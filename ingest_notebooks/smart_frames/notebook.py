# Databricks notebook source
# MAGIC %md
# MAGIC # Smart frame extraction (ingest) — selector v2
# MAGIC
# MAGIC Reads a YAML config (passed via the `config_yaml` widget):
# MAGIC
# MAGIC ```yaml
# MAGIC video_path: /Volumes/.../videos/inbox/foo.mp4
# MAGIC video_name: foo.mp4
# MAGIC video_id: <uuid>
# MAGIC output_dir: /Volumes/.../extracted_frames/foo
# MAGIC pg_host: instance-...database.cloud.databricks.com
# MAGIC pg_dbname: vlm_workbench
# MAGIC pg_user: <sp uuid or user email>
# MAGIC candidate_fps: 1.0
# MAGIC max_frames: 40
# MAGIC max_gap_seconds: null   # optional; default clamp(2*dur/max_frames, 10, 60)
# MAGIC ```
# MAGIC
# MAGIC ## Pipeline (validated on 4 videos — see repo-parent `frame_selection_experiment/`)
# MAGIC
# MAGIC 1. **Candidates** at `candidate_fps` (decord decode, cv2 signals).
# MAGIC 2. **Domain gate** (model-free): per-video scope mask → blurred-gradient
# MAGIC    focus signal → drop blank / glare / red-out / scope-out / bad-exposure /
# MAGIC    **out-of-body** (bright border ⇒ scope withdrawn into the room; PHI
# MAGIC    guard — such frames are never written to the Volume).
# MAGIC 3. **DINOv2 embeddings** (ViT-S/14, weights cached on the Volume) →
# MAGIC    **phase segmentation** via embedding change-points — the surgical
# MAGIC    narrative (tool in → tool out → altered scene → next tool).
# MAGIC 4. **hybrid_coverage selection**: coverage anchors (≥1 frame per
# MAGIC    `max_gap_seconds`) + bookends (first/last usable frame) + best-value
# MAGIC    fill (0.6·grad + 0.4·contrast ranks − 0.5·glare).
# MAGIC 5. **Micro-alignment**: each pick is re-scanned ±0.5 s at native fps and
# MAGIC    replaced when a ≥3 % sharper neighbour exists (mp4/mov only).
# MAGIC 6. **Persist everything**: ALL gated candidates (JPG ≤1024 px +
# MAGIC    `candidates.json` sidecar + `embeddings.npz`) land in
# MAGIC    `<output_dir>/candidates/`; index rows carry `selected` flags so the
# MAGIC    app can re-select (new K / Y) instantly without re-running this job.
# MAGIC
# MAGIC If torch/DINOv2 is unavailable the notebook degrades to the gated
# MAGIC quality+coverage selector (v1.5 behavior) and says so in the status.

# COMMAND ----------
# NOTE: NO in-notebook `%pip install` / `%restart_python`.
# The extra deps (decord, opencv-python-headless==4.12.0.88, psycopg[binary]) are declared
# in the JOB's serverless environment spec (`environments[].spec.dependencies`
# in app.py's ingest submit), so the platform pre-builds the environment BEFORE
# the notebook runs. An in-notebook `%pip install` + `%restart_python` spiked
# the serverless driver's memory and OOM-killed the job ("Execution ran out of
# memory") — even a minimal, no-`--upgrade` install failed while a bare
# `import numpy` succeeded. Declaring deps on the env avoids that spike.
# numpy/pillow/pyyaml come from environment_version 5's base.
# torch is intentionally absent — on this CPU env the DINOv2 phase-segmentation
# stage degrades to the gate+coverage (v1.5) selector.

# COMMAND ----------
import os, json, time, math, traceback
from collections import Counter
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

dbutils.widgets.text("config_yaml", "", "Path to YAML config")
yaml_path = dbutils.widgets.get("config_yaml").strip()
if not yaml_path:
    raise SystemExit("config_yaml widget is required")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}

VIDEO_PATH = cfg["video_path"]
VIDEO_NAME = cfg["video_name"]
VIDEO_ID = cfg["video_id"]
OUTPUT_DIR = cfg["output_dir"]
PG_HOST = cfg["pg_host"]
PG_DBNAME = cfg["pg_dbname"]
PG_USER = cfg["pg_user"]
PG_PORT = int(cfg.get("pg_port", 5432))
CANDIDATE_FPS = float(cfg.get("candidate_fps", 1.0))
MAX_FRAMES = int(cfg.get("max_frames", 40))
MAX_GAP_S = cfg.get("max_gap_seconds")  # None → computed from duration below
PG_ENDPOINT = cfg.get("pg_endpoint", "")
PG_INSTANCE = cfg.get("pg_instance", "")

CAND_DIR = f"{OUTPUT_DIR}/candidates"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CAND_DIR, exist_ok=True)
import glob as _glob
for _stale in _glob.glob(f"{OUTPUT_DIR}/*.jpg") + _glob.glob(f"{CAND_DIR}/*.jpg"):
    try:
        os.remove(_stale)
    except Exception:
        pass
print(f"[ingest] video={VIDEO_NAME} video_id={VIDEO_ID}")
print(f"[ingest] output={OUTPUT_DIR}  candidate_fps={CANDIDATE_FPS}  max_frames={MAX_FRAMES}  max_gap={MAX_GAP_S}")

# COMMAND ----------
from databricks.sdk import WorkspaceClient
import psycopg

_sdk = WorkspaceClient()

def _mint_pg_token() -> str:
    """Mint Lakebase token for this ingest run. Raises on failure."""
    if PG_ENDPOINT:
        try:
            r = _sdk.api_client.do("POST", "/api/2.0/postgres/credentials",
                                   body={"endpoint": PG_ENDPOINT})
            t = (r or {}).get("token")
            if t: return t
        except Exception as e:
            print(f"[ingest] project credential mint failed: {e}")
    if PG_INSTANCE:
        try:
            import uuid as _uuid
            r = _sdk.api_client.do("POST", "/api/2.0/database/credentials/generate",
                                   body={"instance_names": [PG_INSTANCE],
                                         "request_id": str(_uuid.uuid4())})
            t = (r or {}).get("token")
            if t: return t
        except Exception as e:
            print(f"[ingest] instance credential mint failed: {e}")
    raise RuntimeError("could not mint Lakebase token (pg_endpoint and pg_instance both unset)")

# Lakebase tokens expire after ~1 h; a long ingest would fail its final
# writes with the token minted at start. Re-mint once it's 40 min old.
_pg_token = {"value": None, "minted_at": 0.0}

def _pg_password() -> str:
    if not _pg_token["value"] or time.time() - _pg_token["minted_at"] > 40 * 60:
        _pg_token["value"] = _mint_pg_token()
        _pg_token["minted_at"] = time.time()
    return _pg_token["value"]

def _conn():
    return psycopg.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
        user=PG_USER, password=_pg_password(),
        sslmode="require", connect_timeout=15,
    )

def _set_video_status(status: str, message: str = "", duration_s: float = None, n_frames: int = None):
    """Update video status in Lakebase. Failures raise to fail the job loudly."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE videos SET status = %s, status_message = %s,
                duration_s = COALESCE(%s, duration_s),
                n_frames_extracted = COALESCE(%s, n_frames_extracted),
                updated_at = now()
            WHERE id = %s
        """, (status, message, duration_s, n_frames, VIDEO_ID))
    print(f"[ingest][db] status={status}: {message[:100]}")

# COMMAND ----------
import cv2
from decord import VideoReader, cpu

MAX_SIDE = 640    # analysis resolution
SAVE_SIDE = 1024  # persisted candidate JPG resolution (Playground resizes to 1024 anyway)

def _resize_max(rgb, max_side):
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return rgb

def _rank(values):
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(v))
    n = len(v)
    return order / (n - 1) if n > 1 else np.full(n, 0.5)

def _grad_of(gray, m_bool):
    gb = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    return float(np.sqrt(gx ** 2 + gy ** 2)[m_bool].mean())

# COMMAND ----------
try:
    t0 = time.time()
    degrade_reasons = []
    oob_rejection_count = 0

    # Mint Lakebase token inside try so failures fail the job loudly with error status
    _pg_password()

    # Probe with a throwaway full-res reader, then run ALL analysis through a
    # decode-time-resized reader (decord scales inside the decoder). The first
    # v1.5 run OOM'd on serverless CPU (74s, "Execution ran out of memory")
    # decoding full-res frames and downscaling in numpy — decode-time resize
    # keeps peak memory at ~one small frame regardless of source resolution.
    # A separate full-res reader is opened only for writing the final JPGs.
    _probe = VideoReader(VIDEO_PATH, ctx=cpu(0))
    n_total = len(_probe)
    fps = float(_probe.get_avg_fps())
    duration_s = n_total / max(fps, 0.001)
    h0, w0 = _probe[0].asnumpy().shape[:2]
    del _probe
    _scale = min(1.0, MAX_SIDE / max(h0, w0))
    Ws, Hs = (int(w0 * _scale) // 2) * 2, (int(h0 * _scale) // 2) * 2  # decoder wants even dims
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0), width=Ws, height=Hs)

    step = max(1, int(fps / CANDIDATE_FPS))
    indices = list(range(0, n_total, step))
    if MAX_GAP_S is None:
        MAX_GAP_S = float(np.clip(2.0 * duration_s / max(MAX_FRAMES, 1), 10.0, 60.0))
    print(f"[ingest] {n_total} frames @ {fps:.1f}fps = {duration_s:.1f}s "
          f"({w0}x{h0} → analysis {Ws}x{Hs}); {len(indices)} candidates; max_gap={MAX_GAP_S:.0f}s")
    _set_video_status("processing", f"{n_total} frames @ {fps:.1f}fps", duration_s=duration_s)

    # ---- pass A: robust scope mask (constant memory) ----
    # A pixel is "scope" if it is lit (>25) in more than 40% of ≤120 sampled
    # frames — i.e. its 60th-percentile luminance is >25 — counted with one
    # uint16 accumulator instead of stacking frames. A scope withdrawal that
    # lights the whole frame for <40% of the video can't widen the mask (the
    # old running-max mask collapsed to full-frame and disabled the OOB check).
    ok_idx = []
    for idx in indices:
        try:
            cv2.cvtColor(vr[idx].asnumpy(), cv2.COLOR_RGB2GRAY)
        except Exception:
            continue
        ok_idx.append(idx)
    if not ok_idx:
        raise RuntimeError("no decodable candidate frames")

    # Uniform over the whole video (a plain stride would take only the first
    # 120 frames whenever there are fewer than 240 candidates).
    mask_sample_indices = [ok_idx[int(i)] for i in np.unique(
        np.linspace(0, len(ok_idx) - 1, num=min(120, len(ok_idx))).round().astype(int))]
    lit_count = np.zeros((Hs, Ws), dtype=np.uint16)
    n_mask = 0
    for idx in mask_sample_indices:
        try:
            lit_count += cv2.cvtColor(vr[idx].asnumpy(), cv2.COLOR_RGB2GRAY) > 25
            n_mask += 1
        except Exception:
            continue
    if not n_mask:
        raise RuntimeError("no decodable frames for the scope mask")
    mask = (lit_count > 0.4 * n_mask).astype(np.uint8)
    ncc, lab = cv2.connectedComponents(mask)
    if ncc > 1:
        biggest = 1 + int(np.argmax([(lab == i).sum() for i in range(1, ncc)]))
        mask = (lab == biggest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.erode(mask, np.ones((21, 21), np.uint8))

    # Determine if mask represents a true scope or near-full-frame image
    full_frame = mask.mean() > 0.90
    M = mask.astype(bool)
    border = None if full_frame else ~M
    if full_frame:
        print(f"[ingest] scope mask covers {M.mean()*100:.0f}% (full-frame fallback)")
    else:
        print(f"[ingest] scope mask covers {M.mean()*100:.0f}% (border available for OOB check)")

    # ---- pass B: re-decode (small), signals per candidate ----
    # Mask-independent OOB detector: a backstop for when the scope-mask border
    # check can't see the room (full-frame video, mask edge cases). OOB means
    # "lit area is mostly near-white/gray/blue (lights, drapes, walls, title
    # cards) AND has almost no saturated tissue hue".
    # Calibrated 2026-10-09 on 611 real frames from 4 arthroscopy videos
    # (frame_selection_experiment round2) — false-reject on in-body frames:
    #   v1 1.5%, v2 0.0%, web1 0.0%, web2 5.5% (web2's rejects are its white
    #   intro/title frames and blank frames, i.e. correct rejects).
    # The earlier "any 2 of 4 weak flags" rule rejected 6-17% of in-body
    # frames (pale cartilage + haze trip the tissue and contrast flags).
    # Known gap: a close-up of skin/face in a full-frame (no scope circle)
    # video reads as tissue hue — only the border check catches that today.
    TISSUE_HUE_MIN = 0.10       # below this saturated-tissue fraction → suspicious
    NEAR_WHITE_BLUE_MAX = 0.30  # above this near-white/gray/blue fraction → suspicious
    LIT_MIN = 0.02              # <2% lit pixels: nothing to judge → OOB

    def _is_oob_frame(rgb):
        """Mask-independent out-of-body check over LIT pixels only (gray > 25),
        so the black scope vignette doesn't dilute the fractions.
        Returns (oob_flag, tissue_frac, edge_density, white_blue_frac, brightness_std);
        edge density and brightness std are recorded for analysis, not used to decide."""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s = hsv[..., 0], hsv[..., 1]

        lit = g > 25
        if lit.mean() < LIT_MIN:
            return True, 0.0, 0.0, 1.0, 0.0
        h_lit, s_lit, g_lit = h[lit], s[lit], g[lit]

        # Tissue hue: red wraps around in OpenCV's 0-179 hue (0-20 and 160-179) + orange-yellow (30-70)
        tissue_mask = (((h_lit < 20) | (h_lit >= 160)) | ((h_lit >= 30) & (h_lit <= 70))) & (s_lit > 80)
        tissue_frac = float(tissue_mask.mean())

        near_white = (g_lit > 200) & (s_lit < 30)
        blue_tone = (h_lit >= 100) & (h_lit <= 140)
        near_white_blue_frac = float((near_white | blue_tone).mean())

        # 2-D Canny on the full frame, measured inside the eroded lit region so
        # the vignette rim doesn't count as edges.
        inner = cv2.erode(lit.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        if not inner.any():
            inner = lit
        edge_density = float((cv2.Canny(g, 50, 150)[inner] > 0).mean())
        brightness_std = float(g_lit.std())

        oob_flag = near_white_blue_frac > NEAR_WHITE_BLUE_MAX and tissue_frac < TISSUE_HUE_MIN
        return oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std

    candidates = []
    for idx in ok_idx:
        try:
            rgb = vr[idx].asnumpy()
        except Exception:
            continue
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        v = g[M]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hch, sch = hsv[..., 0][M], hsv[..., 1][M]
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        r_, g_, b_ = (rgb[..., i].astype(np.float32) for i in (0, 1, 2))
        rg = r_ - g_
        yb = 0.5 * (r_ + g_) - b_
        colorf = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

        # Compute mask-independent OOB detector signals
        oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

        candidates.append({
            "frame_index": int(idx),
            "timestamp_s": float(idx / fps),
            "grad": _grad_of(g, M),
            "contrast": float(v.std()),
            "sharpness": sharp,
            "colorfulness": float(colorf),
            "whiteout": float((v > 240).mean()),
            "blackout": float((v < 20).mean()),
            "mean_lum": float(v.mean()),
            "red_frac": float((((hch < 10) | (hch > 170)) & (sch > 90)).mean()),
            "border_lum": float(g[border].mean()) if border is not None else 0.0,
            "tissue_frac": float(tissue_frac),
            "edge_density": float(edge_density),
            "near_white_blue_frac": float(near_white_blue_frac),
            "brightness_std": float(brightness_std),
            "oob_detector_flag": bool(oob_flag),
        })
    grad_med = float(np.median([c["grad"] for c in candidates]))
    border_base = float(np.median([c["border_lum"] for c in candidates])) if border is not None else 0.0

    # ---- gate v2 (model-free; thresholds validated on 4 videos) ----
    def _gate(c):
        if c["whiteout"] > 0.35:
            return "glare"
        if c["blackout"] > 0.60:
            return "scope-out"
        if c["mean_lum"] < 18 or c["mean_lum"] > 246:
            return "exposure"
        if c["red_frac"] > 0.85:
            return "red-out"
        # Mask-based OOB check: if border exists and frame has bright border spillover
        if border is not None and c["border_lum"] > max(35.0, 5.0 * border_base):
            return "out-of-body"   # PHI guard: never persisted
        # Mask-independent OOB check: catches frames shot outside body even if mask fails
        if c["oob_detector_flag"]:
            return "out-of-body"   # PHI guard: never persisted
        if c["grad"] < 0.55 * grad_med:
            return "blank"
        return "ok"

    for c in candidates:
        c["gate_reason"] = _gate(c)
    gated = [c for c in candidates if c["gate_reason"] == "ok"]
    drops = Counter(c["gate_reason"] for c in candidates if c["gate_reason"] != "ok")
    oob_rejection_count = drops.get("out-of-body", 0)
    drop_str = ", ".join(f"{k} {v}" for k, v in sorted(drops.items())) or "none"
    print(f"[ingest] gate: {len(gated)}/{len(candidates)} pass "
          f"(dropped: {drop_str}; grad_med={grad_med:.1f}; OOB rejected: {oob_rejection_count})")
    if not gated:
        raise RuntimeError(f"domain gate rejected all {len(candidates)} candidates ({drop_str})")

    # ---- DINOv2 embeddings + phase segmentation (degrades gracefully) ----
    selector_version = "v2"
    embs = None
    try:
        # Cache hub weights on the Volume so they download exactly once per
        # workspace. Derive <volume-root>/local_models/torch_hub from the
        # output dir (…/medical_video/extracted_frames/<stem>).
        vol_root = os.path.dirname(os.path.dirname(OUTPUT_DIR))
        os.environ["TORCH_HOME"] = f"{vol_root}/local_models/torch_hub"
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(dev)
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
        STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
        t_e = time.time()
        chunks = []
        B = 32
        for j in range(0, len(gated), B):
            batch = []
            for c in gated[j:j + B]:
                rgb = vr[c["frame_index"]].asnumpy()
                im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                batch.append(torch.from_numpy(im).permute(2, 0, 1))
            x = torch.stack(batch).to(dev)
            with torch.no_grad():
                chunks.append(dino((x - MEAN) / STD).float().cpu().numpy())
        embs = np.concatenate(chunks, 0)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        print(f"[ingest] DINOv2: embedded {len(gated)} candidates on {dev} in {time.time()-t_e:.0f}s")
    except Exception as e:
        selector_version = "v1.5"
        error_msg = f"{type(e).__name__}: {str(e)[:150]}"
        degrade_reasons.append(f"DINOv2 unavailable — {error_msg}")
        print(f"[ingest][warn] DINOv2 unavailable → falling back to {selector_version}: {error_msg}")

    n_phases = int(np.clip(MAX_FRAMES, 4, 24))
    # embedding-row → timestamp pairing for the sidecar (embs rows stay in
    # "initial gated order + dense additions appended" order even after the
    # pool itself is re-sorted by time below)
    emb_ts = [c["timestamp_s"] for c in gated]
    bound_times: list = []
    if embs is not None and len(gated) > n_phases:
        # Change-point segmentation: split at the largest embedding-novelty
        # jumps (min-gap enforced) → contiguous narrative phases.
        nov = np.array([0.0] + [1 - float(np.dot(embs[i], embs[i - 1])) for i in range(1, len(gated))])
        nov = np.convolve(nov, np.ones(3) / 3, mode="same")
        min_gap = max(1, len(gated) // (n_phases * 2))
        bounds = []
        for ci in np.argsort(-nov):
            if ci == 0:
                continue
            if all(abs(int(ci) - b) >= min_gap for b in bounds):
                bounds.append(int(ci))
            if len(bounds) >= n_phases - 1:
                break
        bound_times = sorted(gated[b]["timestamp_s"] for b in sorted(bounds))

    # ---- adaptive densification around phase boundaries ----
    # Brief story beats (instrument entering/leaving, a transient contact,
    # a title/CT card in edited videos) can fall BETWEEN 1-fps candidates.
    # Re-scan ±2s around each boundary at ~6fps, gate the new frames, and
    # merge them into the pool. Validated: story-completeness 7→8 on a
    # structured laparoscopy video (8/16 final picks came from the dense
    # pass); neutral on continuous single-shot arthroscopy.
    n_dense = 0
    if bound_times and embs is not None and Path(VIDEO_PATH).suffix.lower() in (".mp4", ".mov", ".m4v"):
        existing = {round(c["timestamp_s"], 2) for c in gated}
        dense_step = max(1, int(round(fps / 6.0)))
        new_cands = []
        for bt in bound_times:
            lo = max(0, int((bt - 2.0) * fps))
            hi = min(n_total - 1, int((bt + 2.0) * fps))
            for fi in range(lo, hi + 1, dense_step):
                ts = fi / fps
                if round(ts, 2) in existing:
                    continue
                try:
                    rgb = vr[fi].asnumpy()
                except Exception:
                    continue
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if g.shape != (Hs, Ws):
                    continue
                v = g[M]
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hch, sch = hsv[..., 0][M], hsv[..., 1][M]

                # Compute OOB detector signals
                oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)

                cand = {
                    "frame_index": int(fi),
                    "timestamp_s": float(ts),
                    "grad": _grad_of(g, M),
                    "contrast": float(v.std()),
                    "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()),
                    "colorfulness": 0.0,
                    "whiteout": float((v > 240).mean()),
                    "blackout": float((v < 20).mean()),
                    "mean_lum": float(v.mean()),
                    "red_frac": float((((hch < 10) | (hch > 170)) & (sch > 90)).mean()),
                    "border_lum": float(g[border].mean()) if border is not None else 0.0,
                    "tissue_frac": float(tissue_frac),
                    "edge_density": float(edge_density),
                    "near_white_blue_frac": float(near_white_blue_frac),
                    "brightness_std": float(brightness_std),
                    "oob_detector_flag": bool(oob_flag),
                }
                if _gate(cand) == "ok":
                    cand["gate_reason"] = "ok"
                    new_cands.append(cand)
                    existing.add(round(ts, 2))
        if new_cands:
            # embed the dense additions and extend pool + embedding store
            emb_ts = emb_ts + [c["timestamp_s"] for c in new_cands]
            chunks = []
            for j in range(0, len(new_cands), 32):
                batch = []
                for c in new_cands[j:j + 32]:
                    rgb = vr[c["frame_index"]].asnumpy()
                    im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                    batch.append(torch.from_numpy(im).permute(2, 0, 1))
                x = torch.stack(batch).to(dev)
                with torch.no_grad():
                    chunks.append(dino((x - MEAN) / STD).float().cpu().numpy())
            new_embs = np.concatenate(chunks, 0)
            new_embs = new_embs / (np.linalg.norm(new_embs, axis=1, keepdims=True) + 1e-8)
            embs = np.concatenate([embs, new_embs], 0)
            gated = sorted(gated + new_cands, key=lambda c: c["timestamp_s"])
            n_dense = len(new_cands)
        print(f"[ingest] densify: +{n_dense} gated candidates around {len(bound_times)} boundaries")

    # ---- assign phase_id (bisect over boundary times; works for merged pool) --
    import bisect
    if bound_times:
        for c in gated:
            c["phase_id"] = bisect.bisect_right(bound_times, c["timestamp_s"])
    else:
        for c in gated:
            c["phase_id"] = min(n_phases - 1, int(c["timestamp_s"] / max(duration_s, 1e-6) * n_phases))

    # ---- value + hybrid_coverage selection ----
    g_rank = _rank([c["grad"] for c in gated])
    c_rank = _rank([c["contrast"] for c in gated])
    for c, a, b in zip(gated, g_rank, c_rank):
        c["value"] = 0.6 * a + 0.4 * b - 0.5 * c["whiteout"]

    # NOTE: keep in sync with `_hybrid_coverage_select` in app.py (reselect
    # endpoint) — same algorithm, duplicated because notebooks are imported
    # into the SP home as single files and can't share a module with the app.
    def _select(pool, k, y, dur):
        pool = sorted(pool, key=lambda c: c["timestamp_s"])
        n_min = max(1, math.ceil(dur / y))
        picked, anchor_keys = [], set()
        for i in range(n_min):                      # coverage anchors
            w0, w1 = i * y, min((i + 1) * y, dur + 1e-6)
            inwin = [c for c in pool if w0 <= c["timestamp_s"] < w1]
            if inwin:
                best = max(inwin, key=lambda c: c["value"])
                if best not in picked:
                    picked.append(best)
                    anchor_keys.add(best["timestamp_s"])
        for bk in (pool[0], pool[-1]):              # bookends
            if bk not in picked and len(picked) < max(k, n_min):
                picked.append(bk)
        remaining = sorted([c for c in pool if c not in picked], key=lambda c: -c["value"])
        keff = min(max(k, n_min), len(pool))
        for min_gap in (0.5 * y, 0.33 * y, 0.2 * y, 0.0):   # value fill, relaxing spacing
            for c in remaining:
                if len(picked) >= keff:
                    break
                if c in picked:
                    continue
                if all(abs(c["timestamp_s"] - p["timestamp_s"]) >= min_gap for p in picked):
                    picked.append(c)
            if len(picked) >= keff:
                break
        picked.sort(key=lambda c: c["timestamp_s"])
        return picked, anchor_keys

    selected, anchor_keys = _select(gated, MAX_FRAMES, MAX_GAP_S, duration_s)
    sel_keys = {c["timestamp_s"] for c in selected}
    gaps = np.diff([0.0] + sorted(sel_keys) + [duration_s])
    print(f"[ingest] selected {len(selected)}/{len(gated)} (max_gap={gaps.max():.1f}s, "
          f"{len(anchor_keys)} anchors, phases hit "
          f"{len({c['phase_id'] for c in selected})}/{max(c['phase_id'] for c in gated)+1})")

    # ---- micro-alignment (mp4/mov only; accept only ≥3% sharper) ----
    # After swapping frames, recompute all signals and re-gate to ensure OOB/PHI compliance
    def _compute_frame_signals(rgb, frame_idx, ts):
        """Compute all signals for a frame (used after micro-align swap)."""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        v = g[M]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hch, sch = hsv[..., 0][M], hsv[..., 1][M]
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        r_, g_, b_ = (rgb[..., i].astype(np.float32) for i in (0, 1, 2))
        rg = r_ - g_
        yb = 0.5 * (r_ + g_) - b_
        colorf = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        oob_flag, tissue_frac, edge_density, near_white_blue_frac, brightness_std = _is_oob_frame(rgb)
        return {
            "frame_index": int(frame_idx),
            "timestamp_s": float(ts),
            "grad": _grad_of(g, M),
            "contrast": float(v.std()),
            "sharpness": sharp,
            "colorfulness": float(colorf),
            "whiteout": float((v > 240).mean()),
            "blackout": float((v < 20).mean()),
            "mean_lum": float(v.mean()),
            "red_frac": float((((hch < 10) | (hch > 170)) & (sch > 90)).mean()),
            "border_lum": float(g[border].mean()) if border is not None else 0.0,
            "tissue_frac": float(tissue_frac),
            "edge_density": float(edge_density),
            "near_white_blue_frac": float(near_white_blue_frac),
            "brightness_std": float(brightness_std),
            "oob_detector_flag": bool(oob_flag),
        }

    refined_n = 0
    if Path(VIDEO_PATH).suffix.lower() in (".mp4", ".mov", ".m4v"):
        half = int(round(fps * 0.5))
        for c in selected:
            lo = max(0, c["frame_index"] - half)
            hi = min(n_total - 1, c["frame_index"] + half)
            best_g, best_idx = c["grad"], c["frame_index"]
            for fi in range(lo, hi + 1, max(1, (hi - lo) // 14)):  # ~15 probes
                if fi == c["frame_index"]:
                    continue
                try:
                    g2 = cv2.cvtColor(vr[fi].asnumpy(), cv2.COLOR_RGB2GRAY)
                except Exception:
                    continue
                if g2.shape != (Hs, Ws):
                    continue
                gv = _grad_of(g2, M)
                if gv > best_g:
                    best_g, best_idx = gv, fi
            if best_idx != c["frame_index"] and best_g >= 1.03 * c["grad"]:
                # Candidate swap: recompute all signals for the new frame and re-gate
                try:
                    new_rgb = vr[best_idx].asnumpy()
                    new_signals = _compute_frame_signals(new_rgb, best_idx, best_idx / fps)
                    if _gate(new_signals) == "ok":
                        # Swap passes gate: update the candidate
                        c.update(new_signals)
                        refined_n += 1
                    # else: swap fails gate (OOB/PHI), keep original frame
                except Exception:
                    # Decode error on candidate: keep original
                    pass
        print(f"[ingest] micro-align: refined {refined_n}/{len(selected)} picks "
              f"(all re-gated for OOB/PHI compliance)")
        sel_keys = {c["timestamp_s"] for c in selected}

    # ---- persist candidates + sidecar + register in Lakebase ----
    # (no `# COMMAND ----------` inside try — Databricks would split the cell)
    # Full-res reader ONLY here: sequential single-frame decode + immediate
    # downscale to SAVE_SIDE keeps peak memory at ~one full frame.
    del vr
    vr_full = VideoReader(VIDEO_PATH, ctx=cpu(0))
    video_stem = Path(VIDEO_NAME).stem
    rows = []
    for c in sorted(gated, key=lambda c: c["frame_index"]):
        is_sel = c["timestamp_s"] in sel_keys
        filename = f"{video_stem}_frame_{c['timestamp_s']:07.2f}s.jpg"
        out_path = f"{CAND_DIR}/{filename}"
        Image.fromarray(_resize_max(vr_full[c["frame_index"]].asnumpy(), SAVE_SIDE)).save(
            out_path, format="JPEG", quality=88)
        rows.append({**c, "frame_path": out_path, "frame_name": filename,
                     "selected": is_sel,
                     "is_anchor": c["timestamp_s"] in anchor_keys})

    sidecar = {
        "video_id": VIDEO_ID, "video_name": VIDEO_NAME, "duration_s": duration_s,
        "selector_version": selector_version, "max_frames": MAX_FRAMES,
        "max_gap_seconds": MAX_GAP_S, "gate_drops": dict(drops),
        "oob_rejection_count": oob_rejection_count,
        "candidates": [{k: v for k, v in r.items() if k != "frame_index"} for r in rows],
    }
    with open(f"{OUTPUT_DIR}/candidates.json", "w") as f:
        json.dump(sidecar, f)
    if embs is not None:
        np.savez(f"{OUTPUT_DIR}/embeddings.npz", embs=embs,
                 timestamps=np.array(emb_ts))
    print(f"[ingest] persisted {len(rows)} candidates + sidecar to {OUTPUT_DIR}")

    with _conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM extracted_frames_index WHERE video_id = %s", (VIDEO_ID,))
        for r in rows:
            cur.execute("""
                INSERT INTO extracted_frames_index
                    (video_id, frame_path, frame_name, timestamp_s, score,
                     sharpness, colorfulness, contrast, grad,
                     phase_id, is_anchor, selected, selector_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (frame_path) DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    timestamp_s = EXCLUDED.timestamp_s,
                    score = EXCLUDED.score,
                    sharpness = EXCLUDED.sharpness,
                    colorfulness = EXCLUDED.colorfulness,
                    contrast = EXCLUDED.contrast,
                    grad = EXCLUDED.grad,
                    phase_id = EXCLUDED.phase_id,
                    is_anchor = EXCLUDED.is_anchor,
                    selected = EXCLUDED.selected,
                    selector_version = EXCLUDED.selector_version,
                    updated_at = now()
            """, (
                VIDEO_ID, r["frame_path"], r["frame_name"], r["timestamp_s"],
                r["value"], r["sharpness"], r["colorfulness"], r["contrast"],
                r["grad"], r["phase_id"], r["is_anchor"], r["selected"],
                selector_version,
            ))
    n_sel = sum(1 for r in rows if r["selected"])
    print(f"[ingest][db] registered {len(rows)} candidates ({n_sel} selected)")

    elapsed = time.time() - t0
    status_msg = (f"selected {n_sel} of {len(rows)} candidates ({selector_version}) in {elapsed:.0f}s · "
                  f"OOB rejected: {oob_rejection_count} · dropped: {drop_str}")
    if degrade_reasons:
        status_msg += f" · degraded: {'; '.join(degrade_reasons)}"
    else:
        status_msg += " · re-select available"

    _set_video_status("ready", status_msg, n_frames=n_sel)
    _exit_payload = json.dumps({
        "video_id": VIDEO_ID,
        "n_frames": n_sel,
        "n_candidates": len(rows),
        "gate_drops": dict(drops),
        "oob_rejection_count": oob_rejection_count,
        "selector_version": selector_version,
        "max_gap_seconds": MAX_GAP_S,
        "micro_aligned": refined_n,
        "elapsed_s": elapsed,
        "degrade_reasons": degrade_reasons,
        "status": "ready",
    })

except Exception as e:
    # dbutils.notebook.exit() raises internally, so the happy-path exit must
    # live OUTSIDE the try or it would overwrite status='ready' with 'error'.
    tb = traceback.format_exc()
    print(f"[ingest][error] {e}\n{tb}")
    # Set error status without hiding the original error (_conn re-mints the
    # token if the first mint is what failed or it has since expired).
    try:
        _set_video_status("error", str(e)[:300])
    except Exception as status_err:
        print(f"[ingest][error] failed to set error status: {status_err}")
    raise
else:
    dbutils.notebook.exit(_exit_payload)

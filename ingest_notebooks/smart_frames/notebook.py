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
# MAGIC %pip install -q --upgrade decord pillow numpy opencv-python-headless psycopg[binary] pyyaml
# MAGIC %restart_python
# NOTE: torch is NOT pip-installed — the app submits this notebook on
# serverless GPU (GPU_1xA10 + base_environment=databricks_ai_v4) where torch
# ships preinstalled. On a CPU environment the DINOv2 stage cleanly degrades.

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
_sdk = WorkspaceClient()

def _mint_pg_token() -> str:
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
    raise SystemExit("could not mint Lakebase token (pg_endpoint and pg_instance both unset)")

PG_PASSWORD = _mint_pg_token()

import psycopg

def _conn():
    return psycopg.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
        user=PG_USER, password=PG_PASSWORD,
        sslmode="require", connect_timeout=15,
    )

def _set_video_status(status: str, message: str = "", duration_s: float = None, n_frames: int = None):
    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE videos SET status = %s, status_message = %s,
                    duration_s = COALESCE(%s, duration_s),
                    n_frames_extracted = COALESCE(%s, n_frames_extracted),
                    updated_at = now()
                WHERE id = %s
            """, (status, message, duration_s, n_frames, VIDEO_ID))
        print(f"[ingest][db] status={status}: {message[:100]}")
    except Exception as e:
        print(f"[ingest][db][warn] failed to update video status: {e}")

_set_video_status("processing", "loading video")

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
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    n_total = len(vr)
    fps = float(vr.get_avg_fps())
    duration_s = n_total / max(fps, 0.001)
    step = max(1, int(fps / CANDIDATE_FPS))
    indices = list(range(0, n_total, step))
    if MAX_GAP_S is None:
        MAX_GAP_S = float(np.clip(2.0 * duration_s / max(MAX_FRAMES, 1), 10.0, 60.0))
    print(f"[ingest] {n_total} frames @ {fps:.1f}fps = {duration_s:.1f}s; "
          f"{len(indices)} candidates; max_gap={MAX_GAP_S:.0f}s")
    _set_video_status("processing", f"{n_total} frames @ {fps:.1f}fps", duration_s=duration_s)

    # ---- pass A: grays only → scope mask (memory-flat on long videos) ----
    grays, ok_idx = [], []
    for idx in indices:
        try:
            rgb = _resize_max(vr[idx].asnumpy(), MAX_SIDE)
        except Exception:
            continue
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        ok_idx.append(idx)
    if not grays:
        raise RuntimeError("no decodable candidate frames")
    Hs, Ws = grays[0].shape

    mask = (np.max(np.stack(grays), axis=0) > 25).astype(np.uint8)
    ncc, lab = cv2.connectedComponents(mask)
    if ncc > 1:
        biggest = 1 + int(np.argmax([(lab == i).sum() for i in range(1, ncc)]))
        mask = (lab == biggest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.erode(mask, np.ones((21, 21), np.uint8))
    full_frame = mask.mean() > 0.90
    if full_frame:
        mask = np.ones((Hs, Ws), np.uint8)
    M = mask.astype(bool)
    border = None if full_frame else ~M
    print(f"[ingest] scope mask covers {M.mean()*100:.0f}% "
          f"({'full-frame fallback' if full_frame else 'circle'})")

    # ---- pass B: signals per candidate ----
    candidates = []
    for gi, idx in enumerate(ok_idx):
        try:
            rgb = _resize_max(vr[idx].asnumpy(), MAX_SIDE)
        except Exception:
            continue
        g = grays[gi]
        v = g[M]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hch, sch = hsv[..., 0][M], hsv[..., 1][M]
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        r_, g_, b_ = (rgb[..., i].astype(np.float32) for i in (0, 1, 2))
        rg = r_ - g_
        yb = 0.5 * (r_ + g_) - b_
        colorf = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
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
        if border is not None and c["border_lum"] > max(35.0, 5.0 * border_base):
            return "out-of-body"   # PHI guard: never persisted
        if c["grad"] < 0.55 * grad_med:
            return "blank"
        return "ok"

    for c in candidates:
        c["gate_reason"] = _gate(c)
    gated = [c for c in candidates if c["gate_reason"] == "ok"]
    drops = Counter(c["gate_reason"] for c in candidates if c["gate_reason"] != "ok")
    drop_str = ", ".join(f"{k} {v}" for k, v in sorted(drops.items())) or "none"
    print(f"[ingest] gate: {len(gated)}/{len(candidates)} pass "
          f"(dropped: {drop_str}; grad_med={grad_med:.1f})")
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
                rgb = _resize_max(vr[c["frame_index"]].asnumpy(), MAX_SIDE)
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
        print(f"[ingest][warn] DINOv2 unavailable → falling back to {selector_version}: {str(e)[:200]}")

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
                    rgb = _resize_max(vr[fi].asnumpy(), MAX_SIDE)
                except Exception:
                    continue
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if g.shape != (Hs, Ws):
                    continue
                v = g[M]
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                hch, sch = hsv[..., 0][M], hsv[..., 1][M]
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
                    rgb = _resize_max(vr[c["frame_index"]].asnumpy(), MAX_SIDE)
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
                    g2 = cv2.cvtColor(_resize_max(vr[fi].asnumpy(), MAX_SIDE), cv2.COLOR_RGB2GRAY)
                except Exception:
                    continue
                if g2.shape != (Hs, Ws):
                    continue
                gv = _grad_of(g2, M)
                if gv > best_g:
                    best_g, best_idx = gv, fi
            if best_idx != c["frame_index"] and best_g >= 1.03 * c["grad"]:
                c["frame_index"], c["grad"] = best_idx, best_g
                c["timestamp_s"] = best_idx / fps
                refined_n += 1
        print(f"[ingest] micro-align: refined {refined_n}/{len(selected)} picks")
        sel_keys = {c["timestamp_s"] for c in selected}

    # ---- persist candidates + sidecar + register in Lakebase ----
    # (no `# COMMAND ----------` inside try — Databricks would split the cell)
    video_stem = Path(VIDEO_NAME).stem
    rows = []
    for c in gated:
        is_sel = c["timestamp_s"] in sel_keys
        filename = f"{video_stem}_frame_{c['timestamp_s']:07.2f}s.jpg"
        out_path = f"{CAND_DIR}/{filename}"
        Image.fromarray(_resize_max(vr[c["frame_index"]].asnumpy(), SAVE_SIDE)).save(
            out_path, format="JPEG", quality=88)
        rows.append({**c, "frame_path": out_path, "frame_name": filename,
                     "selected": is_sel,
                     "is_anchor": c["timestamp_s"] in anchor_keys})

    sidecar = {
        "video_id": VIDEO_ID, "video_name": VIDEO_NAME, "duration_s": duration_s,
        "selector_version": selector_version, "max_frames": MAX_FRAMES,
        "max_gap_seconds": MAX_GAP_S, "gate_drops": dict(drops),
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
    _set_video_status(
        "ready",
        f"selected {n_sel} of {len(rows)} candidates ({selector_version}) in {elapsed:.0f}s · "
        f"dropped: {drop_str} · re-select available",
        n_frames=n_sel,
    )
    _exit_payload = json.dumps({
        "video_id": VIDEO_ID,
        "n_frames": n_sel,
        "n_candidates": len(rows),
        "gate_drops": dict(drops),
        "selector_version": selector_version,
        "max_gap_seconds": MAX_GAP_S,
        "micro_aligned": refined_n,
        "elapsed_s": elapsed,
        "status": "ready",
    })

except Exception as e:
    # dbutils.notebook.exit() raises internally, so the happy-path exit must
    # live OUTSIDE the try or it would overwrite status='ready' with 'error'.
    tb = traceback.format_exc()
    print(f"[ingest][error] {e}\n{tb}")
    _set_video_status("error", str(e)[:300])
    raise
else:
    dbutils.notebook.exit(_exit_payload)

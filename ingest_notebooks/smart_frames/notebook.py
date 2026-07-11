# Databricks notebook source
# MAGIC %md
# MAGIC # Smart frame extraction (ingest) — selector v1.5
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
# MAGIC ```
# MAGIC
# MAGIC ## v1.5 (gate v2) — what changed vs v1
# MAGIC
# MAGIC v1 scored every candidate on sharpness+colorfulness+contrast and picked the
# MAGIC best per time window. Validated on 4 videos (2 arthroscopy, 2 laparoscopy;
# MAGIC see `frame_selection_experiment/` in the repo parent), that approach:
# MAGIC   - wasted ~22% of picks on defocused "blank disc" frames (Laplacian
# MAGIC     sharpness *rewards* sensor noise on blank frames),
# MAGIC   - rewarded blood/glare via the colorfulness term.
# MAGIC
# MAGIC v1.5 adds a **model-free domain gate** before selection:
# MAGIC   1. per-video scope mask (endoscope circle; full-frame fallback when the
# MAGIC      video has no vignette) — all stats are computed inside the mask;
# MAGIC   2. blurred-gradient focus signal (Gaussian σ=2 → Sobel; real edges
# MAGIC      survive the blur, noise doesn't — unlike Laplacian);
# MAGIC   3. drops: blank (grad < 0.55×video median), glare (>35% blown pixels),
# MAGIC      red-out (lens against blood: >85% saturated-red), scope-out
# MAGIC      (>60% black), bad exposure, and **out-of-body** (bright content in
# MAGIC      the normally-black border → scope withdrawn into the room; PHI
# MAGIC      guard — these frames are never written to the Volume);
# MAGIC   4. scoring drops colorfulness: 0.6×grad_rank + 0.4×contrast_rank;
# MAGIC   5. bookend rule: the first and last *gated* candidates are always
# MAGIC      included (set-level judging showed every selector missed the
# MAGIC      entry / final-state story beats).

# COMMAND ----------
# MAGIC %pip install -q --upgrade decord pillow numpy opencv-python-headless psycopg[binary] pyyaml
# MAGIC %restart_python

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
SELECTOR_VERSION = "v1.5"
# Lakebase endpoint path — the notebook mints its OWN Postgres token via
# /api/2.0/postgres/credentials as the run identity (the App SP). We never
# accept a pre-minted password: it would leak via the Volume-stored YAML.
PG_ENDPOINT = cfg.get("pg_endpoint", "")
PG_INSTANCE = cfg.get("pg_instance", "")  # legacy Provisioned fallback

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Clear frames from a prior ingest so re-running doesn't accumulate stale picks.
import glob as _glob
for _stale in _glob.glob(f"{OUTPUT_DIR}/*.jpg"):
    try:
        os.remove(_stale)
    except Exception:
        pass
print(f"[ingest] video={VIDEO_NAME} video_id={VIDEO_ID} selector={SELECTOR_VERSION}")
print(f"[ingest] output={OUTPUT_DIR}  candidate_fps={CANDIDATE_FPS}  max_frames={MAX_FRAMES}")

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
# Signals. cv2 for speed (the v1 numpy sliding-window Laplacian was the slow
# path); decord stays the decoder.
import cv2
from decord import VideoReader, cpu

MAX_SIDE = 640  # analysis resolution; picks are re-read at full res for saving

def _small(rgb):
    h, w = rgb.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return rgb

def _rank(values):
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(v))
    n = len(v)
    return order / (n - 1) if n > 1 else np.full(n, 0.5)

# COMMAND ----------
try:
    t0 = time.time()
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    n_total = len(vr)
    fps = float(vr.get_avg_fps())
    duration_s = n_total / max(fps, 0.001)
    step = max(1, int(fps / CANDIDATE_FPS))
    indices = list(range(0, n_total, step))
    print(f"[ingest] {n_total} frames @ {fps:.1f}fps = {duration_s:.1f}s; "
          f"scoring {len(indices)} candidates (1 per {step} frames)")
    _set_video_status("processing", f"{n_total} frames @ {fps:.1f}fps", duration_s=duration_s)

    # ---- pass A: grays only (memory-safe for long videos) → scope mask ----
    grays, ok_idx = [], []
    for idx in indices:
        try:
            rgb = _small(vr[idx].asnumpy())
        except Exception:
            continue
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        ok_idx.append(idx)
    if not grays:
        raise RuntimeError("no decodable candidate frames")
    Hs, Ws = grays[0].shape

    # Scope mask: endoscope footage is a bright circle on a black vignette
    # (~67% of every arthroscopy frame is border). Whole-frame stats are
    # meaningless there — everything below is computed inside this mask.
    mask = (np.max(np.stack(grays), axis=0) > 25).astype(np.uint8)
    ncc, lab = cv2.connectedComponents(mask)
    if ncc > 1:
        biggest = 1 + int(np.argmax([(lab == i).sum() for i in range(1, ncc)]))
        mask = (lab == biggest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.erode(mask, np.ones((21, 21), np.uint8))
    full_frame = mask.mean() > 0.90  # no vignette (e.g. laparoscopy) → whole frame
    if full_frame:
        mask = np.ones((Hs, Ws), np.uint8)
    M = mask.astype(bool)
    border = None if full_frame else ~M
    print(f"[ingest] scope mask covers {M.mean()*100:.0f}% "
          f"({'full-frame fallback' if full_frame else 'circle'})")

    # ---- pass B: re-decode, compute in-mask signals per candidate ----
    candidates = []
    for gi, idx in enumerate(ok_idx):
        try:
            rgb = _small(vr[idx].asnumpy())
        except Exception:
            continue
        g = grays[gi]
        gb = cv2.GaussianBlur(g, (0, 0), 2.0)
        gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1)
        grad = float(np.sqrt(gx ** 2 + gy ** 2)[M].mean())
        v = g[M]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hch, sch = hsv[..., 0][M], hsv[..., 1][M]
        # legacy columns, kept for continuity in the index/UI
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        r_, g_, b_ = (rgb[..., i].astype(np.float32) for i in (0, 1, 2))
        rg = r_ - g_
        yb = 0.5 * (r_ + g_) - b_
        colorf = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        candidates.append({
            "frame_index": int(idx),
            "timestamp_s": float(idx / fps),
            "grad": grad,
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
          f"(dropped: {drop_str}; grad_med={grad_med:.1f}, border_base={border_base:.1f})")
    if not gated:
        raise RuntimeError(f"domain gate rejected all {len(candidates)} candidates ({drop_str})")

    # ---- score + select: bookends first, then best-per-window ----
    g_rank = _rank([c["grad"] for c in gated])
    c_rank = _rank([c["contrast"] for c in gated])
    for c, a, b in zip(gated, g_rank, c_rank):
        c["score"] = 0.6 * a + 0.4 * b

    window_s = max(2.0, duration_s / MAX_FRAMES) if duration_s > 0 else 2.0
    by_time = sorted(gated, key=lambda x: x["timestamp_s"])
    selected = []
    # Bookends: guarantee the entry moment + final state are represented.
    selected.append(by_time[0])
    if len(by_time) > 1 and by_time[-1]["timestamp_s"] - by_time[0]["timestamp_s"] >= 1.0:
        selected.append(by_time[-1])
    for c in sorted(gated, key=lambda x: -x["score"]):
        if len(selected) >= MAX_FRAMES:
            break
        if any(abs(c["timestamp_s"] - s["timestamp_s"]) < window_s for s in selected):
            continue
        selected.append(c)
    selected.sort(key=lambda x: x["timestamp_s"])
    print(f"[ingest] selected {len(selected)} frames "
          f"(window={window_s:.2f}s, incl. bookends at "
          f"{selected[0]['timestamp_s']:.1f}s / {selected[-1]['timestamp_s']:.1f}s)")

    # ---- write picks (full-res) + register in Lakebase ----
    # (note: no `# COMMAND ----------` inside try — Databricks would split the
    # cell and break the try/except across cells)
    video_stem = Path(VIDEO_NAME).stem
    frame_rows = []
    for c in selected:
        ts = c["timestamp_s"]
        filename = f"{video_stem}_frame_{ts:07.2f}s.jpg"
        out_path = f"{OUTPUT_DIR}/{filename}"
        Image.fromarray(vr[c["frame_index"]].asnumpy()).save(out_path, format="JPEG", quality=92)
        frame_rows.append({**c, "frame_path": out_path, "frame_name": filename})

    with _conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM extracted_frames_index WHERE video_id = %s", (VIDEO_ID,))
        for r in frame_rows:
            cur.execute("""
                INSERT INTO extracted_frames_index
                    (video_id, frame_path, frame_name, timestamp_s, score,
                     sharpness, colorfulness, contrast, grad,
                     selected, selector_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s)
                ON CONFLICT (frame_path) DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    timestamp_s = EXCLUDED.timestamp_s,
                    score = EXCLUDED.score,
                    sharpness = EXCLUDED.sharpness,
                    colorfulness = EXCLUDED.colorfulness,
                    contrast = EXCLUDED.contrast,
                    grad = EXCLUDED.grad,
                    selected = true,
                    selector_version = EXCLUDED.selector_version,
                    updated_at = now()
            """, (
                VIDEO_ID, r["frame_path"], r["frame_name"], r["timestamp_s"],
                r["score"], r["sharpness"], r["colorfulness"], r["contrast"],
                r["grad"], SELECTOR_VERSION,
            ))
    print(f"[ingest][db] registered {len(frame_rows)} frames")

    elapsed = time.time() - t0
    _set_video_status(
        "ready",
        f"extracted {len(frame_rows)} frames in {elapsed:.0f}s · "
        f"gated {len(gated)}/{len(candidates)} candidates (dropped: {drop_str})",
        n_frames=len(frame_rows),
    )
    _exit_payload = json.dumps({
        "video_id": VIDEO_ID,
        "n_frames": len(frame_rows),
        "n_candidates": len(candidates),
        "gate_drops": dict(drops),
        "selector_version": SELECTOR_VERSION,
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

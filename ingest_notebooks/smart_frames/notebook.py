# Databricks notebook source
# MAGIC %md
# MAGIC # Smart frame extraction (ingest)
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
# MAGIC pg_password: <oauth bearer minted by the app at submit time>
# MAGIC candidate_fps: 1.0
# MAGIC max_frames: 40
# MAGIC ```
# MAGIC
# MAGIC Extracts best-per-window frames using the same sharpness + colorfulness
# MAGIC + contrast heuristic as ``notebooks/13_smart_frame_extraction.py``,
# MAGIC writes JPGs to ``output_dir/``, and registers them in the Lakebase
# MAGIC ``extracted_frames_index`` table so the app can list them instantly
# MAGIC without touching the Files API.

# COMMAND ----------
# MAGIC %pip install -q --upgrade decord pillow numpy psycopg[binary] pyyaml
# MAGIC %restart_python

# COMMAND ----------
import os, json, time, uuid, math, traceback
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
# Lakebase endpoint path — used by the notebook to mint its OWN Postgres
# token via /api/2.0/postgres/credentials. We deliberately do NOT accept a
# pre-minted password from the caller anymore: writing a bearer to a
# Volume-stored YAML leaks it to any reader with READ VOLUME, and tokens
# live ~1h.
PG_ENDPOINT = cfg.get("pg_endpoint", "")  # e.g. projects/vlm/branches/production/endpoints/primary
PG_INSTANCE = cfg.get("pg_instance", "")  # for legacy Provisioned fallback

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[ingest] video={VIDEO_NAME} video_id={VIDEO_ID}")
print(f"[ingest] output={OUTPUT_DIR}  candidate_fps={CANDIDATE_FPS}  max_frames={MAX_FRAMES}")

# COMMAND ----------
# Mint a fresh Lakebase Postgres token using THIS notebook's identity (which
# is the App SP for SP-submitted runs). Tokens live ~1h, so we mint once at
# the top of the notebook — long-running extractions may need a refresh
# helper if they exceed that, but the smart-frame pipeline is typically <5min.
from databricks.sdk import WorkspaceClient
_sdk = WorkspaceClient()

def _mint_pg_token() -> str:
    # Try Project API first
    if PG_ENDPOINT:
        try:
            r = _sdk.api_client.do("POST", "/api/2.0/postgres/credentials",
                                   body={"endpoint": PG_ENDPOINT})
            t = (r or {}).get("token")
            if t: return t
        except Exception as e:
            print(f"[ingest] project credential mint failed: {e}")
    # Provisioned fallback
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

# Connect to Lakebase to update video status + register frames
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
        print(f"[ingest][db] status={status}: {message[:80]}")
    except Exception as e:
        print(f"[ingest][db][warn] failed to update video status: {e}")

_set_video_status("processing", "loading video")

# COMMAND ----------
# Frame extraction — straightforward port of notebooks/13_smart_frame_extraction.py
from decord import VideoReader, cpu

def _laplacian_var(gray):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(gray.astype(np.float32), 1, mode="reflect")
    win = sliding_window_view(pad, (3, 3))
    lap = (win * kernel).sum(axis=(-1, -2))
    return float(np.var(lap))

def _to_gray(img_np):
    return (0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2])

def _colorfulness(img_np):
    rg = img_np[..., 0].astype(np.float32) - img_np[..., 1].astype(np.float32)
    yb = 0.5 * (img_np[..., 0].astype(np.float32) + img_np[..., 1].astype(np.float32)) - img_np[..., 2].astype(np.float32)
    rg_mean, rg_std = float(np.mean(rg)), float(np.std(rg))
    yb_mean, yb_std = float(np.mean(yb)), float(np.std(yb))
    return math.sqrt(rg_std**2 + yb_std**2) + 0.3 * math.sqrt(rg_mean**2 + yb_mean**2)

def _contrast(gray):
    return float(np.std(gray))

# COMMAND ----------
try:
    t0 = time.time()
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    n_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration_s = n_frames / max(fps, 0.001)
    print(f"[ingest] loaded video: {n_frames} frames, {fps:.1f} fps, {duration_s:.1f}s")
    _set_video_status("processing", f"{n_frames} frames @ {fps:.1f}fps", duration_s=duration_s)

    # Sample 1 candidate per second (or candidate_fps)
    step = max(1, int(fps / CANDIDATE_FPS))
    indices = list(range(0, n_frames, step))
    print(f"[ingest] scoring {len(indices)} candidate frames (1 per {step} frames)")

    candidates = []
    for idx in indices:
        try:
            frame = vr[idx].asnumpy()
        except Exception:
            continue
        gray = _to_gray(frame)
        sharp = _laplacian_var(gray)
        color = _colorfulness(frame)
        contr = _contrast(gray)
        candidates.append({
            "frame_index": int(idx),
            "timestamp_s": float(idx / fps),
            "sharpness": sharp,
            "colorfulness": color,
            "contrast": contr,
        })

    # Rank-normalize within video → [0,1], then weighted sum
    def _rank(values):
        idxs = np.argsort(np.argsort(values))
        n = len(values)
        return [float(i / (n - 1)) if n > 1 else 0.5 for i in idxs]

    s_rank = _rank([c["sharpness"] for c in candidates])
    co_rank = _rank([c["colorfulness"] for c in candidates])
    ct_rank = _rank([c["contrast"] for c in candidates])
    for c, s, co, ct in zip(candidates, s_rank, co_rank, ct_rank):
        c["sharpness_rank"] = s
        c["colorfulness_rank"] = co
        c["contrast_rank"] = ct
        c["score"] = 0.45 * s + 0.25 * co + 0.30 * ct

    # Pick the top frame per ~window, capped at MAX_FRAMES
    window_s = max(2.0, duration_s / MAX_FRAMES) if duration_s > 0 else 2.0
    selected = []
    last_t = -1e9
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if abs(c["timestamp_s"] - last_t) < (window_s * 0.8) and any(
            abs(c["timestamp_s"] - s["timestamp_s"]) < (window_s * 0.5) for s in selected
        ):
            continue
        selected.append(c)
        last_t = c["timestamp_s"]
        if len(selected) >= MAX_FRAMES:
            break
    selected.sort(key=lambda x: x["timestamp_s"])
    print(f"[ingest] selected {len(selected)} smart frames")

    # Write JPGs + register each frame in Lakebase
    # (note: cannot use `# COMMAND ----------` mid-try — Databricks treats
    # any line matching that pattern as a cell separator, splitting the
    # try/except across cells and producing a SyntaxError on the first half)
    video_stem = Path(VIDEO_NAME).stem
    frame_rows = []
    for c in selected:
        ts = c["timestamp_s"]
        frame_idx = c["frame_index"]
        # Same filename convention as the existing extracted_frames dir
        filename = f"{video_stem}_frame_{ts:07.2f}s.jpg"
        out_path = f"{OUTPUT_DIR}/{filename}"
        frame_np = vr[frame_idx].asnumpy()
        Image.fromarray(frame_np).save(out_path, format="JPEG", quality=92)
        frame_rows.append({
            "frame_path": out_path,
            "frame_name": filename,
            "timestamp_s": ts,
            "score": c["score"],
            "sharpness": c["sharpness"],
            "colorfulness": c["colorfulness"],
            "contrast": c["contrast"],
        })

    # Insert via Lakebase. Idempotent: ON CONFLICT (frame_path) DO UPDATE.
    with _conn() as conn, conn.cursor() as cur:
        # First clear any previous frames for this video so re-ingest is clean
        cur.execute("DELETE FROM extracted_frames_index WHERE video_id = %s", (VIDEO_ID,))
        for r in frame_rows:
            cur.execute("""
                INSERT INTO extracted_frames_index
                    (video_id, frame_path, frame_name, timestamp_s, score,
                     sharpness, colorfulness, contrast)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (frame_path) DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    timestamp_s = EXCLUDED.timestamp_s,
                    score = EXCLUDED.score,
                    sharpness = EXCLUDED.sharpness,
                    colorfulness = EXCLUDED.colorfulness,
                    contrast = EXCLUDED.contrast,
                    updated_at = now()
            """, (
                VIDEO_ID, r["frame_path"], r["frame_name"], r["timestamp_s"],
                r["score"], r["sharpness"], r["colorfulness"], r["contrast"],
            ))
    print(f"[ingest][db] registered {len(frame_rows)} frames")

    elapsed = time.time() - t0
    _set_video_status("ready", f"extracted {len(frame_rows)} frames in {elapsed:.0f}s", n_frames=len(frame_rows))
    _exit_payload = json.dumps({
        "video_id": VIDEO_ID,
        "n_frames": len(frame_rows),
        "elapsed_s": elapsed,
        "status": "ready",
    })

except Exception as e:
    # `dbutils.notebook.exit()` raises an exception itself, so it has to
    # live OUTSIDE the try — otherwise the normal happy-path exit gets
    # caught here and overwrites status='ready' with status='error'.
    tb = traceback.format_exc()
    print(f"[ingest][error] {e}\n{tb}")
    _set_video_status("error", str(e)[:300])
    raise
else:
    dbutils.notebook.exit(_exit_payload)

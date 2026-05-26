# Databricks notebook source
# MAGIC %md
# MAGIC # Surgical VLM smoke test
# MAGIC
# MAGIC Pulls `mehti/medgemma-cataract-surgical-analysis` from HuggingFace and
# MAGIC runs it on a handful of frames from the workbench's `eval_frames/` set,
# MAGIC then dumps results to a Volume path so we can inspect raw outputs +
# MAGIC compare to gold labels.
# MAGIC
# MAGIC Purpose: verify the model loads cleanly on 1× A10 and produces usable
# MAGIC output BEFORE integrating it into the app's Local-model list.
# MAGIC
# MAGIC Run as: serverless GPU job, GPU_1xA10, base_environment=databricks_ai_v4.

# COMMAND ----------
# MAGIC %pip install -q --upgrade "transformers>=4.57" accelerate pillow "psycopg[binary]"
# MAGIC %restart_python

# COMMAND ----------
import os, json, time, traceback
from pathlib import Path

os.environ["HF_TOKEN"] = dbutils.secrets.get("hls_g4", "HF_TOKEN")

MODEL_ID = "mehti/medgemma-cataract-surgical-analysis"
EVAL_FRAMES_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/eval_frames"
OUTPUT_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/_surgical_vlm_test"
N_FRAMES = 5
PROMPT = (
    'Identify the surgical instrument by its visible visual features. '
    'Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", '
    '"tissue_condition": "<short>"} where <class> is one of: probe, shaver, '
    'burr, grasper, biter, suture_passer, anchor_driver, electrocautery, '
    'cannula, scissors, drill_guide, trocar, knot_pusher, rasp, '
    'other_metal_tool, no_instrument_visible.'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[surg-vlm-test] model={MODEL_ID}")
print(f"[surg-vlm-test] output_dir={OUTPUT_DIR}")

# COMMAND ----------
import torch, transformers
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU attached. Submit with hardware_accelerator=GPU_1xA10.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.0f}GB")

# COMMAND ----------
# Snapshot the model into a temporary local cache so we can reuse it later.
# This is the same trick the workbench's setup_cache notebook uses.
from huggingface_hub import snapshot_download

t0 = time.time()
local_dir = "/tmp/_surg_vlm_cache"
print(f"[surg-vlm-test] snapshot_download → {local_dir}")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.*"],
)
print(f"[surg-vlm-test] download done in {time.time()-t0:.0f}s")

# COMMAND ----------
from transformers import AutoProcessor, AutoModelForImageTextToText
t0 = time.time()
processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    local_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
model.eval()
print(f"[surg-vlm-test] model loaded in {time.time()-t0:.0f}s")

# COMMAND ----------
# Find some eval frames + their gold labels (read from Lakebase via psycopg)
import psycopg
LAKEBASE_HOST = "instance-6b59171b-cee8-4acc-9209-6c848ffbfbfe.database.cloud.databricks.com"
LAKEBASE_DBNAME = "vlm_workbench"

# Mint a bearer token to use as Postgres password. Falls back to PAT for local dev.
def _mint_pg_token():
    try:
        # In a notebook this is the user's identity, so we use email + the
        # workspace token directly (the user has the right role).
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return w.config.token, w.current_user.me().user_name
    except Exception:
        raise

pg_password, pg_user = _mint_pg_token()
print(f"[surg-vlm-test] connecting to Lakebase as {pg_user}")

try:
    conn = psycopg.connect(
        host=LAKEBASE_HOST, dbname=LAKEBASE_DBNAME,
        user=pg_user, password=pg_password, sslmode="require", connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT frame_path, instrument FROM frame_labels ORDER BY updated_at DESC LIMIT %s",
            (N_FRAMES,),
        )
        labels = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    print(f"[surg-vlm-test] {len(labels)} labeled frames pulled from Lakebase")
except Exception as e:
    print(f"[surg-vlm-test][warn] Lakebase pull failed: {e}")
    # Fall back: scan EVAL_FRAMES_DIR
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    labels = {}
    for entry in w.files.list_directory_contents(EVAL_FRAMES_DIR):
        if entry.name and entry.name.lower().endswith(".jpg") and len(labels) < N_FRAMES:
            labels[entry.path] = "unknown"

# COMMAND ----------
from PIL import Image
def predict(image_path, prompt):
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    in_len = inputs["input_ids"].shape[-1]
    return processor.decode(out[0, in_len:], skip_special_tokens=True)

# COMMAND ----------
results = []
for path, gold in labels.items():
    t0 = time.time()
    try:
        raw = predict(path, PROMPT)
        elapsed = time.time() - t0
        # Try to parse JSON; if not, leave it raw
        parsed = None
        try:
            s, e = raw.find("{"), raw.rfind("}")
            if 0 <= s < e:
                parsed = json.loads(raw[s:e+1])
        except Exception:
            pass
        results.append({
            "frame_path": path, "gold": gold, "raw": raw, "parsed": parsed,
            "elapsed_s": round(elapsed, 2), "ok": True,
        })
        print(f"\n--- {Path(path).name}  gold={gold}  elapsed={elapsed:.1f}s ---")
        print(raw[:400])
    except Exception as e:
        tb = traceback.format_exc()
        results.append({"frame_path": path, "gold": gold, "ok": False, "error": str(e), "trace": tb[-1000:]})
        print(f"FAIL on {path}: {e}")

# COMMAND ----------
# Quick accuracy: did the parsed instrument match gold?
def _extract(p):
    if not isinstance(p, dict): return None
    if isinstance(p.get("instrument"), str):
        v = p["instrument"].strip()
        return v.lower() if v else "no_instrument_visible"
    inst = p.get("instruments")
    if isinstance(inst, list):
        if not inst: return "no_instrument_visible"
        if isinstance(inst[0], dict) and isinstance(inst[0].get("class"), str):
            return inst[0]["class"].strip().lower() or "no_instrument_visible"
    return None

n_correct, n_total = 0, 0
for r in results:
    if not r.get("ok"): continue
    gold = (r.get("gold") or "").lower()
    pred = _extract(r.get("parsed"))
    if gold == "unknown":
        continue
    n_total += 1
    if pred == gold:
        n_correct += 1
    print(f"  gold={gold!r}  pred={pred!r}  match={pred == gold}")
print(f"\n=== {n_correct}/{n_total} match ===")

out_path = f"{OUTPUT_DIR}/result.json"
with open(out_path, "w") as f:
    json.dump({
        "model": MODEL_ID,
        "n_frames": len(results),
        "n_correct": n_correct,
        "n_total": n_total,
        "results": results,
    }, f, indent=2)
print(f"Wrote {out_path}")

dbutils.notebook.exit(json.dumps({
    "model": MODEL_ID,
    "n_correct": n_correct, "n_total": n_total,
    "result_path": out_path,
}))

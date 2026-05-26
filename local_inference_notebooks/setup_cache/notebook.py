# Databricks notebook source
# MAGIC %md
# MAGIC # Pre-download local VLM weights to Volume
# MAGIC
# MAGIC One-shot job that snapshots Qwen3-VL-8B and MedGemma-4B from Hugging Face
# MAGIC into a Unity Catalog Volume so the inference notebook can load them
# MAGIC without re-downloading every cold start. Reads each model's
# MAGIC ``manifest.yaml`` for the HF repo + revision.

# COMMAND ----------
# MAGIC %pip install -q --upgrade huggingface_hub pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import time
import yaml
from pathlib import Path

LOCAL_MODELS_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/local_models"

dbutils.widgets.text("hf_token", "", "HF token override (else read from secret)")
dbutils.widgets.text("hf_secret_scope", "hls_g4", "Databricks secret scope")
dbutils.widgets.text("hf_secret_key", "HF_TOKEN", "Databricks secret key")

# Prefer the widget override; fall back to dbutils.secrets so the job can run
# unattended on a schedule. The token unlocks gated repos like
# google/medgemma-4b-it.
HF_TOKEN = dbutils.widgets.get("hf_token").strip() or None
if not HF_TOKEN:
    try:
        HF_TOKEN = dbutils.secrets.get(
            dbutils.widgets.get("hf_secret_scope"),
            dbutils.widgets.get("hf_secret_key"),
        )
    except Exception as e:
        print(f"[warn] could not read HF token from secret: {e}")
        HF_TOKEN = None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print("HF_TOKEN set (length:", len(HF_TOKEN), ")")
else:
    print("[warn] no HF token available — gated repos will fail")

# COMMAND ----------
from huggingface_hub import snapshot_download

def find_models() -> list[dict]:
    out = []
    for child in sorted(Path(LOCAL_MODELS_DIR).iterdir()):
        manifest = child / "manifest.yaml"
        if not manifest.exists():
            continue
        with manifest.open() as f:
            cfg = yaml.safe_load(f) or {}
        cfg["_dir"] = str(child)
        out.append(cfg)
    return out

models = find_models()
print(f"Found {len(models)} model manifest(s):")
for m in models:
    print(f"  - {m.get('name')} → {m.get('hf_repo')} (rev {m.get('revision','main')})")

# COMMAND ----------
def download(model_cfg: dict) -> dict:
    name = model_cfg["name"]
    hf_repo = model_cfg["hf_repo"]
    revision = model_cfg.get("revision", "main")
    target = f"{model_cfg['_dir']}/snapshot"
    os.makedirs(target, exist_ok=True)
    print(f"\n[{name}] → {target}")
    t0 = time.time()
    snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        local_dir=target,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
        # Skip giant artifacts we don't need for transformers loading
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.*"],
    )
    elapsed = time.time() - t0
    bytes_total = sum(p.stat().st_size for p in Path(target).rglob("*") if p.is_file())
    gb = bytes_total / 1e9
    print(f"[{name}] cached {gb:.2f} GB in {elapsed:.0f}s")
    return {"name": name, "path": target, "bytes": bytes_total, "elapsed_s": elapsed}

import traceback
results = []
for m in models:
    try:
        results.append(download(m))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{m.get('name')}] FAILED: {e}\n{tb}")
        results.append({"name": m.get("name"), "error": str(e), "trace": tb[-2000:]})

# COMMAND ----------
# Dump a structured log to the Volume so callers (the app, the operator) can
# read what actually happened — Databricks SUBMIT_RUN doesn't surface stdout.
LOG_PATH = f"{LOCAL_MODELS_DIR}/_last_cache_run.json"
import json
with open(LOG_PATH, "w") as f:
    json.dump({
        "results": results,
        "token_present": bool(HF_TOKEN),
        "token_length": len(HF_TOKEN) if HF_TOKEN else 0,
    }, f, indent=2)
print(f"Wrote log to {LOG_PATH}")

print("Cache results:")
for r in results:
    print(" ", r)

# Surface the structured result back through dbutils so it shows up in the
# notebook task's notebook_output, visible via `jobs get-run-output`.
dbutils.notebook.exit(json.dumps({"results": results, "token_present": bool(HF_TOKEN)}))

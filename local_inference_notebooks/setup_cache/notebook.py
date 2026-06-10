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

dbutils.widgets.text("local_models_dir", "", "Absolute path under a UC Volume — defaults to env LOCAL_MODELS_DIR")
dbutils.widgets.text("hf_token", "", "HF token override (else read from secret)")
dbutils.widgets.text("hf_secret_scope", "vlmwb_hf", "Databricks secret scope")
dbutils.widgets.text("hf_secret_key", "HF_TOKEN", "Databricks secret key")
dbutils.widgets.text("models_json", "", "JSON list of models to cache, e.g. [{name, hf_repo, revision?}] — if empty, scans LOCAL_MODELS_DIR for manifest.yaml files")

import json as _json
LOCAL_MODELS_DIR = (
    dbutils.widgets.get("local_models_dir").strip()
    or os.environ.get("LOCAL_MODELS_DIR")
    or "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/local_models"
)
MODELS_OVERRIDE = []
_models_raw = dbutils.widgets.get("models_json").strip()
if _models_raw:
    try:
        MODELS_OVERRIDE = _json.loads(_models_raw)
        if not isinstance(MODELS_OVERRIDE, list):
            raise ValueError("models_json must be a JSON list")
    except Exception as e:
        raise SystemExit(f"bad models_json: {e}")
print(f"[setup_cache] LOCAL_MODELS_DIR={LOCAL_MODELS_DIR}")
print(f"[setup_cache] models_override={len(MODELS_OVERRIDE)} entries")

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

def materialize_overrides(specs: list[dict]) -> list[dict]:
    """Given [{name, hf_repo, revision?}], write a manifest.yaml under
    LOCAL_MODELS_DIR/<name>/ so subsequent runs find them via find_models()
    and the inference notebook can resolve snapshot_dir."""
    out = []
    for spec in specs:
        name = spec.get("name")
        hf_repo = spec.get("hf_repo")
        if not (name and hf_repo):
            print(f"[skip] missing name/hf_repo in {spec!r}")
            continue
        d = Path(LOCAL_MODELS_DIR) / name
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "hf_repo": hf_repo,
            "revision": spec.get("revision", "main"),
            "provider": spec.get("provider", "huggingface_local"),
        }
        # Optional manifest fields — only written when the caller specified
        # them. The app's local-run submitter reads each at runtime to pick
        # the right accelerator/env/notebook for this specific model.
        # - accelerator        ("GPU_1xA10" | "GPU_1xH100" | "GPU_8xH100"): needed for
        #                      bigger models that won't fit on A10
        # - base_environment   ("databricks_ai_v4" | "5"): controls the
        #                      preinstalled torch/CUDA/transformers
        # - inference_notebook ("run" | "run_gemma4"): which inference
        #                      notebook to dispatch to (some model families
        #                      need a different env/recipe)
        for opt in ("accelerator", "base_environment", "inference_notebook"):
            if spec.get(opt):
                manifest[opt] = spec[opt]
        with (d / "manifest.yaml").open("w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        manifest["_dir"] = str(d)
        out.append(manifest)
    return out

if MODELS_OVERRIDE:
    models = materialize_overrides(MODELS_OVERRIDE)
else:
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

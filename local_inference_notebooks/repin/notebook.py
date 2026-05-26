# Databricks notebook source
# MAGIC %md
# MAGIC # Re-pin requirements on an existing UC model
# MAGIC
# MAGIC When a previous fine-tune was registered without ``torchvision`` pinned,
# MAGIC Databricks Model Serving installs a CUDA-mismatched torchvision wheel at
# MAGIC deploy time and crashes with ``operator torchvision::nms does not exist``.
# MAGIC This notebook reads the merged-model dir from a YAML config, re-runs
# MAGIC ``mlflow.transformers.log_model`` with the correct pin set, and registers
# MAGIC a new version on the same UC model so it deploys cleanly.

# COMMAND ----------
# MAGIC %pip install -q --upgrade "transformers>=4.57" mlflow pillow safetensors tokenizers
# MAGIC %restart_python

# COMMAND ----------
import os, json, yaml, tempfile, time
import torch
import transformers
import torchvision as _tv

dbutils.widgets.text("config_yaml", "", "Path to YAML config")
yaml_path = dbutils.widgets.get("config_yaml").strip()
if not yaml_path:
    raise SystemExit("config_yaml widget is required")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}

UC_FULL = cfg["uc_full_name"]
MERGED_DIR = cfg["merged_dir"]
OUTPUT_DIR = cfg["output_dir"]
EXPERIMENT_PATH = cfg.get("mlflow_experiment") or "/Users/guanyu.chen@databricks.com/vlmwb-experiments"

print(f"[repin] target: {UC_FULL}")
print(f"[repin] merged_dir: {MERGED_DIR}")
print(f"[repin] transformers={transformers.__version__} torch={torch.__version__} torchvision={_tv.__version__}")

# COMMAND ----------
# Sanity-check that the merged dir is still there (UC Volumes don't get GC'd
# automatically, but worth catching the case where the user deleted it).
if not os.path.isdir(MERGED_DIR):
    raise SystemExit(f"merged dir not found: {MERGED_DIR}. Re-run the original fine-tune.")
missing = [n for n in ("config.json",) if not os.path.exists(os.path.join(MERGED_DIR, n))]
if missing:
    raise SystemExit(f"merged dir is incomplete (missing {missing})")

# COMMAND ----------
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
except Exception as e:
    print(f"[repin][warn] set_experiment failed: {e}")

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "prompt"), ColSpec("binary", "image")]),
    outputs=Schema([ColSpec("string", "generated_text")]),
)

# Capture the optional deps so we can pin them
try:
    import safetensors as _sft
    sft_ver = _sft.__version__
except Exception:
    sft_ver = None
try:
    import tokenizers as _tk
    tk_ver = _tk.__version__
except Exception:
    tk_ver = None
torch_ver = torch.__version__.split("+")[0]
tv_ver = _tv.__version__.split("+")[0]
pip_reqs = [
    f"transformers=={transformers.__version__}",
    f"torch=={torch_ver}",
    f"torchvision=={tv_ver}",
    "accelerate",
    "peft",
    "qwen-vl-utils",
    "Pillow",
]
if sft_ver: pip_reqs.append(f"safetensors=={sft_ver}")
if tk_ver: pip_reqs.append(f"tokenizers=={tk_ver}")
print(f"[repin] new pip_requirements: {pip_reqs}")

# COMMAND ----------
with mlflow.start_run(run_name=f"repin_{UC_FULL.split('.')[-1]}") as run:
    mlflow.log_params({
        "kind": "repin_requirements",
        "uc_model": UC_FULL,
        "merged_dir": MERGED_DIR,
    })
    mlflow.transformers.log_model(
        transformers_model=MERGED_DIR,
        name="finetuned_model",
        registered_model_name=UC_FULL,
        task="image-text-to-text",
        signature=signature,
        metadata={"repin_reason": "torchvision/torch CUDA mismatch fix"},
        pip_requirements=pip_reqs,
    )
    mlflow_url = f"{mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

# COMMAND ----------
# Look up the new version we just registered so we can hand it back.
client = mlflow.tracking.MlflowClient()
vs = client.search_model_versions(filter_string=f"name='{UC_FULL}'", max_results=20)
latest = max(vs, key=lambda v: int(v.version))

os.makedirs(OUTPUT_DIR, exist_ok=True)
result = {
    "uc_model": UC_FULL,
    "new_version": latest.version,
    "mlflow_url": mlflow_url,
    "pip_requirements": pip_reqs,
}
with open(f"{OUTPUT_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"[repin] registered new version: {UC_FULL} v{latest.version}")
dbutils.notebook.exit(json.dumps(result))

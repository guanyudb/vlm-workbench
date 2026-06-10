# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy-test (TRANSFORMERS FLAVOR — broken at serve time; use notebook_pyfunc.py instead)
# MAGIC
# MAGIC NOTE: This notebook uses `mlflow.transformers.log_model`. The model
# MAGIC REGISTERS and RELOADS fine in-notebook, but FAILS at Model Serving load
# MAGIC time with:
# MAGIC
# MAGIC   Exception: Impossible to guess which processor to use. Please provide
# MAGIC   a processor instance or a path/identifier to a processor.
# MAGIC
# MAGIC because mlflowserving calls `transformers.pipeline(**conf)` without
# MAGIC `trust_remote_code=True` — required for Qwen3-VL's custom processor.
# MAGIC
# MAGIC Use `notebook_pyfunc.py` instead — that wraps the model in a
# MAGIC mlflow.pyfunc.PythonModel whose load_context calls AutoModel/AutoProcessor
# MAGIC with trust_remote_code=True directly, sidestepping the broken auto-resolution.
# MAGIC
# MAGIC This notebook is kept so the failure mode is documented for future readers.
# MAGIC
# MAGIC This notebook uses the **known-good pattern** from
# MAGIC `~/Documents/Projects/Cursor/vlm_finetune/02_register_base_model.ipynb`:
# MAGIC
# MAGIC 1. Pin `transformers>=4.57.0` at the top so Qwen3-VL's tokenizer doesn't blow up.
# MAGIC 2. Load model+processor from the on-disk checkpoint dir.
# MAGIC 3. Save them to a local tempdir, then pass the **PATH** to
# MAGIC    `mlflow.transformers.log_model(transformers_model=<path>, ...)`.
# MAGIC    Passing a dict triggers MLflow's Pipeline-creation validation, which
# MAGIC    fails for multimodal Processor models.
# MAGIC 4. Reload via `AutoModelForImageTextToText.from_pretrained(model_dir, trust_remote_code=True)`
# MAGIC    — `mlflow.transformers.load_model` does NOT forward `trust_remote_code`,
# MAGIC    so for custom-arch models we resolve the artifact dir and load manually.
# MAGIC 5. Smoke-test inference on one frame and write `result.json`.
# MAGIC
# MAGIC Reads its config from a YAML file passed via the `config_yaml` widget so
# MAGIC this can be submitted as a Databricks notebook job from the app.
# MAGIC
# MAGIC ## How to read serving build/server logs (the user's specific ask)
# MAGIC
# MAGIC ```bash
# MAGIC # Build logs (docker image build for the served entity):
# MAGIC GET /api/2.0/serving-endpoints/{name}/served-models/{served_model_name}/build-logs
# MAGIC   → {"logs": "<stdout tail>"}
# MAGIC # served_model_name is the entry under config.served_entities[].name
# MAGIC # (the app uses f"{endpoint_name}-srv" in create_serving_endpoint).
# MAGIC
# MAGIC # Server logs (after the container is up):
# MAGIC GET /api/2.0/serving-endpoints/{name}/served-models/{served_model_name}/logs
# MAGIC   → {"logs": "..."}
# MAGIC
# MAGIC # Endpoint state — watch status.state.{ready,config_update}:
# MAGIC GET /api/2.0/serving-endpoints/{name}
# MAGIC
# MAGIC # CLI equivalents:
# MAGIC databricks --profile <p> serving-endpoints get <name>
# MAGIC databricks --profile <p> serving-endpoints get-build-logs <name> <served_model_name>
# MAGIC databricks --profile <p> serving-endpoints logs <name> <served_model_name>
# MAGIC ```

# COMMAND ----------
# MAGIC %pip install --quiet --upgrade pip
# MAGIC %pip install --quiet "transformers>=4.57.0" "accelerate>=0.26.0" "torch>=2.1.0" "torchvision>=0.16.0"
# MAGIC %pip install --quiet "pillow>=10.0.0" "pyyaml>=6.0" qwen-vl-utils
# MAGIC %pip install --quiet "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os, json, time, base64, glob, tempfile, traceback, sys
from pathlib import Path
import yaml

dbutils.widgets.text("config_yaml", "", "Path to YAML config")
yaml_path = dbutils.widgets.get("config_yaml").strip()
if not yaml_path:
    raise SystemExit("config_yaml widget is required")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}

UC_FULL = cfg["uc_full_name"]
MODEL_DIR = cfg["model_dir"]
OUTPUT_DIR = cfg["output_dir"]
PROMPT = cfg.get(
    "prompt",
    "Identify the surgical instrument in this image. Respond with JSON like {\"instrument\": \"<name>\"}.",
)
TEST_IMAGE = cfg.get("test_image")
EXPERIMENT_PATH = cfg.get("mlflow_experiment") or "/Users/guanyu.chen@databricks.com/vlmwb-deploy-test-exp"
TORCH_DTYPE = cfg.get("torch_dtype", "bfloat16")
MAX_NEW_TOKENS = int(cfg.get("max_new_tokens", 200))

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[deploy-test] uc={UC_FULL}")
print(f"[deploy-test] model_dir={MODEL_DIR}")
print(f"[deploy-test] output_dir={OUTPUT_DIR}")
print(f"[deploy-test] test_image={TEST_IMAGE}")

# COMMAND ----------
# Sanity-check the model dir is there and complete enough to load the processor.
if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"model dir not found: {MODEL_DIR}")
for fname in ("config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json"):
    p = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(p):
        print(f"[deploy-test][warn] {fname} missing from {MODEL_DIR}")

# COMMAND ----------
import torch
import transformers
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU attached. Submit this notebook with a serverless GPU accelerator.")

# COMMAND ----------
# Load the checkpoint we plan to register. We load + save it to a local
# tempdir (faster local disk than Volume) and pass the PATH to log_model.
from transformers import AutoProcessor, AutoModelForImageTextToText

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
dtype = dtype_map.get(TORCH_DTYPE, torch.bfloat16)

print(f"[deploy-test] loading model from {MODEL_DIR} dtype={dtype}")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.eval()
print(f"[deploy-test] loaded in {time.time() - t0:.1f}s")

# COMMAND ----------
# Save model + processor to a local tempdir. This is the working pattern:
# passing transformers_model=<path> to log_model avoids MLflow's Pipeline
# validation, which fails for multimodal Processor models.
save_dir = tempfile.mkdtemp(prefix="qwen3vl_deploy_")
print(f"[deploy-test] saving model+processor to {save_dir}")
t0 = time.time()
model.save_pretrained(save_dir, safe_serialization=True)
processor.save_pretrained(save_dir)
print(f"[deploy-test] saved in {time.time() - t0:.1f}s")
print(f"[deploy-test] save_dir contents: {os.listdir(save_dir)}")

# COMMAND ----------
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
except Exception as e:
    print(f"[deploy-test][warn] set_experiment failed ({EXPERIMENT_PATH}): {e}")

# UC requires an explicit signature.
signature = ModelSignature(
    inputs=Schema([ColSpec("string", "prompt"), ColSpec("binary", "image")]),
    outputs=Schema([ColSpec("string", "generated_text")]),
)

# Strip CUDA build tag (e.g. "+cu126") from torch version so pip can resolve.
torch_version = torch.__version__.split("+")[0]
try:
    import torchvision as _tv
    tv_version = _tv.__version__.split("+")[0]
except Exception:
    tv_version = None

pip_reqs = [
    f"transformers=={transformers.__version__}",
    f"torch=={torch_version}",
    "accelerate>=0.26.0",
    "qwen-vl-utils",
    "Pillow>=10.0.0",
]
if tv_version:
    pip_reqs.append(f"torchvision=={tv_version}")
print(f"[deploy-test] pip_requirements: {pip_reqs}")

# COMMAND ----------
run_name = f"deploy-test-{int(time.time())}"
print(f"[deploy-test] registering {UC_FULL}")
with mlflow.start_run(run_name=run_name) as run:
    training_run_id = run.info.run_id
    mlflow.log_params({
        "kind": "deploy_test_transformers",
        "uc_model": UC_FULL,
        "source_model_dir": MODEL_DIR,
        "torch_dtype": TORCH_DTYPE,
        "transformers_version": transformers.__version__,
        "torch_version": torch_version,
    })
    t0 = time.time()
    info = mlflow.transformers.log_model(
        transformers_model=save_dir,         # PATH, not dict
        name="model",
        registered_model_name=UC_FULL,
        task="image-text-to-text",
        torch_dtype=dtype,
        signature=signature,
        metadata={
            "source_model_dir": MODEL_DIR,
            "default_prompt": PROMPT,
        },
        pip_requirements=pip_reqs,
    )
    print(f"[deploy-test] log_model took {time.time() - t0:.1f}s; model_uri={info.model_uri}")
    mlflow_url = f"{mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{training_run_id}"

# COMMAND ----------
# Pull the new version we just registered.
from mlflow.tracking import MlflowClient
client = MlflowClient()
vs = client.search_model_versions(filter_string=f"name='{UC_FULL}'", max_results=20)
latest = max(vs, key=lambda v: int(v.version))
NEW_VERSION = int(latest.version)
print(f"[deploy-test] registered UC version: {UC_FULL} v{NEW_VERSION}")

# COMMAND ----------
# Free the in-memory model so the reload below isn't masked by a still-loaded
# copy of the same weights.
import gc
try:
    del model
    del processor
except Exception:
    pass
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------
# Reload manually — mlflow.transformers.load_model doesn't forward
# trust_remote_code which Qwen3-VL needs for its custom arch.
print(f"[deploy-test] downloading models:/{UC_FULL}/{NEW_VERSION}")
t0 = time.time()
local_dir = mlflow.artifacts.download_artifacts(f"models:/{UC_FULL}/{NEW_VERSION}")
print(f"[deploy-test] download took {time.time() - t0:.1f}s; local_dir={local_dir}")

hits = glob.glob(os.path.join(local_dir, "**/config.json"), recursive=True)
reloaded_dir = os.path.dirname(hits[0]) if hits else local_dir
print(f"[deploy-test] resolved checkpoint dir: {reloaded_dir}")

t0 = time.time()
reloaded_model = AutoModelForImageTextToText.from_pretrained(
    reloaded_dir,
    device_map="auto",
    torch_dtype=dtype,
    trust_remote_code=True,
)
reloaded_processor = AutoProcessor.from_pretrained(reloaded_dir, trust_remote_code=True)
reloaded_model.eval()
print(f"[deploy-test] reloaded in {time.time() - t0:.1f}s")

# COMMAND ----------
# 1-frame inference smoke test.
from PIL import Image
import io as _io

if not TEST_IMAGE or not os.path.exists(TEST_IMAGE):
    raise SystemExit(f"test_image not found: {TEST_IMAGE}")
img = Image.open(TEST_IMAGE).convert("RGB")
print(f"[deploy-test] test image: {TEST_IMAGE}  size={img.size}")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT},
    ],
}]
inputs = reloaded_processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(reloaded_model.device)

t0 = time.time()
with torch.no_grad():
    out_ids = reloaded_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
in_len = inputs["input_ids"].shape[-1]
gen = out_ids[0, in_len:]
generated_text = reloaded_processor.decode(gen, skip_special_tokens=True)
print(f"[deploy-test] inference took {time.time() - t0:.1f}s")
print(f"[deploy-test] generated_text: {generated_text}")

# COMMAND ----------
# Try to parse the JSON answer for the result blob.
parsed_json = None
try:
    parsed_json = json.loads(generated_text.strip())
except Exception:
    # Try to extract JSON from a wrapping string (chat models sometimes pad
    # with backticks or "Answer: ").
    import re
    m = re.search(r"\{[^{}]*\}", generated_text)
    if m:
        try:
            parsed_json = json.loads(m.group(0))
        except Exception:
            parsed_json = None
print(f"[deploy-test] parsed_json: {parsed_json}")

# COMMAND ----------
result = {
    "uc_model": UC_FULL,
    "new_version": NEW_VERSION,
    "mlflow_run_id": training_run_id,
    "mlflow_url": mlflow_url,
    "pip_requirements": pip_reqs,
    "test_image": TEST_IMAGE,
    "generated_text": generated_text[:1000],
    "parsed_json": parsed_json,
    "reload_ok": True,
    "transformers_version": transformers.__version__,
    "torch_version": torch_version,
}
result_path = f"{OUTPUT_DIR}/result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[deploy-test] wrote {result_path}")
dbutils.notebook.exit(json.dumps(result))

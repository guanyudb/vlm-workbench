# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy-test (pyfunc): register VLM via mlflow.pyfunc + transformers>=4.57
# MAGIC
# MAGIC The mlflow.transformers flavor fails at SERVE time for Qwen3-VL because
# MAGIC mlflowserving calls `transformers.pipeline(**conf)` without
# MAGIC `trust_remote_code=True`, which then can't infer the custom processor.
# MAGIC
# MAGIC ```
# MAGIC Exception: Impossible to guess which processor to use. Please provide a
# MAGIC processor instance or a path/identifier to a processor.
# MAGIC ```
# MAGIC
# MAGIC The fix: register as `mlflow.pyfunc` with our own wrapper that loads
# MAGIC `AutoModelForImageTextToText` + `AutoProcessor` with `trust_remote_code=True`.
# MAGIC Combine that with `transformers>=4.57` (no more `AttributeError: 'list'
# MAGIC object has no attribute 'keys'`) and serving works end-to-end.

# COMMAND ----------
# MAGIC %pip install --quiet --upgrade pip
# MAGIC %pip install --quiet "transformers>=4.57.0" "accelerate>=0.26.0" "torch>=2.1.0" "torchvision>=0.16.0"
# MAGIC %pip install --quiet "pillow>=10.0.0" "pyyaml>=6.0" qwen-vl-utils
# MAGIC %pip install --quiet "mlflow[databricks]>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os, json, time, base64, io, sys, traceback
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
print(f"[deploy-test-pyfunc] uc={UC_FULL}")
print(f"[deploy-test-pyfunc] model_dir={MODEL_DIR}")

# COMMAND ----------
import torch, transformers
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU attached.")

# COMMAND ----------
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd

class VLMPyfunc(mlflow.pyfunc.PythonModel):
    """Generic VLM wrapper for Qwen3-VL, Gemma-3/4, and any model loadable
    via AutoProcessor + AutoModelForImageTextToText. Inference path matches
    local_inference_notebooks/run/notebook.py exactly. Uses trust_remote_code=True
    so custom-arch models like Qwen3-VL load correctly."""

    def load_context(self, context):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = context.artifacts["model"]
        cfg_path = os.path.join(context.artifacts.get("cfg", ""), "serve_config.json")
        try:
            with open(cfg_path) as f:
                serve_cfg = json.load(f)
        except Exception:
            serve_cfg = {}
        dtype_str = serve_cfg.get("torch_dtype", "bfloat16")
        self.max_new_tokens = int(serve_cfg.get("max_new_tokens", 200))
        self.default_prompt = serve_cfg.get("default_prompt", "Describe the image.")

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_str, torch.bfloat16)

        print(f"[VLMPyfunc] loading from {model_path} dtype={dtype}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[VLMPyfunc] model loaded")

    def _row_to_image(self, image_val):
        from PIL import Image
        import io as _io
        if image_val is None:
            raise ValueError("image is required")
        if isinstance(image_val, (bytes, bytearray)):
            raw = bytes(image_val)
        elif isinstance(image_val, str):
            raw = base64.b64decode(image_val, validate=False)
        else:
            raise ValueError(f"unsupported image type: {type(image_val)}")
        return Image.open(_io.BytesIO(raw)).convert("RGB")

    def predict(self, context, model_input, params=None):
        import torch
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = list(model_input)

        results = []
        for row in rows:
            prompt = row.get("prompt") or self.default_prompt
            img = self._row_to_image(row.get("image"))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=int(row.get("max_new_tokens") or self.max_new_tokens),
                    do_sample=False,
                )
            in_len = inputs["input_ids"].shape[-1]
            gen = out_ids[0, in_len:]
            text = self.processor.decode(gen, skip_special_tokens=True)
            results.append({"generated_text": text})
        return pd.DataFrame(results)


# COMMAND ----------
# Build serve_config artifact (dtype, max_new_tokens, default_prompt).
cfg_dir = f"{OUTPUT_DIR}/cfg"
os.makedirs(cfg_dir, exist_ok=True)
with open(f"{cfg_dir}/serve_config.json", "w") as f:
    json.dump({
        "torch_dtype": TORCH_DTYPE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "default_prompt": PROMPT,
    }, f)

# COMMAND ----------
torch_ver = torch.__version__.split("+")[0]
try:
    import torchvision as _tv
    tv_ver = _tv.__version__.split("+")[0]
except Exception:
    tv_ver = None

pip_reqs = [
    f"transformers=={transformers.__version__}",
    f"torch=={torch_ver}",
    "accelerate>=0.26.0",
    "qwen-vl-utils",
    "Pillow>=10.0.0",
]
if tv_ver:
    pip_reqs.append(f"torchvision=={tv_ver}")
print(f"[deploy-test-pyfunc] pip_requirements: {pip_reqs}")

# COMMAND ----------
# Build input example + signature.
with open(TEST_IMAGE, "rb") as f:
    example_b64 = base64.b64encode(f.read()).decode("ascii")
input_example = pd.DataFrame([{"prompt": PROMPT, "image": example_b64}])
output_example = pd.DataFrame([{"generated_text": "shaver"}])
signature = infer_signature(input_example, output_example)

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
except Exception as e:
    print(f"[warn] set_experiment failed: {e}")

run_name = f"deploy-test-pyfunc-{int(time.time())}"
with mlflow.start_run(run_name=run_name) as run:
    training_run_id = run.info.run_id
    mlflow.log_params({
        "kind": "deploy_test_pyfunc",
        "uc_model": UC_FULL,
        "source_model_dir": MODEL_DIR,
        "torch_dtype": TORCH_DTYPE,
    })
    info = mlflow.pyfunc.log_model(
        name="vlm",
        python_model=VLMPyfunc(),
        artifacts={"model": MODEL_DIR, "cfg": cfg_dir},
        registered_model_name=UC_FULL,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_reqs,
        metadata={"source_model_dir": MODEL_DIR, "default_prompt": PROMPT},
    )
    print(f"[deploy-test-pyfunc] logged model_uri={info.model_uri}")
    mlflow_url = f"{mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{training_run_id}"

# COMMAND ----------
from mlflow.tracking import MlflowClient
client = MlflowClient()
vs = client.search_model_versions(filter_string=f"name='{UC_FULL}'", max_results=20)
latest = max(vs, key=lambda v: int(v.version))
NEW_VERSION = int(latest.version)
print(f"[deploy-test-pyfunc] registered v{NEW_VERSION}")

# COMMAND ----------
# Reload via pyfunc — proves artifact + requirements are self-contained.
import gc
gc.collect()
torch.cuda.empty_cache()

loaded = mlflow.pyfunc.load_model(f"models:/{UC_FULL}/{NEW_VERSION}")
t0 = time.time()
pred = loaded.predict(input_example)
print(f"[deploy-test-pyfunc] inference took {time.time()-t0:.1f}s")
gen_text = pred.iloc[0]["generated_text"] if hasattr(pred, "iloc") else str(pred)
print(f"[deploy-test-pyfunc] generated_text: {gen_text}")

# COMMAND ----------
parsed_json = None
try:
    parsed_json = json.loads(gen_text.strip())
except Exception:
    import re
    m = re.search(r"\{[^{}]*\}", gen_text)
    if m:
        try:
            parsed_json = json.loads(m.group(0))
        except Exception:
            pass

result = {
    "uc_model": UC_FULL,
    "new_version": NEW_VERSION,
    "mlflow_run_id": training_run_id,
    "mlflow_url": mlflow_url,
    "pip_requirements": pip_reqs,
    "generated_text": gen_text[:1000],
    "parsed_json": parsed_json,
    "reload_ok": True,
    "transformers_version": transformers.__version__,
    "torch_version": torch_ver,
}
with open(f"{OUTPUT_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"[deploy-test-pyfunc] wrote {OUTPUT_DIR}/result.json")
dbutils.notebook.exit(json.dumps(result))

# Databricks notebook source
# MAGIC %md
# MAGIC # Local VLM inference (Qwen3-VL-8B / MedGemma-4B)
# MAGIC
# MAGIC Reads a YAML config (passed via the ``config_yaml`` widget) describing:
# MAGIC ```yaml
# MAGIC model_name: qwen3-vl-8b
# MAGIC model_dir: /Volumes/.../local_models/qwen3-vl-8b/snapshot
# MAGIC torch_dtype: bfloat16
# MAGIC max_new_tokens: 600
# MAGIC prompt: "Identify the surgical instrument..."
# MAGIC frame_paths:
# MAGIC   - /Volumes/.../eval_frames/foo.jpg
# MAGIC output_dir: /Volumes/.../local_runs/<run_id>
# MAGIC ```
# MAGIC Writes ``<output_dir>/results.json`` with one row per frame.

# COMMAND ----------
# MAGIC %pip install -q --upgrade transformers>=4.56.0 accelerate>=1.0.0 pyyaml
# MAGIC dbutils.library.restartPython()
# NOTE: torch + torchvision + CUDA come preinstalled in databricks_ai_v4.

# COMMAND ----------
import os, json, time, traceback, base64
from pathlib import Path
import yaml

dbutils.widgets.text("config_yaml", "", "Path to YAML config")
yaml_path = dbutils.widgets.get("config_yaml").strip()
if not yaml_path:
    raise SystemExit("config_yaml widget is required")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}
print("Loaded config:", {k: v for k, v in cfg.items() if k != "frame_paths"})
print(f"  frame_paths: {len(cfg.get('frame_paths', []))} frames")

MODEL_DIR = cfg["model_dir"]
MODEL_NAME = cfg.get("model_name", "unknown")
DTYPE_STR = cfg.get("torch_dtype", "bfloat16")
MAX_NEW_TOKENS = int(cfg.get("max_new_tokens", 600))
PROMPT = cfg["prompt"]
FRAME_PATHS = list(cfg.get("frame_paths", []))
OUTPUT_DIR = cfg["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# COMMAND ----------
import torch
import transformers
from PIL import Image

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(DTYPE_STR, torch.bfloat16)
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), "dtype", DTYPE)

# COMMAND ----------
# Load model + processor based on family. Qwen3-VL uses AutoProcessor +
# Qwen3VLForConditionalGeneration; MedGemma is a Gemma-3 multimodal that
# loads via AutoProcessor + AutoModelForImageTextToText.
from transformers import AutoProcessor, AutoModelForImageTextToText

t0 = time.time()
print(f"Loading {MODEL_NAME} from {MODEL_DIR} ...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"  loaded in {time.time() - t0:.1f}s")

# COMMAND ----------
def run_one(image_path: str, prompt: str) -> dict:
    started = time.time()
    try:
        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        in_len = inputs["input_ids"].shape[-1]
        gen = out_ids[0, in_len:]
        text = processor.decode(gen, skip_special_tokens=True)
        return {"frame": image_path, "ok": True, "raw": text, "elapsed_s": round(time.time() - started, 2)}
    except Exception as e:
        return {"frame": image_path, "ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": round(time.time() - started, 2)}

# COMMAND ----------
results = []
for i, fp in enumerate(FRAME_PATHS):
    print(f"[{i+1}/{len(FRAME_PATHS)}] {fp}")
    r = run_one(fp, PROMPT)
    print(f"   → ok={r['ok']} elapsed={r['elapsed_s']}s")
    if r["ok"]:
        # Try to extract a JSON object from the text — many prompts ask for JSON.
        # Falling through is fine; the app will parse client-side too.
        try:
            txt = r["raw"]
            start = txt.find("{")
            end = txt.rfind("}")
            if 0 <= start < end:
                r["parsed"] = json.loads(txt[start:end+1])
        except Exception:
            pass
    results.append(r)

# COMMAND ----------
out_path = f"{OUTPUT_DIR}/results.json"
payload = {
    "model_name": MODEL_NAME,
    "model_dir": MODEL_DIR,
    "n_frames": len(FRAME_PATHS),
    "successful": sum(1 for r in results if r["ok"]),
    "results": results,
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {out_path}")
print(f"  successful: {payload['successful']}/{payload['n_frames']}")

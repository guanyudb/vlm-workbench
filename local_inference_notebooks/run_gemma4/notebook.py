# Databricks notebook source
# MAGIC %md
# MAGIC # Local VLM inference (Gemma 4 family)
# MAGIC
# MAGIC Same input/output contract as ``local_inference_notebooks/run/notebook.py``:
# MAGIC reads a YAML config (``config_yaml`` widget), writes
# MAGIC ``<output_dir>/results.json``. The reason for a separate notebook is the
# MAGIC environment requirements differ:
# MAGIC
# MAGIC - Gemma 4 architecture (``Gemma4ForConditionalGeneration``) needs
# MAGIC   ``transformers >= 5.5.0`` — older versions ``raise ValueError:
# MAGIC   The checkpoint you are trying to load has model type 'gemma4' but
# MAGIC   Transformers does not recognize this architecture.``
# MAGIC - The 12 B variant is ~24 GB in bf16, so the manifest pins the
# MAGIC   accelerator to ``GPU_1xH100`` (won't fit on A10's 24 GB total VRAM
# MAGIC   with overhead).
# MAGIC
# MAGIC See ``~/.claude/projects/-Users-guanyu-chen-Documents-Projects-llm-serving/memory/gemma4-breed-normalization-bench.md``
# MAGIC for the full debug history that pinned down these constraints.

# COMMAND ----------
# MAGIC %pip install -q "transformers>=5.5.0" "accelerate>=1.0.0" "compressed-tensors>=0.7" pyyaml pillow
# MAGIC dbutils.library.restartPython()
# NOTE: torch + torchvision + CUDA come preinstalled in databricks_ai_v4.
# We deliberately use transformers (not vLLM) here:
#   - Playground does one-shot per-frame calls, not benchmark throughput
#   - transformers avoids the mistral_common → opencv → SIGABRT chain we
#     hit while trying to use nightly vLLM on the Standard env.
# `compressed-tensors` is for Google's QAT w4a16 Gemma 4 12B variant
# (gemma-4-12B-it-qat-w4a16-ct) — transformers auto-detects the
# quantization config in the model's config.json and routes the
# weights through compressed-tensors at load time. Pip-installing it
# unconditionally is harmless when the model is bf16 (the package just
# isn't touched).

# COMMAND ----------
import os, json, time, traceback
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
print("transformers", transformers.__version__, "torch", torch.__version__,
      "cuda", torch.cuda.is_available(), "dtype", DTYPE)
# Sanity check — fail fast if the env didn't actually upgrade transformers.
# Gemma 4 needs >= 5.5.0; anything older will throw on `from_pretrained`.
_tv = tuple(int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit())
if _tv and _tv < (5, 5):
    raise SystemExit(
        f"transformers {transformers.__version__} is too old for Gemma 4; "
        f"the %pip install + dbutils.library.restartPython at the top of this "
        f"notebook should have given us >= 5.5.0 — check the run log for pip errors."
    )

# COMMAND ----------
# Load via AutoProcessor + AutoModelForImageTextToText. Gemma 4 is
# Gemma4ForConditionalGeneration, an image-text-to-text model in transformers.
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
# Report GPU memory so we can spot OOM risk before we run inference
if torch.cuda.is_available():
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {(total-free)/1e9:.1f} / {total/1e9:.1f} GB used after load")
    except Exception:
        pass

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
        # Match run/notebook.py: best-effort JSON extraction from the
        # generated text so the app's results matrix can render parsed
        # output without a second parse pass.
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

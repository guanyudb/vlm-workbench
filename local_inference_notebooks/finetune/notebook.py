# Databricks notebook source
# MAGIC %md
# MAGIC # VLM Fine-tune (LoRA on Qwen3-VL / MedGemma)
# MAGIC
# MAGIC Parameterized fine-tune notebook. Reads everything from a YAML config
# MAGIC handed in via the `config_yaml` widget. The YAML carries:
# MAGIC
# MAGIC ```yaml
# MAGIC run_id: ft_<ts>
# MAGIC base_model_name: qwen3-vl-8b           # key in local_models/
# MAGIC base_model_dir: /Volumes/.../local_models/qwen3-vl-8b/snapshot
# MAGIC uc_catalog: hls_amer_catalog
# MAGIC uc_schema: guanyu_chen
# MAGIC uc_model_name: vlmwb_ft_qwen3vl_<ts>   # → <catalog>.<schema>.<name>
# MAGIC mlflow_experiment: /Users/<user>/vlm-workbench
# MAGIC output_dir: /Volumes/.../finetune_runs/<run_id>
# MAGIC training_data:
# MAGIC   - { image_path: "/Volumes/...", instrument: "shaver", anatomy: "...", tissue_condition: "..." }
# MAGIC train_prompt: "Identify the surgical instrument ..."
# MAGIC lora: { r: 16, alpha: 32, dropout: 0.05 }
# MAGIC training:
# MAGIC   num_epochs: 3
# MAGIC   per_device_train_batch_size: 1
# MAGIC   gradient_accumulation_steps: 8
# MAGIC   learning_rate: 2.0e-4
# MAGIC ```
# MAGIC
# MAGIC Writes:
# MAGIC - LoRA adapter + merged weights to ``<output_dir>/final/``
# MAGIC - MLflow run with training metrics + registered model
# MAGIC - ``<output_dir>/manifest.yaml`` — Playground-compatible manifest for the new model
# MAGIC - ``<output_dir>/result.json`` — summary with run_id, UC name, accuracy on holdout, etc.

# COMMAND ----------
# MAGIC %pip install -q --upgrade "transformers>=4.57" "accelerate>=1.0" "peft>=0.13" \
# MAGIC   "trl>=0.11" "datasets" "qwen-vl-utils" "pyyaml" mlflow pillow \
# MAGIC   safetensors tokenizers
# MAGIC # NOTE: torch + torchvision come preinstalled in databricks_ai_v4 with
# MAGIC # matching CUDA versions. We intentionally don't reinstall them; doing
# MAGIC # so risks pulling a torchvision wheel for a different CUDA than the
# MAGIC # torch the base image ships, which crashes the model loader at serve
# MAGIC # time ("operator torchvision::nms does not exist").
# MAGIC %restart_python

# COMMAND ----------
import os, json, time, sys, traceback
from pathlib import Path
import yaml

dbutils.widgets.text("config_yaml", "", "Path to YAML config")
yaml_path = dbutils.widgets.get("config_yaml").strip()
if not yaml_path:
    raise SystemExit("config_yaml widget is required")
with open(yaml_path) as f:
    cfg = yaml.safe_load(f) or {}

RUN_ID = cfg["run_id"]
BASE_MODEL_NAME = cfg["base_model_name"]
BASE_MODEL_DIR = cfg["base_model_dir"]
UC_CATALOG = cfg["uc_catalog"]
UC_SCHEMA = cfg["uc_schema"]
UC_MODEL_NAME = cfg["uc_model_name"]
UC_FULL = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
OUTPUT_DIR = cfg["output_dir"]
TRAIN_DATA = list(cfg.get("training_data", []))
TRAIN_PROMPT = cfg["train_prompt"]
EXPERIMENT_PATH = cfg.get("mlflow_experiment") or "/Users/guanyu.chen@databricks.com/vlmwb-experiments"

LORA = cfg.get("lora", {})
LORA_R = int(LORA.get("r", 16))
LORA_ALPHA = int(LORA.get("alpha", 32))
LORA_DROPOUT = float(LORA.get("dropout", 0.05))

TRC = cfg.get("training", {})
NUM_EPOCHS = float(TRC.get("num_epochs", 3))
BATCH_SIZE = int(TRC.get("per_device_train_batch_size", 1))
GRAD_ACCUM = int(TRC.get("gradient_accumulation_steps", 8))
LR = float(TRC.get("learning_rate", 2e-4))
WARMUP_RATIO = float(TRC.get("warmup_ratio", 0.03))
WEIGHT_DECAY = float(TRC.get("weight_decay", 0.0))
MAX_LENGTH = int(TRC.get("max_length", 1024))

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[ft] run_id={RUN_ID} base={BASE_MODEL_NAME}")
print(f"[ft] training rows: {len(TRAIN_DATA)}  epochs={NUM_EPOCHS}  bs={BATCH_SIZE}*{GRAD_ACCUM}  lr={LR}")
print(f"[ft] UC target: {UC_FULL}")
print(f"[ft] output_dir: {OUTPUT_DIR}")

# COMMAND ----------
# HF token: gives us access to gated repos like Qwen3-VL/MedGemma when
# transformers wants to validate the tokenizer config against HF.
try:
    # Defaults match the workbench's per-deploy scope name. The app threads
    # the actual configured values in via cfg — the defaults only matter for
    # standalone notebook runs.
    hf_scope = cfg.get("hf_secret_scope", "vlmwb_hf")
    hf_key = cfg.get("hf_secret_key", "HF_TOKEN")
    os.environ["HF_TOKEN"] = dbutils.secrets.get(hf_scope, hf_key)
except Exception as e:
    print(f"[ft][warn] HF_TOKEN secret not available: {e}")

# COMMAND ----------
import torch, transformers, mlflow
print("transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
if not torch.cuda.is_available() or n_gpu == 0:
    raise RuntimeError("No GPU attached. Submit this notebook with a serverless GPU accelerator.")
print(f"[ft] GPUs: {n_gpu}")
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.0f}GB")

# COMMAND ----------
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

print(f"[ft] loading base model from {BASE_MODEL_DIR}")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)
print(f"[ft] base model loaded in {time.time() - t0:.0f}s")
model.enable_input_require_grads()

# COMMAND ----------
# LoRA. Target modules vary by model family; the default below works for both
# Qwen3-VL and Gemma3.
lora_target_modules = LORA.get("target_modules") or [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------
# Build dataset: (image, prompt, json-answer). We construct the assistant
# response as JSON matching the Playground's expected shape so the fine-tuned
# model emits the same structure the rest of the app understands.
from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

def _format_answer(row: dict) -> str:
    obj = {"instrument": row.get("instrument", "")}
    if row.get("anatomy"):
        obj["anatomy"] = row["anatomy"]
    if row.get("tissue_condition"):
        obj["tissue_condition"] = row["tissue_condition"]
    return json.dumps(obj)

class VLMFTDataset(Dataset):
    """Dataset that emits whatever the processor returns. Qwen3-VL added
    M-RoPE support, which means the model now requires ``mm_token_type_ids``
    alongside ``image_grid_thw``. Hard-coding a subset of processor outputs
    breaks every time the model family adds a new field, so we forward all
    non-input fields untouched and rely on the collator to stack them."""

    def __init__(self, records, processor, prompt, max_length=1024):
        self.records = records
        self.processor = processor
        self.prompt = prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        answer = _format_answer(rec)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_inputs, videos=vid_inputs,
            padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        # Mask the user prefix so we only train on the assistant span.
        prefix_messages = [messages[0]]
        prefix_text = self.processor.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        prefix_inputs = self.processor(
            text=[prefix_text], images=img_inputs, videos=vid_inputs,
            padding=False, return_tensors="pt",
        )
        prefix_len = prefix_inputs["input_ids"].shape[1]
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[attention_mask == 0] = -100

        # Forward EVERY field the processor returned (input_ids, attention_mask,
        # pixel_values, image_grid_thw, mm_token_type_ids, ...). Squeezing the
        # batch dim leaves us with a per-example tensor for the collator to
        # re-stack. Anything that isn't a tensor we skip.
        out = {"labels": labels}
        for k, v in inputs.items():
            if k in ("input_ids", "attention_mask"):
                out[k] = v.squeeze(0)
            elif hasattr(v, "squeeze"):
                # Per-image tensors typically have shape (1, ...); strip the batch dim.
                try:
                    out[k] = v.squeeze(0) if v.shape[0] == 1 else v
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out


def vlm_collate(batch):
    """Stack every tensor field. Per-example tensors that differ in shape
    (e.g. variable-length pixel_values for different image sizes) are left
    as-is; the model handles them. Non-tensor fields are passed through
    if every example has the same Python value."""
    import torch
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[k] = torch.stack(vals)
            except Exception:
                # Shapes differ — concatenate along dim 0 instead (matches the
                # per-token pixel_values layout for Qwen3-VL with variable
                # image patch counts).
                out[k] = torch.cat([v.unsqueeze(0) if v.dim() == 0 else v for v in vals], dim=0)
        else:
            out[k] = vals[0]
    return out

# Hold out the last 10% (or at least 1 row) for in-loop eval. The Playground
# accuracy chip is the real validation; here we just want training stability.
if len(TRAIN_DATA) < 4:
    raise SystemExit(f"Need at least 4 training examples; got {len(TRAIN_DATA)}")
holdout_n = max(1, len(TRAIN_DATA) // 10)
train_rows = TRAIN_DATA[:-holdout_n]
val_rows = TRAIN_DATA[-holdout_n:]
print(f"[ft] split: train={len(train_rows)}  val={len(val_rows)}")
train_ds = VLMFTDataset(train_rows, processor, TRAIN_PROMPT, MAX_LENGTH)
val_ds = VLMFTDataset(val_rows, processor, TRAIN_PROMPT, MAX_LENGTH)

# COMMAND ----------
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    logging_steps=1,
    eval_strategy="steps" if len(val_rows) > 0 else "no",
    eval_steps=max(5, max(1, len(train_rows) // 4)),
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    optim="adamw_torch",
    gradient_checkpointing=True,
    report_to="mlflow",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# COMMAND ----------
# MLflow tracking. The Workspace tracking URI lets us run, log, and register
# in Unity Catalog without extra config.
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
except Exception as e:
    print(f"[ft][warn] set_experiment failed ({EXPERIMENT_PATH}): {e}; falling back to default")

with mlflow.start_run(run_name=RUN_ID) as run:
    training_run_id = run.info.run_id
    mlflow.log_params({
        "base_model": BASE_MODEL_NAME,
        "base_model_dir": BASE_MODEL_DIR,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_rows) > 0 else None,
        data_collator=vlm_collate,
    )
    print("[ft] training start")
    t0 = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - t0
    print(f"[ft] training done in {train_elapsed:.0f}s  loss={train_result.training_loss:.4f}")
    mlflow.log_metric("final_train_loss", train_result.training_loss)
    mlflow.log_metric("train_elapsed_s", train_elapsed)

    # COMMAND ----------
    # Save LoRA adapter + merge to a deployable full model. We register the
    # merged checkpoint so Playground can load it like any other base model.
    adapter_dir = f"{OUTPUT_DIR}/lora_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    print("[ft] merging LoRA into base weights ...")
    merged_model = model.merge_and_unload()
    merged_dir = f"{OUTPUT_DIR}/merged"
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    print(f"[ft] merged model saved to {merged_dir}")

    # COMMAND ----------
    # Register in Unity Catalog. We use the merged checkpoint so Playground's
    # local-inference notebook can `from_pretrained(<snapshot_dir>)` it
    # without needing PEFT at inference time.
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "prompt"), ColSpec("binary", "image")]),
        outputs=Schema([ColSpec("string", "generated_text")]),
    )
    print(f"[ft] logging + registering {UC_FULL}")
    # Pin everything the runtime needs to import the model. We hit
    # "operator torchvision::nms does not exist" at serve time when
    # torchvision wasn't pinned — the conda solver picked a CUDA-11.8 wheel
    # against a CUDA-12.6 torch, and the import failed. Pinning torchvision
    # to the exact version captured here forces a matching pair. Also pin
    # safetensors + tokenizers since transformers imports them at module-load.
    import torchvision as _tv
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
    # Strip the +cuXXX suffix on torch/torchvision to make the version a valid
    # pip identifier (e.g. 2.7.1+cu118 → 2.7.1).
    torch_ver = torch.__version__.split('+')[0]
    tv_ver = _tv.__version__.split('+')[0]
    pip_reqs = [
        f"transformers=={transformers.__version__}",
        f"torch=={torch_ver}",
        f"torchvision=={tv_ver}",
        "accelerate",
        "peft",
        "qwen-vl-utils",
        "Pillow",
    ]
    if sft_ver:
        pip_reqs.append(f"safetensors=={sft_ver}")
    if tk_ver:
        pip_reqs.append(f"tokenizers=={tk_ver}")
    print(f"[ft] pinning pip_requirements: {pip_reqs}")
    # IMPORTANT: pass torch_dtype=torch.bfloat16 so the reloaded model uses
    # bfloat16 at serve time. Without this, MLflow stores torch_dtype: null
    # in MLmodel and the HF pipeline reloads in fp32, OOM'ing a 24 GB A10.
    # The working pattern is also to pass a PATH (merged_dir) — not a dict —
    # so MLflow's Pipeline-creation validation doesn't fight the multimodal
    # Processor. We're already passing the path; we only needed to add dtype.
    mlflow.transformers.log_model(
        transformers_model=merged_dir,
        name="finetuned_model",
        registered_model_name=UC_FULL,
        task="image-text-to-text",
        torch_dtype=torch.bfloat16,
        signature=signature,
        metadata={
            "source_model": BASE_MODEL_DIR,
            "n_train_examples": len(train_rows),
            "train_loss": float(train_result.training_loss),
        },
        pip_requirements=pip_reqs,
    )
    mlflow_url = f"{mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{training_run_id}"
    print(f"[ft] mlflow run url: {mlflow_url}")

# COMMAND ----------
# Drop a Playground-compatible manifest pointing at the merged-model dir,
# so the new fine-tuned model appears automatically in Playground's Local
# section the next time list_local_models polls.
LOCAL_MODELS_DIR = cfg.get("local_models_dir") or os.environ.get(
    "LOCAL_MODELS_DIR",
    "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/local_models",
)
manifest_name = UC_MODEL_NAME  # short, sluggable identifier
manifest_dir = f"{LOCAL_MODELS_DIR}/{manifest_name}"
os.makedirs(manifest_dir, exist_ok=True)
manifest = {
    "name": manifest_name,
    "display_name": f"{manifest_name} (ft)",
    "hf_repo": "",
    "accelerator": "GPU_1xA10",
    "base_environment": "databricks_ai_v4",
    "torch_dtype": "bfloat16",
    "max_new_tokens": 400,
    "snapshot_dir": merged_dir,
    "notes": f"Fine-tuned from {BASE_MODEL_NAME} on {len(train_rows)} labeled frames (run {RUN_ID}). UC: {UC_FULL}",
    "uc_model": UC_FULL,
    "base_model": BASE_MODEL_NAME,
    "run_id": RUN_ID,
    "mlflow_run_id": training_run_id,
}
with open(f"{manifest_dir}/manifest.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
print(f"[ft] wrote Playground manifest: {manifest_dir}/manifest.yaml")

# COMMAND ----------
# Summary blob the backend reads via /api/training/finetune/<run_id>.
result = {
    "run_id": RUN_ID,
    "uc_model": UC_FULL,
    "mlflow_run_id": training_run_id,
    "mlflow_url": mlflow_url,
    "manifest_name": manifest_name,
    "merged_dir": merged_dir,
    "adapter_dir": adapter_dir,
    "n_train": len(train_rows),
    "n_val": len(val_rows),
    "train_loss": float(train_result.training_loss),
    "train_elapsed_s": float(train_elapsed),
}
result_path = f"{OUTPUT_DIR}/result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[ft] wrote result: {result_path}")

dbutils.notebook.exit(json.dumps(result))

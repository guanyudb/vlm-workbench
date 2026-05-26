# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy Prompt Optimizer — generic, parameterized
# MAGIC
# MAGIC Real DSPy framework (>=2.5). You fill in **just the CONFIG block at the top**:
# MAGIC task definition, dataset loader, task model, optimizer choice. The notebook
# MAGIC handles the rest: builds the `dspy.Signature` dynamically, wraps your task
# MAGIC model as a custom `dspy.LM` (image-aware), runs the chosen optimizer
# MAGIC (`BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `MIPROv2`, or
# MAGIC `COPRO`), evaluates baseline vs optimized, and saves both the dspy program
# MAGIC and a human-readable prompt+demos export.
# MAGIC
# MAGIC ## Supported configurations
# MAGIC
# MAGIC | Modality | Task model providers | Prompt-model (MIPROv2 only) |
# MAGIC |---|---|---|
# MAGIC | image    | `huggingface_local` (Qwen-VL etc.), `databricks_fmapi` (Claude/GPT/Gemini vision) | `databricks_fmapi` |
# MAGIC | text     | `huggingface_local`, `databricks_fmapi`                                          | `databricks_fmapi` |
# MAGIC
# MAGIC ## Optimizers (DSPy teleprompters)
# MAGIC
# MAGIC | type            | Best for                                                |
# MAGIC |-----------------|---------------------------------------------------------|
# MAGIC | BootstrapFewShot | Quick win — bootstrap demos from successful traces      |
# MAGIC | BootstrapFewShotWithRandomSearch | Tries multiple demo combos                |
# MAGIC | MIPROv2         | Joint instruction + demo Bayesian search (best, slowest) |
# MAGIC | COPRO           | Instruction-only optimization (no demos)                |
# MAGIC
# MAGIC ## How to use
# MAGIC
# MAGIC 1. Edit the **`# === CONFIG ===`** cell.
# MAGIC 2. Run all cells.
# MAGIC 3. Find optimized prompt at `OUTPUT_DIR/<task_name>/optimized_program.json`
# MAGIC    and a human-readable export at `OUTPUT_DIR/<task_name>/optimized_prompt.md`.

# COMMAND ----------

# MAGIC %pip install "dspy-ai>=2.5" "transformers>=4.57" accelerate "torchvision>=0.20" pillow timm einops sentencepiece openai databricks-sdk pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## === CONFIG ===  (edit these for your task)

# COMMAND ----------

import os, json

# -- Where to save outputs --
OUTPUT_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/dspy_runs"

# -- Task definition --
TASK = {
    "name": "arthroscopy_instrument_id",
    "description": (
        "Identify the primary surgical instrument visible in a single frame "
        "from a knee arthroscopy video."
    ),
    "input_modality": "image",                    # "image" or "text"

    # The signature's input field — both name (used in code) and description
    # (shown to the LM by DSPy). Pick a name that matches the keys in EVAL_DATA.
    "input_field_name": "frame",
    "input_field_desc": "single frame from a knee arthroscopy procedure",

    # The signature's output field
    "output_field_name": "instrument",
    "output_field_desc": "one of the vocabulary classes",

    # Used for parsing the model's output back into a label string.
    # Options: "class_in_text" (vocabulary required), "exact_match", "json_field"
    "metric": "class_in_text",
    "vocabulary": [
        "probe", "shaver", "burr", "grasper", "biter", "suture_passer",
        "anchor_driver", "electrocautery", "cannula", "scissors", "drill_guide",
        "trocar", "knot_pusher", "rasp", "other_metal_tool", "no_instrument_visible",
    ],
    # If metric == "json_field":
    "json_field_path": None,
}

# -- Dataset loader --
# Return a list of dicts. Each dict needs:
#   - TASK["input_field_name"]   (image path for image modality, str for text)
#   - TASK["output_field_name"]  (the gold label — string)
#   - "id" (optional, for tracking)
def load_dataset():
    EVAL_FRAMES_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/eval_frames"
    TRACK1 = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/experiments_results/01_aigateway.json"
    sonnet_labels = {}
    with open(TRACK1) as f:
        t1 = json.load(f)
    for r in t1["results"]:
        if r["model"] != "databricks-claude-sonnet-4-5" or not r.get("ok"):
            continue
        parsed = r.get("parsed") or {}
        inst = parsed.get("instruments", [])
        if isinstance(inst, list) and inst and isinstance(inst[0], dict):
            cls = inst[0].get("class")
            if cls and cls in TASK["vocabulary"]:
                sonnet_labels[r["frame"]] = cls
    out = []
    for fname in sorted(os.listdir(EVAL_FRAMES_DIR)):
        if not fname.lower().endswith(".jpg"):
            continue
        if fname not in sonnet_labels:
            continue
        out.append({
            TASK["input_field_name"]: os.path.join(EVAL_FRAMES_DIR, fname),
            TASK["output_field_name"]: sonnet_labels[fname],
            "id": fname,
        })
    return out

# -- Train / dev split --
TRAIN_FRAC = 0.6
SEED = 42

# -- DSPy module type --
USE_CHAIN_OF_THOUGHT = False   # if True, dspy.ChainOfThought instead of dspy.Predict

# -- Task model: the model whose program/prompt is being optimized --
TASK_MODEL = {
    "provider": "huggingface_local",          # "huggingface_local" or "databricks_fmapi"
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",  # for huggingface_local
    # "endpoint": "databricks-meta-llama-4-maverick",  # for databricks_fmapi
    "max_new_tokens": 400,
    "temperature": 0.0,
    "image_max_side": 896,
}

# -- Prompt model (only used by MIPROv2 / COPRO to PROPOSE candidate instructions) --
# Always FMAPI text model; recommended a strong reasoner.
PROMPT_MODEL = {
    "provider": "databricks_fmapi",
    "endpoint": "databricks-claude-sonnet-4-6",
    "max_tokens": 2000,
    "temperature": 0.7,
}

# -- Optimizer choice --
OPTIMIZER = {
    "type": "BootstrapFewShot",   # BootstrapFewShot | BootstrapFewShotWithRandomSearch | MIPROv2 | COPRO
    # BootstrapFewShot / BootstrapFewShotWithRandomSearch
    "max_bootstrapped_demos": 4,
    "max_labeled_demos": 4,
    "max_rounds": 1,
    # BootstrapFewShotWithRandomSearch
    "num_candidate_programs": 6,
    # MIPROv2
    "num_candidates": 10,
    "init_temperature": 1.4,
    # COPRO
    "breadth": 5, "depth": 3,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## YAML override (for non-interactive runs)

# COMMAND ----------

dbutils.widgets.text("config_yaml", "", "Path to YAML config (optional)")

_yaml_path = dbutils.widgets.get("config_yaml").strip()
_inline_eval_data = None
if _yaml_path:
    import yaml
    print(f"Loading config override from {_yaml_path}")
    with open(_yaml_path) as _f:
        _cfg = yaml.safe_load(_f) or {}
    if "output_dir" in _cfg: OUTPUT_DIR = _cfg["output_dir"]
    if "task" in _cfg: TASK = {**TASK, **_cfg["task"]}
    if "task_model" in _cfg: TASK_MODEL = {**TASK_MODEL, **_cfg["task_model"]}
    if "prompt_model" in _cfg: PROMPT_MODEL = {**PROMPT_MODEL, **_cfg["prompt_model"]}
    if "optimizer" in _cfg: OPTIMIZER = {**OPTIMIZER, **_cfg["optimizer"]}
    if "train_frac" in _cfg: TRAIN_FRAC = float(_cfg["train_frac"])
    if "use_chain_of_thought" in _cfg: USE_CHAIN_OF_THOUGHT = bool(_cfg["use_chain_of_thought"])
    if "eval_data" in _cfg: _inline_eval_data = list(_cfg["eval_data"])
    print(f"  task: {TASK.get('name')}  optimizer: {OPTIMIZER.get('type')}")
    print(f"  task_model: {TASK_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup (no edits below this line)

# COMMAND ----------

import io, base64, time, random, traceback, copy, re
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import dspy

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("dspy:", dspy.__version__)

os.makedirs(OUTPUT_DIR, exist_ok=True)
TASK_OUT = os.path.join(OUTPUT_DIR, TASK["name"])
os.makedirs(TASK_OUT, exist_ok=True)
print("output:", TASK_OUT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom DSPy LM that wraps the task model
# MAGIC
# MAGIC Handles OpenAI-style chat messages with mixed text + `image_url` content
# MAGIC blocks for both `huggingface_local` and `databricks_fmapi` providers.

# COMMAND ----------

class TaskLM(dspy.LM):
    """One LM that supports image OR text inputs, HF-local OR Databricks FMAPI."""

    def __init__(self, cfg, modality):
        super().__init__(model=f"local-{cfg.get('model_id', cfg.get('endpoint', 'unknown'))}",
                         model_type="chat")
        self.cfg = cfg
        self.modality = modality
        self.kwargs = {"temperature": cfg["temperature"], "max_tokens": cfg.get("max_new_tokens", 400)}
        self.history = []
        if cfg["provider"] == "huggingface_local":
            self._load_hf()
        elif cfg["provider"] == "databricks_fmapi":
            self._load_fmapi()
        else:
            raise ValueError(cfg["provider"])

    def _load_hf(self):
        from transformers import (AutoProcessor, AutoModelForImageTextToText,
                                  AutoModelForCausalLM, AutoTokenizer)
        _hf_scope = (_cfg.get("hf_secret_scope") if "_cfg" in dir() else None) or "hls_g4"
        _hf_key = (_cfg.get("hf_secret_key") if "_cfg" in dir() else None) or "HF_TOKEN"
        os.environ["HF_TOKEN"] = dbutils.secrets.get(_hf_scope, _hf_key)
        print(f"Loading {self.cfg['model_id']}...")
        if self.modality == "image":
            self.processor = AutoProcessor.from_pretrained(self.cfg["model_id"], token=os.environ["HF_TOKEN"])
            self.hf_model = AutoModelForImageTextToText.from_pretrained(
                self.cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
                token=os.environ["HF_TOKEN"],
            ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_id"], token=os.environ["HF_TOKEN"])
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
                token=os.environ["HF_TOKEN"],
            ).eval()
        print("loaded")

    def _load_fmapi(self):
        from openai import OpenAI
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        token = w.tokens.create(comment="dspy-task", lifetime_seconds=7200).token_value
        self.client = OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")
        self.endpoint = self.cfg["endpoint"]

    # ---- helpers ----
    def _decode_image_url(self, url: str) -> Image.Image:
        if url.startswith("data:image"):
            b64 = url.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if url.startswith("file://") or url.startswith("/"):
            return Image.open(url.replace("file://", "")).convert("RGB")
        raise ValueError(f"Unsupported image url: {url[:30]}")

    def _shrink(self, pil):
        max_side = self.cfg.get("image_max_side", 896)
        pil2 = pil.copy(); pil2.thumbnail((max_side, max_side), Image.LANCZOS)
        return pil2

    def _flatten(self, messages):
        """Pull out plain text + list of images from OpenAI-style messages."""
        text_parts, images = [], []
        for m in messages:
            content = m.get("content", "")
            role = m.get("role", "user")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        text_parts.append(c["text"])
                    elif c.get("type") == "image_url":
                        images.append(self._decode_image_url(c["image_url"]["url"]))
            else:
                text_parts.append(str(content))
        return "\n".join(text_parts), images

    # ---- main entry point DSPy calls ----
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        text, images = self._flatten(messages)

        if self.cfg["provider"] == "huggingface_local":
            if self.modality == "image":
                images = [self._shrink(img) for img in images]
                qcontent = []
                for img in images:
                    qcontent.append({"type": "image", "image": img})
                qcontent.append({"type": "text", "text": text})
                qmsgs = [{"role": "user", "content": qcontent}]
                inputs = self.processor.apply_chat_template(
                    qmsgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(self.hf_model.device)
                with torch.no_grad():
                    out = self.hf_model.generate(
                        **inputs, max_new_tokens=kwargs.get("max_tokens", self.cfg["max_new_tokens"]),
                        do_sample=(self.cfg["temperature"] > 0),
                        temperature=max(self.cfg["temperature"], 1e-5),
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                return [self.processor.decode(gen, skip_special_tokens=True)]
            else:
                ids = self.tokenizer(text, return_tensors="pt").to(self.hf_model.device)
                with torch.no_grad():
                    out = self.hf_model.generate(
                        **ids, max_new_tokens=kwargs.get("max_tokens", self.cfg["max_new_tokens"]),
                        do_sample=(self.cfg["temperature"] > 0),
                        temperature=max(self.cfg["temperature"], 1e-5),
                    )
                return [self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)]

        # databricks_fmapi
        content = [{"type": "text", "text": text}]
        for img in images:
            buf = io.BytesIO()
            self._shrink(img).save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = self.client.chat.completions.create(
            model=self.endpoint,
            messages=[{"role": "user", "content": content}],
            max_tokens=kwargs.get("max_tokens", self.cfg["max_new_tokens"]),
            temperature=self.cfg["temperature"],
        )
        return [resp.choices[0].message.content or ""]

task_lm = TaskLM(TASK_MODEL, TASK["input_modality"])
dspy.configure(lm=task_lm)
print("DSPy LM configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: prompt model (for MIPROv2 / COPRO)

# COMMAND ----------

prompt_lm = None
if OPTIMIZER["type"] in ("MIPROv2", "COPRO"):
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    pm_token = w.tokens.create(comment="dspy-prompt-model", lifetime_seconds=7200).token_value
    pm_endpoint = PROMPT_MODEL["endpoint"]
    pm_base = f"{w.config.host}/serving-endpoints"
    # DSPy can talk to OpenAI-compatible endpoints directly
    prompt_lm = dspy.LM(
        f"openai/{pm_endpoint}",
        api_base=pm_base,
        api_key=pm_token,
        max_tokens=PROMPT_MODEL["max_tokens"],
        temperature=PROMPT_MODEL["temperature"],
        model_type="chat",
    )
    print(f"Prompt model: {pm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build dspy.Signature dynamically

# COMMAND ----------

def build_signature():
    """Dynamically construct a dspy.Signature class from TASK config."""
    in_name = TASK["input_field_name"]
    out_name = TASK["output_field_name"]

    if TASK["input_modality"] == "image":
        input_field = dspy.InputField(desc=TASK["input_field_desc"], format=dspy.Image)
    else:
        input_field = dspy.InputField(desc=TASK["input_field_desc"])

    out_desc = TASK["output_field_desc"]
    vocab = TASK.get("vocabulary")
    if vocab:
        out_desc += f". One of: {', '.join(vocab)}"

    output_field = dspy.OutputField(desc=out_desc)

    attrs = {
        "__doc__": TASK["description"],
        in_name: input_field,
        out_name: output_field,
    }
    if TASK["input_modality"] == "image":
        attrs["__annotations__"] = {in_name: dspy.Image, out_name: str}
    else:
        attrs["__annotations__"] = {in_name: str, out_name: str}
    Sig = type("DynamicSignature", (dspy.Signature,), attrs)
    return Sig

Signature = build_signature()
print("Signature:", Signature.__doc__)
print("  input :", TASK["input_field_name"])
print("  output:", TASK["output_field_name"])

base_program = dspy.ChainOfThought(Signature) if USE_CHAIN_OF_THOUGHT else dspy.Predict(Signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build dspy.Examples + train/dev split

# COMMAND ----------

# If a YAML provided eval_data inline, use it; otherwise fall back to load_dataset().
if _inline_eval_data is not None:
    in_field = TASK["input_field_name"]; out_field = TASK["output_field_name"]
    raw = []
    for ex in _inline_eval_data:
        raw.append({
            in_field: ex.get("input"),
            out_field: ex.get("expected"),
            "id": ex.get("id"),
        })
    print(f"Using {len(raw)} examples from YAML config")
else:
    raw = load_dataset()
    print(f"Loaded {len(raw)} examples from in-notebook loader")

def to_example(row):
    in_name = TASK["input_field_name"]
    out_name = TASK["output_field_name"]
    if TASK["input_modality"] == "image":
        img = Image.open(row[in_name]).convert("RGB")
        max_side = TASK_MODEL.get("image_max_side", 896)
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        return dspy.Example(**{
            in_name: dspy.Image.from_PIL(img),
            out_name: row[out_name],
            "_id": row.get("id", "?"),
        }).with_inputs(in_name)
    else:
        return dspy.Example(**{
            in_name: row[in_name],
            out_name: row[out_name],
            "_id": row.get("id", "?"),
        }).with_inputs(in_name)

examples = [to_example(r) for r in raw]
random.shuffle(examples)
n_train = int(TRAIN_FRAC * len(examples))
trainset = examples[:n_train]
devset = examples[n_train:]
print(f"trainset={len(trainset)}  devset={len(devset)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metric

# COMMAND ----------

def _strip_md(t):
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip().rstrip("`").strip()
    return t

def _walk(d, path):
    cur = d
    for tok in re.findall(r"[^.\[\]]+", path):
        if tok.isdigit():
            try: cur = cur[int(tok)]
            except: return None
        else:
            if isinstance(cur, dict): cur = cur.get(tok)
            else: return None
        if cur is None: return None
    return cur

def _extract_class_from_parsed(d):
    """Mirrors the Playground UI's primaryClass: handles both
    {"instrument": "shaver"} and {"instruments": [{"class": "..."}]}, and
    treats {"instruments": []} or empty instrument string as
    'no_instrument_visible'. Without this an empty-frame response gets
    misclassified, tanking the optimizer score."""
    if not isinstance(d, dict):
        return None
    if "instrument" in d:
        v = d["instrument"]
        if isinstance(v, str):
            return v.strip().lower() or "no_instrument_visible"
    if "instruments" in d:
        lst = d["instruments"]
        if isinstance(lst, list):
            if len(lst) == 0:
                return "no_instrument_visible"
            if isinstance(lst[0], dict):
                c = lst[0].get("class")
                if isinstance(c, str):
                    return c.strip().lower() or "no_instrument_visible"
    return "no_instrument_visible"


def normalize(text):
    metric = TASK["metric"]
    if metric == "exact_match":
        return (text or "").strip().lower()
    t = _strip_md(text or "")
    parsed_dict = None
    if "{" in t and "}" in t:
        try:
            start, end = t.find("{"), t.rfind("}")
            if 0 <= start < end:
                parsed_dict = json.loads(t[start:end + 1])
        except Exception:
            parsed_dict = None
    if parsed_dict is not None:
        for path in [TASK.get("json_field_path"), *TASK.get("json_field_path_alts", [])]:
            if not path:
                continue
            v = _walk(parsed_dict, path)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        c = _extract_class_from_parsed(parsed_dict)
        if c:
            return c
    if metric == "json_field":
        return "?"
    text_l = (text or "").lower()
    for v in TASK["vocabulary"]:
        if v in text_l:
            return v
    return "?"

def metric_fn(example, pred, trace=None):
    out_name = TASK["output_field_name"]
    target = (getattr(example, out_name, "") or "").strip().lower()
    raw = getattr(pred, out_name, "")
    return normalize(raw) == target

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline + optimization

# COMMAND ----------

def evaluate(program, dataset):
    matched = 0; rows = []
    for ex in dataset:
        in_val = getattr(ex, TASK["input_field_name"])
        try:
            pred = program(**{TASK["input_field_name"]: in_val})
            raw = getattr(pred, TASK["output_field_name"], "")
            predicted = normalize(raw)
        except Exception as e:
            raw = f"[error: {str(e)[:120]}]"
            predicted = "?"
        target = (getattr(ex, TASK["output_field_name"], "") or "").lower()
        ok = (predicted == target)
        if ok: matched += 1
        rows.append({
            "id": getattr(ex, "_id", "?"),
            "expected": target,
            "predicted": predicted,
            "match": ok,
            "raw": (raw or "")[:300],
        })
    score = matched / len(dataset) if dataset else 0
    return score, rows

print("\n=== Baseline (unoptimized) on devset ===")
t0 = time.time()
base_score, base_rows = evaluate(base_program, devset)
print(f"  baseline: {base_score:.1%}  elapsed={time.time()-t0:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the chosen optimizer

# COMMAND ----------

from dspy.teleprompt import (BootstrapFewShot, BootstrapFewShotWithRandomSearch,
                              MIPROv2, COPRO)

opt_t0 = time.time()
opt_type = OPTIMIZER["type"]
print(f"\n=== Optimizer: {opt_type} ===")

try:
    if opt_type == "BootstrapFewShot":
        opt = BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=OPTIMIZER["max_bootstrapped_demos"],
            max_labeled_demos=OPTIMIZER["max_labeled_demos"],
            max_rounds=OPTIMIZER["max_rounds"],
        )
        optimized = opt.compile(base_program, trainset=trainset)
    elif opt_type == "BootstrapFewShotWithRandomSearch":
        opt = BootstrapFewShotWithRandomSearch(
            metric=metric_fn,
            max_bootstrapped_demos=OPTIMIZER["max_bootstrapped_demos"],
            max_labeled_demos=OPTIMIZER["max_labeled_demos"],
            num_candidate_programs=OPTIMIZER["num_candidate_programs"],
        )
        optimized = opt.compile(base_program, trainset=trainset, valset=devset)
    elif opt_type == "MIPROv2":
        opt = MIPROv2(
            metric=metric_fn,
            prompt_model=prompt_lm,
            task_model=None,            # use the configured dspy.lm
            num_candidates=OPTIMIZER["num_candidates"],
            init_temperature=OPTIMIZER["init_temperature"],
        )
        optimized = opt.compile(
            base_program.deepcopy(),
            trainset=trainset, valset=devset,
            max_bootstrapped_demos=OPTIMIZER["max_bootstrapped_demos"],
            max_labeled_demos=OPTIMIZER["max_labeled_demos"],
            requires_permission_to_run=False,
        )
    elif opt_type == "COPRO":
        opt = COPRO(metric=metric_fn, prompt_model=prompt_lm,
                    breadth=OPTIMIZER["breadth"], depth=OPTIMIZER["depth"])
        optimized = opt.compile(base_program, trainset=trainset, eval_kwargs={"display_progress": False})
    else:
        raise ValueError(opt_type)
    print(f"  optimization done in {time.time()-opt_t0:.0f}s")
    optimized_ok = True
except Exception as e:
    print(f"  optimization FAILED: {e}")
    traceback.print_exc()
    optimized = base_program
    optimized_ok = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate optimized + save

# COMMAND ----------

opt_score, opt_rows = evaluate(optimized, devset)
print(f"\nOptimized devset score: {opt_score:.1%}  (baseline: {base_score:.1%})")

# Run on the full dataset for downstream comparison with other tracks
full_score, full_rows = evaluate(optimized, examples)
print(f"Optimized full-set score: {full_score:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export results

# COMMAND ----------

prog_path = os.path.join(TASK_OUT, "optimized_program.json")
try:
    optimized.save(prog_path)
    print(f"Saved program -> {prog_path}")
except Exception as e:
    print(f"  could not save program: {e}")

def extract_human_readable(program):
    """Pull out the final instruction text + few-shot demos from a compiled program."""
    out = []
    for name, sub in program.named_predictors():
        sig = sub.signature
        out.append(f"## Predictor: {name}")
        out.append(f"Instructions:\n{sig.instructions}\n")
        demos = getattr(sub, "demos", []) or []
        out.append(f"Demos: {len(demos)}")
        for i, d in enumerate(demos):
            out.append(f"\n### demo[{i}]")
            for k, v in d.items():
                if k.startswith("_"): continue
                preview = str(v)[:300] if not isinstance(v, dspy.Image) else "<image>"
                out.append(f"- **{k}**: {preview}")
    return "\n".join(out)

md_path = os.path.join(TASK_OUT, "optimized_prompt.md")
with open(md_path, "w") as f:
    f.write(f"# DSPy Optimized Program — {TASK['name']}\n\n")
    f.write(f"- Optimizer: {opt_type}\n")
    f.write(f"- Optimization succeeded: {optimized_ok}\n")
    f.write(f"- Baseline devset: {base_score:.1%}\n")
    f.write(f"- Optimized devset: {opt_score:.1%}\n")
    f.write(f"- Optimized full set: {full_score:.1%}\n\n")
    f.write(extract_human_readable(optimized))
print(f"Saved markdown -> {md_path}")

run_path = os.path.join(TASK_OUT, "run.json")
with open(run_path, "w") as f:
    json.dump({
        "task": TASK,
        "task_model": TASK_MODEL,
        "prompt_model": PROMPT_MODEL if prompt_lm else None,
        "optimizer": OPTIMIZER,
        "use_chain_of_thought": USE_CHAIN_OF_THOUGHT,
        "n_train": len(trainset),
        "n_dev": len(devset),
        "n_full": len(examples),
        "baseline_devset_score": base_score,
        "optimized_devset_score": opt_score,
        "optimized_full_score": full_score,
        "optimized_ok": optimized_ok,
        "results_full": full_rows,
    }, f, indent=2)
print(f"Saved run -> {run_path}")

dbutils.notebook.exit(json.dumps({
    "task_name": TASK["name"],
    "optimizer": opt_type,
    "baseline_devset": base_score,
    "optimized_devset": opt_score,
    "optimized_full_score": full_score,
    "optimized_ok": optimized_ok,
    "run_path": run_path,
    "program_path": prog_path,
    "prompt_md_path": md_path,
}))

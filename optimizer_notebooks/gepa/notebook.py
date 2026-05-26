# Databricks notebook source
# MAGIC %md
# MAGIC # GEPA Prompt Optimizer — generic, parameterized
# MAGIC
# MAGIC Reflective prompt optimization. You fill in **just the CONFIG block at the top**
# MAGIC (eval data, seed prompt, task model, reflector model). The notebook handles
# MAGIC the rest: loads models, evaluates the seed prompt, identifies failures,
# MAGIC asks the reflector LLM to revise the prompt, evaluates the revision,
# MAGIC iterates, tracks best-so-far, and saves results.
# MAGIC
# MAGIC ## Supported configurations
# MAGIC
# MAGIC | Modality | Task model providers | Reflector model providers |
# MAGIC |---|---|---|
# MAGIC | image    | `huggingface_local` (Qwen-VL, etc.), `databricks_fmapi` (Claude/GPT/Gemini vision) | `databricks_fmapi` |
# MAGIC | text     | `huggingface_local`, `databricks_fmapi`                              | `databricks_fmapi` |
# MAGIC
# MAGIC ## Compute
# MAGIC
# MAGIC | Task model = | Recommended compute |
# MAGIC |---|---|
# MAGIC | `huggingface_local` (8B+) | 8× H100 |
# MAGIC | `huggingface_local` (≤4B) | 1× A10 |
# MAGIC | `databricks_fmapi` only   | serverless CPU |
# MAGIC
# MAGIC ## How to use
# MAGIC
# MAGIC 1. Edit the **`# === CONFIG ===`** cell to your task.
# MAGIC 2. Run all cells.
# MAGIC 3. Find the optimized prompt in `OUTPUT_DIR/<task_name>/best_prompt.txt`
# MAGIC    and full results in `OUTPUT_DIR/<task_name>/run.json`.

# COMMAND ----------

# MAGIC %pip install "transformers>=4.57" accelerate "torchvision>=0.20" pillow timm einops sentencepiece openai databricks-sdk pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## === CONFIG ===  (edit these for your task)

# COMMAND ----------

import os, json

# -- Where to save outputs --
OUTPUT_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/gepa_runs"

# -- Task definition --
TASK = {
    "name": "arthroscopy_instrument_id",          # used in output paths
    "description": (
        "Classify the primary surgical instrument visible in a single frame "
        "from a knee arthroscopy video."
    ),
    "input_modality": "image",                    # "image" or "text"
    # How to extract a comparable label from the model's raw output.
    # Options: "exact_match" | "class_in_text" | "json_field"
    "metric": "json_field",
    # Used when metric == "json_field": dotted path into the parsed JSON, e.g.
    #   "instruments[0].class"
    "json_field_path": "instruments[0].class",
    # Used when metric == "class_in_text": a list of allowed class strings; the
    # first one found in the model's output is taken as its prediction.
    "vocabulary": [
        "probe", "shaver", "burr", "grasper", "biter", "suture_passer",
        "anchor_driver", "electrocautery", "cannula", "scissors", "drill_guide",
        "trocar", "knot_pusher", "rasp", "other_metal_tool", "no_instrument_visible",
    ],
}

# -- Eval dataset --
# A list of dicts. Each dict needs:
#   - "input":    image path (image modality) OR string (text modality)
#   - "expected": gold label / expected output (string)
#
# This example reads frames from a Volume + their Sonnet-4.5 pseudo-labels.
# Replace this with your own loader.
def load_eval_data():
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
            if cls:
                sonnet_labels[r["frame"]] = cls
    examples = []
    for fname in sorted(os.listdir(EVAL_FRAMES_DIR)):
        if not fname.lower().endswith(".jpg"):
            continue
        if fname not in sonnet_labels:
            continue
        examples.append({
            "input": os.path.join(EVAL_FRAMES_DIR, fname),
            "expected": sonnet_labels[fname],
            "id": fname,
        })
    return examples

EVAL_DATA = None  # filled in later via load_eval_data() — keep as None to defer

# -- Optional: hold out a fraction for validation (avoid fitting to the whole set) --
HOLDOUT_FRAC = 0.0   # 0 = use all data for both eval and reflection. 0.3 = 30% holdout.

# -- Seed prompt --
# This is the starting point that GEPA will refine. Use whatever you have today.
SEED_PROMPT = """\
Identify the primary surgical instrument visible in this knee arthroscopy frame.

Vocabulary (pick exactly one):
probe, shaver, burr, grasper, biter, suture_passer, anchor_driver,
electrocautery, cannula, scissors, drill_guide, trocar, knot_pusher, rasp,
other_metal_tool, no_instrument_visible.

Respond with strict JSON:
{"instruments": [{"class": "<one of vocab>", "confidence": 0.0-1.0,
"evidence": "<short text>"}], "anatomy": "<short text>",
"tissue_condition": "<short text>"}
"""

# -- Task model: the model whose prompt we are optimizing --
TASK_MODEL = {
    "provider": "huggingface_local",            # "huggingface_local" or "databricks_fmapi"
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",   # for huggingface_local
    # "endpoint": "databricks-meta-llama-4-maverick",  # for databricks_fmapi
    "max_new_tokens": 400,
    "temperature": 0.0,
    "image_max_side": 896,
}

# -- Reflector model: meta-optimizer; reads failures and proposes a new prompt --
REFLECTOR_MODEL = {
    "provider": "databricks_fmapi",
    "endpoint": "databricks-claude-sonnet-4-6",
    "max_tokens": 1500,
    "temperature": 0.4,
}

# -- GEPA loop --
OPTIMIZER = {
    "n_rounds": 5,
    "n_failure_samples": 6,
    # If reflector should REPLACE its previous revision (not append to it).
    # We've seen "append" mode bloat prompts and regress accuracy.
    "reflection_mode": "replace",   # "replace" or "append"
    "early_stop_patience": 2,       # stop after N rounds with no improvement
    "seed": 42,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## YAML override (for non-interactive runs)
# MAGIC
# MAGIC If a YAML config is supplied via the `config_yaml` widget, its values
# MAGIC override the CONFIG block above. This lets the workbench backend kick
# MAGIC the notebook off as a Databricks Job with `notebook_params={"config_yaml": "/Volumes/.../<id>.yaml"}`.

# COMMAND ----------

dbutils.widgets.text("config_yaml", "", "Path to YAML config (optional)")

_yaml_path = dbutils.widgets.get("config_yaml").strip()
if _yaml_path:
    import yaml
    print(f"Loading config override from {_yaml_path}")
    with open(_yaml_path) as _f:
        _cfg = yaml.safe_load(_f) or {}

    # Top-level overrides
    if "output_dir" in _cfg:
        OUTPUT_DIR = _cfg["output_dir"]
    if "task" in _cfg:
        TASK = {**TASK, **_cfg["task"]}
    if "seed_prompt" in _cfg:
        SEED_PROMPT = _cfg["seed_prompt"]
    if "task_model" in _cfg:
        TASK_MODEL = {**TASK_MODEL, **_cfg["task_model"]}
    if "reflector_model" in _cfg:
        REFLECTOR_MODEL = {**REFLECTOR_MODEL, **_cfg["reflector_model"]}
    if "optimizer" in _cfg:
        OPTIMIZER = {**OPTIMIZER, **_cfg["optimizer"]}
    if "holdout_frac" in _cfg:
        HOLDOUT_FRAC = float(_cfg["holdout_frac"])

    # Inline eval data — backend pre-resolves snapshots into this list
    if "eval_data" in _cfg:
        EVAL_DATA = list(_cfg["eval_data"])
    else:
        EVAL_DATA = None
    print(f"  task: {TASK.get('name')}  optimizer: {OPTIMIZER.get('reflection_mode')} × {OPTIMIZER.get('n_rounds')} rounds")
    print(f"  task_model: {TASK_MODEL}")
    print(f"  reflector:  {REFLECTOR_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup (no edits below this line)

# COMMAND ----------

import io, json, time, copy, random, base64, traceback, re
from typing import Any, Dict, List, Optional, Callable
import numpy as np

random.seed(OPTIMIZER["seed"]); np.random.seed(OPTIMIZER["seed"])

os.makedirs(OUTPUT_DIR, exist_ok=True)
TASK_OUT = os.path.join(OUTPUT_DIR, TASK["name"])
os.makedirs(TASK_OUT, exist_ok=True)
print(f"Output dir: {TASK_OUT}")

EVAL_DATA = load_eval_data() if EVAL_DATA is None else EVAL_DATA
print(f"Loaded {len(EVAL_DATA)} eval examples")

# Optional holdout split
if HOLDOUT_FRAC > 0:
    rng = random.Random(OPTIMIZER["seed"])
    shuffled = list(EVAL_DATA)
    rng.shuffle(shuffled)
    n_holdout = int(HOLDOUT_FRAC * len(shuffled))
    HOLDOUT = shuffled[:n_holdout]
    OPTIM_SET = shuffled[n_holdout:]
else:
    HOLDOUT = []
    OPTIM_SET = list(EVAL_DATA)
print(f"Optimization set: {len(OPTIM_SET)}  holdout: {len(HOLDOUT)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task model adapter
# MAGIC
# MAGIC Two providers, one interface. `predict(prompt, example_input) -> raw_text`.

# COMMAND ----------

import torch
from PIL import Image

class TaskModel:
    def __init__(self, cfg, modality):
        self.cfg = cfg
        self.modality = modality
        self.provider = cfg["provider"]
        if self.provider == "huggingface_local":
            self._load_hf()
        elif self.provider == "databricks_fmapi":
            self._load_fmapi()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _load_hf(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
        os.environ["HF_TOKEN"] = dbutils.secrets.get("hls_g4", "HF_TOKEN")
        print(f"Loading {self.cfg['model_id']}...")
        if self.modality == "image":
            self.processor = AutoProcessor.from_pretrained(self.cfg["model_id"], token=os.environ["HF_TOKEN"])
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
                token=os.environ["HF_TOKEN"],
            ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_id"], token=os.environ["HF_TOKEN"])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
                token=os.environ["HF_TOKEN"],
            ).eval()
        print("loaded")

    def _load_fmapi(self):
        from openai import OpenAI
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        token = w.tokens.create(comment="gepa-task", lifetime_seconds=7200).token_value
        self.client = OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")

    def _load_pil(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((self.cfg["image_max_side"], self.cfg["image_max_side"]), Image.LANCZOS)
        return img

    def _b64_pil(self, pil):
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @torch.no_grad()
    def _predict_hf_image(self, prompt, image_path):
        img = self._load_pil(image_path)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg["max_new_tokens"],
            do_sample=(self.cfg["temperature"] > 0),
            temperature=max(self.cfg["temperature"], 1e-5),
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def _predict_hf_text(self, prompt, text_input):
        full = f"{prompt}\n\nInput: {text_input}\n\nResponse:"
        ids = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids, max_new_tokens=self.cfg["max_new_tokens"],
            do_sample=(self.cfg["temperature"] > 0),
            temperature=max(self.cfg["temperature"], 1e-5),
        )
        return self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _predict_fmapi_image(self, prompt, image_path):
        pil = self._load_pil(image_path)
        b64 = self._b64_pil(pil)
        resp = self.client.chat.completions.create(
            model=self.cfg["endpoint"],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            max_tokens=self.cfg["max_new_tokens"],
            temperature=self.cfg["temperature"],
        )
        return resp.choices[0].message.content or ""

    def _predict_fmapi_text(self, prompt, text_input):
        resp = self.client.chat.completions.create(
            model=self.cfg["endpoint"],
            messages=[{"role": "user", "content": f"{prompt}\n\n{text_input}"}],
            max_tokens=self.cfg["max_new_tokens"],
            temperature=self.cfg["temperature"],
        )
        return resp.choices[0].message.content or ""

    def predict(self, prompt, example_input):
        if self.provider == "huggingface_local":
            if self.modality == "image":
                return self._predict_hf_image(prompt, example_input)
            return self._predict_hf_text(prompt, example_input)
        if self.provider == "databricks_fmapi":
            if self.modality == "image":
                return self._predict_fmapi_image(prompt, example_input)
            return self._predict_fmapi_text(prompt, example_input)

task_lm = TaskModel(TASK_MODEL, TASK["input_modality"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reflector model adapter

# COMMAND ----------

class ReflectorModel:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg["provider"] != "databricks_fmapi":
            raise ValueError("Reflector currently only supports databricks_fmapi")
        from openai import OpenAI
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        token = w.tokens.create(comment="gepa-reflector", lifetime_seconds=7200).token_value
        self.client = OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")

    def call(self, content_blocks):
        resp = self.client.chat.completions.create(
            model=self.cfg["endpoint"],
            messages=[{"role": "user", "content": content_blocks}],
            max_tokens=self.cfg["max_tokens"],
            temperature=self.cfg["temperature"],
        )
        return resp.choices[0].message.content or ""

reflector = ReflectorModel(REFLECTOR_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metric / parsing
# MAGIC
# MAGIC `parse(raw_text) -> normalized_label` then `score(normalized, expected) -> bool`.

# COMMAND ----------

def _strip_md_fences(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip().rstrip("`").strip()
    return t

def _walk_json_path(d, path):
    """Walk a path like 'instruments[0].class' through a parsed dict."""
    cur = d
    for tok in re.findall(r"[^.\[\]]+", path):
        if tok.isdigit():
            try:
                cur = cur[int(tok)]
            except (IndexError, TypeError):
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(tok)
            else:
                return None
        if cur is None:
            return None
    return cur

def _extract_class_from_parsed(d) -> str | None:
    """Extract the predicted class from a parsed JSON object, mirroring the
    Playground frontend's ``primaryClass`` semantics:
      - {"instrument": "shaver"}                → "shaver"
      - {"instrument": ""}                       → "no_instrument_visible"
      - {"instruments": [{"class": "shaver"}]}   → "shaver"
      - {"instruments": []}                      → "no_instrument_visible"
      - parsed object exists, no recognizable field → "no_instrument_visible"
      - not a dict / can't be interpreted        → None
    Without this, GEPA's score collapses because Qwen3-VL emits
    ``{"instruments": []}`` for empty frames, which used to fail JSON-path
    walk and then mismatch the vocab on the substring fallback (the literal
    ``no_instrument_visible`` token never appears in the model's prose)."""
    if not isinstance(d, dict):
        return None
    if "instrument" in d:
        v = d["instrument"]
        if isinstance(v, str):
            return v.strip() or "no_instrument_visible"
    if "instruments" in d:
        lst = d["instruments"]
        if isinstance(lst, list):
            if len(lst) == 0:
                return "no_instrument_visible"
            if isinstance(lst[0], dict):
                c = lst[0].get("class")
                if isinstance(c, str):
                    return c.strip() or "no_instrument_visible"
    return "no_instrument_visible"


def parse_prediction(raw: str) -> str:
    metric = TASK["metric"]
    if metric == "exact_match":
        return (raw or "").strip()
    # All non-exact metrics: try JSON parse first (robust to prompt/reasoning
    # mentioning vocab words). Fall back to substring scan only when no
    # parseable JSON object is present.
    t = _strip_md_fences(raw or "")
    parsed_dict = None
    if "{" in t and "}" in t:
        try:
            start, end = t.find("{"), t.rfind("}")
            if 0 <= start < end:
                parsed_dict = json.loads(t[start:end + 1])
        except Exception:
            parsed_dict = None
    if parsed_dict is not None:
        # First try any explicit json_field_path(s) declared by the task config
        # — these win even when the path happens to point at something else.
        for path in [TASK.get("json_field_path"), *TASK.get("json_field_path_alts", [])]:
            if not path:
                continue
            v = _walk_json_path(parsed_dict, path)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Otherwise apply the same shape-agnostic extractor the Playground UI
        # uses so identical predictions score the same here and there.
        c = _extract_class_from_parsed(parsed_dict)
        if c:
            return c
    if metric == "json_field":
        return "?"
    # class_in_text: vocabulary substring scan as last resort
    text = (raw or "").lower()
    for v in TASK.get("vocabulary", []):
        if v in text:
            return v
    return "?"

def score_prediction(predicted: str, expected: str) -> bool:
    return (predicted or "").strip().lower() == (expected or "").strip().lower()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval loop

# COMMAND ----------

def evaluate(prompt: str, dataset: List[Dict]) -> Dict:
    rows = []
    matched = 0
    for ex in dataset:
        t0 = time.time()
        try:
            raw = task_lm.predict(prompt, ex["input"])
        except Exception as e:
            raw = f"[error: {str(e)[:200]}]"
        elapsed = time.time() - t0
        predicted = parse_prediction(raw)
        ok = score_prediction(predicted, ex["expected"])
        if ok: matched += 1
        rows.append({
            "id": ex.get("id", ex["input"][:60]),
            "expected": ex["expected"],
            "predicted": predicted,
            "match": ok,
            "raw": (raw or "")[:500],
            "elapsed_s": round(elapsed, 2),
        })
    return {"score": matched / len(dataset) if dataset else 0,
            "matched": matched, "total": len(dataset), "rows": rows}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reflection step

# COMMAND ----------

REFLECTION_INSTRUCTION_REPLACE = """\
You are an expert prompt engineer optimizing instructions for a {provider} model.

Task description: {task_description}

I will show you:
1. The CURRENT PROMPT being used.
2. The current accuracy on the eval set.
3. A handful of FAILURE CASES — each shows the model's prediction and the gold label.{image_clause}

Your job: write a REVISED prompt that should reduce these specific failure modes
while not regressing on what already works. Make targeted changes — clarify a
class definition, add a discriminative cue between confused classes, remove
ambiguity. Do NOT just append more rules to what's there: REWRITE the relevant
sections so the prompt stays focused. Keep the same output format. Keep length
similar to the current prompt (smaller models lose signal in verbose prompts).

Output ONLY the revised prompt text — no preamble, no commentary, no markdown
fences.
"""

REFLECTION_INSTRUCTION_APPEND = """\
You are an expert prompt engineer optimizing instructions for a {provider} model.

Task description: {task_description}

Below: current prompt, accuracy, and FAILURE CASES.{image_clause}

Append targeted clarifications to the existing prompt to fix these failures.
Output the full revised prompt. No preamble or fences.
"""

def build_reflection_request(current_prompt, eval_result, failures, modality):
    inst = (REFLECTION_INSTRUCTION_REPLACE if OPTIMIZER["reflection_mode"] == "replace"
            else REFLECTION_INSTRUCTION_APPEND)
    image_clause = (
        "\nFor each failure you'll see the input image, the model's prediction, and the gold label."
        if modality == "image" else ""
    )
    inst = inst.format(
        provider=TASK_MODEL["provider"],
        task_description=TASK["description"],
        image_clause=image_clause,
    )
    blocks = [{"type": "text", "text": inst}]
    blocks.append({"type": "text",
                   "text": f"\n=== ACCURACY: {eval_result['score']:.1%} "
                           f"({eval_result['matched']}/{eval_result['total']}) ===\n"})
    blocks.append({"type": "text", "text": "\n=== CURRENT PROMPT ===\n" + current_prompt})
    blocks.append({"type": "text", "text": "\n=== FAILURE CASES ==="})

    for i, fc in enumerate(failures):
        blocks.append({"type": "text",
                       "text": f"\nFailure {i+1}: predicted '{fc['predicted']}', gold '{fc['expected']}'."})
        if modality == "image":
            try:
                pil = task_lm._load_pil(_orig_input_lookup(fc["id"]))
                b64 = task_lm._b64_pil(pil)
                blocks.append({"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:
                blocks.append({"type": "text", "text": f"[image unavailable: {e}]"})
        else:
            txt = _orig_input_lookup(fc["id"])
            blocks.append({"type": "text", "text": f"\nINPUT: {str(txt)[:1000]}"})
    blocks.append({"type": "text", "text": "\n\nRevised prompt:"})
    return blocks

# Helper to recover original input from row id
_id_to_input = {ex.get("id", ex["input"][:60]): ex["input"] for ex in EVAL_DATA}
def _orig_input_lookup(row_id): return _id_to_input.get(row_id, row_id)

def reflect(current_prompt, eval_result):
    failures = [r for r in eval_result["rows"] if not r["match"]]
    if not failures:
        return None, []
    sample = random.sample(failures, min(OPTIMIZER["n_failure_samples"], len(failures)))
    blocks = build_reflection_request(current_prompt, eval_result, sample, TASK["input_modality"])
    revised = reflector.call(blocks)
    revised = _strip_md_fences(revised)
    return revised, sample

# COMMAND ----------

# MAGIC %md
# MAGIC ## GEPA loop

# COMMAND ----------

# MLflow tracking: log every round's score so the optimization trajectory is
# visible alongside fine-tune runs in the same experiment. Wraps the whole
# loop in a single run (one row in the MLflow UI per optimize submission).
try:
    import mlflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_PATH",
        "/Users/guanyu.chen@databricks.com/vlmwb-experiments"))
    _ml_run = mlflow.start_run(run_name=TASK.get("name", "gepa"))
    mlflow.log_params({
        "kind": "gepa",
        "task_model_provider": TASK_MODEL.get("provider"),
        "task_model": TASK_MODEL.get("endpoint") or TASK_MODEL.get("model_id"),
        "reflector_model": REFLECTOR_MODEL.get("endpoint"),
        "n_rounds": OPTIMIZER.get("n_rounds"),
        "n_examples": len(OPTIM_SET),
        "reflection_mode": OPTIMIZER.get("reflection_mode"),
    })
    _ml_available = True
except Exception as e:
    print(f"[gepa][warn] MLflow tracking disabled: {e}")
    _ml_run = None
    _ml_available = False

history = []
best_prompt = SEED_PROMPT
best_score = -1.0
best_round = 0
no_improve = 0
current_prompt = SEED_PROMPT

t_total = time.time()
for round_idx in range(OPTIMIZER["n_rounds"]):
    print(f"\n=== Round {round_idx} ===")
    t0 = time.time()
    eval_result = evaluate(current_prompt, OPTIM_SET)
    elapsed = time.time() - t0
    print(f"  score={eval_result['score']:.1%}  ({eval_result['matched']}/{eval_result['total']})  elapsed={elapsed:.0f}s")
    history.append({
        "round": round_idx,
        "prompt": current_prompt,
        "score": eval_result["score"],
        "matched": eval_result["matched"],
        "total": eval_result["total"],
        "rows": eval_result["rows"],
    })
    if _ml_available:
        try:
            mlflow.log_metric("round_score", eval_result["score"], step=round_idx)
            mlflow.log_metric("round_matched", eval_result["matched"], step=round_idx)
            mlflow.log_metric("round_elapsed_s", elapsed, step=round_idx)
        except Exception:
            pass
    if eval_result["score"] > best_score:
        best_score = eval_result["score"]
        best_prompt = current_prompt
        best_round = round_idx
        no_improve = 0
        print(f"  ** new best: {best_score:.1%}")
    else:
        no_improve += 1
        if no_improve >= OPTIMIZER["early_stop_patience"]:
            print(f"  early stop after {no_improve} rounds without improvement")
            break

    if round_idx == OPTIMIZER["n_rounds"] - 1:
        break

    # Reflect (always reflect from BEST prompt, not last prompt — avoids drift in append/replace mixes)
    print(f"  reflecting on {min(OPTIMIZER['n_failure_samples'], eval_result['total']-eval_result['matched'])} failures via {REFLECTOR_MODEL['endpoint']}...")
    base_for_reflection = best_prompt if OPTIMIZER["reflection_mode"] == "replace" else current_prompt
    base_eval = evaluate(base_for_reflection, OPTIM_SET) if base_for_reflection != current_prompt else eval_result
    try:
        revised, _ = reflect(base_for_reflection, base_eval)
        if revised:
            print(f"  revised prompt: {len(revised)} chars")
            current_prompt = revised
        else:
            print("  no failures to reflect on; stopping")
            break
    except Exception as e:
        print(f"  reflection failed: {e}; stopping")
        traceback.print_exc()
        break

print(f"\nTotal optimization time: {time.time() - t_total:.0f}s")
print(f"Best round={best_round}  best_score={best_score:.1%}")

if _ml_available:
    try:
        mlflow.log_metric("best_score_optim", best_score)
        mlflow.log_metric("best_round", best_round)
        mlflow.log_metric("total_elapsed_s", time.time() - t_total)
        # Log the optimized prompt as an artifact for easy comparison in MLflow.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(best_prompt)
            _bp = f.name
        mlflow.log_artifact(_bp, artifact_path="prompts")
    except Exception:
        pass
    try:
        mlflow.end_run()
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Holdout evaluation (if configured)

# COMMAND ----------

holdout_result = None
if HOLDOUT:
    print("Evaluating BEST prompt on HOLDOUT...")
    holdout_result = evaluate(best_prompt, HOLDOUT)
    print(f"  holdout score: {holdout_result['score']:.1%} ({holdout_result['matched']}/{holdout_result['total']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save

# COMMAND ----------

best_prompt_path = os.path.join(TASK_OUT, "best_prompt.txt")
with open(best_prompt_path, "w") as f:
    f.write(best_prompt)

run_path = os.path.join(TASK_OUT, "run.json")
with open(run_path, "w") as f:
    json.dump({
        "task": TASK,
        "task_model": TASK_MODEL,
        "reflector_model": REFLECTOR_MODEL,
        "optimizer": OPTIMIZER,
        "n_examples_optim": len(OPTIM_SET),
        "n_examples_holdout": len(HOLDOUT),
        "seed_prompt": SEED_PROMPT,
        "best_prompt": best_prompt,
        "best_round": best_round,
        "best_score_optim": best_score,
        "holdout_score": holdout_result["score"] if holdout_result else None,
        "round_scores": [h["score"] for h in history],
        "history": history,
    }, f, indent=2)

print(f"\nSaved:\n  {best_prompt_path}\n  {run_path}")
print(f"\nRound trajectory: {[round(h['score'], 3) for h in history]}")
print(f"Best optim score: {best_score:.1%}  best round: {best_round}")
if holdout_result:
    print(f"Holdout score:    {holdout_result['score']:.1%}")

# Return summary so caller (a Job) can see top-line via get-run-output
dbutils.notebook.exit(json.dumps({
    "task_name": TASK["name"],
    "best_score_optim": best_score,
    "holdout_score": holdout_result["score"] if holdout_result else None,
    "best_round": best_round,
    "round_scores": [h["score"] for h in history],
    "best_prompt_path": best_prompt_path,
    "run_path": run_path,
}))

# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow GenAI primitives — verification (defensive)
# MAGIC
# MAGIC The earlier version vanished in 40s with status SUCCESS but no output.
# MAGIC This rewrite writes a status file FIRST, then attempts each primitive
# MAGIC in a try/except that always falls through, then dumps the result to
# MAGIC the Volume so we can see exactly which call failed and how.

# COMMAND ----------
import os, sys, json, time, traceback

OUTPUT_DIR = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/_mlflow_genai_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)
result = {
    "py_version": sys.version,
    "started_at": time.time(),
    "steps": [],
    "imports": {},
}

# Sanity write so we know the notebook ran at all
with open(f"{OUTPUT_DIR}/started.txt", "w") as f:
    f.write(f"started {time.time()}\npy={sys.version}\n")
print(f"[verify] notebook started, OUTPUT_DIR={OUTPUT_DIR}")

# COMMAND ----------
# Try to import mlflow at the version already in env_v5 first. If it's too
# old, pip install + manually re-import (no %restart_python — that magic
# swallows the output and can short-circuit job runs).
def _try_import_mlflow():
    try:
        import mlflow
        return mlflow.__version__
    except Exception as e:
        return f"FAIL: {e}"

result["imports"]["before_install"] = _try_import_mlflow()
print(f"[verify] mlflow before install: {result['imports']['before_install']}")

# COMMAND ----------
# Only pip install if mlflow < 3.0
import subprocess, importlib
def _maybe_upgrade():
    try:
        import mlflow
        major = int(mlflow.__version__.split('.')[0])
        if major >= 3:
            return f"already {mlflow.__version__}, skipping pip install"
    except Exception:
        pass
    # Use subprocess directly so output is captured
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "mlflow>=3.0", "psycopg[binary]", "--quiet"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return f"pip rc={out.returncode}, stderr_tail={out.stderr[-300:]}"

result["imports"]["pip_install"] = _maybe_upgrade()
print(f"[verify] pip: {result['imports']['pip_install']}")

# Force re-import (in case we just installed a newer version)
for k in list(sys.modules.keys()):
    if k == "mlflow" or k.startswith("mlflow."):
        del sys.modules[k]
import mlflow
result["imports"]["after"] = mlflow.__version__
result["imports"]["has_genai"] = hasattr(mlflow, "genai")
print(f"[verify] mlflow after: {mlflow.__version__}  has_genai={hasattr(mlflow, 'genai')}")
if hasattr(mlflow, "genai"):
    result["imports"]["genai_dir"] = [x for x in dir(mlflow.genai) if not x.startswith('_')]
    print(f"[verify] genai exports: {result['imports']['genai_dir']}")

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
EXPERIMENT_PATH = "/Users/guanyu.chen@databricks.com/vlmwb-experiments"
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
    result["experiment"] = {"ok": True, "path": EXPERIMENT_PATH}
except Exception as e:
    result["experiment"] = {"ok": False, "error": str(e)[:300]}
print(f"[verify] experiment: {result['experiment']}")

# COMMAND ----------
def step(name, fn):
    t0 = time.time()
    rec = {"step": name}
    try:
        out = fn()
        rec["ok"] = True
        rec["out"] = str(out)[:1500]
    except Exception as e:
        rec["ok"] = False
        rec["error"] = str(e)[:500]
        rec["trace_tail"] = traceback.format_exc()[-1500:]
    rec["elapsed_s"] = round(time.time() - t0, 2)
    result["steps"].append(rec)
    print(f"\n=== {name}  {'OK' if rec.get('ok') else 'FAIL'}  ({rec['elapsed_s']}s) ===")
    if rec.get("ok"):
        print(rec["out"][:400])
    else:
        print(rec.get("error"))
        print(rec.get("trace_tail", "")[:600])
    # Persist after every step so we have partial progress even on crash
    with open(f"{OUTPUT_DIR}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    return rec

# COMMAND ----------
DEFAULT_PROMPT = (
    'Identify the surgical instrument by its visible visual features. '
    'Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", '
    '"tissue_condition": "<short>"} where <class> is one of: probe, shaver, '
    'burr, grasper, biter, suture_passer, anchor_driver, electrocautery, '
    'cannula, scissors, drill_guide, trocar, knot_pusher, rasp, '
    'other_metal_tool, no_instrument_visible.'
)
# Prompts + datasets in MLflow 3.x are UC-managed — names must be three-part
# (catalog.schema.name), same convention as registered models.
UC_CATALOG = "hls_amer_catalog"
UC_SCHEMA = "guanyu_chen"
PROMPT_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.vlmwb_instrument_id"
DS_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.vlmwb_eval_labels"

# COMMAND ----------
# Step 1: register_prompt
def s1():
    p = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=DEFAULT_PROMPT,
        commit_message="vlmwb: baseline instrument-ID prompt",
        tags={"source": "verification", "vlmwb_role": "task_prompt"},
    )
    return {
        "name": getattr(p, "name", None),
        "version": getattr(p, "version", None),
        "uri": getattr(p, "uri", None),
        "type": type(p).__name__,
    }
step("register_prompt", s1)

# COMMAND ----------
# Step 1b: load_prompt by URI
def s1b():
    p = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/1")
    return {"template_len": len(p.template), "version": p.version, "type": type(p).__name__}
step("load_prompt", s1b)

# COMMAND ----------
# Step 2: create_dataset
def s2():
    from mlflow.genai import datasets as gds
    print(f"  datasets dir: {[x for x in dir(gds) if not x.startswith('_')]}")
    try:
        ds = gds.create_dataset(name=DS_NAME)
    except Exception as e:
        # Maybe it already exists — find it
        print(f"  create_dataset raised: {e}; trying get_dataset")
        ds = gds.get_dataset(DS_NAME)
    print(f"  dataset type: {type(ds).__name__}")
    print(f"  dataset dir: {[x for x in dir(ds) if not x.startswith('_')][:25]}")
    rows = [
        {"inputs": {"prompt": DEFAULT_PROMPT, "image_path": "/Volumes/.../frame_0003.jpg"},
         "expectations": {"instrument": "no_instrument_visible"}},
        {"inputs": {"prompt": DEFAULT_PROMPT, "image_path": "/Volumes/.../frame_0124.jpg"},
         "expectations": {"instrument": "probe"}},
        {"inputs": {"prompt": DEFAULT_PROMPT, "image_path": "/Volumes/.../frame_0145.jpg"},
         "expectations": {"instrument": "shaver"}},
    ]
    ds.merge_records(rows)
    return {"name": getattr(ds, 'name', '?'), "n_records": len(rows)}
step("create_dataset", s2)

# COMMAND ----------
# Step 3: custom scorer
def s3():
    from mlflow.genai.scorers import scorer

    @scorer
    def instrument_match(outputs, expectations):
        if not isinstance(outputs, dict):
            return 0.0
        pred = (outputs.get("instrument") or "").strip().lower()
        gold = (expectations.get("instrument") or "").strip().lower()
        return 1.0 if pred and pred == gold else 0.0

    # CustomScorer is a pydantic object — use .name (not __name__)
    return {"defined": getattr(instrument_match, "name", "?"), "type": type(instrument_match).__name__}
step("define_scorer", s3)

# COMMAND ----------
# Step 4: evaluate
def s4():
    from mlflow.genai.scorers import scorer
    from mlflow.genai import datasets as gds

    @scorer
    def instrument_match(outputs, expectations):
        if not isinstance(outputs, dict):
            return 0.0
        pred = (outputs.get("instrument") or "").strip().lower()
        gold = (expectations.get("instrument") or "").strip().lower()
        return 1.0 if pred and pred == gold else 0.0

    def predict(prompt: str, image_path: str):
        # Stub: always returns "probe" so we can verify scoring works without
        # actually calling a model.
        return {"instrument": "probe", "anatomy": "?", "tissue_condition": "?"}

    ds = gds.get_dataset(DS_NAME)
    eval_result = mlflow.genai.evaluate(
        data=ds,
        predict_fn=predict,
        scorers=[instrument_match],
    )
    return {
        "type": type(eval_result).__name__,
        "run_id": getattr(eval_result, "run_id", None) or "?",
        "summary": str(eval_result)[:500],
    }
step("evaluate", s4)

# COMMAND ----------
result["finished_at"] = time.time()
with open(f"{OUTPUT_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)
n_ok = sum(1 for s in result["steps"] if s.get("ok"))
n_total = len(result["steps"])
print(f"\n=== Summary: {n_ok}/{n_total} steps succeeded ===")
print(f"Wrote {OUTPUT_DIR}/result.json")

dbutils.notebook.exit(json.dumps({"ok": n_ok, "total": n_total, "mlflow_version": mlflow.__version__}))

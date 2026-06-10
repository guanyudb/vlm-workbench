"""FastAPI backend for the Surgical VLM Workbench (Phase 1: Playground only).

Endpoints:
  GET  /api/health                    — liveness
  GET  /api/videos                    — list .mp4 files in $VOLUME_PATH
  GET  /api/frames                    — list pre-extracted frames in $EXTRACTED_FRAMES_DIR
  GET  /api/frames/image              — stream a single frame JPEG (?path=...)
  GET  /api/models                    — vision-capable databricks-* chat endpoints
  POST /api/playground/run            — kick off a parallel multi-model run, stream via SSE

Volume access uses the Databricks Files API (`w.files.*`) because Databricks
Apps don't FUSE-mount Volumes — `os.listdir("/Volumes/...")` returns empty.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
# mlflow.genai requires the Databricks SDK auth path. Set BEFORE the first
# `import mlflow` anywhere in the process. MLflow's env-var parser only
# accepts the literal "1" / "True" — "true" doesn't trip the boolean check.
# Also pin tracking + registry URIs so MLflow doesn't fall back to a local
# sqlite file when no client has called set_tracking_uri yet (which it does
# inside `mlflow.genai.register_prompt` before our hook runs).
os.environ.setdefault("MLFLOW_ENABLE_DB_SDK", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vlm-workbench")

app = FastAPI(title="Surgical VLM Workbench")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Databricks Apps' runtime sets DATABRICKS_HOST to a bare hostname (no
# scheme). Mlflow/SDK code paths that build URLs by string concat then
# produce schemeless URLs and `requests` raises `MissingSchema`. Normalize
# once at boot so every downstream consumer sees a fully-qualified URL.
_dh = os.environ.get("DATABRICKS_HOST", "").strip()
if _dh and not _dh.startswith(("http://", "https://")):
    os.environ["DATABRICKS_HOST"] = "https://" + _dh.lstrip("/")

# UC layout — sourced via DAB secret bindings (UC_CATALOG/UC_SCHEMA/VOLUME_NAME)
# so a fresh workspace deploys without editing the source. We accept the
# pre-DAB env var names (CATALOG/SCHEMA, hardcoded VOLUME_PATH) as fallbacks
# so existing prod deploys keep working until they migrate to the bundle.
UC_CATALOG_BOOT = os.environ.get("UC_CATALOG") or os.environ.get("CATALOG") or "hls_amer_catalog"
UC_SCHEMA_BOOT = os.environ.get("UC_SCHEMA") or os.environ.get("SCHEMA") or "guanyu_chen"
VOLUME_NAME_BOOT = os.environ.get("VOLUME_NAME", "medical_video")
VOLUME_PATH = os.environ.get(
    "VOLUME_PATH",
    f"/Volumes/{UC_CATALOG_BOOT}/{UC_SCHEMA_BOOT}/{VOLUME_NAME_BOOT}",
)
EXTRACTED_FRAMES_DIR = os.environ.get(
    "EXTRACTED_FRAMES_DIR",
    os.path.join(VOLUME_PATH, "extracted_frames"),
)
EVAL_FRAMES_DIR = os.environ.get(
    "EVAL_FRAMES_DIR",
    os.path.join(VOLUME_PATH, "eval_frames"),
)
STUDIO_CACHE_DIR = os.environ.get(
    "STUDIO_CACHE_DIR",
    os.path.join(VOLUME_PATH, "studio_analyses"),
)
# ── Task configuration (vocabulary + output schema + prompt template) ────
#
# All workbench task semantics live in one editable JSON file on the Volume:
#   <VOLUME_PATH>/config/task_config.json
#
# Single source of truth — replaces the five places that used to hardcode
# the instrument vocabulary (Playground.tsx, Label.tsx, app.yaml's
# DEFAULT_PROMPT, the optimizer's vocab seed, and Studio's per-frame prompt).
# Setup tab edits this; Playground reads `rendered_prompt`; Label reads
# `vocabulary`; snapshots embed the active config at save-time.
TASK_CONFIG_PATH = f"{VOLUME_PATH}/config/task_config.json"

_DEFAULT_TASK_CONFIG: Dict[str, object] = {
    "vocabulary": [
        "probe", "shaver", "burr", "grasper", "biter", "suture_passer",
        "anchor_driver", "electrocautery", "cannula", "scissors",
        "drill_guide", "trocar", "knot_pusher", "rasp", "other_metal_tool",
        "no_instrument_visible",
    ],
    # response_schema declares (a) the JSON shape the LLM is told to return,
    # and (b) the JSONPath-ish lenses every UI uses to extract fields from
    # the parsed response. Editing this is the supported way to change the
    # response contract without touching code.
    "response_schema": {
        "shape": {
            "instruments": [
                {"class": "<one of vocab>", "confidence": 0.0, "evidence": "<short text>"}
            ],
            "anatomy": "<short text>",
            "tissue_condition": "<short text>",
        },
        "primary_class_path": "instruments[0].class",
        "evidence_path": "instruments[0].evidence",
        "confidence_path": "instruments[0].confidence",
        "aux_fields": [
            {"path": "anatomy", "label": "Anatomy"},
            {"path": "tissue_condition", "label": "Tissue condition"},
        ],
    },
    "prompt_template": (
        "Identify the surgical instrument by its visible visual features:\n\n"
        "PROBE vs SHAVER (most common confusion):\n"
        "- probe = thin, narrow solid metal rod, uniform diameter end-to-end, "
        "tip may have a tiny hook or blunt end; NO opening, NO rotating head\n"
        "- shaver = wider hollow tube with a distinct rectangular or oval "
        "side-opening (aspiration window) near the tip; shaft is noticeably "
        "thicker than a probe\n\n"
        "Other instruments:\n"
        "- burr = round/spherical or oval abrasive head at tip\n"
        "- grasper or biter = two metal jaws that open/close at a hinge\n"
        "- anchor_driver = ribbed or screw-threaded shaft\n"
        "- electrocautery = smooth wand with flat, angled, or hook-shaped distal tip\n"
        "- cannula = transparent or yellow plastic hollow tube\n\n"
        "Vocabulary: {{vocabulary}}.\n"
        "Respond with strict JSON matching this shape: {{response_schema}}"
    ),
}

_task_config_cache: Optional[Dict[str, object]] = None


def _read_task_config() -> Dict[str, object]:
    """Read task_config.json from the Volume. Returns the default config if
    the file doesn't exist yet (fresh deploy) — so Playground/Label always
    have *something* to render. The default is the same config the prod
    workspace has been using; existing workspaces are unaffected."""
    global _task_config_cache
    if _task_config_cache is not None:
        return _task_config_cache
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        with w.files.download(TASK_CONFIG_PATH).contents as f:
            data = json.loads(_drain_bytes(f).decode("utf-8"))
        # Backfill any missing top-level keys from the default (forward-compat
        # if a user saved a partial config before we added a field).
        for k, v in _DEFAULT_TASK_CONFIG.items():
            data.setdefault(k, v)
        _task_config_cache = data
        return data
    except Exception as e:
        log.info(f"[task_config] using built-in default ({e.__class__.__name__})")
        _task_config_cache = dict(_DEFAULT_TASK_CONFIG)
        return _task_config_cache


def _write_task_config(cfg: Dict[str, object]) -> None:
    """Atomically write task_config.json to the Volume. Uses tmp + rename so
    a partial write can never be observed by a concurrent reader."""
    from databricks.sdk import WorkspaceClient
    global _task_config_cache
    # Validate minimally — bad shape would silently break every tab
    if not isinstance(cfg.get("vocabulary"), list) or not cfg["vocabulary"]:
        raise ValueError("vocabulary must be a non-empty list")
    if not isinstance(cfg.get("prompt_template"), str) or not cfg["prompt_template"].strip():
        raise ValueError("prompt_template must be a non-empty string")
    w = WorkspaceClient()
    cfg_dir = TASK_CONFIG_PATH.rsplit("/", 1)[0]
    try:
        w.files.create_directory(cfg_dir)
    except Exception:
        pass  # already exists
    tmp_path = TASK_CONFIG_PATH + ".tmp"
    payload = json.dumps(cfg, indent=2).encode("utf-8")
    w.files.upload(tmp_path, payload, overwrite=True)
    # Volumes' "rename" is delete+upload; emulate atomicity by uploading the
    # final path after the tmp one has been fully written.
    w.files.upload(TASK_CONFIG_PATH, payload, overwrite=True)
    try:
        w.files.delete(tmp_path)
    except Exception:
        pass
    _task_config_cache = cfg


def _render_prompt_template(cfg: Dict[str, object]) -> str:
    """Render `prompt_template` with substitutions:
      - `{{vocabulary}}` → comma-joined vocab list
      - `{{response_schema}}` → JSON pretty-print of `response_schema.shape`
    Tiny custom substitution (not Jinja) so the template stays editable by hand."""
    tpl = str(cfg.get("prompt_template") or "")
    vocab = ", ".join(str(v) for v in (cfg.get("vocabulary") or []))
    shape = ((cfg.get("response_schema") or {}) if isinstance(cfg.get("response_schema"), dict) else {}).get("shape", {})
    shape_str = json.dumps(shape, separators=(", ", ": "))
    return tpl.replace("{{vocabulary}}", vocab).replace("{{response_schema}}", shape_str)


def _resolve_path(obj: object, path: str) -> object:
    """Resolve a JSONPath-ish lens like "instruments[0].class" against a
    parsed response. Returns None if any segment is missing rather than
    raising — the UI degrades gracefully when a model returns a partial
    response."""
    if obj is None or not path:
        return None
    cur: object = obj
    # Tokenize: split on '.' then expand '[N]' inside each segment
    import re as _re
    for seg in path.split("."):
        m = _re.match(r"^([^\[\]]*)((?:\[\d+\])*)$", seg)
        if not m:
            return None
        name, idxs = m.group(1), m.group(2)
        if name:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(name)
        for ix in _re.findall(r"\[(\d+)\]", idxs):
            if not isinstance(cur, list):
                return None
            i = int(ix)
            cur = cur[i] if 0 <= i < len(cur) else None
        if cur is None:
            return None
    return cur


def _drain_bytes(stream) -> bytes:
    """Read a streaming response into memory. Files SDK varies between
    sync iterables and file-like objects."""
    if hasattr(stream, "read"):
        return stream.read()
    return b"".join(stream)


def _default_rendered_prompt() -> str:
    """The prompt used to seed Playground when there's no localStorage draft.
    Reads from task_config so prompt edits propagate without a code change."""
    try:
        return _render_prompt_template(_read_task_config())
    except Exception:
        # Pre-DAB env var fallback — keeps existing prod deploys unchanged.
        return os.environ.get(
            "DEFAULT_PROMPT",
            "Identify the surgical instrument visible in this arthroscopy frame.",
        )


DEFAULT_PROMPT = _default_rendered_prompt()
STUDIO_PER_FRAME_MODEL = os.environ.get("STUDIO_PER_FRAME_MODEL", "databricks-claude-sonnet-4-5")
STUDIO_VALIDATOR_MODEL = os.environ.get("STUDIO_VALIDATOR_MODEL", "databricks-gpt-5-2")
STUDIO_AUDIO_ENDPOINT = os.environ.get("STUDIO_AUDIO_ENDPOINT", "whisper-transcription")
STUDIO_MAX_FRAMES = int(os.environ.get("STUDIO_MAX_FRAMES", "30"))
ALLOWED_PATH_PREFIXES = [
    os.path.normpath(EXTRACTED_FRAMES_DIR),
    os.path.normpath(EVAL_FRAMES_DIR),
    os.path.normpath(VOLUME_PATH),
]


# ── Lazy SDK clients ─────────────────────────────────────────────────────

_w = None


def _workspace():
    global _w
    if _w is None:
        from databricks.sdk import WorkspaceClient
        _w = WorkspaceClient()
    return _w


def _openai_client():
    """OpenAI-compatible client pointed at Databricks FMAPI.

    Mints a fresh M2M OAuth token from the app SP's credentials each call
    (via the cached helper) and constructs a new OpenAI client. We do NOT
    cache the OpenAI client globally — the previous implementation cached
    a 1-hour PAT inside the OpenAI instance, which started returning 403
    "Invalid access token" after expiry until the app was redeployed.
    The `_lakebase_token` helper handles the OAuth refresh with a 50-min
    in-memory cache, so this stays cheap."""
    from openai import OpenAI
    w = _workspace()
    token = _lakebase_token() or w.config.token
    if not token:
        raise RuntimeError("could not obtain a bearer token for FMAPI")
    return OpenAI(api_key=token, base_url=f"{w.config.host}/serving-endpoints")


# ── Lakebase Postgres connection (M2M OAuth → token-as-password) ─────────

# Databricks Apps' `valueFrom` env-var injection for database resources is
# inconsistent — some runtimes inject PG*, some don't. The robust pattern
# (matching production apps in this workspace) is to mint an OAuth token at
# request time using the SP's M2M creds (DATABRICKS_CLIENT_ID/SECRET/HOST,
# always injected) and use it as the Postgres password.
# Postgres connection params. The canonical Apps `postgres:` resource binding
# (added 2026) auto-injects PGHOST/PGDATABASE/PGUSER/PGPORT/PGSSLMODE — read
# those first. Fall back to the older LAKEBASE_* env vars so existing prod
# deploys (which still bind via the legacy `database:` Provisioned API) keep
# working without an immediate migration.
LAKEBASE_HOST = (
    os.environ.get("PGHOST")
    or os.environ.get("LAKEBASE_HOST")
    or "instance-6b59171b-cee8-4acc-9209-6c848ffbfbfe.database.cloud.databricks.com"
)
LAKEBASE_DBNAME = (
    os.environ.get("PGDATABASE")
    or os.environ.get("LAKEBASE_DBNAME")
    or "vlm_workbench"
)
LAKEBASE_PORT = int(os.environ.get("PGPORT") or os.environ.get("LAKEBASE_PORT") or 5432)
LAKEBASE_SSLMODE = os.environ.get("PGSSLMODE", "require")

_token_cache: Dict[str, object] = {"token": None, "expires_at": 0}


def _lakebase_token() -> Optional[str]:
    """Fetch (and cache) an OAuth token usable as the Postgres password."""
    now = time.time()
    if _token_cache["token"] and float(_token_cache["expires_at"]) > now + 60:
        return str(_token_cache["token"])
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if client_id and client_secret and host:
        host_no_scheme = host.replace("https://", "").replace("http://", "")
        try:
            import requests
            resp = requests.post(
                f"https://{host_no_scheme}/oidc/v1/token",
                data={"grant_type": "client_credentials", "scope": "all-apis"},
                auth=(client_id, client_secret),
                timeout=30,
            )
            if resp.status_code == 200:
                tok = resp.json().get("access_token", "")
                if tok:
                    _token_cache["token"] = tok
                    # Tokens are 1 hour; cache for 50 min to be safe
                    _token_cache["expires_at"] = now + 50 * 60
                    return tok
            log.warning(f"M2M OAuth failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            log.warning(f"M2M OAuth fetch failed: {e}")
    # Last-resort: workspace SDK token (works for local dev with PAT)
    try:
        w = _workspace()
        return w.config.token
    except Exception as e:
        log.warning(f"workspace token fetch failed: {e}")
        return None


def _pg_user() -> Optional[str]:
    # Lakebase Postgres role name. The Apps `postgres:` binding auto-injects
    # PGUSER = the SP's client_id (UUID). For PAT-based local dev, fall back
    # to the workspace user email (which the personal Lakebase role uses).
    return os.environ.get("PGUSER") or os.environ.get("DATABRICKS_CLIENT_ID") or _ws_user_email()


def _ws_user_email() -> Optional[str]:
    try:
        w = _workspace()
        me = w.current_user.me()
        return getattr(me, "user_name", None)
    except Exception:
        return None


def _pg_conn():
    import psycopg
    user = _pg_user()
    pw = _lakebase_token()
    if not (user and pw):
        raise RuntimeError("Lakebase auth unavailable (no DATABRICKS_CLIENT_ID/SECRET/HOST + no PAT)")
    return psycopg.connect(
        host=LAKEBASE_HOST, port=LAKEBASE_PORT, user=user, password=pw,
        dbname=LAKEBASE_DBNAME, sslmode="require", autocommit=True,
    )


def _lakebase_available() -> bool:
    """Lakebase is available iff we have credentials capable of producing a
    Postgres password (either the SP's M2M creds, a PGPASSWORD, or a PAT)."""
    return bool(
        os.environ.get("DATABRICKS_CLIENT_ID")
        or os.environ.get("PGPASSWORD")
        or os.environ.get("DATABRICKS_TOKEN")
    )


# ── Files API helpers (Volume-safe alternative to os.listdir / open) ─────

def _ls_dir(path: str) -> List[Dict[str, object]]:
    """List entries in a Volume directory using the Files API.
    Returns a list of {name, path, is_directory, size_bytes}."""
    w = _workspace()
    out: List[Dict[str, object]] = []
    try:
        for entry in w.files.list_directory_contents(path):
            out.append({
                "name": getattr(entry, "name", None),
                "path": getattr(entry, "path", None),
                "is_directory": bool(getattr(entry, "is_directory", False)),
                "size_bytes": int(getattr(entry, "file_size", 0) or 0),
            })
    except Exception as e:
        log.warning(f"_ls_dir({path}) failed: {e}")
    return out


def _download_bytes(path: str) -> bytes:
    """Read a file from a Volume via the Files API."""
    w = _workspace()
    resp = w.files.download(path)
    contents = resp.contents
    if hasattr(contents, "read"):
        return contents.read()
    if isinstance(contents, (bytes, bytearray)):
        return bytes(contents)
    return bytes(contents)


# ── Health ────────────────────────────────────────────────────────────────

# ── Global runs tracker ──────────────────────────────────────────────────
# Lists every workbench-submitted Databricks Job run from the last 24h so
# the React navbar pill can show in-flight + recent runs across all tabs
# (Library ingest, Setup cache, Optimize, Train) in one place.

# Map run_name prefix → human-readable kind shown in the UI.
# Every workbench-submitted run gets a `vlmwb_` prefix so the runs pill
# doesn't surface unrelated workflows in shared workspaces (e.g., other
# users' `ingest_confluence_*` jobs were polluting the list).
_RUN_KIND_PATTERNS = [
    ("vlmwb_ingest_",        "ingest"),
    ("vlmwb_setup_cache_",   "cache"),
    ("vlmwb_optimize_gepa_", "optimize"),
    ("vlmwb_optimize_dspy_", "optimize"),
    ("vlmwb_finetune_",      "train"),
    ("vlmwb_repin_",         "repin"),
    ("vlmwb_local_",         "local-inference"),
]


def _infer_run_kind(run_name: str) -> str:
    n = (run_name or "").lower()
    for prefix, kind in _RUN_KIND_PATTERNS:
        if n.startswith(prefix):
            return kind
    return "other"


@app.get("/api/runs/active")
def list_runs():
    """Recent workbench-submitted runs (active + finished in the last 24h).
    Filters by `run_name` prefix patterns so we don't surface unrelated
    runs from other workspace users. The SDK doesn't expose a
    `run_as_user_name` filter at this version, so name-prefix filtering
    is the only practical guard in a shared workspace."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    since = int((time.time() - 24 * 3600) * 1000)
    host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
    sp = os.environ.get("DATABRICKS_CLIENT_ID") or ""
    out: List[Dict[str, object]] = []
    try:
        # limit=25 is the Jobs API per-page cap; iterator paginates as needed.
        for r in w.jobs.list_runs(limit=25, start_time_from=since):
            name = r.run_name or ""
            kind = _infer_run_kind(name)
            if kind == "other":
                continue
            # Originally I added a "creator must be this SP" guard here, but
            # `run_as_user_name` for an Apps SP is its display name (e.g.
            # `app-40zbx9 vlm-workbench`), not the UUID — so checking for the
            # client_id substring excluded every legitimate run. The
            # `run_name`-prefix filter (above) already prevents leakage of
            # unrelated jobs in shared workspaces; rely on that alone.
            s = r.state
            life = (s.life_cycle_state.value if s and s.life_cycle_state else "") or ""
            res = (s.result_state.value if s and s.result_state else "") or ""
            out.append({
                "run_id": r.run_id,
                "name": name,
                "kind": kind,
                "life_cycle_state": life,
                "result_state": res,
                "state_message": (s.state_message if s else None),
                "start_time": r.start_time,
                "end_time": r.end_time,
                "run_url": f"{host}/#job/runs/{r.run_id}" if host else None,
            })
    except Exception as e:
        log.warning(f"list_runs failed: {e}")
    return {"runs": out, "fetched_at": int(time.time() * 1000)}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "uc_catalog": UC_CATALOG_BOOT,
        "uc_schema": UC_SCHEMA_BOOT,
        "volume_name": VOLUME_NAME_BOOT,
        "volume_path": VOLUME_PATH,
        "extracted_frames_dir": EXTRACTED_FRAMES_DIR,
        "eval_frames_dir": EVAL_FRAMES_DIR,
    }


# ── Task configuration endpoints ─────────────────────────────────────────

@app.get("/api/task-config")
def get_task_config():
    """Return the active task config plus its derived `rendered_prompt`.
    Falls back to the built-in default if no config has been saved yet."""
    cfg = _read_task_config()
    return {**cfg, "rendered_prompt": _render_prompt_template(cfg)}


class TaskConfigIn(BaseModel):
    vocabulary: List[str]
    response_schema: Optional[Dict[str, object]] = None
    prompt_template: str


class HFModelSpec(BaseModel):
    name: str            # slug used for the directory + manifest
    hf_repo: str         # Hugging Face repo id (e.g. "Qwen/Qwen3-VL-8B-Instruct")
    revision: Optional[str] = "main"
    # Optional manifest extras that flow through setup_cache → manifest.yaml
    # → the local-run submitter. Used to pin bigger/different models to the
    # right GPU + inference notebook without forking the whole flow.
    accelerator: Optional[str] = None        # e.g. "GPU_1xH100"
    base_environment: Optional[str] = None   # e.g. "databricks_ai_v4"
    inference_notebook: Optional[str] = None  # e.g. "run_gemma4"


class HFTokenIn(BaseModel):
    token: str
    models: Optional[List[HFModelSpec]] = None  # if present, kick off setup_cache job


# Preset model catalogue surfaced to the Setup tab so the UI doesn't need
# to hardcode HF repo paths. Add new entries here as more local-served
# models become useful.
LOCAL_MODEL_PRESETS: List[Dict[str, str]] = [
    {"name": "qwen3-vl-8b",   "hf_repo": "Qwen/Qwen3-VL-8B-Instruct", "label": "Qwen3-VL-8B (Apache-2.0)"},
    {"name": "medgemma-4b-it","hf_repo": "google/medgemma-4b-it",     "label": "MedGemma-4B-it (gated)"},
    # Gemma 4 family — needs transformers >= 5.5.0 (the `run_gemma4`
    # variant pins it). Three sizes available for the workbench:
    #
    #   • E4B (MatFormer effective ~4 B, ~16 GB raw weights at bf16):
    #     fits A10 in bf16. Smallest/fastest.
    #   • 12B at bf16 (~24 GB weights): won't fit A10's 24 GB total VRAM
    #     once activations + KV cache are accounted for — pinned to
    #     1×H100. Best raw quality.
    #   • 12B int4 (Google's official QAT w4a16 variant, ~6 GB weights):
    #     fits A10 with headroom. Slight accuracy loss vs bf16, much
    #     cheaper to serve. Requires `compressed-tensors` in the
    #     notebook env (added to run_gemma4's pip install).
    {"name": "gemma-4-e4b-it", "hf_repo": "google/gemma-4-E4B-it",
     "label": "Gemma 4 E4B-it · ~4B effective, fits A10 (Apache-2.0)",
     "accelerator": "GPU_1xA10", "inference_notebook": "run_gemma4"},
    {"name": "gemma-4-12b-it", "hf_repo": "google/gemma-4-12B-it",
     "label": "Gemma 4 12B-it · bf16, needs H100 (Apache-2.0)",
     "accelerator": "GPU_1xH100", "inference_notebook": "run_gemma4"},
    {"name": "gemma-4-12b-it-int4", "hf_repo": "google/gemma-4-12B-it-qat-w4a16-ct",
     "label": "Gemma 4 12B-it int4 · QAT w4a16, fits A10 (Apache-2.0)",
     "accelerator": "GPU_1xA10", "inference_notebook": "run_gemma4"},
]


@app.get("/api/setup/local-model-presets")
def get_local_model_presets():
    """Catalogue of model presets the Setup tab offers to cache. Each entry
    has (name, hf_repo, label). The HF token paste dialog uses these to
    render checkboxes."""
    return {"presets": LOCAL_MODEL_PRESETS}


@app.post("/api/setup/hf-token")
def store_hf_token(req: HFTokenIn):
    """Write the user's HF token into the configured scope/key, creating
    the scope if it doesn't exist yet (App SP owns the scope it creates,
    so subsequent put_secret succeeds). If `models` is non-empty, also
    kick off the setup_cache job to download those weights to the Volume.

    Returns {ok, scope, key, len, models_run_id?, models_run_url?}."""
    tok = (req.token or "").strip()
    if not tok:
        raise HTTPException(400, "token is empty")
    if not (tok.startswith("hf_") or tok.startswith("api_")):
        raise HTTPException(400, "token doesn't look like an HF token (expected `hf_...`)")
    scope = HF_SECRET_SCOPE
    key = HF_SECRET_KEY
    if not scope or not key:
        raise HTTPException(503, "HF scope/key not configured — set hf_secret_scope and hf_secret_key in variable-overrides.json")

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    # Ensure scope exists. If the deployer pre-owns it, we just put. If it
    # doesn't exist yet, App SP creates it (and becomes owner — so the
    # subsequent put + grant succeed without admin help).
    try:
        existing = {s.name for s in w.secrets.list_scopes()}
    except Exception as e:
        raise HTTPException(500, f"could not list secret scopes: {e}")
    if scope not in existing:
        try:
            w.secrets.create_scope(scope=scope)
        except Exception as e:
            raise HTTPException(500, f"could not create scope '{scope}': {e}")

    try:
        w.secrets.put_secret(scope=scope, key=key, string_value=tok)
    except Exception as e:
        msg = str(e)
        if "403" in msg or "PERMISSION" in msg.upper():
            raise HTTPException(403,
                f"App SP can't write to scope '{scope}'. Either grant the SP WRITE "
                f"on the scope, or point hf_secret_scope in variable-overrides.json "
                f"at a scope you own. Underlying error: {e}")
        log.exception("hf token put failed")
        raise HTTPException(500, f"put failed: {e}")

    out: Dict[str, object] = {"ok": True, "scope": scope, "key": key, "len": len(tok)}

    # Optional: kick off setup_cache job for the selected models
    if req.models:
        try:
            notebook_path = _resolve_setup_cache_notebook()
            models_json = json.dumps([m.dict() for m in req.models])
            body = {
                "run_name": f"vlmwb_setup_cache_{int(time.time())}",
                "tasks": [{
                    "task_key": "cache",
                    "notebook_task": {
                        "notebook_path": notebook_path,
                        "source": "WORKSPACE",
                        "base_parameters": {
                            "local_models_dir": LOCAL_MODELS_DIR,
                            "hf_secret_scope": scope,
                            "hf_secret_key": key,
                            "models_json": models_json,
                        },
                    },
                    "environment_key": "cpu_env",
                }],
                "environments": [{"environment_key": "cpu_env", "spec": {"environment_version": "5"}}],
                "queue": {"enabled": True},
            }
            host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
            run = w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body=body)
            run_id = run.get("run_id") if isinstance(run, dict) else None
            out["models_run_id"] = run_id
            out["models_run_url"] = f"{host}/#job/runs/{run_id}" if run_id and host else None
            out["models"] = [m.dict() for m in req.models]
        except Exception as e:
            log.exception("setup_cache submit failed")
            out["models_error"] = f"{type(e).__name__}: {e}"

    return out


@app.put("/api/task-config")
def put_task_config(req: TaskConfigIn):
    cfg: Dict[str, object] = {
        "vocabulary": [v.strip() for v in req.vocabulary if v.strip()],
        "response_schema": req.response_schema or _DEFAULT_TASK_CONFIG["response_schema"],
        "prompt_template": req.prompt_template,
    }
    try:
        _write_task_config(cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("write task_config failed")
        raise HTTPException(500, f"write failed: {e}")
    return {**cfg, "rendered_prompt": _render_prompt_template(cfg)}


@app.get("/api/debug/lakebase")
def debug_lakebase():
    """Comprehensive Lakebase Postgres diagnostic for the App SP. Each step
    runs independently so a downstream failure doesn't mask an upstream one.
    Returns the full picture: env, token mint, TCP connect, role identity,
    schema visibility, and DDL capability."""
    out: Dict[str, object] = {
        "env": {
            "LAKEBASE_HOST": LAKEBASE_HOST,
            "LAKEBASE_PORT": LAKEBASE_PORT,
            "LAKEBASE_DBNAME": LAKEBASE_DBNAME,
            "DATABRICKS_CLIENT_ID_present": bool(os.environ.get("DATABRICKS_CLIENT_ID")),
            "DATABRICKS_CLIENT_ID_first8": (os.environ.get("DATABRICKS_CLIENT_ID") or "")[:8],
            "DATABRICKS_CLIENT_SECRET_present": bool(os.environ.get("DATABRICKS_CLIENT_SECRET")),
            "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST"),
        },
        "steps": [],
    }
    def step(name: str, fn):
        try:
            out["steps"].append({"name": name, "ok": True, "result": fn()})
        except Exception as e:
            out["steps"].append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:400]})

    # 1. mint token
    tok_holder: Dict[str, object] = {}
    def _mint():
        t = _lakebase_token()
        tok_holder["token"] = t
        return {"len": len(t or ""), "first20": (t or "")[:20]}
    step("mint_oauth_token", _mint)

    # 2. computed user
    step("compute_pg_user", lambda: {"user": _pg_user()})

    # 3. raw TCP connect (no auth) — distinguishes network vs auth failures
    def _tcp():
        import socket
        s = socket.create_connection((LAKEBASE_HOST, LAKEBASE_PORT), timeout=10)
        s.close()
        return "tcp_open"
    step("tcp_connect", _tcp)

    # 4. authenticated connect + current identity
    def _connect_and_select():
        import psycopg
        with psycopg.connect(
            host=LAKEBASE_HOST, port=LAKEBASE_PORT,
            user=_pg_user(), password=tok_holder.get("token"),
            dbname=LAKEBASE_DBNAME, sslmode="require", autocommit=True,
            connect_timeout=10,
        ) as conn, conn.cursor() as cur:
            cur.execute("SELECT current_user, current_database(), version()")
            row = cur.fetchone()
            return {"current_user": row[0], "current_database": row[1], "version": row[2][:80]}
    step("connect_select", _connect_and_select)

    # 5. schema visibility — what schemas can the role see?
    def _list_schemas():
        import psycopg
        with psycopg.connect(
            host=LAKEBASE_HOST, port=LAKEBASE_PORT,
            user=_pg_user(), password=tok_holder.get("token"),
            dbname=LAKEBASE_DBNAME, sslmode="require", autocommit=True,
            connect_timeout=10,
        ) as conn, conn.cursor() as cur:
            cur.execute("SELECT schema_name FROM information_schema.schemata ORDER BY 1")
            return [r[0] for r in cur.fetchall()]
    step("list_schemas", _list_schemas)

    # 6. CREATE TABLE — the real failure mode for _ensure_ingest_tables()
    def _create_tmp_table():
        import psycopg
        with psycopg.connect(
            host=LAKEBASE_HOST, port=LAKEBASE_PORT,
            user=_pg_user(), password=tok_holder.get("token"),
            dbname=LAKEBASE_DBNAME, sslmode="require", autocommit=True,
            connect_timeout=10,
        ) as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS vlmwb_debug_probe (id INT, note TEXT)")
            cur.execute("INSERT INTO vlmwb_debug_probe VALUES (1, 'ok') ON CONFLICT DO NOTHING")
            cur.execute("SELECT count(*) FROM vlmwb_debug_probe")
            n = cur.fetchone()[0]
            cur.execute("DROP TABLE vlmwb_debug_probe")
            return {"rows_inserted_and_dropped": n}
    step("ddl_probe", _create_tmp_table)

    return out


@app.get("/api/debug/mlflow")
def debug_mlflow():
    """Surface mlflow + env var diagnostics so we can debug the GenAI auth chain."""
    out: Dict[str, object] = {
        "MLFLOW_ENABLE_DB_SDK": os.environ.get("MLFLOW_ENABLE_DB_SDK"),
        "DATABRICKS_CLIENT_ID_present": bool(os.environ.get("DATABRICKS_CLIENT_ID")),
        "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST"),
    }
    with _mlflow_auth_scope():
        out["DATABRICKS_TOKEN_present"] = bool(os.environ.get("DATABRICKS_TOKEN"))
        out["DATABRICKS_CLIENT_ID_in_scope"] = bool(os.environ.get("DATABRICKS_CLIENT_ID"))
        try:
            import mlflow
            out["mlflow_version"] = mlflow.__version__
            out["has_genai"] = hasattr(mlflow, "genai")
            try:
                mlflow.set_tracking_uri("databricks")
                mlflow.set_registry_uri("databricks-uc")
                mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
                out["tracking_uri"] = mlflow.get_tracking_uri()
                out["registry_uri"] = mlflow.get_registry_uri()
                out["experiment_path"] = MLFLOW_EXPERIMENT_PATH
                p = mlflow.genai.register_prompt(
                    name=GENAI_PROMPT_NAME,
                    template="DEBUG probe " + str(time.time()),
                    commit_message="debug",
                )
                out["register_ok"] = {"version": p.version, "uri": p.uri}
            except Exception as e:
                import traceback
                out["register_err"] = str(e)[:500]
                out["register_trace"] = traceback.format_exc()[-1500:]
        except Exception as e:
            out["import_err"] = str(e)
    return out


# ── Setup preflight ─────────────────────────────────────────────────────
#
# Runs every check the app needs to function end-to-end. Used by the Setup
# tab so a new deploy can see what's working / broken in one place, with
# remediations inline.

class SetupCheck(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    remediation: Optional[str] = None
    docs_url: Optional[str] = None


@app.get("/api/setup/check")
def setup_check():
    checks: List[SetupCheck] = []

    # 1. Workspace SDK / M2M OAuth
    try:
        w = _workspace()
        host = w.config.host
        checks.append(SetupCheck(
            name="Workspace credentials",
            ok=True,
            detail=f"connected to {host}",
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="Workspace credentials",
            ok=False,
            detail=str(e)[:200],
            remediation="The app's service principal needs DATABRICKS_CLIENT_ID/SECRET injected via the App's resource bindings. Re-deploy via the bundle.",
        ))

    # 2. Lakebase reachable + labels table present
    try:
        if not _lakebase_available():
            raise RuntimeError("DATABRICKS_CLIENT_ID/SECRET not set — app SP can't mint a Postgres token")
        _ensure_labels_table()
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM frame_labels")
            n_labels = cur.fetchone()[0]
        checks.append(SetupCheck(
            name="Lakebase Postgres",
            ok=True,
            detail=f"connected · frame_labels rows: {n_labels}",
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="Lakebase Postgres",
            ok=False,
            detail=str(e)[:240],
            remediation=(
                "1. Confirm the App is bound to a Lakebase database in app.yaml. "
                "2. Run the postdeploy job to create a Postgres role for the App SP "
                f"({os.environ.get('DATABRICKS_CLIENT_ID','<sp-id>')})."
            ),
        ))

    # 3. SQL warehouse — used for Delta sync. PENDING is a benign warming
    # state (a cold warehouse needs ~30-60s to spin up); only treat actual
    # FAILED/CANCELED states as a real failure.
    try:
        wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
        if not wh_id:
            raise RuntimeError("DATABRICKS_WAREHOUSE_ID not set")
        w = _workspace()
        resp = w.api_client.do(
            "POST", "/api/2.0/sql/statements",
            body={"warehouse_id": wh_id, "statement": "SELECT 1", "wait_timeout": "10s"},
        )
        state = (resp.get("status") or {}).get("state")
        if state in ("SUCCEEDED", "PENDING", "RUNNING"):
            detail = (f"warehouse {wh_id} ready"
                      if state == "SUCCEEDED" else f"warehouse {wh_id} warming ({state})")
            checks.append(SetupCheck(name="SQL warehouse", ok=True, detail=detail))
        else:
            raise RuntimeError(f"SELECT 1 returned state={state}")
    except Exception as e:
        checks.append(SetupCheck(
            name="SQL warehouse",
            ok=False,
            detail=str(e)[:240],
            remediation="Bind a SQL warehouse via app.yaml `resources.sql_warehouse.permission: CAN_USE`. The bundle does this automatically.",
        ))

    # 4. UC volume — verify the data volume exists and is readable
    try:
        items = _ls_dir(VOLUME_PATH)
        n = sum(1 for _ in items)
        checks.append(SetupCheck(
            name="UC data volume",
            ok=True,
            detail=f"{VOLUME_PATH} · {n} items",
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="UC data volume",
            ok=False,
            detail=f"{VOLUME_PATH}: {str(e)[:200]}",
            remediation=(
                "1. Run `bundle deploy` — the bundle creates the managed volume. "
                "2. Confirm the SP has READ/WRITE VOLUME (the postdeploy job grants this)."
            ),
        ))

    # 5. Local models cache
    try:
        local_models = list_local_models()
        ready = [m for m in local_models if m.ready]
        all_count = len(local_models)
        checks.append(SetupCheck(
            name="Local model cache",
            ok=len(ready) > 0,
            detail=f"{len(ready)} ready of {all_count} (in {LOCAL_MODELS_DIR})",
            remediation=(
                "Open the HuggingFace token check below → click 'Set token in-app' → "
                "paste your HF token and check the models you want to cache. The "
                "setup_cache job downloads weights from HF into the Volume; ~5–30 min "
                "depending on model size and network."
            ) if len(ready) == 0 else None,
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="Local model cache",
            ok=False,
            detail=str(e)[:200],
        ))

    # 6. HF token secret — check the configured scope (HF_SECRET_SCOPE), not
    # whichever happens to be alphabetically-first with "hf" in the name.
    try:
        w = _workspace()
        scopes = [s.name for s in w.secrets.list_scopes()]
        target_scope = HF_SECRET_SCOPE or ""
        target_key = HF_SECRET_KEY or ""
        if not target_scope:
            raise RuntimeError("HF_SECRET_SCOPE not set — check resources/app.yml binding")
        if target_scope not in scopes:
            raise RuntimeError(
                f"configured HF scope `{target_scope}` does not exist. Click "
                f"'Set token in-app' below to create it + paste a token."
            )
        # Check whether the token secret actually exists (not just the scope)
        existing_keys = [k.key for k in w.secrets.list_secrets(scope=target_scope)]
        if target_key not in existing_keys:
            raise RuntimeError(
                f"scope `{target_scope}` exists but key `{target_key}` is missing. "
                f"Click 'Set token in-app' below to paste a token."
            )
        # Probe ACL — we want to know if SP has READ
        sp = os.environ.get("DATABRICKS_CLIENT_ID")
        detail = f"scope `{target_scope}` · key `{target_key}` present"
        if sp:
            try:
                acls = w.secrets.list_acls(scope=target_scope)
                sp_acl = next((a for a in acls if a.principal == sp), None)
                if sp_acl:
                    detail = f"scope `{target_scope}` · key `{target_key}` · SP has {sp_acl.permission}"
            except Exception:
                pass  # MANAGE needed for list_acls; existence-only is fine
        checks.append(SetupCheck(
            name="HuggingFace token secret",
            ok=True,
            detail=detail,
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="HuggingFace token secret",
            ok=False,
            detail=str(e)[:200],
            remediation=(
                "1. `databricks secrets create-scope hls_g4` (or whatever scope name) "
                "2. `databricks secrets put-secret hls_g4 HF_TOKEN` (paste your HF token) "
                "3. `databricks secrets put-acl hls_g4 <SP_ID> READ`."
            ),
        ))

    # 7. Bundled notebooks — actively import each into the SP home (idempotent).
    # That's what real Playground/Train/Optimize calls do anyway, and it's the
    # only way to verify the source files are actually present in the deploy.
    notebook_resolvers = [
        ("optimizer/gepa", lambda: _resolve_optimizer_notebook("gepa")),
        ("optimizer/dspy", lambda: _resolve_optimizer_notebook("dspy")),
        ("local_inference/run", _resolve_local_inference_notebook),
        ("local_inference/run_gemma4", lambda: _resolve_local_inference_notebook_variant("run_gemma4")),
        ("local_inference/finetune", _resolve_finetune_notebook),
        ("ingest/smart_frames", _resolve_ingest_notebook),
    ]
    missing: List[str] = []
    for name, resolver in notebook_resolvers:
        try:
            resolver()
        except Exception as e:
            missing.append(f"{name}: {str(e)[:80]}")
    if missing:
        checks.append(SetupCheck(
            name="Bundled notebooks",
            ok=False,
            detail=f"{len(missing)} of {len(notebook_resolvers)} fail to resolve",
            remediation=(
                "Each bundled notebook should be present in the App's deployed source tree. "
                "Run `databricks apps deploy` from a fresh build; the bundle's sync.paths "
                "copies everything under the source root.\n\nFailures:\n  - "
                + "\n  - ".join(missing)
            ),
        ))
    else:
        checks.append(SetupCheck(
            name="Bundled notebooks",
            ok=True,
            detail=f"{len(notebook_resolvers)}/{len(notebook_resolvers)} resolve cleanly",
        ))

    # 8. MLflow experiment (workspace-level run tracking, separate from UC prompts).
    # Default is under the SP's own home (/Users/<sp_uuid>/vlmwb-experiments)
    # which the SP owns implicitly. If the operator overrode MLFLOW_EXPERIMENT_PATH
    # to a different folder, the SP needs CAN_MANAGE on it. We exercise the full
    # flow (create-or-get + start_run + log_param + end_run) to catch every gate.
    try:
        with _mlflow_auth_scope():
            import mlflow
            mlflow.set_tracking_uri("databricks")
            exp = mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
            exp_id = getattr(exp, "experiment_id", None)
            # Quick write probe — start_run/log_param/end_run all hit the
            # tracking server with the SP's identity.
            with mlflow.start_run(experiment_id=exp_id, run_name="vlmwb_setup_probe", nested=False) as run:
                mlflow.log_param("probe", "ok")
            detail = f"experiment `{MLFLOW_EXPERIMENT_PATH}` (id={exp_id}) writable"
        checks.append(SetupCheck(
            name="MLflow experiment",
            ok=True,
            detail=detail,
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="MLflow experiment",
            ok=False,
            detail=str(e)[:200],
            remediation=(
                f"App SP needs CAN_MANAGE on the experiment folder `{MLFLOW_EXPERIMENT_PATH}`. "
                f"By default we put the experiment under the SP's own home "
                f"(/Users/<sp_client_id>/vlmwb-experiments) which the SP owns — no action "
                f"needed there. If MLFLOW_EXPERIMENT_PATH was overridden to a folder under "
                f"a user's home, that user must:\n"
                f"  1. Create the folder if missing (Workspace UI → New → Folder)\n"
                f"  2. Share with the App SP (Workspace UI → ⋯ → Permissions → "
                f"Add `{os.environ.get('DATABRICKS_CLIENT_ID', '<sp_client_id>')}` with "
                f"`Can Manage`)"
            ),
        ))

    # 9. MLflow GenAI auth path (UC-managed prompts/datasets)
    try:
        with _mlflow_auth_scope():
            import mlflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
            # Try a no-op load on the canonical prompt name — if perm denied,
            # we'll learn here
            try:
                mlflow.genai.load_prompt(f"prompts:/{GENAI_PROMPT_NAME}/1")
                detail = f"prompt `{GENAI_PROMPT_NAME}` v1 readable"
            except Exception:
                # Try the actual permission check by attempting to register a
                # throwaway version we won't keep — but only if the schema name
                # matches the configured one (otherwise we're just polluting
                # the user's schema)
                try:
                    p = mlflow.genai.register_prompt(
                        name=GENAI_PROMPT_NAME,
                        template="PREFLIGHT probe — safe to delete",
                        commit_message="preflight check",
                    )
                    detail = f"register works (v{p.version})"
                except Exception as ex_inner:
                    if "permission" in str(ex_inner).lower():
                        raise RuntimeError(
                            "App SP can register prompts via MLflow but UC blocks it: "
                            f"{str(ex_inner)[:200]}"
                        )
                    raise
        checks.append(SetupCheck(
            name="MLflow GenAI prompts (UC)",
            ok=True,
            detail=detail,
        ))
    except Exception as e:
        checks.append(SetupCheck(
            name="MLflow GenAI prompts (UC)",
            ok=False,
            detail=str(e)[:240],
            remediation=(
                "UC-managed MLflow prompts have a permission gate that ALL PRIVILEGES "
                "on the schema doesn't cover. Add the App SP as a schema co-owner via "
                "the UC UI (Catalog Explorer → Permissions → grant CAN_MANAGE), or run "
                "prompt registration from a notebook executing as a user with schema "
                "ownership."
            ),
        ))

    n_ok = sum(1 for c in checks if c.ok)
    return {
        "checks": [c.model_dump() for c in checks],
        "n_ok": n_ok,
        "n_total": len(checks),
        "ready": n_ok == len(checks),
    }


@app.get("/api/debug/env")
def debug_env():
    """List which Lakebase-related env vars exist (presence only, not values)."""
    keys_of_interest = [
        "PGHOST", "PGUSER", "PGPASSWORD", "PGDATABASE", "PGPORT",
        "DATABRICKS_DATABASE_HOST", "DATABRICKS_DATABASE_USER",
        "DATABRICKS_DATABASE_PASSWORD", "DATABRICKS_DATABASE_NAME",
        "DATABRICKS_DATABASE_PORT", "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET", "DATABRICKS_HOST",
    ]
    return {
        "present": {k: bool(os.environ.get(k)) for k in keys_of_interest},
        "lakebase_available": _lakebase_available(),
        # Show all env keys that contain DB-ish patterns (just the names)
        "all_db_keys": sorted([
            k for k in os.environ.keys()
            if any(t in k.upper() for t in ("PG", "DATABASE", "POSTGRES", "LAKEBASE"))
        ]),
    }


# ── Videos ────────────────────────────────────────────────────────────────

class VideoEntry(BaseModel):
    name: str
    path: str
    size_bytes: int


@app.get("/api/videos", response_model=List[VideoEntry])
def list_videos():
    """List videos that have been ingested into Lakebase.

    Lakebase's `videos` table is the single source of truth — it's
    populated by the Library tab's ingest pipeline and contains only
    videos that have been (or are being) frame-extracted. Studio,
    Playground, and every other consumer should read from here so
    the views stay consistent across tabs. Image batches (`kind='image_batch'`)
    are excluded — Studio is video-only."""
    out: List[VideoEntry] = []
    if not _lakebase_available():
        return out
    try:
        _ensure_ingest_tables()
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT name, path, COALESCE(size_bytes, 0)
                FROM videos
                WHERE kind = 'video' AND status = 'ready'
                ORDER BY name
            """)
            for name, path, size_bytes in cur.fetchall():
                out.append(VideoEntry(name=name, path=path, size_bytes=int(size_bytes)))
    except Exception as e:
        log.warning(f"list_videos from Lakebase failed: {e}")
    return out


# ── Frames ────────────────────────────────────────────────────────────────

class FrameEntry(BaseModel):
    name: str
    path: str
    timestamp_s: Optional[float] = None
    video: Optional[str] = None


def _parse_timestamp(name: str) -> Optional[float]:
    """Filename pattern: <video>_frame_<seconds>s.jpg → returns seconds."""
    m = re.search(r"_frame_(\d+(?:\.\d+)?)s", name)
    return float(m.group(1)) if m else None


@app.get("/api/frames", response_model=List[FrameEntry])
def list_frames(video: Optional[str] = None, source: str = "eval"):
    """List frames available to the Playground.

    `source=eval`      → curated 31-frame eval set (flat dir)
    `source=extracted` → smart-extracted frames (per-video subfolders)
    `source=labeled`   → every frame in the ground-truth labels table
    """
    out: List[FrameEntry] = []

    if source == "labeled":
        if not _lakebase_available():
            return out
        _ensure_labels_table()
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT frame_path FROM frame_labels ORDER BY updated_at DESC"
            )
            paths = [r[0] for r in cur.fetchall()]
        for p in paths:
            name = os.path.basename(p)
            # Try to recover video attribution from the path or filename.
            v: Optional[str] = None
            if EXTRACTED_FRAMES_DIR in p:
                # /Volumes/.../extracted_frames/<video>/<file>
                rel = p[len(EXTRACTED_FRAMES_DIR):].lstrip("/")
                head = rel.split("/", 1)[0]
                if head:
                    v = head
            if v is None and "_frame_" in name:
                v = name.split("_frame_")[0]
            out.append(FrameEntry(
                name=name,
                path=p,
                timestamp_s=_parse_timestamp(name),
                video=v,
            ))
        if video:
            out = [f for f in out if (f.video or "") == video]
        return out

    if source == "eval":
        for entry in _ls_dir(EVAL_FRAMES_DIR):
            name = entry["name"]
            if not name or entry["is_directory"]:
                continue
            if not name.lower().endswith(".jpg"):
                continue
            out.append(FrameEntry(
                name=name,
                path=str(entry["path"] or os.path.join(EVAL_FRAMES_DIR, name)),
                timestamp_s=_parse_timestamp(name),
                video=name.split("_frame_")[0] if "_frame_" in name else None,
            ))
        out.sort(key=lambda f: f.name)
        return out

    # source == "extracted" — has per-video subfolders.
    # Prefer the Lakebase index when populated: it's instant (single SQL query)
    # vs. walking N subdirectories via the Files API for every Playground load.
    # Falls back to the Files-API path scan if the table is empty/unavailable.
    if _lakebase_available():
        try:
            _ensure_ingest_tables()
            with _pg_conn() as conn, conn.cursor() as cur:
                if video:
                    cur.execute("""
                        SELECT efi.frame_path, efi.frame_name, efi.timestamp_s, v.name
                        FROM extracted_frames_index efi
                        JOIN videos v ON v.id = efi.video_id
                        WHERE v.name = %s OR v.name = %s
                        ORDER BY efi.timestamp_s
                    """, (video, video.replace(".mp4", "") if video else ""))
                else:
                    cur.execute("""
                        SELECT efi.frame_path, efi.frame_name, efi.timestamp_s, v.name
                        FROM extracted_frames_index efi
                        JOIN videos v ON v.id = efi.video_id
                        ORDER BY v.name, efi.timestamp_s
                    """)
                rows = cur.fetchall()
            if rows:
                for r in rows:
                    out.append(FrameEntry(
                        name=r[1],
                        path=r[0],
                        timestamp_s=r[2],
                        video=(r[3] or "").replace(".mp4", "") or r[3],
                    ))
                return out
        except Exception as e:
            log.warning(f"[frames] index lookup failed, falling back to scan: {e}")

    for top in _ls_dir(EXTRACTED_FRAMES_DIR):
        if not top["is_directory"]:
            continue
        video_dir_name = top["name"]
        if not video_dir_name:
            continue
        if video and video_dir_name != video and video_dir_name != video.replace(".mp4", ""):
            continue
        video_dir_path = str(top["path"] or os.path.join(EXTRACTED_FRAMES_DIR, video_dir_name))
        for entry in _ls_dir(video_dir_path):
            name = entry["name"]
            if not name or entry["is_directory"]:
                continue
            if not name.lower().endswith(".jpg"):
                continue
            out.append(FrameEntry(
                name=name,
                path=str(entry["path"] or os.path.join(video_dir_path, name)),
                timestamp_s=_parse_timestamp(name),
                video=video_dir_name,
            ))
    out.sort(key=lambda f: (f.video or "", f.name))
    return out


def _check_allowed_path(path: str) -> str:
    norm = os.path.normpath(path)
    if not any(norm.startswith(p) for p in ALLOWED_PATH_PREFIXES):
        raise HTTPException(403, f"Path not in allowed directories")
    return norm


@app.get("/api/frames/image")
def get_frame_image(path: str):
    """Stream a single frame JPEG via the Files API."""
    abs_path = _check_allowed_path(path)
    try:
        data = _download_bytes(abs_path)
    except Exception as e:
        raise HTTPException(404, f"Frame not found or unreadable: {e}")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=300"},
    )


# ── Models (auto-enumerate vision-capable FMAPI endpoints) ───────────────

class ModelEntry(BaseModel):
    name: str
    task: str
    ready: bool
    vision: Optional[bool] = None


_models_cache: Dict[str, object] = {"data": None, "ts": 0}


@app.get("/api/models", response_model=List[ModelEntry])
def list_models(refresh: bool = False):
    if not refresh and _models_cache["data"] is not None and (time.time() - float(_models_cache["ts"]) < 300):
        return _models_cache["data"]

    w = _workspace()
    resp = w.api_client.do("GET", "/api/2.0/serving-endpoints")
    out: List[ModelEntry] = []
    for e in resp.get("endpoints", []):
        name = e.get("name", "")
        if not name.startswith("databricks-"):
            continue
        task = e.get("task") or ""
        if task != "llm/v1/chat":
            continue
        ready = e.get("state", {}).get("ready") == "READY"
        # Prefer the explicit capability flag from endpoint metadata. The
        # endpoint authoritatively knows; substring matching on the name
        # mis-classifies new family members (e.g. `databricks-claude-fable-5`
        # is text-only — sending it `[{type: image}, {type: text}]` content
        # crashes the gateway's validator with "'list' object has no
        # attribute 'strip'"). Only fall back to substring matching when
        # the endpoint didn't report capabilities at all.
        caps = e.get("capabilities") or {}
        if "image_input" in caps:
            vision = bool(caps["image_input"])
        else:
            vision = any(tag in name for tag in [
                "claude-opus", "claude-sonnet", "claude-haiku",
                "gemini", "gpt-5", "gemma-3", "llama-4-maverick",
            ])
        out.append(ModelEntry(name=name, task=task, ready=ready, vision=vision))

    out.sort(key=lambda m: ((not m.vision), (not m.ready), m.name))
    _models_cache["data"] = out
    _models_cache["ts"] = time.time()
    return out


# ── Playground run (SSE) ─────────────────────────────────────────────────

class PlaygroundRequest(BaseModel):
    frame_paths: List[str] = Field(..., description="Volume paths to frame JPEGs")
    model_names: List[str] = Field(..., description="FMAPI endpoint names to query")
    prompt: str = DEFAULT_PROMPT
    # Default budget bumped to accommodate reasoning models (e.g.
    # `databricks-claude-fable-5`), which spend most of their tokens on
    # an internal (private) thinking trace before emitting the answer.
    # At 600 tokens, fable consumed all of them on reasoning and returned
    # no text block — empirically 2000 leaves enough room for the answer.
    max_tokens: int = 2000
    temperature: float = 0.0


def _load_b64(path: str, max_side: int = 1024, quality: int = 85) -> str:
    data = _download_bytes(path)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_strict_json(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if t.startswith("```"):
        try:
            t = t.split("```", 2)[1]
            if t.startswith("json"):
                t = t[4:]
            t = t.strip().rstrip("`").strip()
        except Exception:
            pass
    try:
        return json.loads(t)
    except Exception:
        return None


def _call_responses_api(client, model_name: str, frame_b64: str, prompt: str,
                        max_tokens: int) -> str:
    """Fallback for endpoints that only accept the Responses API
    (e.g. databricks-gpt-5-5-pro). Uses the OpenAI SDK's responses.create
    if available; otherwise raises so the caller surfaces the original error.
    """
    if not hasattr(client, "responses"):
        raise RuntimeError("OpenAI SDK lacks responses.create — upgrade openai>=1.40")
    resp = client.responses.create(
        model=model_name,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame_b64}"},
            ],
        }],
        max_output_tokens=max_tokens,
    )
    # Responses API surfaces text via .output_text on recent SDKs
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    # Older SDK shape: iterate output blocks
    chunks: List[str] = []
    for block in (getattr(resp, "output", None) or []):
        for c in (getattr(block, "content", None) or []):
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks)


def _extract_text(content) -> str:
    """Normalize chat-completion `message.content` to a plain string.

    Most endpoints return a string. Reasoning-flavored Claude variants
    (e.g. `databricks-claude-fable-5`) return a LIST of typed content
    blocks instead:

        [
          {"type": "reasoning", "summary": [...]},
          {"type": "text", "text": "{\\"instrument\\": ...}"}
        ]

    Returning the list as-is breaks every downstream consumer that does
    `text.strip()` or `text.find("{")` — manifests as
    `'list' object has no attribute 'strip'` in the Playground cell.
    Concatenate every `type: text` block's text and drop the reasoning
    blocks (the summary is intentionally empty in the response anyway —
    Anthropic keeps the actual chain-of-thought private)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _run_one(model_name: str, frame_name: str, frame_b64: str, prompt: str,
             max_tokens: int, temperature: float) -> dict:
    """Run one (model × frame) call. Some endpoints have quirks we work around:
    - claude-opus-4-7 (and other newer Anthropic reasoning models) reject the
      `temperature` parameter. On that error, retry without it.
    - gpt-5-5-pro only accepts the Responses API. On that error, retry via the
      responses.create endpoint."""
    client = _openai_client()
    t0 = time.time()

    def _chat(features: set) -> str:
        """One attempt with the given feature set. `features` is the set of
        optional kwargs we're willing to send; we drop members one at a time
        in the caller based on which parameter the endpoint rejected."""
        kwargs = dict(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ],
            }],
            max_tokens=max_tokens,
        )
        if "temperature" in features:
            kwargs["temperature"] = temperature
        # Force JSON-mode output so the per-cell parser doesn't hit "unparsed"
        # when a model chooses prose. The prompts already ask for strict
        # JSON; this enforces it. Endpoints that reject `response_format`
        # cascade through the retry below.
        if "response_format" in features:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return _extract_text(resp.choices[0].message.content)

    def _chat_with_retries(features: set, attempts_left: int = 4) -> str:
        """Cascading feature-drop retry. Some endpoints reject multiple
        optional params (e.g. claude-fable-5 rejects BOTH `response_format`
        and `temperature`). Earlier code dropped only one feature per
        attempt and surfaced the second rejection as the user-visible
        error. This loop drops features one at a time based on the error
        message until either a request succeeds or all features have been
        dropped."""
        try:
            return _chat(features)
        except Exception as e:
            if attempts_left <= 0:
                raise
            msg = str(e).lower()
            # response_format rejection — drop it and retry. Gateway phrases
            # the error two ways depending on adapter version:
            #   OpenAI-style:    "response_format is not supported"
            #   Anthropic-style: "Response format type json_object is not
            #                     supported for this model." (fable-5)
            if ("response_format" in msg or "response format" in msg) \
                    and "response_format" in features:
                return _chat_with_retries(features - {"response_format"}, attempts_left - 1)
            # temperature rejection — Anthropic reasoning models often
            # disallow it. "does not support" + "temperature" matches both
            # OpenAI's and Anthropic's phrasings.
            if "temperature" in msg and "does not support" in msg \
                    and "temperature" in features:
                return _chat_with_retries(features - {"temperature"}, attempts_left - 1)
            raise

    try:
        try:
            text = _chat_with_retries({"temperature", "response_format"})
        except Exception as e:
            msg = str(e)
            # Retry via Responses API for endpoints that require it.
            if "responses api" in msg.lower() or "/serving-endpoints/responses" in msg.lower():
                text = _call_responses_api(client, model_name, frame_b64, prompt, max_tokens)
            else:
                raise
        return {
            "model": model_name, "frame": frame_name,
            "ok": True, "elapsed_s": round(time.time() - t0, 2),
            "raw": text, "parsed": _parse_strict_json(text),
        }
    except Exception as e:
        return {
            "model": model_name, "frame": frame_name,
            "ok": False, "elapsed_s": round(time.time() - t0, 2),
            "error": str(e)[:300],
        }


@app.post("/api/playground/run")
async def playground_run(req: PlaygroundRequest):
    """Run all (model × frame) calls in parallel and return results as one
    JSON array when finished.

    We previously used SSE for incremental delivery, but the Databricks Apps
    HTTP/2 ingress buffers streamed responses regardless of `X-Accel-Buffering`.
    With 4-16 calls completing in <10s, batched is plenty fast and far more
    robust than streaming through that proxy.
    """
    if not req.frame_paths or not req.model_names:
        raise HTTPException(400, "Need at least one frame and one model")

    frames_b64: Dict[str, str] = {}
    for p in req.frame_paths:
        _check_allowed_path(p)
        try:
            frames_b64[os.path.basename(p)] = _load_b64(p)
        except Exception as e:
            raise HTTPException(400, f"Failed to read {p}: {e}")

    tasks = [(m, fn, b64) for m in req.model_names for fn, b64 in frames_b64.items()]
    total = len(tasks)
    log.info(f"playground_run: {len(req.model_names)} models × {len(frames_b64)} frames = {total} tasks")

    t_start = time.time()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=min(16, total)) as pool:
        futures = [
            loop.run_in_executor(
                pool, _run_one, m, fn, b64, req.prompt, req.max_tokens, req.temperature,
            )
            for (m, fn, b64) in tasks
        ]
        results = await asyncio.gather(*futures)

    successful = sum(1 for r in results if r.get("ok"))
    return {
        "total": total,
        "successful": successful,
        "elapsed_s": round(time.time() - t_start, 2),
        "models": req.model_names,
        "frames": list(frames_b64.keys()),
        "results": results,
    }


# ── Prompt optimization (kicks off GEPA/DSPy notebook as a Databricks Job) ──

def _sp_home() -> str:
    sp_id = os.environ.get("DATABRICKS_CLIENT_ID")
    if not sp_id:
        raise RuntimeError("DATABRICKS_CLIENT_ID not set; cannot resolve SP home")
    return f"/Workspace/Users/{sp_id}"


_NOTEBOOK_CACHE: Dict[str, str] = {}


def _resolve_optimizer_notebook(kind: str) -> str:
    """Notebooks are bundled inside the app source tree as plain ``.py``
    files. Databricks notebook-task jobs need a workspace-registered
    notebook (not a raw file), and the user's home dir ACL doesn't allow
    the app SP to read. So at first call, we import the bundled .py
    into the SP's own home as a real notebook; subsequent calls reuse
    the imported path. The SP has full access to its own home, so this
    sidesteps every cross-user ACL issue."""
    if kind in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[kind]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "optimizer_notebooks", kind,
    )
    candidates = [
        os.path.join(base_dir, "notebook.py"),
        os.path.join(base_dir, "notebook"),
    ]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        try:
            existing = os.listdir(base_dir)
        except FileNotFoundError:
            existing = "(dir missing)"
        raise RuntimeError(
            f"bundled optimizer notebook missing in {base_dir}; tried {candidates}; "
            f"dir contents: {existing}"
        )
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/optimizer_notebooks/{kind}"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[kind] = target_path
    return target_path
OPTIMIZER_RUNS_DIR = os.environ.get("OPTIMIZER_RUNS_DIR", f"{VOLUME_PATH}/optimizer_runs")

# ── Local (open-weight) models served via serverless GPU jobs ─────────
LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR", f"{VOLUME_PATH}/local_models")
LOCAL_RUNS_DIR = os.environ.get("LOCAL_RUNS_DIR", f"{VOLUME_PATH}/local_runs")

# HF secret scope/key the bundled notebooks read via dbutils.secrets.get.
# Threaded into every YAML config so the notebooks don't need to know which
# scope holds the token in *this* workspace.
HF_SECRET_SCOPE = os.environ.get("HF_SECRET_SCOPE", "hls_g4")
HF_SECRET_KEY = os.environ.get("HF_SECRET_KEY", "HF_TOKEN")


def _resolve_local_inference_notebook() -> str:
    """Bundled inference notebook → SP-home as a real notebook."""
    cache_key = "local_inference_run"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_inference_notebooks", "run",
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        raise RuntimeError(f"local inference notebook missing in {base_dir}")
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/local_inference_notebooks/run"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


def _resolve_local_inference_notebook_variant(variant: str) -> str:
    """Family-specific inference notebooks (e.g. ``run_gemma4`` for Gemma 4
    which needs transformers >= 5.5.0). Same import-to-SP-home pattern as
    ``_resolve_local_inference_notebook``; the variant name is the subdir
    under ``local_inference_notebooks/``. Manifest's
    ``inference_notebook`` field picks which one. Falls back to the
    default ``run`` notebook when the variant subdir is missing."""
    if not variant or variant == "run":
        return _resolve_local_inference_notebook()
    # Reject path-traversal attempts — the manifest is user-editable.
    if "/" in variant or ".." in variant or variant.startswith("."):
        raise RuntimeError(f"refusing variant with suspicious chars: {variant!r}")
    cache_key = f"local_inference_{variant}"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_inference_notebooks", variant,
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        log.warning(f"variant {variant!r} not found at {base_dir} — falling back to default run notebook")
        return _resolve_local_inference_notebook()
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/local_inference_notebooks/{variant}"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


class LocalModelEntry(BaseModel):
    name: str                # subdir name, also the canonical id
    display_name: str
    hf_repo: str
    accelerator: str         # GPU_1xA10 | GPU_8xH100 | ... (NOT GPU_A10 — silently maps to CPU)
    snapshot_dir: Optional[str]  # path to cached weights, None if not yet downloaded
    ready: bool              # True if snapshot dir exists and is non-empty
    notes: Optional[str] = None


@app.get("/api/local-models", response_model=List[LocalModelEntry])
def list_local_models():
    """Scan LOCAL_MODELS_DIR for subdirs containing manifest.yaml. Each
    surfaces as a selectable model in Playground; ``ready`` reflects whether
    weights have been pre-downloaded to ``snapshot/``.

    NOTE: Apps don't FUSE-mount Volumes, so we use the Files API
    (``_ls_dir``/``_download_bytes``) — not ``os.listdir`` — to reach the
    Volume."""
    import yaml
    out: List[LocalModelEntry] = []
    children = _ls_dir(LOCAL_MODELS_DIR)
    for child in sorted(children, key=lambda c: c.get("name") or ""):
        if not child.get("is_directory"):
            continue
        name = child.get("name") or ""
        sub = child.get("path") or f"{LOCAL_MODELS_DIR}/{name}"
        manifest_path = f"{sub}/manifest.yaml"
        try:
            cfg = yaml.safe_load(_download_bytes(manifest_path).decode("utf-8")) or {}
        except Exception as e:
            log.warning(f"[local-models] no/bad manifest under {sub}: {e}")
            continue
        # Manifests can override snapshot_dir to point at a cache that lives
        # elsewhere in the Volume (e.g. downloaded by a different notebook).
        # Falls back to the conventional <sub>/snapshot location.
        snapshot_dir = cfg.get("snapshot_dir") or f"{sub}/snapshot"
        ready = False
        try:
            snap_entries = _ls_dir(snapshot_dir)
            snap_names = [e.get("name") or "" for e in snap_entries]
            has_config = any(n == "config.json" for n in snap_names)
            has_weights = any(n.endswith((".safetensors", ".bin")) for n in snap_names)
            ready = bool(has_config and has_weights)
        except Exception:
            ready = False
        out.append(LocalModelEntry(
            name=cfg.get("name", name),
            display_name=cfg.get("display_name", name),
            hf_repo=cfg.get("hf_repo", ""),
            accelerator=cfg.get("accelerator", "GPU_1xA10"),
            snapshot_dir=snapshot_dir if ready else None,
            ready=ready,
            notes=(cfg.get("notes") or "").strip() or None,
        ))
    return out


class LocalRunRequest(BaseModel):
    model_name: str          # one of LOCAL_MODELS_DIR subdirs
    frame_paths: List[str]
    prompt: str
    max_new_tokens: Optional[int] = None


@app.post("/api/playground/run/local")
def submit_local_run(req: LocalRunRequest):
    """Submit a serverless GPU job that loads the model from Volume cache and
    runs inference for the given frames + prompt. Returns run_id immediately;
    poll ``/api/playground/run/local/<run_id>`` for status + results."""
    import yaml
    if not req.frame_paths:
        raise HTTPException(400, "frame_paths is empty")
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    # Validate frames live under VOLUME_PATH
    for fp in req.frame_paths:
        _check_allowed_path(fp)

    # Look up the manifest via Files API (Apps can't FUSE-mount Volumes)
    manifest_path = f"{LOCAL_MODELS_DIR}/{req.model_name}/manifest.yaml"
    try:
        manifest = yaml.safe_load(_download_bytes(manifest_path).decode("utf-8")) or {}
    except Exception:
        raise HTTPException(404, f"unknown local model '{req.model_name}'")
    snapshot_dir = manifest.get("snapshot_dir") or f"{LOCAL_MODELS_DIR}/{req.model_name}/snapshot"
    try:
        snap_names = [e.get("name") or "" for e in _ls_dir(snapshot_dir)]
        if not (any(n == "config.json" for n in snap_names) and
                any(n.endswith((".safetensors", ".bin")) for n in snap_names)):
            raise HTTPException(409, f"model '{req.model_name}' not yet cached — run setup_cache first")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(409, f"model '{req.model_name}' snapshot dir unreachable")
    accelerator = manifest.get("accelerator", "GPU_1xA10")
    base_environment = manifest.get("base_environment", "databricks_ai_v4")

    run_id = f"local_{int(time.time())}_{req.model_name.replace('/', '_')}"
    out_dir = f"{LOCAL_RUNS_DIR}/{run_id}"
    yaml_path = f"{LOCAL_RUNS_DIR}/{run_id}.yaml"

    cfg = {
        "model_name": req.model_name,
        "model_dir": snapshot_dir,
        "torch_dtype": manifest.get("torch_dtype", "bfloat16"),
        "max_new_tokens": int(req.max_new_tokens or manifest.get("max_new_tokens", 600)),
        "prompt": req.prompt,
        "frame_paths": req.frame_paths,
        "output_dir": out_dir,
        # Threaded so the notebook stays workspace-agnostic
        "hf_secret_scope": HF_SECRET_SCOPE,
        "hf_secret_key": HF_SECRET_KEY,
        "local_models_dir": LOCAL_MODELS_DIR,
    }
    yaml_text = yaml.safe_dump(cfg, default_flow_style=False, allow_unicode=True)
    w = _workspace()
    try:
        w.files.upload(yaml_path, contents=io.BytesIO(yaml_text.encode("utf-8")), overwrite=True)
    except Exception as e:
        raise HTTPException(500, f"failed to write YAML config: {e}")

    # The manifest decides which inference notebook to use. The default
    # `run` notebook handles Qwen3-VL + MedGemma + most other multimodal
    # transformers models. Family-specific variants (`run_gemma4`, …) live
    # alongside it and override env / package requirements.
    notebook_variant = manifest.get("inference_notebook") or "run"
    notebook_path = _resolve_local_inference_notebook_variant(notebook_variant)
    # Correct serverless GPU contract: pair `compute.hardware_accelerator` on
    # the task with a `base_environment: databricks_ai_v4` spec — which is the
    # GPU image that ships with torch + CUDA + ML libs preinstalled. Using
    # `environment_version: "5"` here silently lands on a CPU image (no torch),
    # which manifests as multi-minute hangs followed by ModuleNotFoundError.
    body = {
        "run_name": f"vlmwb_local_{req.model_name}_{run_id}",
        "tasks": [{
            "task_key": "infer",
            "notebook_task": {
                "notebook_path": notebook_path,
                "source": "WORKSPACE",
                "base_parameters": {"config_yaml": yaml_path},
            },
            "environment_key": "gpu_env",
            "compute": {"hardware_accelerator": accelerator},
        }],
        "queue": {"enabled": True},
        "environments": [{
            "environment_key": "gpu_env",
            "spec": {"base_environment": base_environment},
        }],
        "performance_target": "PERFORMANCE_OPTIMIZED",
    }
    try:
        resp = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=body)
        databricks_run_id = resp.get("run_id")
    except Exception as e:
        raise HTTPException(500, f"job submit failed: {e}")
    if not databricks_run_id:
        raise HTTPException(500, "job submit returned no run_id")
    log.info(f"[local-run] submitted {req.model_name} run_id={databricks_run_id} yaml={yaml_path}")
    return {
        "run_id": run_id,
        "databricks_run_id": str(databricks_run_id),
        "yaml_path": yaml_path,
        "output_dir": out_dir,
        "model_name": req.model_name,
        "n_frames": len(req.frame_paths),
    }


@app.get("/api/playground/run/local/{run_id}")
def get_local_run_status(run_id: str, databricks_run_id: str):
    """Live status of a local inference run + parsed results when ready."""
    w = _workspace()
    try:
        resp = w.api_client.do("GET", f"/api/2.1/jobs/runs/get?run_id={databricks_run_id}")
    except Exception as e:
        raise HTTPException(500, f"runs/get failed: {e}")
    state = resp.get("state", {}) or {}
    life = state.get("life_cycle_state")
    result_state = state.get("result_state")
    out: Dict[str, object] = {
        "run_id": run_id,
        "databricks_run_id": databricks_run_id,
        "life_cycle_state": life,
        "result_state": result_state,
        "state_message": state.get("state_message"),
        "run_page_url": resp.get("run_page_url"),
        "results": None,
        "model_name": None,
        "successful": None,
        "n_frames": None,
    }
    if life == "TERMINATED" and result_state == "SUCCESS":
        results_path = f"{LOCAL_RUNS_DIR}/{run_id}/results.json"
        try:
            data = _download_bytes(results_path)
            obj = json.loads(data.decode("utf-8", errors="replace"))
            out["results"] = obj.get("results")
            out["model_name"] = obj.get("model_name")
            out["successful"] = obj.get("successful")
            out["n_frames"] = obj.get("n_frames")
            out["results_path"] = results_path
        except Exception as e:
            out["state_message"] = f"results parse error: {e}"
    return out


# ── Fine-tuning ─────────────────────────────────────────────────────────

FINETUNE_RUNS_DIR = os.environ.get(
    "FINETUNE_RUNS_DIR",
    f"{VOLUME_PATH}/finetune_runs",
)
UC_CATALOG = UC_CATALOG_BOOT
UC_SCHEMA = UC_SCHEMA_BOOT
def _default_mlflow_experiment_path() -> str:
    """Place the experiment under the SP's own workspace home — that's the
    one path we're guaranteed CAN_MANAGE on. Putting it under the human
    user's home requires ACL grants that aren't reliably in place."""
    sp = os.environ.get("DATABRICKS_CLIENT_ID")
    if sp:
        return f"/Users/{sp}/vlmwb-experiments"
    return "/Shared/vlmwb-experiments"


MLFLOW_EXPERIMENT_PATH = os.environ.get(
    "MLFLOW_EXPERIMENT_PATH",
    _default_mlflow_experiment_path(),
)


def _resolve_finetune_notebook() -> str:
    """Bundle the fine-tune notebook into the SP's home (same pattern as the
    optimizer + local-inference notebooks)."""
    cache_key = "finetune_run"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_inference_notebooks", "finetune",
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        raise RuntimeError(f"finetune notebook missing in {base_dir}")
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/local_inference_notebooks/finetune"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


DEFAULT_FT_PROMPT = (
    'Identify the surgical instrument by its visible visual features. '
    'Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", '
    '"tissue_condition": "<short>"} where <class> is one of: probe, shaver, burr, '
    'grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, '
    'scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, '
    'no_instrument_visible.'
)


def _safe_uc_name(s: str) -> str:
    """Validate a 3-part UC table name. Lets letters, digits, _, and dots
    through; rejects anything else so we can safely interpolate the name
    into a SQL statement without enabling injection."""
    import re as _re
    if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$", s or ""):
        raise HTTPException(400, f"invalid UC table name {s!r}; expected catalog.schema.table")
    return s


def _query_delta_labels(
    delta_table: str,
    *,
    delta_version: Optional[int] = None,
    instruments: Optional[List[str]] = None,
    video_names: Optional[List[str]] = None,
    labeled_since: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Read labels from a UC Delta table via the configured SQL warehouse.
    Supports the same composable filters as the Lakebase query (minus
    snapshot — snapshots only exist in Lakebase).

    Required Delta columns: frame_path, instrument, anatomy,
    tissue_condition, created_at. The Label tab's `Sync to Delta` button
    writes a table with exactly that shape; users can also point this at
    any UC Delta with the same columns.

    Inputs are validated (catalog/schema/table identifier regex, instrument
    strings quote-escaped) and the warehouse runs as the App SP — so the
    only access is whatever the SP has on the target table."""
    table = _safe_uc_name(delta_table)
    # Build a parameterized statement. The Databricks SQL Statement Execution
    # API supports named parameters (:name) for values; identifiers must be
    # interpolated (validated above).
    where_clauses: List[str] = ["1=1"]
    named_params: Dict[str, object] = {}

    if instruments:
        # Build :inst0, :inst1, ... param markers
        keys = [f"inst{i}" for i in range(len(instruments))]
        for k, v in zip(keys, instruments):
            named_params[k] = v
        where_clauses.append(f"instrument IN ({', '.join(':' + k for k in keys)})")
    if video_names:
        # `frame_path` contains the video name as a directory segment
        ors = []
        for i, v in enumerate(video_names):
            k = f"vid{i}"
            named_params[k] = f"%/{v}/%"
            ors.append(f"frame_path LIKE :{k}")
        where_clauses.append("(" + " OR ".join(ors) + ")")
    if labeled_since:
        named_params["since"] = labeled_since
        where_clauses.append("created_at >= TIMESTAMP(:since)")

    version_clause = f" VERSION AS OF {int(delta_version)}" if delta_version is not None else ""
    sql = (
        "SELECT frame_path, instrument, anatomy, tissue_condition "
        f"FROM {table}{version_clause} "
        f"WHERE {' AND '.join(where_clauses)} "
        "ORDER BY created_at"
    )

    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise HTTPException(503, "DATABRICKS_WAREHOUSE_ID not set — Delta data source needs a SQL warehouse")
    w = _workspace()
    body: Dict[str, object] = {
        "warehouse_id": warehouse_id,
        "statement": sql,
        "wait_timeout": "30s",
        "disposition": "INLINE",
    }
    if named_params:
        body["parameters"] = [{"name": k, "value": str(v)} for k, v in named_params.items()]
    try:
        resp = w.api_client.do("POST", "/api/2.0/sql/statements", body=body)
    except Exception as e:
        raise HTTPException(500, f"Delta query submit failed: {e}")
    status = (resp.get("status") or {})
    state = status.get("state")
    if state != "SUCCEEDED":
        err = (status.get("error") or {}).get("message") or state
        raise HTTPException(500, f"Delta query failed ({state}): {err}")
    data = ((resp.get("result") or {}).get("data_array") or [])
    return [
        {"image_path": r[0], "instrument": r[1], "anatomy": r[2], "tissue_condition": r[3]}
        for r in data
    ]


def _query_finetune_labels(
    *,
    data_source: str = "lakebase",
    delta_table: Optional[str] = None,
    delta_version: Optional[int] = None,
    snapshot_id: Optional[str] = None,
    snapshot_ids: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    video_names: Optional[List[str]] = None,
    labeled_since: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Resolve labels matching the (composable) Refine-tab filters.
    Used by both the live preview endpoint and the actual training submit
    so the matching set stays identical across the two.

    `data_source`:
      - "lakebase" (default): live Postgres `frame_labels` table
      - "delta":              UC Delta table via SQL warehouse

    Filters all compose with AND:
      - instruments      : restrict to these instrument classes
      - snapshot_id      : legacy single-snapshot field (Lakebase only)
      - snapshot_ids     : multi-select snapshots (Lakebase only; union of their frame_paths)
      - video_names      : labels for frames extracted from these videos
      - labeled_since    : ISO date — labels created at or after this date

    For Lakebase, joins `frame_labels` against `extracted_frames_index` so
    orphan labels pointing at re-ingested-then-deleted JPGs don't sneak
    into training and break the notebook with FileNotFoundError mid-epoch.
    """
    if data_source == "delta":
        target = delta_table or f"{UC_CATALOG}.{UC_SCHEMA}.frame_labels_delta"
        # snapshot_ids is intentionally ignored — snapshots live in Lakebase
        return _query_delta_labels(
            target,
            delta_version=delta_version,
            instruments=instruments,
            video_names=video_names,
            labeled_since=labeled_since,
        )
    if data_source != "lakebase":
        raise HTTPException(400, f"unknown data_source {data_source!r}; expected lakebase|delta")
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured — set data_source='delta' to read from a Delta table instead")
    # Combine snapshot_id (legacy) + snapshot_ids (new) into one list
    snap_ids = list(snapshot_ids or [])
    if snapshot_id and snapshot_id not in snap_ids:
        snap_ids.append(snapshot_id)

    snapshot_frame_paths: Optional[List[str]] = None
    if snap_ids:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT frame_paths FROM snapshots WHERE id = ANY(%s)",
                (snap_ids,),
            )
            rows = cur.fetchall()
        if not rows:
            raise HTTPException(404, f"none of snapshots {snap_ids} found")
        # Union of every snapshot's frame_paths
        snapshot_frame_paths = sorted({fp for r in rows for fp in (r[0] or [])})

    with _pg_conn() as conn, conn.cursor() as cur:
        clauses: List[str] = ["efi.frame_path IS NOT NULL"]
        params: List[object] = []
        if instruments:
            clauses.append("fl.instrument = ANY(%s)")
            params.append(instruments)
        if snapshot_frame_paths is not None:
            clauses.append("fl.frame_path = ANY(%s)")
            params.append(snapshot_frame_paths)
        if video_names:
            # Frames track their source video via the videos JOIN — frame_path
            # contains the video name as a path segment, so a LIKE pattern is
            # the simplest cross-platform match without restructuring the schema.
            # Build N ORs explicitly so each pattern is parameterized.
            like_clauses = []
            for v in video_names:
                like_clauses.append("fl.frame_path LIKE %s")
                params.append(f"%/{v}/%")
            clauses.append("(" + " OR ".join(like_clauses) + ")")
        if labeled_since:
            clauses.append("fl.created_at >= %s::timestamptz")
            params.append(labeled_since)
        where = " WHERE " + " AND ".join(clauses)
        cur.execute(
            f"SELECT fl.frame_path, fl.instrument, fl.anatomy, fl.tissue_condition "
            f"FROM frame_labels fl "
            f"INNER JOIN extracted_frames_index efi ON efi.frame_path = fl.frame_path"
            f"{where} "
            f"ORDER BY fl.created_at",
            tuple(params),
        )
        rows = cur.fetchall()
    return [
        {"image_path": r[0], "instrument": r[1], "anatomy": r[2], "tissue_condition": r[3]}
        for r in rows
    ]


class PreviewRequest(BaseModel):
    instruments: Optional[List[str]] = None
    snapshot_ids: Optional[List[str]] = None
    video_names: Optional[List[str]] = None
    labeled_since: Optional[str] = None
    data_source: str = "lakebase"
    delta_table: Optional[str] = None
    delta_version: Optional[int] = None


@app.post("/api/training/preview")
def preview_finetune_labels(req: PreviewRequest):
    """Live count + class breakdown for the Refine tab's filter UI.
    Same logic as the training submit's data pull, so the count the user
    sees in the preview is exactly what the job will train on."""
    # For lakebase source, gracefully return an empty preview when
    # Postgres isn't reachable (the Refine tab will still render). For
    # Delta source we want the user to see the actual error since they
    # explicitly chose it.
    if req.data_source == "lakebase" and not _lakebase_available():
        return {"total": 0, "by_instrument": []}
    try:
        rows = _query_finetune_labels(
            data_source=req.data_source,
            delta_table=req.delta_table,
            delta_version=req.delta_version,
            snapshot_ids=req.snapshot_ids,
            instruments=req.instruments,
            video_names=req.video_names,
            labeled_since=req.labeled_since,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Surface as a typed 4xx so the UI can render the message in the
        # preview area without breaking the whole tab.
        raise HTTPException(500, f"preview failed: {e}")
    counts: Dict[str, int] = {}
    for r in rows:
        c = r["instrument"] or "(unlabeled)"
        counts[c] = counts.get(c, 0) + 1
    return {
        "total": len(rows),
        "by_instrument": [{"instrument": k, "n": v}
                          for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
    }


class FineTuneRequest(BaseModel):
    base_model_name: str  # one of local_models/ entries (e.g. "qwen3-vl-8b")
    uc_model_name: Optional[str] = None  # e.g. "vlmwb_ft_qwen3vl_<ts>". Auto-generated if None.
    train_prompt: Optional[str] = None

    # ── Label source ────────────────────────────────────────────────────
    # `lakebase` (default) reads from the live Postgres frame_labels table.
    # `delta` reads from a UC Delta table — by default the one the Label
    # tab's "Sync to Delta" button writes (frame_labels_delta), but the
    # user can override with any 3-part Delta name (`<catalog>.<schema>.<table>`)
    # as long as it exposes `frame_path / instrument / anatomy /
    # tissue_condition / created_at` columns. Delta is useful for:
    #   • training on a stable, versioned snapshot (Lakebase mutates)
    #   • training on labels imported from other tools
    #   • time-travel via `delta_version`
    data_source: str = "lakebase"
    delta_table: Optional[str] = None                       # NEW — Delta override
    delta_version: Optional[int] = None                     # NEW — Delta time-travel
    # ── Label filters (composed with AND) ───────────────────────────────
    # Each is optional; an empty / null value means "no restriction on this
    # dimension". When multiple are provided, the resulting frame set is
    # the intersection. Frontend (Refine tab) shows a live preview of how
    # many frames match the current filter combo via /api/training/preview.
    label_filter_instruments: Optional[List[str]] = None   # by instrument class
    snapshot_id: Optional[str] = None                       # legacy single-snapshot filter
    snapshot_ids: Optional[List[str]] = None                # NEW — multiple snapshots (Lakebase only)
    video_names: Optional[List[str]] = None                 # NEW — by source video
    labeled_since: Optional[str] = None                     # NEW — ISO date; labels created on/after

    # ── Hyperparams (everything threaded through to the notebook YAML) ──
    num_epochs: float = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05                              # NEW
    per_device_batch_size: int = 1                          # NEW
    grad_accum_steps: int = 8                               # NEW
    warmup_ratio: float = 0.03                              # NEW
    weight_decay: float = 0.0                               # NEW
    max_length: int = 1024                                  # NEW
    accelerator: str = "GPU_8xH100"  # bigger model + bigger batch wants more GPUs


@app.post("/api/training/finetune")
def kick_off_finetune(req: FineTuneRequest):
    """Submit a fine-tune job:
    1. Pull all rows from frame_labels (optionally filtered).
    2. Build YAML config + drop into Volume.
    3. Submit GPU job that runs local_inference_notebooks/finetune/notebook.py.
    Returns immediately with run_id; the Train UI polls for status."""
    import yaml

    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured — fine-tune needs the labels table")
    _ensure_labels_table()

    # Resolve base model snapshot dir from its manifest
    manifest_path = f"{LOCAL_MODELS_DIR}/{req.base_model_name}/manifest.yaml"
    try:
        base_manifest = yaml.safe_load(_download_bytes(manifest_path).decode("utf-8")) or {}
    except Exception:
        raise HTTPException(404, f"unknown base model '{req.base_model_name}'")
    base_model_dir = (
        base_manifest.get("snapshot_dir")
        or f"{LOCAL_MODELS_DIR}/{req.base_model_name}/snapshot"
    )

    # Pull labeled training data. Filters compose with AND. The
    # data_source picker (lakebase | delta) lets users either hit the
    # live Lakebase frame_labels table or read a stable Delta snapshot
    # (e.g., what the Label tab's `Sync to Delta` button writes). Same
    # downstream pipeline regardless of source.
    training_data = _query_finetune_labels(
        data_source=req.data_source,
        delta_table=req.delta_table,
        delta_version=req.delta_version,
        snapshot_id=req.snapshot_id,
        snapshot_ids=req.snapshot_ids,
        instruments=req.label_filter_instruments,
        video_names=req.video_names,
        labeled_since=req.labeled_since,
    )
    if len(training_data) < 4:
        raise HTTPException(
            400,
            f"need at least 4 labeled frames to fine-tune; have {len(training_data)}. "
            f"Open the Label tab to add more."
        )

    run_id = f"ft_{int(time.time())}_{req.base_model_name.replace('/', '_').replace('-', '')[:18]}"
    uc_model_name = req.uc_model_name or run_id  # short, sluggable
    out_dir = f"{FINETUNE_RUNS_DIR}/{run_id}"
    yaml_path = f"{FINETUNE_RUNS_DIR}/{run_id}.yaml"

    cfg: Dict[str, object] = {
        "run_id": run_id,
        "base_model_name": req.base_model_name,
        "base_model_dir": base_model_dir,
        "uc_catalog": UC_CATALOG,
        "uc_schema": UC_SCHEMA,
        "uc_model_name": uc_model_name,
        "mlflow_experiment": MLFLOW_EXPERIMENT_PATH,
        "output_dir": out_dir,
        "training_data": training_data,
        "train_prompt": (req.train_prompt or "").strip() or DEFAULT_FT_PROMPT,
        # Workspace-specific knobs threaded through so the notebook stays portable
        "hf_secret_scope": HF_SECRET_SCOPE,
        "hf_secret_key": HF_SECRET_KEY,
        "local_models_dir": LOCAL_MODELS_DIR,
        "lora": {
            "r": req.lora_r,
            "alpha": req.lora_alpha,
            "dropout": req.lora_dropout,
        },
        "training": {
            "num_epochs": req.num_epochs,
            "per_device_train_batch_size": req.per_device_batch_size,
            "gradient_accumulation_steps": req.grad_accum_steps,
            "learning_rate": req.learning_rate,
            "warmup_ratio": req.warmup_ratio,
            "weight_decay": req.weight_decay,
            "max_length": req.max_length,
        },
    }
    yaml_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

    w = _workspace()
    try:
        w.files.create_directory(FINETUNE_RUNS_DIR)
    except Exception:
        pass
    try:
        w.files.upload(yaml_path, contents=io.BytesIO(yaml_text.encode("utf-8")), overwrite=True)
    except Exception as e:
        raise HTTPException(500, f"failed to write YAML: {e}")

    notebook_path = _resolve_finetune_notebook()
    body = {
        "run_name": f"vlmwb_finetune_{run_id}",
        "tasks": [{
            "task_key": "finetune",
            "notebook_task": {
                "notebook_path": notebook_path,
                "source": "WORKSPACE",
                "base_parameters": {"config_yaml": yaml_path},
            },
            "environment_key": "gpu_env",
            "compute": {"hardware_accelerator": req.accelerator},
        }],
        "queue": {"enabled": True},
        "environments": [{
            "environment_key": "gpu_env",
            "spec": {"base_environment": "databricks_ai_v4"},
        }],
        "performance_target": "PERFORMANCE_OPTIMIZED",
    }
    try:
        resp = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=body)
        databricks_run_id = resp.get("run_id")
    except Exception as e:
        raise HTTPException(500, f"job submit failed: {e}")
    if not databricks_run_id:
        raise HTTPException(500, "job submit returned no run_id")
    log.info(f"[finetune] submitted run_id={databricks_run_id} ({run_id}) base={req.base_model_name}")
    return {
        "run_id": run_id,
        "databricks_run_id": str(databricks_run_id),
        "yaml_path": yaml_path,
        "output_dir": out_dir,
        "uc_model_name": uc_model_name,
        "n_train": len(training_data),
    }


@app.get("/api/training/finetune/{run_id}")
def get_finetune_status(run_id: str, databricks_run_id: str):
    """Live status of a fine-tune run + the result blob when ready."""
    w = _workspace()
    try:
        resp = w.api_client.do("GET", f"/api/2.1/jobs/runs/get?run_id={databricks_run_id}")
    except Exception as e:
        raise HTTPException(500, f"runs/get failed: {e}")
    state = resp.get("state", {}) or {}
    life = state.get("life_cycle_state")
    result_state = state.get("result_state")
    out: Dict[str, object] = {
        "run_id": run_id,
        "databricks_run_id": databricks_run_id,
        "life_cycle_state": life,
        "result_state": result_state,
        "state_message": state.get("state_message"),
        "run_page_url": resp.get("run_page_url"),
        "result": None,
    }
    if life == "TERMINATED" and result_state == "SUCCESS":
        result_path = f"{FINETUNE_RUNS_DIR}/{run_id}/result.json"
        try:
            data = _download_bytes(result_path)
            out["result"] = json.loads(data.decode("utf-8", errors="replace"))
        except Exception as e:
            out["state_message"] = f"result parse error: {e}"
    return out


@app.get("/api/training/runs")
def list_finetune_runs(limit: int = 25):
    """List past fine-tune runs from MLflow. Returns lightweight summaries
    the Train tab can show in a history strip."""
    out: List[Dict[str, object]] = []
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        client = mlflow.tracking.MlflowClient()
        try:
            exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH)
        except Exception:
            exp = None
        if exp is None:
            return out
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
        for r in runs:
            params = r.data.params
            metrics = r.data.metrics
            out.append({
                "mlflow_run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "base_model": params.get("base_model"),
                "n_train": params.get("n_train"),
                "lora_r": params.get("lora_r"),
                "learning_rate": params.get("learning_rate"),
                "final_train_loss": metrics.get("final_train_loss"),
                "train_elapsed_s": metrics.get("train_elapsed_s"),
            })
    except Exception as e:
        log.warning(f"list_finetune_runs failed: {e}")
    return out


# ── Model deployment (serving endpoints) ────────────────────────────────
#
# Surface every UC-registered model produced by the fine-tune workflow + every
# live serving endpoint, and let the user deploy / undeploy from one page. A
# successfully deployed endpoint automatically appears in Playground's AI
# Gateway dropdown (since /api/models enumerates all serving endpoints), so
# the same UI handles A/B testing the fine-tuned model against base Qwen +
# AI Gateway models.

DEPLOY_TAG_KEY = "vlmwb_managed"  # tag we put on every endpoint we create


class DeployableModel(BaseModel):
    full_name: str           # e.g. hls_amer_catalog.guanyu_chen.vlmwb_ft_xxx
    catalog: str
    schema: str
    name: str
    versions: List[str]      # sorted desc, latest first
    base_model: Optional[str] = None  # from the MLflow run's params, if available
    n_train: Optional[str] = None
    train_loss: Optional[float] = None
    updated_at: Optional[str] = None


@app.get("/api/deploy/models", response_model=List[DeployableModel])
def list_deployable_models():
    """List UC models in the workbench's catalog.schema. Pulls every model
    version + cross-references with MLflow runs so the UI can show base_model
    and training metadata.

    IMPLEMENTATION NOTE: MLflow's `search_registered_models(filter_string=...)`
    is NOT supported for UC-managed models — calling it with a filter raises
    `MlflowException: Argument 'filter_string' is unsupported for models in
    the Unity Catalog.` And the unfiltered version paginates against the
    workspace's entire registry, so models in our schema may not appear in
    the first page. Use the UC REST API directly to scope the query.
    """
    out: List[DeployableModel] = []
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.tracking.MlflowClient()
        w = _workspace()
        # UC REST: GET /api/2.1/unity-catalog/models?catalog_name=...&schema_name=...
        try:
            uc_resp = w.api_client.do(
                "GET", "/api/2.1/unity-catalog/models",
                query={"catalog_name": UC_CATALOG, "schema_name": UC_SCHEMA, "max_results": 100},
            )
            uc_models = (uc_resp or {}).get("registered_models", [])
        except Exception as e:
            log.warning(f"UC models query failed: {e}")
            uc_models = []
        for uc in uc_models:
            full = uc.get("full_name") or f"{UC_CATALOG}.{UC_SCHEMA}.{uc.get('name','')}"
            if not full.startswith(f"{UC_CATALOG}.{UC_SCHEMA}."):
                continue
            short = full.rsplit(".", 1)[-1]
            try:
                vlist = client.search_model_versions(filter_string=f"name='{full}'", max_results=20)
                versions = sorted([v.version for v in vlist], key=int, reverse=True)
            except Exception:
                versions = []
            base_model = None
            n_train = None
            train_loss = None
            updated_at = None
            # Pull the latest version's MLflow run for metadata
            if versions:
                try:
                    latest = next(v for v in vlist if v.version == versions[0])
                    if getattr(latest, "run_id", None):
                        r = client.get_run(latest.run_id)
                        params = r.data.params
                        metrics = r.data.metrics
                        base_model = params.get("base_model")
                        n_train = params.get("n_train")
                        train_loss = metrics.get("final_train_loss")
                        if r.info.end_time:
                            import datetime
                            updated_at = datetime.datetime.fromtimestamp(
                                r.info.end_time / 1000
                            ).isoformat()
                except Exception:
                    pass
            out.append(DeployableModel(
                full_name=full, catalog=UC_CATALOG, schema=UC_SCHEMA,
                name=short, versions=versions, base_model=base_model,
                n_train=n_train, train_loss=train_loss, updated_at=updated_at,
            ))
    except Exception as e:
        log.warning(f"list_deployable_models failed: {e}")
    # Sort newest first
    out.sort(key=lambda m: m.updated_at or "", reverse=True)
    return out


class ServingEndpointRow(BaseModel):
    name: str
    state: str
    config_state: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    workload_size: Optional[str] = None
    workload_type: Optional[str] = None
    creator: Optional[str] = None
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    invocation_url: Optional[str] = None
    managed: bool = False  # True if this endpoint carries our vlmwb_managed tag


@app.get("/api/deploy/endpoints", response_model=List[ServingEndpointRow])
def list_serving_endpoints(only_managed: bool = False):
    """List serving endpoints in the workspace. Filters to ones tagged by
    this app (``vlmwb_managed=true``) when ``only_managed=true``."""
    w = _workspace()
    try:
        resp = w.api_client.do("GET", "/api/2.0/serving-endpoints")
    except Exception as e:
        raise HTTPException(500, f"list endpoints failed: {e}")
    items = (resp.get("endpoints") or [])
    out: List[ServingEndpointRow] = []
    for ep in items:
        tags = ep.get("tags") or []
        managed = any(t.get("key") == DEPLOY_TAG_KEY for t in tags)
        if only_managed and not managed:
            continue
        cfg = ep.get("config") or {}
        served = (cfg.get("served_entities") or cfg.get("served_models") or [{}])[0]
        state = (ep.get("state") or {}).get("ready") or "?"
        config_state = (ep.get("state") or {}).get("config_update")
        out.append(ServingEndpointRow(
            name=ep.get("name") or "",
            state=str(state),
            config_state=config_state,
            model=served.get("entity_name") or served.get("model_name"),
            version=str(served.get("entity_version") or served.get("model_version") or ""),
            workload_size=served.get("workload_size"),
            workload_type=served.get("workload_type"),
            creator=ep.get("creator"),
            creation_timestamp=ep.get("creation_timestamp"),
            last_updated_timestamp=ep.get("last_updated_timestamp"),
            invocation_url=ep.get("endpoint_url"),
            managed=managed,
        ))
    return out


class DeployRequest(BaseModel):
    model_full_name: str        # e.g. hls_amer_catalog.guanyu_chen.vlmwb_ft_xxx
    model_version: Optional[str] = None  # default = latest
    endpoint_name: Optional[str] = None  # default = sanitized model name + ts
    workload_type: str = "GPU_LARGE"     # GPU_SMALL | GPU_MEDIUM | GPU_LARGE
    workload_size: str = "Small"         # Small | Medium | Large
    scale_to_zero: bool = True


@app.post("/api/deploy/endpoints")
def create_serving_endpoint(req: DeployRequest):
    """Create a serving endpoint from a UC-registered model. Returns
    immediately; poll /api/deploy/endpoints/{name} until state=READY."""
    import re
    from databricks.sdk import WorkspaceClient
    mlflow_client = None
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow_client = mlflow.tracking.MlflowClient()
    except Exception as e:
        raise HTTPException(500, f"mlflow not available: {e}")

    # Resolve version
    if req.model_version:
        version = req.model_version
    else:
        try:
            vlist = mlflow_client.search_model_versions(
                filter_string=f"name='{req.model_full_name}'", max_results=20,
            )
            if not vlist:
                raise HTTPException(404, f"no versions found for {req.model_full_name}")
            version = sorted([v.version for v in vlist], key=int, reverse=True)[0]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"version lookup failed: {e}")

    # Build endpoint name. Names: lowercase alnum + dashes, ≤63 chars.
    if req.endpoint_name:
        name = req.endpoint_name
    else:
        short = req.model_full_name.rsplit(".", 1)[-1]
        name = re.sub(r"[^a-z0-9-]+", "-", short.lower()).strip("-")[:48] + f"-{int(time.time())}"
    name = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:63]

    body = {
        "name": name,
        "config": {
            "served_entities": [{
                "name": name + "-srv",
                "entity_name": req.model_full_name,
                "entity_version": str(version),
                "workload_size": req.workload_size,
                "workload_type": req.workload_type,
                "scale_to_zero_enabled": bool(req.scale_to_zero),
            }],
            "traffic_config": {
                "routes": [{"served_model_name": name + "-srv", "traffic_percentage": 100}],
            },
        },
        "tags": [
            {"key": DEPLOY_TAG_KEY, "value": "true"},
            {"key": "vlmwb_source_model", "value": req.model_full_name},
            {"key": "vlmwb_source_version", "value": str(version)},
        ],
    }
    w = _workspace()
    try:
        resp = w.api_client.do("POST", "/api/2.0/serving-endpoints", body=body)
    except Exception as e:
        # Common case: endpoint already exists. Fall through to UPDATE.
        msg = str(e)
        if "RESOURCE_ALREADY_EXISTS" in msg or "already exists" in msg.lower():
            try:
                resp = w.api_client.do(
                    "PUT",
                    f"/api/2.0/serving-endpoints/{name}/config",
                    body=body["config"],
                )
            except Exception as e2:
                raise HTTPException(500, f"update endpoint failed: {e2}")
        else:
            raise HTTPException(500, f"create endpoint failed: {e}")
    log.info(f"[deploy] created endpoint {name} → {req.model_full_name}/{version}")
    return {
        "name": name,
        "model_full_name": req.model_full_name,
        "version": str(version),
        "workload_type": req.workload_type,
        "workload_size": req.workload_size,
        "endpoint": resp,
    }


@app.get("/api/deploy/endpoints/{name}")
def get_serving_endpoint(name: str):
    w = _workspace()
    try:
        return w.api_client.do("GET", f"/api/2.0/serving-endpoints/{name}")
    except Exception as e:
        raise HTTPException(404, f"endpoint not found: {e}")


class RepinRequest(BaseModel):
    model_full_name: str
    version: Optional[str] = None  # default = latest


@app.post("/api/deploy/repin-requirements")
def repin_model_requirements(req: RepinRequest):
    """Re-register a UC model with corrected pip_requirements. Used when an
    existing model was logged without ``torchvision``/``safetensors`` pinned,
    causing Model Serving to install incompatible CUDA wheels and fail at
    load time with ``operator torchvision::nms does not exist``.

    This kicks off a small GPU job (we need torch + transformers + the
    merged-model dir on disk) that runs ``mlflow.transformers.log_model``
    with the right requirements + bumps the UC model version. Returns
    immediately; the new version then deploys cleanly."""
    notebook_path = _resolve_repin_notebook()

    # Pull the source model's artifact dir from its MLflow run, since that's
    # where the merged checkpoint actually lives.
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.tracking.MlflowClient()
        if req.version:
            mv = client.get_model_version(req.model_full_name, req.version)
        else:
            vlist = client.search_model_versions(
                filter_string=f"name='{req.model_full_name}'", max_results=20,
            )
            if not vlist:
                raise HTTPException(404, f"no versions for {req.model_full_name}")
            mv = max(vlist, key=lambda v: int(v.version))
        run = client.get_run(mv.run_id) if mv.run_id else None
        # Find the merged-dir path from the original run's params/metadata
        merged_dir = None
        if run:
            # First try the metadata field we set
            tags = run.data.tags or {}
            for k in ("merged_dir", "vlmwb.merged_dir"):
                if k in tags:
                    merged_dir = tags[k]; break
            params = run.data.params or {}
            if not merged_dir:
                # Fall back: the result.json we wrote in the original run
                # carries the merged_dir; convention-derive from run_id.
                # Easier path: derive from base_model_name + ts in the run name.
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"mlflow lookup failed: {e}")

    if not merged_dir:
        raise HTTPException(
            400,
            "could not resolve merged_dir for this model from the MLflow run. "
            "Retrain via the Train tab — the new run will use the corrected "
            "pip pinning automatically."
        )

    run_id = f"repin_{int(time.time())}_{req.model_full_name.split('.')[-1][:16]}"
    out_dir = f"{FINETUNE_RUNS_DIR}/{run_id}"
    import yaml
    cfg = {
        "run_id": run_id,
        "uc_full_name": req.model_full_name,
        "merged_dir": merged_dir,
        "output_dir": out_dir,
        "mlflow_experiment": MLFLOW_EXPERIMENT_PATH,
    }
    yaml_path = f"{FINETUNE_RUNS_DIR}/{run_id}.yaml"
    w = _workspace()
    try:
        w.files.create_directory(FINETUNE_RUNS_DIR)
    except Exception:
        pass
    try:
        w.files.upload(yaml_path, contents=io.BytesIO(
            yaml.safe_dump(cfg, sort_keys=False).encode("utf-8")
        ), overwrite=True)
    except Exception as e:
        raise HTTPException(500, f"yaml upload failed: {e}")

    body = {
        "run_name": f"vlmwb_repin_{run_id}",
        "tasks": [{
            "task_key": "repin",
            "notebook_task": {
                "notebook_path": notebook_path,
                "source": "WORKSPACE",
                "base_parameters": {"config_yaml": yaml_path},
            },
            "environment_key": "gpu_env",
            "compute": {"hardware_accelerator": "GPU_1xA10"},
        }],
        "queue": {"enabled": True},
        "environments": [{
            "environment_key": "gpu_env",
            "spec": {"base_environment": "databricks_ai_v4"},
        }],
        "performance_target": "PERFORMANCE_OPTIMIZED",
    }
    try:
        resp = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=body)
    except Exception as e:
        raise HTTPException(500, f"job submit failed: {e}")
    return {
        "run_id": run_id,
        "databricks_run_id": str(resp.get("run_id")),
        "merged_dir": merged_dir,
        "model_full_name": req.model_full_name,
    }


def _resolve_repin_notebook() -> str:
    cache_key = "repin_requirements"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_inference_notebooks", "repin",
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        raise RuntimeError(f"repin notebook missing in {base_dir}")
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/local_inference_notebooks/repin"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


@app.delete("/api/deploy/endpoints/{name}")
def delete_serving_endpoint(name: str):
    """Remove a serving endpoint. Only allowed for endpoints tagged as managed
    by this app — refuses to touch ones created elsewhere."""
    w = _workspace()
    try:
        info = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{name}")
    except Exception as e:
        raise HTTPException(404, f"endpoint not found: {e}")
    tags = info.get("tags") or []
    if not any(t.get("key") == DEPLOY_TAG_KEY for t in tags):
        raise HTTPException(
            403,
            f"refusing to delete '{name}' — it isn't tagged as vlmwb-managed. "
            f"Delete from the Databricks UI if you really want to remove it.",
        )
    try:
        w.api_client.do("DELETE", f"/api/2.0/serving-endpoints/{name}")
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")
    return {"deleted": name}


class OptimizeRequest(BaseModel):
    snapshot_id: str
    optimizer: str = "gepa"  # "gepa" | "dspy"
    teacher_model: str       # source of pseudo-labels (must be a model in the snapshot)
    student_model: str       # the FMAPI endpoint whose prompt we're optimizing
    n_rounds: int = 5        # GEPA only
    dspy_optimizer_type: str = "BootstrapFewShot"  # DSPy only
    seed_prompt_override: Optional[str] = None     # default = snapshot.prompt
    reflection_mode: str = "replace"
    early_stop_patience: int = 2


def _instrument_class_from_result(parsed: object) -> Optional[str]:
    """Extract the primary class from either Studio or Playground prompt format."""
    if not isinstance(parsed, dict):
        return None
    direct = parsed.get("instrument")
    if isinstance(direct, str) and direct:
        return direct
    inst_list = parsed.get("instruments")
    if isinstance(inst_list, list) and inst_list and isinstance(inst_list[0], dict):
        cls = inst_list[0].get("class")
        if isinstance(cls, str) and cls:
            return cls
    return None


def _build_eval_data_from_snapshot(snap: dict, teacher_model: str) -> List[dict]:
    """Use the teacher model's predictions on each frame as the gold label."""
    teacher_rows = [r for r in (snap.get("results") or [])
                    if r.get("model") == teacher_model and r.get("ok")]
    # Map frame name → teacher's predicted class
    label_by_frame: Dict[str, str] = {}
    for r in teacher_rows:
        cls = _instrument_class_from_result(r.get("parsed"))
        if cls:
            label_by_frame[r.get("frame", "")] = cls
    out = []
    frame_paths = snap.get("frame_paths") or []
    for path in frame_paths:
        fname = os.path.basename(path)
        if fname not in label_by_frame:
            continue
        out.append({"input": path, "expected": label_by_frame[fname], "id": fname})
    return out


# Sentinel value for the teacher_model field meaning "use ground-truth labels
# from the frame_labels table rather than any model's predictions."
GOLD_TEACHER = "__gold__"


def _build_eval_data_from_gold_labels(snap: dict) -> List[dict]:
    """Use ground-truth labels from the frame_labels table as the gold answer
    for each frame in the snapshot. Frames without a label are dropped."""
    frame_paths = snap.get("frame_paths") or []
    if not frame_paths:
        return []
    _ensure_labels_table()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT frame_path, instrument FROM frame_labels WHERE frame_path = ANY(%s)",
            (frame_paths,),
        )
        label_by_path = {r[0]: r[1] for r in cur.fetchall()}
    out = []
    for path in frame_paths:
        inst = label_by_path.get(path)
        if not inst:
            continue
        out.append({"input": path, "expected": inst, "id": os.path.basename(path)})
    return out


@app.post("/api/playground/optimize")
def kick_off_optimize(req: OptimizeRequest):
    """Compose a YAML config from a snapshot and submit a notebook-task
    Databricks Job. Returns a `run_id` the frontend can poll."""
    if req.optimizer not in ("gepa", "dspy"):
        raise HTTPException(400, "optimizer must be 'gepa' or 'dspy'")
    try:
        notebook_path = _resolve_optimizer_notebook(req.optimizer)
    except Exception as e:
        raise HTTPException(500, f"failed to resolve optimizer notebook path: {e}")

    # Resolve snapshot
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase unavailable")
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT prompt, frame_paths, model_names, results FROM snapshots WHERE id = %s",
                        (req.snapshot_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"snapshot {req.snapshot_id} not found")
            snap = {
                "prompt": row[0], "frame_paths": row[1],
                "model_names": row[2], "results": row[3],
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"snapshot lookup failed: {e}")

    # Two teacher modes:
    #   - GOLD_TEACHER → pull `expected` from the labels table (supervised).
    #   - any other model name → use that model's snapshot predictions as
    #     pseudo-labels (the original behavior).
    if req.teacher_model == GOLD_TEACHER:
        eval_data = _build_eval_data_from_gold_labels(snap)
        if not eval_data:
            raise HTTPException(
                400,
                "No gold labels found for any frame in this snapshot. Open the "
                "Label tab and label some of these frames before retrying."
            )
    else:
        if req.teacher_model not in (snap["model_names"] or []):
            raise HTTPException(400,
                f"teacher_model '{req.teacher_model}' wasn't in the snapshot's model list")
        eval_data = _build_eval_data_from_snapshot(snap, req.teacher_model)
        if not eval_data:
            raise HTTPException(400,
                f"No usable predictions from teacher '{req.teacher_model}' in snapshot — "
                "check that this model produced parseable JSON during the Playground run.")

    seed_prompt = req.seed_prompt_override or snap["prompt"]
    # Snapshot the seed prompt into the Prompt Registry so we have a stable,
    # versioned reference to whatever we started optimizing from. Non-fatal.
    seed_prompt_info = _register_genai_prompt(
        seed_prompt,
        commit_message=f"seed for {req.optimizer} optimize of snapshot {req.snapshot_id[:8]}",
        tags={
            "vlmwb_role": "seed",
            "snapshot_id": req.snapshot_id,
            "optimizer": req.optimizer,
            "student_model": req.student_model,
        },
    )
    run_id = f"opt_{int(time.time())}_{req.snapshot_id[:8]}_{req.optimizer}"
    out_dir = f"{OPTIMIZER_RUNS_DIR}/{run_id}"

    # Detect whether the student is a local (HF-from-Volume) model or an AI
    # Gateway endpoint. The GEPA/DSPy notebook understands two providers:
    #   - databricks_fmapi → calls an FMAPI endpoint via OpenAI-compatible client
    #   - huggingface_local → loads the model in-process (needs a GPU job)
    # Without this branch a local student name like "qwen3-vl-8b" was being
    # sent as an FMAPI endpoint, every request 404'd, and every round scored 0.
    import yaml
    local_manifest = None
    try:
        local_manifest = yaml.safe_load(
            _download_bytes(f"{LOCAL_MODELS_DIR}/{req.student_model}/manifest.yaml")
            .decode("utf-8")
        )
    except Exception:
        local_manifest = None

    if local_manifest:
        local_snapshot_dir = (
            local_manifest.get("snapshot_dir")
            or f"{LOCAL_MODELS_DIR}/{req.student_model}/snapshot"
        )
        task_model_cfg = {
            "provider": "huggingface_local",
            "model_id": local_snapshot_dir,
            "max_new_tokens": 400,
            "temperature": 0.0,
            "image_max_side": 896,
        }
    else:
        task_model_cfg = {
            "provider": "databricks_fmapi",
            "endpoint": req.student_model,
            "max_new_tokens": 400,
            "temperature": 0.0,
            "image_max_side": 896,
        }

    cfg: Dict[str, object] = {
        "output_dir": out_dir,
        # Workspace-specific knobs the bundled notebook reads via _cfg.get(...)
        "hf_secret_scope": HF_SECRET_SCOPE,
        "hf_secret_key": HF_SECRET_KEY,
        "task": {
            "name": run_id,
            "description": "Prompt optimization for arthroscopy instrument ID",
            "input_modality": "image",
            "metric": "class_in_text",
            "vocabulary": [
                "probe", "shaver", "burr", "grasper", "biter", "suture_passer",
                "anchor_driver", "electrocautery", "cannula", "scissors",
                "drill_guide", "trocar", "knot_pusher", "rasp",
                "other_metal_tool", "no_instrument_visible",
            ],
        },
        "seed_prompt": seed_prompt,
        "eval_data": eval_data,
        "task_model": task_model_cfg,
    }
    if req.optimizer == "gepa":
        cfg["reflector_model"] = {
            "provider": "databricks_fmapi",
            "endpoint": "databricks-claude-sonnet-4-6",
            "max_tokens": 1500, "temperature": 0.4,
        }
        cfg["optimizer"] = {
            "n_rounds": req.n_rounds,
            "n_failure_samples": 6,
            "reflection_mode": req.reflection_mode,
            "early_stop_patience": req.early_stop_patience,
            "seed": 42,
        }
    else:
        cfg["prompt_model"] = {
            "provider": "databricks_fmapi",
            "endpoint": "databricks-claude-sonnet-4-6",
            "max_tokens": 2000, "temperature": 0.7,
        }
        cfg["optimizer"] = {
            "type": req.dspy_optimizer_type,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 4,
            "max_rounds": 1,
            "num_candidate_programs": 6,
            "num_candidates": 10,
            "init_temperature": 1.4,
            "breadth": 5, "depth": 3,
        }

    yaml_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    yaml_path = f"{OPTIMIZER_RUNS_DIR}/{run_id}.yaml"

    w = _workspace()
    # Make sure the runs dir exists in volume; ignore if it already does
    try:
        w.files.create_directory(OPTIMIZER_RUNS_DIR)
    except Exception:
        pass
    try:
        w.files.upload(yaml_path, contents=io.BytesIO(yaml_text.encode("utf-8")), overwrite=True)
    except Exception as e:
        raise HTTPException(500, f"failed to write YAML config: {e}")

    # When the student is a local HF model, the optimizer notebook loads it
    # in-process and we need a GPU. For pure-FMAPI students, serverless CPU
    # is sufficient (no model weights live in the notebook process).
    if local_manifest:
        accelerator = local_manifest.get("accelerator", "GPU_1xA10")
        base_environment = local_manifest.get("base_environment", "databricks_ai_v4")
        body = {
            "run_name": f"vlmwb_optimize_{req.optimizer}_{run_id}",
            "tasks": [{
                "task_key": "optimize",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "source": "WORKSPACE",
                    "base_parameters": {"config_yaml": yaml_path},
                },
                "environment_key": "gpu_env",
                "compute": {"hardware_accelerator": accelerator},
            }],
            "queue": {"enabled": True},
            "environments": [{
                "environment_key": "gpu_env",
                "spec": {"base_environment": base_environment},
            }],
            "performance_target": "PERFORMANCE_OPTIMIZED",
        }
    else:
        body = {
            "run_name": f"vlmwb_optimize_{req.optimizer}_{run_id}",
            "tasks": [{
                "task_key": "optimize",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "base_parameters": {"config_yaml": yaml_path},
                },
            }],
            "performance_target": "PERFORMANCE_OPTIMIZED",
        }
    try:
        resp = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=body)
        databricks_run_id = resp.get("run_id")
    except Exception as e:
        raise HTTPException(500, f"job submit failed: {e}")
    if not databricks_run_id:
        raise HTTPException(500, "job submit returned no run_id")
    log.info(f"[optimize] submitted {req.optimizer} run_id={databricks_run_id} yaml={yaml_path}")

    return {
        "run_id": run_id,
        "databricks_run_id": str(databricks_run_id),
        "yaml_path": yaml_path,
        "output_dir": out_dir,
        "n_eval_examples": len(eval_data),
        "optimizer": req.optimizer,
        "student_model": req.student_model,
        "teacher_model": req.teacher_model,
    }


@app.get("/api/playground/optimize/{run_id}")
def get_optimize_status(run_id: str, databricks_run_id: str):
    """Fetch live status of an optimization run + best_prompt if ready."""
    w = _workspace()
    try:
        resp = w.api_client.do("GET", f"/api/2.1/jobs/runs/get?run_id={databricks_run_id}")
    except Exception as e:
        raise HTTPException(500, f"runs/get failed: {e}")

    state = resp.get("state", {}) or {}
    life = state.get("life_cycle_state")
    result_state = state.get("result_state")
    out: Dict[str, object] = {
        "run_id": run_id,
        "databricks_run_id": databricks_run_id,
        "life_cycle_state": life,
        "result_state": result_state,
        "state_message": state.get("state_message"),
        "run_page_url": resp.get("run_page_url"),
        "best_prompt": None,
        "score": None,
        "history": None,
    }

    if life == "TERMINATED" and result_state == "SUCCESS":
        # Try to pull best_prompt.txt + run.json from the output dir
        out_dir = f"{OPTIMIZER_RUNS_DIR}/{run_id}/{run_id}"  # task name == run_id, OUTPUT_DIR/<task>
        # Compatibility: try both layouts
        candidates = [
            f"{OPTIMIZER_RUNS_DIR}/{run_id}/{run_id}/best_prompt.txt",
            f"{OPTIMIZER_RUNS_DIR}/{run_id}/{run_id}/optimized_prompt.md",
        ]
        for path in candidates:
            try:
                data = _download_bytes(path)
                out["best_prompt"] = data.decode("utf-8", errors="replace")
                out["best_prompt_path"] = path
                break
            except Exception:
                continue
        # And the run summary
        for run_path in [
            f"{OPTIMIZER_RUNS_DIR}/{run_id}/{run_id}/run.json",
        ]:
            try:
                data = _download_bytes(run_path)
                obj = json.loads(data.decode("utf-8", errors="replace"))
                out["score"] = obj.get("best_score") or obj.get("optimized_full_score") or obj.get("optimized_full_eval_score")
                out["history"] = [h.get("score") for h in obj.get("history", [])] if obj.get("history") else None
                out["run_path"] = run_path
                break
            except Exception:
                continue

        # Register the best prompt as a new version in the UC Prompt Registry
        # so it's discoverable + reusable from outside the workbench. Only do
        # this once — guard by an idempotency tag check would be nicer but
        # mlflow.genai.register_prompt is happy to no-op when the template is
        # byte-identical to the latest version.
        if out.get("best_prompt"):
            tags = {
                "vlmwb_role": "best",
                "optimize_run_id": run_id,
            }
            if out.get("score") is not None:
                tags["score"] = f"{float(out['score']):.4f}"
            prompt_info = _register_genai_prompt(
                str(out["best_prompt"]),
                commit_message=f"GEPA best (round {out.get('history') and len(out['history']) - 1}, "
                               f"score {out.get('score')}) — run {run_id}",
                tags=tags,
            )
            if prompt_info:
                out["registered_prompt"] = prompt_info

    return out


# ── Snapshots (Playground configs frozen + recallable) ──────────────────

class SnapshotIn(BaseModel):
    name: str
    notes: Optional[str] = None
    frame_paths: List[str]
    model_names: List[str]
    prompt: str
    best_model: Optional[str] = None
    results: List[dict]
    elapsed_s: Optional[float] = None


class SnapshotSummary(BaseModel):
    id: str
    name: str
    n_frames: int
    n_models: int
    best_model: Optional[str]
    created_at: str
    created_by: Optional[str]


_SNAPSHOTS_TABLE_READY = False


def _ensure_snapshots_table():
    """Lazily create the snapshots table on first save. Idempotent."""
    global _SNAPSHOTS_TABLE_READY
    if _SNAPSHOTS_TABLE_READY:
        return
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name         text NOT NULL,
                notes        text,
                frame_paths  text[] NOT NULL,
                model_names  text[] NOT NULL,
                prompt       text NOT NULL,
                best_model   text,
                results      jsonb NOT NULL DEFAULT '[]'::jsonb,
                elapsed_s    double precision,
                created_by   text,
                created_at   timestamptz NOT NULL DEFAULT now()
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS snapshots_created_at_idx "
            "ON snapshots (created_at DESC)"
        )
    _SNAPSHOTS_TABLE_READY = True


@app.post("/api/playground/snapshots", response_model=dict)
def save_snapshot(snap: SnapshotIn):
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured — snapshots require Postgres")
    _ensure_snapshots_table()
    created_by = os.environ.get("DATABRICKS_USER_EMAIL") or "vlm-workbench"
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO snapshots (name, notes, frame_paths, model_names, prompt,
                                       best_model, results, elapsed_s, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id, created_at
            """, (
                snap.name, snap.notes, snap.frame_paths, snap.model_names, snap.prompt,
                snap.best_model, json.dumps(snap.results), snap.elapsed_s, created_by,
            ))
            row = cur.fetchone()
            return {"id": str(row[0]), "created_at": row[1].isoformat()}
    except Exception as e:
        log.error(f"save_snapshot failed: {e}")
        raise HTTPException(500, f"Failed to save snapshot: {e}")


@app.get("/api/playground/snapshots", response_model=List[SnapshotSummary])
def list_snapshots(limit: int = 30):
    if not _lakebase_available():
        return []
    _ensure_snapshots_table()
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, array_length(frame_paths, 1), array_length(model_names, 1),
                       best_model, created_at, created_by
                FROM snapshots
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            return [
                {
                    "id": str(r[0]), "name": r[1],
                    "n_frames": int(r[2] or 0), "n_models": int(r[3] or 0),
                    "best_model": r[4],
                    "created_at": r[5].isoformat(),
                    "created_by": r[6],
                }
                for r in cur.fetchall()
            ]
    except Exception as e:
        log.warning(f"list_snapshots failed: {e}")
        return []


@app.get("/api/playground/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    if not _lakebase_available():
        raise HTTPException(404, "no snapshot store")
    _ensure_snapshots_table()
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, notes, frame_paths, model_names, prompt,
                       best_model, results, elapsed_s, created_at, created_by
                FROM snapshots WHERE id = %s
            """, (snapshot_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "snapshot not found")
            return {
                "id": str(row[0]),
                "name": row[1], "notes": row[2],
                "frame_paths": row[3], "model_names": row[4],
                "prompt": row[5], "best_model": row[6],
                "results": row[7], "elapsed_s": row[8],
                "created_at": row[9].isoformat(),
                "created_by": row[10],
            }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"get_snapshot failed: {e}")
        raise HTTPException(500, f"Failed to fetch snapshot: {e}")


@app.delete("/api/playground/snapshots/{snapshot_id}")
def delete_snapshot(snapshot_id: str):
    if not _lakebase_available():
        raise HTTPException(404, "no snapshot store")
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM snapshots WHERE id = %s", (snapshot_id,))
            return {"deleted": cur.rowcount}
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")


# ── Ground-truth frame labels ───────────────────────────────────────────

_LABELS_TABLE_READY = False


def _ensure_labels_table():
    global _LABELS_TABLE_READY
    if _LABELS_TABLE_READY:
        return
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frame_labels (
                frame_path       text PRIMARY KEY,
                instrument       text NOT NULL,
                anatomy          text,
                tissue_condition text,
                notes            text,
                source           text NOT NULL DEFAULT 'manual',
                labeled_by       text,
                created_at       timestamptz NOT NULL DEFAULT now(),
                updated_at       timestamptz NOT NULL DEFAULT now()
            )
        """)
    _LABELS_TABLE_READY = True


class FrameLabelIn(BaseModel):
    frame_path: str
    instrument: str
    anatomy: Optional[str] = None
    tissue_condition: Optional[str] = None
    notes: Optional[str] = None
    source: str = "manual"  # "manual" | "snapshot:<id>:<model>" | "import"


class FrameLabelRow(BaseModel):
    frame_path: str
    instrument: str
    anatomy: Optional[str] = None
    tissue_condition: Optional[str] = None
    notes: Optional[str] = None
    source: str
    labeled_by: Optional[str]
    created_at: str
    updated_at: str


def _row_to_label(row) -> Dict[str, object]:
    return {
        "frame_path": row[0],
        "instrument": row[1],
        "anatomy": row[2],
        "tissue_condition": row[3],
        "notes": row[4],
        "source": row[5],
        "labeled_by": row[6],
        "created_at": row[7].isoformat() if row[7] else "",
        "updated_at": row[8].isoformat() if row[8] else "",
    }


@app.get("/api/labels")
def list_labels(frame_paths: Optional[str] = None, limit: int = 500):
    """Return labels. If ``frame_paths`` is provided (comma-separated), only
    those frames; otherwise the most recent ``limit`` labels."""
    if not _lakebase_available():
        return []
    _ensure_labels_table()
    paths = [p for p in (frame_paths or "").split(",") if p.strip()]
    with _pg_conn() as conn, conn.cursor() as cur:
        if paths:
            cur.execute(
                "SELECT frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by, created_at, updated_at "
                "FROM frame_labels WHERE frame_path = ANY(%s)",
                (paths,),
            )
        else:
            cur.execute(
                "SELECT frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by, created_at, updated_at "
                "FROM frame_labels ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()
    return [_row_to_label(r) for r in rows]


@app.post("/api/labels")
def upsert_label(body: FrameLabelIn):
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _check_allowed_path(body.frame_path)
    if not body.instrument.strip():
        raise HTTPException(400, "instrument is required")
    _ensure_labels_table()
    labeled_by = os.environ.get("DATABRICKS_USER_EMAIL") or _ws_user_email() or "vlm-workbench"
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO frame_labels (frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (frame_path) DO UPDATE SET
                instrument = EXCLUDED.instrument,
                anatomy = EXCLUDED.anatomy,
                tissue_condition = EXCLUDED.tissue_condition,
                notes = EXCLUDED.notes,
                source = EXCLUDED.source,
                labeled_by = EXCLUDED.labeled_by,
                updated_at = now()
            RETURNING frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by, created_at, updated_at
            """,
            (
                body.frame_path, body.instrument.strip(),
                (body.anatomy or "").strip() or None,
                (body.tissue_condition or "").strip() or None,
                (body.notes or "").strip() or None,
                body.source,
                labeled_by,
            ),
        )
        row = cur.fetchone()
    return _row_to_label(row)


class FrameLabelsBatchIn(BaseModel):
    rows: List[FrameLabelIn]


@app.post("/api/labels/batch")
def upsert_labels_batch(body: FrameLabelsBatchIn):
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    if not body.rows:
        return {"inserted": 0, "rows": []}
    _ensure_labels_table()
    labeled_by = os.environ.get("DATABRICKS_USER_EMAIL") or _ws_user_email() or "vlm-workbench"
    out = []
    with _pg_conn() as conn, conn.cursor() as cur:
        for r in body.rows:
            _check_allowed_path(r.frame_path)
            if not r.instrument.strip():
                continue
            cur.execute(
                """
                INSERT INTO frame_labels (frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (frame_path) DO UPDATE SET
                    instrument = EXCLUDED.instrument,
                    anatomy = EXCLUDED.anatomy,
                    tissue_condition = EXCLUDED.tissue_condition,
                    notes = EXCLUDED.notes,
                    source = EXCLUDED.source,
                    labeled_by = EXCLUDED.labeled_by,
                    updated_at = now()
                RETURNING frame_path, instrument, anatomy, tissue_condition, notes, source, labeled_by, created_at, updated_at
                """,
                (
                    r.frame_path, r.instrument.strip(),
                    (r.anatomy or "").strip() or None,
                    (r.tissue_condition or "").strip() or None,
                    (r.notes or "").strip() or None,
                    r.source,
                    labeled_by,
                ),
            )
            out.append(_row_to_label(cur.fetchone()))
    return {"inserted": len(out), "rows": out}


@app.delete("/api/labels")
def delete_label(frame_path: str):
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _ensure_labels_table()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM frame_labels WHERE frame_path = %s", (frame_path,))
        return {"deleted": cur.rowcount}


DELTA_LABELS_TABLE = os.environ.get(
    "DELTA_LABELS_TABLE",
    f"{UC_CATALOG}.{UC_SCHEMA}.frame_labels_delta",
)


def _execute_sql(query: str, params: Optional[List[object]] = None) -> Dict[str, object]:
    """Execute a SQL statement via the configured SQL warehouse. Returns the
    statement-execution response so callers can inspect rowcount/status."""
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID not set")
    w = _workspace()
    body: Dict[str, object] = {
        "warehouse_id": warehouse_id,
        "statement": query,
        "wait_timeout": "30s",
    }
    if params:
        body["parameters"] = [{"value": p} for p in params]
    return w.api_client.do("POST", "/api/2.0/sql/statements", body=body)


@app.post("/api/labels/sync-to-delta")
def sync_labels_to_delta():
    """Mirror every row in Lakebase frame_labels into a Delta table in UC.
    Idempotent — uses CREATE OR REPLACE for the snapshot; the Delta table is
    a frozen view of the labels at sync time. Designed to be cheap to call
    after every labeling session and to power downstream Spark/training jobs."""
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _ensure_labels_table()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT frame_path, instrument, anatomy, tissue_condition, notes,
                   source, labeled_by, created_at, updated_at
            FROM frame_labels
            ORDER BY updated_at DESC
        """)
        rows = cur.fetchall()
    if not rows:
        # Still create an empty table so downstream consumers don't 404 on the
        # first sync; just create + truncate.
        try:
            _execute_sql(f"""
                CREATE TABLE IF NOT EXISTS {DELTA_LABELS_TABLE} (
                    frame_path STRING NOT NULL,
                    instrument STRING NOT NULL,
                    anatomy STRING,
                    tissue_condition STRING,
                    notes STRING,
                    source STRING NOT NULL,
                    labeled_by STRING,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    synced_at TIMESTAMP NOT NULL
                ) USING DELTA
            """)
            _execute_sql(f"DELETE FROM {DELTA_LABELS_TABLE}")
        except Exception as e:
            raise HTTPException(500, f"create empty delta failed: {e}")
        return {"rows_synced": 0, "delta_table": DELTA_LABELS_TABLE}

    # Build a single multi-row INSERT. For up to a few thousand labels this is
    # well under the SQL statement limit and avoids the cost of repeated
    # round-trips. If the table grows past ~10k rows we'd switch to staging
    # the rows in a Volume CSV + COPY INTO.
    try:
        _execute_sql(f"""
            CREATE OR REPLACE TABLE {DELTA_LABELS_TABLE} (
                frame_path STRING NOT NULL,
                instrument STRING NOT NULL,
                anatomy STRING,
                tissue_condition STRING,
                notes STRING,
                source STRING NOT NULL,
                labeled_by STRING,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                synced_at TIMESTAMP NOT NULL
            ) USING DELTA
        """)
    except Exception as e:
        raise HTTPException(500, f"create delta table failed: {e}")

    def _sql_lit(v):
        if v is None:
            return "NULL"
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        # datetime → ISO string
        return "'" + str(v).replace("'", "''") + "'"

    values_clauses = []
    for r in rows:
        frame_path, instrument, anatomy, tissue, notes, source, labeled_by, created_at, updated_at = r
        values_clauses.append(
            f"({_sql_lit(frame_path)}, {_sql_lit(instrument)}, {_sql_lit(anatomy)}, "
            f"{_sql_lit(tissue)}, {_sql_lit(notes)}, {_sql_lit(source)}, "
            f"{_sql_lit(labeled_by)}, {_sql_lit(created_at)}, {_sql_lit(updated_at)}, "
            f"current_timestamp())"
        )

    # Insert in batches of 500 to stay well clear of statement size limits.
    BATCH = 500
    inserted = 0
    for i in range(0, len(values_clauses), BATCH):
        chunk = values_clauses[i:i + BATCH]
        try:
            _execute_sql(
                f"INSERT INTO {DELTA_LABELS_TABLE} "
                f"(frame_path, instrument, anatomy, tissue_condition, notes, source, "
                f"labeled_by, created_at, updated_at, synced_at) VALUES "
                + ",".join(chunk)
            )
        except Exception as e:
            raise HTTPException(500, f"insert batch [{i}:{i+len(chunk)}] failed: {e}")
        inserted += len(chunk)
    # Also push to a UC-managed MLflow GenAI dataset so the labels become
    # discoverable in the GenAI experiment UI ("Datasets" tab) and usable via
    # mlflow.genai.evaluate(). This is best-effort — never blocks the Delta
    # sync from succeeding.
    try:
        ds_uri = _sync_labels_to_genai_dataset()
        return {
            "rows_synced": inserted,
            "delta_table": DELTA_LABELS_TABLE,
            "genai_dataset": ds_uri,
        }
    except Exception as e:
        log.warning(f"[labels-sync] mlflow dataset sync failed (non-fatal): {e}")
        return {"rows_synced": inserted, "delta_table": DELTA_LABELS_TABLE}


GENAI_DATASET_NAME = os.environ.get(
    "GENAI_DATASET_NAME",
    f"{UC_CATALOG}.{UC_SCHEMA}.vlmwb_gold_labels",
)


def _sync_labels_to_genai_dataset() -> str:
    """Push every Lakebase frame_labels row into a UC-managed MLflow GenAI
    evaluation dataset. Idempotent — uses merge_records keyed by frame_path.

    Returns the dataset name (which is also its URI in MLflow's prompt-style
    notation: ``mlflow.genai.datasets.get_dataset(<name>)``)."""
    _ensure_labels_table()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT frame_path, instrument, anatomy, tissue_condition
            FROM frame_labels
        """)
        rows = cur.fetchall()
    if not rows:
        return GENAI_DATASET_NAME

    with _mlflow_auth_scope():
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        from mlflow.genai import datasets as gds
        try:
            ds = gds.create_dataset(name=GENAI_DATASET_NAME)
        except Exception:
            # Already exists — fetch it
            ds = gds.get_dataset(GENAI_DATASET_NAME)
        # MLflow GenAI dataset rows: {"inputs": {...}, "expectations": {...}}.
        records = [
            {
                "inputs": {"image_path": r[0]},
                "expectations": {
                    "instrument": r[1],
                    "anatomy": r[2] or "",
                    "tissue_condition": r[3] or "",
                },
            }
            for r in rows
        ]
        ds.merge_records(records)
        log.info(f"[labels-sync] merged {len(records)} rows into {GENAI_DATASET_NAME}")
        return GENAI_DATASET_NAME


# ── MLflow Prompt Registry (UC-managed prompts) ──────────────────────────

GENAI_PROMPT_NAME = os.environ.get(
    "GENAI_PROMPT_NAME",
    f"{UC_CATALOG}.{UC_SCHEMA}.vlmwb_instrument_id",
)


class _mlflow_auth_scope:
    """Context manager that swaps env vars to use a minted bearer token (PAT
    path) only inside the block. Outside the block, the OAuth creds remain
    the active auth — required because WorkspaceClient refuses to start with
    both OAuth + PAT configured ("more than one authorization method").

    Why we need this dance: mlflow.genai's REST client takes the "OAuth"
    code path when DATABRICKS_CLIENT_ID is set, and that path raises
    "set MLFLOW_ENABLE_DB_SDK to true" even with the flag set (some
    string-check quirk we couldn't satisfy). Using a PAT sidesteps the
    flag entirely, but globally setting DATABRICKS_TOKEN breaks every other
    SDK consumer in the app. So: scope it."""
    def __init__(self):
        self._saved_id = None
        self._saved_secret = None
        self._saved_token = None
        self._set = False

    def __enter__(self):
        tok = _lakebase_token()
        if not tok:
            return self
        self._saved_id = os.environ.pop("DATABRICKS_CLIENT_ID", None)
        self._saved_secret = os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
        self._saved_token = os.environ.get("DATABRICKS_TOKEN")
        os.environ["DATABRICKS_TOKEN"] = tok
        self._set = True
        return self

    def __exit__(self, *_):
        if not self._set:
            return
        if self._saved_id is not None:
            os.environ["DATABRICKS_CLIENT_ID"] = self._saved_id
        if self._saved_secret is not None:
            os.environ["DATABRICKS_CLIENT_SECRET"] = self._saved_secret
        if self._saved_token is None:
            os.environ.pop("DATABRICKS_TOKEN", None)
        else:
            os.environ["DATABRICKS_TOKEN"] = self._saved_token


def _prime_mlflow_auth():
    """Deprecated alias — kept for the existing call sites until they migrate."""
    pass


def _register_genai_prompt(template: str, commit_message: str,
                           tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, object]]:
    """Register a new version of the workbench's prompt. Returns
    {name, version, uri} on success, None on failure (never raises)."""
    try:
        with _mlflow_auth_scope():
            import mlflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
            p = mlflow.genai.register_prompt(
                name=GENAI_PROMPT_NAME,
                template=template,
                commit_message=commit_message[:240],
                tags=tags or {},
            )
            return {
                "name": getattr(p, "name", GENAI_PROMPT_NAME),
                "version": getattr(p, "version", None),
                "uri": getattr(p, "uri", None),
            }
    except Exception as e:
        log.warning(f"[prompt-registry] register failed (non-fatal): {e}")
        return None


class RegisterPromptRequest(BaseModel):
    template: str
    commit_message: str = ""
    tags: Optional[Dict[str, str]] = None


@app.post("/api/mlflow/prompts/register")
def register_prompt_endpoint(req: RegisterPromptRequest):
    """Manually register a prompt version (used by the UI's prompt-versioning
    flow). The Optimize submit + complete hooks also call _register_genai_prompt
    automatically."""
    out = _register_genai_prompt(req.template, req.commit_message or "manual", req.tags)
    if not out:
        raise HTTPException(500, "registration failed — see app logs")
    return out


@app.get("/api/mlflow/prompts/{name}")
def list_prompt_versions(name: str, limit: int = 20):
    """List versions of a registered prompt by full UC name (or short name —
    we'll prefix it with the configured catalog.schema if it's not three-part)."""
    if name.count(".") < 2:
        name = f"{UC_CATALOG}.{UC_SCHEMA}.{name}"
    try:
        with _mlflow_auth_scope():
            import mlflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            # UC prompt registries don't accept search_prompts with name=
            # filter; they want catalog/schema separately. Easier path: just
            # try loading versions 1..N until we hit "not found".
            out = []
            for v in range(1, limit + 1):
                try:
                    p = mlflow.genai.load_prompt(f"prompts:/{name}/{v}")
                    out.append({
                        "version": p.version,
                        "template_preview": p.template[:200],
                        "uri": f"prompts:/{name}/{v}",
                    })
                except Exception:
                    break
            return {"name": name, "versions": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"prompt lookup failed: {e}")


@app.get("/api/mlflow/datasets/labels-summary")
def genai_dataset_summary():
    """Compact info about the labels dataset (name + row count)."""
    try:
        with _mlflow_auth_scope():
            import mlflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
            from mlflow.genai import datasets as gds
            try:
                ds = gds.get_dataset(GENAI_DATASET_NAME)
            except Exception:
                return {"name": GENAI_DATASET_NAME, "exists": False, "n_records": 0}
            # Different SDK versions expose count differently — best effort
            n = None
            for attr in ("num_records", "n_records", "size"):
                if hasattr(ds, attr):
                    try:
                        n = int(getattr(ds, attr))
                        break
                    except Exception:
                        pass
            return {"name": GENAI_DATASET_NAME, "exists": True, "n_records": n}
    except Exception as e:
        return {"name": GENAI_DATASET_NAME, "exists": False, "error": str(e)[:200]}


@app.get("/api/labels/stats")
def labels_stats():
    if not _lakebase_available():
        return {"total": 0, "by_instrument": []}
    _ensure_labels_table()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM frame_labels")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT instrument, count(*) FROM frame_labels "
            "GROUP BY instrument ORDER BY count(*) DESC"
        )
        by_inst = [{"instrument": r[0], "n": r[1]} for r in cur.fetchall()]
    return {"total": total, "by_instrument": by_inst}


# ── Auto-ingest pipeline ────────────────────────────────────────────────
#
# Drop a video into `/Volumes/.../videos/inbox/`, hit POST /api/videos/ingest
# (or rely on the file-arrival Job we register once), and the smart-frame
# extractor runs on a GPU job. Frames land in
# `/Volumes/.../extracted_frames/<video_stem>/` and metadata is registered in
# the Lakebase `videos` + `extracted_frames_index` tables so the app can list
# them instantly without scanning the Volume.

VIDEOS_INBOX_DIR = os.environ.get("VIDEOS_INBOX_DIR", f"{VOLUME_PATH}/videos/inbox")
IMAGES_INBOX_DIR = os.environ.get("IMAGES_INBOX_DIR", f"{VOLUME_PATH}/images")
EXTRACTED_FRAMES_DIR_DEFAULT = os.environ.get(
    "EXTRACTED_FRAMES_DIR", EXTRACTED_FRAMES_DIR,
)

_INGEST_TABLES_READY = False


def _ensure_ingest_tables():
    """Lazily create the videos + extracted_frames_index tables. Adds the
    ``kind`` column on existing tables so older deploys keep working."""
    global _INGEST_TABLES_READY
    if _INGEST_TABLES_READY:
        return
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name          text NOT NULL UNIQUE,
                path          text NOT NULL,
                kind          text NOT NULL DEFAULT 'video',  -- 'video' | 'image_batch'
                size_bytes    bigint,
                duration_s    double precision,
                status        text NOT NULL DEFAULT 'pending',
                status_message text,
                n_frames_extracted int,
                ingested_by   text,
                created_at    timestamptz NOT NULL DEFAULT now(),
                updated_at    timestamptz NOT NULL DEFAULT now()
            )
        """)
        # Add kind column on pre-existing tables that don't have it yet.
        cur.execute("""
            ALTER TABLE videos ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'video'
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS extracted_frames_index (
                frame_path   text PRIMARY KEY,
                video_id     uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                frame_name   text NOT NULL,
                timestamp_s  double precision NOT NULL,
                score        double precision,
                sharpness    double precision,
                colorfulness double precision,
                contrast     double precision,
                created_at   timestamptz NOT NULL DEFAULT now(),
                updated_at   timestamptz NOT NULL DEFAULT now()
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS extracted_frames_video_idx "
            "ON extracted_frames_index (video_id, timestamp_s)"
        )
    _INGEST_TABLES_READY = True


def _resolve_ingest_notebook() -> str:
    cache_key = "ingest_smart_frames"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ingest_notebooks", "smart_frames",
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        raise RuntimeError(f"ingest notebook missing in {base_dir}")
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/ingest_notebooks/smart_frames"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


def _resolve_setup_cache_notebook() -> str:
    """Mirror of _resolve_ingest_notebook for the setup_cache notebook.
    Imports the source under the App SP's home so the SP can run it."""
    cache_key = "setup_cache"
    if cache_key in _NOTEBOOK_CACHE:
        return _NOTEBOOK_CACHE[cache_key]
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local_inference_notebooks", "setup_cache",
    )
    candidates = [os.path.join(base_dir, "notebook.py"), os.path.join(base_dir, "notebook")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if not src_path:
        raise RuntimeError(f"setup_cache notebook missing in {base_dir}")
    with open(src_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, Language
    w = WorkspaceClient()
    target_dir = f"{_sp_home()}/local_inference_notebooks/setup_cache"
    target_path = f"{target_dir}/notebook"
    w.workspace.mkdirs(target_dir)
    w.workspace.import_(
        path=target_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=content_b64,
        overwrite=True,
    )
    _NOTEBOOK_CACHE[cache_key] = target_path
    return target_path


def _list_inbox_videos() -> List[Dict[str, object]]:
    """List MP4 files in the inbox via the Files API. Returns name + path + size."""
    out: List[Dict[str, object]] = []
    for entry in _ls_dir(VIDEOS_INBOX_DIR):
        name = entry.get("name") or ""
        if entry.get("is_directory") or not name.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        out.append({
            "name": name,
            "path": entry.get("path") or f"{VIDEOS_INBOX_DIR}/{name}",
            "size_bytes": int(entry.get("size_bytes") or 0),
            "kind": "video",
        })
    return out


def _list_image_batches() -> List[Dict[str, object]]:
    """List top-level directories under IMAGES_INBOX_DIR, plus a sentinel for
    images dropped at the root. Each directory is a "batch" — the equivalent of
    a video for ingest purposes. Returns name + path + count + total size."""
    out: List[Dict[str, object]] = []
    if not VIDEOS_INBOX_DIR:
        return out
    root_files: List[Dict[str, object]] = []
    for entry in _ls_dir(IMAGES_INBOX_DIR):
        name = entry.get("name") or ""
        path = entry.get("path") or f"{IMAGES_INBOX_DIR}/{name}"
        if entry.get("is_directory"):
            # Enumerate images inside the batch dir
            n_images = 0
            total_size = 0
            try:
                for sub in _ls_dir(path):
                    sname = (sub.get("name") or "").lower()
                    if sname.endswith((".jpg", ".jpeg", ".png")):
                        n_images += 1
                        total_size += int(sub.get("size_bytes") or 0)
            except Exception:
                pass
            if n_images > 0:
                out.append({
                    "name": name,
                    "path": path,
                    "size_bytes": total_size,
                    "n_images": n_images,
                    "kind": "image_batch",
                })
        elif name.lower().endswith((".jpg", ".jpeg", ".png")):
            root_files.append({"name": name, "path": path, "size_bytes": int(entry.get("size_bytes") or 0)})
    # If there are loose images at the root, treat them as a single "uploads" batch
    if root_files:
        out.append({
            "name": "uploads",
            "path": IMAGES_INBOX_DIR,
            "size_bytes": sum(f["size_bytes"] for f in root_files),
            "n_images": len(root_files),
            "kind": "image_batch",
            "_root_files": root_files,
        })
    return out


class IngestRequest(BaseModel):
    video_name: Optional[str] = None  # if None, ingest every new video in inbox
    candidate_fps: float = 1.0
    max_frames: int = 40
    force: bool = False  # if True, re-ingest even if status=ready


class IngestResponse(BaseModel):
    submitted: List[Dict[str, object]]
    skipped: List[Dict[str, object]]


@app.get("/api/ingest/videos")
def list_ingest_videos():
    """Combined view: Lakebase-registered rows (both video + image_batch kinds)
    plus any inbox files / image dirs not yet registered. Powers the Library
    panel in the UI."""
    if not _lakebase_available():
        inbox_videos = _list_inbox_videos()
        inbox_images = _list_image_batches()
        merged = inbox_videos + inbox_images
        return {"videos": [{**v, "status": "pending", "id": None, "n_frames_extracted": None} for v in merged]}
    _ensure_ingest_tables()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id::text, name, path, kind, size_bytes, duration_s, status,
                   status_message, n_frames_extracted, created_at, updated_at
            FROM videos ORDER BY updated_at DESC
        """)
        rows = cur.fetchall()
    by_name: Dict[str, Dict[str, object]] = {}
    for r in rows:
        by_name[r[1]] = {
            "id": r[0], "name": r[1], "path": r[2], "kind": r[3],
            "size_bytes": r[4], "duration_s": r[5], "status": r[6],
            "status_message": r[7], "n_frames_extracted": r[8],
            "created_at": r[9].isoformat() if r[9] else None,
            "updated_at": r[10].isoformat() if r[10] else None,
        }
    # Merge inbox files (videos + image batches) that haven't been registered yet
    for v in _list_inbox_videos() + _list_image_batches():
        if v["name"] not in by_name:
            by_name[v["name"]] = {
                "id": None, "name": v["name"], "path": v["path"],
                "kind": v.get("kind", "video"),
                "size_bytes": v["size_bytes"], "duration_s": None,
                "status": "pending", "status_message": None,
                "n_frames_extracted": v.get("n_images"),
                "created_at": None, "updated_at": None,
            }
    return {"videos": list(by_name.values())}


# ── Library upload (videos + image batches) ─────────────────────────────
#
# Streams uploads from the browser straight into the Volume's inbox folders
# so users don't have to use the catalog explorer or CLI. The App SP needs
# WRITE VOLUME (granted by postdeploy.py).
_ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
_ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_basename(name: str) -> str:
    base = os.path.basename(name or "").strip()
    # Strip directory traversal + reserved chars; pathlib's PurePosixPath would
    # do it too but we want to be obvious.
    return "".join(c for c in base if c.isalnum() or c in "._- ") or "upload.bin"


@app.post("/api/library/upload")
async def library_upload(
    kind: str = Form(...),                 # "video" | "image_batch"
    batch_name: Optional[str] = Form(None), # required if kind == "image_batch"
    file: UploadFile = File(...),
):
    """Stream a single file into the appropriate Volume inbox.
    For image batches, all files in a batch must use the same `batch_name`
    so they land in the same subfolder.
    Multi-file uploads from the browser issue this endpoint once per file."""
    from databricks.sdk import WorkspaceClient
    fname = _safe_basename(file.filename or "upload")
    ext = os.path.splitext(fname)[1].lower()
    if kind == "video":
        if ext not in _ALLOWED_VIDEO_EXT:
            raise HTTPException(400, f"unsupported video extension {ext!r}; allowed: {sorted(_ALLOWED_VIDEO_EXT)}")
        dest_dir = VIDEOS_INBOX_DIR
    elif kind == "image_batch":
        if not (batch_name or "").strip():
            raise HTTPException(400, "batch_name required for image_batch uploads")
        if ext not in _ALLOWED_IMAGE_EXT:
            raise HTTPException(400, f"unsupported image extension {ext!r}; allowed: {sorted(_ALLOWED_IMAGE_EXT)}")
        # Reuse same sanitizer for the folder name
        safe_batch = "".join(c for c in batch_name.strip() if c.isalnum() or c in "._-") or "batch"
        dest_dir = f"{IMAGES_INBOX_DIR}/{safe_batch}"
    else:
        raise HTTPException(400, f"kind must be 'video' or 'image_batch', got {kind!r}")

    w = WorkspaceClient()
    try:
        w.files.create_directory(dest_dir)
    except Exception:
        pass  # already exists

    dest_path = f"{dest_dir}/{fname}"
    body = await file.read()
    if not body:
        raise HTTPException(400, "empty upload")
    try:
        w.files.upload(dest_path, body, overwrite=True)
    except Exception as e:
        log.exception("upload failed")
        raise HTTPException(500, f"upload failed: {e}")
    return {"path": dest_path, "size_bytes": len(body), "kind": kind}


@app.post("/api/videos/ingest", response_model=IngestResponse)
def kick_off_ingest(req: IngestRequest):
    """Submit the smart-frame extractor for one (or all pending) video(s).
    Each video runs as its own serverless GPU-free Databricks job — the
    extractor is CPU-only, so we keep cost low and just use base env_v5."""
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _ensure_ingest_tables()

    inbox = {v["name"]: v for v in _list_inbox_videos()}
    if req.video_name:
        if req.video_name not in inbox:
            raise HTTPException(404, f"video '{req.video_name}' not in inbox ({VIDEOS_INBOX_DIR})")
        targets = [inbox[req.video_name]]
    else:
        targets = list(inbox.values())

    if not targets:
        raise HTTPException(404, f"no videos in inbox ({VIDEOS_INBOX_DIR})")

    submitted: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    # Mint a fresh Lakebase token to hand to the notebook so it can connect
    # back to Postgres as the SP. (The notebook's exec context doesn't share
    # the app's env vars.)
    #
    # SECURITY: do NOT include the Postgres password in the config we write
    # to the Volume — anyone with READ VOLUME could read it. The notebook
    # mints its own token from inside, using the SP identity it runs as.
    pg_user = _pg_user()
    if not pg_user:
        raise HTTPException(500, "could not resolve Lakebase user for the ingest job")
    # Endpoint paths for the notebook's mint step. We pass both styles so the
    # notebook auto-detects Project vs legacy Provisioned at runtime.
    lb_instance = os.environ.get("PGDATABASE_INSTANCE") or os.environ.get("LAKEBASE_INSTANCE") or ""
    lb_branch = os.environ.get("LAKEBASE_BRANCH", "production")
    # `PGAPPNAME` is set when the postgres: binding is active; from it we can
    # infer the endpoint. Otherwise we leave pg_endpoint empty and rely on
    # pg_instance for Provisioned deploys.
    pg_endpoint = ""
    pg_instance = ""
    # If LAKEBASE_HOST looks like a Project endpoint host, use the var path
    if "ep-" in LAKEBASE_HOST and lb_instance:
        pg_endpoint = f"projects/{lb_instance}/branches/{lb_branch}/endpoints/primary"
    else:
        pg_instance = lb_instance

    notebook_path = _resolve_ingest_notebook()
    w = _workspace()

    for tgt in targets:
        name = tgt["name"]
        with _pg_conn() as conn, conn.cursor() as cur:
            # Reserve a videos row (or reuse existing) — we need a stable
            # video_id BEFORE submitting the job so the notebook can update
            # status as it progresses.
            cur.execute("""
                INSERT INTO videos (name, path, size_bytes, ingested_by, status)
                VALUES (%s, %s, %s, %s, 'queued')
                ON CONFLICT (name) DO UPDATE SET
                    path = EXCLUDED.path,
                    size_bytes = EXCLUDED.size_bytes,
                    status = CASE
                        WHEN videos.status IN ('ready', 'processing') AND NOT %s THEN videos.status
                        ELSE 'queued'
                    END,
                    status_message = NULL,
                    updated_at = now()
                RETURNING id::text, status
            """, (name, tgt["path"], tgt["size_bytes"],
                  os.environ.get("DATABRICKS_USER_EMAIL") or "vlm-workbench",
                  bool(req.force)))
            video_id, status = cur.fetchone()

        if status == "ready" and not req.force:
            skipped.append({"name": name, "reason": "already ingested", "video_id": video_id})
            continue

        video_stem = os.path.splitext(name)[0]
        output_dir = f"{EXTRACTED_FRAMES_DIR_DEFAULT}/{video_stem}"
        ingest_cfg = {
            "video_path": tgt["path"],
            "video_name": name,
            "video_id": video_id,
            "output_dir": output_dir,
            "pg_host": LAKEBASE_HOST,
            "pg_port": LAKEBASE_PORT,
            "pg_dbname": LAKEBASE_DBNAME,
            "pg_user": pg_user,
            # No `pg_password` — the notebook mints its own at runtime via
            # /api/2.0/postgres/credentials using its SP identity. We pass
            # the endpoint/instance so it knows which to ask for.
            "pg_endpoint": pg_endpoint,
            "pg_instance": pg_instance,
            "candidate_fps": req.candidate_fps,
            "max_frames": req.max_frames,
        }
        import yaml
        ingest_yaml = yaml.safe_dump(ingest_cfg, sort_keys=False, allow_unicode=True)
        yaml_path = f"{VOLUME_PATH}/ingest_runs/{video_id}.yaml"
        try:
            w.files.create_directory(f"{VOLUME_PATH}/ingest_runs")
        except Exception:
            pass
        try:
            w.files.upload(yaml_path, contents=io.BytesIO(ingest_yaml.encode("utf-8")), overwrite=True)
        except Exception as e:
            with _pg_conn() as conn, conn.cursor() as cur:
                cur.execute("UPDATE videos SET status='error', status_message=%s WHERE id=%s",
                            (f"failed to write yaml: {e}"[:300], video_id))
            skipped.append({"name": name, "reason": f"yaml upload failed: {e}", "video_id": video_id})
            continue

        body = {
            "run_name": f"vlmwb_ingest_{video_stem}",
            "tasks": [{
                "task_key": "ingest",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "source": "WORKSPACE",
                    "base_parameters": {"config_yaml": yaml_path},
                },
                "environment_key": "cpu_env",
            }],
            "queue": {"enabled": True},
            "environments": [{
                "environment_key": "cpu_env",
                "spec": {"environment_version": "5"},
            }],
            "performance_target": "PERFORMANCE_OPTIMIZED",
        }
        try:
            resp = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=body)
            databricks_run_id = resp.get("run_id")
        except Exception as e:
            with _pg_conn() as conn, conn.cursor() as cur:
                cur.execute("UPDATE videos SET status='error', status_message=%s WHERE id=%s",
                            (f"job submit failed: {e}"[:300], video_id))
            skipped.append({"name": name, "reason": f"job submit failed: {e}", "video_id": video_id})
            continue

        log.info(f"[ingest] submitted run_id={databricks_run_id} video={name}")
        submitted.append({
            "name": name,
            "video_id": video_id,
            "databricks_run_id": str(databricks_run_id),
            "yaml_path": yaml_path,
        })

    return IngestResponse(submitted=submitted, skipped=skipped)


class IngestImagesRequest(BaseModel):
    batch_name: Optional[str] = None  # if None, ingest every batch
    force: bool = False


@app.post("/api/images/ingest")
def kick_off_image_ingest(req: IngestImagesRequest):
    """Register an image batch directly in Lakebase — no GPU job, no smart-
    frame extraction. Each image becomes a frame row pointing at its existing
    path, so Playground's "Smart-extracted" source picks it up immediately."""
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _ensure_ingest_tables()

    inbox = {b["name"]: b for b in _list_image_batches()}
    if req.batch_name:
        if req.batch_name not in inbox:
            raise HTTPException(404, f"image batch '{req.batch_name}' not found in {IMAGES_INBOX_DIR}")
        targets = [inbox[req.batch_name]]
    else:
        targets = list(inbox.values())
    if not targets:
        raise HTTPException(404, f"no image batches in {IMAGES_INBOX_DIR}")

    registered: List[Dict[str, object]] = []
    for batch in targets:
        name = batch["name"]
        # Reserve / update the videos row
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO videos (name, path, kind, size_bytes, ingested_by, status)
                VALUES (%s, %s, 'image_batch', %s, %s, 'processing')
                ON CONFLICT (name) DO UPDATE SET
                    path = EXCLUDED.path,
                    kind = 'image_batch',
                    size_bytes = EXCLUDED.size_bytes,
                    status = 'processing',
                    status_message = NULL,
                    updated_at = now()
                RETURNING id::text
            """, (name, batch["path"], int(batch.get("size_bytes") or 0),
                  os.environ.get("DATABRICKS_USER_EMAIL") or "vlm-workbench"))
            video_id = cur.fetchone()[0]

        # Enumerate images. Either a real subdir of IMAGES_INBOX_DIR, or the
        # synthetic "uploads" batch which already carries _root_files.
        image_entries = batch.get("_root_files")
        if image_entries is None:
            image_entries = []
            for sub in _ls_dir(batch["path"]):
                sname = (sub.get("name") or "").lower()
                if sname.endswith((".jpg", ".jpeg", ".png")):
                    image_entries.append({
                        "name": sub.get("name"),
                        "path": sub.get("path") or f"{batch['path']}/{sub.get('name')}",
                        "size_bytes": int(sub.get("size_bytes") or 0),
                    })

        # Register each image as a frame. timestamp_s = index in sorted order
        # so the Library/Playground UIs can still sort consistently.
        image_entries.sort(key=lambda x: x.get("name") or "")
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM extracted_frames_index WHERE video_id = %s", (video_id,))
            for i, img in enumerate(image_entries):
                cur.execute("""
                    INSERT INTO extracted_frames_index
                        (video_id, frame_path, frame_name, timestamp_s, score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (frame_path) DO UPDATE SET
                        video_id = EXCLUDED.video_id,
                        timestamp_s = EXCLUDED.timestamp_s,
                        updated_at = now()
                """, (video_id, img["path"], img["name"], float(i), 1.0))
            cur.execute("""
                UPDATE videos SET status='ready',
                    status_message=%s,
                    n_frames_extracted=%s,
                    updated_at=now()
                WHERE id=%s
            """, (f"registered {len(image_entries)} images", len(image_entries), video_id))
        registered.append({
            "name": name, "video_id": video_id, "n_images": len(image_entries),
        })
    return {"registered": registered}


@app.delete("/api/videos/{video_id}")
def delete_video_index(video_id: str):
    """Remove a video and its extracted frames index from Lakebase. Does NOT
    delete the JPG files in the Volume — they're cheap to keep around."""
    if not _lakebase_available():
        raise HTTPException(503, "Lakebase not configured")
    _ensure_ingest_tables()
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM videos WHERE id = %s", (video_id,))
        return {"deleted": cur.rowcount}


# ── Video streaming (Volume → range-supported HTTP) ─────────────────────

@app.get("/api/videos/{video_name}/stream")
def stream_video(video_name: str, request: Request):
    """Stream a video from the Volume with HTTP range support so the browser
    can seek without downloading the whole file."""
    if "/" in video_name or ".." in video_name:
        raise HTTPException(400, "Invalid video name")
    abs_path = os.path.join(VOLUME_PATH, video_name)
    _check_allowed_path(abs_path)

    # Get full size via metadata
    w = _workspace()
    try:
        meta = w.files.get_metadata(abs_path)
        size = int(getattr(meta, "content_length", 0) or 0)
    except Exception as e:
        raise HTTPException(404, f"Video not found: {e}")
    # Force video/mp4 for .mp4 files — Files API metadata returns
    # application/octet-stream which browsers refuse to play.
    ctype = "video/mp4" if video_name.lower().endswith(".mp4") else (
        getattr(meta, "content_type", None) or "application/octet-stream"
    )

    range_header = request.headers.get("range") or request.headers.get("Range")
    start, end = 0, size - 1 if size else None
    if range_header and size:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
    chunk_len = (end - start + 1) if size else None

    # Files API doesn't support partial reads directly via download(), so we
    # download the whole file but only return the requested byte range.
    # For typical 50-100MB videos this is fine on the app's network.
    data = _download_bytes(abs_path)
    if size and (start > 0 or end != size - 1):
        body = data[start:end + 1]
        status = 206
        headers = {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(body)),
        }
    else:
        body = data
        status = 200
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(body)),
        }
    return Response(content=body, status_code=status, media_type=ctype, headers=headers)


# ── Studio: visual analysis pipeline ─────────────────────────────────────

class StudioAnalyzeRequest(BaseModel):
    video_name: str
    per_frame_model: Optional[str] = None
    validator_model: Optional[str] = None
    prompt: Optional[str] = None         # if absent, app uses the default GEPA prompt
    snapshot_id: Optional[str] = None    # when present: override model+prompt from this snapshot
    max_frames: int = STUDIO_MAX_FRAMES
    force: bool = False  # ignore cache


def _resolve_studio_config(req: "StudioAnalyzeRequest") -> tuple[str, str, str]:
    """Decide per-frame model + validator + prompt for this analyze call.
    Snapshot wins, then explicit request fields, then defaults."""
    per_frame = req.per_frame_model or STUDIO_PER_FRAME_MODEL
    validator = req.validator_model or STUDIO_VALIDATOR_MODEL
    prompt = req.prompt or PER_FRAME_PROMPT
    if req.snapshot_id and _lakebase_available():
        try:
            with _pg_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT prompt, best_model, model_names FROM snapshots WHERE id = %s",
                    (req.snapshot_id,))
                row = cur.fetchone()
                if row:
                    snap_prompt, best, models = row
                    if not req.prompt and snap_prompt:
                        prompt = snap_prompt
                    if not req.per_frame_model:
                        per_frame = best or (models[0] if models else per_frame)
        except Exception as e:
            log.warning(f"snapshot lookup failed for {req.snapshot_id}: {e}")
    return per_frame, validator, prompt


def _cache_path(video_name: str) -> str:
    safe = video_name.replace("/", "_").replace("..", "_")
    return os.path.join(STUDIO_CACHE_DIR, f"{safe}.json")


def _read_cache(video_name: str) -> Optional[dict]:
    """Return cached analysis. Tries Lakebase first, then volume file."""
    # 1. Lakebase
    if _lakebase_available():
        try:
            with _pg_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT payload FROM video_analyses WHERE video_name = %s", (video_name,))
                row = cur.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            log.warning(f"_read_cache lakebase failed: {e}")
    # 2. Volume file fallback
    try:
        data = _download_bytes(_cache_path(video_name))
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _write_cache(video_name: str, payload: dict) -> str:
    """Persist analysis. Lakebase is authoritative; volume file is a backup."""
    storage_label = "file"
    # 1. Lakebase
    if _lakebase_available():
        try:
            with _pg_conn() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO video_analyses (video_name, duration_s, n_frames,
                        per_frame_model, validator_model, payload, elapsed_s, cached_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NOW())
                    ON CONFLICT (video_name) DO UPDATE SET
                        duration_s = EXCLUDED.duration_s,
                        n_frames = EXCLUDED.n_frames,
                        per_frame_model = EXCLUDED.per_frame_model,
                        validator_model = EXCLUDED.validator_model,
                        payload = EXCLUDED.payload,
                        elapsed_s = EXCLUDED.elapsed_s,
                        cached_at = NOW()
                """, (
                    video_name,
                    float(payload.get("duration_s") or 0),
                    int(payload.get("n_frames") or 0),
                    payload.get("per_frame_model"),
                    payload.get("validator_model"),
                    json.dumps(payload),
                    float(payload.get("total_elapsed_s") or 0),
                ))
            storage_label = "lakebase"
        except Exception as e:
            log.warning(f"_write_cache lakebase failed: {e}")
    # 2. Volume file backup (always)
    path = _cache_path(video_name)
    body = json.dumps(payload, indent=2).encode("utf-8")
    w = _workspace()
    try:
        try:
            w.files.create_directory(STUDIO_CACHE_DIR)
        except Exception:
            pass
        w.files.upload(path, contents=io.BytesIO(body), overwrite=True)
    except Exception as e:
        log.warning(f"_write_cache file failed: {e}")
    return f"{storage_label}:{path}"


def _read_audio_cache(video_name: str) -> Optional[dict]:
    if not _lakebase_available():
        return None
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT endpoint, text, segments, raw_response, cached_at "
                "FROM audio_transcripts WHERE video_name = %s", (video_name,))
            row = cur.fetchone()
            if not row:
                return None
            ep, text, segments, raw, cached_at = row
            return {
                "video_name": video_name,
                "endpoint": ep,
                "text": text,
                "segments": segments or [],
                "raw_response": raw,
                "cached_at": cached_at.timestamp() if cached_at else None,
                "from_cache": True,
            }
    except Exception as e:
        log.warning(f"_read_audio_cache failed: {e}")
        return None


def _write_audio_cache(video_name: str, endpoint: str, text: Optional[str],
                       segments: Optional[List[dict]], raw_response: object) -> None:
    if not _lakebase_available():
        return
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audio_transcripts (video_name, endpoint, text, segments, raw_response)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (video_name) DO UPDATE SET
                    endpoint = EXCLUDED.endpoint,
                    text = EXCLUDED.text,
                    segments = EXCLUDED.segments,
                    raw_response = EXCLUDED.raw_response,
                    cached_at = NOW()
            """, (
                video_name, endpoint, text,
                json.dumps(segments or []),
                json.dumps(raw_response) if raw_response is not None else None,
            ))
    except Exception as e:
        log.warning(f"_write_audio_cache failed: {e}")


def _gather_frames_for_video(video_name: str, max_frames: int) -> List[Dict[str, str]]:
    """Pick frames for a video from the existing extracted_frames/ cache.
    Returns a list of {name, path, timestamp_s, b64} ordered by timestamp.
    """
    base = video_name.replace(".mp4", "")
    video_dir = os.path.join(EXTRACTED_FRAMES_DIR, base)
    out: List[Dict[str, str]] = []
    for entry in _ls_dir(video_dir):
        name = entry["name"]
        if not name or entry["is_directory"] or not name.lower().endswith(".jpg"):
            continue
        ts = _parse_timestamp(name) or 0.0
        out.append({"name": name, "path": str(entry["path"] or os.path.join(video_dir, name)), "timestamp_s": ts})
    out.sort(key=lambda f: float(f["timestamp_s"]))
    if len(out) > max_frames:
        # Even-stride downsample
        stride = len(out) / float(max_frames)
        out = [out[int(i * stride)] for i in range(max_frames)]
    # Pre-encode all frames once
    for f in out:
        try:
            f["b64"] = _load_b64(f["path"])
        except Exception as e:
            f["b64"] = None
            f["error"] = f"encode failed: {e}"
    return out


PER_FRAME_PROMPT = """\
Identify the surgical instrument visible in this knee arthroscopy frame using these visual features:

PROBE vs SHAVER (most common confusion):
- probe = thin, narrow solid metal rod, uniform diameter end-to-end, tip may have a tiny hook or blunt end; NO opening, NO rotating head
- shaver = wider hollow tube with a distinct rectangular or oval side-opening (aspiration window) near the tip; shaft is noticeably thicker than a probe

Other instruments:
- burr = round/spherical or oval abrasive head at tip
- grasper or biter = two metal jaws that open/close at a hinge
- anchor_driver = ribbed or screw-threaded shaft
- electrocautery = smooth wand with flat, angled, or hook-shaped distal tip
- cannula = transparent or yellow plastic hollow tube

Vocabulary: probe, shaver, burr, grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, no_instrument_visible.
Respond with strict JSON: {"instrument": "<class>", "confidence": 0.0-1.0, "anatomy": "<short>", "tissue_condition": "<short>", "evidence": "<short>"}
"""


def _per_frame_call(model_name: str, frame_b64: str, frame_name: str) -> dict:
    return _per_frame_call_with_prompt(model_name, frame_b64, frame_name, PER_FRAME_PROMPT)


def _per_frame_call_with_prompt(model_name: str, frame_b64: str, frame_name: str, prompt: str) -> dict:
    client = _openai_client()
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ],
            }],
            max_tokens=400, temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return {"frame": frame_name, "ok": True, "raw": text, "parsed": _parse_strict_json(text),
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"frame": frame_name, "ok": False, "error": str(e)[:300],
                "elapsed_s": round(time.time() - t0, 2)}


VALIDATOR_PROMPT = """\
You are reviewing per-frame predictions from a vision model on a knee arthroscopy video.
Each frame has a timestamp, a predicted instrument class, anatomy, and an evidence note.

Your task:
1. Identify any obvious errors / inconsistencies in the per-frame predictions
   (e.g. an isolated 'shaver' label between many 'probe' frames is suspicious).
2. Group consecutive frames into 3-8 SECTIONS — each section is a coherent
   phase of the procedure (same/similar instrument or task). Section count
   should reflect video length: ~1 section per 30-60s of video.
3. Write a 1-2 sentence narrative for each section describing what's happening.

Respond with strict JSON of this shape:
{
  "summary": "<2-3 sentence overview of the whole procedure>",
  "sections": [
    {
      "title": "<short title>",
      "start_s": <float>,
      "end_s": <float>,
      "primary_instrument": "<class>",
      "narrative": "<1-2 sentence description>",
      "frame_indexes": [<integer>, ...]
    }
  ],
  "anomalies": [
    {"frame": "<frame_name>", "issue": "<why this prediction looks wrong>", "suggested_class": "<class or null>"}
  ]
}
"""


def _extract_instrument(parsed: Optional[dict]) -> str:
    """Pull the primary instrument class out of either prompt format:
       Studio per-frame:  {"instrument": "...", ...}
       Playground style:  {"instruments": [{"class": "..."}, ...], ...}"""
    if not isinstance(parsed, dict):
        return "?"
    direct = parsed.get("instrument")
    if isinstance(direct, str) and direct:
        return direct
    inst_list = parsed.get("instruments")
    if isinstance(inst_list, list) and inst_list and isinstance(inst_list[0], dict):
        cls = inst_list[0].get("class")
        if isinstance(cls, str) and cls:
            return cls
    return "?"


def _extract_evidence(parsed: Optional[dict]) -> str:
    if not isinstance(parsed, dict):
        return ""
    if isinstance(parsed.get("evidence"), str):
        return parsed["evidence"]
    inst_list = parsed.get("instruments")
    if isinstance(inst_list, list) and inst_list and isinstance(inst_list[0], dict):
        ev = inst_list[0].get("evidence")
        if isinstance(ev, str):
            return ev
    return ""


def _validator_call(model_name: str, per_frame_results: List[dict], duration_s: float) -> dict:
    """Send the per-frame results to a stronger model, ask it to organize +
    sanity-check. Pure text input — no images to keep the call cheap."""
    rows = []
    for i, r in enumerate(per_frame_results):
        ts = r.get("timestamp_s", 0.0)
        parsed = r.get("parsed") or {}
        instrument = _extract_instrument(parsed) if r.get("ok") else "ERROR"
        evidence = (_extract_evidence(parsed) or "")[:80]
        anatomy = (parsed.get("anatomy") or "")[:60]
        rows.append(f"[{i}] t={ts:6.1f}s  inst={instrument:20s}  anatomy={anatomy:30s}  ev={evidence}")
    payload = (
        f"Video duration: {duration_s:.1f}s\n"
        f"Per-frame predictions ({len(per_frame_results)}):\n" + "\n".join(rows)
    )
    client = _openai_client()
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": VALIDATOR_PROMPT + "\n\n" + payload},
            ],
            max_tokens=2000, temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return {"ok": True, "raw": text, "parsed": _parse_strict_json(text),
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500], "elapsed_s": round(time.time() - t0, 2)}


@app.get("/api/studio/analysis/{video_name}")
def studio_get_analysis(video_name: str):
    cached = _read_cache(video_name)
    if cached:
        return cached
    raise HTTPException(404, "No analysis cached for this video — POST /api/studio/analyze first")


@app.post("/api/studio/analyze")
async def studio_analyze(req: StudioAnalyzeRequest):
    per_frame_model, validator_model, per_frame_prompt = _resolve_studio_config(req)

    # Cache key = (video, per_frame_model, snapshot_id) to avoid mixing configs.
    # We keep a single cache slot per video for now; if user runs with a
    # different snapshot, we re-run rather than serving stale.
    if not req.force:
        cached = _read_cache(req.video_name)
        if cached and cached.get("per_frame_model") == per_frame_model \
                and cached.get("validator_model") == validator_model \
                and (cached.get("snapshot_id") or None) == (req.snapshot_id or None):
            cached["from_cache"] = True
            return cached

    t_start = time.time()
    # Step 1: gather frames
    log.info(f"[studio] gathering frames for {req.video_name}")
    frames = _gather_frames_for_video(req.video_name, req.max_frames)
    if not frames:
        raise HTTPException(404,
            f"No extracted frames for '{req.video_name}'. Run notebook 13 to extract first.")
    duration_s = max(float(f["timestamp_s"]) for f in frames)

    # Step 2: per-frame VLM in parallel
    log.info(f"[studio] {len(frames)} frames × {per_frame_model}")
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=min(16, len(frames))) as pool:
        futures = [
            loop.run_in_executor(pool, _per_frame_call_with_prompt,
                                 per_frame_model, f["b64"], f["name"], per_frame_prompt)
            for f in frames if f.get("b64")
        ]
        per_frame_results_raw = await asyncio.gather(*futures)
    by_name = {f["name"]: f for f in frames}
    per_frame_results = []
    for r in per_frame_results_raw:
        f = by_name.get(r["frame"], {})
        r["timestamp_s"] = f.get("timestamp_s")
        r["path"] = f.get("path")
        per_frame_results.append(r)
    per_frame_results.sort(key=lambda r: float(r.get("timestamp_s") or 0))

    # Step 3: validator
    log.info(f"[studio] validating with {validator_model}")
    validator = _validator_call(validator_model, per_frame_results, duration_s)

    payload = {
        "video_name": req.video_name,
        "duration_s": duration_s,
        "n_frames": len(per_frame_results),
        "per_frame_model": per_frame_model,
        "validator_model": validator_model,
        "snapshot_id": req.snapshot_id,
        "prompt_used": per_frame_prompt[:500],     # truncated for size
        "per_frame": per_frame_results,
        "validator": validator,
        "total_elapsed_s": round(time.time() - t_start, 2),
        "cached_at": time.time(),
        "from_cache": False,
    }
    # ── Step 4: MLflow scoring (gold-labels-aware) ──────────────────────
    # If any of the analyzed frames have ground-truth labels, compute per-class
    # accuracy / precision / recall and log to MLflow under the workbench
    # experiment. Cheap, deterministic, and gives us a single place to compare
    # Studio analysis quality across runs (different snapshots, different
    # videos, different per_frame_models). No-op when no labels overlap.
    try:
        payload["mlflow"] = _score_studio_with_mlflow(
            video_name=req.video_name,
            per_frame_model=per_frame_model,
            snapshot_id=req.snapshot_id,
            per_frame=per_frame_results,
            validator=validator,
            total_elapsed_s=payload["total_elapsed_s"],
        )
    except Exception as e:
        log.warning(f"[studio] mlflow scoring failed (non-fatal): {e}")
        payload["mlflow"] = None

    cache_path = _write_cache(req.video_name, payload)
    payload["cache_path"] = cache_path
    log.info(f"[studio] done in {payload['total_elapsed_s']}s, cached at {cache_path}")
    return payload


def _score_studio_with_mlflow(
    video_name: str,
    per_frame_model: str,
    snapshot_id: Optional[str],
    per_frame: List[dict],
    validator: dict,
    total_elapsed_s: float,
) -> Optional[Dict[str, object]]:
    """Score a Studio analysis against the gold-labels table, log to MLflow,
    and return a small summary the UI can show. Returns None if no overlap."""
    if not _lakebase_available():
        return None
    _ensure_labels_table()

    frame_paths = [f.get("path") for f in per_frame if f.get("path")]
    if not frame_paths:
        return None
    with _pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT frame_path, instrument FROM frame_labels WHERE frame_path = ANY(%s)",
            (frame_paths,),
        )
        labels_by_path = {r[0]: r[1] for r in cur.fetchall()}
    if not labels_by_path:
        return None

    def _pred(parsed):
        # Mirror the Playground's primaryClass: handle both shapes + empty-list.
        if not isinstance(parsed, dict):
            return None
        if isinstance(parsed.get("instrument"), str):
            v = parsed["instrument"].strip()
            return v.lower() if v else "no_instrument_visible"
        lst = parsed.get("instruments")
        if isinstance(lst, list):
            if len(lst) == 0:
                return "no_instrument_visible"
            if isinstance(lst[0], dict) and isinstance(lst[0].get("class"), str):
                v = lst[0]["class"].strip()
                return v.lower() if v else "no_instrument_visible"
        return None

    n_total = 0
    n_correct = 0
    per_class: Dict[str, Dict[str, int]] = {}  # class → {tp, fp, fn}
    for f in per_frame:
        path = f.get("path")
        gold = labels_by_path.get(path)
        if not gold:
            continue
        gold_l = gold.strip().lower()
        pred = _pred(f.get("parsed")) if f.get("ok") else None
        n_total += 1
        if pred == gold_l:
            n_correct += 1
            per_class.setdefault(gold_l, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        else:
            per_class.setdefault(gold_l, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            if pred:
                per_class.setdefault(pred, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    if n_total == 0:
        return None
    accuracy = n_correct / n_total

    # MLflow logging — wrapped to never break the parent request.
    mlflow_url = None
    try:
        import mlflow
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        with mlflow.start_run(run_name=f"studio_{video_name}") as run:
            mlflow.log_params({
                "kind": "studio",
                "video_name": video_name,
                "per_frame_model": per_frame_model,
                "snapshot_id": snapshot_id or "",
                "n_per_frame": len(per_frame),
                "n_gold_overlap": n_total,
            })
            mlflow.log_metric("accuracy_vs_gold", accuracy)
            mlflow.log_metric("n_correct", n_correct)
            mlflow.log_metric("n_total", n_total)
            mlflow.log_metric("total_elapsed_s", total_elapsed_s)
            # Per-class precision / recall (only for classes that appear)
            for cls, counts in per_class.items():
                tp = counts["tp"]; fp = counts["fp"]; fn = counts["fn"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                # Use safe metric names (no spaces / special chars)
                safe = "".join(c if c.isalnum() else "_" for c in cls)[:40]
                mlflow.log_metric(f"precision_{safe}", precision)
                mlflow.log_metric(f"recall_{safe}", recall)
            mlflow_url = f"{mlflow.get_tracking_uri()}#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    except Exception as e:
        log.warning(f"[studio] mlflow log failed: {e}")

    return {
        "n_gold_overlap": n_total,
        "accuracy_vs_gold": accuracy,
        "per_class": {
            cls: {
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0,
                "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0,
            } for cls, counts in per_class.items()
        },
        "mlflow_url": mlflow_url,
    }


# ── Studio: audio transcription ──────────────────────────────────────────

def _extract_audio_to_wav(video_bytes: bytes, target_sr: int = 16000) -> bytes:
    """Decode the audio track of an MP4 (or any container PyAV reads) and
    re-encode as a 16-bit mono WAV at `target_sr` Hz. Returns the WAV bytes.

    Used because the Whisper endpoint has a 16MB request limit which the
    original MP4 typically exceeds. Mono 16kHz s16le for a 4-min clip is
    ~7-8 MB — well under the limit and the canonical Whisper input format.
    """
    import av
    import wave
    in_buf = io.BytesIO(video_bytes)
    container = av.open(in_buf)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        raise RuntimeError("video has no audio stream")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                wf.writeframes(bytes(resampled.planes[0]))
    container.close()
    return out_buf.getvalue()


def _extract_segments(resp: object) -> tuple[Optional[str], List[dict]]:
    """Best-effort extraction of (full_text, segments) from a Whisper-style
    response. Whisper variants return wildly different shapes — handle the
    common ones."""
    text: Optional[str] = None
    segs: List[dict] = []
    if not resp:
        return text, segs

    # Possible shapes:
    #   {"predictions": [{"text": "...", "segments": [...] }]}
    #   {"predictions": [{"text": "...", "chunks": [{"timestamp":[s,e],"text":...}]}]}
    #   {"outputs": [{"text": "...", "segments": [...]}]}
    #   [{"text": "...", "segments": [...]}]
    #   {"text": "...", "segments": [...]}
    candidates = []
    if isinstance(resp, dict):
        if "predictions" in resp:
            candidates = resp["predictions"]
        elif "outputs" in resp:
            candidates = resp["outputs"]
        else:
            candidates = [resp]
    elif isinstance(resp, list):
        candidates = resp

    if not candidates:
        return text, segs
    first = candidates[0]
    if not isinstance(first, dict):
        return str(first), []

    text = first.get("text") or first.get("transcription") or first.get("full_text")
    raw_segs = first.get("segments") or first.get("chunks") or first.get("words") or []
    for s in raw_segs:
        if not isinstance(s, dict):
            continue
        # huggingface/openai whisper styles
        ts = s.get("timestamp") or s.get("ts")
        start = s.get("start")
        end = s.get("end")
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            start = ts[0] if start is None else start
            end = ts[1] if end is None else end
        if start is None and end is None:
            continue
        segs.append({
            "start": float(start) if start is not None else 0.0,
            "end": float(end) if end is not None else float(start or 0.0),
            "text": s.get("text") or s.get("word") or "",
        })
    return text, segs


@app.post("/api/studio/audio/transcribe")
def studio_audio_transcribe(req: dict):
    """Transcribe the audio track of a video.

    Behaviour:
      1. Return cached transcript if present.
      2. Otherwise download the video bytes, send to the configured Whisper
         endpoint with a few candidate request shapes, parse segments, cache.

    Set `force=true` to bypass cache.
    """
    video_name = req.get("video_name")
    force = bool(req.get("force"))
    if not video_name:
        raise HTTPException(400, "video_name required")
    if not force:
        cached = _read_audio_cache(video_name)
        if cached:
            return cached

    abs_path = os.path.join(VOLUME_PATH, video_name)
    _check_allowed_path(abs_path)
    try:
        video_bytes = _download_bytes(abs_path)
    except Exception as e:
        raise HTTPException(404, f"Video not found: {e}")

    # Extract just the audio track to a 16kHz mono WAV — Whisper's preferred
    # input. The 16MB serving-endpoint request limit makes sending the raw
    # MP4 infeasible for typical surgery videos.
    try:
        wav_bytes = _extract_audio_to_wav(video_bytes)
    except RuntimeError as e:
        if "no audio stream" in str(e).lower():
            payload = {
                "video_name": video_name,
                "endpoint": STUDIO_AUDIO_ENDPOINT,
                "status": "no_audio_stream",
                "message": "This video has no audio track — nothing to transcribe.",
                "text": None, "segments": [], "raw_response": None,
                "from_cache": False,
            }
            _write_audio_cache(video_name, STUDIO_AUDIO_ENDPOINT, None, [],
                               {"status": "no_audio_stream"})
            return payload
        raise HTTPException(500, f"Audio extraction failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Audio extraction failed: {e}")
    log.info(f"audio extract: video={len(video_bytes)/1e6:.1f}MB → wav={len(wav_bytes)/1e6:.1f}MB")

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    w = _workspace()
    # We learned from probing that the endpoint accepts dataframe_records with
    # audio_base64, but try a couple of alternates first in case the schema differs.
    candidates = [
        ("dataframe_records.audio_base64", {"dataframe_records": [{"audio_base64": audio_b64}]}),
        ("dataframe_records.audio_bytes",  {"dataframe_records": [{"audio_bytes": audio_b64}]}),
        ("dataframe_split.audio_base64",   {"dataframe_split": {"columns": ["audio_base64"], "data": [[audio_b64]]}}),
    ]
    last_err = None
    for label, body in candidates:
        try:
            resp = w.api_client.do(
                "POST",
                f"/serving-endpoints/{STUDIO_AUDIO_ENDPOINT}/invocations",
                body=body,
            )
            text, segments = _extract_segments(resp)
            payload = {
                "video_name": video_name,
                "endpoint": STUDIO_AUDIO_ENDPOINT,
                "shape": label,
                "text": text,
                "segments": segments,
                "raw_response": resp,
                "from_cache": False,
            }
            _write_audio_cache(video_name, STUDIO_AUDIO_ENDPOINT, text, segments, resp)
            return payload
        except Exception as e:
            last_err = f"{label}: {str(e)[:200]}"
            log.info(f"audio shape {label} failed: {last_err}")
            continue
    raise HTTPException(502, f"All Whisper input shapes failed. Last: {last_err}")


@app.get("/api/studio/audio/{video_name}")
def studio_audio_get(video_name: str):
    cached = _read_audio_cache(video_name)
    if not cached:
        raise HTTPException(404, "no cached transcript — POST /api/studio/audio/transcribe first")
    return cached


# ── SPA serving — keep last ──────────────────────────────────────────────

if os.path.isdir(os.path.join(STATIC_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404)
    candidate = os.path.join(STATIC_DIR, full_path)
    if full_path and os.path.isfile(candidate):
        return FileResponse(candidate)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

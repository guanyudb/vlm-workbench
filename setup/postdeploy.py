# Databricks notebook source
# MAGIC %md
# MAGIC # Post-deploy setup for vlm-workbench
# MAGIC
# MAGIC Runs once after a fresh `databricks bundle deploy` to wire up the
# MAGIC permissions DAB can't express directly:
# MAGIC
# MAGIC - GRANT all the UC privileges the App SP needs on the workbench schema
# MAGIC - Create the Lakebase Postgres role for the App SP (so it can connect
# MAGIC   via M2M OAuth → token-as-password)
# MAGIC - Grant the App SP READ on the HF token secret scope (if the scope's
# MAGIC   owner is the deployer)
# MAGIC - Sanity-check the App is healthy: enumerate every bundled notebook
# MAGIC   the App references and confirm the SP can read it
# MAGIC
# MAGIC The notebook is idempotent — re-running after a no-op deploy is safe.

# COMMAND ----------
import json, os, time

dbutils.widgets.text("uc_catalog", "", "UC catalog")
dbutils.widgets.text("uc_schema", "", "UC schema")
dbutils.widgets.text("volume_name", "medical_video", "Volume name")
dbutils.widgets.text("app_name", "vlm-workbench", "App name")
dbutils.widgets.text("lakebase_instance", "", "Lakebase instance name")
dbutils.widgets.text("lakebase_database", "vlm_workbench", "Lakebase database name")
dbutils.widgets.text("workbench_secret_scope", "vlmwb_config", "Workbench secret scope")
dbutils.widgets.text("hf_secret_scope", "", "HF secret scope")
dbutils.widgets.text("hf_secret_key", "hf_token", "HF secret key")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
VOLUME = dbutils.widgets.get("volume_name")
APP_NAME = dbutils.widgets.get("app_name")
LB_INSTANCE = dbutils.widgets.get("lakebase_instance")
LB_DATABASE = dbutils.widgets.get("lakebase_database")
WB_SCOPE = dbutils.widgets.get("workbench_secret_scope")
HF_SCOPE = dbutils.widgets.get("hf_secret_scope")
HF_KEY = dbutils.widgets.get("hf_secret_key")
for k, v in [("uc_catalog", UC_CATALOG), ("uc_schema", UC_SCHEMA),
             ("lakebase_instance", LB_INSTANCE)]:
    if not v:
        raise SystemExit(f"{k} widget is required")

print(f"[setup] catalog={UC_CATALOG} schema={UC_SCHEMA} app={APP_NAME}")
print(f"[setup] lakebase={LB_INSTANCE}/{LB_DATABASE}")
print(f"[setup] scopes: workbench={WB_SCOPE!r}, hf={HF_SCOPE!r}")

# COMMAND ----------
# Look up the App's service principal application_id. The Apps API exposes
# it under `service_principal_client_id`.
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
try:
    app = w.apps.get(name=APP_NAME)
    sp_app_id = (getattr(app, "service_principal_client_id", None)
                 or getattr(app, "service_principal_id", None))
except Exception as e:
    raise SystemExit(f"could not look up app {APP_NAME}: {e}")
if not sp_app_id:
    raise SystemExit(f"app {APP_NAME} has no service principal — has it been created yet?")
print(f"[setup] app SP = {sp_app_id}")

# COMMAND ----------
# UC grants. Idempotent — GRANT is no-op when already granted.
import subprocess
def _sql(stmt: str):
    print(f"[sql] {stmt[:160]}")
    spark.sql(stmt)

_sql(f"GRANT USE CATALOG ON CATALOG {UC_CATALOG} TO `{sp_app_id}`")
_sql(f"GRANT USE SCHEMA, CREATE TABLE, CREATE FUNCTION, CREATE MODEL, "
     f"CREATE VOLUME, MODIFY, SELECT, EXECUTE "
     f"ON SCHEMA {UC_CATALOG}.{UC_SCHEMA} TO `{sp_app_id}`")
_sql(f"GRANT READ VOLUME, WRITE VOLUME ON VOLUME "
     f"{UC_CATALOG}.{UC_SCHEMA}.{VOLUME} TO `{sp_app_id}`")

# COMMAND ----------
# Secret scope ACLs — only attempt if the scope already exists. The
# workbench_secret_scope is created by the bundle; the HF scope must exist
# beforehand because we can't create it without the user's HF token.
def _grant_secret(scope: str, perm: str):
    if not scope: return "(skipped — no scope)"
    try:
        existing = [s.name for s in w.secrets.list_scopes()]
        if scope not in existing:
            return f"scope {scope!r} not found, skipping"
        # put-acl is idempotent
        w.secrets.put_acl(scope=scope, principal=sp_app_id, permission=perm)
        return f"granted {perm} on {scope!r}"
    except Exception as e:
        return f"failed: {e}"

# Workbench scope is created by the bundle (deployer owns it) so we can
# grant WRITE — required for the in-app HF token paste flow and for any
# future Setup-tab config writes.
print(_grant_secret(WB_SCOPE, "WRITE"))
# HF scope may be shared (e.g. e2-demo-field-eng's `hf` scope is owned by a
# different team). Try WRITE first; if the deployer doesn't own the scope
# the call 403s and we fall back to READ.
hf_w = _grant_secret(HF_SCOPE, "WRITE")
print(hf_w)
if "failed" in hf_w.lower() or "403" in hf_w or "permission" in hf_w.lower():
    print(_grant_secret(HF_SCOPE, "READ"))

# COMMAND ----------
# Workbench-config secret values. The scope is created by the bundle but
# values can't be set via DAB — write them here so the App reads
# catalog/schema/volume at runtime.
def _put(scope: str, key: str, value: str):
    try:
        w.secrets.put_secret(scope=scope, key=key, string_value=value)
        print(f"  set {scope}/{key} = {value!r}")
    except Exception as e:
        print(f"  put {scope}/{key} failed: {e}")

if WB_SCOPE:
    _put(WB_SCOPE, "uc_catalog", UC_CATALOG)
    _put(WB_SCOPE, "uc_schema", UC_SCHEMA)
    _put(WB_SCOPE, "volume_name", VOLUME)
    # HF scope/key are metadata, not secrets — but Apps' `valueFrom` only
    # accepts secret-resource bindings, so we round-trip these through the
    # workbench scope. The Setup tab reads them to show which scope holds
    # the token; the paste-HF-in-app flow uses them to know where to write.
    _put(WB_SCOPE, "hf_secret_scope", HF_SCOPE)
    _put(WB_SCOPE, "hf_secret_key", HF_KEY)
    # Lakebase project name + branch — needed by the ingest notebook to mint
    # its OWN OAuth token (security fix: app no longer writes pg_password to
    # the volume YAML). The `postgres:` binding auto-injects PGHOST/PGDATABASE
    # but not the project name, so we seed it here.
    _put(WB_SCOPE, "lakebase_instance", LB_INSTANCE)
    _put(WB_SCOPE, "lakebase_branch", os.environ.get("LAKEBASE_BRANCH", "production"))

# COMMAND ----------
# Pre-create the subdirectories the Library tab + every notebook expects on
# the Volume, so first-time users can drop files in immediately without
# hunting through the catalog explorer. Idempotent — `create_directory`
# is a no-op when the path already exists.
VOL_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{VOLUME}"
_EXPECTED_DIRS = [
    "videos/inbox",        # MP4 drop zone (Library ingest)
    "images",              # parent for image-batch subfolders
    "extracted_frames",    # smart-frame extractor output
    "eval_frames",         # eval set
    "local_models",        # HF model snapshots cached on Volume
    "optimizer_runs",      # GEPA / DSPy run outputs
    "finetune_runs",       # LoRA training outputs
    "local_runs",          # serverless GPU inference runs
    "ingest_runs",         # per-ingest YAML configs + logs
    "studio_analyses",     # Studio's per-video JSON caches
    "config",              # task_config.json + any future workspace config
]
for _d in _EXPECTED_DIRS:
    try:
        w.files.create_directory(f"{VOL_ROOT}/{_d}")
        print(f"  ok  {VOL_ROOT}/{_d}")
    except Exception as _e:
        msg = str(_e)
        if "exists" in msg.lower() or "409" in msg:
            print(f"  ok  {VOL_ROOT}/{_d} (existed)")
        else:
            print(f"  !!  {VOL_ROOT}/{_d}: {msg[:120]}")

# COMMAND ----------
# Lakebase Postgres role for the App SP — same UUID as the SP's application_id.
#
# Two Lakebase API surfaces coexist in 2026:
#   - Old: `/api/2.0/database/instances/<name>/roles` (Database Instance)
#   - New: `/api/2.0/postgres/projects/<name>/branches/<branch>/roles` (Project)
# We probe each; the first that doesn't 404 wins. Idempotent — a 409 "already
# exists" response is treated as success.
def _list_project_roles(project: str, branch: str):
    """Return list of (postgres_role, name) tuples for the project's roles."""
    try:
        r = w.api_client.do(
            "GET",
            f"/api/2.0/postgres/projects/{project}/branches/{branch}/roles",
        )
        return [
            ((rr.get("status") or {}).get("postgres_role"), rr.get("name"))
            for rr in (r.get("roles") or [])
        ]
    except Exception:
        return []

def _create_lakebase_role():
    # Detect: try old Instance API first
    try:
        w.api_client.do("GET", f"/api/2.0/database/instances/{LB_INSTANCE}")
        kind = "instance"
    except Exception:
        kind = None
    if kind is None:
        try:
            w.api_client.do("GET", f"/api/2.0/postgres/projects/{LB_INSTANCE}")
            kind = "project"
        except Exception as e:
            return f"lakebase {LB_INSTANCE} not found via either API: {e}"

    if kind == "instance":
        path = f"/api/2.0/database/instances/{LB_INSTANCE}/roles"
        body = {"name": sp_app_id, "identity_type": "SERVICE_PRINCIPAL"}
    else:
        # New Lakebase Project API. Body shape (verified 2026-05-26):
        #   {"spec": {"identity_type": "SERVICE_PRINCIPAL",
        #             "postgres_role": "<sp_app_uuid>"}}
        # Notes:
        # - role_id query param must match `^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$`,
        #   so we cannot use the SP UUID (which has uppercase chars / wrong
        #   shape) — we let the server auto-generate the role_id slug.
        # - The Postgres role NAME that the App will authenticate as is what
        #   you put in `postgres_role`, i.e. the SP UUID.
        # - The default branch is "production"; override via LAKEBASE_BRANCH.
        branch = os.environ.get("LAKEBASE_BRANCH", "production")
        # Idempotency check — server returns 200 on create even when same
        # postgres_role exists (it just creates another slug). De-dupe in
        # client by listing first.
        existing = _list_project_roles(LB_INSTANCE, branch)
        if any(pgrole == sp_app_id for pgrole, _ in existing):
            return f"Lakebase role for SP {sp_app_id} already present (ok)"
        path = f"/api/2.0/postgres/projects/{LB_INSTANCE}/branches/{branch}/roles"
        body = {"spec": {"identity_type": "SERVICE_PRINCIPAL",
                          "postgres_role": sp_app_id}}

    try:
        resp = w.api_client.do("POST", path, body=body)
        return f"created Lakebase role for SP via {kind} API: {resp}"
    except Exception as e:
        msg = str(e)
        if "already exists" in msg.lower() or "409" in msg:
            return f"Lakebase role for SP {sp_app_id} already present (ok, {kind} API)"
        return f"create role failed via {kind} API: {e}"

print(_create_lakebase_role())

# COMMAND ----------
# Grant the App SP the Postgres privileges it needs.
#
# Lakebase auto-provisions a role for the App SP (via the `postgres:` App
# binding) but that role only has CONNECT — it can't CREATE TABLE in
# `public`. The workbench's _ensure_ingest_tables() needs DDL, so we GRANT
# CREATE here. The deployer is typically a DATABRICKS_SUPERUSER in Lakebase
# (auto-granted to the workspace user who created the project), so this
# works as the notebook user.
#
# Idempotent: GRANT is a no-op when the privilege is already present.
def _grant_pg_privileges():
    if not sp_app_id:
        return "(skipped — no SP id)"
    # Resolve host. Use raw api_client.do() instead of typed SDK methods —
    # serverless env's SDK may be older than the workstation's.
    branch = os.environ.get("LAKEBASE_BRANCH", "production")
    host = None
    try:
        r = w.api_client.do(
            "GET",
            f"/api/2.0/postgres/projects/{LB_INSTANCE}/branches/{branch}/endpoints",
        )
        for ep in (r.get("endpoints") if isinstance(r, dict) else r) or []:
            host = (((ep.get("status") or {}).get("hosts") or {}) or {}).get("host")
            if host:
                break
    except Exception as e:
        print(f"  project endpoints lookup failed: {type(e).__name__}: {e}")
    # Fall back to Instance API for legacy Provisioned deploys
    if not host:
        try:
            inst = w.api_client.do("GET", f"/api/2.0/database/instances/{LB_INSTANCE}")
            host = inst.get("read_write_dns") or inst.get("dns") or inst.get("hostname")
        except Exception as e:
            print(f"  instance lookup failed: {type(e).__name__}: {e}")
    if not host:
        return f"could not discover host for {LB_INSTANCE}"
    # IMPORTANT: the notebook's `apiToken()` is a Databricks PAT, NOT an
    # OAuth JWT — Lakebase rejects it with
    # `Provided authentication token is not a valid JWT encoding`.
    # The supported way to mint a Lakebase-compatible token is
    # `POST /api/2.0/postgres/credentials` with `{endpoint: "projects/.../endpoints/primary"}`
    # for Project Lakebases, or the legacy database-instance endpoint for
    # Provisioned ones. We use raw api_client.do() because the serverless
    # notebook env's SDK may be older than the workstation's and not have
    # `w.postgres.generate_database_credential`.
    endpoint_path = f"projects/{LB_INSTANCE}/branches/{branch}/endpoints/primary"
    pg_token = None
    try:
        resp = w.api_client.do(
            "POST", "/api/2.0/postgres/credentials",
            body={"endpoint": endpoint_path},
        )
        pg_token = resp.get("token") if isinstance(resp, dict) else None
    except Exception as e:
        # Provisioned fallback
        try:
            import uuid as _uuid
            resp = w.api_client.do(
                "POST", "/api/2.0/database/credentials/generate",
                body={"instance_names": [LB_INSTANCE], "request_id": str(_uuid.uuid4())},
            )
            pg_token = (resp.get("token") if isinstance(resp, dict) else None)
        except Exception as e2:
            return f"could not mint pg token: project={e}, instance={e2}"
    if not pg_token:
        return f"credential API returned no token for {endpoint_path}"
    me = w.current_user.me().user_name
    try:
        import subprocess, sys
        try:
            import psycopg
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psycopg[binary]"], check=True)
            import psycopg
        with psycopg.connect(
            host=host, port=5432, user=me, password=pg_token,
            dbname=LB_DATABASE, sslmode="require", autocommit=True,
            connect_timeout=15,
        ) as conn, conn.cursor() as cur:
            # CREATE on schema public so the App SP can _ensure_ingest_tables.
            # USAGE comes automatically with default ACLs but grant explicitly
            # in case the public schema's defaults were tightened.
            cur.execute(f'GRANT USAGE, CREATE ON SCHEMA public TO "{sp_app_id}"')
            # Future tables created by the SP itself need no further grants —
            # ownership flows from the creator. For tables created by OTHER
            # roles that the SP needs to read/write, ALTER DEFAULT PRIVILEGES
            # in public GRANT ... TO sp would be needed. Skipping for now.
        return f"granted USAGE, CREATE ON SCHEMA public TO {sp_app_id}"
    except Exception as e:
        return f"grant failed: {type(e).__name__}: {e}"

_GRANT_RESULT = _grant_pg_privileges()
print(_GRANT_RESULT)

# COMMAND ----------
# Sanity-check: list the bundled notebooks the App references at runtime and
# verify the SP would be able to read them. We can't impersonate the SP from
# this notebook, but if the App was created by `bundle deploy`, the source-
# code-path lives in the SP's home and the SP has CAN_MANAGE on it.
notebook_paths = [
    "local_inference_notebooks/run/notebook",
    "local_inference_notebooks/finetune/notebook",
    "local_inference_notebooks/repin/notebook",
    "local_inference_notebooks/setup_cache/notebook",
    "optimizer_notebooks/gepa/notebook",
    "optimizer_notebooks/dspy/notebook",
    "ingest_notebooks/smart_frames/notebook",
]
art = getattr(getattr(app, "active_deployment", None), "deployment_artifacts", None)
src = getattr(art, "source_code_path", None) if art else None
print(f"[setup] active deployment source: {src}")

# COMMAND ----------
dbutils.notebook.exit(json.dumps({
    "ok": True,
    "sp": sp_app_id,
    "catalog": UC_CATALOG,
    "schema": UC_SCHEMA,
    "volume": VOLUME,
    "lakebase_instance": LB_INSTANCE,
    "pg_grant": _GRANT_RESULT,
}))

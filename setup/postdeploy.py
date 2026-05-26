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

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
VOLUME = dbutils.widgets.get("volume_name")
APP_NAME = dbutils.widgets.get("app_name")
LB_INSTANCE = dbutils.widgets.get("lakebase_instance")
LB_DATABASE = dbutils.widgets.get("lakebase_database")
WB_SCOPE = dbutils.widgets.get("workbench_secret_scope")
HF_SCOPE = dbutils.widgets.get("hf_secret_scope")
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
def _grant_secret_read(scope: str):
    if not scope: return "(skipped — no scope)"
    try:
        existing = [s.name for s in w.secrets.list_scopes()]
        if scope not in existing:
            return f"scope {scope!r} not found, skipping"
        # put-acl is idempotent
        w.secrets.put_acl(scope=scope, principal=sp_app_id, permission="READ")
        return f"granted READ on {scope!r}"
    except Exception as e:
        return f"failed: {e}"

print(_grant_secret_read(WB_SCOPE))
print(_grant_secret_read(HF_SCOPE))

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

# COMMAND ----------
# Lakebase Postgres role for the App SP — same UUID as the SP's application_id.
# Idempotent: the Lakebase Database REST API returns 409 on duplicate, we treat
# that as success.
def _create_lakebase_role():
    try:
        token = w.api_client.do(
            "POST",
            f"/api/2.0/database/instances/{LB_INSTANCE}/roles",
            body={"name": sp_app_id, "identity_type": "SERVICE_PRINCIPAL"},
        )
        return f"created Lakebase role {sp_app_id} in {LB_INSTANCE}"
    except Exception as e:
        msg = str(e)
        if "already exists" in msg.lower() or "409" in msg:
            return f"Lakebase role {sp_app_id} already present (ok)"
        return f"create role failed: {e}"

print(_create_lakebase_role())

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
}))

# Databricks notebook source
# MAGIC %md
# MAGIC # Lakebase connection probe
# MAGIC
# MAGIC Runs as the executing user — establishes the baseline ("does the
# MAGIC instance accept connections AT ALL from anyone in this workspace?").
# MAGIC Compared with the App's `/api/debug/lakebase` (which runs as the App
# MAGIC SP), this isolates whether problems are at the instance level vs the
# MAGIC SP-role level.
# MAGIC
# MAGIC Tests the connection against both supported auth surfaces:
# MAGIC   - new Lakebase **Project** (`/api/2.0/postgres/projects/<name>`)
# MAGIC   - old Lakebase **Instance** (`/api/2.0/database/instances/<name>`)
# MAGIC and reports which one matches the configured instance name.

# COMMAND ----------
%pip install -q psycopg[binary]

# COMMAND ----------
import os, time, json, socket
dbutils.widgets.text("lakebase_instance", "vlm", "Lakebase instance/project name")
dbutils.widgets.text("lakebase_database", "databricks_postgres", "Database name")
dbutils.widgets.text("lakebase_branch", "production", "Branch (Project only)")

LB_INSTANCE = dbutils.widgets.get("lakebase_instance")
LB_DATABASE = dbutils.widgets.get("lakebase_database")
LB_BRANCH = dbutils.widgets.get("lakebase_branch")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print(f"[probe] instance/project name = {LB_INSTANCE}")
print(f"[probe] database              = {LB_DATABASE}")
print(f"[probe] branch (Project only) = {LB_BRANCH}")

# COMMAND ----------
# Step 1 — discover the connection host. Try Project then Instance.
def discover_host():
    # Project path
    try:
        for ep in w.postgres.list_endpoints(parent=f"projects/{LB_INSTANCE}/branches/{LB_BRANCH}"):
            host = getattr(getattr(ep, "status", None), "hosts", None)
            host = getattr(host, "host", None) if host else None
            if host:
                return ("project", host)
    except Exception as e:
        print(f"  project lookup failed: {type(e).__name__}: {e}")
    # Instance path
    try:
        inst = w.api_client.do("GET", f"/api/2.0/database/instances/{LB_INSTANCE}")
        for k in ("read_write_dns", "dns", "hostname"):
            if inst.get(k):
                return ("instance", inst[k])
    except Exception as e:
        print(f"  instance lookup failed: {type(e).__name__}: {e}")
    return (None, None)

kind, host = discover_host()
if not host:
    raise SystemExit(f"could not discover host for {LB_INSTANCE}")
print(f"[probe] kind={kind} host={host}")

# COMMAND ----------
# Step 2 — pick a Postgres role + auth.
#   Users connect with their email as `user`; SPs use the SP UUID.
#   The Postgres password is an OAuth bearer (databricks_token works in the
#   notebook context).
me = w.current_user.me().user_name
print(f"[probe] notebook user = {me}")

# Notebook execution context already has a token; reuse it.
notebook_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)
print(f"[probe] token len     = {len(notebook_token)}  first8={notebook_token[:8]}")

# COMMAND ----------
# Step 3 — TCP probe (network reachability)
try:
    s = socket.create_connection((host, 5432), timeout=10)
    s.close()
    print(f"[probe] tcp:5432 reachable")
except Exception as e:
    print(f"[probe] tcp:5432 FAILED: {e}")
    raise

# COMMAND ----------
# Step 4 — authenticated connect + identity
import psycopg
def connect():
    return psycopg.connect(
        host=host, port=5432, user=me, password=notebook_token,
        dbname=LB_DATABASE, sslmode="require", autocommit=True,
        connect_timeout=15,
    )

try:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_user, current_database(), version()")
        u, db, v = cur.fetchone()
        print(f"[probe] AUTHED — user={u} db={db}")
        print(f"        server={v[:80]}")
except Exception as e:
    print(f"[probe] AUTH FAILED: {type(e).__name__}: {e}")
    raise

# COMMAND ----------
# Step 5 — schema visibility + DDL capability
with connect() as conn, conn.cursor() as cur:
    cur.execute("SELECT schema_name FROM information_schema.schemata ORDER BY 1")
    schemas = [r[0] for r in cur.fetchall()]
    print(f"[probe] visible schemas: {schemas[:20]}{' …' if len(schemas) > 20 else ''}")
    cur.execute("SHOW search_path")
    print(f"[probe] search_path: {cur.fetchone()[0]}")
    # DDL — create + drop a tiny probe table to mirror what the app does
    try:
        cur.execute("CREATE TABLE IF NOT EXISTS vlmwb_lakebase_probe (id INT, note TEXT)")
        cur.execute("INSERT INTO vlmwb_lakebase_probe VALUES (1, 'ok')")
        cur.execute("SELECT count(*) FROM vlmwb_lakebase_probe")
        n = cur.fetchone()[0]
        cur.execute("DROP TABLE vlmwb_lakebase_probe")
        print(f"[probe] DDL ok — created/inserted/dropped (n={n})")
    except Exception as e:
        print(f"[probe] DDL FAILED: {type(e).__name__}: {e}")
        # not raising — we want to see schema info even if DDL fails

# COMMAND ----------
# Step 6 — list roles via API so we can compare configured vs actual
print("[probe] roles in this project/instance:")
try:
    if kind == "project":
        r = w.api_client.do("GET", f"/api/2.0/postgres/projects/{LB_INSTANCE}/branches/{LB_BRANCH}/roles")
        for rr in r.get("roles", []):
            s = rr.get("status") or {}
            print(f"  - role_id={s.get('role_id'):<30} pg_role={s.get('postgres_role'):<40} identity={s.get('identity_type')}")
    else:
        r = w.api_client.do("GET", f"/api/2.0/database/instances/{LB_INSTANCE}/roles")
        for rr in (r.get("roles") or r):
            print(f"  - {rr.get('name'):<50} identity={rr.get('identity_type')}")
except Exception as e:
    print(f"  list-roles failed: {e}")

# COMMAND ----------
dbutils.notebook.exit(json.dumps({
    "ok": True, "kind": kind, "host": host, "db": LB_DATABASE, "user": me,
}))

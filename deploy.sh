#!/bin/bash
# One-command deploy of vlm-workbench to any Databricks workspace.
#
# Usage:
#   ./deploy.sh <target> [--profile <profile>]
#
# Targets:
#   dev          — your primary dev workspace      (reads .databricks/bundle/dev/variable-overrides.json)
#   dev_e2fe     — secondary test workspace        (reads .databricks/bundle/dev_e2fe/variable-overrides.json)
#   aws | azure | gcp — production cloud targets   (reads module.env)
#
# Examples:
#   ./deploy.sh dev_e2fe                          # uses DATABRICKS_CONFIG_PROFILE or DEFAULT
#   ./deploy.sh dev_e2fe --profile e2FE           # explicit profile
#   ./deploy.sh aws                               # prod aws (reads module.env)
#
# What it does (modelled on genesis-workbench's pattern):
#   1. ./build.sh                  — builds the React SPA into static/
#   2. seed required secrets       — breaks the chicken-and-egg where DAB
#                                    binds to secrets that don't exist yet
#   3. bundle deploy --auto-approve — creates/updates App + schema + volume
#                                    + secret scope + jobs, syncs source
#   4. bundle run vlmwb_postdeploy_setup
#                                 — grants UC + Postgres privileges to App SP,
#                                   seeds workspace-config secrets, pre-creates
#                                   volume subdirs, creates Lakebase role
#   5. bundle run vlm_workbench_app — deploys source + starts the App
#   6. prints the URL
#
# Idempotent — re-run any time.

set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dev|dev_e2fe|aws|azure|gcp> [--profile <profile>]"
  exit 1
fi
TARGET_ARG="$1"; shift
PROFILE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

case "$TARGET_ARG" in
  dev)       TARGET=dev      ; USES_OVERRIDES=true ;;
  dev_e2fe)  TARGET=dev_e2fe ; USES_OVERRIDES=true ;;
  aws)       TARGET=prod_aws ; USES_OVERRIDES=false ;;
  azure)     TARGET=prod_azure; USES_OVERRIDES=false ;;
  gcp)       TARGET=prod_gcp ; USES_OVERRIDES=false ;;
  *)         echo "🚫 unknown target '$TARGET_ARG'. Pick one of: dev, dev_e2fe, aws, azure, gcp"; exit 1 ;;
esac

PROFILE_ARG=""
PROFILE_DISPLAY="${DATABRICKS_CONFIG_PROFILE:-DEFAULT}"
if [ -n "$PROFILE" ]; then
  PROFILE_ARG="--profile $PROFILE"
  PROFILE_DISPLAY="$PROFILE"
fi
DBX="databricks $PROFILE_ARG"

echo "═════════════════════════════════════════════════════════"
echo "  vlm-workbench deploy"
echo "    target  = $TARGET"
echo "    profile = $PROFILE_DISPLAY"
echo "═════════════════════════════════════════════════════════"

# ── 0. Variable source ────────────────────────────────────────────────────
if [ "$USES_OVERRIDES" = true ]; then
  OVERRIDES=".databricks/bundle/$TARGET/variable-overrides.json"
  if [ ! -f "$OVERRIDES" ]; then
    echo "🚫 Missing $OVERRIDES."
    echo "   Copy from .databricks/bundle/dev_e2fe/variable-overrides.json and edit."
    exit 1
  fi
  echo "▶️ Using overrides: $OVERRIDES"
  read_var() { python3 -c "import json; print((json.load(open('$OVERRIDES'))).get('$1',''))"; }
  UC_CATALOG=$(read_var uc_catalog)
  UC_SCHEMA=$(read_var uc_schema)
  HF_SECRET_SCOPE=$(read_var hf_secret_scope)
  HF_SECRET_KEY=$(read_var hf_secret_key); HF_SECRET_KEY="${HF_SECRET_KEY:-hf_token}"
  WB_SCOPE=$(read_var workbench_secret_scope); WB_SCOPE="${WB_SCOPE:-vlmwb_config}"
  LB_INSTANCE=$(read_var lakebase_instance)
  LB_DATABASE=$(read_var lakebase_database); LB_DATABASE="${LB_DATABASE:-databricks_postgres}"
else
  if [ ! -f module.env ]; then
    echo "🚫 module.env not found. Copy module.env.sample, fill it in, then re-run."
    exit 1
  fi
  source module.env
  UC_CATALOG="${BUNDLE_VAR_uc_catalog}"
  UC_SCHEMA="${BUNDLE_VAR_uc_schema}"
  HF_SECRET_SCOPE="${BUNDLE_VAR_hf_secret_scope}"
  HF_SECRET_KEY="${BUNDLE_VAR_hf_secret_key:-HF_TOKEN}"
  WB_SCOPE="${BUNDLE_VAR_workbench_secret_scope:-vlmwb_config}"
  LB_INSTANCE="${BUNDLE_VAR_lakebase_instance}"
  LB_DATABASE="${BUNDLE_VAR_lakebase_database:-databricks_postgres}"
fi

for pair in "uc_catalog:$UC_CATALOG" "uc_schema:$UC_SCHEMA" \
            "hf_secret_scope:$HF_SECRET_SCOPE" "lakebase_instance:$LB_INSTANCE"; do
  k="${pair%%:*}"; v="${pair#*:}"
  if [ -z "$v" ]; then echo "🚫 $k is empty — fill it in your overrides/module.env"; exit 1; fi
done

# ── 1. Build the SPA ──────────────────────────────────────────────────────
echo
echo "▶️ [1/5] Building React SPA"
./build.sh

# ── 2. Seed required secrets (DAB binds to these — they must exist first) ─
echo
echo "▶️ [2/5] Pre-seeding workbench secrets (idempotent)"
# Ensure the scope exists. `secrets create-scope` 400s if it does — swallow.
$DBX secrets create-scope "$WB_SCOPE" 2>/dev/null || true

put_if_absent() {
  local key="$1"; local val="$2"
  # We can't "get-secret" the value back (write-only), so always re-put.
  # This is idempotent — overwriting with the same value is a no-op.
  echo "$val" | $DBX secrets put-secret "$WB_SCOPE" "$key" --string-value "$val" >/dev/null
  echo "    seeded $WB_SCOPE/$key"
}

put_if_absent uc_catalog       "$UC_CATALOG"
put_if_absent uc_schema        "$UC_SCHEMA"
put_if_absent volume_name      "${BUNDLE_VAR_volume_name:-medical_video}"
put_if_absent hf_secret_scope  "$HF_SECRET_SCOPE"
put_if_absent hf_secret_key    "$HF_SECRET_KEY"

# ── 3. Bundle deploy ──────────────────────────────────────────────────────
echo
echo "▶️ [3/5] Bundle deploy (target=$TARGET)"
$DBX bundle deploy -t "$TARGET" --auto-approve

# ── 4. Post-deploy setup ─────────────────────────────────────────────────
echo
echo "▶️ [4/5] Running post-deploy setup (UC GRANTs + Lakebase role + dirs)"
$DBX bundle run -t "$TARGET" vlmwb_postdeploy_setup

# ── 5. Deploy + start the App ─────────────────────────────────────────────
echo
echo "▶️ [5/5] Deploying + starting the App"
$DBX bundle run -t "$TARGET" vlm_workbench_app

APP_NAME="${BUNDLE_VAR_app_name:-vlm-workbench}"
APP_URL=$($DBX apps get "$APP_NAME" --output json \
  | python3 -c "import sys, json; print(json.load(sys.stdin).get('url',''))")

echo
echo "═════════════════════════════════════════════════════════"
echo "  ✅ Deploy complete"
echo "     App:    $APP_URL"
echo "     Setup:  $APP_URL/  (open the Setup tab to verify all health checks pass)"
echo "     Debug:  $APP_URL/api/debug/lakebase  (deep Lakebase diagnostic)"
echo "═════════════════════════════════════════════════════════"

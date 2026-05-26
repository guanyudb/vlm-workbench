#!/bin/bash
# One-line deploy of vlm-workbench to a fresh Databricks workspace.
#
# Usage:
#   ./deploy.sh aws      # or azure / gcp / dev
#
# Prerequisites:
#   - Databricks CLI authenticated against the target workspace (DEFAULT_PROFILE
#     env var, or `databricks auth login`)
#   - module.env populated (copy from module.env.sample)
#   - The HF token secret scope already exists with HF_TOKEN inside it
#
# What it does:
#   1. Loads module.env (catalog, schema, warehouse_id, lakebase instance, ...)
#   2. Builds the React SPA (`build.sh`)
#   3. Runs `databricks bundle deploy -t prod_<cloud>` — creates the App,
#      schema, volume, secret scope; syncs all source code to the workspace
#   4. Sets the workbench-config secret values (catalog/schema/volume) since
#      DAB can't write secret values, only scopes
#   5. Runs the post-deploy job that GRANTs all required UC privileges to the
#      App SP and creates the Lakebase Postgres role for it
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dev|aws|azure|gcp>"
  exit 1
fi
case "$1" in
  dev)   TARGET=dev ;;
  aws)   TARGET=prod_aws ;;
  azure) TARGET=prod_azure ;;
  gcp)   TARGET=prod_gcp ;;
  *)     echo "Usage: $0 <dev|aws|azure|gcp>"; exit 1 ;;
esac

PROFILE="${DATABRICKS_CONFIG_PROFILE:-DEFAULT}"

if [ ! -f module.env ]; then
  echo "🚫 module.env not found. Copy module.env.sample, fill it in, then re-run."
  exit 1
fi
source module.env

# Validate required vars
for v in BUNDLE_VAR_uc_catalog BUNDLE_VAR_uc_schema BUNDLE_VAR_sql_warehouse_id \
         BUNDLE_VAR_lakebase_instance BUNDLE_VAR_hf_secret_scope; do
  if [ -z "${!v}" ]; then
    echo "🚫 module.env is missing $v. See module.env.sample."
    exit 1
  fi
done

echo "▶️ Building SPA"
./build.sh

echo
echo "▶️ Deploying bundle (target=$TARGET, profile=$PROFILE)"
databricks bundle deploy -t "$TARGET" --profile "$PROFILE"

echo
echo "▶️ Looking up the App SP id"
APP_NAME="${BUNDLE_VAR_app_name:-vlm-workbench}"
SP_ID=$(databricks apps get "$APP_NAME" --profile "$PROFILE" --output json \
  | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('service_principal_client_id') or d.get('service_principal_id') or '')")
if [ -z "$SP_ID" ]; then
  echo "🚫 Could not resolve App SP id for $APP_NAME. Check the deploy logs."
  exit 1
fi
echo "App SP: $SP_ID"

echo
echo "▶️ Running post-deploy setup job"
BUNDLE_JSON=$(databricks bundle summary -t "$TARGET" --profile "$PROFILE" --output json)
SETUP_JOB_ID=$(echo "$BUNDLE_JSON" \
  | python3 -c "import sys, json; d=json.load(sys.stdin); j=d.get('resources',{}).get('jobs',{}).get('vlmwb_postdeploy_setup',{}); print(j.get('id',''))")
if [ -z "$SETUP_JOB_ID" ]; then
  echo "⚠️  vlmwb_postdeploy_setup job id not found — skipping post-deploy step."
else
  RUN=$(databricks jobs run-now "$SETUP_JOB_ID" --profile "$PROFILE" --output json)
  RUN_ID=$(echo "$RUN" | python3 -c "import sys, json; print(json.load(sys.stdin).get('run_id',''))")
  echo "Post-deploy run id: $RUN_ID"
  echo "Watch progress: databricks jobs get-run $RUN_ID --profile $PROFILE"
fi

echo
echo "✅ Deploy complete."
echo "   App URL: $(databricks apps get $APP_NAME --profile $PROFILE --output json | python3 -c 'import sys, json; print(json.load(sys.stdin).get(\"url\",\"\"))')"

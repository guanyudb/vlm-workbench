#!/usr/bin/env bash
# Build the React frontend → static/ then verify the output exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "→ Installing frontend deps + building"
cd frontend
npm install
npm run build
cd ..

echo "→ Verifying static/ output"
test -f static/index.html
test -d static/assets
ls -lh static/

echo
echo "Build complete. Next:"
echo "  databricks workspace import-dir . <ws-path> --profile <p>"
echo "  databricks apps deploy <app-name> --source-code-path <ws-path> --profile <p>"

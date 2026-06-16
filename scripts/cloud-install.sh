#!/bin/bash
# Installed by SessionStart hook — runs after the repo is cloned (cloud + local).
set -euo pipefail

# Skip on local machines unless you want the same bootstrap everywhere.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

echo "==> claude-gateway: installing Python deps..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

UPLOAD_DIR="${UPLOAD_DIR:-/tmp/gateway-uploads}"
mkdir -p "$UPLOAD_DIR"

if [ -n "${API_KEY:-}" ] && [ ! -f .env ]; then
  cat > .env <<EOF
API_KEY=${API_KEY}
MAX_CONCURRENT=${MAX_CONCURRENT:-5}
TIMEOUT=${TIMEOUT:-120}
PORT=${PORT:-8000}
UPLOAD_DIR=${UPLOAD_DIR}
MAX_FILE_SIZE=${MAX_FILE_SIZE:-10485760}
EOF
fi

echo "==> claude-gateway: ready (uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000})"
exit 0

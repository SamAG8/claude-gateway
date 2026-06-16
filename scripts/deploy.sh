#!/usr/bin/env bash
# scripts/deploy.sh — update to a target commit, sync deps, restart the service.
#
# Runs ON THE SERVER. The CI/CD workflow (.github/workflows/ci-cd.yml) invokes
# it over SSH after tests pass; you can also run it by hand:
#
#   ./scripts/deploy.sh            # deploy the latest origin/main
#   ./scripts/deploy.sh <git-sha>  # deploy a specific commit
#
# Assumes the one-time server bootstrap is already done (see docs/deployment.md):
# the repo is cloned at APP_DIR, a .venv exists, .env is filled in, the `claude`
# CLI is installed + authenticated, and the claude-gateway systemd unit is
# installed and enabled. Override APP_DIR / SERVICE / VENV via env if your layout
# differs from the defaults below.
set -euo pipefail

REF="${1:-origin/main}"
APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE="${SERVICE:-claude-gateway}"
VENV="${VENV:-$APP_DIR/.venv}"

cd "$APP_DIR"

echo "==> Updating $APP_DIR to $REF ..."
git fetch --all --prune
git reset --hard "$REF"

echo "==> Syncing dependencies into $VENV ..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r requirements.txt

echo "==> Restarting $SERVICE ..."
sudo systemctl restart "$SERVICE"

# Health gate: read PORT from .env (default 8000) and poll /health. A failing
# probe exits non-zero so CI marks the deploy red instead of going quietly bad.
PORT="$(sed -n 's/^PORT=//p' .env 2>/dev/null | tr -d '[:space:]')"
PORT="${PORT:-8000}"
echo "==> Waiting for health on http://127.0.0.1:${PORT}/health ..."
for _ in $(seq 1 15); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "==> Deploy OK — $SERVICE healthy on :${PORT} ($(git rev-parse --short HEAD))"
    exit 0
  fi
  sleep 1
done

echo "!! Health check failed — $SERVICE did not respond on :${PORT}" >&2
systemctl status "$SERVICE" --no-pager --lines=30 >&2 || true
exit 1

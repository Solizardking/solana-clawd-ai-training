#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This file contains only a random local API secret; never print it in logs.
if [[ -z "${CHART_API_KEY:-}" ]]; then
  mkdir -p outputs/chart-agent
  if [[ ! -f outputs/chart-agent/api-key ]]; then
    (umask 077; .venv-charts/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > outputs/chart-agent/api-key)
  fi
  CHART_API_KEY="$(cat outputs/chart-agent/api-key)"
  export CHART_API_KEY
fi
exec .venv-charts/bin/python -m uvicorn chart_agent.server:app --host 127.0.0.1 --port "${CHART_PORT:-8090}" --no-access-log

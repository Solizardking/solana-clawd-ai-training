#!/usr/bin/env bash
set -euo pipefail

# Deploy the Model Kit site + API to a Fly Machine from model-kit/.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${FLY_APP:-solana-clawd-model-kit}"
REGION="${FLY_REGION:-iad}"

cd "$ROOT"

if ! command -v flyctl >/dev/null 2>&1; then
  echo "flyctl is required. Install: curl -L https://fly.io/install.sh | sh" >&2
  exit 1
fi

if [[ -z "${FLY_API_TOKEN:-}" ]] && ! flyctl auth whoami >/dev/null 2>&1; then
  echo "Authenticate with flyctl auth login or set FLY_API_TOKEN." >&2
  exit 1
fi

# A pasted deploy token in FLY_ORG is not an org slug; flyctl would 404.
if [[ -n "${FLY_ORG:-}" ]] && { [[ ${#FLY_ORG} -gt 64 ]] || [[ "${FLY_ORG}" == "${FLY_API_TOKEN:-}" ]]; }; then
  echo "Ignoring FLY_ORG because it does not look like an organization slug." >&2
  unset FLY_ORG
fi

if ! flyctl status --app "$APP" >/dev/null 2>&1; then
  echo "Creating Fly app $APP in $REGION"
  create_args=("$APP")
  if [[ -n "${FLY_ORG:-}" ]]; then
    create_args+=(--org "$FLY_ORG")
  fi
  flyctl apps create "${create_args[@]}"
fi

flyctl deploy "$ROOT" \
  --config "$ROOT/fly.toml" \
  --app "$APP" \
  --primary-region "$REGION" \
  --ha=false \
  --yes \
  --remote-only

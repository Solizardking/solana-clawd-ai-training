#!/usr/bin/env bash
# Launch the Solana transaction foundation CPT+SFT training job on Hugging Face Jobs.
#
# Required:
#   HF_TOKEN or a working `hf auth login`
#
# Usage:
#   bash scripts/launch_transaction_foundation_hf_job.sh a100-large 6h

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLAVOR="${1:-a100-large}"
TIMEOUT="${2:-6h}"
CONFIG="${CONFIG:-nvidia/configs/solana_tx_foundation.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/solana-tx-foundation-1.5b}"
HUB_MODEL_ID="${HUB_MODEL_ID:-solanaclawd/solana-tx-foundation-1.5b}"
RUN_NAME="${WANDB_RUN_NAME:-tx-foundation-1.5b-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-outputs/job-launches}"
LOG_FILE="${LOG_DIR}/tx-foundation-launch-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "HF_TOKEN is required, or run: hf auth login" >&2
    exit 1
  fi
  export HF_TOKEN="$(hf auth token 2>/dev/null || echo "")"
fi

echo "Launching transaction foundation HF Job on ${FLAVOR} (timeout ${TIMEOUT})"
echo "Config: ${CONFIG}"
echo "Hub model: ${HUB_MODEL_ID}"
echo "Launch log: ${LOG_FILE}"

mkdir -p "$LOG_DIR"

JOB_SECRET_ARGS=(--secrets HF_TOKEN)
JOB_ENV_ARGS=(
  --env HF_HOME=/data/hf_cache
  --env HF_DATASETS_CACHE=/data/hf_cache/datasets
  --env TRANSFORMERS_CACHE=/data/hf_cache
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  JOB_SECRET_ARGS+=(--secrets WANDB_API_KEY)
  JOB_ENV_ARGS+=(--env WANDB_PROJECT=solana-clawd-tx-foundation --env "WANDB_RUN_NAME=$RUN_NAME")
else
  echo "WANDB_API_KEY is not set; launching without W&B tracking." >&2
fi

set +e
hf jobs uv run nvidia/blueprints/transaction-foundation-model/train.py \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  "${JOB_SECRET_ARGS[@]}" \
  "${JOB_ENV_ARGS[@]}" \
  --label solana-clawd-tx-foundation \
  --detach \
  -- \
  --config "$CONFIG" \
  --stage both \
  --output-dir "$OUTPUT_DIR" \
  --hub-model-id "$HUB_MODEL_ID" \
  --push 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
  if grep -q "402 Payment Required\\|Pre-paid credit balance is insufficient" "$LOG_FILE"; then
    cat >&2 <<'EOF'

ERROR: Hugging Face Jobs launch failed because prepaid Jobs credits are insufficient.

Next actions:
  1. Add Hugging Face Jobs credits to the authenticated account.
  2. Re-run: bash scripts/launch_transaction_foundation_hf_job.sh a100-large 6h
  3. Watch:  bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>

No transaction foundation job was launched.
EOF
  else
    echo "ERROR: Hugging Face Jobs launch failed. See ${LOG_FILE}" >&2
  fi
  exit "$status"
fi

echo
echo "Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <JOB_ID> --follow"
echo "  hf jobs inspect <JOB_ID>"
echo "  bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>"
echo
echo "After completion:"
echo "  bash scripts/after_transaction_foundation_job.sh"

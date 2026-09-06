#!/usr/bin/env bash
# Stage 2: train clawd-ws live-data tool-calling on top of the stage-1 adapter.
#
# Prerequisite: stage 1 must have COMPLETED and pushed its adapter. Check with:
#   ./scripts/watch_hf_job.sh <STAGE1_JOB_ID>
#
# Required:
#   HF_TOKEN         or an existing `hf auth login` session
# Optional:
#   RESUME_ADAPTER   stage-1 adapter to continue (default below)
#   BASE_MODEL       must be the SAME base stage 1 trained against
#   DATASET_REPO     live-data dataset id
#
# Usage:
#   ./scripts/launch_nemotron35_live_data_job.sh            # h200, 1h
#   ./scripts/launch_nemotron35_live_data_job.sh h200 2h

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

FLAVOR="${1:-h200}"
TIMEOUT="${2:-1h}"
CONFIG_PATH="configs/nemotron35_live_data_lora.yaml"
TEMPLATE_PATH="configs/nemotron35_chat_template_genmask.jinja"
BASE_MODEL="${BASE_MODEL:-mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-BF16}"
RESUME_ADAPTER="${RESUME_ADAPTER:-solanaclawd/solana-clawd-nemotron35-lightning-30b-uncensored-lora}"
DATASET_REPO="${DATASET_REPO:-solanaclawd/solana-clawd-live-data}"
HUB_MODEL_ID="${HUB_MODEL_ID:-solanaclawd/solana-clawd-nemotron35-lightning-30b-live-lora}"
RUN_NAME="${WANDB_RUN_NAME:-nemotron35-live-data-lora-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$BASE_MODEL" in
  *NVFP4*)
    echo "Refusing to train on an NVFP4 checkpoint: $BASE_MODEL" >&2
    exit 1
    ;;
esac

if [[ -z "${HF_TOKEN:-}" ]]; then
  if HF_TOKEN="$(hf auth token 2>/dev/null)"; then
    export HF_TOKEN
    echo "Using Hugging Face token from existing hf auth session."
  else
    echo "HF_TOKEN is required, or run: hf auth login" >&2
    exit 1
  fi
fi

# Fail before burning GPU time if stage 1 never published its adapter.
if ! hf api "models/$RESUME_ADAPTER" >/dev/null 2>&1; then
  if ! curl -sf -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/api/models/$RESUME_ADAPTER" >/dev/null 2>&1; then
    echo "Stage-1 adapter not found on the Hub: $RESUME_ADAPTER" >&2
    echo "Wait for stage 1 to finish, or set RESUME_ADAPTER to an existing adapter." >&2
    exit 1
  fi
fi
echo "Stage-1 adapter found: $RESUME_ADAPTER"

JOB_SECRET_ARGS=(--secrets HF_TOKEN)
JOB_ENV_ARGS=(
  --env HF_HOME=/data/hf_cache
  --env HF_DATASETS_CACHE=/data/hf_cache/datasets
  --env TRANSFORMERS_CACHE=/data/hf_cache
  --env TRITON_CACHE_DIR=/data/triton_cache
)

# Local files in the arg list are uploaded flat into /data. The patched chat
# template must ride along or assistant_only_loss silently turns off and the
# model learns to emit tool results instead of fetching them.
TRAIN_ARGS=(
  --config "$CONFIG_PATH"
  --ship scripts/sft_runtime.py
  --ship scripts/qwen38_multimodal.py
  --ship "$TEMPLATE_PATH"
  --base-model "$BASE_MODEL"
  --resume-adapter "$RESUME_ADAPTER"
  --dataset-repo "$DATASET_REPO"
  --output-dir /data/outputs/nemotron35-live-data-lora
  --hub-model-id "$HUB_MODEL_ID"
  --push
  --no-quant
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  JOB_SECRET_ARGS+=(--secrets WANDB_API_KEY)
  JOB_ENV_ARGS+=(
    --env WANDB_PROJECT=solana-clawd-nemotron35
    --env "WANDB_RUN_NAME=$RUN_NAME"
  )
  TRAIN_ARGS+=(--wandb)
else
  echo "WANDB_API_KEY is not set; launching without W&B tracking." >&2
fi

echo "Launching stage-2 live-data LoRA on $FLAVOR (timeout $TIMEOUT)"
echo "  base model:  $BASE_MODEL"
echo "  resuming:    $RESUME_ADAPTER"
echo "  dataset:     $DATASET_REPO"
echo "  hub model:   $HUB_MODEL_ID"
echo

hf jobs uv run scripts/train_lora.py \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  "${JOB_SECRET_ARGS[@]}" \
  "${JOB_ENV_ARGS[@]}" \
  --label solana-clawd-live-data \
  --detach \
  -- \
  "${TRAIN_ARGS[@]}"

echo
echo "Job submitted. Monitor with:"
echo "  LABEL=solana-clawd-live-data ./scripts/watch_hf_job.sh"

#!/usr/bin/env bash
# Launch a separate NVIDIA Trading Factory LoRA training job on Hugging Face.
#
# This does not touch or cancel any currently running HF job.
#
# Required:
#   HF_TOKEN        Hugging Face token with dataset/model/job access, or
#                   an existing `hf auth login` session
#
# Optional:
#   WANDB_API_KEY  Weights & Biases API key. If absent, launch without W&B.
#
# Usage:
#   ./scripts/launch_trading_factory_hf_job.sh
#   ./scripts/launch_trading_factory_hf_job.sh a100-large 4h

set -euo pipefail

FLAVOR="${1:-a100-large}"
TIMEOUT="${2:-4h}"
DATASET_REPO="${DATASET_REPO:-solanaclawd/solana-clawd-nvidia-trading-factory-instruct}"
BASE_MODEL="${BASE_MODEL:-NousResearch/Hermes-3-Llama-3.1-8B}"
HUB_MODEL_ID="${HUB_MODEL_ID:-solanaclawd/solana-nvidia-trading-factory-8b-lora}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
RUN_NAME="${WANDB_RUN_NAME:-nvidia-trading-factory-8b-lora-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if HF_TOKEN="$(hf auth token 2>/dev/null)"; then
    export HF_TOKEN
    echo "Using Hugging Face token from existing hf auth session."
  else
    echo "HF_TOKEN is required, or run: hf auth login" >&2
    exit 1
  fi
fi

JOB_SECRET_ARGS=(--secrets HF_TOKEN)
# /data is a shared HF bucket mount across all of this account's jobs. Caching a
# 16GB base model there fails with `OSError: [Errno 5] Input/output error` on
# read-back when a concurrent job is also hammering the bucket. Default to the
# container's local disk instead; set HF_CACHE_ROOT to opt back into a shared
# path when a run genuinely needs the cache to persist between jobs.
# Keep this array non-empty: the shebang can resolve to bash 3.2 on macOS,
# where "${JOB_ENV_ARGS[@]}" on an empty array aborts under `set -u`.
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/.cache/huggingface}"
JOB_ENV_ARGS=(
  --env "HF_HOME=$HF_CACHE_ROOT"
  --env "HF_DATASETS_CACHE=$HF_CACHE_ROOT/datasets"
  --env "TRANSFORMERS_CACHE=$HF_CACHE_ROOT"
)
# `hf jobs uv run` uploads the script plus every argument that is an existing
# local file into a flat /data mount. train_lora.py imports sft_runtime and
# qwen38_multimodal from its own directory, so they must ride along via --ship
# or the job dies with ModuleNotFoundError.
TRAIN_ARGS=(
  --config configs/nvidia_trading_factory_lora_config.yaml
  --ship scripts/sft_runtime.py
  --ship scripts/qwen38_multimodal.py
  --dataset-repo "$DATASET_REPO"
  --base-model "$BASE_MODEL"
  --output-dir "${OUTPUT_DIR:-/outputs/solana-nvidia-trading-factory-8b-lora}"
  --hub-model-id "$HUB_MODEL_ID"
  --num-epochs "$NUM_EPOCHS"
  --push
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  JOB_SECRET_ARGS+=(--secrets WANDB_API_KEY)
  JOB_ENV_ARGS+=(--env WANDB_PROJECT=solana-clawd-trading-factory --env "WANDB_RUN_NAME=$RUN_NAME")
  TRAIN_ARGS+=(--wandb)
else
  echo "WANDB_API_KEY is not set; launching without W&B tracking." >&2
fi

hf jobs uv run scripts/train_lora.py \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  "${JOB_SECRET_ARGS[@]}" \
  "${JOB_ENV_ARGS[@]}" \
  --label solana-clawd-trading-factory \
  --detach \
  -- \
  "${TRAIN_ARGS[@]}"

#!/usr/bin/env bash
# Launch NVIDIA Nemotron 3.5 Lightning 30B-A3B LoRA training on Hugging Face Jobs.
#
# Required:
#   HF_TOKEN        Hugging Face token with dataset/model/job access, or an
#                   existing `hf auth login` session.
# Optional:
#   WANDB_API_KEY   Weights & Biases API key. Absent -> launch without W&B.
#   BASE_MODEL      Override the base checkpoint (see below).
#   DATASET_REPO    Override the Hub dataset id.
#   MAMBA_KERNELS=1 Also install mamba-ssm + causal-conv1d for the fused
#                   selective-scan path. Much faster, but these build from
#                   source against CUDA and can add 15-30min to job startup or
#                   fail outright. Off by default; transformers falls back to a
#                   slower pure-PyTorch mamba path.
#
# Usage:
#   ./scripts/launch_nemotron35_hf_job.sh                  # h200, 6h
#   ./scripts/launch_nemotron35_hf_job.sh h200 8h
#   BASE_MODEL=mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-BF16 \
#     ./scripts/launch_nemotron35_hf_job.sh
#
# Hardware sizing: the model is ~60GB in bf16.
#   h200      (141GB)  recommended, comfortable headroom
#   a100-large (80GB)  tight but workable at seq 4096 / batch 1
#   a100x4             use if you want a larger effective batch
#
# Monitor:
#   ./scripts/watch_hf_job.sh <JOB_ID>
#   hf jobs logs <JOB_ID> --follow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

FLAVOR="${1:-h200}"
TIMEOUT="${2:-6h}"
CONFIG_PATH="configs/nemotron35_lightning_lora.yaml"
BASE_MODEL="${BASE_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
DATASET_REPO="${DATASET_REPO:-solanaclawd/solana-clawd-repo-corpus}"
HUB_MODEL_ID="${HUB_MODEL_ID:-solanaclawd/solana-clawd-nemotron35-lightning-30b-lora}"
RUN_NAME="${WANDB_RUN_NAME:-nemotron35-lightning-30b-lora-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$BASE_MODEL" in
  *NVFP4*)
    echo "Refusing to train on an NVFP4 checkpoint." >&2
    echo "  '$BASE_MODEL' is a ModelOpt MIXED_PRECISION export (FP8 attention +" >&2
    echo "  W4A16_NVFP4 experts) built for TensorRT-LLM / vLLM inference on" >&2
    echo "  Blackwell. PEFT cannot attach LoRA adapters to it." >&2
    echo "  Use a -BF16 checkpoint instead." >&2
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

JOB_SECRET_ARGS=(--secrets HF_TOKEN)
JOB_ENV_ARGS=(
  --env HF_HOME=/data/hf_cache
  --env HF_DATASETS_CACHE=/data/hf_cache/datasets
  --env TRANSFORMERS_CACHE=/data/hf_cache
  --env TRITON_CACHE_DIR=/data/triton_cache
)
DEP_ARGS=()
if [[ "${MAMBA_KERNELS:-0}" == "1" ]]; then
  echo "Requesting fused mamba kernels (mamba-ssm, causal-conv1d)."
  DEP_ARGS+=(--with mamba-ssm --with causal-conv1d)
fi

# `hf jobs uv run` uploads the script plus every argument that is an existing
# local file into a flat /data mount. train_lora.py imports sft_runtime and
# qwen38_multimodal from its own directory, so they must ride along via --ship
# or the job dies with ModuleNotFoundError.
TRAIN_ARGS=(
  --config "$CONFIG_PATH"
  --ship scripts/sft_runtime.py
  --ship scripts/qwen38_multimodal.py
  --base-model "$BASE_MODEL"
  --dataset-repo "$DATASET_REPO"
  --output-dir /data/outputs/nemotron35-lightning-30b-clawd-lora
  --hub-model-id "$HUB_MODEL_ID"
  --num-epochs 1
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

echo "Launching Nemotron 3.5 Lightning LoRA on $FLAVOR (timeout $TIMEOUT)"
echo "  base model:  $BASE_MODEL"
echo "  dataset:     $DATASET_REPO"
echo "  config:      $CONFIG_PATH"
echo "  hub model:   $HUB_MODEL_ID"
echo

hf jobs uv run scripts/train_lora.py \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  "${JOB_SECRET_ARGS[@]}" \
  "${JOB_ENV_ARGS[@]}" \
  ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} \
  --label solana-clawd-nemotron35 \
  --detach \
  -- \
  "${TRAIN_ARGS[@]}"

echo
echo "Job submitted. Monitor with:"
echo "  hf jobs ps"
echo "  ./scripts/watch_hf_job.sh <JOB_ID>"

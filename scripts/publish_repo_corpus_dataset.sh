#!/usr/bin/env bash
# Build, process, and publish the repo-corpus dataset to the Hugging Face Hub.
#
# The corpus is derived from this repo's own knowledge directories:
#   memory/ nvidia/ ollama/ library/ data/ docs/ configs/ programs/ studio/
#
# Requires either HF_TOKEN in the environment or an existing `hf auth login`
# session. Tokens are never accepted as CLI arguments.
#
# Usage:
#   ./scripts/publish_repo_corpus_dataset.sh
#   REPO_ID=myorg/my-corpus ./scripts/publish_repo_corpus_dataset.sh
#
# To also fold in the large root-level instruction corpus (trainingday.jsonl,
# ~27k examples) so the published dataset is the one the Nemotron config trains
# on, set COMBINE=1.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${REPO_ID:-solanaclawd/solana-clawd-repo-corpus}"
JSONL="${JSONL:-data/repo_corpus_sft.jsonl}"
CARD="${CARD:-data/repo_corpus_dataset_card.md}"
COMBINE="${COMBINE:-1}"
EXTRA_INPUT="${EXTRA_INPUT:-trainingday.jsonl}"

if [[ "$COMBINE" == "1" ]]; then
  PROCESSED_DIR="${PROCESSED_DIR:-data/combined_processed}"
else
  PROCESSED_DIR="${PROCESSED_DIR:-data/repo_corpus_processed}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "HF_TOKEN is required, or run: hf auth login" >&2
    exit 1
  fi
fi

echo "[1/3] Building repo corpus from source directories"
python3 scripts/build_repo_corpus_dataset.py --output "$JSONL"

echo "[2/3] Splitting and pushing to $REPO_ID"
PREPARE_INPUTS=("$JSONL")
if [[ "$COMBINE" == "1" && -f "$EXTRA_INPUT" ]]; then
  echo "      including $EXTRA_INPUT"
  PREPARE_INPUTS+=("$EXTRA_INPUT")
fi

python3 scripts/prepare_dataset.py \
  --input "${PREPARE_INPUTS[@]}" \
  --output "$PROCESSED_DIR" \
  --train-ratio 0.9 \
  --eval-ratio 0.05 \
  --seed 42 \
  --push \
  --repo-id "$REPO_ID"

echo "[3/3] Uploading dataset card"
hf upload "$REPO_ID" "$CARD" README.md \
  --repo-type dataset \
  --commit-message "Update repo corpus dataset card"

echo
echo "Published: https://huggingface.co/datasets/$REPO_ID"
echo "Train with:"
echo "  DATASET_REPO=$REPO_ID ./scripts/launch_nemotron35_hf_job.sh h200 6h"

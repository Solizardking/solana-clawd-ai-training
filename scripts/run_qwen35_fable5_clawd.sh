#!/usr/bin/env bash
# Clawd Fable training lane.
#
# Legacy script name retained so older commands keep working.
#
# Steps:
#   1. Prep data    - combine armand0e/claude-fable-5-claude-code,
#                    Glint-Research/Fable-5-traces, and local project context
#   2. Local train  - small local smoke adapter
#   3. Cloud train  - HF Jobs adapter for solanaclawd/clawd-fable-lora
#   4. Release      - merge adapter into full solanaclawd/clawd-fable model
#
# Usage:
#   cd ai-training/
#   bash scripts/run_qwen35_fable5_clawd.sh prep
#   bash scripts/run_qwen35_fable5_clawd.sh local
#   bash scripts/run_qwen35_fable5_clawd.sh cloud
#   bash scripts/run_qwen35_fable5_clawd.sh release
#   bash scripts/run_qwen35_fable5_clawd.sh all

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CLOUD_CONFIG="configs/qwen35_fable5_clawd_lora.yaml"
LOCAL_CONFIG="configs/qwen35_fable5_clawd_lora_mac.yaml"
DATA="data/clawd_fable_sft.jsonl"
LOCAL_OUTPUT="outputs/clawd-fable-lora-local"
MERGED_OUTPUT="outputs/clawd-fable-merged"
GPU="${GPU:-a100-large}"
EPOCHS="${EPOCHS:-1}"
MAX_LOCAL_RECORDS="${MAX_LOCAL_RECORDS:-2000}"
MAX_GLINT_FILES="${MAX_GLINT_FILES:-400}"
MAX_GLINT_RECORDS="${MAX_GLINT_RECORDS:-800}"

MODE="${1:-prep}"

echo "============================================================"
echo " Clawd Fable | AliesTaha/fable-traces + Clawd + Glint"
echo "============================================================"
echo "  mode:     $MODE"
echo "  cloud:    $CLOUD_CONFIG"
echo "  local:    $LOCAL_CONFIG"
echo "  data:     $DATA"
echo "  gpu:      $GPU"
echo "  epochs:   $EPOCHS"
echo "  local records cap: $MAX_LOCAL_RECORDS"
echo "  glint files cap:   $MAX_GLINT_FILES"
echo "  glint records cap: $MAX_GLINT_RECORDS"
echo

prep_data() {
  echo "[1] Preparing ChatML dataset..."
  "$PYTHON" scripts/prepare_clawd_code_dataset.py \
    --format chatml \
    --datasets clawd-code glint local \
    --epochs "$EPOCHS" \
    --max-glint-files "$MAX_GLINT_FILES" \
    --max-glint-records "$MAX_GLINT_RECORDS" \
    --max-local-records "$MAX_LOCAL_RECORDS" \
    --output "$DATA"
  echo
}

local_train() {
  echo "[2] Local smoke train..."
  echo "  Base model: AliesTaha/fable-traces"
  echo "  Output:     $LOCAL_OUTPUT"
  "$PYTHON" scripts/train_lora.py --config "$LOCAL_CONFIG"
  echo

  local results="$LOCAL_OUTPUT/train_results.json"
  if [[ -f "$results" ]]; then
    "$PYTHON" - "$results" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
loss = data.get("train_loss")
runtime = float(data.get("train_runtime", 0) or 0)
steps_per_second = float(data.get("train_steps_per_second", 0) or 0)
steps = int(round(runtime * steps_per_second)) if runtime and steps_per_second else data.get("global_step", "?")
print(f"  loss:     {loss:.4f}" if isinstance(loss, (int, float)) else f"  loss:     {loss}")
print(f"  steps:    {steps}")
print(f"  runtime:  {runtime:.1f}s")
PY
  fi
  echo
}

cloud_train() {
  echo "[3] Launching HF Jobs adapter train on $GPU..."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "  ERROR: HF_TOKEN is not set. Export a write token first."
    exit 1
  fi

  echo "  Adapter repo: solanaclawd/clawd-fable-lora"
  echo "  Full repo:    solanaclawd/clawd-fable after merge"
  echo
  read -r -p "  Launch paid HF Job? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "  Aborted."
    exit 0
  fi

  bash scripts/launch_hf_jobs.sh "$GPU" qwen35_fable5_clawd
  echo
  echo "  Monitor: hf jobs ps"
  echo "           hf jobs logs <JOB_ID> --follow"
}

release_full_model() {
  echo "[4] Merging adapter into full Transformers model..."
  "$PYTHON" scripts/merge_lora_to_full_model.py \
    --base-model AliesTaha/fable-traces \
    --adapter "${ADAPTER:-solanaclawd/clawd-fable-lora}" \
    --output-dir "$MERGED_OUTPUT" \
    --hub-model-id solanaclawd/clawd-fable \
    "${PUSH_RELEASE:+--push}"
  echo
}

case "$MODE" in
  prep)
    prep_data
    ;;
  local|smoke)
    prep_data
    local_train
    ;;
  cloud)
    prep_data
    cloud_train
    ;;
  release)
    release_full_model
    ;;
  all)
    prep_data
    local_train
    cloud_train
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Try: prep, local, cloud, release, all" >&2
    exit 1
    ;;
esac

echo "Done."

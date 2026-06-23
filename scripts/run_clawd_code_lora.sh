#!/usr/bin/env bash
# Full pipeline: DeepSolana-GPT2 × clawd-code-lora
#
# Steps:
#   1. Prep training data from armand0e/clawd-fable-5-clawd-code
#   2. Train LoRA adapter on ordlibrary/DeepSolana-GPT2
#   3. Print results summary
#
# Usage:
#   cd ai-training/
#   bash scripts/run_clawd_code_lora.sh            # full run
#   bash scripts/run_clawd_code_lora.sh --dry-run  # show what would run
#   bash scripts/run_clawd_code_lora.sh --clean    # wipe output then full run
#   EPOCHS=5 bash scripts/run_clawd_code_lora.sh   # override epoch count

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
CONFIG="configs/deepsol_clawd_code_lora_mac.yaml"
DATA="data/clawd_code_deepsol_sft.jsonl"
OUTPUT="outputs/deepsol-clawd-code-lora-mac"
LOG="outputs/deepsol_clawd_code_train.log"
EPOCHS="${EPOCHS:-3}"
DRY_RUN=0
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --clean)   CLEAN=1 ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           clawd-code-lora  ·  DeepSolana-GPT2           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  config:  $CONFIG"
echo "  data:    $DATA"
echo "  output:  $OUTPUT"
echo "  python:  $PYTHON"
echo "  epochs:  $EPOCHS"
echo

# ── Step 0: clean ────────────────────────────────────────────────────────────
if [[ "$CLEAN" == "1" ]]; then
  echo "[0/3] Cleaning previous output..."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] would rm -rf $OUTPUT $LOG"
  else
    rm -rf "$OUTPUT" "$LOG"
    echo "  Cleaned."
  fi
  echo
fi

# ── Step 1: prep data ────────────────────────────────────────────────────────
echo "[1/3] Preparing clawd-code training data..."
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  [dry-run] $PYTHON scripts/prepare_clawd_code_dataset.py --epochs $EPOCHS"
else
  "$PYTHON" scripts/prepare_clawd_code_dataset.py --epochs "$EPOCHS"
fi
echo

# ── Step 2: train ────────────────────────────────────────────────────────────
echo "[2/3] Training LoRA adapter..."
echo "  (progress bar below — Ctrl+C to stop)"
echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  [dry-run] $PYTHON scripts/train_lora.py --config $CONFIG --dry-run"
else
  "$PYTHON" scripts/train_lora.py --config "$CONFIG" 2>&1 | tee "$LOG"
fi
echo

# ── Step 3: results ──────────────────────────────────────────────────────────
echo "[3/3] Results"
RESULTS="$OUTPUT/train_results.json"
if [[ -f "$RESULTS" ]]; then
  "$PYTHON" - "$RESULTS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  loss:           {d.get('train_loss', '?'):.4f}")
print(f"  token_accuracy: {d.get('mean_token_accuracy', 0)*100:.1f}%")
print(f"  runtime:        {d.get('train_runtime', 0)/60:.1f} min")
print(f"  samples/sec:    {d.get('train_samples_per_second', '?'):.1f}")
PY
else
  echo "  (no results file — training may have been interrupted)"
fi

echo
echo "  adapter saved → $OUTPUT/adapter_model.safetensors"
echo
echo "  Next steps:"
echo "    Push to Hub:  hf upload model ordlibrary/DeepSolana-GPT2-clawd-code-lora $OUTPUT"
echo "    Run eval:     python3 scripts/evaluate.py --config $CONFIG"
echo "    Ollama:       see ollama/Modelfile.clawd-code"

#!/usr/bin/env bash
# Qwen3.5-9B-Fable5 + clawd-code + Glint traces → cloud LoRA
#
# Steps:
#   1. Prep data   — merge armand0e/claude-fable-5-claude-code + Glint-Research/Fable-5-traces
#                    in ChatML format
#   2. Smoke test  — 20-step local MPS run to verify pipeline (optional)
#   3. Cloud train — launch on HF Jobs A100-80GB (~1h for full 3-epoch run)
#   4. Monitor     — tail the job logs
#
# Usage:
#   cd ai-training/
#   bash scripts/run_qwen35_fable5_clawd.sh prep        # only prep data
#   bash scripts/run_qwen35_fable5_clawd.sh smoke       # prep + local 20-step smoke
#   bash scripts/run_qwen35_fable5_clawd.sh cloud       # prep + launch HF Jobs
#   bash scripts/run_qwen35_fable5_clawd.sh all         # prep + smoke + cloud

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
CLOUD_CONFIG="configs/qwen35_fable5_clawd_lora.yaml"
SMOKE_CONFIG="configs/qwen35_fable5_clawd_lora_mac.yaml"
DATA="data/clawd_fable5_qwen35_sft.jsonl"
GPU="${GPU:-a100-large}"

MODE="${1:-prep}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║    Qwen3.5-9B-Fable5  ·  clawd + glint  ·  LoRA         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  mode:    $MODE"
echo "  cloud:   $CLOUD_CONFIG"
echo "  smoke:   $SMOKE_CONFIG"
echo "  data:    $DATA"
echo "  gpu:     $GPU"
echo

# ── Step 1: Prep data ─────────────────────────────────────────────────────────
if [[ "$MODE" == "prep" || "$MODE" == "smoke" || "$MODE" == "cloud" || "$MODE" == "all" ]]; then
  echo "[1] Preparing ChatML dataset (clawd-code + Glint Fable-5 traces)..."
  "$PYTHON" scripts/prepare_clawd_code_dataset.py \
    --format chatml \
    --datasets clawd-code glint \
    --epochs 1 \
    --output "$DATA"
  echo
fi

# ── Step 2: Smoke test (local MPS) ────────────────────────────────────────────
if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  echo "[2] Local MPS smoke test (20 steps)..."
  echo "  NOTE: First run downloads Qwen3.5-9B (~18GB) — this takes a few minutes."
  "$PYTHON" scripts/train_lora.py --config "$SMOKE_CONFIG"
  echo
  echo "  Smoke test results:"
  SMOKE_RESULTS="outputs/qwen35-fable5-clawd-lora-mac-smoke/train_results.json"
  if [[ -f "$SMOKE_RESULTS" ]]; then
    "$PYTHON" - "$SMOKE_RESULTS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"    loss:     {d.get('train_loss', '?'):.4f}")
print(f"    steps:    {int(d.get('train_steps_per_second', 0) * d.get('train_runtime', 0))}")
print(f"    runtime:  {d.get('train_runtime', 0):.1f}s")
PY
  fi
  echo
fi

# ── Step 3: Cloud launch ──────────────────────────────────────────────────────
if [[ "$MODE" == "cloud" || "$MODE" == "all" ]]; then
  echo "[3] Launching on HF Jobs ($GPU)..."

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "  ERROR: HF_TOKEN not set. Export it first:"
    echo "    export HF_TOKEN=\$(cat ~/.hf_token)"
    exit 1
  fi

  # Confirm before spending cloud compute
  echo "  Base model:    TeichAI/Qwen3.5-9B-Fable-5-v1"
  echo "  Data records:  $(wc -l < "$DATA" 2>/dev/null || echo '?')"
  echo "  GPU:           $GPU"
  echo "  Est. cost:     ~\$8-12 on A100-80GB"
  echo
  read -r -p "  Launch? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "  Aborted."
    exit 0
  fi

  bash scripts/launch_hf_jobs.sh "$GPU" qwen35_fable5_clawd
  echo
  echo "  Watch job:  bash scripts/watch_qwen35_fable5_job.sh"
  echo "  HF Hub:     https://huggingface.co/solanaclawd/Qwen3.5-9B-Fable5-clawd-lora"
fi

echo
echo "Done. Next steps:"
echo "  cloud run:  bash scripts/run_qwen35_fable5_clawd.sh cloud"
echo "  release:    bash scripts/release_clawd_code_lora.sh      (for DeepSolana-GPT2)"

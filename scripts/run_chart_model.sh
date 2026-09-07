#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_dir="${CHART_MODEL_DIR:-outputs/chart-agent/models}"
set --
if [[ -n "${CHART_LORA_FILE:-}" ]]; then
  [[ -f "$CHART_LORA_FILE" ]] || { echo "Configured LoRA file is missing" >&2; exit 1; }
  set -- --lora "$CHART_LORA_FILE" --lora-init-without-apply
fi
exec llama-server \
  "$@" \
  --model "$model_dir/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q4_K_M.gguf" \
  --mmproj "$model_dir/mmproj-F16.gguf" \
  --alias clawd-chart-fable --host 127.0.0.1 --port "${LLAMA_PORT:-8091}" \
  --ctx-size "${CHART_CONTEXT:-8192}" --n-gpu-layers "${CHART_GPU_LAYERS:-99}" \
  --parallel 1 --jinja --chat-template-kwargs '{"enable_thinking":false}'

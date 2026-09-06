# vLLM server for the merged Clawd trading factory model.
#
# Serves an OpenAI-compatible API at /v1 so the Gradio Space, agent code, and
# any OpenAI SDK client can all point at the same backend.
FROM vllm/vllm-openai:latest

ENV MODEL_ID=solanaclawd/solana-nvidia-trading-factory-8b \
    HF_HOME=/models/hf \
    VLLM_LOGGING_LEVEL=INFO

# Baking the weights in would make a ~16GB image and a slow deploy; they are
# fetched into the /models volume on first boot and cached across restarts.
EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-lc", "\
exec python3 -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_ID\" \
  --served-model-name \"$MODEL_ID\" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len ${MAX_MODEL_LEN:-8192} \
  --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} \
  --dtype bfloat16 \
  ${API_KEY:+--api-key \"$API_KEY\"} \
  --enable-auto-tool-choice --tool-call-parser hermes \
"]

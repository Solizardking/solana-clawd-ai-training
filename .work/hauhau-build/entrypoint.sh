#!/bin/sh
# Serve the exact Hugging Face GGUF:
#   https://huggingface.co/ordlibrary/hauhau-qwen36-uncensored
# Reuse llama.cpp's HF hub cache when present, otherwise curl+resume onto
# the Fly volume. llama-server does not bind :8080 until weights are mapped.
set -eu

export LLAMA_CACHE="${LLAMA_CACHE:-/models}"
mkdir -p "$LLAMA_CACHE"

# Choose a verified GGUF release. Explicit overrides remain supported.
PROFILE="${HAUHAU_MODEL_PROFILE:-qwen36}"
case "$PROFILE" in
  qwen36)
    DEFAULT_REPO="ordlibrary/hauhau-qwen36-uncensored"
    DEFAULT_QUANT="IQ2_M"
    DEFAULT_FILE="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
    DEFAULT_BYTES="11659235456"
    DEFAULT_REVISION="main"
    ;;
  trading-factory-q4|trading-factory-q5)
    DEFAULT_REPO="solanaclawd/solana-nvidia-trading-factory-8b-GGUF"
    DEFAULT_REVISION="03954ae95d9d5d5ad86ed7679afb5f120988ec2e"
    if [ "$PROFILE" = trading-factory-q4 ]; then
      DEFAULT_QUANT="Q4_K_M"
      DEFAULT_BYTES="4920740064"
    else
      DEFAULT_QUANT="Q5_K_M"
      DEFAULT_BYTES="5732993248"
    fi
    DEFAULT_FILE="solana-trading-factory-8b-${DEFAULT_QUANT}.gguf"
    ;;
  *) echo "Unknown HAUHAU_MODEL_PROFILE: $PROFILE" >&2; exit 1 ;;
esac
HF_ID="${HAUHAU_HF_ID:-${DEFAULT_REPO}:${DEFAULT_QUANT}}"
HF_REPO="${HAUHAU_HF_REPO:-${HF_ID%%:*}}"
HF_REVISION="${HAUHAU_HF_REVISION:-$DEFAULT_REVISION}"
GGUF="${HAUHAU_GGUF_FILE:-$DEFAULT_FILE}"
GGUF_BYTES="${HAUHAU_GGUF_BYTES:-$DEFAULT_BYTES}"
PORT="${PORT:-8080}"
CTX="${LLAMA_CTX:-8192}"
THREADS="${LLAMA_THREADS:-8}"
# Keep each repository/revision in its own cache to avoid stale release reuse.
if [ "$PROFILE" = qwen36 ]; then HAUHAU_LEGACY_CACHE="$LLAMA_CACHE"; fi
LLAMA_CACHE="${LLAMA_CACHE}/${HF_REPO}/${HF_REVISION}"
mkdir -p "$LLAMA_CACHE"
DEST="${LLAMA_CACHE}/${GGUF}"
URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/${GGUF}"

size_of() {
  if [ -f "$1" ]; then
    wc -c < "$1" | tr -d ' '
  else
    echo 0
  fi
}

# llama.cpp -hf stores the blob under models--<org>--<repo>/blobs/<sha>
# (exact byte size). Prefer that over a second Hugging Face download.
find_cached_gguf() {
  if [ "$(size_of "$DEST")" = "$GGUF_BYTES" ]; then
    printf '%s\n' "$DEST"
    return 0
  fi
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if [ "$(size_of "$path")" = "$GGUF_BYTES" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done <<EOF
$(find "${HAUHAU_LEGACY_CACHE:-$LLAMA_CACHE}" -type f ! -name '*.partial' 2>/dev/null)
EOF
  return 1
}

if MODEL="$(find_cached_gguf)"; then
  echo "Using cached GGUF at ${MODEL} ($(size_of "$MODEL") bytes)"
else
  echo "Downloading ${GGUF} ($(size_of "${DEST}.partial")/${GGUF_BYTES} bytes) from Hugging Face..."
  curl -L --fail --retry 12 --retry-all-errors --retry-delay 5 \
    -C - \
    -o "${DEST}.partial" \
    "$URL"
  PARTIAL="$(size_of "${DEST}.partial")"
  if [ "$PARTIAL" != "$GGUF_BYTES" ]; then
    echo "Download incomplete: got ${PARTIAL}, expected ${GGUF_BYTES}" >&2
    exit 1
  fi
  mv "${DEST}.partial" "$DEST"
  MODEL="$DEST"
  echo "Download complete: ${DEST}"
fi

SERVER=""
for candidate in /app/llama-server /llama-server llama-server; do
  if [ -x "$candidate" ]; then
    SERVER="$candidate"
    break
  fi
  if command -v "$candidate" >/dev/null 2>&1; then
    SERVER="$candidate"
    break
  fi
done
if [ -z "$SERVER" ]; then
  echo "llama-server binary not found in image" >&2
  exit 1
fi

WWW="${HAUHAU_WWW:-/www}"

set -- "$SERVER" \
  -m "$MODEL" \
  --host "${LLAMA_HOST:-0.0.0.0}" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --threads "$THREADS" \
  --jinja \
  --alias "$HF_ID"

# Holder login page at / (Clawd wallet connect). --path replaces the
# default UI; disabling the web UI unmounts that path so / 404s.
# llama-server does not parse wallets — issuance stays on the desk SIWS handler.
if [ -d "$WWW" ]; then
  set -- "$@" --path "$WWW"
fi

# Operator llama.cpp secret only (single --api-key). Per-holder clawd_sk_
# keys are issued by the gateway — never hand this Fly secret to holders.
if [ -n "${HAUHAU_API_KEY:-}" ]; then
  set -- "$@" --api-key "$HAUHAU_API_KEY"
fi

echo "Starting llama.cpp for ${HF_ID} (file=${MODEL} ctx=${CTX} threads=${THREADS})"

if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT:-}" ]; then
  OTLP_BASE="${OTEL_EXPORTER_OTLP_ENDPOINT%/}"
  case "$OTLP_BASE" in
    */v1/traces) OTLP_URL="$OTLP_BASE" ;;
    *) OTLP_URL="${OTLP_BASE}/v1/traces" ;;
  esac
  NS="$(date +%s)000000000"
  SVC="${OTEL_SERVICE_NAME:-hauhau}"
  curl -fsS -m 2 -X POST "$OTLP_URL" \
    -H "content-type: application/json" \
    -d "{\"resourceSpans\":[{\"resource\":{\"attributes\":[{\"key\":\"service.name\",\"value\":{\"stringValue\":\"${SVC}\"}}]},\"scopeSpans\":[{\"scope\":{\"name\":\"solgpt.hauhau\"},\"spans\":[{\"traceId\":\"$(openssl rand -hex 16 2>/dev/null || echo 00000000000000000000000000000001)\",\"spanId\":\"$(openssl rand -hex 8 2>/dev/null || echo 0000000000000001)\",\"name\":\"hauhau.boot\",\"kind\":1,\"startTimeUnixNano\":\"${NS}\",\"endTimeUnixNano\":\"${NS}\",\"attributes\":[{\"key\":\"hauhau.model\",\"value\":{\"stringValue\":\"${HF_ID}\"}}]}]}]}]}" \
    >/dev/null 2>&1 || true
fi

exec "$@"

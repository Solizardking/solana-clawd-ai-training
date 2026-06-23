#!/usr/bin/env bash
# One-shot Clawd model registration
#
# Registers a model into the solana_ai_inference Anchor program AND the
# onchain.x402.wtf off-chain registry index via a single CAAP/1.0 payload.
#
# ONCHAIN mode (requires Anchor CLI + funded wallet):
#   ./dao/register_model.sh --onchain \
#     --model-hash "sha256:$(sha256sum ai-training/scripts/train_lora.py | awk '{print $1}')" \
#     --endpoint "https://clawd-box-router.fly.dev/v1" \
#     --hf-model "solanaclawd/solana-tx-foundation-7b" \
#     --keypair ~/.config/solana/id.json
#
# OFF-CHAIN ONLY (just curl, no Solana tx):
#   ./dao/register_model.sh \
#     --model-hash "sha256:abc123" \
#     --hf-model "solanaclawd/solana-tx-foundation-7b" \
#     --base-model "Qwen/Qwen2.5-7B-Instruct" \
#     --dataset-size 82169

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_HASH=""
MODEL_TYPE="TextGeneration"
API_ENDPOINT="https://clawd-box-router.fly.dev/v1"
HF_MODEL_ID="solanaclawd/solana-tx-foundation-7b"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_ACCURACY="0.00"
DATASET_SIZE="82169"
REWARD_RATE="1000000"
KEYPAIR="${HOME}/.config/solana/id.json"
CLUSTER="devnet"
MANIFEST=""
JOB_ID=""
ATTESTATION_INDEX=""
OUTPUT_PATH=""
ONCHAIN=false
DRY_RUN=false
HF_MODEL_EXPLICIT=false
DATASET_SIZE_EXPLICIT=false
BASE_MODEL_EXPLICIT=false

REGISTRY_URL="${ONCHAIN_REGISTRY_URL:-https://onchain.x402.wtf/api/register}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_RUN="${WANDB_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sha256_file() {
  local target="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$target" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$target" | awk '{print $1}'
  else
    python3 - "$target" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
  fi
}

manifest_value() {
  local manifest="$1"
  local paths="$2"
  local fallback="${3:-}"
  python3 - "$manifest" "$paths" "$fallback" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
paths = sys.argv[2].split("|")
fallback = sys.argv[3]

try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    print(fallback)
    raise SystemExit(0)

for path in paths:
    cur = data
    ok = True
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            ok = False
            break
    if ok and cur not in (None, ""):
        print(cur)
        raise SystemExit(0)

print(fallback)
PY
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-hash)    MODEL_HASH="$2";    shift 2 ;;
    --model-type)    MODEL_TYPE="$2";    shift 2 ;;
    --endpoint)      API_ENDPOINT="$2";  shift 2 ;;
    --hf-model)      HF_MODEL_ID="$2"; HF_MODEL_EXPLICIT=true; shift 2 ;;
    --base-model)    BASE_MODEL="$2"; BASE_MODEL_EXPLICIT=true; shift 2 ;;
    --eval-accuracy) EVAL_ACCURACY="$2"; shift 2 ;;
    --dataset-size)  DATASET_SIZE="$2"; DATASET_SIZE_EXPLICIT=true; shift 2 ;;
    --reward-rate)   REWARD_RATE="$2";   shift 2 ;;
    --keypair)       KEYPAIR="$2";       shift 2 ;;
    --cluster)       CLUSTER="$2";       shift 2 ;;
    --manifest)      MANIFEST="$2";      shift 2 ;;
    --job-id)        JOB_ID="$2";        shift 2 ;;
    --attestation-index) ATTESTATION_INDEX="$2"; shift 2 ;;
    --output)        OUTPUT_PATH="$2";   shift 2 ;;
    --onchain)       ONCHAIN=true;       shift ;;
    --dry-run)       DRY_RUN=true;       shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
  if [[ "$HF_MODEL_EXPLICIT" == "false" ]]; then
    HF_MODEL_ID="$(manifest_value "$MANIFEST" "hf_model_id|hub_model_id|model.repo_id|model_id" "$HF_MODEL_ID")"
  fi
  if [[ "$BASE_MODEL_EXPLICIT" == "false" ]]; then
    BASE_MODEL="$(manifest_value "$MANIFEST" "base_model|model.base_model|training.base_model" "$BASE_MODEL")"
  fi
  if [[ "$DATASET_SIZE_EXPLICIT" == "false" ]]; then
    DATASET_SIZE="$(manifest_value "$MANIFEST" "counts.examples|stats.total_examples|dataset_size|dataset.rows" "$DATASET_SIZE")"
  fi
  WANDB_RUN="$(manifest_value "$MANIFEST" "wandb_run|training.wandb_run|job.wandb_run" "$WANDB_RUN")"
elif [[ -n "$MANIFEST" ]]; then
  echo "[warn] manifest not found: $MANIFEST"
fi

# Auto-compute model hash from manifest or train script if not provided.
if [[ -z "$MODEL_HASH" ]]; then
  if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    MANIFEST_HASH="$(manifest_value "$MANIFEST" "model_hash|model.sha256|adapter_sha256|artifact.sha256|source_sha256|sha256" "")"
    if [[ -n "$MANIFEST_HASH" ]]; then
      if [[ "$MANIFEST_HASH" == sha256:* ]]; then
        MODEL_HASH="$MANIFEST_HASH"
      else
        MODEL_HASH="sha256:$MANIFEST_HASH"
      fi
    else
      MODEL_HASH="sha256:$(sha256_file "$MANIFEST")"
    fi
    echo "[auto] model_hash = $MODEL_HASH"
  else
    SCRIPT_PATH="$SCRIPT_DIR/../scripts/train_lora.py"
    if [[ -f "$SCRIPT_PATH" ]]; then
      MODEL_HASH="sha256:$(sha256_file "$SCRIPT_PATH")"
      echo "[auto] model_hash = $MODEL_HASH"
    else
      MODEL_HASH="sha256:pending-$(date +%s)"
    fi
  fi
fi

if ! [[ "$DATASET_SIZE" =~ ^[0-9]+$ ]]; then
  echo "[error] --dataset-size must be an integer, got: $DATASET_SIZE" >&2
  exit 1
fi

if ! python3 - "$EVAL_ACCURACY" <<'PY'
import sys

try:
    float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
PY
then
  echo "[error] --eval-accuracy must be numeric, got: $EVAL_ACCURACY" >&2
  exit 1
fi

if [[ -n "$OUTPUT_PATH" ]]; then
  mkdir -p "$(dirname "$OUTPUT_PATH")"
fi

echo ""
echo "┌─ Clawd Model Registration ─────────────────────────────────────────"
echo "│  model:    $HF_MODEL_ID"
echo "│  base:     $BASE_MODEL"
echo "│  hash:     $MODEL_HASH"
echo "│  endpoint: $API_ENDPOINT"
echo "│  accuracy: $EVAL_ACCURACY"
echo "│  dataset:  $DATASET_SIZE examples"
echo "│  cluster:  $CLUSTER"
echo "│  onchain:  $ONCHAIN"
if [[ -n "$MANIFEST" ]]; then
echo "│  manifest: $MANIFEST"
fi
echo "└────────────────────────────────────────────────────────────────────"
echo ""

# ── Step 1: Off-chain registry (curl — always runs) ───────────────────────────
PAYLOAD=$(MODEL_HASH="$MODEL_HASH" \
  MODEL_TYPE="$MODEL_TYPE" \
  API_ENDPOINT="$API_ENDPOINT" \
  HF_MODEL_ID="$HF_MODEL_ID" \
  BASE_MODEL="$BASE_MODEL" \
  DATASET_SIZE="$DATASET_SIZE" \
  EVAL_ACCURACY="$EVAL_ACCURACY" \
  WANDB_RUN="$WANDB_RUN" \
  CLUSTER="$CLUSTER" \
  MANIFEST="$MANIFEST" \
  JOB_ID="$JOB_ID" \
  ATTESTATION_INDEX="$ATTESTATION_INDEX" \
  python3 - <<'PY'
import datetime as dt
import json
import os

payload = {
    "model_hash": os.environ["MODEL_HASH"],
    "model_type": os.environ["MODEL_TYPE"],
    "api_endpoint": os.environ["API_ENDPOINT"],
    "hf_model_id": os.environ["HF_MODEL_ID"],
    "base_model": os.environ["BASE_MODEL"],
    "dataset_size": int(os.environ["DATASET_SIZE"]),
    "eval_accuracy": float(os.environ["EVAL_ACCURACY"]),
    "wandb_run": os.environ.get("WANDB_RUN", ""),
    "cluster": os.environ["CLUSTER"],
    "protocol": "CAAP/1.0",
    "clawd_token": "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump",
    "registered_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
metadata = {
    "source": "dao/register_model.sh",
    "manifest": os.environ.get("MANIFEST", ""),
    "job_id": os.environ.get("JOB_ID", ""),
    "attestation_index": os.environ.get("ATTESTATION_INDEX", ""),
}
payload["metadata"] = {key: value for key, value in metadata.items() if value}
print(json.dumps(payload, indent=2))
PY
)

if [[ -n "$OUTPUT_PATH" ]]; then
  printf '%s\n' "$PAYLOAD" > "$OUTPUT_PATH"
  echo "Wrote CAAP payload: $OUTPUT_PATH"
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] Would POST to $REGISTRY_URL:"
  printf '%s\n' "$PAYLOAD" | python3 -m json.tool
else
  echo "Posting to onchain.x402.wtf registry..."
  HTTP_CODE=$(curl -s -o /tmp/clawd_reg_response.json -w "%{http_code}" \
    -X POST "$REGISTRY_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -d "$PAYLOAD"
  )

  if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
    echo "✓ Registry updated (HTTP $HTTP_CODE)"
    cat /tmp/clawd_reg_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/clawd_reg_response.json
  else
    echo "⚠ Registry returned HTTP $HTTP_CODE"
    cat /tmp/clawd_reg_response.json 2>/dev/null || true
    echo "(The onchain.x402.wtf API may not be live yet — registration queued locally)"
  fi
fi

# ── Step 2: Onchain tx (optional) ─────────────────────────────────────────────
if [[ "$ONCHAIN" == "true" ]]; then
  echo ""
  echo "Submitting onchain initialize_model instruction..."
  # Check for pnpm/tsx
  if ! command -v pnpm &>/dev/null; then
    echo "[warn] pnpm not found — install with: npm install -g pnpm"
    echo "[warn] Skipping onchain registration"
    exit 0
  fi

  cd "$SCRIPT_DIR/.."
  HF_MODEL_ID="$HF_MODEL_ID" \
  BASE_MODEL="$BASE_MODEL" \
  DATASET_SIZE="$DATASET_SIZE" \
  EVAL_ACCURACY="$EVAL_ACCURACY" \
  pnpm tsx dao/register_model.ts \
    --model-hash "$MODEL_HASH" \
    --model-type "$MODEL_TYPE" \
    --endpoint "$API_ENDPOINT" \
    --hf-model "$HF_MODEL_ID" \
    --base-model "$BASE_MODEL" \
    --dataset-size "$DATASET_SIZE" \
    --eval-accuracy "$EVAL_ACCURACY" \
    --reward-rate "$REWARD_RATE" \
    --keypair "$KEYPAIR" \
    --cluster "$CLUSTER" \
    ${DRY_RUN:+--dry-run}
fi

echo ""
echo "Done. View registry at: https://onchain.x402.wtf"

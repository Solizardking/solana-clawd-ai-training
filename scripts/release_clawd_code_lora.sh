#!/usr/bin/env bash
# Release pipeline: DeepSolana-GPT2 clawd-code LoRA
#
# Steps:
#   1. Eval — run evaluate.py against the local adapter
#   2. Merge — merge LoRA into base weights, save full model
#   3. GGUF  — convert merged model to Q8_0 GGUF via llama.cpp
#   4. Ollama — create local ollama model and smoke-test it
#   5. Hub   — push adapter to ordlibrary/DeepSolana-GPT2-clawd-code-lora
#
# Usage:
#   cd ai-training/
#   bash scripts/release_clawd_code_lora.sh           # all steps
#   bash scripts/release_clawd_code_lora.sh --skip-eval
#   bash scripts/release_clawd_code_lora.sh --dry-run

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
ADAPTER="outputs/deepsol-clawd-code-lora-mac"
MERGED="outputs/deepsol-clawd-code-merged"
GGUF_DIR="ollama"
GGUF_FILE="$GGUF_DIR/deepsol-clawd-code-q8.gguf"
MODELFILE="$GGUF_DIR/Modelfile.clawd-code"
OLLAMA_NAME="deepsol-clawd-code"
HF_REPO="ordlibrary/DeepSolana-GPT2-clawd-code-lora"
CONVERT="${CONVERT:-/opt/homebrew/bin/convert_hf_to_gguf.py}"

SKIP_EVAL=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-eval) SKIP_EVAL=1 ;;
    --dry-run)   DRY_RUN=1  ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] $*"
  else
    eval "$*"
  fi
}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      clawd-code-lora  ·  Release Pipeline               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# ── Step 1: Eval (GPT-2 completion style — no chat template) ─────────────────
if [[ "$SKIP_EVAL" == "0" ]]; then
  echo "[1/5] Running eval (GPT-2 completion inference)..."
  if [[ "$DRY_RUN" == "0" ]]; then
    "$PYTHON" - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json, pathlib

PROMPTS = [
    "Human: What is a Solana program account?\n\nAssistant:",
    "Human: Write a Rust function to create a PDA.\n\nAssistant:",
    "Human: Explain the difference between lamports and SOL.\n\nAssistant:",
    "Human: How do you send a transaction on Solana using web3.js?\n\nAssistant:",
]

print("  Loading model + LoRA...")
tok = AutoTokenizer.from_pretrained("ordlibrary/DeepSolana-GPT2")
model = AutoModelForCausalLM.from_pretrained(
    "ordlibrary/DeepSolana-GPT2", torch_dtype=torch.float16, device_map="cpu"
)
model = PeftModel.from_pretrained(model, "outputs/deepsol-clawd-code-lora-mac")
model.eval()

results = []
for prompt in PROMPTS:
    t0 = time.time()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, temperature=0.4, do_sample=True,
                             pad_token_id=tok.eos_token_id)
    new_tokens = out[0][ids.shape[-1]:]
    completion = tok.decode(new_tokens, skip_special_tokens=True).split("Human:")[0].strip()
    latency = time.time() - t0
    results.append({"prompt": prompt.split("\n")[0][7:], "completion": completion, "latency_s": round(latency,2)})
    print(f"  Q: {prompt.split(chr(10))[0][7:50]}...")
    print(f"  A: {completion[:140]}")
    print()

outdir = pathlib.Path("outputs/eval-deepsol-clawd-code")
outdir.mkdir(parents=True, exist_ok=True)
json.dump(results, open(outdir / "results.json", "w"), indent=2)
print(f"  Eval results → {outdir}/results.json")
PY
  else
    echo "  [dry-run] GPT-2 completion eval skipped"
  fi
  echo
else
  echo "[1/5] Eval skipped."
  echo
fi

# ── Step 2: Merge adapter into base weights ───────────────────────────────────
echo "[2/5] Merging LoRA adapter into base weights..."
run "$PYTHON - << 'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('  Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    'ordlibrary/DeepSolana-GPT2',
    torch_dtype=torch.float16,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained('ordlibrary/DeepSolana-GPT2')

print('  Attaching LoRA...')
model = PeftModel.from_pretrained(model, 'outputs/deepsol-clawd-code-lora-mac')

print('  Merging and unloading LoRA...')
model = model.merge_and_unload()

print('  Saving merged model → outputs/deepsol-clawd-code-merged/')
model.save_pretrained('outputs/deepsol-clawd-code-merged', safe_serialization=True)
tokenizer.save_pretrained('outputs/deepsol-clawd-code-merged')
print('  Done.')
PY"
echo

# ── Step 3: Convert to GGUF ──────────────────────────────────────────────────
echo "[3/5] Converting to GGUF Q8_0..."
run "$PYTHON $CONVERT \
  $MERGED \
  --outtype q8_0 \
  --outfile $GGUF_FILE"
echo "  GGUF → $GGUF_FILE"
echo

# ── Step 4: Ollama create + smoke test ───────────────────────────────────────
echo "[4/5] Creating Ollama model '$OLLAMA_NAME'..."
run "cd $GGUF_DIR && ollama create $OLLAMA_NAME -f Modelfile.clawd-code && cd $ROOT"

echo "  Smoke-testing model..."
PROMPT="Write a Solana program that creates a counter account."
echo "  Prompt: $PROMPT"
if [[ "$DRY_RUN" == "0" ]]; then
  ollama run "$OLLAMA_NAME" "$PROMPT" --verbose 2>&1 | head -40
else
  echo "  [dry-run] ollama run $OLLAMA_NAME \"$PROMPT\""
fi
echo

# ── Step 5: Push to HF Hub ───────────────────────────────────────────────────
echo "[5/5] Pushing adapter to $HF_REPO..."
run "hf upload model $HF_REPO $ADAPTER"
echo "  Model card: https://huggingface.co/$HF_REPO"
echo

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Release complete!                                       ║"
echo "║  Ollama:  ollama run $OLLAMA_NAME                       ║"
echo "║  HF Hub:  https://huggingface.co/$HF_REPO               ║"
echo "╚══════════════════════════════════════════════════════════╝"

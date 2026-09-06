#!/usr/bin/env bash
# Preflight the Hugging Face training setup.
#
# Checks everything that can be checked before spending GPU minutes: CLI
# version, auth, python deps, config validity, dataset resolution, LoRA target
# module compatibility against the real model architecture, and Hub reachability
# of the base model and dataset.
#
# Safe to run without a token -- auth-dependent checks report SKIP instead of
# failing, so this doubles as a "what do I still need?" report.
#
# Usage:
#   ./scripts/preflight_hf_training.sh
#   ./scripts/preflight_hf_training.sh configs/nemotron35_lightning_lora.yaml

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/nemotron35_lightning_lora.yaml}"
PY="${PY:-python3}"
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

PASS=0
FAIL=0
SKIP=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); }

echo "Preflight: $CONFIG"
echo "Python:    $PY"
echo

echo "[1] Tooling"
if command -v hf >/dev/null 2>&1; then
  ok "hf CLI present ($(hf version 2>/dev/null | head -1))"
else
  bad "hf CLI missing -- pip install --upgrade huggingface_hub"
fi

if "$PY" - <<'PY' >/dev/null 2>&1
import torch, transformers, peft, trl, datasets, accelerate, yaml
PY
then
  ok "python training deps importable (torch/transformers/peft/trl/datasets)"
else
  bad "missing python deps -- $PY -m pip install -r requirements.txt"
fi
echo

echo "[2] Authentication"
if [[ -n "${HF_TOKEN:-}" ]]; then
  ok "HF_TOKEN set in environment"
elif hf auth whoami >/dev/null 2>&1; then
  ok "logged in as $(hf auth whoami 2>/dev/null | head -1)"
else
  skip "not authenticated -- run 'hf auth login' or export HF_TOKEN"
  echo "        Needed to publish datasets, launch jobs, and push adapters."
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ok "WANDB_API_KEY set (runs will be tracked)"
else
  skip "WANDB_API_KEY unset (training runs without W&B tracking)"
fi
echo

echo "[3] Config + dataset resolution"
if [[ -f "$CONFIG" ]]; then
  ok "config exists: $CONFIG"
  if "$PY" scripts/train_lora.py --config "$CONFIG" --dry-run >/tmp/preflight_dry.log 2>&1; then
    ok "dry-run resolved: $(grep -o 'train=[0-9]* eval=[0-9]*' /tmp/preflight_dry.log | tail -1)"
  else
    bad "dry-run failed -- see /tmp/preflight_dry.log"
    tail -5 /tmp/preflight_dry.log | sed 's/^/        /'
  fi
else
  bad "config not found: $CONFIG"
fi
echo

echo "[4] LoRA target modules vs. real architecture"
"$PY" - "$CONFIG" <<'PY'
import sys, warnings, yaml
warnings.filterwarnings("ignore")

cfg = yaml.safe_load(open(sys.argv[1]))
lora, mid = cfg.get("lora") or {}, cfg.get("base_model")
targets = lora.get("target_modules")
if not (mid and targets):
    print("  \033[33mSKIP\033[0m  config has no base_model/target_modules")
    sys.exit(0)
if "NVFP4" in mid:
    print(f"  \033[31mFAIL\033[0m  base_model is an NVFP4 export, not trainable: {mid}")
    sys.exit(1)
try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model

    hf = AutoConfig.from_pretrained(mid)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(hf, dtype=torch.bfloat16)
    kw = dict(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.05),
        bias=lora.get("bias", "none"), task_type=lora.get("task_type", "CAUSAL_LM"),
        target_modules=targets,
    )
    if lora.get("exclude_modules"):
        kw["exclude_modules"] = lora["exclude_modules"]
    m = get_peft_model(model, LoraConfig(**kw))
    n = sum(1 for k, _ in m.named_modules() if k.endswith("lora_A.default"))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if n == 0:
        print("  \033[31mFAIL\033[0m  target_modules matched 0 modules")
        sys.exit(1)
    print(f"  \033[32mPASS\033[0m  {mid.split('/')[-1]}: {n} adapters, {trainable:,} trainable params")
except Exception as e:
    print(f"  \033[33mSKIP\033[0m  could not build model skeleton: {type(e).__name__}: {str(e)[:110]}")
PY
echo

echo "[5] Hub reachability"
"$PY" - "$CONFIG" <<'PY'
import sys, json, urllib.request, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
checks = [("models", cfg.get("base_model")), ("datasets", cfg.get("dataset_repo"))]
for kind, rid in checks:
    if not rid:
        continue
    url = f"https://huggingface.co/api/{kind}/{rid}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        gated = d.get("gated")
        note = " (GATED - accept terms on the model page)" if gated and gated != False else ""
        print(f"  \033[32mPASS\033[0m  {kind[:-1]} reachable: {rid}{note}")
    except Exception as e:
        code = getattr(e, "code", None)
        if code == 401:
            print(f"  \033[33mSKIP\033[0m  {kind[:-1]} needs auth: {rid}")
        elif code == 404:
            print(f"  \033[31mFAIL\033[0m  {kind[:-1]} not found: {rid}")
        else:
            print(f"  \033[33mSKIP\033[0m  {kind[:-1]} check failed for {rid}: {e}")
PY
echo

echo "Summary: $PASS pass, $FAIL fail, $SKIP skip"
if (( FAIL > 0 )); then
  echo "Resolve the FAIL items before launching."
  exit 1
fi
echo "No blocking failures. Launch with:"
echo "  ./scripts/launch_nemotron35_hf_job.sh h200 6h"

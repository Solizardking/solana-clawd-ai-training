# Clawd Fable

Clawd Fable is the new Fable-trace lane for the Solana Clawd model family.

- Base model: `AliesTaha/fable-traces`
- Adapter output: `solanaclawd/clawd-fable-lora`
- Full merged model: `solanaclawd/clawd-fable`
- Training data: `armand0e/claude-fable-5-claude-code`, `Glint-Research/Fable-5-traces`, and local `ai-training` / `trading_factory` context

## Local Train

```bash
cd ai-training
bash scripts/run_qwen35_fable5_clawd.sh local
```

This writes `data/clawd_fable_sft.jsonl` and trains a small smoke adapter in
`outputs/clawd-fable-lora-local`.

## Cloud Train

```bash
cd ai-training
export HF_TOKEN=hf_...
bash scripts/run_qwen35_fable5_clawd.sh cloud
```

The cloud lane trains and publishes the LoRA adapter to
`solanaclawd/clawd-fable-lora`.

## Merge And Publish Full Model

```bash
python3 scripts/merge_lora_to_full_model.py \
  --base-model AliesTaha/fable-traces \
  --adapter solanaclawd/clawd-fable-lora \
  --output-dir outputs/clawd-fable-merged \
  --hub-model-id solanaclawd/clawd-fable \
  --push
```

After that, the final model supports the standard Transformers loader:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("solanaclawd/clawd-fable")
model = AutoModelForCausalLM.from_pretrained("solanaclawd/clawd-fable")
```

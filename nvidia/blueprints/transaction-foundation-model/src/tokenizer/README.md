# Solana trading tokenizer (Nemotron)

Extends `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` with PUMP-MCP and
SOL GPT tool names. Tokenizer files only — do not load 30B NVFP4 weights.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ordlibrary/solana-clawd-nemotron-trading-tokenizer"
)
messages = [{"role": "user", "content": "Who are you?"}]
print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
```

Train locally:

```bash
python3 scripts/train_solana_trading_tokenizer.py --push
```

Pump tape: `wss://clawd-ws.fly.dev/ws` via `python3 scripts/clawd_ws_client.py`.

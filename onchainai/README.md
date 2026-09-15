# onchainai

Catalog + studio for Solana Clawd Hugging Face models/datasets and local deploy recipes.

## Contents

- `index.html` — live catalog (loads `models.json`)
- `studio.html` — local Ollama/OpenAI-compatible chat UI
- `models.json` — snapshot of `solanaclawd` + `ordlibrary` HF artifacts
- symlinks: `deploy/chart-agent`, `deploy/nemotron`, `chart_agent`

## Featured LoRAs / weights

- `solanaclawd/solana-clawd-core-ai-1.5b-lora`
- `solanaclawd/solana-clawd-1.5b-lora`
- `solanaclawd/solana-nvidia-trading-factory-8b-lora`
- `solanaclawd/solana-nvidia-trading-factory-8b` (+ GGUF)
- `ordlibrary/deepsol-clawd-code`
- dataset `ordlibrary/solana-clawd-model-kit`

## Dev

```bash
cd onchainai && npm run dev
```

Refresh catalog:

```bash
# from this folder, re-run the HF catalog script / ask the agent to refresh models.json
```

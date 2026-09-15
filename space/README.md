---
title: Clawd Model Kit
emoji: 🦞
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - solana
  - defi
  - crypto
  - agent
  - lora
  - gguf
  - peft
  - trading
  - constitutional-ai
  - nvidia
  - fine-tuning
short_description: Solana AI — GGUF trading factory, models, model kit
---

# 🦞 Clawd Model Kit

**Solana-Native AI Agent Ecosystem**

| Tab | What's inside |
|-----|--------------|
| 🦞 Chat | Self-hosted chat (LoRA / GGUF via `CLAWD_INFERENCE_URL`) |
| 📦 GGUF | [`solana-nvidia-trading-factory-8b-GGUF`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-GGUF) — Q4_K_M + Q5_K_M |
| 📊 Benchmark | 18-MCQ Solana Knowledge Benchmark — 94.4% (17/18) |
| 🏭 Trading Factory | NVIDIA Blueprint signal discovery + Nemotron distillation |
| 🤖 Ecosystem | Models + datasets with status and links |
| 🔧 Model Kit | Fork → train → eval → register onchain |

## Featured GGUF

[`solanaclawd/solana-nvidia-trading-factory-8b-GGUF`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-GGUF)

| File | Notes |
|------|-------|
| `solana-trading-factory-8b-Q4_K_M.gguf` | Recommended for llama.cpp / Ollama |
| `solana-trading-factory-8b-Q5_K_M.gguf` | Higher fidelity |

```bash
huggingface-cli download solanaclawd/solana-nvidia-trading-factory-8b-GGUF \
  solana-trading-factory-8b-Q4_K_M.gguf --local-dir ./models
llama-server -m ./models/solana-trading-factory-8b-Q4_K_M.gguf --port 8080
```

Space secrets for live Chat:

- `CLAWD_INFERENCE_URL` — OpenAI-compatible base (e.g. `http://host:8080/v1`)
- `CLAWD_INFERENCE_MODEL` — `solana-trading-factory-8b-Q4_K_M` (or your Ollama tag)
- `CLAWD_INFERENCE_KEY` — optional
- `HF_TOKEN` — optional, hub downloads / whoami

## Models

| Model | Base | Status |
|-------|------|--------|
| [`…/solana-nvidia-trading-factory-8b-GGUF`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-GGUF) | Hermes-3-8B → GGUF | LIVE |
| [`…/solana-nvidia-trading-factory-8b-lora`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-lora) | Hermes-3-8B | LIVE |
| [`…/solana-clawd-core-ai-1.5b-lora`](https://huggingface.co/solanaclawd/solana-clawd-core-ai-1.5b-lora) | Qwen2.5-1.5B | LIVE (94.4% MCQ) |
| [`…/solana-clawd-1.5b-lora`](https://huggingface.co/solanaclawd/solana-clawd-1.5b-lora) | Qwen2.5-1.5B | LIVE |

## Sibling demo

`space/spark-4b-lora-demo` — ZeroGPU Gradio for `ordlibrary/clawd-spark-4b-lora-pilot`.

## Token

```
$CLAWD · 8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump · Solana
```

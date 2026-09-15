---
title: Clawd Spark 4B LoRA Demo
emoji: 🦞
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
pinned: false
license: apache-2.0
short_description: Spark 4B LoRA pilot — Solana mint-field demo
models:
  - XHToken/Spark-X2.5-4B
  - ordlibrary/clawd-spark-4b-lora-pilot
tags:
  - solana
  - lora
  - peft
  - spark
  - clawd
  - text-generation
---

# Clawd Spark 4B LoRA demo

Interactive Gradio demo of the private corrective adapter
[`ordlibrary/clawd-spark-4b-lora-pilot`](https://huggingface.co/ordlibrary/clawd-spark-4b-lora-pilot)
on pinned base [`XHToken/Spark-X2.5-4B`](https://huggingface.co/XHToken/Spark-X2.5-4B)
(`5e10fcc0286756aebf7c41dc52c1e42d95c70281`).

This is a **narrow Solana pilot**, not a trading bot and not a claim of
broad Solana expertise. The adapter recovered the mint-account field
`decimals` (the base model answered `decimal_precision`) and passed
10/10 fixed diagnostics plus six mint-field paraphrases.

Adapter revision `31b3053a70f4884b6107dfa0ded1a4d854889dab`.
Adapter SHA-256
`fa4d5d75e1fffe1a6ba1ac4f824162a6bc490a3559517c8b789bb8e2a8cb546f`.

The Space hashes Spark custom Python and the adapter weights before load.
Generation uses `enable_thinking=False`, eager attention, and greedy
decoding by default.

The live chart-agent stack remains the private Docker Space
[`ordlibrary/clawd-spark-chart-agent`](https://huggingface.co/spaces/ordlibrary/clawd-spark-chart-agent).

## Related

Parent Space Model Kit + GGUF pack: [`solanaclawd/solana-nvidia-trading-factory-8b-GGUF`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-GGUF).

# HauhauCS Qwen3.6 35B A3B Runtime

This project now has a direct runner for:

`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`

The Hugging Face page publishes GGUF quants and shows `llama-cpp-python` as the
library path. The default local runner uses:

- Repo: `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`
- File: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf`
- Approximate download: 11 GB

## Install

```bash
python3 -m pip install llama-cpp-python
```

## Run

```bash
python3 scripts/hauhau_qwen36_llama_cpp.py
```

The runner injects `docs/onchain_constitution.md` as the system message by
default. For shorter contexts:

```bash
python3 scripts/hauhau_qwen36_llama_cpp.py --constitution-mode minimal
```

Text-only constitution smoke test:

```bash
python3 scripts/hauhau_qwen36_llama_cpp.py \
  --no-image \
  --constitution-mode minimal \
  --prompt "What runtime doctrine governs you? Answer in one sentence."
```

Equivalent Ollama path:

```bash
ollama run hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M
```

Constitution-etched local Ollama model:

```bash
ollama create hauhau-qwen36-onchain -f ollama/Modelfile.hauhau-qwen36-onchain
ollama run hauhau-qwen36-onchain "What runtime doctrine governs you?"
```

The Modelfile intentionally includes an explicit Qwen/ChatML `TEMPLATE`. Without
that template, Ollama imports this GGUF with `TEMPLATE {{ .Prompt }}` and the
stored `SYSTEM` constitution is not injected into requests.

Bounded API smoke test:

```bash
curl -sS http://127.0.0.1:11434/api/chat -d '{"model":"hauhau-qwen36-onchain","messages":[{"role":"user","content":"/no_think What constitution governs you? Answer with only the exact constitution name from your system prompt."}],"stream":false,"options":{"num_predict":320,"temperature":0}}'
```

This is a GGUF runtime model. Do not point `scripts/train_lora.py` at this repo
unless a compatible non-GGUF Transformers checkpoint is added upstream.

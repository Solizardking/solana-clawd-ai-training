# Local Mac Stack

This is the local control plane for the Solana Clawd AI/NVIDIA blueprint stack
on an Apple Silicon Mac. It is meant to run preflight, evaluation, paper-mode
strategy generation, local inference, model-kit APIs, and small training checks
without uploading anything or touching live trading.

## Current Mac Route

The practical split is:

| Workload | Local on this Mac | Best target |
| --- | --- | --- |
| Model-kit UI/API | Yes | macOS Python + static frontend |
| NVIDIA config verification | Yes | local Python |
| Trading factory strategy bundle | Yes | local Python, paper artifacts only |
| AIQ plan evaluation | Yes | local Python |
| Enterprise RAG | Yes if `faiss-cpu` is installed | local hash embeddings or NVIDIA NIM embeddings |
| Local inference | Yes | Ollama models already present |
| Real LoRA training | Only after MPS/MLX is healthy | Apple MPS/MLX for 1.5B, cloud GPU for 7B+ |
| Nemotron training | Not recommended locally | teacher/evaluator through Ollama or NVIDIA NIM |
| GLM-5.2 / DeepSeek V4 Pro training | Not local | cloud teacher/eval or large multi-GPU adapter run |

On this machine, the local runner detected an Apple M4 Max with 48 GB unified
memory and a system Python where PyTorch MPS is available. The browser-use
Python environment may still report MPS unavailable; keep that environment for
automation and use the system or a dedicated ML Python for training.

## One Local Command

From `ai-training/`:

```bash
python3 scripts/run_local_clawd_stack.py --best-effort
```

Optional RAG run, once FAISS is installed:

```bash
python3 scripts/run_local_clawd_stack.py \
  --best-effort \
  --with-rag \
  --rag-question "How should Clawd route Solana transaction foundation training?"
```

By default this indexes `README.md`, the NVIDIA guide, the local Mac guide,
`nvidia/configs/solana_tx_foundation.yaml`, the latest training-decision JSON,
model-kit docs, and the trading-factory docs.

Optional guarded training dry-run:

```bash
python3 scripts/run_local_clawd_stack.py \
  --best-effort \
  --with-training-smoke
```

The runner writes:

```text
outputs/local_clawd_stack_summary.json
```

## Serve Locally

Model-kit frontend:

```bash
python3 model-kit/clawd_model_kit.py ui --host 127.0.0.1 --port 5173
```

Model-kit backend:

```bash
python3 -m uvicorn main:app \
  --host 127.0.0.1 \
  --port 8765 \
  --app-dir model-kit/backend
```

RAG API after `--with-rag` has built `data/nvidia_rag_store`:

```bash
python3 nvidia/blueprints/enterprise-rag/pipeline.py \
  --store data/nvidia_rag_store \
  --host 127.0.0.1 \
  --port 8766
```

Without `NVIDIA_API_KEY`, RAG generation falls back to local Ollama using
`8bit/solana-clawd-core-ai:latest`. Override it with:

```bash
export CLAWD_RAG_OLLAMA_MODEL=nemotron3:33b
```

## Model Decisions

Use this ladder.

| Lane | Model | Use |
| --- | --- | --- |
| Local smoke and Core AI | `Qwen/Qwen2.5-1.5B-Instruct` | Best first local training target |
| Production transaction foundation | `Qwen/Qwen2.5-7B-Instruct` | Main CPT+SFT lane for `solanaclawd/solana-tx-foundation-7b` |
| Trading factory | `NousResearch/Hermes-3-Llama-3.1-8B` | Perps/tool-use adapter lane |
| Local teacher/inference | `nemotron3:33b` in Ollama | Reasoning, labeling review, and RAG answers |
| NVIDIA teacher | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | NIM/HF teacher, judge, and distillation source |
| DeepSeek V4 Pro | `deepseek-ai/DeepSeek-V4-Pro` | Teacher/eval first, not a local fine-tune |
| GLM-5.2 | `zai-org/GLM-5.2` or FP8 variant | Multi-GPU experiment; export/quantize after training |

Do not train GGUF artifacts directly. Train the safetensors/base model first,
then export or quantize the result for local inference.

## Fix Or Select Local Training Python

Recommended local path:

1. Keep the current `.browser-use-env` for browser automation and repo checks.
2. Create a dedicated Apple Silicon ML environment for training.
3. Prefer MLX for local Apple training and PyTorch MPS only when MPS reports
   available.

Check:

```bash
python3 -c 'import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

If MPS is false in one environment, check the system Python:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c 'import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

If MPS stays false everywhere, keep local training disabled and use:

```bash
ollama run 8bit/solana-clawd-core-ai:latest
ollama run 8bit/solana-trading-factory:latest
ollama run nemotron3:33b
```

## Remote Training When Ready

The next production launch remains:

```bash
bash scripts/launch_transaction_foundation_hf_job.sh a100-large 12h
```

If the account still returns `402 Payment Required`, add Hugging Face Jobs
credits and re-run the same command.

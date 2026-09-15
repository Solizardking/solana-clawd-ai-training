# Hosting the Clawd Trading Factory model

One merged model feeds every surface. Build it once, serve it four ways.

## Artifacts

| Repo | What it is | Use for |
|---|---|---|
| [`solanaclawd/solana-nvidia-trading-factory-8b-lora`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-lora) | LoRA adapter, 168 MB | Further training, PEFT loading |
| [`solanaclawd/solana-nvidia-trading-factory-8b`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b) | Merged bf16 weights, 16 GB | vLLM, Inference Endpoints |
| [`solanaclawd/solana-nvidia-trading-factory-8b-GGUF`](https://huggingface.co/solanaclawd/solana-nvidia-trading-factory-8b-GGUF) | Q4_K_M ~4.9 GB, Q5_K_M ~5.7 GB | Ollama, llama.cpp, Jetson |

Base is `NousResearch/Hermes-3-Llama-3.1-8B`, so **the Llama 3.1 Community
License governs the weights** — that carries redistribution and naming
obligations the Apache-2.0 adapter does not. Read it before making the merged
repo public or serving it commercially.

Rebuild the merged model and GGUFs with:

```bash
# merge (GPU, ~3 min)
hf jobs uv run scripts/merge_lora_to_full_model.py \
  --flavor a100-large --timeout 1h --secrets HF_TOKEN \
  --env HF_HOME=/root/.cache/huggingface -- \
  --base-model NousResearch/Hermes-3-Llama-3.1-8B \
  --adapter solanaclawd/solana-nvidia-trading-factory-8b-lora \
  --output-dir /outputs/solana-nvidia-trading-factory-8b \
  --hub-model-id solanaclawd/solana-nvidia-trading-factory-8b --push
```

> After any re-merge, check `tokenizer_config.json` still says
> `"tokenizer_class": "PreTrainedTokenizerFast"` and still contains
> `chat_template`. transformers 5.x re-serializes it as `TokenizersBackend` and
> drops the template, which breaks llama.cpp conversion and older vLLM. Fix by
> re-uploading the base model's `tokenizer_config.json` verbatim — a LoRA merge
> never changes the tokenizer.

## 1. Public demo — Gradio Space (free CPU)

`solanaclawd/clawd-model-kit` runs on **cpu-basic** and is only a client, so the
demo costs nothing. It needs a backend URL because the HF Router cannot serve
LoRA adapters or the Hermes-3 base — every Clawd model returns
`model_not_supported` there.

Set these as Space secrets:

```
CLAWD_INFERENCE_URL   = https://clawd-trading-factory-vllm.fly.dev/v1
CLAWD_INFERENCE_KEY   = <your API key, if the backend requires one>
CLAWD_INFERENCE_MODEL = solanaclawd/solana-nvidia-trading-factory-8b
```

Without `CLAWD_INFERENCE_URL` the chat tab explains what to configure instead of
surfacing a raw provider error.

## 2. OpenAI-compatible API — HF Inference Endpoints

Requires a token with **Manage Inference Endpoints** scope; a plain write token
returns `missing permissions: inference.endpoints.read`.

```bash
hf endpoints deploy clawd-trading-factory \
  --repo solanaclawd/solana-nvidia-trading-factory-8b \
  --framework vllm --task text-generation \
  --accelerator gpu --vendor aws --region us-east-1 \
  --instance-type nvidia-l4 --instance-size x1 \
  --min-replica 0 --max-replica 1 --scale-to-zero-timeout 15
```

`--min-replica 0` is what keeps this from billing around the clock; expect a
cold start on the first request after idle.

## 3. Self-hosted vLLM — Fly.io

Config in [`deploy/vllm/`](../deploy/vllm/). Needs GPU access enabled on the Fly
org.

```bash
cd deploy/vllm
fly launch --no-deploy --name clawd-trading-factory-vllm
fly volumes create clawd_models --size 40 --region ord
fly secrets set HF_TOKEN=... API_KEY=...     # API_KEY optional, gates the endpoint
fly deploy
curl https://clawd-trading-factory-vllm.fly.dev/v1/models
```

Tool calling is on via `--enable-auto-tool-choice --tool-call-parser hermes`, so
the model's 13 perps tools arrive as real `tool_calls` rather than prose.

## 4. Local — Ollama

```bash
hf download solanaclawd/solana-nvidia-trading-factory-8b-GGUF \
  solana-trading-factory-8b-Q4_K_M.gguf --local-dir ollama/
cd ollama && ollama create solana-trading-factory -f Modelfile.trading-factory-finetuned
ollama run solana-trading-factory
```

Ollama also exposes an OpenAI-compatible API at `http://localhost:11434/v1`, so
it works as a `CLAWD_INFERENCE_URL` for local Space development.

## 5. Edge — NVIDIA Jetson Orin Nano

Q4_K_M is ~4.9 GB, so it fits the **8 GB** Orin Nano / Orin Nano Super with room
for an 4k context. It does **not** fit the 4 GB board — use Q3_K_M or a smaller
base there.

Orin's 8 GB is *unified* memory shared with the OS and display, so budget
conservatively: keep `num_ctx` at 4096, run headless, and expect roughly 5–10
tok/s generation.

```bash
# JetPack 6.x, CUDA-enabled llama.cpp
sudo nvpmodel -m 0 && sudo jetson_clocks          # max clocks
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87   # 87 = Orin
cmake --build build --config Release -j "$(nproc)"

hf download solanaclawd/solana-nvidia-trading-factory-8b-GGUF \
  solana-trading-factory-8b-Q4_K_M.gguf --local-dir ~/models

./build/bin/llama-server \
  -m ~/models/solana-trading-factory-8b-Q4_K_M.gguf \
  --n-gpu-layers 999 --ctx-size 4096 --host 0.0.0.0 --port 8000
```

`--n-gpu-layers 999` offloads everything to the iGPU; drop it if you hit OOM.
`llama-server` speaks the OpenAI API too, so an Orin on the LAN can back the
Space or an agent directly.

## Cost shape

| Surface | Idle | In use |
|---|---|---|
| Gradio Space (cpu-basic) | $0 | $0 |
| HF Endpoint, `--min-replica 0` | $0 | ~$0.80/hr (L4) |
| Fly vLLM, `auto_stop_machines` | volume only | ~$1.25/hr (L40S) |
| Ollama / Orin Nano | $0 | $0 |

Run **one** hosted backend and point everything at it. Two GPUs serving the same
8B model is the most common way to double this bill for no benefit.

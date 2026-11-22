---
name: local-ai-onboarding
description: Onboard any user to local AI with the Solana Model Kit — assess their hardware, pick a Clawd model that fits (Core AI 1.5B, Trading Factory 8B, or Hauhau Qwen3.6 35B-A3B), install Ollama or llama.cpp, serve it OpenAI-compatible, verify it answers, then personalize it with the Model Kit training lanes and the official Solana datasets. Use when a user or agent wants to run a Solana-native model locally, set up local inference, or fine-tune a Clawd model on their own machine.
---

# Local AI Onboarding — Solana Model Kit

Goal: take a user from zero to a working local Solana-native model, then
to a model that is *theirs* — fine-tuned with the Model Kit in this repo
(`model-kit/`) on the official datasets. Any agent can run this skill.
Everything is local-first: models run on the user's machine, keys never
leave it, and every publishing side effect is gated behind an explicit
`--yes` plus the user's own approval.

## Canonical facts

| Thing | Value |
| --- | --- |
| Training repo | https://github.com/Solizardking/solana-clawd-ai-training |
| Model Kit CLI | `model-kit/bin/clawd-model-kit` (run `doctor` first) |
| Model Kit site | https://models.x402.wtf |
| HF orgs | https://huggingface.co/solanaclawd · https://huggingface.co/ordlibrary |
| Flagship model | `ordlibrary/hauhau-qwen36-uncensored` (Qwen3.6-35B-A3B, IQ2_M GGUF, 11.66 GB) |
| $CLAWD mint | `8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump` |

## Step 1 — Assess the machine (read-only)

Record: OS + arch, total RAM (unified memory on Apple Silicon), GPU +
VRAM if any, free disk. Do not pull anything yet — model files are
multi-GB and the tier choice depends on these numbers.

## Step 2 — Pick the tier that fits

| Tier | Hardware | Model | Get it |
| --- | --- | --- | --- |
| Starter | 8–16 GB RAM, any modern laptop | Core AI 1.5B (~986 MB) | Ollama `ordlibrary/core-ai-clawd-1.5b:latest` or HF `ordlibrary/core-ai-clawd-1.5b` |
| Mid | 16–24 GB RAM / 8 GB+ VRAM | Trading Factory 8B (Q4_K_M, 4.92 GB) | HF `solanaclawd/solana-nvidia-trading-factory-8b-GGUF`; Ollama `8bit/solana-trading-factory:8b-lora-20260620` |
| Flagship | 24 GB+ unified RAM or 16 GB+ VRAM | Hauhau Qwen3.6-35B-A3B IQ2_M (11.66 GB) | HF `ordlibrary/hauhau-qwen36-uncensored`; Ollama `hf.co/ordlibrary/hauhau-qwen36-uncensored:latest` |

Never pull a multi-GB model by default — confirm the tier and free
disk with the user first.

## Step 3 — Install a runtime and serve

Ollama (simplest):

```bash
ollama pull hf.co/ordlibrary/hauhau-qwen36-uncensored:latest
ollama serve   # OpenAI-compatible API on http://127.0.0.1:11434/v1
```

llama.cpp (needed for the proven Hauhau recipe):

```bash
llama-server -m Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf \
  --alias clawd/hauhau-uncensored -c 131072 --port 8080
```

Proven recipe (Apple M4 Max, 48 GB, llama.cpp/Metal, verified
2026-11-21): **stock defaults are the optimum** — ~83.5 tok/s decode at
ctx 131072; flash-attention and KV-q8 both measured *slower*. The
recipe is filed in `sybil-solutions/local-ai-registry`
(recipe `apple-m4-max-48gb`, PR #488), weights pinned to
`ordlibrary/hauhau-qwen36-uncensored@007d4f0ccbea6236d954d0079fa7c7ea03e5f111`.

Hauhau serving note: clients must send
`chat_template_kwargs: {enable_thinking: false}` or replies can come
back empty.

## Step 4 — Verify before you celebrate

```bash
curl http://127.0.0.1:11434/v1/chat/completions -d '{
  "model": "<served-name>",
  "messages": [{"role":"user","content":"What is a PDA on Solana?"}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

A real answer about program-derived addresses = the local model works.
Record the machine, model file + SHA, context size, and measured tok/s
— that record is what a registry recipe or a training baseline is
built from. An HTTP 200 with an empty body is not a pass.

## Step 5 — Make it theirs (Model Kit training lanes)

From a checkout of this repo:

```bash
model-kit/bin/clawd-model-kit doctor
model-kit/bin/clawd-model-kit init
# user drops PDFs / JSONL / CSV / notebooks / markdown into data/incoming/
model-kit/bin/clawd-model-kit one-shot data/incoming \
  --output-prefix data/model_kit/my-run --train-dry-run
```

Train for real only after the dry run looks right:

```bash
model-kit/bin/clawd-model-kit train --lane custom \
  --dataset-path data/model_kit/my-run_processed \
  --output-dir outputs/my-solana-lora --train-dry-run   # drop the flag to run
```

### Datasets to train on (all public, chat `messages` format)

| Dataset | Use |
| --- | --- |
| `solanaclawd/solana-clawd-core-ai-instruct` (35,173 ex.) | Core Solana knowledge SFT |
| `solanaclawd/solana-clawd-realtime-research-instruct` (29,058 ex.) | Research / reasoning SFT |
| `solanaclawd/solana-tx-foundation-unified` (82,169 ex.) | Transaction CPT + SFT |
| `solanaclawd/solana-clawd-instruct` | General Solana instruct |
| `ordlibrary/solana-clawd-model-kit` | The kit: `solana_clawd_reasoning_tooling_sft.jsonl` (~133 MB) + `clawd_masterpiece_sft.jsonl` (~166 MB) as the main SFT mix, `tx_foundation_cpt_clean.jsonl` (~21 MB) as the CPT leg |
| `ordlibrary/clawd-chart-foundation-training` | 41,214 supervised rows with held-out val/test splits + 3,345 document chunks (text rows for text runs; 1,712 chart images only for vision runs) |
| `ordlibrary/clawd-agentic-layer-whitepaper` | Small doctrine leg — Clawd Agentic Layer / Six-Law alignment |

Review each dataset's manifest and quality report before training;
mixed provenance is documented, not hidden.

## Step 6 — Optional: publish and register (user-approved, `--yes`-gated)

- Upload a dataset or model to the **user's own** Hugging Face account
  (`model-kit/bin/clawd-model-kit upload ... --yes`). The user enters
  their own HF token on their own machine — never route it through
  chat or an agent tool.
- Register the model with CAAP/1.0: `clawd-model-kit register`
  dry-run first; live POST only with `--live --yes`.
- Users can also earn $CLAWD for good training data via TrainToEarn —
  see `onchainai/SKILL.md` in this repo.

## Hard rules for agents

- **Brain/Hands separation:** the model never sees a signing keypair.
  Never ask for, receive, store, or transmit seed phrases, private
  keys, wallet passwords, or HF tokens.
- **Verify, don't promise:** report measured tok/s, real endpoint
  responses, and ledger/recipe records — never projected numbers.
- **Side effects stay gated:** HF uploads, remote HF Jobs, Ollama
  pushes, and registry POSTs all require the user's explicit approval
  of that exact action, matching the kit's `--yes` gates.
- **Model outputs are not transactions.** Anything a local model
  suggests must be parsed, validated, simulated, and risk-checked by
  separate code before it touches a chain. Live trading is outside
  this skill.
- **Brand spelling:** it is **Clawd**, never "Claud".

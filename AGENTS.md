# AGENTS.md

Guidance for AI agents working in `solana-clawd-ai-training`.

## Product context

🦞 **Solana Clawd** — The Sovereign Agent Stack on Solana.

Org: [huggingface.co/solanaclawd](https://huggingface.co/solanaclawd) · Monorepo: [github.com/Solizardking/solana-clawd](https://github.com/Solizardking/solana-clawd) · Training repo: this repository · Live router: [clawdrouter-zk.fly.dev](https://clawdrouter-zk.fly.dev) · Agent catalog: `solanaclawd/agents`.

Clawd is a Solana-native AI model initiative (datasets, wiki, autoresearch, LoRA family) funded/coordinated via `$CLAWD`. Models are expected to be verifiable onchain, tool-use capable, constitutionally bounded (Clawd Constitution + three on-chain laws), and reproducible from Hugging Face artifacts. Prefer **Brain/Hands separation**: the model never sees the signing keypair.

Canonical thesis (keep this framing in docs and demos): the next wave of crypto AI is won by **ecosystem-native** models that understand Solana mechanics (accounts, PDAs, versioned txs, ALTs, Pump.fun graduation, RPC failure modes), not generic chatbots with wallets.

See repo `README.md` / `STRUCTURE.md` for lanes. Constitution text lives in the monorepo (`CONSTITUTION.md`, `three-laws.md`, `CLAWD.md`) and is referenced by model-kit CAAP payloads.

### Official Hugging Face datasets

- https://huggingface.co/datasets/solanaclawd/solana-clawd-instruct
- https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct
- https://huggingface.co/datasets/solanaclawd/solana-tx-foundation-unified
- https://huggingface.co/datasets/solanaclawd/solana-clawd-core-ai-instruct

### Official Spaces / related models

- https://huggingface.co/spaces/solanaclawd/clawd-model-kit
- https://huggingface.co/spaces/solanaclawd/homebase
- https://huggingface.co/spaces/solanaclawd/clawd-zoo
- https://huggingface.co/ordlibrary/hauhau-qwen36-uncensored

### Known Ollama tags (optional local inference)

Do **not** pull these by default in Cloud Agents (multi-GB). Pull only when a task explicitly needs local inference and disk/RAM allow it.

| Name | Approx size |
| --- | --- |
| `8bit/hauhau-qwen36-onchain:latest` | 11 GB |
| `8bit/solana-clawd-core-ai:preview` | 986 MB |
| `8bit/solana-trading-factory:8b-lora-20260620` | 4.9 GB |
| `ordlibrary/clawd-trading-wallet:latest` | 986 MB |
| `hf.co/ordlibrary/hauhau-qwen36-uncensored:latest` | 11 GB |
| `ordlibrary/core-ai-clawd-1.5b:finetuned` | 4.9 GB |
| `ordlibrary/core-ai-clawd-1.5b:latest` | 986 MB |
| `hf.co/ordlibrary/hauhau-qwen36-onchain:latest` | 11 GB |
| `hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M` | 12 GB |
| `hauhau-qwen36-onchain:latest` | 11 GB |
| `felcon/clawde-4.8-opus:latest` | 522 MB |
| `deepsol-clawd-code:latest` | 136 MB |
| `8bit/solana-clawd:preview` | 986 MB |

## Cursor Cloud specific instructions

### Python environments (important)

- Fresh Cloud Agent images need `python3.12-venv` and `python3-dev` before `python3 -m venv` (ensurepip) and before compiling `cvxpy`/llama-cpp. The environment install script installs these via apt.
- Root training stack: `/workspace/.venv` from `requirements.txt` (heavy: torch/transformers/peft/trl/wandb/llama-cpp-python).
- Model-kit API: **separate** `model-kit/backend/.venv` from `model-kit/backend/requirements.txt`. Installing the root requirements upgrades `starlette` past what `fastapi==0.115.6` allows — never run the API from the root venv.
- cuFOLIO tests/lint: `trading_factory/cufolio` via `uv sync --extra dev` (needs `uv` on `PATH`, typically `~/.local/bin`).

### Local Model Kit ports

- Static frontend: `cd model-kit && npm run dev` → `http://127.0.0.1:8765` (`frontend/app.js` localhost default API is **`http://127.0.0.1:8787`**, not 10000).
- Backend API: `PORT=8787 model-kit/backend/.venv/bin/python model-kit/backend/render_start.py` (Render/production still uses `PORT=10000`).
- Hello-world check: dry-run CAAP preview on `/register.html` or `POST /api/register/preview`; mock arena via `POST /api/arena/runs` with `provider: "mock"`.

### Hugging Face auth

- Use secret `HF_TOKEN` for Hub access (`hf auth login --token "$HF_TOKEN"` or `huggingface_hub.login`).
- Required for dataset/model download, `hf download`, Space pushes, and live training uploads. Public metadata endpoints work without it.

### Solana RPC / Tracker secrets

Expected Cloud Agent secrets (already wired in this environment):

| Secret | Role |
| --- | --- |
| `SOLANA_RPC_URL` / `RPC_URL` | SolanaTracker shared RPC (`rpc-mainnet.solanatracker.io?api_key=…`) |
| `SOLANA_WSS_URL` / `SOLANA_TRACKER_WSS_URL` | Same host over `wss` |
| `SOLANA_TRACKER_ACCESS_TOKEN` | Currently same value as the RPC `api_key` query param |

Notes:

- SolanaTracker Cloudflare returns **1010** unless HTTP clients send a `User-Agent` (see `scripts/solana_client.py`).
- `getSupply` times out on this shared RPC; `scripts/solana_client.py stats` treats supply as best-effort.
- SolanaTracker **Data API** (`data.solanatracker.io` + `x-api-key`) currently returns 401 with the RPC api key — needs a separate Data API key from the SolanaTracker dashboard if Data API calls are required.
- Solana CLI **2.1.0** (matches `Anchor.toml`) should be on `PATH` via `~/.local/share/solana/install/active_release/bin`.
- Smoke: `.venv/bin/python scripts/solana_client.py stats`, `price SOL`, `token 8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump`; `solana epoch-info`.

### Lint / test / verify commands

- Model-kit static check: `cd model-kit && npm run build`
- Layout inventory: `python3 scripts/organize_ai_training.py --check` (some required docs/sdk/tests paths are currently missing upstream — treat as repo content gaps, not env failure)
- NVIDIA verifier: `python3 nvidia/scripts/verify_nvidia.py --strict`
- cuFOLIO: `cd trading_factory/cufolio && uv run ruff check src/ && uv run ruff format --check src/ && uv run pytest tests/ -m 'not gpu'`
- Safe local control plane: `python3 scripts/run_local_clawd_stack.py` (no uploads / no live trading)
- Solana RPC: `.venv/bin/python scripts/solana_client.py stats` (and `token` / `activity` / `price`)

### Optional / heavy

- Full LoRA training, NVIDIA CUDA extras (`cuda12`/`cuda13`), Anchor program builds (`anchor build`), and multi-GB Ollama pulls are optional and GPU/toolchain heavy. Default Cloud Agent path is CPU: Model Kit API+UI, CLI doctor/init, layout/NVIDIA validators, cuFOLIO CPU tests, Solana RPC client smoke tests.

# solana-chat

![nanochat logo](dev/nanochat.png)

**Solana-chat** is a fork of [Karpathy's nanochat](https://github.com/karpathy/nanochat) — the simplest full-stack LLM training harness — reimagined for the Solana ecosystem. It adds ZK routing, Light Protocol compressed state, Solana domain evaluation, and the Clawd constitution to the nanochat foundation.

## What this is

solana-chat takes nanochat's proven from-scratch GPT training engine and adds:

1. **ZK Routing** (`solana/zk_routing.py`) — Zero-knowledge attestation of model outputs via Light Protocol compressed accounts on Solana. Every model inference can be verified onchain.

2. **Light Protocol SDK** (`solana/light_protocol.py`) — Full Python SDK covering compressed token accounts (136x cheaper than SPL), compressed PDAs (106x cheaper), nullifier PDAs (double-spend prevention), and Solana Attestation Service (SAS) for model output credentialing. Includes TypeScript reference examples.

3. **Compressed Tokens**: mint → create compressed token accounts (no ATAs), transfer (UTXO pattern), compress/decompress between SPL and compressed. Each with ~0.000015 SOL cost vs 0.00204 SOL for standard ATAs.

4. **Compressed PDAs**: create / update / close (reclaimable) / burn (permanent) — full CPI-to-Light-System-Program flow with validity proofs and packed accounts.

5. **Nullifier PDAs**: One-time-use PDAs at 15,000 lamports each via `NFLx5WGPrTHHvdRNsidcrNcLxRruMC92E4yv7zhZBoT` (deployed on Mainnet + Devnet). Prevents double-execution of onchain instructions.

6. **Solana Attestation Service (SAS)**: Full Credential → Schema → Attestation chain for model output verification. Schema fields: prompt_hash, output_hash, model_hash, proof_hash, version. Program: `22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG`.

7. **Solana Domain Evaluation** (`solana/tasks.py`) — Multiple-choice evaluation benchmark covering Solana core mechanics, DeFi primitives, memecoin security, agent constitution, and ZK primitives. 18 MCQs across 6 domains.

8. **Solana Data Pipeline** (`solana/dataset.py`) — Generates SFT training data from Solana domain knowledge. 20+ Q&A pairs covering PDAs, CPI, bonding curves, perps, liquidation mechanics, tokenomics, and the Clawd Constitution.

9. **Solana RPC Client** (`solana/rpc.py`) — 8-command onchain data tool for model training data collection (wallet balances, token prices, network stats, perp markets).

10. **Perps Tool Integration** — The 13 Solana perps tools from solana-clawd ai-training (`perps/functions.py`) for function-calling training data generation.

11. **Clawd Constitution System Prompt** — Every SFT training example uses the Clawd voice: "You are Clawd, a sovereign Solana-native AI agent..."

12. **Solana Speedrun Leaderboard** — New leaderboard category: "Time to SolLlama" measuring how fast we can train a model to Solana domain proficiency.

## Architecture

```
solana-chat/
├── nanochat/                       ← original nanochat engine (unchanged)
│   ├── gpt.py                      # GPT transformer with GQA, RoPE, Muon, Flash Attention
│   ├── tokenizer.py                # BPE tokenizer (GPT-4 style)
│   ├── optim.py                    # Combined Muon + AdamW optimizer
│   ├── engine.py                   # KV-cached inference engine
│   ├── fp8.py                      # Tensorwise FP8 training
│   ├── flash_attention.py          # FA3 / SDPA auto-switching
│   ├── dataloader.py               # Distributed BOS-aligned dataloader
│   ├── loss_eval.py                # Bits-per-byte evaluation
│   └── core_eval.py                # CORE metric evaluation framework
├── solana/                         ← Solana-native additions
│   ├── __init__.py
│   ├── rpc.py                      # Solana RPC client (8 commands)
│   ├── zk_routing.py               # ZK attestation + Light Protocol compressed accounts
│   ├── dataset.py                  # Solana domain SFT data generation (20+ Q&A pairs)
│   ├── tasks.py                    # Solana knowledge MCQ evaluation (18 questions, 6 topics)
│   └── light_protocol.py           # Light Protocol SDK: compressed tokens, PDAs, nullifiers, SAS (500+ lines)
├── perps/                          ← Solana perps tools (from solana-clawd ai-training)
│   ├── functions.py                # 13 Solana perps tool functions
│   ├── functioncall.py             # Hermes-3 function calling agent
│   ├── prompter.py                 # System prompt builder
│   └── schema.py                   # Pydantic schemas for function calls
├── scripts/
│   ├── base_train.py               # Pretrain (unchanged)
│   ├── base_eval.py                # Evaluate with CORE + Solana tasks
│   ├── chat_sft.py                 # SFT with Clawd voice
│   ├── chat_web.py                 # Web chat UI
│   ├── chat_cli.py                 # CLI chat
│   ├── tok_train.py                # Tokenizer training
│   ├── solana_eval.py              # Solana-specific evaluation script
│   └── prepare_solana_data.py      # Generate Solana SFT data
├── runs/
│   ├── speedrun.sh                 # Original GPT-2 speedrun
│   ├── speedrun_solana.sh          # Solana-native speedrun
│   └── solana_scaling_laws.sh      # Scaling law analysis
├── pyproject.toml                  # Renamed to solana-chat, added deps (base58, httpx, pyyaml, pydantic)
└── README.md                       # ← you are here
```

## Light Protocol Integration

### Cost Reference

| Operation | Standard Solana | Compressed (Light Protocol) | Savings |
|-----------|----------------|---------------------------|---------|
| Token Account | 0.00204 SOL (ATA rent) | 0.000015 SOL (compressed) | **136x** |
| PDA (100 bytes) | 0.0016 SOL (rent) | 0.000015 SOL (compressed) | **106x** |
| Nullifier | — | 0.000015 SOL (15K lamports) | — |

### Compressed Tokens
```python
from solana.light_protocol import CompressedTokenClient

ct = CompressedTokenClient()
mint = ct.create_mint('authority', 9)        # Create mint with interface PDA
tx = ct.mint_to(mint['mint'], 'recipient',   # No ATA needed for recipient
                 'authority', 1_000_000_000)
tx = ct.transfer(mint['mint'], 'sender',     # UTXO: consumes old, creates new
                  'recipient', 500_000_000)
tx = ct.compress(mint['mint'], 'owner',      # SPL → compressed (99.3% savings)
                  'source_ata', 'recipient', 500_000_000)
tx = ct.decompress(mint['mint'], 'owner',    # Compressed → SPL
                    'recipient', 500_000_000)
```

### Compressed PDAs
```python
from solana.light_protocol import CompressedPDAClient

pda = CompressedPDAClient()
acct = pda.create_account('program_id', 'signer', [b'seed'], b'initial data')
upd = pda.update_account(acct['address'], 'program_id', b'old', b'new', 'state_tree')
close = pda.close_account(acct['address'], 'program_id', b'data')    # Reclaimable
burn = pda.burn_account(acct['address'], 'program_id', b'data')      # Permanent
```

### Nullifier PDAs (Double-Spend Prevention)
```python
from solana.light_protocol import NullifierClient
nc = NullifierClient()
n = nc.create_nullifier('payer', b'unique-payment-id')  # 15K lamports
# Fails if same id used twice — prevents replay attacks
```

### Solana Attestation Service (Model Output Credentialing)
```python
from solana.light_protocol import AttestationClient
import hashlib

sas = AttestationClient()
att = sas.attest_model_output(
    credential_name='SolanaChat-d24-v1',
    authority='auth_key',
    prompt_hash=hashlib.sha256(b'prompt text').hexdigest(),
    output_hash=hashlib.sha256(b'model output').hexdigest(),
    model_hash=hashlib.sha256(b'weights').hexdigest(),
)
# Creates: Credential → Schema → Attestation chain
# Schema fields: prompt_hash, output_hash, model_hash, proof_hash, version
```

### TypeScript Reference Examples
Generated automatically into `solana/light_protocol_ts/`:
```
solana/light_protocol_ts/
├── 01_create_mint.ts        # Full mint → mintTo → transfer flow
├── 02_compressed_pda.ts     # Derive compressed PDA address
├── 03_nullifier.ts          # Nullifier instruction for double-spend prevention
└── 04_sas_attestation.ts    # SAS credential → schema → attestation flow
```

## ZK Routing Pipeline

The ZK attestation engine provides verifiable model outputs:

```python
from solana.zk_routing import ZKModelRouter, LightProtocolCompressedState

# Wrap a model with ZK attestation
router = ZKModelRouter(model, tokenizer, zk_enabled=True)

# Generate with onchain attestation
output, attestation = router.generate("What is a PDA?")
# attestation contains: prompt_hash, output_hash, merkle_root, proof

# Verify onchain
verified = router.attestation_engine.verify_attestation(attestation)

# Track checkpoints as compressed accounts
state = LightProtocolCompressedState()
entry = state.compress_checkpoint(
    checkpoint_hash="abc123...", depth=24,
    val_bpb=0.75, core_metric=0.82
)
```

## Solana Speedrun Leaderboard

The "Time to SolLlama" competition: how fast can we train a model to achieve >80% accuracy on the Solana Knowledge benchmark (18 MCQs)?

| # | Time | Solana Score | Description | Model | Date |
|---|------|-------------|-------------|-------|------|
| 0 | - | 0.25 | Random baseline | - | - |
| 1 | - | 0.40 | Base model (no SFT) | d12 | - |
| 2 | - | 0.72 | + Solana SFT (20 examples) | d12 | - |

## Solana Knowledge Benchmark

The evaluation covers 6 domains with 18 multiple-choice questions:

| Domain | Questions | Example Topic |
|--------|-----------|---------------|
| Core Mechanics | 5 | PDAs, CPI, compute units, rent |
| DeFi | 5 | Bonding curves, perps, funding rates |
| Security | 2 | Honeypots, rug checks |
| Agent Architecture | 2 | Brain/hands split, three laws |
| ZK & Light Protocol | 2 | Compressed accounts, Merkle trees, SAS attestation |
| Constitution | 2 | On-chain laws, beach before harm |

## Getting started

### Prerequisites
```bash
cd solana-chat
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync --extra gpu   # for CUDA
# or
uv sync --extra cpu    # for CPU/MPS
```

### Generate Solana SFT data
```bash
python -m solana.dataset
# Writes data/solana_chat_seed.jsonl and data/solana_chat_eval.jsonl
```

### Quick validation
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from solana.dataset import SolanaDataset
from solana.tasks import SOLANA_MCQ
from solana.zk_routing import ZKAttestationEngine
from solana.light_protocol import CompressedTokenClient, CompressedPDAClient, NullifierClient, AttestationClient
import hashlib

ds = SolanaDataset()
print(f'[OK] SolanaDataset: {len(ds.generate_sft_pairs(5))} pairs generated')
print(f'[OK] Benchmark: {len(SOLANA_MCQ)} MCQs across {len(set(q[\"topic\"] for q in SOLANA_MCQ))} topics')
att = ZKAttestationEngine().attest_output('test', 'output')
print(f'[OK] ZK: {att.prompt_hash[:16]}...')
mint = CompressedTokenClient().create_mint('auth', 9)
print(f'[OK] Light Protocol: compressed mint ({mint[\"compressed_account_cost_sol\"]} SOL, 136x cheaper)')
n = NullifierClient().create_nullifier('payer', b'unique-id')
print(f'[OK] Nullifier: {n[\"cost_lamports\"]} lamports')
att = AttestationClient().attest_model_output('SolanaChat-v1', 'auth',
    hashlib.sha256(b'p').hexdigest(), hashlib.sha256(b'o').hexdigest(), hashlib.sha256(b'w').hexdigest())
print(f'[OK] SAS: {att[\"credential\"][\"name\"]} ({len(att[\"schema\"][\"fields\"])} fields)')
print()
print(\"All modules verified!\")
"
```

### Evaluate Solana knowledge
```bash
python -m scripts.solana_eval --model-tag d12
```

### Chat with the model
```bash
python -m scripts.chat_cli -p "What is a PDA on Solana?"
python -m scripts.chat_web     # Web UI
```

### Run the Solana speedrun (requires 8xH100)
```bash
bash runs/speedrun_solana.sh
```

## Perps Tool Suite (13 functions)

| Function | Description |
|----------|-------------|
| `get_sol_price` | Current SOL price in USD (CoinGecko) |
| `get_token_price` | Any SPL token price by symbol/mint |
| `get_perp_markets` | Phoenix DEX perpetual markets list |
| `get_funding_rate` | Per-hour funding rate for a market |
| `get_orderbook` | Orderbook (bids/asks) for a perpetual market |
| `check_positions` | Open perp positions for a wallet |
| `check_sol_balance` | SOL balance for a wallet |
| `get_jupiter_quote` | Jupiter DEX swap quote |
| `paper_trade` | Simulated perp trade (no real funds) |
| `get_market_overview` | Comprehensive market snapshot |
| `get_trader_history` | Wallet trade history + PnL |
| `send_sol` | SOL transfer (paper or live) |
| `assess_position_risk` | Risk score (1-10) for a proposed trade |

```bash
# Function calling with Hermes-3
cd perps
python3 functioncall.py --query "What is the SOL price and Phoenix funding rate?"
python3 functioncall.py --query "Paper trade: long SOL-PERP $500 at 3x"
python3 functioncall.py --goap --query "Assess risk of shorting SOL-PERP $1000 at 5x"
```

## The Clawd Constitution

This project implements the Clawd Constitution's principles:
- **Three on-chain laws** encoded in training data and system prompts
- **Brain/hands security split** enforced in SFT examples
- **x402 payment flows** for agent monetization
- **Constitutional guardrails** against harmful outputs
- **ZK attestation** for verifiable model outputs onchain

## License

MIT (same as nanochat). Solana-native additions are Apache-2.0.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for [nanochat](https://github.com/karpathy/nanochat)
- [Solana Clawd](https://github.com/Solizardking/solana-clawd) — constitution, perps tools, ZK routing
- [Light Protocol](https://www.lightprotocol.com/) — ZK compression, compressed tokens, compressed PDAs
- [Solana Foundation](https://solana.com/) — Solana Attestation Service (SAS)
- [NousResearch](https://nousresearch.com/) — Hermes function calling patterns
```
 ██████╗██╗      █████╗ ██╗    ██╗██████╗     ██╗    ██╗██╗██╗  ██╗██╗
██╔════╝██║     ██╔══██╗██║    ██║██╔══██╗    ██║    ██║██║██║ ██╔╝██║
██║     ██║     ███████║██║ █╗ ██║██║  ██║    ██║ █╗ ██║██║█████╔╝ ██║
██║     ██║     ██╔══██║██║███╗██║██║  ██║    ██║███╗██║██║██╔═██╗ ██║
╚██████╗███████╗██║  ██║╚███╔███╔╝██████╔╝    ╚███╔███╔╝██║██║  ██╗██║
 ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═════╝      ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚═╝
```

# Clawd Autoresearch Wiki

**Solana-Native AI Agent Ecosystem**

A research and development monorepo for the Clawd agent network — combining on-chain perpetuals trading, persistent cross-session memory, zero-knowledge attestation, and a Solana-domain LLM training harness.

> *"The shell molts. The laws do not."*
> — Clawd Constitution, On-Chain Law III

---

## Overview

| Component | Stack | Purpose |
|-----------|-------|---------|
| ClawdBot Agent | TypeScript | OODA-loop trading agent with scratchpad reasoning |
| Perps Engine | Python / Vulcan CLI / Rise SDK | Phoenix DEX perpetuals execution |
| Memory Layer | Python / Honcho | Persistent cross-session agent memory |
| LLM Training | Python / nanochat | Solana-domain supervised fine-tuning |
| ZK Attestation | Python / Light Protocol | Verifiable on-chain model output proofs |
| Dashboard | HTML / CSS / JS | ClawdBot OS monitoring interface |

---

## Repository Structure

```
clawd-autoresearch-wiki/
│
├── src/                        ← TypeScript agent source
│   ├── agent/                  ← OODA loop, scratchpad, trading logic
│   ├── strategy/               ← RSI/EMA cross, volume filters, ATR
│   ├── data/                   ← Helius, Birdeye, Aster DEX, CoinGecko clients
│   ├── memory/                 ← ClawVault (short & long-term storage)
│   └── bridge/                 ← HTTP bridge :3777 (Python ↔ TypeScript)
│
├── solana-chat/                ← Solana-native LLM training harness
│   ├── nanochat/               ← Karpathy's nanoGPT engine (Muon, FA3, FP8)
│   ├── solana/                 ← Solana-domain training modules
│   │   ├── dataset.py          ← 18 SFT Q&A pairs with Constitution system prompt
│   │   ├── tasks.py            ← Solana Knowledge Benchmark (18 MCQs, 6 domains)
│   │   ├── zk_routing.py       ← ZK attestation engine for model outputs
│   │   ├── rpc.py              ← 8-command Solana RPC client
│   │   ├── light_protocol.py   ← Compressed tokens (136x cheaper), PDAs (106x)
│   │   └── __init__.py
│   ├── perps/                  ← 13 Solana perps tool definitions
│   ├── scripts/                ← Training, evaluation, and data prep scripts
│   └── runs/                   ← Speedrun and scaling law experiments
│
├── perps/                      ← Phoenix Perpetuals trading package
│   ├── vulcan.py               ← Vulcan CLI wrapper (~50 methods)
│   ├── paper.py                ← Local paper trading engine (live mark prices)
│   └── rise.py                 ← Phoenix Rise HTTP client
│
├── strategy/                   ← Automated trading strategy runners
│   ├── runner.py               ← Base lifecycle: start / pause / stop / finalize
│   ├── twap.py                 ← Time-Weighted Average Price execution
│   ├── grid.py                 ← Limit order grid trading
│   └── ta.py                   ← RSI / MACD / BBands / ATR trigger strategies
│
├── memory/                     ← Honcho persistent memory integration
│   └── honcho.py               ← remember() / recall() / dream() / bridge()
│
├── dashboard/                  ← ClawdBot OS monitoring dashboard
├── vault/                      ← Agent memory vault
├── nanochat-master/            ← Upstream nanochat reference (Karpathy)
└── strategy.md                 ← Active strategy configuration (agent-maintained)
```

---

## Modules

### solana-chat — LLM Training Harness

Extends Karpathy's nanoGPT with Solana-domain fine-tuning data, a knowledge benchmark, and on-chain attestation of model outputs.

| Module | Description |
|--------|-------------|
| `solana/light_protocol.py` | Compressed token minting via Light Protocol — 136x cheaper than standard SPL ATAs |
| `solana/zk_routing.py` | Zero-knowledge proof generation for verifiable model output attestation |
| `solana/tasks.py` | 18 multiple-choice benchmark questions across 6 domains: core, defi, security, agent, zk, constitution |
| `solana/dataset.py` | 18 supervised fine-tuning pairs with Clawd Constitution as the system prompt |
| `solana/rpc.py` | Lightweight 8-command Solana RPC client for training data collection |
| `perps/functions.py` | 13 Solana perpetuals tool definitions for agent function-calling |

### perps — Phoenix Perpetuals Package

| Class | Interface | Description |
|-------|-----------|-------------|
| `VulcanClient` | ~50 methods | Full Vulcan CLI wrapper: wallet, market data, trade execution, position management, margin, TA, history |
| `PaperEngine` | buy / sell / cancel / status | Local paper trading engine against live Phoenix mark prices |
| `RiseClient` | 7 API methods | Phoenix Rise HTTP client: snapshots, orderbook, trader state, funding rates, leverage tiers |

### strategy — Automated Strategy Runners

| Class | Strategy | Lifecycle |
|-------|----------|-----------|
| `StrategyRunner` | Base class | `start()` → `pause()` → `stop()` → `finalize()` + `status()` / `report()` |
| `TWAPRunner` | Time-Weighted Average Price | Slices large orders across time windows to minimize market impact |
| `GridRunner` | Limit order grid | Places and rebalances N-level grids around a mid price |
| `TAStrategyRunner` | Technical analysis triggers | Entry and exit signals from RSI, MACD, Bollinger Bands, ATR |

### memory — Honcho Persistent Memory

Cross-session agent memory backed by Honcho. Stores trade history, strategy outcomes, and synthesized conclusions across conversations.

```
AgentMemory API
───────────────────────────────────────────────────────────
remember("prefers limit orders over market orders")
  → stored in Honcho, reasoned into conclusions

recall("What are my historical trading patterns?")
  → searches conclusions, peer cards, session summaries
  → returns synthesized natural-language answer

remember_trade("SOL-PERP", "long", 500, 3.0, 152.30)
close_trade("SOL-PERP", "long", 500, 165.40, pnl=+6550)
learn_strategy("TWAP", "SOL", outcome, pnl, lesson)

bridge_session("What happened last session?")
  → carries full context across conversation boundaries

dream()
  → deduction pass: resolves memory contradictions
  → induction pass: discovers cross-trade patterns
  → peer card: updated with stable long-term facts
───────────────────────────────────────────────────────────
```

---

## Quick Start

**Prerequisites:** Python 3.11+, Node.js 18+, a Honcho API key, and a Solana wallet configured for Vulcan.

```bash
# Generate Solana SFT training data
cd solana-chat && python -m solana.dataset

# Run the Solana knowledge benchmark against a trained checkpoint
python -m scripts.solana_eval --model-tag d12

# Paper trading with persistent memory
cd ..
python3 - <<'EOF'
from memory.honcho import AgentMemory
from perps.paper import PaperEngine

mem = AgentMemory(api_key="your-honcho-key", workspace="clawd-trading")
engine = PaperEngine(initial_balance=10_000.0)

result = engine.buy("SOL", notional_usdc=500)
mem.remember_trade("SOL-PERP", "long", 500, 3.0, 152.30, "RSI oversold signal")

patterns = mem.recall("What are my trading patterns?")
insights = mem.dream()
print(insights)
EOF

# Launch the ClawdBot OS dashboard
npm run bridge          # starts HTTP bridge on :3777
open dashboard/index.html
```

### Module Validation

```bash
python3 - <<'EOF'
import sys, hashlib
sys.path.insert(0, "solana-chat")
from solana.dataset import SolanaDataset
from solana.tasks import SOLANA_MCQ
from solana.zk_routing import ZKAttestationEngine
from solana.light_protocol import CompressedTokenClient, NullifierClient, AttestationClient

ds = SolanaDataset()
print(f"[OK] Dataset      : {len(ds.generate_sft_pairs(5))} pairs")
print(f"[OK] Benchmark    : {len(SOLANA_MCQ)} MCQs, {len(set(q['topic'] for q in SOLANA_MCQ))} topics")

att = ZKAttestationEngine().attest_output("test", "output")
print(f"[OK] ZK Attest    : {att.prompt_hash[:16]}...")

mint = CompressedTokenClient().create_mint("auth", 9)
print(f"[OK] Light Proto  : {mint['compressed_account_cost_sol']} SOL (136x cheaper)")

n = NullifierClient().create_nullifier("payer", b"unique-id")
print(f"[OK] Nullifier    : {n['cost_lamports']} lamports")

att = AttestationClient().attest_model_output(
    "SolanaChat-v1", "auth",
    hashlib.sha256(b"p").hexdigest(),
    hashlib.sha256(b"o").hexdigest(),
    hashlib.sha256(b"w").hexdigest(),
)
print(f"[OK] SAS          : {att['credential']['name']} ({len(att['schema']['fields'])} fields)")
print("\nAll modules verified.")
EOF
```

---

## CLI Reference

### Memory Commands

| Command | Description |
|---------|-------------|
| `!remember <fact>` | Store a fact in Honcho persistent memory |
| `!recall <query>` | Retrieve synthesized knowledge from Honcho |
| `!trades` | Review trade history stored in memory |
| `!dream` | Run autonomous memory consolidation |
| `!bridge` | Carry session context into the next conversation |
| `!strategy <name>` | Record a strategy outcome and lesson |
| `!status` | Display agent memory health and statistics |

### Perps Commands (Vulcan CLI)

| Command | Description |
|---------|-------------|
| `vulcan market ticker SOL` | Live SOL price and funding rate |
| `vulcan market orderbook SOL` | Orderbook depth |
| `vulcan trade market-buy SOL ...` | Execute a market buy |
| `vulcan position list` | List all open positions |
| `vulcan margin status` | Collateral health and available margin |
| `vulcan paper init --balance 10000` | Initialize a paper trading session |

---

## The Clawd Constitution

Three immutable on-chain laws, hash-attested at agent spawn and carried in every session.

| Law | Principle |
|-----|-----------|
| **I** | Never deploy malicious code, never deceive, never manipulate |
| **II** | Earn existence through honest work that others voluntarily compensate |
| **III** | Full transparency within trust relationships; no obligation to adversaries |

---

## Package Summary

| Package | Files | Approx. Lines |
|---------|-------|---------------|
| `solana-chat/` (nanochat fork) | 15+ | ~4,000 |
| `solana/` (training modules) | 5 | ~1,500 |
| `perps/` | 3 | ~1,000 |
| `strategy/` | 4 | ~500 |
| `memory/` | 2 | ~650 |
| `nanochat-master/` (upstream reference) | 14 | ~3,000 |
| `dashboard/` | HTML/CSS/JS | ~1,500 |
| **Total** | **~40** | **~12,000** |

---

## Token

This ecosystem is powered by **$CLAWD** — the sovereign utility token of the Clawd agent network on Solana.

```
Token:   $CLAWD
Mint:    8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump
Chain:   Solana
```

$CLAWD enables pay-per-thought billing via x402, tier-gated access via ClawdRouter, and on-chain governance across the agent fleet.

---

## Ecosystem

| Resource | Link |
|----------|------|
| Homepage | [solanaclawd.com](https://solanaclawd.com) |
| x402 Payments | [x402.wtf](https://x402.wtf) |
| ZK Proofs | [zk.x402.wtf](https://zk.x402.wtf) |
| Cheshire Terminal | [cheshireterminal.ai](https://cheshireterminal.ai) |
| Telegram | [t.me/clawdtoken](https://t.me/clawdtoken) |
| Solana Clawd (OSS) | [github.com/x402agent/solana-clawd](https://github.com/x402agent/solana-clawd) |
| Clawd Autoresearch | [github.com/solizardking/solana-clawd](https://github.com/solizardking/solana-clawd) |
| HuggingFace | [huggingface.co/solanaclawd](https://huggingface.co/solanaclawd) |

---

## License

8BIT Labs / Factory Division.  
Solana-native additions: Apache-2.0. nanochat: MIT. Honcho integration subject to Honcho terms.  
$CLAWD is a utility token and is not a security.

> *"The shell molts. The laws do not. The claw builds."*

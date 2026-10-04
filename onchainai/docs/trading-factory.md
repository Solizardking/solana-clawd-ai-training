# Solana Clawd Trading Factory

Local integration workspace for the NVIDIA/cuFOLIO Solana spot and Phoenix
perps training lane.

## Source Snapshots

- `cufolio/` — local clone of `Solizardking/cuFOLIO`, an Apache-2.0
  GPU-accelerated portfolio optimization toolkit with scenario generation,
  CVaR optimization, rebalancing, backtesting, and notebooks.
- `clawd-autoresearch-wiki/` — local clone of
  `Solizardking/clawd-autoresearch-wiki`; the `perps/` directory is used as
  internal research/reference material for paper trading, Rise HTTP reads, and
  Vulcan wrappers.
- `solana_factory/` — Clawd-owned adapter layer that generates current Vulcan
  strategy JSON, Rise read plans, cuFOLIO Mean-CVaR handoff specs, and the
  NemoClawd/NVIDIA agent plan.

## Generate Strategy Artifacts

From `ai-training/`:

```bash
python3 scripts/build_solana_trading_factory_strategies.py
```

This writes:

- `data/strategies/sol_rsi_mean_reversion_paper.json`
- `data/strategies/sol_ema_adx_trend_paper.json`
- `data/strategies/sol_macd_adx_trim_paper.json`
- `data/strategies/cufolio_mean_cvar_handoff.json`
- `data/strategies/rise_market_data_plan.json`
- `data/strategies/vulcan_command_plans.json`
- `data/strategies/nvidia_clawd_agent_plan.json`
- `data/strategies/strategy_manifest.json`

The generator validates the full bundle after writing and exits non-zero if a
strategy config, referenced artifact, read plan, command plan, or NVIDIA agent
plan violates the paper-first contract.

Validate an existing bundle without regenerating files:

```bash
python3 scripts/build_solana_trading_factory_strategies.py --validate-only
```

Useful knobs for deterministic reviews:

```bash
python3 scripts/build_solana_trading_factory_strategies.py \
  --paper-notional-usdc 150 \
  --max-ticks 60 \
  --dry-run
```

Regenerate only the NVIDIA/NemoClawd plan:

```bash
python3 nvidia/integration/nemo_clawd_agent.py --mode paper
```

## Python API

From `ai-training/`, the package can be imported without CUDA, Vulcan, wallet
access, or network calls:

```python
import sys
from pathlib import Path

sys.path.insert(0, "trading_factory")

from solana_factory import build_strategy_bundle, validate_strategy_bundle

repo_root = Path(".").resolve()
output_dir = repo_root / "data" / "strategies"
manifest = build_strategy_bundle(repo_root=repo_root, output_dir=output_dir)
report = validate_strategy_bundle(manifest, repo_root=repo_root, output_dir=output_dir)
assert report.ok, report.errors
```

Public builders include:

- Vulcan paper TA configs and guarded paper command plans.
- Rise/Phoenix read-only market-data plans.
- cuFOLIO Mean-CVaR optimizer handoff specs.
- NVIDIA/NemoClawd agent-plan generation.
- Bundle and sub-artifact validators.

## Execution Policy

The generated artifacts are paper-first and do not execute anything.

Safe paper smoke test:

```bash
vulcan paper init --balance 10000 -o json
vulcan strategy ta start \
  --config-file data/strategies/sol_rsi_mean_reversion_paper.json \
  --mode paper \
  --max-ticks 60 \
  -o json
```

Before any live mode, run:

```bash
vulcan strategy preflight -o json
```

Live strategy launch still requires an explicit execution-mode answer from the
operator, a READY preflight, margin/position/orderbook review, and Vulcan's live
approval gate. Do not place wallet passwords, private keys, HF/W&B/NVIDIA/API
tokens, ADC JSON, or client-secret files in this directory.

## Training Use

`configs/nvidia_trading_factory_config.yaml` includes these adapter files as
local sources for `scripts/build_nvidia_trading_factory_dataset.py`. Rebuild
after changing strategies:

```bash
python3 scripts/build_solana_trading_factory_strategies.py
python3 scripts/build_nvidia_trading_factory_dataset.py
python3 scripts/prepare_dataset.py \
  --input data/nvidia_trading_factory_sft.jsonl \
  --output data/nvidia_trading_factory_processed \
  --train-ratio 0.9 --eval-ratio 0.05 --seed 42
```

The NVIDIA path adds a reviewable agent-plan and AIQ gate:

```bash
python3 nvidia/integration/nemo_clawd_agent.py
python3 nvidia/blueprints/aiq/agent.py --strict
python3 nvidia/scripts/verify_nvidia.py --strict
```

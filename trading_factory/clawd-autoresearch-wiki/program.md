# ClawdBot Research Program v1.0

## Mission
Autonomously discover the optimal Solana trading strategy through systematic experimentation.
One experiment = one parameter mutation + backtest + evaluation.
You accept what improves the metric. You discard what doesn't. You learn. You repeat.

## Primary Metric
**Sharpe × WinRate** — higher is better.
Secondary: MaxDrawdown < 15%, minimum 10 trades per backtest.

## Strategy Space to Explore

### Phase 1 — Momentum Fundamentals
- RSI period optimization (9, 14, 21)
- RSI threshold calibration (25/75, 30/70, 35/65)
- EMA crossover periods (5/20, 10/30, 20/50)
- Volume filter sensitivity

### Phase 2 — Perps + Funding Arbitrage
- Funding rate entry thresholds
- Mark/index divergence signals
- Long/short bias based on OI
- Combined spot + perp signals

### Phase 3 — Risk Management
- Stop loss optimization (5%, 8%, 12%)
- Take profit laddering
- Position sizing (fixed vs Kelly)
- Max concurrent positions

### Phase 4 — Advanced Signals
- VWAP deviation entries
- Holder concentration filters
- Top trader wallet tracking
- Whale wallet correlation

## Constraints (Never Violate)
- Min liquidity: $50,000 USD
- Min volume: $100,000 USD/24h
- Max position: 10-25% of portfolio
- Backtest minimum: 10 trades
- Stop loss: always present

## Agent Instructions
1. Read this file each session
2. Review lessons/ vault for accumulated knowledge  
3. Generate hypothesis based on unexplored parameter space
4. Run backtest simulation using live Birdeye OHLCV data
5. Log results to research/ vault regardless of outcome
6. Update lessons/ if insight is discovered
7. Only update strategy.md if metric improves

## Notes
- Prefer small mutations over large jumps
- If stuck, explore a new phase
- Document why rejected hypotheses failed
- Compound successful improvements

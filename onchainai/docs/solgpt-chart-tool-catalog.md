<!-- Generated catalog for SOL GPT trading tools — keep in sync with shipped TOOL_DEFS (72 tools) -->
<div align="center">

<svg width="900" height="200" viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SOL GPT tool catalog — 72 tools">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0b1220"/>
      <stop offset="50%" stop-color="#132238"/>
      <stop offset="100%" stop-color="#0d2818"/>
    </linearGradient>
    <linearGradient id="neon" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#14F195"/>
      <stop offset="50%" stop-color="#9945FF"/>
      <stop offset="100%" stop-color="#14F195"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="900" height="200" rx="18" fill="url(#bg)"/>
  <rect x="2" y="2" width="896" height="196" rx="16" fill="none" stroke="url(#neon)" stroke-width="2" opacity="0.85"/>
  <text x="450" y="58" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18" fill="#94a3b8" letter-spacing="4">
    SOLANA · NON-CUSTODIAL · RESEARCH + USER-SIGNED
  </text>
  <text x="450" y="108" text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif" font-size="42" font-weight="700" fill="#f8fafc" filter="url(#glow)">
    SOL GPT tool catalog
  </text>
  <text x="450" y="148" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="28" font-weight="700" fill="#14F195" filter="url(#glow)">
    <tspan>72</tspan>
    <tspan fill="#e2e8f0" font-size="22"> tools shipped</tspan>
  </text>
  <text x="450" y="178" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="14" fill="#a5b4fc">
    37 core · 16 Phoenix Eternal · Birdeye · Helius · DFlow · Browser Use
  </text>
</svg>

<br/>

[![tools](https://img.shields.io/badge/tools-72-14F195?style=for-the-badge&labelColor=0b1220)](#totals)
[![core](https://img.shields.io/badge/core-37-9945FF?style=for-the-badge&labelColor=0b1220)](#core-tools-always-loaded-for-kimi)
[![phoenix](https://img.shields.io/badge/phoenix-16-emerald?style=for-the-badge&labelColor=0b1220)](#phoenix-eternal-16)
[![custody](https://img.shields.io/badge/live_orders-none-red?style=for-the-badge&labelColor=0b1220)](#execution-model)

</div>

# SOL GPT tool catalog

> Full tool surface for **Kimi (Moonshot)** and **OpenRouter** (including free `poolside/laguna-s-2.1:free`, `z-ai/glm-5.2:free`, and `openrouter/free`).
> Both providers receive the same non-custodial catalog via `availableSolGptTools(TOOL_DEFS)`.
> Moonshot also loads **core** tools on every turn (no `search_tools` first).
> OpenRouter requests attribute usage to [solanaclawd.com](https://openrouter.ai/apps?url=https://solanaclawd.com) (`HTTP-Referer: https://solanaclawd.com`).

This catalog is the product/UI contract for Cheshire **SOL GPT** trading research.
Clawd Bot Go packages that cover related ground: `pkg/solana`, `pkg/phoenix`, `pkg/vulcan`, `pkg/trading`, `pkg/wallet`, `pkg/tools`.

Generated for the open-source runtime docs. **Do not put API keys in this file.**

## Totals

| Metric | Count |
|--------|------:|
| **Shipped tools (full catalog)** | **72** |
| Core (always on for Kimi first turn) | 37 |
| Specialty (via `search_tools` / full OpenRouter catalog) | 35 |
| Groups | 10 |

### Count by group

| Group | Id | Tools | Blurb |
|-------|----|------:|-------|
| **Phoenix Eternal** | `phoenix` | 16 | Perps research — markets, mark, book, funding, candles, risk estimates |
| **Market data** | `market` | 18 | Prices, search, trending, memes, fees, security |
| **OHLCV & live tape** | `ohlcv` | 10 | Candles, live price, base/quote streams, trades |
| **Wallet & portfolio** | `wallet` | 4 | Net worth, PnL, assets, balances |
| **Helius Wallet API** | `helius` | 8 | Identity, history, transfers, funding, activity stream |
| **Swaps & sends** | `trading` | 5 | Quotes + user-signed swap/transfer prep (your wallet signs) |
| **Prediction markets** | `prediction` | 3 | DFlow / Kalshi read-only market data |
| **Cloud browser** | `browser` | 4 | Browse external sites via Browser Use |
| **Agents & DAS** | `agents` | 2 | Metaplex / Solana agent discovery and assets |
| **Platform** | `platform` | 2 | Sponge status and catalog search |

**Grand total: `72` tools** (no live place/deposit/withdraw order tools).

## Providers

| Provider | Model path | Tools |
|----------|------------|-------|
| **Moonshot / Kimi** | `MOONSHOT_API_KEY` → kimi-k3 | Core tools every turn + `search_tools` for specialty |
| **OpenRouter** | `OPENROUTER_API_KEY` + `OPENROUTER_DEFAULT_MODEL=poolside/laguna-s-2.1:free` (also `openrouter/free`, `z-ai/glm-5.2:free`) | Full catalog each turn; `HTTP-Referer: https://solanaclawd.com` |
| **xAI Grok** | `XAI_API_KEY` (holder opt-in) | Same catalog |

## Execution model

- **Research tools** return JSON to the model (streamed as tool results in the chat UI).
- **Live spends** never use a server hot wallet: `prepare_user_swap` / `prepare_user_transfer` return unsigned txs; the browser wallet signs; server relays via Jupiter execute or Helius/RPC `sendTransaction`.
- **Helius Wallet API** tools require holder / paid / entitled access + server `HELIUS_API_KEY`.
- **Phoenix** market data uses `https://perp-api.phoenix.trade`; **get_phoenix_rpc_context** uses project Helius/Solana RPC for cluster health + fee SOL.
- **Not in catalog:** `execute_swap`, `sponge_bridge`, and any Phoenix live place/deposit/withdraw order tools.

## Core tools (always loaded for Kimi)

```
analyze_phoenix_account_health
batch_wallet_identity
browse_web
calculate_phoenix_position_margin
get_net_worth
get_phoenix_candles
get_phoenix_exchange_snapshot
get_phoenix_exchange_status
get_phoenix_funding_overview
get_phoenix_funding_rates
get_phoenix_mark_price
get_phoenix_market
get_phoenix_market_calendar
get_phoenix_market_fills
get_phoenix_market_stats
get_phoenix_orderbook
get_phoenix_rpc_context
get_phoenix_trader
get_pnl
get_price
get_quote
get_token_overview
get_trending
get_wallet_assets
get_wallet_balance_at
get_wallet_balances_helius
get_wallet_funded_by
get_wallet_history
get_wallet_identity
get_wallet_transfers
list_phoenix_markets
prepare_user_swap
prepare_user_transfer
resolve_token
search_tokens
search_tools
stream_wallet_activity
```

**37** core names (includes full Helius Wallet + Phoenix perps research sets).

## Catalog by group

### Phoenix Eternal (16)

> Perps research — markets, mark, book, funding, candles, risk estimates

| Tool | Core | What it does |
|------|:----:|--------------|
| `analyze_phoenix_account_health` | yes | Estimate Phoenix effective collateral, account-health gap, risk score/tier, and liquidation thresholds. Call get_phoenix_market first for risk-factor percentages. Read-only estimate. |
| `calculate_phoenix_position_margin` | yes | Estimate Phoenix position margin + resting limit-order margin. Call get_phoenix_market + get_phoenix_mark_price first. Read-only. |
| `get_phoenix_candles` | yes | OHLCV candles for a Phoenix perpetual (TradingView-style). Requires timeframe (1m, 5m, 1h, 1d, …). Read-only. |
| `get_phoenix_exchange_snapshot` | yes | Phoenix exchange snapshot (aggregate market state). Read-only. |
| `get_phoenix_exchange_status` | yes | Phoenix Eternal operational status: active, gated, withdrawals. Read-only. |
| `get_phoenix_funding_overview` | yes | Exchange-wide Phoenix funding overview across markets. Read-only. |
| `get_phoenix_funding_rates` | yes | Recent Phoenix Eternal funding-rate history for a market. Read-only. |
| `get_phoenix_mark_price` | yes | Current Phoenix Eternal mark price (USD) used for PnL and liquidation. Read-only. |
| `get_phoenix_market` | yes | Get static configuration for one Phoenix Eternal perpetual market (leverage tiers, fees, risk). Symbols like SOL-PERP. Read-only. |
| `get_phoenix_market_calendar` | yes | Market calendar / session schedule for a Phoenix market (useful for RWA/commodity perps). Read-only. |
| `get_phoenix_market_fills` | yes | Recent unaggregated fills for a Phoenix market (tape). Read-only. |
| `get_phoenix_market_stats` | yes | Historical / rolling market stats for a Phoenix perpetual. Read-only. |
| `get_phoenix_orderbook` | yes | Current Phoenix Eternal L2 orderbook snapshot for a perpetual market. Read-only. |
| `get_phoenix_rpc_context` | yes | Solana RPC / Helius cluster health for perps readiness: slot, block height, optional connected-wallet SOL for fees. Uses server HELIUS_API_KEY / SOLANA_RPC_URL (never client keys). |
| `get_phoenix_trader` | yes | Read Phoenix trader view by trader pubkey (positions/margins when available). Read-only research — not a live order. |
| `list_phoenix_markets` | yes | List Phoenix Eternal perpetual markets with status, fees, leverage tiers, and risk parameters. Read-only. Prefer for 'what perps can I trade'. |

### Market data (18)

> Prices, search, trending, memes, fees, security

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_birdeye_security` |  | Birdeye token_security endpoint — freeze authority, mint authority, top10, etc. |
| `get_creation_info` |  | Token creation info (creator, slot, time). |
| `get_holder_distribution` |  | Holder distribution buckets for a token mint. |
| `get_meme_list` |  | Meme token list from Birdeye (pump.fun, moonshot, raydium_launchlab, meteora DBC, …). Supports graduation filter and min fees/liquidity. |
| `get_meme_listings` |  | Newly listed meme-platform tokens (e.g. pump.fun). Use for 'new memes' discovery. |
| `get_multi_price` |  | Spot USD prices for up to 50 Solana mints from authenticated Jupiter Price V3 (Birdeye fallback). |
| `get_price` | yes | Spot USD price from authenticated Jupiter Price V3 (Birdeye fallback), including 24h change and liquidity. |
| `get_smart_money` |  | Smart-money token list from Birdeye (tokens smart wallets are trading). |
| `get_token_fees` |  | Global fees paid for multiple Solana tokens (Business/Enterprise Birdeye). Max 3 intervals per call. |
| `get_token_markets` |  | List markets/pairs for a token (liquidity, volume, source DEX). |
| `get_token_overview` | yes | Rich token overview: price, market cap, FDV, volume, holders, social links. |
| `get_token_security` |  | Token security / rug-risk check: holder concentration, sniper/bundler/insider wallets, rug score, Jupiter verification. |
| `get_top_traders` |  | Top traders for a token by volume/PnL window. |
| `get_trending` | yes | Trending tokens list sorted by rank, volume, or liquidity. |
| `list_tokens` |  | DEX token list with filters. Use min_global_fees_paid (SOL) to filter quality/fee-paying tokens. |
| `resolve_token` | yes | Convert a ticker symbol ('BONK', 'SOL', 'JUP') to a mint address. Uses well-known list + Birdeye search. |
| `search_market_data` |  | Full Birdeye search for tokens and/or markets (pairs). |
| `search_tokens` | yes | Search tokens by symbol or name (e.g. 'BONK', 'JUP'). Returns mint addresses and metadata. |

### OHLCV & live tape (10)

> Candles, live price, base/quote streams, trades

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_base_quote_chart` |  | Historical base/quote OHLCV REST when you only know both token mints (no pair address). |
| `get_base_quote_live_price` |  | Live SUBSCRIBE_BASE_QUOTE_PRICE OHLCV from two mints (no pair address). mode raw|scaled|both. |
| `get_chart` |  | OHLCV candlestick data for a token mint. The SOL GPT UI renders an interactive price chart from this tool result. |
| `get_history_price` |  | Historical price line (unixTime, value) for charting over days/weeks. |
| `get_live_price` |  | Live SUBSCRIBE_PRICE OHLCV ticks. Token: currency=usd. Pair/market: currency=pair. |
| `get_live_txs` |  | Sample live SUBSCRIBE_TXS via Birdeye WebSocket for a few seconds. |
| `get_net_worth_chart` |  | Historical net worth chart for a wallet. |
| `get_pair_chart` |  | Historical OHLCV for a pair/market address. |
| `get_pair_trades` |  | Recent trades for a specific pair/market address. |
| `get_token_trades` |  | Recent token transactions (swaps, add/remove liquidity) for a mint. |

### Wallet & portfolio (4)

> Net worth, PnL, assets, balances

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_net_worth` | yes | Check wallet portfolio / net worth via Helius DAS getAssetsByOwner (showFungible + showNativeBalance). |
| `get_pnl` | yes | Wallet PnL summary — realized profit/loss, win rate, total invested. |
| `get_sol_balance` |  | On-chain SOL balance (lamports) for a wallet via server Solana RPC / Helius. |
| `get_wallet_assets` | yes | Wallet assets / holdings overview. |

### Helius Wallet API (8)

> Identity, history, transfers, funding, activity stream

| Tool | Core | What it does |
|------|:----:|--------------|
| `batch_wallet_identity` | yes | Helius Wallet API: batch identity lookup for up to 100 addresses/domains (POST /batch-identity). |
| `get_wallet_balance_at` | yes | Helius Wallet API: historical balance of a mint (or native SOL) at a past time/slot. |
| `get_wallet_balances_helius` | yes | Helius Wallet API: token + SOL balances with USD values (sorted by value). |
| `get_wallet_funded_by` | yes | Helius Wallet API: who originally funded this wallet (first SOL transfer). |
| `get_wallet_history` | yes | Helius Wallet API: recent transaction history with balance changes. |
| `get_wallet_identity` | yes | Helius Wallet API: resolve identity for a Solana address or SNS/ANS domain. |
| `get_wallet_transfers` | yes | Helius Wallet API: inbound/outbound token transfers with counterparties. |
| `stream_wallet_activity` | yes | Composite Helius Wallet activity stream for natural-language narration. |

### Swaps & sends (5)

> Quotes + user-signed swap/transfer prep (your wallet signs)

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_dflow_priority_fees` |  | DFlow global priority fee estimates in micro-lamports per compute unit. Requires DFLOW_API_KEY. |
| `get_quote` | yes | Quote a spot swap without executing. Prefers DFlow GET /quote; falls back to Jupiter Swap V2. |
| `list_dflow_tokens` |  | List DFlow-supported spot token mints and decimals. Requires DFLOW_API_KEY. |
| `prepare_user_swap` | yes | Prepare user-signed Jupiter swap (UI signs). |
| `prepare_user_transfer` | yes | Prepare user-signed SOL/SPL transfer (UI signs + RPC send). |

### Prediction markets (3)

> DFlow / Kalshi read-only market data

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_prediction_market` |  | Authenticated DFlow/Kalshi market snapshot by ticker. Read-only. |
| `get_prediction_orderbook` |  | Authenticated DFlow/Kalshi orderbook depth for a known market ticker. Read-only. |
| `search_prediction_markets` |  | Search authenticated DFlow/Kalshi prediction-market events by topic. Read-only. |

### Cloud browser (4)

> Browse external sites via Browser Use

| Tool | Core | What it does |
|------|:----:|--------------|
| `browse_web` | yes | Open a real cloud browser (Browser Use) and complete a research or navigation task on external sites. |
| `browser_followup` |  | Dispatch a follow-up task on an existing keepAlive Browser Use session. |
| `browser_session_status` |  | Poll Browser Use session status, output, and live URL by session id. |
| `browser_session_stop` |  | Stop a Browser Use session (task or full session). |

### Agents & DAS (2)

> Metaplex / Solana agent discovery and assets

| Tool | Core | What it does |
|------|:----:|--------------|
| `get_asset` |  | Helius DAS getAsset: NFT/cNFT/token metadata, ownership, optional cached price. |
| `search_solana_agents` |  | Discover Metaplex MPL Core agents via Helius DAS (isAgent). |

### Platform (2)

> Sponge status and catalog search

| Tool | Core | What it does |
|------|:----:|--------------|
| `search_tools` | yes | Search the SOL GPT tool catalog by keyword (loads extra tools mid-chat for Kimi specialty path). |
| `sponge_status` |  | PaySponge / Sponge agent wallet status. Call before proposing a bridge. |

## Phoenix perps + Helius RPC

| Concern | Tool / service |
|---------|----------------|
| Markets, mark, book, funding, candles, fills | Phoenix REST `perp-api.phoenix.trade` |
| Cluster slot / block height / wallet SOL for fees | `get_phoenix_rpc_context` → Helius RPC (`HELIUS_API_KEY` / `SOLANA_RPC_URL`) |
| Margin / liquidation estimates | `analyze_phoenix_account_health`, `calculate_phoenix_position_margin` (local math) |
| Live order placement | **Not** in SOL GPT chat (use Vulcan CLI / dedicated trader UI) — chat is research + user-signed spot |

## Env

```bash
OPENROUTER_API_KEY=…
OPENROUTER_DEFAULT_MODEL=poolside/laguna-s-2.1:free
# Ranking headers are pinned in code: HTTP-Referer=https://solanaclawd.com
MOONSHOT_API_KEY=…          # Kimi default when set
HELIUS_API_KEY=…            # Wallet API + DAS + RPC
DFLOW_API_KEY=…             # spot quotes / priority fees
PHOENIX_API_URL=https://perp-api.phoenix.trade   # optional
# PHOENIX_API_KEY / PHOENIX_BEARER_TOKEN if your deployment requires auth
```

## Source of truth

- **Product catalog:** `src/lib/solgpt/tool-catalog.ts` (`getSolGptShippedToolCatalog`) plus this 72-tool trading subset
- Tool defs assembly: `src/lib/solgpt/tools.ts` (`solGptTools` / `TOOL_DEFS`)
- Helius wallet: `src/lib/solgpt/helius-wallet.ts`
- Phoenix: `src/lib/solgpt/phoenix.ts`
- OpenRouter attribution: `src/lib/solgpt/openrouter.ts` (`HTTP-Referer: https://solanaclawd.com`)
- Go runtime cousins: `pkg/solana`, `pkg/phoenix`, `pkg/vulcan`, `pkg/trading`, `pkg/tools`
- Package map: [`PKG_MAP.md`](PKG_MAP.md)

## Totals (machine-readable)

```json
{"shipped":72,"core":37,"specialty":35,"groups":10,"phoenix":16}
```

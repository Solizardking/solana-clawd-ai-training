# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PUMP-MCP and SOL GPT tool names used as atomic trading tokens.

These strings are added onto the Nemotron tokenizer so tool calls encode as
a single id instead of character/BPE fragments.
"""

from __future__ import annotations

from typing import Iterable

# Exact MCP tool names from PUMP-MCP-main/src/index.ts (server.tool).
PUMP_MCP_TOOLS: tuple[str, ...] = (
    "get-token-info",
    "create-token",
    "buy-token",
    "sell-token",
    "list-accounts",
    "get-account-balance",
    "generate-image",
    "get-fee-tier",
    "list-free-models",
    "free-router-chat",
    "list-skills",
    "get-skill",
    "rerank-docs",
)

# SOL GPT shipped catalog (desk + Phoenix + Helius DAS + OHLCV + Imperial).
# 72 core names used in tests/docs; extras below stay in the same vocab.
SOL_GPT_TOOLS: tuple[str, ...] = (
    "analyze_phoenix_account_health",
    "batch_wallet_identity",
    "browse_web",
    "calculate_phoenix_position_margin",
    "get_asset",
    "get_chart",
    "get_imperial_flash_markets",
    "get_imperial_funding_rates",
    "get_imperial_gmtrade_funding_rates",
    "get_imperial_gmtrade_liquidity",
    "get_imperial_gmtrade_markets",
    "get_imperial_mark_prices",
    "get_imperial_orders",
    "get_imperial_phoenix_depth",
    "get_imperial_phoenix_mark_prices",
    "get_imperial_phoenix_markets",
    "get_imperial_positions",
    "get_imperial_priority_fee",
    "get_imperial_route",
    "get_imperial_stats_markets",
    "get_imperial_stats_open_interest",
    "get_imperial_stats_summary",
    "get_imperial_stats_volume",
    "get_imperial_status",
    "get_net_worth",
    "get_phoenix_candles",
    "get_phoenix_exchange_snapshot",
    "get_phoenix_exchange_status",
    "get_phoenix_funding_overview",
    "get_phoenix_funding_rates",
    "get_phoenix_mark_price",
    "get_phoenix_market",
    "get_phoenix_market_calendar",
    "get_phoenix_market_fills",
    "get_phoenix_market_stats",
    "get_phoenix_my_trader_state",
    "get_phoenix_orderbook",
    "get_phoenix_rpc_context",
    "get_phoenix_trader",
    "get_pnl",
    "get_price",
    "get_quote",
    "get_token_overview",
    "get_trending",
    "get_wallet_assets",
    "get_wallet_balance_at",
    "get_wallet_balances_helius",
    "get_wallet_funded_by",
    "get_wallet_history",
    "get_wallet_identity",
    "get_wallet_transfers",
    "list_phoenix_markets",
    "prepare_phoenix_cancel_all",
    "prepare_phoenix_deposit",
    "prepare_phoenix_limit_order",
    "prepare_phoenix_market_order",
    "prepare_phoenix_order",
    "prepare_phoenix_register_trader",
    "prepare_phoenix_withdraw",
    "prepare_user_payment",
    "prepare_user_swap",
    "prepare_user_transfer",
    "resolve_token",
    "search_solana_agents",
    "search_tokens",
    "search_tools",
    "st_das_get_asset",
    "st_das_get_asset_proof",
    "st_das_get_assets_by_owner",
    "st_get_chart",
    "st_get_graduating_tokens",
    "st_get_price",
)

SOL_GPT_EXTRA_TOOLS: tuple[str, ...] = (
    "get_birdeye_price",
    "get_birdeye_multi_price",
    "get_multi_price",
    "get_meme_token_detail",
    "get_meme_token_list",
    "get_smart_money",
    "get_wallet_status",
    "get_address_overview",
    "get_transaction_overview",
    "scan_pump_token",
    "search_birdeye",
    "solana_token_info",
    "solana_top_traders",
    "solana_trending",
    "solana_price",
    "st_das_get_assets_by_authority",
    "st_das_get_assets_by_creator",
    "st_das_get_assets_by_group",
    "st_das_get_nft_editions",
    "st_das_get_signatures_for_asset",
    "st_das_get_token_accounts",
    "st_das_search_assets",
    "st_get_first_buyers",
    "st_get_graduated_tokens",
    "st_get_latest_tokens",
    "st_get_multi_tokens",
    "st_get_multiple_prices",
    "st_get_price_history",
    "st_get_token",
    "st_get_token_holders",
    "st_get_token_stats",
    "st_get_token_top_holders",
    "st_get_token_top_traders",
    "st_get_token_trades",
    "st_get_trending_tokens",
    "st_get_wallet",
    "st_get_wallet_pnl",
    "stream_wallet_activity",
)

STREAM_TOKENS: tuple[str, ...] = (
    "token-launch",
    "token-enriched",
    "clawd-ws",
    "pump.fun",
)

NEMOTRON_TOKENIZER_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
DEFAULT_HUB_REPO = "ordlibrary/solana-clawd-nemotron-trading-tokenizer"


def all_trading_tool_tokens(extra: Iterable[str] | None = None) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for group in (
        PUMP_MCP_TOOLS,
        SOL_GPT_TOOLS,
        SOL_GPT_EXTRA_TOOLS,
        STREAM_TOKENS,
        tuple(extra or ()),
    ):
        for name in group:
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)

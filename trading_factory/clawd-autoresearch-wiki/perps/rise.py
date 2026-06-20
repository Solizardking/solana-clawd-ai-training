"""
Python bindings for the Phoenix Rise SDK.

Rise is the developer-facing SDK for Phoenix perpetuals. This module provides
Python wrappers for HTTP market data, order placement via the Vulcan CLI,
and type-safe order packet construction.

Usage:
    from perps.rise import RiseClient, RiseConfig

    client = RiseClient()
    snapshot = client.get_snapshot()
    market = client.get_market("SOL")

See: https://docs.phoenix.trade/sdk
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


PHOENIX_API_BASE = "https://perp-api.phoenix.trade"
DEFAULT_RPC = "https://api.mainnet-beta.solana.com"


@dataclass
class RiseConfig:
    """Configuration for Rise SDK HTTP client."""
    api_url: str = PHOENIX_API_BASE
    rpc_url: str = DEFAULT_RPC
    api_key: str = ""


class RiseClient:
    """Python HTTP client for Phoenix Rise SDK market data.

    Provides read-only access to exchange state, market data, and trader info.
    For write operations (orders, margin), use VulcanClient.
    """

    def __init__(self, config: RiseConfig | None = None):
        self.config = config or RiseConfig()

    def _fetch(self, path: str) -> dict[str, Any]:
        """Fetch JSON from the Phoenix API."""
        url = f"{self.config.api_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def get_snapshot(self) -> dict[str, Any]:
        """Exchange-wide state snapshot plus every market's current config."""
        return self._fetch("/v1/exchange/snapshot")

    def get_market(self, symbol: str) -> dict[str, Any]:
        """One market's fees, risk, funding cadence, and configuration."""
        return self._fetch(f"/v1/exchange/markets/{symbol}")

    def get_orderbook(self, symbol: str) -> dict[str, Any]:
        """HTTP L2 orderbook snapshot for one market."""
        return self._fetch(f"/v1/orderbook/{symbol}")

    def get_trader_state(self, authority: str, pda_index: int = 0) -> dict[str, Any]:
        """Trader-centric view of collateral, positions, orders, and triggers."""
        return self._fetch(f"/v1/traders/{authority}?traderPdaIndex={pda_index}")

    def get_market_stats_history(self, symbol: str, interval: str = "1h",
                                  limit: int = 100) -> dict[str, Any]:
        """Time-series market stats for frontends and analytics."""
        return self._fetch(f"/v1/markets/{symbol}/stats?interval={interval}&limit={limit}")

    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Historical funding rate data."""
        return self._fetch(f"/v1/funding/{symbol}?limit={limit}")

    def get_leverage_tiers(self, symbol: str) -> dict[str, Any]:
        """Leverage tier schedule for a market."""
        return self._fetch(f"/v1/exchange/markets/{symbol}/leverage-tiers")
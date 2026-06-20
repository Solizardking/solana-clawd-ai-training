"""
TWAP (Time-Weighted Average Price) strategy runner for Phoenix perpetuals.

Splits a large order into smaller slices executed at fixed time intervals
to minimize market impact.

Usage:
    from strategy.twap import TWAPRunner, TWAPConfig
    from perps.vulcan import VulcanClient

    config = TWAPConfig(
        symbol="SOL", side="long", total_size=1000.0,
        num_slices=10, duration_minutes=30
    )
    runner = TWAPRunner(config, vulcan_client=VulcanClient())
    runner.start()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from strategy.runner import StrategyRunner, StrategyConfig


@dataclass
class TWAPConfig(StrategyConfig):
    """Configuration for TWAP execution."""
    num_slices: int = 10
    duration_minutes: int = 30
    max_slippage_bps: int = 50
    aggressive: bool = False  # If True, slices are market orders


class TWAPRunner(StrategyRunner):
    """TWAP strategy runner.

    Splits total_size into num_slices and executes one slice per interval.
    """

    def __init__(self, config: TWAPConfig, vulcan_client=None, paper_engine=None):
        super().__init__(config, vulcan_client, paper_engine)
        self.twap_config = config
        self._slice_size = config.total_size / config.num_slices
        self._slice_interval = (config.duration_minutes * 60) / config.num_slices
        self._slices_executed = 0

    def execute_tick(self) -> dict[str, Any]:
        """Execute one TWAP slice."""
        if self._slices_executed >= self.twap_config.num_slices:
            return {"status": "completed", "slices": self._slices_executed}

        remaining_slices = self.twap_config.num_slices - self._slices_executed
        slice_size = self.twap_config.total_size / remaining_slices

        result = {"slices_executed": 0, "fills": []}

        if self.paper and self.twap_config.paper_mode:
            if self.twap_config.side == "long":
                r = self.paper.buy(self.twap_config.symbol, notional_usdc=slice_size)
            else:
                r = self.paper.sell(self.twap_config.symbol, notional_usdc=slice_size)
            if r.get("ok"):
                result["slices_executed"] = 1
                result["fills"].append(r.get("data", {}).get("fill"))
            else:
                return {"status": "error", "error": r.get("error")}
        elif self.vulcan:
            notional = slice_size
            if self.twap_config.side == "long":
                r = self.vulcan.trade_market_buy(self.twap_config.symbol, notional_usdc=notional)
            else:
                r = self.vulcan.trade_market_sell(self.twap_config.symbol, notional_usdc=notional)
            if r.ok:
                result["slices_executed"] = 1
                result["fills"].append(r.data)
                self.state.total_filled += notional
            else:
                return {"status": "error", "error": r.error}

        self._slices_executed += 1
        result["slices_total"] = self._slices_executed
        result["remaining"] = self.twap_config.num_slices - self._slices_executed
        return result

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["data"].update({
            "strategy_type": "twap",
            "num_slices": self.twap_config.num_slices,
            "slices_executed": self._slices_executed,
            "slice_size_usdc": self._slice_size,
            "total_target_usdc": self.twap_config.total_size,
            "completion_pct": (self._slices_executed / self.twap_config.num_slices) * 100
            if self.twap_config.num_slices > 0 else 0,
        })
        return base
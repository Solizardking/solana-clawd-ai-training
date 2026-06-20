"""
Technical Analysis (TA) driven strategy runner for Phoenix perpetuals.

Evaluates indicator-based trigger specs and executes trades when conditions
are met. Supports RSI, MACD, Bollinger Bands, ATR, VWAP, ADX, Stoch.

Usage:
    from strategy.ta import TAStrategyRunner, TAConfig

    config = TAConfig(
        symbol="SOL", side="long", total_size=1000.0,
        entry_spec={"indicator":"rsi","timeframe":"1h","op":"lt","threshold":30},
        exit_spec={"indicator":"rsi","timeframe":"1h","op":"gt","threshold":70},
    )
    runner = TAStrategyRunner(config, vulcan_client=VulcanClient())
    runner.start()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from strategy.runner import StrategyRunner, StrategyConfig


@dataclass
class TAConfig(StrategyConfig):
    """Configuration for TA-driven strategy."""
    entry_spec: dict | None = None  # Trigger spec for entry
    exit_spec: dict | None = None   # Trigger spec for exit
    timeframe: str = "1h"
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 4.0
    max_concurrent_positions: int = 1


class TAStrategyRunner(StrategyRunner):
    """Technical Analysis driven strategy runner.

    At each tick:
    1. Compute requested indicators
    2. Evaluate entry trigger
    3. If triggered, enter position
    4. Evaluate exit trigger
    5. If triggered, exit position
    """

    def __init__(self, config: TAConfig, vulcan_client=None, paper_engine=None):
        super().__init__(config, vulcan_client, paper_engine)
        self.ta_config = config
        self._position_open = False
        self._last_atr: float = 0

    def _compute_indicator(self, spec: dict) -> dict[str, Any]:
        """Compute a technical indicator via Vulcan CLI."""
        if self.vulcan:
            r = self.vulcan.ta_compute(
                symbol=self.ta_config.symbol,
                indicator=spec.get("indicator", "rsi"),
                timeframe=spec.get("timeframe", self.ta_config.timeframe),
                params=spec.get("params"),
            )
            if r.ok and r.data:
                return r.data
        return {"value": None, "error": "Cannot compute indicator"}

    def _evaluate_trigger(self, spec: dict) -> bool:
        """Evaluate a trigger spec via Vulcan CLI."""
        if self.vulcan and spec:
            params = {
                "indicator": spec.get("indicator", "rsi"),
                "timeframe": spec.get("timeframe", self.ta_config.timeframe),
                "op": spec.get("op", "lt"),
                "threshold": spec.get("threshold", 30),
            }
            # Use Vulcan ta_compute + compare locally for more control
            result = self._compute_indicator(spec)
            value = result.get("value")
            if value is not None:
                op = spec.get("op", "lt")
                threshold = spec.get("threshold", 30)
                try:
                    val = float(value)
                    thr = float(threshold)
                    if op == "lt":
                        return val < thr
                    elif op == "gt":
                        return val > thr
                    elif op == "lte":
                        return val <= thr
                    elif op == "gte":
                        return val >= thr
                    elif op == "cross_above":
                        # Cross above = was below, now above
                        return val > thr
                    elif op == "cross_below":
                        return val < thr
                except (ValueError, TypeError):
                    pass
        return False

    def execute_tick(self) -> dict[str, Any]:
        """Execute one TA tick — check triggers and act."""
        result = {
            "indicators": {},
            "entry_triggered": False,
            "exit_triggered": False,
            "position_open": self._position_open,
            "action": "none",
        }

        # Get market snapshot
        if self.vulcan:
            r = self.vulcan.ta_report(self.ta_config.symbol, self.ta_config.timeframe)
            if r.ok and r.data:
                result["indicators"] = r.data
                if isinstance(r.data, dict):
                    self._last_atr = float(r.data.get("atr", 0))
            elif self.vulcan:
                # Fallback: compute individual indicators
                indicators = ["rsi", "macd", "bbands", "atr"]
                for ind in indicators:
                    r2 = self.vulcan.ta_compute(self.ta_config.symbol, ind)
                    if r2.ok and r2.data:
                        result["indicators"][ind] = r2.data

        # Check entry trigger
        if not self._position_open and self.ta_config.entry_spec:
            triggered = self._evaluate_trigger(self.ta_config.entry_spec)
            result["entry_triggered"] = triggered
            if triggered:
                if self.paper and self.ta_config.paper_mode:
                    if self.ta_config.side == "long":
                        r3 = self.paper.buy(self.ta_config.symbol,
                                           notional_usdc=self.ta_config.total_size)
                    else:
                        r3 = self.paper.sell(self.ta_config.symbol,
                                            notional_usdc=self.ta_config.total_size)
                    if r3.get("ok"):
                        self._position_open = True
                        self.state.total_filled = self.ta_config.total_size
                        result["action"] = "entry_filled"
                        result["fill"] = r3.get("data", {}).get("fill")
                elif self.vulcan:
                    if self.ta_config.side == "long":
                        r3 = self.vulcan.trade_market_buy(self.ta_config.symbol,
                                                         notional_usdc=self.ta_config.total_size)
                    else:
                        r3 = self.vulcan.trade_market_sell(self.ta_config.symbol,
                                                          notional_usdc=self.ta_config.total_size)
                    if r3.ok:
                        self._position_open = True
                        self.state.total_filled = self.ta_config.total_size
                        result["action"] = "entry_filled"

        # Check exit trigger
        if self._position_open and self.ta_config.exit_spec:
            triggered = self._evaluate_trigger(self.ta_config.exit_spec)
            result["exit_triggered"] = triggered
            if triggered:
                if self.paper and self.ta_config.paper_mode:
                    if self.ta_config.side == "long":
                        r4 = self.paper.sell(self.ta_config.symbol,
                                           notional_usdc=self.ta_config.total_size)
                    else:
                        r4 = self.paper.buy(self.ta_config.symbol,
                                           notional_usdc=self.ta_config.total_size)
                    if r4.get("ok"):
                        self._position_open = False
                        result["action"] = "exit_filled"
                elif self.vulcan:
                    if self.ta_config.side == "long":
                        r4 = self.vulcan.trade_market_sell(self.ta_config.symbol,
                                                          notional_usdc=self.ta_config.total_size)
                    else:
                        r4 = self.vulcan.trade_market_buy(self.ta_config.symbol,
                                                         notional_usdc=self.ta_config.total_size)
                    if r4.ok:
                        self._position_open = False
                        result["action"] = "exit_filled"

        return result
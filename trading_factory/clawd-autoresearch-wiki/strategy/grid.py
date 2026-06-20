"""
Grid trading strategy runner for Phoenix perpetuals.

Places buy and sell limit orders at fixed price intervals within a range.
Profits from mean reversion by buying low and selling high.

Usage:
    from strategy.grid import GridRunner, GridConfig

    config = GridConfig(symbol="SOL", lower_price=140.0, upper_price=160.0,
                        num_orders=10, total_size=1000.0)
    runner = GridRunner(config, vulcan_client=VulcanClient())
    runner.start()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from strategy.runner import StrategyRunner, StrategyConfig


@dataclass
class GridConfig(StrategyConfig):
    """Configuration for grid trading."""
    lower_price: float = 140.0
    upper_price: float = 160.0
    num_orders: int = 10
    rebalance_threshold: float = 0.01  # Rebalance when price moves this much


class GridRunner(StrategyRunner):
    """Grid trading strategy runner.

    Places N limit orders evenly spaced between lower_price and upper_price.
    On the buy side: places buy limit orders at each grid level.
    On the sell side: places sell limit orders at each grid level.
    """

    def __init__(self, config: GridConfig, vulcan_client=None, paper_engine=None):
        super().__init__(config, vulcan_client, paper_engine)
        self.grid_config = config
        self._grid_levels: list[float] = []
        self._placed_orders: dict[str, list[str]] = {}  # level -> order ids
        self._generate_grid()

    def _generate_grid(self):
        """Generate evenly spaced grid levels between lower and upper price."""
        step = (self.grid_config.upper_price - self.grid_config.lower_price) / max(self.grid_config.num_orders - 1, 1)
        self._grid_levels = [
            round(self.grid_config.lower_price + i * step, 2)
            for i in range(self.grid_config.num_orders)
        ]

    def execute_tick(self) -> dict[str, Any]:
        """Execute one grid tick — place/rebalance orders at grid levels."""
        result = {"orders_placed": 0, "orders_filled": 0, "grid_levels": self._grid_levels}

        if self.vulcan:
            r = self.vulcan.market_ticker(self.grid_config.symbol)
            if r.ok and r.data:
                mark = float(r.data.get("mark_price", 0))
            else:
                return {"status": "error", "error": "Cannot fetch mark price"}
        else:
            mark = 150.0

        # Cancel old orders
        if self._placed_orders:
            for level, order_ids in self._placed_orders.items():
                for oid in order_ids:
                    if self.vulcan:
                        self.vulcan.trade_cancel(self.grid_config.symbol, oid)
                    elif self.paper:
                        self.paper.cancel(oid)
            self._placed_orders = {}

        # Place new orders at levels that make sense given current price
        level_size = self.grid_config.total_size / self.grid_config.num_orders
        buy_levels = [l for l in self._grid_levels if l < mark]
        sell_levels = [l for l in self._grid_levels if l > mark]

        order_ids = []
        for level in buy_levels:
            if self.paper and self.grid_config.paper_mode:
                r = self.paper.buy(self.grid_config.symbol,
                                   notional_usdc=level_size,
                                   order_type="limit",
                                   price=level)
                if r.get("ok"):
                    oid = r["data"]["order"]["id"]
                    order_ids.append(oid)
                    result["orders_placed"] += 1
            elif self.vulcan:
                lots = str(round(level_size / level, 4))
                r = self.vulcan.trade_limit_buy(self.grid_config.symbol, lots, level)
                if r.ok and r.data:
                    order_ids.append(str(r.data))
                    result["orders_placed"] += 1

        for level in sell_levels:
            if self.paper and self.grid_config.paper_mode:
                r = self.paper.sell(self.grid_config.symbol,
                                    notional_usdc=level_size,
                                    order_type="limit",
                                    price=level)
                if r.get("ok"):
                    oid = r["data"]["order"]["id"]
                    order_ids.append(oid)
                    result["orders_placed"] += 1
            elif self.vulcan:
                lots = str(round(level_size / level, 4))
                r = self.vulcan.trade_limit_sell(self.grid_config.symbol, lots, level)
                if r.ok and r.data:
                    order_ids.append(str(r.data))
                    result["orders_placed"] += 1

        self._placed_orders[self.grid_config.symbol] = order_ids
        return result
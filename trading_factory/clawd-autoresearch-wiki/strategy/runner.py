"""
Base strategy runner for Clawd perps trading strategies.

Provides the common lifecycle: tick loop, pause/resume/finalize,
ledger persistence, and status reporting.

All strategy runners extend this base class.
"""
from __future__ import annotations

import json
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class StrategyStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINALIZED = "finalized"
    ERROR = "error"


@dataclass
class StrategyConfig:
    """Base configuration for any strategy runner."""
    symbol: str = "SOL"
    side: str = "long"  # long or short
    total_size: float = 1000.0  # USDC notional
    max_runtime_minutes: int = 60
    tick_interval_seconds: int = 5
    paper_mode: bool = True
    leverage: float = 1.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0


@dataclass
class StrategyState:
    """Persistence state for a strategy run."""
    run_id: str = ""
    status: str = "pending"
    config: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    paused_at: float = 0.0
    finalized_at: float = 0.0
    ticks: int = 0
    total_filled: float = 0.0
    avg_fill_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "config": self.config,
            "started_at": self.started_at,
            "paused_at": self.paused_at,
            "finalized_at": self.finalized_at,
            "ticks": self.ticks,
            "total_filled": self.total_filled,
            "avg_fill_price": self.avg_fill_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "error": self.error,
        }


class StrategyRunner(ABC):
    """Abstract base class for strategy runners.

    Lifecycle:
        start() -> tick loop (runs until completed/paused/stopped)
        pause() -> pause at next safe point
        stop() -> stop permanently
        finalize() -> stop + clean up orders/positions
        status() -> current state
        report() -> final metrics
    """

    def __init__(self, config: StrategyConfig, vulcan_client=None, paper_engine=None):
        self.config = config
        self.vulcan = vulcan_client
        self.paper = paper_engine
        self.state = StrategyState(
            config=asdict(config),
            started_at=time.time(),
        )
        self._should_stop = False
        self._should_pause = False
        self._ledger: list[dict[str, Any]] = []
        self._tick_count = 0

    @abstractmethod
    def execute_tick(self) -> dict[str, Any]:
        """Execute one tick of the strategy. Returns tick result."""
        ...

    def start(self) -> dict[str, Any]:
        """Start the strategy tick loop."""
        self.state.status = StrategyStatus.RUNNING.value
        self.state.started_at = time.time()
        self.state.run_id = f"{self.config.symbol}_{int(time.time())}"

        deadline = time.time() + self.config.max_runtime_minutes * 60
        last_tick = 0

        while time.time() < deadline:
            if self._should_stop:
                self.state.status = StrategyStatus.FINALIZED.value
                self.state.finalized_at = time.time()
                break
            if self._should_pause:
                self.state.status = StrategyStatus.PAUSED.value
                self.state.paused_at = time.time()
                break

            now = time.time()
            if now - last_tick >= self.config.tick_interval_seconds:
                try:
                    result = self.execute_tick()
                    self._tick_count += 1
                    self.state.ticks = self._tick_count
                    ledger_entry = {
                        "tick": self._tick_count,
                        "timestamp": now,
                        "result": result,
                    }
                    self._ledger.append(ledger_entry)
                except Exception as e:
                    self.state.status = StrategyStatus.ERROR.value
                    self.state.error = str(e)
                    break
                last_tick = now
            else:
                time.sleep(0.1)  # Prevent busy-waiting

        return self.status()

    def pause(self, reason: str = "user_request") -> dict[str, Any]:
        """Request the strategy to pause at the next safe point."""
        self._should_pause = True
        return {"ok": True, "data": {"status": "pause_requested", "reason": reason}}

    def stop(self, reason: str = "user_request") -> dict[str, Any]:
        """Request the strategy to stop permanently."""
        self._should_stop = True
        return {"ok": True, "data": {"status": "stop_requested", "reason": reason}}

    def finalize(self, cancel_orders: bool = True,
                 close_position: bool = True) -> dict[str, Any]:
        """Stop the strategy and clean up."""
        self._should_stop = True
        self.state.status = StrategyStatus.FINALIZED.value
        self.state.finalized_at = time.time()

        cleanups = []
        if cancel_orders and self.vulcan:
            r = self.vulcan.trade_cancel_all(self.config.symbol)
            cleanups.append({"action": "cancel_all", "result": r.ok})
        if close_position and self.vulcan:
            r = self.vulcan.position_close(self.config.symbol)
            cleanups.append({"action": "close_position", "result": r.ok})

        return {
            "ok": True,
            "data": {
                "status": "finalized",
                "cleanups": cleanups,
                "state": self.state.to_dict(),
            },
        }

    def status(self) -> dict[str, Any]:
        """Get current strategy status."""
        return {"ok": True, "data": self.state.to_dict()}

    def report(self) -> dict[str, Any]:
        """Get final or latest strategy report."""
        return {
            "ok": True,
            "data": {
                "run_id": self.state.run_id,
                "status": self.state.status,
                "symbol": self.config.symbol,
                "duration_seconds": (self.state.finalized_at or time.time()) - self.state.started_at
                if self.state.started_at else 0,
                "ticks": self._tick_count,
                "total_filled": self.state.total_filled,
                "avg_fill_price": self.state.avg_fill_price,
                "realized_pnl": self.state.realized_pnl,
                "ledger_size": len(self._ledger),
                "config": asdict(self.config),
            },
        }

    def save_ledger(self, path: str = "") -> None:
        """Save strategy ledger to disk."""
        if not path:
            path = f"strategy_ledger_{self.state.run_id}.json"
        with open(path, "w") as f:
            json.dump({
                "state": self.state.to_dict(),
                "ledger": self._ledger,
                "config": asdict(self.config),
            }, f, indent=2)
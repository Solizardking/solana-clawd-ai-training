"""
NVIDIA Blueprint 4 signal bridge for the clawd-autoresearch-wiki perps package.

Connects ai-training/nvidia/blueprints/signal-discovery/signals.py to the
wiki's VulcanClient + NemotronTrader stack.

Usage:
    from perps.nvidia_signals import NvidiaSignalBridge

    bridge = NvidiaSignalBridge()
    composite = bridge.scan("SOL")
    print(composite.direction, composite.strength)
    bridge.run_and_trade("SOL", budget_usdc=100)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from perps.vulcan import VulcanClient
from perps.paper import PaperEngine


# Path to ai-training/nvidia/blueprints/signal-discovery
_SIGNAL_PATH = (
    Path(__file__).parents[4]
    / "ai-training"
    / "nvidia"
    / "blueprints"
    / "signal-discovery"
)


@dataclass
class CompositeSignal:
    market: str
    direction: str      # long | short | neutral
    strength: float     # 0–1
    signals: list[dict]
    recommended_command: str


def _load_signal_scanner():
    if str(_SIGNAL_PATH) not in sys.path:
        sys.path.insert(0, str(_SIGNAL_PATH))
    from signals import scan_all, score_signals
    return scan_all, score_signals


class NvidiaSignalBridge:
    """
    Runs NVIDIA Blueprint 4 signal detectors and optionally executes
    paper trades via VulcanClient.
    """

    def __init__(
        self,
        vulcan: VulcanClient | None = None,
        paper: PaperEngine | None = None,
        threshold: float = 0.35,
    ):
        self._vc = vulcan or VulcanClient()
        self._paper = paper or PaperEngine()
        self.threshold = threshold

    def scan(self, market: str) -> CompositeSignal:
        """Run all signal detectors, return composite result."""
        try:
            scan_all, score_signals = _load_signal_scanner()
            results = scan_all(market)
            direction, strength = score_signals(results)
            signals = [
                {"name": s.name, "direction": s.direction,
                 "strength": round(s.strength, 4), "reason": s.reason}
                for s in results
            ]
        except Exception as e:
            return CompositeSignal(market, "neutral", 0.0, [],
                                   f"# signal scan error: {e}")

        if direction != "neutral" and strength >= self.threshold:
            side = "buy" if direction == "long" else "sell"
            cmd = f"vulcan paper {side} {market} --notional-usdc 100 --type market"
        else:
            cmd = f"# hold — strength {strength:.2f} below threshold {self.threshold}"

        return CompositeSignal(market, direction, strength, signals, cmd)

    def run_and_trade(self, market: str, budget_usdc: float = 100.0) -> dict:
        """Scan signals and execute paper trade if above threshold."""
        composite = self.scan(market)
        print(f"[nvidia-signals] {market}: {composite.direction}  strength={composite.strength:.3f}")
        for s in composite.signals:
            print(f"  [{s['direction']:7s}] {s['name']:15s} {s['strength']:.2f}  {s['reason']}")

        result = None
        if composite.strength >= self.threshold and composite.direction != "neutral":
            side = "buy" if composite.direction == "long" else "sell"
            fn = self._vc.paper_buy if side == "buy" else self._vc.paper_sell
            r = fn(market, budget_usdc)
            result = {"ok": r.ok, "data": r.data, "error": r.error}
            print(f"  paper {side} ${budget_usdc:.0f} — {'OK' if r.ok else r.error}")
        else:
            print(f"  hold — no trade")

        return {
            "market": market,
            "direction": composite.direction,
            "strength": composite.strength,
            "signals": composite.signals,
            "paper_result": result,
        }

    def scan_many(self, markets: list[str]) -> list[CompositeSignal]:
        return [self.scan(m) for m in markets]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="SOL")
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    bridge = NvidiaSignalBridge(threshold=args.threshold)
    result = bridge.run_and_trade(args.market, args.budget)
    print(json.dumps({k: v for k, v in result.items() if k != "signals"}, indent=2))

"""
Blueprint 2 — End-to-end Solana portfolio optimizer with Clawd trust gates.

Flow:
  1. Fetch live prices from Phoenix / Jupiter via RPC
  2. Generate Monte Carlo scenarios (cuML KDE)
  3. Run Mean-CVaR optimization (cuFOLIO / CVXPY)
  4. Enforce Clawd trust gates before any live execution
  5. Emit Vulcan paper or live trade commands
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from scenarios import generate_scenarios, historical_returns
from mean_cvar import optimize, OptResult
from phoenix_prices import fetch_price_bundle, PHOENIX_MARKETS


TRUST_GATES = {
    "observer":    "Read-only; emit allocation plan, no orders.",
    "paper":       "Execute in Vulcan paper mode. No real funds.",
    "delegated":   "Execute live with human confirmation per order.",
    "auto":        "Execute live automatically. High risk. Requires explicit --gate auto.",
}


@dataclass
class AllocationPlan:
    assets: list[str]
    weights: list[float]
    notional_usdc: float
    gross_notional_usdc: float
    cash_usdc: float
    allocations_usdc: dict[str, float]
    expected_return: float
    cvar: float
    sharpe: float
    solver: str
    scenario_backend: str
    scenario_count: int
    price_sources: dict[str, dict]
    constraints: dict[str, Any]
    gate: str
    warnings: list[str]


@dataclass
class PortfolioRun:
    schema_version: str
    generated_at: str
    blueprint: str
    trust_gate: dict[str, str]
    plan: AllocationPlan
    vulcan_commands: list[str]
    execution: dict[str, Any]


def _fetch_mock_prices(assets: list[str], n_days: int = 365) -> dict[str, list[float]]:
    """Stub: returns synthetic price history. Replace with real RPC/API calls."""
    rng = np.random.default_rng(seed=sum(ord(c) for c in "".join(assets)))
    base = {"SOL": 150, "BTC": 65000, "ETH": 3500, "BONK": 0.00003, "JTO": 3.5, "JUP": 0.8}
    prices = {}
    for a in assets:
        start = base.get(a, 10.0)
        drift = 0.001
        vol = 0.04
        daily = 1 + rng.normal(drift, vol, n_days)
        daily[0] = 1.0
        prices[a] = (start * np.cumprod(daily)).tolist()
    return prices


def build_plan(
    assets: list[str],
    budget: float,
    cvar_alpha: float,
    max_cvar: float,
    max_leverage: float,
    gate: str,
    use_phoenix: bool = True,
    n_days: int = 90,
    n_scenarios: int = 5000,
    max_cardinality: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> AllocationPlan:
    if gate not in TRUST_GATES:
        print(f"ERROR: unknown gate '{gate}'. Valid: {list(TRUST_GATES)}", file=sys.stderr)
        sys.exit(1)

    if use_phoenix:
        if verbose:
            print(f"[portfolio-opt] fetching live prices ({n_days}d)...", file=sys.stderr)
        price_bundle = fetch_price_bundle(assets, n_days=n_days, verbose=verbose)
        prices = price_bundle.prices
        price_sources = price_bundle.sources
    else:
        prices = _fetch_mock_prices(assets)
        price_sources = {
            asset: {
                "asset": asset,
                "source": "synthetic_mock",
                "days": len(series),
                "last_price": float(series[-1]),
                "synthetic": True,
                "notes": ["--no-phoenix synthetic run"],
            }
            for asset, series in prices.items()
        }
    rets, names = historical_returns(prices)
    sc = generate_scenarios(rets, names, n_scenarios=n_scenarios, seed=seed)
    result: OptResult = optimize(
        sc.scenarios, sc.assets,
        cvar_alpha=cvar_alpha,
        max_cvar=max_cvar,
        max_leverage=max_leverage,
        max_cardinality=max_cardinality,
    )

    weight_sum = float(result.weights.sum())
    gross_notional = float(budget * weight_sum)
    cash_usdc = float(max(0.0, budget * max(0.0, 1.0 - min(weight_sum, 1.0))))
    alloc = {a: float(w * budget) for a, w in zip(result.assets, result.weights)}
    warnings = []
    if result.cvar > max_cvar + 1e-6:
        warnings.append(f"optimized CVaR {result.cvar:.4f} exceeds max_cvar {max_cvar:.4f}")
    if gate in {"delegated", "auto"}:
        warnings.append("live gate selected; commands are emitted only after --allow-live --yes")
    return AllocationPlan(
        assets=result.assets,
        weights=result.weights.tolist(),
        notional_usdc=budget,
        gross_notional_usdc=gross_notional,
        cash_usdc=cash_usdc,
        allocations_usdc=alloc,
        expected_return=result.expected_return,
        cvar=result.cvar,
        sharpe=result.sharpe,
        solver=result.solver,
        scenario_backend=sc.backend,
        scenario_count=sc.n_scenarios,
        price_sources=price_sources,
        constraints={
            "cvar_alpha": cvar_alpha,
            "max_cvar": max_cvar,
            "max_leverage": max_leverage,
            "max_cardinality": max_cardinality,
            "days": n_days,
            "seed": seed,
        },
        gate=gate,
        warnings=warnings,
    )


def emit_vulcan_commands(plan: AllocationPlan) -> list[str]:
    """Translate allocation plan → Vulcan CLI commands."""
    cmds = []
    mode = "paper" if plan.gate in ("paper", "observer") else "live"
    for asset, usdc in plan.allocations_usdc.items():
        if usdc < 1.0:
            continue
        cmd = (
            f"vulcan trade market-buy {asset} "
            f"--notional-usdc {usdc:.2f} "
            f"--mode {mode}"
        )
        if plan.gate == "observer":
            cmd = f"# [observer — not executed] {cmd}"
        cmds.append(cmd)
    return cmds


def build_run(plan: AllocationPlan, allow_live: bool) -> PortfolioRun:
    commands = emit_vulcan_commands(plan)
    return PortfolioRun(
        schema_version="2026-06-22",
        generated_at=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        blueprint="nvidia/quantitative-portfolio-optimization",
        trust_gate={
            "gate": plan.gate,
            "description": TRUST_GATES[plan.gate],
            "live_allowed": str(bool(allow_live)).lower(),
        },
        plan=plan,
        vulcan_commands=commands,
        execution={
            "executed": False,
            "reason": "blueprint emits reviewable commands only; execution remains external and paper/live gated",
            "paper_default": plan.gate in {"observer", "paper"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clawd portfolio optimizer (Blueprint 2)")
    parser.add_argument("--assets", nargs="+", default=["SOL", "BTC", "ETH"])
    parser.add_argument("--budget", type=float, default=1000.0)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--max-cvar", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--gate", default="paper", choices=list(TRUST_GATES))
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--no-phoenix", action="store_true", help="Use synthetic prices instead of live")
    parser.add_argument("--days", type=int, default=90, help="Days of price history to fetch")
    parser.add_argument("--n-scenarios", type=int, default=5000)
    parser.add_argument("--max-cardinality", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Write machine-readable handoff JSON")
    parser.add_argument("--allow-live", action="store_true", help="Allow delegated/auto command emission")
    parser.add_argument("--yes", action="store_true", help="Confirm live gate review")
    args = parser.parse_args()

    if args.gate in {"delegated", "auto"} and not (args.allow_live and args.yes):
        print(
            "ERROR: delegated/auto gates require --allow-live --yes. "
            "Use --gate observer or --gate paper for local runs.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"[portfolio-opt] gate={args.gate}: {TRUST_GATES[args.gate]}", file=sys.stderr)
    plan = build_plan(
        args.assets, args.budget, args.cvar_alpha,
        args.max_cvar, args.max_leverage, args.gate,
        use_phoenix=not args.no_phoenix,
        n_days=args.days,
        n_scenarios=args.n_scenarios,
        max_cardinality=args.max_cardinality,
        seed=args.seed,
        verbose=not args.json,
    )
    run = build_run(plan, allow_live=args.allow_live and args.yes)

    if args.json:
        print(json.dumps(asdict(run), indent=2, sort_keys=True))
    else:
        print(f"\nAllocation plan ({plan.solver})")
        print(
            f"  E[ret]={plan.expected_return:.4f}  CVaR={plan.cvar:.4f}  "
            f"Sharpe={plan.sharpe:.3f}  scenarios={plan.scenario_count} ({plan.scenario_backend})"
        )
        print(f"  gross=${plan.gross_notional_usdc:.2f}  cash=${plan.cash_usdc:.2f}")
        for a, usdc in plan.allocations_usdc.items():
            w = plan.weights[plan.assets.index(a)]
            print(f"  {a:8s}  {w*100:5.1f}%  ${usdc:8.2f}")
        for warning in plan.warnings:
            print(f"  WARNING: {warning}")

    print("\n[portfolio-opt] Vulcan commands:")
    for cmd in run.vulcan_commands:
        print(f"  {cmd}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(run), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[portfolio-opt] wrote {out_path}")


if __name__ == "__main__":
    main()

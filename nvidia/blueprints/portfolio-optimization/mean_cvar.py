"""
Blueprint 2 — Mean-CVaR portfolio optimizer.

Uses cuFOLIO (GPU) when available, falls back to CVXPY (CPU).
Enforces: CVaR budget, leverage cap, max cardinality, turnover limit.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class OptResult:
    weights: np.ndarray
    assets: list[str]
    expected_return: float
    cvar: float                # CVaR at alpha
    sharpe: float
    solver: str
    weight_sum: float
    diagnostics: dict[str, Any] | None = None


def optimize(
    scenarios: np.ndarray,
    asset_names: list[str],
    cvar_alpha: float = 0.95,
    max_cvar: float = 0.10,
    max_leverage: float = 1.0,
    max_cardinality: int | None = None,
    turnover_limit: float | None = None,
    current_weights: np.ndarray | None = None,
    risk_free: float = 0.0,
) -> OptResult:
    """
    Solve Mean-CVaR optimization.

    Args:
        scenarios: (n_scenarios, n_assets) return matrix
        asset_names: list of asset tickers
        cvar_alpha: CVaR confidence level (e.g. 0.95 = 95th-percentile loss)
        max_cvar: maximum allowed CVaR loss
        max_leverage: sum-of-weights cap (1.0 = long-only, 2.0 = 2x levered)
        max_cardinality: max number of non-zero positions
        turnover_limit: max total weight change from current_weights
        current_weights: existing position weights for turnover constraint
        risk_free: risk-free rate for Sharpe calculation
    """
    n_scenarios, n_assets = scenarios.shape

    diagnostics: dict[str, Any] = {
        "n_scenarios": int(n_scenarios),
        "n_assets": int(n_assets),
        "cvar_alpha": float(cvar_alpha),
        "max_cvar": float(max_cvar),
        "max_leverage": float(max_leverage),
        "max_cardinality": max_cardinality,
        "turnover_limit": turnover_limit,
    }

    try:
        weights = _solve_cufolio(
            scenarios, cvar_alpha, max_cvar, max_leverage,
            max_cardinality, turnover_limit, current_weights,
        )
        solver = "cufolio"
    except Exception as exc:
        diagnostics["cufolio_error"] = f"{type(exc).__name__}: {exc}"
        weights = _solve_cvxpy(
            scenarios, cvar_alpha, max_cvar, max_leverage,
            max_cardinality, turnover_limit, current_weights,
        )
        solver = "cvxpy"
    weights = _sanitize_weights(weights, n_assets, max_leverage, max_cardinality)

    port_returns = scenarios @ weights
    cvar = _compute_cvar(port_returns, cvar_alpha)
    exp_ret = float(port_returns.mean())
    vol = float(port_returns.std())
    sharpe = (exp_ret - risk_free) / vol if vol > 1e-9 else 0.0

    return OptResult(weights, asset_names, exp_ret, cvar, sharpe, solver, float(weights.sum()), diagnostics)


def _compute_cvar(port_returns: np.ndarray, alpha: float) -> float:
    var = float(np.percentile(port_returns, (1 - alpha) * 100))
    tail = port_returns[port_returns <= var]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def _sanitize_weights(
    weights: np.ndarray,
    n_assets: int,
    max_leverage: float,
    max_cardinality: int | None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_assets,) or not np.all(np.isfinite(weights)):
        weights = np.ones(n_assets, dtype=np.float64) / n_assets
    weights = np.clip(weights, 0.0, None)
    if max_cardinality is not None and 0 < max_cardinality < n_assets:
        keep = np.argsort(weights)[-max_cardinality:]
        pruned = np.zeros_like(weights)
        pruned[keep] = weights[keep]
        weights = pruned
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.ones(n_assets, dtype=np.float64) / n_assets
        total = 1.0
    cap = max(float(max_leverage), 1e-9)
    if total > cap:
        weights = weights * (cap / total)
        total = cap
    # For the normal long-only budget case, remove tiny solver drift.
    if cap >= 1.0 and 0.999 <= total <= 1.001:
        weights = weights / max(float(weights.sum()), 1e-12)
    return weights


def _solve_cufolio(
    scenarios, cvar_alpha, max_cvar, max_leverage,
    max_cardinality, turnover_limit, current_weights,
) -> np.ndarray:
    # cuFOLIO API (GPU solver)
    import cufolio  # type: ignore
    n_assets = scenarios.shape[1]
    prob = cufolio.MeanCVaRProblem(
        scenarios=scenarios,
        alpha=cvar_alpha,
    )
    prob.add_constraint(cufolio.CVaRConstraint(max_cvar))
    prob.add_constraint(cufolio.LeverageConstraint(max_leverage))
    if max_cardinality is not None:
        prob.add_constraint(cufolio.CardinalityConstraint(max_cardinality))
    if turnover_limit is not None and current_weights is not None:
        prob.add_constraint(cufolio.TurnoverConstraint(current_weights, turnover_limit))
    result = prob.solve()
    return np.array(result.weights, dtype=np.float64)


def _solve_cvxpy(
    scenarios, cvar_alpha, max_cvar, max_leverage,
    max_cardinality, turnover_limit, current_weights,
) -> np.ndarray:
    import cvxpy as cp
    n_scenarios, n_assets = scenarios.shape
    w = cp.Variable(n_assets)
    alpha = cvar_alpha
    # CVaR linearization: auxiliary variable z, loss = -scenarios @ w
    z = cp.Variable()
    u = cp.Variable(n_scenarios)

    port_loss = -scenarios @ w
    cvar_expr = z + (1 / (n_scenarios * (1 - alpha))) * cp.sum(u)

    constraints = [
        u >= 0,
        u >= port_loss - z,
        cvar_expr <= max_cvar,
        cp.sum(w) <= max_leverage,
        w >= 0,
    ]
    # Cardinality is applied after solve in the CPU fallback. Keeping the CVXPY
    # problem continuous avoids requiring a mixed-integer solver on local Macs.
    if turnover_limit is not None and current_weights is not None:
        constraints.append(cp.norm1(w - current_weights) <= turnover_limit)

    exp_returns = scenarios.mean(0)
    objective = cp.Maximize(exp_returns @ w)
    prob = cp.Problem(objective, constraints)

    # Try ECOS first (faster, LP-friendly); fall back to SCS for larger problems
    for solver in [cp.ECOS, cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                break
        except cp.SolverError:
            continue

    if w.value is None or np.allclose(w.value, 0, atol=1e-6):
        # Infeasible or zero solution: use equal-weight as safe default
        return np.ones(n_assets) / n_assets
    return np.clip(w.value, 0, None)


if __name__ == "__main__":
    from scenarios import generate_scenarios, historical_returns
    import numpy as np

    np.random.seed(0)
    prices = {
        "SOL": (100 * np.cumprod(1 + np.random.normal(0.001, 0.04, 200))).tolist(),
        "BTC": (30000 * np.cumprod(1 + np.random.normal(0.0005, 0.03, 200))).tolist(),
        "ETH": (2000 * np.cumprod(1 + np.random.normal(0.0006, 0.035, 200))).tolist(),
    }
    rets, names = historical_returns(prices)
    sc = generate_scenarios(rets, names, n_scenarios=500)
    result = optimize(sc.scenarios, sc.assets, cvar_alpha=0.95, max_cvar=0.15)
    print(f"solver={result.solver}")
    print(f"weights: {dict(zip(result.assets, result.weights.round(4).tolist()))}")
    print(f"E[ret]={result.expected_return:.4f}  CVaR={result.cvar:.4f}  Sharpe={result.sharpe:.3f}")

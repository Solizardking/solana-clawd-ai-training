"""
Blueprint 2 — cuML KDE Monte Carlo scenario generator.

Generates return scenarios for Solana DeFi assets using GPU-accelerated
kernel density estimation when cuML is available, falling back to scipy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class ScenarioResult:
    assets: list[str]
    scenarios: np.ndarray      # shape (n_scenarios, n_assets)
    n_scenarios: int
    backend: str               # "cuml", "scipy", or "bootstrap"
    diagnostics: dict[str, Any] | None = None


def generate_scenarios(
    returns: np.ndarray,
    asset_names: list[str],
    n_scenarios: int = 10_000,
    bandwidth: float = 0.1,
    seed: int = 42,
) -> ScenarioResult:
    """
    Fit KDE on historical returns, sample n_scenarios Monte Carlo paths.

    Args:
        returns: (n_days, n_assets) array of daily log returns
        asset_names: list of asset ticker strings
        n_scenarios: number of scenarios to sample
        bandwidth: KDE bandwidth (Scott's rule if 0)
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be a 2D matrix, got shape={returns.shape}")
    if returns.shape[0] < 2:
        raise ValueError("at least two return rows are required to generate scenarios")
    if returns.shape[1] != len(asset_names):
        raise ValueError("asset_names length must match returns column count")
    n_assets = returns.shape[1]
    diagnostics = {
        "return_rows": int(returns.shape[0]),
        "asset_count": int(n_assets),
        "bandwidth": bandwidth,
        "seed": seed,
    }
    try:
        from cuml.neighbors import KernelDensity as cuKDE
        import cupy as cp

        gpu_returns = cp.asarray(returns, dtype=cp.float32)
        samples = np.zeros((n_scenarios, n_assets), dtype=np.float32)
        for i in range(n_assets):
            col = gpu_returns[:, i : i + 1]
            bw = bandwidth if bandwidth > 0 else float(col.std()) * (len(returns) ** -0.2)
            kde = cuKDE(bandwidth=bw, kernel="gaussian")
            kde.fit(col)
            s = kde.sample(n_scenarios)
            samples[:, i] = cp.asnumpy(s).ravel()
        return ScenarioResult(asset_names, samples, n_scenarios, "cuml", diagnostics)

    except ImportError:
        pass

    try:
        from scipy.stats import gaussian_kde

        samples = np.zeros((n_scenarios, n_assets), dtype=np.float32)
        for i in range(n_assets):
            col = returns[:, i]
            if float(np.std(col)) < 1e-12:
                samples[:, i] = col[-1]
                continue
            bw = bandwidth if bandwidth > 0 else "scott"
            kde = gaussian_kde(col, bw_method=bw)
            samples[:, i] = kde.resample(n_scenarios, seed=seed + i)[0]
        return ScenarioResult(asset_names, samples, n_scenarios, "scipy", diagnostics)
    except ImportError:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, returns.shape[0], size=n_scenarios)
        samples = returns[idx, :].astype(np.float32)
        return ScenarioResult(asset_names, samples, n_scenarios, "bootstrap", diagnostics)


def historical_returns(prices: dict[str, list[float]]) -> tuple[np.ndarray, list[str]]:
    """Convert price dict → log-return matrix (n_days-1, n_assets)."""
    assets = sorted(prices.keys())
    if not assets:
        raise ValueError("prices must contain at least one asset")
    cols = []
    for a in assets:
        p = np.array(prices[a], dtype=np.float64)
        if len(p) < 2:
            raise ValueError(f"{a} needs at least two prices")
        if np.any(p <= 0):
            raise ValueError(f"{a} contains non-positive prices")
        cols.append(np.diff(np.log(p)))
    min_len = min(len(c) for c in cols)
    matrix = np.column_stack([c[-min_len:] for c in cols])
    return matrix, assets


if __name__ == "__main__":
    import json, sys
    # demo with synthetic prices
    np.random.seed(42)
    prices = {
        "SOL":  (100 * np.cumprod(1 + np.random.normal(0.001, 0.04, 365))).tolist(),
        "BTC":  (30000 * np.cumprod(1 + np.random.normal(0.0005, 0.03, 365))).tolist(),
        "ETH":  (2000 * np.cumprod(1 + np.random.normal(0.0006, 0.035, 365))).tolist(),
        "BONK": (0.00001 * np.cumprod(1 + np.random.normal(0.002, 0.08, 365))).tolist(),
    }
    rets, names = historical_returns(prices)
    result = generate_scenarios(rets, names, n_scenarios=1000)
    print(f"backend={result.backend} scenarios={result.scenarios.shape}")
    print(f"mean returns: { dict(zip(result.assets, result.scenarios.mean(0).tolist())) }")

"""
Vulcan CLI wrapper for Clawd agents.

Vulcan is the official command-line tool for Phoenix perpetual futures on Solana.
This module provides a Python subprocess wrapper with JSON output parsing.

Usage:
    from perps.vulcan import VulcanClient

    vc = VulcanClient()
    ticker = vc.market_ticker("SOL")
    orderbook = vc.market_orderbook("SOL", depth=5)
    result = vc.trade_market_buy("SOL", notional_usdc=100, tp=250, sl=180)

See: https://docs.phoenix.trade/cli
"""
from __future__ import annotations

import enum
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OutputFormat(enum.Enum):
    TABLE = "table"
    JSON = "json"


@dataclass
class VulcanConfig:
    """Configuration for the Vulcan CLI."""
    wallet: str = ""
    rpc_url: str = ""
    api_url: str = ""
    dry_run: bool = False
    output: OutputFormat = OutputFormat.JSON
    verbose: bool = False

    def to_args(self) -> list[str]:
        args = []
        if self.wallet:
            args.extend(["-w", self.wallet])
        if self.rpc_url:
            args.extend(["--rpc-url", self.rpc_url])
        if self.api_url:
            args.extend(["--api-url", self.api_url])
        if self.dry_run:
            args.append("--dry-run")
        args.extend(["-o", self.output.value])
        if self.verbose:
            args.append("-v")
        return args


@dataclass
class VulcanResult:
    """Parsed result from a Vulcan CLI command."""
    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    raw: str = ""

    @classmethod
    def from_json(cls, raw: str) -> "VulcanResult":
        try:
            parsed = json.loads(raw)
            return cls(
                ok=parsed.get("ok", False),
                data=parsed.get("data"),
                error=parsed.get("error"),
                raw=raw,
            )
        except (json.JSONDecodeError, KeyError):
            return cls(ok=False, raw=raw, error={"message": "Failed to parse output"})


class VulcanClient:
    """Subprocess wrapper around the Vulcan CLI for Phoenix perps.

    All commands use -o json for structured output.
    """

    def __init__(self, binary: str = "vulcan", config: VulcanConfig | None = None):
        self.binary = binary
        self.config = config or VulcanConfig()

    def _run(self, *args: str) -> VulcanResult:
        """Execute a vulcan command and return parsed result."""
        cmd = [self.binary]
        cmd.extend(self.config.to_args())
        cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return VulcanResult(
                    ok=False,
                    error={"message": result.stderr.strip() or "Command failed"},
                    raw=result.stdout + result.stderr,
                )
            return VulcanResult.from_json(result.stdout)
        except subprocess.TimeoutExpired:
            return VulcanResult(
                ok=False,
                error={"message": "Command timed out (30s)", "retryable": True},
            )
        except FileNotFoundError:
            return VulcanResult(
                ok=False,
                error={"message": f"Vulcan CLI not found: {self.binary}. Install: cargo install vulcan-cli"},
            )
        except Exception as e:
            return VulcanResult(
                ok=False,
                error={"message": str(e)},
            )

    # ── Wallet Management ──

    def wallet_create(self, name: str) -> VulcanResult:
        """Generate a new Solana keypair."""
        return self._run("wallet", "create", "--name", name)

    def wallet_list(self) -> VulcanResult:
        """List all stored wallets."""
        return self._run("wallet", "list")

    def wallet_balance(self, name: str | None = None) -> VulcanResult:
        """Show SOL and USDC balances."""
        if name:
            return self._run("wallet", "balance", name)
        return self._run("wallet", "balance")

    def wallet_import(self, name: str, source: str, fmt: str = "base58") -> VulcanResult:
        """Import a wallet from base58 string, bytes, or file."""
        return self._run("wallet", "import", "--name", name, f"--format={fmt}", source)

    # ── Market Data ──

    def market_list(self) -> VulcanResult:
        """List all available perpetual markets."""
        return self._run("market", "list")

    def market_ticker(self, symbol: str) -> VulcanResult:
        """Current price, 24h volume, open interest, funding rate."""
        return self._run("market", "ticker", symbol)

    def market_orderbook(self, symbol: str, depth: int = 10) -> VulcanResult:
        """L2 orderbook snapshot."""
        return self._run("market", "orderbook", symbol, "--depth", str(depth))

    def market_candles(self, symbol: str, interval: str = "1h",
                       limit: int = 20, indicators: list[str] | None = None) -> VulcanResult:
        """OHLCV candles with optional indicators."""
        args = ["market", "candles", symbol, "--interval", interval, "--limit", str(limit)]
        if indicators:
            args.extend(["--with-indicators", ",".join(indicators)])
        return self._run(*args)

    def market_trades(self, symbol: str, limit: int = 20) -> VulcanResult:
        """Recent trades."""
        return self._run("market", "trades", symbol, "--limit", str(limit))

    def market_info(self, symbol: str) -> VulcanResult:
        """Detailed market config (tick size, lot size, fees, leverage tiers)."""
        return self._run("market", "info", symbol)

    def funding_rates(self, symbol: str, limit: int = 20) -> VulcanResult:
        """Historical funding rates."""
        return self._run("market", "funding-rates", symbol, "--limit", str(limit))

    # ── Order Management ──

    def trade_market_buy(self, symbol: str, size: str | None = None,
                         tokens: float | None = None, notional_usdc: float | None = None,
                         tp: float | None = None, sl: float | None = None,
                         isolated: bool = False, collateral: float | None = None,
                         reduce_only: bool = False) -> VulcanResult:
        """Place a market buy order."""
        args = ["trade", "market-buy", symbol]
        if size:
            args.append(size)
        elif tokens:
            args.extend(["--tokens", str(tokens)])
        elif notional_usdc:
            args.extend(["--notional-usdc", str(notional_usdc)])
        if tp:
            args.extend(["--tp", str(tp)])
        if sl:
            args.extend(["--sl", str(sl)])
        if isolated:
            args.append("--isolated")
            if collateral:
                args.extend(["--collateral", str(collateral)])
        if reduce_only:
            args.append("--reduce-only")
        return self._run(*args)

    def trade_market_sell(self, symbol: str, size: str | None = None,
                          tokens: float | None = None, notional_usdc: float | None = None,
                          tp: float | None = None, sl: float | None = None,
                          isolated: bool = False, collateral: float | None = None,
                          reduce_only: bool = False) -> VulcanResult:
        """Place a market sell order."""
        args = ["trade", "market-sell", symbol]
        if size:
            args.append(size)
        elif tokens:
            args.extend(["--tokens", str(tokens)])
        elif notional_usdc:
            args.extend(["--notional-usdc", str(notional_usdc)])
        if tp:
            args.extend(["--tp", str(tp)])
        if sl:
            args.extend(["--sl", str(sl)])
        if isolated:
            args.append("--isolated")
            if collateral:
                args.extend(["--collateral", str(collateral)])
        if reduce_only:
            args.append("--reduce-only")
        return self._run(*args)

    def trade_limit_buy(self, symbol: str, size: str, price: float,
                        tp: float | None = None, sl: float | None = None,
                        isolated: bool = False, collateral: float | None = None,
                        reduce_only: bool = False) -> VulcanResult:
        """Place a limit buy order."""
        args = ["trade", "limit-buy", symbol, size, str(price)]
        if tp:
            args.extend(["--tp", str(tp)])
        if sl:
            args.extend(["--sl", str(sl)])
        if isolated:
            args.append("--isolated")
            if collateral:
                args.extend(["--collateral", str(collateral)])
        if reduce_only:
            args.append("--reduce-only")
        return self._run(*args)

    def trade_limit_sell(self, symbol: str, size: str, price: float,
                         tp: float | None = None, sl: float | None = None,
                         isolated: bool = False, collateral: float | None = None,
                         reduce_only: bool = False) -> VulcanResult:
        """Place a limit sell order."""
        args = ["trade", "limit-sell", symbol, size, str(price)]
        if tp:
            args.extend(["--tp", str(tp)])
        if sl:
            args.extend(["--sl", str(sl)])
        if isolated:
            args.append("--isolated")
            if collateral:
                args.extend(["--collateral", str(collateral)])
        if reduce_only:
            args.append("--reduce-only")
        return self._run(*args)

    def trade_cancel(self, symbol: str, *order_ids: str) -> VulcanResult:
        """Cancel specific orders by ID."""
        return self._run("trade", "cancel", symbol, *order_ids)

    def trade_cancel_all(self, symbol: str | None = None) -> VulcanResult:
        """Cancel all open orders."""
        args = ["trade", "cancel-all"]
        if symbol:
            args.append(symbol)
        return self._run(*args)

    def trade_orders(self, symbol: str | None = None) -> VulcanResult:
        """List open orders."""
        args = ["trade", "orders"]
        if symbol:
            args.append(symbol)
        return self._run(*args)

    def trade_set_tpsl(self, symbol: str, tp_price: float | None = None,
                       sl_price: float | None = None) -> VulcanResult:
        """Set take-profit or stop-loss on an existing position."""
        args = ["trade", "set-tpsl", symbol]
        if tp_price:
            args.extend(["--tp", str(tp_price)])
        if sl_price:
            args.extend(["--sl", str(sl_price)])
        return self._run(*args)

    # ── Position Management ──

    def position_list(self) -> VulcanResult:
        """List all open positions."""
        return self._run("position", "list")

    def position_show(self, symbol: str) -> VulcanResult:
        """Detailed view of a specific position."""
        return self._run("position", "show", symbol)

    def position_close(self, symbol: str) -> VulcanResult:
        """Close an entire position."""
        return self._run("position", "close", symbol)

    def position_close_all(self) -> VulcanResult:
        """Close every open position."""
        return self._run("position", "close-all")

    def position_reduce(self, symbol: str, size: str) -> VulcanResult:
        """Reduce a position by size."""
        return self._run("position", "reduce", symbol, size)

    # ── Collateral Management ──

    def margin_status(self) -> VulcanResult:
        """Show cross-margin health, equity, maintenance margin."""
        return self._run("margin", "status")

    def margin_deposit(self, amount: float) -> VulcanResult:
        """Deposit USDC collateral."""
        return self._run("margin", "deposit", str(amount))

    def margin_withdraw(self, amount: float) -> VulcanResult:
        """Withdraw USDC collateral."""
        return self._run("margin", "withdraw", str(amount))

    # ── Trader Account ──

    def account_register(self, code: str) -> VulcanResult:
        """Register a trader account with access/referral/invite code."""
        return self._run("account", "register", "--access-code", code)

    def account_info(self) -> VulcanResult:
        """Show trader account details."""
        return self._run("account", "info")

    # ── Paper Trading ──

    def paper_init(self, balance: float = 10000.0, fee_bps: int = 0) -> VulcanResult:
        """Initialize paper trading account."""
        return self._run("paper", "init", "--balance", str(balance), "--fee-bps", str(fee_bps))

    def paper_status(self) -> VulcanResult:
        """Show paper account status."""
        return self._run("paper", "status")

    def paper_buy(self, symbol: str, notional_usdc: float,
                  order_type: str = "market", price: float | None = None) -> VulcanResult:
        """Place a paper buy order."""
        args = ["paper", "buy", symbol, "--notional-usdc", str(notional_usdc), "--type", order_type]
        if price:
            args.extend(["--price", str(price)])
        return self._run(*args)

    def paper_sell(self, symbol: str, notional_usdc: float,
                   order_type: str = "market", price: float | None = None) -> VulcanResult:
        """Place a paper sell order."""
        args = ["paper", "sell", symbol, "--notional-usdc", str(notional_usdc), "--type", order_type]
        if price:
            args.extend(["--price", str(price)])
        return self._run(*args)

    # ── Portfolio ──

    def portfolio(self, include: list[str] | None = None) -> VulcanResult:
        """Portfolio snapshot (margin, positions, orders)."""
        args = ["portfolio"]
        if include:
            args.extend(["--include", ",".join(include)])
        return self._run(*args)

    # ── Strategy ──

    def strategy_twap_start(self, symbol: str, side: str, total_size: str,
                            num_slices: int, duration_minutes: int) -> VulcanResult:
        """Start a TWAP strategy run."""
        return self._run(
            "strategy", "twap", "start", symbol, side,
            "--total-size", total_size,
            "--num-slices", str(num_slices),
            "--duration-minutes", str(duration_minutes),
        )

    def strategy_grid_start(self, symbol: str, lower_price: float, upper_price: float,
                            num_orders: int, total_size: str) -> VulcanResult:
        """Start a grid trading run."""
        return self._run(
            "strategy", "grid", "start", symbol,
            "--lower-price", str(lower_price),
            "--upper-price", str(upper_price),
            "--num-orders", str(num_orders),
            "--total-size", total_size,
        )

    def strategy_runs(self, limit: int = 10) -> VulcanResult:
        """List persisted strategy runs."""
        return self._run("strategy", "runs", "--limit", str(limit))

    def strategy_pause(self, run_id: str) -> VulcanResult:
        """Pause a running strategy."""
        return self._run("strategy", "pause", run_id)

    def strategy_stop(self, run_id: str) -> VulcanResult:
        """Stop a strategy permanently."""
        return self._run("strategy", "stop", run_id)

    # ── History ──

    def history_trades(self, symbol: str | None = None, limit: int = 20) -> VulcanResult:
        """Past trade and fill history."""
        args = ["history", "trades", "--limit", str(limit)]
        if symbol:
            args.extend(["--symbol", symbol])
        return self._run(*args)

    def history_pnl(self, resolution: str = "hourly", limit: int = 24) -> VulcanResult:
        """PnL over time."""
        return self._run("history", "pnl", "--resolution", resolution, "--limit", str(limit))

    # ── Technical Analysis ──

    def ta_compute(self, symbol: str, indicator: str, timeframe: str = "1h",
                   period: int | None = None, params: dict | None = None) -> VulcanResult:
        """Compute a technical indicator."""
        args = ["ta", "compute", symbol, "--indicator", indicator, "--timeframe", timeframe]
        if period:
            args.extend(["--period", str(period)])
        if params:
            args.extend(["--params", json.dumps(params)])
        return self._run(*args)

    def ta_signal(self, symbol: str, spec: dict) -> VulcanResult:
        """Evaluate a trigger spec against the latest indicator value."""
        return self._run("ta", "signal", symbol, "--spec", json.dumps(spec))

    def ta_report(self, symbol: str, timeframe: str = "1h") -> VulcanResult:
        """Multi-indicator snapshot (RSI, MACD, BBands, ATR, ADX)."""
        return self._run("ta", "report", symbol, "--timeframe", timeframe)

    # ── Auth ──

    def auth_login(self) -> VulcanResult:
        """Log in to the Phoenix API."""
        return self._run("auth", "login")

    def auth_status(self) -> VulcanResult:
        """Show Phoenix API session status."""
        return self._run("auth", "status")

    # ── Status ──

    def status(self) -> VulcanResult:
        """Check configuration, connectivity, wallet, and registration."""
        return self._run("status")

    def version(self) -> VulcanResult:
        """Print version and build information."""
        return self._run("version")
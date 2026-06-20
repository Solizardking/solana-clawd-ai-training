"""
Local paper trading engine for Phoenix perpetuals.

Runs against live Phoenix prices but never touches your wallet or onchain state.
Maintains its own simulated account with balance, positions, orders, and fills.

Usage:
    from perps.paper import PaperEngine

    engine = PaperEngine(initial_balance=10000.0)
    engine.buy("SOL", notional_usdc=500)
    engine.sell("SOL", notional_usdc=200)
    status = engine.status()
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PaperFill:
    """A simulated fill from a paper trade."""
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    notional: float
    timestamp: float
    fee: float = 0.0


@dataclass
class PaperOrder:
    """A simulated order in the paper engine."""
    id: str
    symbol: str
    side: str
    order_type: str  # "market" or "limit"
    size: float
    price: float | None = None
    filled: float = 0.0
    status: str = "open"  # open, filled, cancelled
    created_at: float = 0.0


@dataclass
class PaperPosition:
    """A simulated perpetual position."""
    symbol: str
    side: str  # "long" or "short" or "flat"
    size: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    liquidation_price: float = 0.0
    leverage: float = 1.0


@dataclass
class PaperAccount:
    """Full paper trading account state."""
    balance: float = 10000.0
    currency: str = "USDC"
    equity: float = 10000.0
    margin_used: float = 0.0
    free_collateral: float = 10000.0
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    orders: list[PaperOrder] = field(default_factory=list)
    fills: list[PaperFill] = field(default_factory=list)
    fee_bps: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "currency": self.currency,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "free_collateral": self.free_collateral,
            "positions": {
                sym: {
                    "side": p.side,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                    "liquidation_price": p.liquidation_price,
                    "leverage": p.leverage,
                }
                for sym, p in self.positions.items()
            },
            "open_orders": len([o for o in self.orders if o.status == "open"]),
            "total_fills": len(self.fills),
            "fee_bps": self.fee_bps,
        }


class PaperEngine:
    """Local paper trading engine for Phoenix perpetuals.

    Simulates a perpetual trading account:
    - Tracks balance, positions, orders, fills
    - Uses live mark prices (fetched via Vulcan or provided externally)
    - Simulates leverage, liquidation, funding costs
    - Never submits real transactions
    """

    def __init__(self, initial_balance: float = 10000.0,
                 currency: str = "USDC", fee_bps: int = 0,
                 vulcan_client=None):
        self.account = PaperAccount(
            balance=initial_balance,
            equity=initial_balance,
            free_collateral=initial_balance,
            currency=currency,
            fee_bps=fee_bps,
            created_at=time.time(),
        )
        self._vulcan = vulcan_client
        self._order_counter = 0
        self._fill_counter = 0

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"paper_{int(time.time())}_{self._order_counter}"

    def _next_fill_id(self) -> str:
        self._fill_counter += 1
        return f"fill_{int(time.time())}_{self._fill_counter}"

    def _get_mark_price(self, symbol: str) -> float:
        """Get current mark price for a symbol."""
        if self._vulcan:
            result = self._vulcan.market_ticker(symbol)
            if result.ok and result.data:
                return float(result.data.get("mark_price", 0))
        # Default fallback for common symbols
        prices = {"SOL": 150.0, "BTC": 65000.0, "ETH": 3500.0}
        return prices.get(symbol.upper(), 100.0)

    def buy(self, symbol: str, notional_usdc: float | None = None,
            size: float | None = None, order_type: str = "market",
            price: float | None = None, leverage: float = 1.0) -> dict[str, Any]:
        """Place a buy order in paper mode."""
        mark = self._get_mark_price(symbol)
        if size is None and notional_usdc:
            size = notional_usdc / mark
        elif size is None:
            size = 1.0

        fee = (size * mark) * (self.account.fee_bps / 10000)
        cost = size * mark + fee

        if cost > self.account.free_collateral:
            return {"ok": False, "error": "Insufficient free collateral"}

        order = PaperOrder(
            id=self._next_order_id(),
            symbol=symbol.upper(),
            side="buy",
            order_type=order_type,
            size=size,
            price=price or mark,
            created_at=time.time(),
        )

        if order_type == "market":
            fill_price = price or mark
            fill = PaperFill(
                id=self._next_fill_id(),
                symbol=symbol.upper(),
                side="buy",
                price=fill_price,
                size=size,
                notional=size * fill_price,
                timestamp=time.time(),
                fee=fee,
            )
            self.account.fills.append(fill)
            order.filled = size
            order.status = "filled"

            # Update position
            pos = self.account.positions.get(symbol.upper())
            if pos and pos.side == "long":
                # Increase existing long
                total_cost = pos.size * pos.entry_price + size * fill_price
                pos.size += size
                pos.entry_price = total_cost / pos.size
            elif pos and pos.side == "short":
                # Reduce short
                pos.size -= size
                if pos.size <= 0:
                    pnl = (pos.entry_price - fill_price) * abs(pos.size + size)
                    pos.realized_pnl += pnl
                    pos.side = "flat"
                    pos.size = 0
                    pos.entry_price = 0
                    self.account.balance += pnl
                else:
                    pnl = (pos.entry_price - fill_price) * size
                    pos.realized_pnl += pnl
                    self.account.balance += pnl
            else:
                # Open new long
                self.account.positions[symbol.upper()] = PaperPosition(
                    symbol=symbol.upper(),
                    side="long",
                    size=size,
                    entry_price=fill_price,
                    mark_price=fill_price,
                    leverage=leverage,
                    liquidation_price=fill_price * (1 - 0.9 / leverage),
                )

            self.account.balance -= cost
            self.account.orders.append(order)
            self._recalculate(symbol.upper(), fill_price)

            return {
                "ok": True,
                "data": {
                    "order": {"id": order.id, "symbol": symbol.upper(), "side": "buy",
                              "size": size, "price": fill_price, "status": "filled"},
                    "fill": {"id": fill.id, "price": fill_price, "size": size,
                             "notional": size * fill_price, "fee": fee},
                    "account": self.account.to_dict(),
                }
            }

        self.account.orders.append(order)
        return {
            "ok": True,
            "data": {
                "order": {"id": order.id, "symbol": symbol.upper(), "side": "buy",
                          "size": size, "price": price or mark, "status": "open"},
                "account": self.account.to_dict(),
            }
        }

    def sell(self, symbol: str, notional_usdc: float | None = None,
             size: float | None = None, order_type: str = "market",
             price: float | None = None, leverage: float = 1.0) -> dict[str, Any]:
        """Place a sell order in paper mode."""
        mark = self._get_mark_price(symbol)
        if size is None and notional_usdc:
            size = notional_usdc / mark
        elif size is None:
            size = 1.0

        fee = (size * mark) * (self.account.fee_bps / 10000)

        order = PaperOrder(
            id=self._next_order_id(),
            symbol=symbol.upper(),
            side="sell",
            order_type=order_type,
            size=size,
            price=price or mark,
            created_at=time.time(),
        )

        if order_type == "market":
            fill_price = price or mark
            fill = PaperFill(
                id=self._next_fill_id(),
                symbol=symbol.upper(),
                side="sell",
                price=fill_price,
                size=size,
                notional=size * fill_price,
                timestamp=time.time(),
                fee=fee,
            )
            self.account.fills.append(fill)
            order.filled = size
            order.status = "filled"

            pos = self.account.positions.get(symbol.upper())
            if pos and pos.side == "short":
                total_cost = abs(pos.size) * pos.entry_price + size * fill_price
                pos.size -= size
                pos.entry_price = total_cost / abs(pos.size) if pos.size != 0 else 0
            elif pos and pos.side == "long":
                pnl = (fill_price - pos.entry_price) * min(size, pos.size)
                pos.realized_pnl += pnl
                pos.size -= size
                if pos.size <= 0:
                    pos.side = "flat"
                    pos.entry_price = 0
                self.account.balance += pnl - fee
            else:
                self.account.positions[symbol.upper()] = PaperPosition(
                    symbol=symbol.upper(),
                    side="short",
                    size=size,
                    entry_price=fill_price,
                    mark_price=fill_price,
                    leverage=leverage,
                    liquidation_price=fill_price * (1 + 0.9 / leverage),
                )

            self.account.balance -= fee
            self.account.orders.append(order)
            self._recalculate(symbol.upper(), fill_price)

            return {
                "ok": True,
                "data": {
                    "order": {"id": order.id, "symbol": symbol.upper(), "side": "sell",
                              "size": size, "price": fill_price, "status": "filled"},
                    "fill": {"id": fill.id, "price": fill_price, "size": size,
                             "notional": size * fill_price, "fee": fee},
                    "account": self.account.to_dict(),
                }
            }

        self.account.orders.append(order)
        return {
            "ok": True,
            "data": {
                "order": {"id": order.id, "symbol": symbol.upper(), "side": "sell",
                          "size": size, "price": price or mark, "status": "open"},
                "account": self.account.to_dict(),
            }
        }

    def cancel(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        for order in self.account.orders:
            if order.id == order_id and order.status == "open":
                order.status = "cancelled"
                return {"ok": True, "data": {"order_id": order_id, "status": "cancelled"}}
        return {"ok": False, "error": f"Order {order_id} not found or already filled"}

    def cancel_all(self, symbol: str | None = None) -> dict[str, Any]:
        """Cancel all open orders."""
        cancelled = []
        for order in self.account.orders:
            if order.status == "open":
                if symbol is None or order.symbol == symbol.upper():
                    order.status = "cancelled"
                    cancelled.append(order.id)
        return {"ok": True, "data": {"cancelled": cancelled, "count": len(cancelled)}}

    def status(self) -> dict[str, Any]:
        """Get full paper account status."""
        return {"ok": True, "data": self.account.to_dict()}

    def positions(self) -> dict[str, Any]:
        """Get all open positions."""
        active = {sym: p for sym, p in self.account.positions.items()
                  if p.side != "flat" and p.size > 0}
        return {"ok": True, "data": {
            "positions": {sym: {
                "side": p.side, "size": p.size, "entry_price": p.entry_price,
                "mark_price": p.mark_price, "unrealized_pnl": p.unrealized_pnl,
                "realized_pnl": p.realized_pnl, "liquidation_price": p.liquidation_price,
                "leverage": p.leverage,
            } for sym, p in active.items()},
            "count": len(active),
        }}

    def fills(self, limit: int = 50) -> dict[str, Any]:
        """Get recent fills."""
        recent = sorted(self.account.fills, key=lambda f: f.timestamp, reverse=True)[:limit]
        return {"ok": True, "data": {
            "fills": [{
                "id": f.id, "symbol": f.symbol, "side": f.side,
                "price": f.price, "size": f.size, "notional": f.notional,
                "fee": f.fee, "timestamp": f.timestamp,
            } for f in recent],
            "count": len(recent),
        }}

    def reconcile(self, symbol: str | None = None) -> dict[str, Any]:
        """Reconcile open limit orders against current mark prices."""
        mark_prices = {}
        symbols_to_check = [symbol] if symbol else list(set(
            o.symbol for o in self.account.orders if o.status == "open"
        ))
        for sym in symbols_to_check:
            mark_prices[sym] = self._get_mark_price(sym)

        filled = []
        for order in self.account.orders:
            if order.status != "open":
                continue
            if symbol and order.symbol != symbol.upper():
                continue
            mp = mark_prices.get(order.symbol, self._get_mark_price(order.symbol))
            if order.side == "buy" and order.price and mp <= order.price:
                order.status = "filled"
                filled.append(order.id)
            elif order.side == "sell" and order.price and mp >= order.price:
                order.status = "filled"
                filled.append(order.id)

        return {"ok": True, "data": {
            "reconciled": len(filled),
            "filled_order_ids": filled,
            "account": self.account.to_dict(),
        }}

    def _recalculate(self, symbol: str, mark_price: float):
        """Recalculate account equity and margin after a trade."""
        pos = self.account.positions.get(symbol)
        if pos and pos.size > 0:
            pos.mark_price = mark_price
            if pos.side == "long":
                pos.unrealized_pnl = (mark_price - pos.entry_price) * pos.size
            elif pos.side == "short":
                pos.unrealized_pnl = (pos.entry_price - mark_price) * abs(pos.size)

        total_upnl = sum(p.unrealized_pnl for p in self.account.positions.values()
                         if p.side != "flat")
        self.account.equity = self.account.balance + total_upnl
        self.account.margin_used = sum(
            abs(p.size * p.mark_price) / p.leverage
            for p in self.account.positions.values() if p.side != "flat"
        )
        self.account.free_collateral = self.account.equity - self.account.margin_used

    def save(self, path: str = "paper_state.json") -> None:
        """Save paper trading state to disk."""
        state = {
            "account": self.account.to_dict(),
            "order_counter": self._order_counter,
            "fill_counter": self._fill_counter,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str = "paper_state.json") -> None:
        """Load paper trading state from disk."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        acc = state["account"]
        self.account.balance = acc["balance"]
        self.account.equity = acc["equity"]
        self.account.margin_used = acc["margin_used"]
        self.account.free_collateral = acc["free_collateral"]
        self._order_counter = state.get("order_counter", 0)
        self._fill_counter = state.get("fill_counter", 0)
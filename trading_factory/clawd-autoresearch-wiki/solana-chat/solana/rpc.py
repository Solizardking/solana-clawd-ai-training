"""
Solana RPC Client — Solana-native data tools for model training and evaluation.

8 RPC commands for the training pipeline:
  wallet   <address>           SOL balance + SPL token portfolio
  tx       <signature>         Full transaction details
  token    <mint_address>      SPL token metadata, price, supply
  activity <address>           Recent transaction history
  nft      <address>           NFT portfolio (SPL + cNFT)
  whales   [--min-sol N]       Large SOL transfers in latest block
  stats                        Network health: slot, epoch, TPS, SOL price
  price    <mint_or_symbol>    Quick price lookup

Environment:
  SOLANA_RPC_URL   Override RPC endpoint
  HELIUS_API_KEY   Enable Helius DAS for cNFT/token data

Usage as data tool for training:
  from solana.rpc import SolanaClient
  client = SolanaClient()
  stats = client.stats()
  price = client.price("SOL")
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_RPC = "https://api.mainnet-beta.solana.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

KNOWN_SYMBOLS: dict[str, str] = {
    "SOL":  "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "JUP":  "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "WIF":  "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JTO":  "jtojtomepa8beP8AuQc6eXt5FriJwfFMwjx2ZEfchqd",
    "PYTH": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "DRIFT":"DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7",
    "mSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "CLAWD":"8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump",
}

COINGECKO_IDS: dict[str, str] = {
    "So11111111111111111111111111111111111111112": "solana",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "bonk",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": "jupiter-exchange-solana",
    "DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7": "drift-protocol",
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": "pyth-network",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "marinade-staked-sol",
}


class SolanaClient:
    """Minimal Solana RPC client for training pipeline data collection."""

    def __init__(self, rpc_url: str | None = None):
        self.rpc_url = rpc_url or os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC)
        self.helius_key = os.environ.get("HELIUS_API_KEY", "")

    def _fetch(self, url: str, data: bytes | None = None,
               headers: dict | None = None, retries: int = 2) -> Any:
        req = urllib.request.Request(url, data=data, headers=headers or {},
                                     method="POST" if data else "GET")
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception:
                if attempt < retries:
                    time.sleep(1)
                    continue
                raise
        return None

    def rpc(self, method: str, params: list | None = None) -> Any:
        payload = json.dumps({"jsonrpc": "2.0", "id": 1,
                              "method": method, "params": params or []}).encode()
        result = self._fetch(self.rpc_url, data=payload,
                             headers={"Content-Type": "application/json"})
        if result and "error" in result:
            raise RuntimeError(f"RPC error {method}: {result['error']}")
        return result.get("result") if result else None

    def _coingecko(self, path: str, params: dict | None = None) -> Any:
        qs = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        url = f"{COINGECKO_BASE}/{path}?{qs}" if qs else f"{COINGECKO_BASE}/{path}"
        try:
            return self._fetch(url)
        except Exception:
            return None

    def get_sol_price(self) -> float:
        data = self._coingecko("simple/price",
                               {"ids": "solana", "vs_currencies": "usd"})
        return float(data["solana"]["usd"]) if data and "solana" in data else 0.0

    def get_token_price_usd(self, mint: str) -> float:
        cg_id = COINGECKO_IDS.get(mint, "")
        if not cg_id:
            return 0.0
        data = self._coingecko("simple/price",
                               {"ids": cg_id, "vs_currencies": "usd"})
        return float(data[cg_id]["usd"]) if data and cg_id in data else 0.0

    def stats(self) -> dict:
        """Network stats: slot, epoch, TPS, SOL price."""
        epoch_result = self.rpc("getEpochInfo")
        slot = epoch_result.get("absoluteSlot", "?") if epoch_result else "?"
        epoch = epoch_result.get("epoch", "?") if epoch_result else "?"
        supply_result = self.rpc("getSupply")
        circulating = supply_result.get("value", {}).get("circulating", 0) if supply_result else 0
        if circulating:
            circulating /= 1e9
        perf = self.rpc("getRecentPerformanceSamples", [1])
        tps = 0.0
        if perf and isinstance(perf, list) and perf:
            s = perf[0]
            tps = s.get("numTransactions", 0) / max(s.get("samplePeriodSecs", 60), 1)
        sol_price = self.get_sol_price()
        return {
            "slot": slot, "epoch": epoch, "tps": round(tps, 1),
            "circulating_sol": circulating, "sol_price_usd": sol_price,
        }

    def wallet(self, address: str, limit: int = 20) -> dict:
        """Get SOL balance + SPL token portfolio. Returns dict for training."""
        result = self.rpc("getBalance", [address])
        sol_balance = (result.get("value", 0) if result else 0) / 1e9
        sol_price = self.get_sol_price()
        tokens = []
        result2 = self.rpc("getTokenAccountsByOwner", [
            address,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"},
        ])
        accounts = result2.get("value", []) if result2 else []
        for acct in accounts:
            info = acct.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            balance_raw = int(info.get("tokenAmount", {}).get("amount", "0"))
            decimals = info.get("tokenAmount", {}).get("decimals", 0)
            amount = balance_raw / (10 ** decimals) if decimals else balance_raw
            if amount > 0:
                mint = info.get("mint", "")
                name = next((k for k, v in KNOWN_SYMBOLS.items() if v == mint), mint[:8])
                tokens.append({"mint": mint, "name": name, "amount": amount})
        return {
            "address": address,
            "sol_balance": sol_balance,
            "sol_usd": sol_balance * sol_price,
            "token_count": len(tokens),
            "tokens": tokens[:limit],
        }


def format_sol(n: float) -> str:
    return f"{n:,.4f} SOL"


def format_usd(n: float) -> str:
    if n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n:,.2f}"
    return f"${n:.4f}"
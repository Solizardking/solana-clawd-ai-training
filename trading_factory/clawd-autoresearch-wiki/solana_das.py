"""
Helius Solana DAS + SPL helper for autoresearch.

This script adds a lightweight server-side integration for querying Solana
assets, NFTs, compressed NFTs, and fungible token data via Helius DAS and
standard Solana JSON-RPC methods.

Examples:
    export HELIUS_API_KEY="<your_api_key>"

    # DAS: single asset
    uv run solana_das.py get-asset EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v --show-fungible

    # DAS: wallet assets (NFTs + tokens)
    uv run solana_das.py owner-assets 86xCnPeV69n6t3DnyGvkKobf9FdN2H9oiVDdaMpo2MMY --token-type all --show-fungible --show-native-balance

    # DAS: advanced search
    uv run solana_das.py search --params '{"ownerAddress":"86xCnPeV69n6t3DnyGvkKobf9FdN2H9oiVDdaMpo2MMY","tokenType":"fungible","limit":50}'

    # SPL: token account balance
    uv run solana_das.py token-balance 3emsAVdmGKERbHjmGfQ6oZ1e35dkf5iYcS6U4CPKFVaa

    # SPL: token accounts by owner
    uv run solana_das.py token-accounts 86xCnPeV69n6t3DnyGvkKobf9FdN2H9oiVDdaMpo2MMY

    # SPL: mint supply
    uv run solana_das.py token-supply EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v

    # SPL: largest holders
    uv run solana_das.py token-largest he1iusmfkpAdwvxLNGV8Y1iSbj4rUy6yMhEA3fotn9A

    # Generic passthrough (for methods like getTransactionsForAddress)
    uv run solana_das.py rpc getTransactionsForAddress --params '{"address":"<WALLET>","limit":100}'
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

MAINNET_RPC = "https://mainnet.helius-rpc.com/"
DEVNET_RPC = "https://devnet.helius-rpc.com/"

TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


class HeliusAPIError(RuntimeError):
    """Raised when a Helius RPC call fails."""


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON params must decode to an object/dict")
    return data


@dataclass
class HeliusClient:
    api_key: str
    network: str = "mainnet"
    endpoint: Optional[str] = None
    timeout: float = 20.0
    max_retries: int = 3
    backoff_seconds: float = 0.75

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Missing API key. Set HELIUS_API_KEY or pass --api-key.")

        if self.endpoint:
            base = self.endpoint.rstrip("/")
        else:
            base = (
                DEVNET_RPC.rstrip("/")
                if self.network == "devnet"
                else MAINNET_RPC.rstrip("/")
            )

        if "api-key=" in base:
            self.url = base
        else:
            separator = "&" if "?" in base else "?"
            self.url = f"{base}{separator}api-key={self.api_key}"

    def rpc(self, method: str, params: Any) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": f"helius-{uuid.uuid4().hex[:10]}",
            "method": method,
            "params": params,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()

                if "error" in body:
                    raise HeliusAPIError(str(body["error"]))
                return body.get("result")
            except (requests.RequestException, ValueError, HeliusAPIError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = self.backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)

        raise HeliusAPIError(
            f"RPC {method} failed after {self.max_retries} attempts: {last_error}"
        )

    # -------------------- DAS methods --------------------
    def get_asset(
        self, asset_id: str, display_options: Optional[Dict[str, Any]] = None
    ) -> Any:
        params: Dict[str, Any] = {"id": asset_id}
        if display_options:
            params["displayOptions"] = display_options
        return self.rpc("getAsset", params)

    def get_asset_batch(self, ids: list[str]) -> Any:
        return self.rpc("getAssetBatch", {"ids": ids})

    def get_asset_proof(self, asset_id: str) -> Any:
        return self.rpc("getAssetProof", {"id": asset_id})

    def get_assets_by_owner(
        self,
        owner_address: str,
        page: int = 1,
        limit: int = 100,
        token_type: Optional[str] = None,
        display_options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "ownerAddress": owner_address,
            "page": page,
            "limit": limit,
        }
        if token_type:
            params["tokenType"] = token_type
        if display_options:
            params["displayOptions"] = display_options
        return self.rpc("getAssetsByOwner", params)

    def search_assets(self, params: Dict[str, Any]) -> Any:
        return self.rpc("searchAssets", params)

    def get_signatures_for_asset(
        self, asset_id: str, page: int = 1, limit: int = 100
    ) -> Any:
        return self.rpc(
            "getSignaturesForAsset",
            {
                "id": asset_id,
                "page": page,
                "limit": limit,
            },
        )

    # -------------------- SPL / RPC methods --------------------
    def get_token_account_balance(self, token_account: str) -> Any:
        return self.rpc("getTokenAccountBalance", [token_account])

    def get_token_accounts_by_owner(
        self,
        owner_address: str,
        program_id: Optional[str] = TOKEN_PROGRAM_ID,
        mint: Optional[str] = None,
        encoding: str = "jsonParsed",
    ) -> Any:
        if mint:
            filter_obj: Dict[str, Any] = {"mint": mint}
        elif program_id:
            filter_obj = {"programId": program_id}
        else:
            raise ValueError("Either program_id or mint must be provided")

        return self.rpc(
            "getTokenAccountsByOwner",
            [owner_address, filter_obj, {"encoding": encoding}],
        )

    def get_token_supply(self, mint: str) -> Any:
        return self.rpc("getTokenSupply", [mint])

    def get_token_largest_accounts(self, mint: str) -> Any:
        return self.rpc("getTokenLargestAccounts", [mint])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helius Solana DAS + SPL CLI")
    parser.add_argument(
        "--api-key",
        default=os.getenv("HELIUS_API_KEY"),
        help="Helius API key (or set HELIUS_API_KEY)",
    )
    parser.add_argument(
        "--network",
        choices=["mainnet", "devnet"],
        default=os.getenv("HELIUS_NETWORK", "mainnet"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HELIUS_RPC_URL"),
        help="Optional custom RPC URL (with or without api-key query)",
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("HELIUS_TIMEOUT", "20"))
    )
    parser.add_argument(
        "--retries", type=int, default=int(os.getenv("HELIUS_RETRIES", "3"))
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_get_asset = sub.add_parser("get-asset", help="DAS getAsset")
    p_get_asset.add_argument("id", help="Asset mint/address")
    p_get_asset.add_argument(
        "--show-fungible", action="store_true", help="Include fungible token fields"
    )

    p_get_asset_batch = sub.add_parser("get-asset-batch", help="DAS getAssetBatch")
    p_get_asset_batch.add_argument("ids", nargs="+", help="Asset IDs")

    p_asset_proof = sub.add_parser("asset-proof", help="DAS getAssetProof")
    p_asset_proof.add_argument("id", help="Asset ID")

    p_owner = sub.add_parser("owner-assets", help="DAS getAssetsByOwner")
    p_owner.add_argument("owner", help="Owner wallet address")
    p_owner.add_argument("--page", type=int, default=1)
    p_owner.add_argument("--limit", type=int, default=100)
    p_owner.add_argument(
        "--token-type",
        choices=["fungible", "nonFungible", "regularNft", "compressedNft", "all"],
        default=None,
    )
    p_owner.add_argument("--show-fungible", action="store_true")
    p_owner.add_argument("--show-native-balance", action="store_true")
    p_owner.add_argument("--show-inscription", action="store_true")

    p_search = sub.add_parser("search", help="DAS searchAssets (raw params JSON)")
    p_search.add_argument(
        "--params", required=True, help="JSON object for searchAssets params"
    )

    p_asset_sigs = sub.add_parser("asset-signatures", help="DAS getSignaturesForAsset")
    p_asset_sigs.add_argument("id", help="Asset ID")
    p_asset_sigs.add_argument("--page", type=int, default=1)
    p_asset_sigs.add_argument("--limit", type=int, default=100)

    p_balance = sub.add_parser("token-balance", help="RPC getTokenAccountBalance")
    p_balance.add_argument("token_account", help="Token account address")

    p_token_accounts = sub.add_parser(
        "token-accounts", help="RPC getTokenAccountsByOwner"
    )
    p_token_accounts.add_argument("owner", help="Owner wallet address")
    p_token_accounts.add_argument("--mint", default=None, help="Optional mint filter")
    p_token_accounts.add_argument(
        "--program-id",
        default=TOKEN_PROGRAM_ID,
        help="Token program ID when mint is not provided",
    )
    p_token_accounts.add_argument(
        "--encoding", default="jsonParsed", choices=["jsonParsed", "base64"]
    )

    p_supply = sub.add_parser("token-supply", help="RPC getTokenSupply")
    p_supply.add_argument("mint", help="Token mint address")

    p_largest = sub.add_parser("token-largest", help="RPC getTokenLargestAccounts")
    p_largest.add_argument("mint", help="Token mint address")

    p_rpc = sub.add_parser("rpc", help="Generic RPC passthrough")
    p_rpc.add_argument("method", help="RPC method name")
    p_rpc.add_argument("--params", default="{}", help="JSON params (object or array)")

    return parser


def _display_options_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    opts: Dict[str, Any] = {}
    if getattr(args, "show_fungible", False):
        # Helius docs show both variants across pages; keep compatibility.
        opts["showFungible"] = True
        opts["showFungibleTokens"] = True
    if getattr(args, "show_native_balance", False):
        opts["showNativeBalance"] = True
    if getattr(args, "show_inscription", False):
        opts["showInscription"] = True
    return opts or None


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        client = HeliusClient(
            api_key=args.api_key,
            network=args.network,
            endpoint=args.endpoint,
            timeout=args.timeout,
            max_retries=args.retries,
        )

        if args.command == "get-asset":
            result = client.get_asset(
                args.id, display_options=_display_options_from_args(args)
            )
        elif args.command == "get-asset-batch":
            result = client.get_asset_batch(args.ids)
        elif args.command == "asset-proof":
            result = client.get_asset_proof(args.id)
        elif args.command == "owner-assets":
            result = client.get_assets_by_owner(
                args.owner,
                page=args.page,
                limit=args.limit,
                token_type=args.token_type,
                display_options=_display_options_from_args(args),
            )
        elif args.command == "search":
            result = client.search_assets(_safe_json_loads(args.params))
        elif args.command == "asset-signatures":
            result = client.get_signatures_for_asset(
                args.id, page=args.page, limit=args.limit
            )
        elif args.command == "token-balance":
            result = client.get_token_account_balance(args.token_account)
        elif args.command == "token-accounts":
            result = client.get_token_accounts_by_owner(
                args.owner,
                program_id=args.program_id,
                mint=args.mint,
                encoding=args.encoding,
            )
        elif args.command == "token-supply":
            result = client.get_token_supply(args.mint)
        elif args.command == "token-largest":
            result = client.get_token_largest_accounts(args.mint)
        elif args.command == "rpc":
            raw_params = json.loads(args.params)
            result = client.rpc(args.method, raw_params)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2

        _print_json(result)
        return 0

    except Exception as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

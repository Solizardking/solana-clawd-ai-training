#!/usr/bin/env python3
"""Train/extend the Nemotron tokenizer with Solana trading + PUMP-MCP tool tokens.

Tokenizer files only from nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.
Does not download 30B NVFP4 weights.

Usage:
  python3 scripts/train_solana_trading_tokenizer.py
  python3 scripts/train_solana_trading_tokenizer.py --push
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SRC = ROOT / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(TOKENIZER_SRC) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_SRC))

from tokenizer.corpus import iter_secret_free_corpus  # noqa: E402
from tokenizer.hf_trading import (  # noqa: E402
    apply_user_chat_template,
    decode_ids,
    encode_text,
    is_atomic_tool_token,
    push_tokenizer,
    save_tokenizer,
    tool_token_id,
    train_or_extend_tokenizer,
)
from tokenizer.trading_tokens import (  # noqa: E402
    DEFAULT_HUB_REPO,
    NEMOTRON_TOKENIZER_ID,
    PUMP_MCP_TOOLS,
    SOL_GPT_TOOLS,
)

DEFAULT_PUMP_MCP = Path(
    "/Users/8bit/Downloads/solgpt---nl-trading-desk (5)/PUMP-MCP-main"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "solana-trading-tokenizer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pump-mcp", type=Path, default=DEFAULT_PUMP_MCP)
    parser.add_argument("--pretrained", default=NEMOTRON_TOKENIZER_ID)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--train-new", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_HUB_REPO)
    parser.add_argument("--private", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pump_root = args.pump_mcp if args.pump_mcp.exists() else None
    corpus = list(
        iter_secret_free_corpus(
            pump_mcp_root=pump_root,
            repo_root=ROOT,
            max_docs=args.max_docs,
        )
    )
    tokenizer = train_or_extend_tokenizer(
        output_dir=args.output,
        corpus=corpus,
        pretrained=args.pretrained,
        train_new=args.train_new,
    )
    checks = {
        "pretrained": args.pretrained,
        "output": str(args.output),
        "corpus_docs": len(corpus),
        "atomic": {},
        "roundtrip": {},
        "chat_template": None,
        "hub_repo": None,
    }
    sample_names = (
        "prepare_user_swap",
        "get_price",
        "list_phoenix_markets",
        *PUMP_MCP_TOOLS[:3],
        SOL_GPT_TOOLS[0],
    )
    for name in sample_names:
        atomic = is_atomic_tool_token(tokenizer, name)
        encoded = encode_text(tokenizer, name, add_special_tokens=False)
        decoded = decode_ids(tokenizer, encoded)
        checks["atomic"][name] = {
            "ok": atomic,
            "id": tool_token_id(tokenizer, name),
            "n_ids": len(encoded),
        }
        checks["roundtrip"][name] = decoded == name
        if not atomic or decoded != name:
            print(f"FAIL atomic/roundtrip {name!r} ids={encoded} decoded={decoded!r}", file=sys.stderr)
            return 1
    mint = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump"
    mint_ids = encode_text(tokenizer, mint, add_special_tokens=False)
    if decode_ids(tokenizer, mint_ids) != mint or not mint_ids:
        print("FAIL mint round-trip", file=sys.stderr)
        return 1
    rendered = apply_user_chat_template(tokenizer, "Who are you?")
    checks["chat_template"] = bool(rendered)
    if args.push:
        repo = push_tokenizer(
            tokenizer,
            args.repo_id,
            token=os.environ.get("HF_TOKEN"),
            private=args.private,
        )
        checks["hub_repo"] = repo
    save_tokenizer(tokenizer, args.output)
    print(json.dumps(checks, indent=2))
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

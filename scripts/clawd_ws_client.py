#!/usr/bin/env python3
"""Health-check and tokenize live Pump.fun frames from clawd-ws.

Canonical tape: wss://clawd-ws.fly.dev/ws (no query string, no subscribe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SRC = ROOT / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(TOKENIZER_SRC) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_SRC))

from tokenizer.clawd_ws import (  # noqa: E402
    DEFAULT_HTTP,
    DEFAULT_WS,
    fetch_health,
    frame_to_text,
    recv_pump_frames,
)
from tokenizer.hf_trading import encode_text, load_nemotron_tokenizer  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "outputs" / "solana-trading-tokenizer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-url", default=DEFAULT_HTTP + "/health")
    parser.add_argument("--ws-url", default=DEFAULT_WS)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--health-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    health = fetch_health(args.health_url, timeout=args.timeout)
    report: dict = {"health": health, "frames": []}
    if health.get("status") != "ok":
        print(json.dumps(report, indent=2))
        print("FAIL health status is not ok", file=sys.stderr)
        return 1
    if args.health_only:
        print(json.dumps(report, indent=2))
        return 0
    frames = recv_pump_frames(args.ws_url, timeout=args.timeout, max_frames=args.frames)
    tokenizer = None
    if args.tokenizer.exists():
        tokenizer = load_nemotron_tokenizer(str(args.tokenizer))
    for frame in frames:
        text = frame_to_text(frame)
        entry = {"type": frame.get("type"), "text": text[:500], "n_ids": None}
        if tokenizer is not None:
            ids = encode_text(tokenizer, text, add_special_tokens=False)
            entry["n_ids"] = len(ids)
            if not ids:
                print(json.dumps(report, indent=2))
                print("FAIL empty input_ids for live frame", file=sys.stderr)
                return 1
        report["frames"].append(entry)
    print(json.dumps(report, indent=2))
    if not frames:
        print("FAIL no pump frames received", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

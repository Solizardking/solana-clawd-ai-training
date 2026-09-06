#!/usr/bin/env python3
"""Runtime bridge that injects live clawd-ws data into a model at inference time.

This is the *only* layer that can serve current data. Weights cannot hold live
prices -- they are stale the moment training ends. The model is taught (by
scripts/build_live_data_dataset.py) to emit one of the tool calls declared here
instead of guessing; this module executes that call against
https://clawd-ws.fly.dev and returns a frame to drop back into context.

Tool schemas are emitted in both OpenAI/Ollama ("function") and Anthropic
("input_schema") shapes so the same definitions drive any runtime.

CLI:
    python3 scripts/clawd_ws_tools.py --list
    python3 scripts/clawd_ws_tools.py --schema openai
    python3 scripts/clawd_ws_tools.py --call get_pump_stream_status
    python3 scripts/clawd_ws_tools.py --call get_recent_token_launches \
        --args '{"limit": 3, "require_socials": true}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tokenizer.clawd_ws import (  # noqa: E402
    DEFAULT_HTTP,
    DEFAULT_WS,
    fetch_health,
    recv_pump_frames,
)

# Fields that are URLs or free text and blow up context without adding signal.
_VERBOSE_FIELDS = ("metadataUri", "imageUri", "description")


TOOLS: dict[str, dict[str, Any]] = {
    "get_pump_stream_status": {
        "description": (
            "Get the live health and connection status of the clawd-ws Pump.fun tape "
            "(total launches seen, connected clients, which upstream feeds are up). "
            "Use this to confirm the stream is live before reporting on it, or when "
            "asked how many tokens have launched."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_recent_token_launches": {
        "description": (
            "Read the newest Pump.fun token launches off the live clawd-ws tape. "
            "Returns real frames with mint, creator, symbol, marketCapSol, and social "
            "links. Use this whenever asked what is launching now, or to inspect a "
            "current launch. Never answer from memory -- launches are seconds old."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many launch frames to return (1-25).",
                    "default": 5,
                },
                "require_socials": {
                    "type": "boolean",
                    "description": "Only return launches that declare a website, twitter, or telegram.",
                    "default": False,
                },
                "min_market_cap_sol": {
                    "type": "number",
                    "description": "Drop launches below this marketCapSol.",
                    "default": 0,
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to hold the socket open waiting for frames.",
                    "default": 30,
                },
                "concise": {
                    "type": "boolean",
                    "description": "Strip metadataUri/imageUri/description to save context.",
                    "default": True,
                },
            },
            "required": [],
        },
    },
}


def _concise(frame: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in frame.items() if k not in _VERBOSE_FIELDS}


def get_pump_stream_status(timeout: float = 15.0) -> dict[str, Any]:
    health = fetch_health(DEFAULT_HTTP + "/health", timeout=timeout)
    # The health payload embeds RPC URLs with api-key query strings. Never let
    # those reach a model context or a log.
    return {
        "source": DEFAULT_HTTP + "/health",
        "status": health.get("status"),
        "total_launches": health.get("totalLaunches"),
        "clients": health.get("clients"),
        "uptime_seconds": health.get("uptime"),
        "feeds": {
            k: health.get(k)
            for k in ("solana", "birdeye", "tracker", "analyzer", "jupiter", "dflow", "helius")
        },
    }


def get_recent_token_launches(
    limit: int = 5,
    require_socials: bool = False,
    min_market_cap_sol: float = 0.0,
    timeout: float = 30.0,
    concise: bool = True,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 25))
    # Over-read so filters still have candidates to work with.
    budget = limit * 6 if (require_socials or min_market_cap_sol > 0) else limit + 4
    frames = recv_pump_frames(DEFAULT_WS, timeout=timeout, max_frames=budget)

    launches: list[dict[str, Any]] = []
    for frame in frames:
        if frame.get("type") != "token-launch":
            continue
        if min_market_cap_sol and (frame.get("marketCapSol") or 0) < min_market_cap_sol:
            continue
        if require_socials and not any(frame.get(k) for k in ("website", "twitter", "telegram")):
            continue
        launches.append(_concise(frame) if concise else frame)
        if len(launches) >= limit:
            break

    return {
        "source": DEFAULT_WS,
        "returned": len(launches),
        "frames_seen": len(frames),
        "filters": {
            "require_socials": require_socials,
            "min_market_cap_sol": min_market_cap_sol,
        },
        "launches": launches,
    }


EXECUTORS = {
    "get_pump_stream_status": get_pump_stream_status,
    "get_recent_token_launches": get_recent_token_launches,
}


def openai_schema() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
        for name, spec in TOOLS.items()
    ]


def anthropic_schema() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": spec["description"], "input_schema": spec["parameters"]}
        for name, spec in TOOLS.items()
    ]


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in EXECUTORS:
        return {"error": f"unknown tool {name!r}", "available": sorted(EXECUTORS)}
    try:
        return EXECUTORS[name](**(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List tool names and descriptions")
    p.add_argument("--schema", choices=["openai", "anthropic"], help="Print tool schemas")
    p.add_argument("--call", metavar="TOOL", help="Execute a tool against the live stream")
    p.add_argument("--args", default="{}", help="JSON arguments for --call")
    args = p.parse_args()

    if args.list:
        for name, spec in TOOLS.items():
            print(f"{name}\n    {spec['description']}\n")
        return 0
    if args.schema:
        print(json.dumps(openai_schema() if args.schema == "openai" else anthropic_schema(), indent=2))
        return 0
    if args.call:
        result = call_tool(args.call, json.loads(args.args))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if "error" in result else 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

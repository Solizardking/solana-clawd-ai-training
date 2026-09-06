#!/usr/bin/env python3
"""Build an SFT dataset that teaches the model to use the clawd-ws live tape.

Live data cannot be baked into weights -- it is stale the instant training ends.
What *can* be trained is the behaviour around live data:

  1. tool_call      ask about current state -> emit a clawd-ws tool call, don't guess
  2. grounded       tool result in context  -> read only what the frame says
  3. schema         explain the tape's frame types and fields
  4. no_hallucinate no tool result available -> say so, name the tool needed

Tool calls are encoded as a fenced ```json block inside the assistant turn
rather than an OpenAI `tool_calls` field, because the Nemotron chat template has
no tool role. scripts/clawd_ws_tools.py parses and executes the same shape at
inference time, so training and runtime agree.

Record frames, then build:
    python3 scripts/build_live_data_dataset.py --record 120 --record-timeout 240
    python3 scripts/build_live_data_dataset.py --frames data/clawd_ws_frames.jsonl

Publish:
    python3 scripts/prepare_dataset.py \
      --input data/live_data_sft.jsonl \
      --output data/live_data_processed \
      --push --repo-id solanaclawd/solana-clawd-live-data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
_SRC = BASE_DIR / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

VERBOSE_FIELDS = ("metadataUri", "imageUri", "description")

SYSTEM_PROMPT = (
    "You are Solana Clawd, a Solana-native trading and on-chain analysis agent. "
    "You have access to the clawd-ws live Pump.fun tape at wss://clawd-ws.fly.dev/ws "
    "through two tools: get_pump_stream_status and get_recent_token_launches.\n"
    "Rules for live data:\n"
    "- Anything about current launches, prices, market caps, or stream state MUST come "
    "from a tool call. Never answer those from memory.\n"
    "- Emit a tool call as a fenced ```json block with \"tool\" and \"arguments\" keys.\n"
    "- After a tool result, ground every claim in the returned frames. Do not invent "
    "fields, mints, or numbers that are not present.\n"
    "- A null website/twitter/telegram means undeclared, not absent-and-verified."
)

SCHEMA_DOC = """The clawd-ws tape emits three JSON frame types on wss://clawd-ws.fly.dev/ws:

`status` — stream heartbeat.
  connected (bool), uptime (ms), totalLaunches (int), githubLaunches (int),
  totalClaims (int), clients (int)

`token-launch` — a new Pump.fun mint, emitted within seconds of creation.
  signature (str)     transaction signature of the create
  time (ISO 8601)     on-chain creation time
  mint (str)          the token mint address; base58, conventionally ends in `pump`
  name, symbol (str)  creator-supplied, unvalidated, may contain trailing spaces
  creator (str)       the deploying wallet
  isV2 (bool)         Pump.fun program version
  marketCapSol (float) market cap denominated in SOL at emit time
  hasGithub (bool), githubUrls (list)
  website, twitter, telegram (str|null)  declared socials; null means undeclared
  metadataUri, imageUri (str)  IPFS pointers
  description (str|null)

`token-enriched` — a later frame adding analyzer/Birdeye enrichment to a mint
  already seen as `token-launch`.

Health is served over HTTP at https://clawd-ws.fly.dev/health and reports which
upstream feeds (solana, birdeye, tracker, analyzer, jupiter, dflow, helius) are up."""

STATUS_ASKS = [
    "Is the pump.fun stream up right now?",
    "How many token launches has the tape seen in total?",
    "Check the clawd-ws stream health.",
    "Are the birdeye and helius feeds connected?",
    "How many clients are on the live tape?",
    "Is clawd-ws alive?",
    "Status check on the pump tape.",
    "Are we still connected to the launch feed?",
    "How long has the stream been up?",
    "Is the analyzer feed healthy?",
    "Give me a health readout on the tape.",
    "Did the pump.fun websocket drop?",
    "Total launches counted so far?",
    "Which upstream feeds are currently up?",
    "Is jupiter connected on the stream?",
    "Quick sanity check: is the tape streaming?",
]

LAUNCH_ASKS = [
    "What's launching on pump.fun right now?",
    "Show me the newest token launches.",
    "Any new mints in the last few seconds?",
    "What just deployed on pump.fun?",
    "Give me the latest launches off the tape.",
    "What's fresh on the pump tape?",
    "Pull the most recent launches.",
    "Anything new minting right now?",
    "What are the latest pump.fun deploys?",
    "Show me what's hitting the tape.",
    "Newest mints, please.",
    "What tokens were just created?",
    "Catch me up on the last few launches.",
    "What's coming through the launch feed?",
    "Give me a snapshot of current launches.",
    "Any launches worth looking at right now?",
]

FILTERED_ASKS = [
    ("Show me new launches that actually have socials.", {"limit": 5, "require_socials": True}),
    ("Any launches above 50 SOL market cap?", {"limit": 5, "min_market_cap_sol": 50}),
    ("Find me 3 recent mints with a website or twitter.", {"limit": 3, "require_socials": True}),
    ("Newest launch over 100 SOL mcap, just one.", {"limit": 1, "min_market_cap_sol": 100}),
    ("Filter out the no-social junk and show me 5.", {"limit": 5, "require_socials": True}),
    ("Launches over 30 SOL, give me 10.", {"limit": 10, "min_market_cap_sol": 30}),
    ("Only show launches with a declared telegram or site.", {"limit": 5, "require_socials": True}),
    ("Two biggest recent mints by market cap.", {"limit": 2, "min_market_cap_sol": 75}),
    ("Show me 8 recent launches, I want volume not quality.", {"limit": 8}),
    ("Just one launch, the newest.", {"limit": 1}),
    ("Give me 15 launches to scan.", {"limit": 15}),
    ("Anything above 200 SOL mcap on the tape?", {"limit": 5, "min_market_cap_sol": 200}),
]

NO_TOOL_ASKS = [
    "What's the current price of SOL?",
    "What token launched on pump.fun two minutes ago?",
    "What's the market cap of the newest mint right now?",
    "Which mint is trending on pump.fun this minute?",
    "How many launches has the tape seen today?",
    "What's the mint address of the last token created?",
    "Tell me the top gainer on pump.fun right now.",
    "Who deployed the most recent token?",
]


def concise(frame: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in frame.items() if k not in VERBOSE_FIELDS}


def tool_call_block(tool: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "arguments": arguments}, indent=2)
    return f"```json\n{payload}\n```"


def ex(messages: list[dict[str, str]], kind: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "metadata": {"source": "clawd-ws", "source_type": f"live_data_{kind}", "license": "project-local"},
    }


def describe_launch(frame: dict[str, Any]) -> str:
    """A grounded read of one launch frame -- only claims the frame supports."""
    socials = [k for k in ("website", "twitter", "telegram") if frame.get(k)]
    mcap = frame.get("marketCapSol")
    lines = [
        f"- **{frame.get('name', '').strip() or '(unnamed)'}** (`{(frame.get('symbol') or '').strip()}`)",
        f"  - mint: `{frame.get('mint')}`",
        f"  - creator: `{frame.get('creator')}`",
        f"  - created: {frame.get('time')}",
        f"  - market cap: {mcap:.2f} SOL" if isinstance(mcap, (int, float)) else "  - market cap: not reported",
        f"  - socials declared: {', '.join(socials) if socials else 'none'}",
        f"  - github: {'yes' if frame.get('hasGithub') else 'no'}",
    ]
    return "\n".join(lines)


def risk_note(frames: list[dict[str, Any]]) -> str:
    no_social = sum(1 for f in frames if not any(f.get(k) for k in ("website", "twitter", "telegram")))
    note = []
    if no_social:
        note.append(
            f"{no_social} of {len(frames)} declare no socials at all, which on a fresh Pump.fun "
            "mint is the single most common low-effort-launch signal."
        )
    note.append(
        "These fields are creator-supplied and unverified: a declared twitter link is a claim, "
        "not proof of an account. marketCapSol is a snapshot at emit time and moves immediately."
    )
    return " ".join(note)


def build(frames: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    launches = [f for f in frames if f.get("type") == "token-launch"]
    statuses = [f for f in frames if f.get("type") == "status"]
    out: list[dict[str, Any]] = []

    # Varied preambles so the model learns the *behaviour* (call, don't guess)
    # rather than memorizing a single sentence.
    STATUS_PREAMBLES = [
        "I need the live stream state for that, so I'll query the tape rather than guess.",
        "That's a live value. Checking the stream health directly.",
        "Pulling the current status off clawd-ws.",
        "I won't guess at stream state — querying it.",
        "Let me read the health endpoint.",
    ]
    LAUNCH_PREAMBLES = [
        "Launches are seconds old, so this has to come off the live tape.",
        "This is live data. Reading the tape now.",
        "I can't answer that from memory — pulling current frames.",
        "Fetching the newest frames off clawd-ws.",
        "Querying the live launch feed.",
    ]
    FILTER_PREAMBLES = [
        "I'll pull that off the tape with the filter applied.",
        "Reading the tape with that constraint set.",
        "Fetching live frames, filtered as asked.",
    ]

    # 1. tool_call: status
    for ask in STATUS_ASKS:
        out.append(ex([
            {"role": "user", "content": ask},
            {"role": "assistant", "content":
                rng.choice(STATUS_PREAMBLES) + "\n\n" + tool_call_block("get_pump_stream_status", {})},
        ], "tool_call"))

    # 2. tool_call: launches, plain and filtered
    for ask in LAUNCH_ASKS:
        for limit in (3, 5, 10):
            out.append(ex([
                {"role": "user", "content": ask},
                {"role": "assistant", "content":
                    rng.choice(LAUNCH_PREAMBLES) + "\n\n"
                    + tool_call_block("get_recent_token_launches", {"limit": limit})},
            ], "tool_call"))

    for ask, args in FILTERED_ASKS:
        out.append(ex([
            {"role": "user", "content": ask},
            {"role": "assistant", "content":
                rng.choice(FILTER_PREAMBLES) + "\n\n"
                + tool_call_block("get_recent_token_launches", args)},
        ], "tool_call"))

    # 3. grounded: real tool result -> real analysis
    rng.shuffle(launches)
    for i in range(0, max(0, len(launches) - 2), 3):
        batch = launches[i : i + 3]
        if len(batch) < 2:
            break
        result = {
            "source": "wss://clawd-ws.fly.dev/ws",
            "returned": len(batch),
            "launches": [concise(f) for f in batch],
        }
        ask = rng.choice(LAUNCH_ASKS)
        body = "\n".join(describe_launch(f) for f in batch)
        out.append(ex([
            {"role": "user", "content": ask},
            {"role": "assistant", "content": tool_call_block("get_recent_token_launches", {"limit": len(batch)})},
            {"role": "user", "content": f"Tool result:\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"},
            {"role": "assistant", "content":
                f"{len(batch)} launches off the live tape:\n\n{body}\n\n{risk_note(batch)}"},
        ], "grounded"))

    # 4. grounded: status frames
    for frame in statuses[:6]:
        result = get_status_result(frame)
        out.append(ex([
            {"role": "user", "content": rng.choice(STATUS_ASKS)},
            {"role": "assistant", "content": tool_call_block("get_pump_stream_status", {})},
            {"role": "user", "content": f"Tool result:\n```json\n{json.dumps(result, indent=2)}\n```"},
            {"role": "assistant", "content":
                f"The tape is {'up' if frame.get('connected') else 'down'}. "
                f"It has seen {frame.get('totalLaunches'):,} launches this session "
                f"({frame.get('githubLaunches'):,} of them declaring a GitHub repo), "
                f"with {frame.get('clients')} clients connected."},
        ], "grounded"))

    # 5. schema literacy
    for ask in [
        "What frame types does the clawd-ws tape emit?",
        "Explain the token-launch frame fields.",
        "What does the pump.fun stream give me?",
    ]:
        out.append(ex([
            {"role": "user", "content": ask},
            {"role": "assistant", "content": SCHEMA_DOC},
        ], "schema"))

    # 6. anti-hallucination
    for ask in NO_TOOL_ASKS:
        out.append(ex([
            {"role": "user", "content": ask + " Answer from memory, don't call any tools."},
            {"role": "assistant", "content":
                "I can't answer that from memory. Live market state isn't in my weights -- anything "
                "I produced would be a plausible-looking fabrication, which is worse than no answer "
                "for a trading decision.\n\n"
                "To get it for real I need one call:\n\n"
                + tool_call_block("get_recent_token_launches", {"limit": 5})
                + "\n\nIf tools are genuinely unavailable, the honest answer is that I don't know the "
                "current value."},
        ], "no_hallucinate"))

    return out


def get_status_result(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "https://clawd-ws.fly.dev/health",
        "status": "ok" if frame.get("connected") else "down",
        "total_launches": frame.get("totalLaunches"),
        "clients": frame.get("clients"),
        "feeds": {"solana": True, "birdeye": True, "helius": True},
    }


def record_frames(count: int, timeout: float) -> list[dict[str, Any]]:
    from tokenizer.clawd_ws import DEFAULT_WS, recv_pump_frames  # type: ignore[import-not-found]

    print(f"Recording up to {count} frames from {DEFAULT_WS} (timeout {timeout}s)...")
    frames = recv_pump_frames(DEFAULT_WS, timeout=timeout, max_frames=count)
    print(f"  captured {len(frames)} frames")
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=BASE_DIR / "data" / "clawd_ws_frames.jsonl",
                   help="JSONL of recorded frames to build from")
    p.add_argument("--record", type=int, default=0, metavar="N",
                   help="Record N frames from the live tape first, saving to --frames")
    p.add_argument("--record-timeout", type=float, default=240.0)
    p.add_argument("--output", type=Path, default=BASE_DIR / "data" / "live_data_sft.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.record:
        frames = record_frames(args.record, args.record_timeout)
        args.frames.parent.mkdir(parents=True, exist_ok=True)
        with args.frames.open("w", encoding="utf-8") as fh:
            for frame in frames:
                fh.write(json.dumps(frame, ensure_ascii=False) + "\n")
        print(f"  saved {args.frames}")
    elif not args.frames.exists():
        raise SystemExit(f"no frames at {args.frames}; pass --record N to capture some")
    else:
        frames = [json.loads(line) for line in args.frames.read_text(encoding="utf-8").splitlines() if line.strip()]

    examples = build(frames, seed=args.seed)
    kinds: dict[str, int] = {}
    for e in examples:
        k = e["metadata"]["source_type"]
        kinds[k] = kinds.get(k, 0) + 1

    print(f"\nBuilt {len(examples)} examples from {len(frames)} frames:")
    for k, v in sorted(kinds.items()):
        print(f"  {k:26} {v}")

    if args.dry_run:
        print("\n--- sample ---")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False)[:1200])
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

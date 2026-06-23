#!/usr/bin/env python3
"""
Convert armand0e/claude-fable-5-claude-code agent traces to training JSONL.

The dataset contains full Claude Code sessions with tool calls, system prompts,
and tool results. We extract clean user→assistant dialogue pairs, stripping
system messages and tool results.

Two output formats:
  text   (default) — GPT-2-style plain text, no chat template:
    {"text": "Human: ...\n\nAssistant: ..."}

  chatml — Qwen / ChatML chat template format:
    {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}

Usage:
  cd ai-training/
  python3 scripts/prepare_clawd_code_dataset.py                          # text, gpt-2
  python3 scripts/prepare_clawd_code_dataset.py --format chatml          # chatml, qwen
  python3 scripts/prepare_clawd_code_dataset.py --output data/my.jsonl --epochs 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DATASET_REPO = "armand0e/claude-fable-5-claude-code"  # upstream HF repo name — not ours to rename
DEFAULT_OUTPUT_TEXT   = Path("data/clawd_code_deepsol_sft.jsonl")    # GPT-2 plain text
DEFAULT_OUTPUT_CHATML = Path("data/clawd_code_qwen35_sft.jsonl")     # Qwen / ChatML
MAX_TOKENS_APPROX = 900   # stay well under GPT-2's 1024 hard limit
MAX_TOKENS_CHATML = 2048  # Qwen3.5 supports 262K; 2048 captures full pairs safely
CHARS_PER_TOKEN   = 3.5   # conservative chars-per-token estimate


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"


def _extract_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """
    Walk the message list and collect (user_content, assistant_content) pairs.
    Skips system messages and tool results. Skips empty assistant turns.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            # content blocks — join text parts
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        if role == "user":
            pending_user = content
        elif role == "assistant" and content and pending_user:
            pairs.append((pending_user, content))
            pending_user = None
        # skip system / tool / empty assistant

    return pairs


def _format_pair_text(user: str, assistant: str, max_chars: int) -> str:
    """GPT-2 plain text format (no chat template)."""
    half = max_chars // 2
    return f"Human: {_trim(user, half)}\n\nAssistant: {_trim(assistant, half)}"


def _format_pair_chatml(user: str, assistant: str, max_chars: int) -> str:
    """Qwen/ChatML format — SFTTrainer recognises <|im_start|>assistant for loss masking."""
    half = max_chars // 2
    u = _trim(user, half)
    a = _trim(assistant, half)
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"


def build_dataset(
    output_path: Path,
    repeat_epochs: int = 1,
    fmt: str = "text",
    verbose: bool = True,
) -> int:
    from datasets import load_dataset

    if verbose:
        print(f"[prepare] loading {DATASET_REPO} ...")
    ds = load_dataset(DATASET_REPO, split="train")

    if fmt == "chatml":
        max_chars = int(MAX_TOKENS_CHATML * CHARS_PER_TOKEN)
        format_fn = _format_pair_chatml
    else:
        max_chars = int(MAX_TOKENS_APPROX * CHARS_PER_TOKEN)
        format_fn = _format_pair_text

    records: list[dict] = []

    for row in ds:
        messages = row.get("messages", [])
        pairs = _extract_pairs(messages)

        for user, assistant in pairs:
            text = format_fn(user, assistant, max_chars)
            if len(text.strip()) < 40:
                continue
            records.append({"text": text})

        # Fallback: all-tool session with no extractable pairs
        if not pairs and row.get("prompt", "").strip():
            prompt_text = _trim(row["prompt"].strip(), max_chars)
            if fmt == "chatml":
                records.append({"text": f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"})
            else:
                records.append({"text": f"Human: {prompt_text}\n\nAssistant:"})

    # Repeat for tiny datasets so the trainer sees more steps
    base = records[:]
    for _ in range(repeat_epochs - 1):
        records.extend(base)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[prepare] wrote {len(records)} records → {output_path}")
        print(f"          ({len(base)} unique pairs × {repeat_epochs} repeat epochs)")

    return len(records)


GLINT_REPO = "Glint-Research/Fable-5-traces"


def _glint_extract_text(content: list) -> str:
    """Extract text from a Glint content block list, skipping pure thinking blocks."""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            t = (block.get("text") or "").strip()
            if t:
                parts.append(t)
        elif btype == "thinking":
            t = (block.get("text") or "").strip()
            if t:
                parts.append(f"<think>\n{t}\n</think>")
        elif btype == "toolCall":
            name = block.get("name", "")
            args = block.get("arguments") or {}
            args_str = json.dumps(args, ensure_ascii=False)
            parts.append(f"<tool_call>{{'name': '{name}', 'arguments': {args_str}}}</tool_call>")
    return "\n".join(parts).strip()


def build_glint_dataset(
    repeat_epochs: int = 1,
    fmt: str = "text",
    verbose: bool = True,
) -> list[dict]:
    """Load Glint-Research/Fable-5-traces and return record list.

    The dataset uses HF Agent Traces format with type='message' rows.
    Messages alternate user→assistant; we pair them and format accordingly.
    """
    from datasets import load_dataset

    if verbose:
        print(f"[prepare] loading {GLINT_REPO} ...")
    ds = load_dataset(GLINT_REPO, split="train")

    if fmt == "chatml":
        max_chars = int(MAX_TOKENS_CHATML * CHARS_PER_TOKEN)
    else:
        max_chars = int(MAX_TOKENS_APPROX * CHARS_PER_TOKEN)
    half = max_chars // 2

    # Collect all message rows in order, then pair user→assistant
    message_rows = [r for r in ds if r.get("type") == "message"]
    user_rows    = [r for r in message_rows if r.get("message", {}).get("role") == "user"]
    asst_rows    = [r for r in message_rows if r.get("message", {}).get("role") == "assistant"]

    records: list[dict] = []
    for u_row, a_row in zip(user_rows, asst_rows):
        user_text = _glint_extract_text(u_row["message"].get("content", []))
        asst_text = _glint_extract_text(a_row["message"].get("content", []))

        if not user_text or not asst_text:
            continue

        u = _trim(user_text, half)
        a = _trim(asst_text, half)

        if fmt == "chatml":
            text = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        else:
            text = f"Human: {u}\n\nAssistant: {a}"

        if len(text.strip()) >= 40:
            records.append({"text": text})

    base = records[:]
    for _ in range(repeat_epochs - 1):
        records.extend(base)

    if verbose:
        print(f"[prepare] Glint: {len(base)} unique records × {repeat_epochs} epochs = {len(records)}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Output JSONL path (auto-selected by format)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Repeat each source dataset N times (default: 3)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "chatml"],
        default="text",
        help="text=GPT-2 plain text (default), chatml=Qwen/ChatML format",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["clawd-code", "glint"],
        default=["clawd-code"],
        help="Which source datasets to include (default: clawd-code only)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    fmt = args.format
    verbose = not args.quiet

    # Auto-select output path based on format and dataset mix
    if args.output:
        out = Path(args.output)
    elif "glint" in args.datasets and fmt == "chatml":
        out = Path("data/clawd_fable5_qwen35_sft.jsonl")
    elif fmt == "chatml":
        out = DEFAULT_OUTPUT_CHATML
    else:
        out = DEFAULT_OUTPUT_TEXT

    all_records: list[dict] = []

    if "clawd-code" in args.datasets:
        from datasets import load_dataset as _ld

        if verbose:
            print(f"[prepare] loading {DATASET_REPO} ...")
        ds_cc = _ld(DATASET_REPO, split="train")
        max_chars = int((MAX_TOKENS_CHATML if fmt == "chatml" else MAX_TOKENS_APPROX) * CHARS_PER_TOKEN)
        format_fn = _format_pair_chatml if fmt == "chatml" else _format_pair_text
        cc_records: list[dict] = []
        for row in ds_cc:
            pairs = _extract_pairs(row.get("messages", []))
            for user, assistant in pairs:
                t = format_fn(user, assistant, max_chars)
                if len(t.strip()) >= 40:
                    cc_records.append({"text": t})
            if not pairs and row.get("prompt", "").strip():
                pt = _trim(row["prompt"].strip(), max_chars)
                if fmt == "chatml":
                    cc_records.append({"text": f"<|im_start|>user\n{pt}<|im_end|>\n<|im_start|>assistant\n"})
                else:
                    cc_records.append({"text": f"Human: {pt}\n\nAssistant:"})
        base_cc = cc_records[:]
        for _ in range(args.epochs - 1):
            cc_records.extend(base_cc)
        if verbose:
            print(f"[prepare] clawd-code: {len(base_cc)} unique pairs × {args.epochs} = {len(cc_records)}")
        all_records.extend(cc_records)

    if "glint" in args.datasets:
        glint_records = build_glint_dataset(
            repeat_epochs=args.epochs,
            fmt=fmt,
            verbose=verbose,
        )
        all_records.extend(glint_records)

    import random
    random.shuffle(all_records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[prepare] wrote {len(all_records)} records → {out}")
    print(f"[prepare] done. {len(all_records)} total training records.")


if __name__ == "__main__":
    main()

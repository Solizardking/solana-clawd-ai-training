#!/usr/bin/env python3
"""
Convert Fable traces and local Solana Clawd project context to training JSONL.

The Hugging Face trace datasets contain full Claude Code sessions with tool
calls, system prompts, and tool results. We extract clean user→assistant
dialogue pairs, stripping system messages and tool results.

The local project source mode ingests selected repository files as supervised
context records. It is intentionally conservative: generated dependency folders,
binary artifacts, large model outputs, and secret-like files are skipped.

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
  python3 scripts/prepare_clawd_code_dataset.py --format chatml --datasets clawd-code glint local
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
LOCAL_CONTEXT_CHARS = 6000

DEFAULT_LOCAL_PATHS = [
    "trading_factory",
    "trading_factory/clawd-autoresearch-wiki",
    "trading_factory/cufolio",
    "trading_factory/solana_factory",
    "trading_factory/README.md",
    "wandb",
    ".gitignore",
    "Anchor.toml",
    "Cargo.lock",
    "Cargo.toml",
    "README.md",
    "requirements.txt",
    "solana1_yourgpt.jsonl",
    "STRUCTURE.md",
    "trainingday.jsonl",
]

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "target",
    "venv",
}

SKIP_FILE_SUFFIXES = {
    ".7z",
    ".arrow",
    ".bin",
    ".bmp",
    ".dylib",
    ".faiss",
    ".gif",
    ".gguf",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mp4",
    ".otf",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pyc",
    ".safetensors",
    ".so",
    ".sqlite",
    ".ttf",
    ".wasm",
    ".webp",
    ".zip",
}

TEXT_FILE_SUFFIXES = {
    "",
    ".css",
    ".csv",
    ".env.example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SECRET_PATTERNS = {
    "api_key": re.compile(r"\b(?:sk|hf|nvapi|xox[baprs])[-_][A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)?PRIVATE KEY-----"),
    "solana_keypair": re.compile(r"\[(?:\s*\d{1,3}\s*,){31,}\s*\d{1,3}\s*\]"),
    "wallet_seed": re.compile(r"\b(?:seed phrase|mnemonic|private key)\b", re.IGNORECASE),
}


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


def _format_messages_chatml(messages: list[dict], max_chars: int) -> str:
    parts: list[str] = []
    budget = max_chars
    for msg in messages:
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        content = str(content).strip()
        if not content:
            continue
        content = _trim(content, max(256, budget // 2))
        budget -= len(content)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if budget <= 0:
            break
    return "\n".join(parts)


def _format_messages_text(messages: list[dict], max_chars: int) -> str:
    user = ""
    assistant = ""
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role == "user" and not user:
            user = content
        elif role == "assistant" and not assistant:
            assistant = content
        if user and assistant:
            break
    return _format_pair_text(user or "Use the project context.", assistant or "", max_chars)


def _safe_read_text(path: Path, max_bytes: int) -> str | None:
    if path.stat().st_size > max_bytes and path.suffix.lower() != ".jsonl":
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None
    if any(pattern.search(text) for pattern in SECRET_PATTERNS.values()):
        return None
    return text


def _should_skip_path(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.name == ".DS_Store":
        return True
    suffix = path.suffix.lower()
    if suffix in SKIP_FILE_SUFFIXES:
        return True
    if suffix not in TEXT_FILE_SUFFIXES and path.name not in {".gitignore"}:
        return True
    return False


def _iter_local_files(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            continue
        if path.is_file():
            if not _should_skip_path(path) and path not in seen:
                seen.add(path)
                files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and not _should_skip_path(child):
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
    return files


def _chunk_text(text: str, chunk_chars: int = LOCAL_CONTEXT_CHARS) -> list[str]:
    clean = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_chars)
        if end < len(clean):
            newline = clean.rfind("\n", start, end)
            if newline > start + chunk_chars // 2:
                end = newline
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _format_local_context_record(rel_path: str, chunk: str, fmt: str) -> dict:
    user = f"Use this Solana Clawd project context from `{rel_path}`."
    assistant = f"Project context from `{rel_path}`:\n\n{chunk}"
    if fmt == "chatml":
        text = (
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )
    else:
        text = f"Human: {user}\n\nAssistant: {assistant}"
    return {"text": text}


def _records_from_jsonl(path: Path, fmt: str, max_chars: int, max_records: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = row.get("messages")
            if isinstance(messages, list):
                text = _format_messages_chatml(messages, max_chars) if fmt == "chatml" else _format_messages_text(messages, max_chars)
                if "<|im_start|>assistant" in text or "Assistant:" in text:
                    records.append({"text": text})
            elif isinstance(row.get("text"), str):
                records.append({"text": _trim(row["text"], max_chars)})
    return records


def build_local_project_dataset(
    paths: list[Path],
    fmt: str,
    max_records: int = 2000,
    max_file_bytes: int = 500_000,
    verbose: bool = True,
) -> list[dict]:
    """Build SFT records from selected local repo paths."""
    root = Path.cwd().resolve()
    max_chars = int((MAX_TOKENS_CHATML if fmt == "chatml" else MAX_TOKENS_APPROX) * CHARS_PER_TOKEN)
    records: list[dict] = []

    files = _iter_local_files(paths)
    if verbose:
        print(f"[prepare] local project files discovered: {len(files)}")

    for file_path in files:
        if len(records) >= max_records:
            break
        rel_path = file_path.resolve()
        try:
            rel_label = rel_path.relative_to(root).as_posix()
        except ValueError:
            rel_label = rel_path.as_posix()

        if file_path.suffix.lower() == ".jsonl":
            room = max_records - len(records)
            jsonl_records = _records_from_jsonl(file_path, fmt=fmt, max_chars=max_chars, max_records=room)
            records.extend(jsonl_records)
            continue

        text = _safe_read_text(file_path, max_bytes=max_file_bytes)
        if text is None:
            continue
        for chunk in _chunk_text(text):
            if len(records) >= max_records:
                break
            records.append(_format_local_context_record(rel_label, chunk, fmt))

    if verbose:
        print(f"[prepare] local project: {len(records)} records")
    return records


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


def _glint_extract_text(content) -> str:
    """Extract text from a Glint content block list, skipping pure thinking blocks."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                parts.append(text)
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


def _glint_pair_rows(rows: list[dict], fmt: str, max_chars: int) -> list[dict]:
    """Pair Glint message rows into training records without assuming a dataset schema."""
    half = max_chars // 2
    pending_user: str | None = None
    records: list[dict] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") not in {None, "message"} and "message" not in row:
            continue

        message = row.get("message")
        if not isinstance(message, dict):
            if isinstance(row.get("messages"), list):
                text = _format_messages_chatml(row["messages"], max_chars) if fmt == "chatml" else _format_messages_text(row["messages"], max_chars)
                if text:
                    records.append({"text": text})
            continue

        role = message.get("role")
        content = _glint_extract_text(message.get("content", []))
        if not content:
            continue

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user:
            user_text = _trim(pending_user, half)
            asst_text = _trim(content, half)
            if fmt == "chatml":
                text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{asst_text}<|im_end|>"
            else:
                text = f"Human: {user_text}\n\nAssistant: {asst_text}"
            if len(text.strip()) >= 40:
                records.append({"text": text})
            pending_user = None

    return records


def _flatten_glint_json(obj) -> list[dict]:
    if isinstance(obj, dict):
        if "message" in obj or "messages" in obj:
            return [obj]
        rows: list[dict] = []
        for value in obj.values():
            if isinstance(value, (list, dict)):
                rows.extend(_flatten_glint_json(value))
        return rows
    if isinstance(obj, list):
        rows = []
        for item in obj:
            rows.extend(_flatten_glint_json(item))
        return rows
    return []


def build_glint_dataset_from_hub_files(
    repeat_epochs: int,
    fmt: str,
    max_chars: int,
    verbose: bool,
    max_files: int | None = None,
    max_records: int | None = None,
) -> list[dict]:
    """Fallback loader for Glint traces with heterogeneous JSON schemas."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [
        name for name in api.list_repo_files(GLINT_REPO, repo_type="dataset")
        if name.endswith((".json", ".jsonl"))
    ]
    if max_files and max_files > 0:
        files = files[:max_files]
    if verbose:
        print(f"[prepare] Glint fallback: parsing {len(files)} JSON files")

    records: list[dict] = []
    for name in files:
        if max_records and max_records > 0 and len(records) >= max_records:
            break
        local_path = Path(hf_hub_download(GLINT_REPO, filename=name, repo_type="dataset"))
        rows: list[dict] = []
        if name.endswith(".jsonl"):
            with local_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    try:
                        rows.extend(_flatten_glint_json(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                rows.extend(_flatten_glint_json(json.loads(local_path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        records.extend(_glint_pair_rows(rows, fmt=fmt, max_chars=max_chars))
        if max_records and max_records > 0 and len(records) >= max_records:
            records = records[:max_records]
            break

    base = records[:]
    for _ in range(repeat_epochs - 1):
        records.extend(base)

    if verbose:
        print(f"[prepare] Glint fallback: {len(base)} unique records × {repeat_epochs} epochs = {len(records)}")
    return records


def build_glint_dataset(
    repeat_epochs: int = 1,
    fmt: str = "text",
    verbose: bool = True,
    max_files: int | None = None,
    max_records: int | None = None,
) -> list[dict]:
    """Load Glint-Research/Fable-5-traces and return record list.

    The dataset uses HF Agent Traces format with type='message' rows.
    Messages alternate user→assistant; we pair them and format accordingly.
    """
    from datasets import load_dataset

    if verbose:
        print(f"[prepare] loading {GLINT_REPO} via file-by-file fallback ...")

    # Glint trace files have heterogeneous JSON schemas, which makes
    # datasets.load_dataset() fail while casting to one Arrow schema. The
    # file-by-file loader is slower but reliable and can be capped for local
    # smoke runs.
    max_chars = int((MAX_TOKENS_CHATML if fmt == "chatml" else MAX_TOKENS_APPROX) * CHARS_PER_TOKEN)
    return build_glint_dataset_from_hub_files(
        repeat_epochs=repeat_epochs,
        fmt=fmt,
        max_chars=max_chars,
        verbose=verbose,
        max_files=max_files,
        max_records=max_records,
    )


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
        choices=["clawd-code", "glint", "local"],
        default=["clawd-code"],
        help="Which source datasets to include (default: clawd-code only)",
    )
    parser.add_argument(
        "--local-path",
        action="append",
        default=[],
        help="Local file or directory to include when --datasets includes local. Defaults to the Clawd Fable project path set.",
    )
    parser.add_argument("--max-local-records", type=int, default=2000)
    parser.add_argument("--max-local-file-bytes", type=int, default=500_000)
    parser.add_argument("--max-glint-files", type=int, default=400, help="Cap Glint JSON files for local prep. Use 0 for all files.")
    parser.add_argument("--max-glint-records", type=int, default=800, help="Cap Glint records for local prep. Use 0 for all records.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    fmt = args.format
    verbose = not args.quiet

    # Auto-select output path based on format and dataset mix
    if args.output:
        out = Path(args.output)
    elif ("glint" in args.datasets or "local" in args.datasets) and fmt == "chatml":
        out = Path("data/clawd_fable_sft.jsonl")
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
            max_files=args.max_glint_files or None,
            max_records=args.max_glint_records or None,
        )
        all_records.extend(glint_records)

    if "local" in args.datasets:
        local_paths = [Path(p) for p in (args.local_path or DEFAULT_LOCAL_PATHS)]
        local_records = build_local_project_dataset(
            paths=local_paths,
            fmt=fmt,
            max_records=args.max_local_records,
            max_file_bytes=args.max_local_file_bytes,
            verbose=verbose,
        )
        all_records.extend(local_records)

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

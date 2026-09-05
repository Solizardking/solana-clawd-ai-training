# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret-free corpus iterator for the Solana trading tokenizer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Iterable

from .trading_tokens import all_trading_tool_tokens

FORBIDDEN_DIR_NAMES = {
    "node_modules",
    "keys",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".hf",
    ".cache",
    "wandb",
    "target",
}

FORBIDDEN_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.backup",
    ".env.production",
    ".env.development",
    "id.json",
    "wallet.json",
    "keypair.json",
}

FORBIDDEN_SUFFIXES = {
    ".pem",
    ".key",
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pkl",
    ".arrow",
    ".parquet",
    ".faiss",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".lock",
    ".lockb",
}

TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".json",
    ".jsonl",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".rs",
    ".idl",
}

PUMP_MCP_RELATIVE = ("src", "docs", "idl", "README.md", "package.json")

MAX_FILE_BYTES = 1_500_000
MAX_JSONL_LINES = 400


def is_secret_path(path: Path) -> bool:
    name = path.name
    if name in FORBIDDEN_FILE_NAMES:
        return True
    if name.startswith(".env"):
        return True
    lowered = name.lower()
    if "keypair" in lowered or lowered.endswith("-wallet.json"):
        return True
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    return any(part in FORBIDDEN_DIR_NAMES for part in path.parts)


def _read_text_file(path: Path) -> str | None:
    if is_secret_path(path) or not path.is_file():
        return None
    if path.stat().st_size > MAX_FILE_BYTES:
        return None
    suffix = path.suffix.lower()
    if suffix not in TEXT_SUFFIXES and path.name not in {"README.md", "package.json"}:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if suffix == ".jsonl":
        lines = text.splitlines()[:MAX_JSONL_LINES]
        return "\n".join(_jsonl_line_text(line) for line in lines if line.strip())
    return text


def _jsonl_line_text(line: str) -> str:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    if isinstance(payload, dict):
        parts: list[str] = []
        for key in ("instruction", "input", "output", "text", "content", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        if payload.get("messages") and isinstance(payload["messages"], list):
            for msg in payload["messages"]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
        return "\n".join(parts) if parts else line
    return line


def iter_pump_mcp_docs(root: Path) -> Iterator[str]:
    root = Path(root)
    if not root.exists():
        return
    for rel in PUMP_MCP_RELATIVE:
        target = root / rel
        if target.is_file():
            text = _read_text_file(target)
            if text:
                yield text
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file():
                text = _read_text_file(path)
                if text:
                    yield text


def iter_local_training_docs(repo_root: Path) -> Iterator[str]:
    repo_root = Path(repo_root)
    files = [
        repo_root / "solana1_yourgpt.jsonl",
        repo_root / "trainingday.jsonl",
    ]
    dirs = [
        repo_root / "configs",
        repo_root / "nvidia",
        repo_root / "trading_factory",
        repo_root / "data",
        repo_root / "memory",
    ]
    for path in files:
        text = _read_text_file(path)
        if text:
            yield text
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                text = _read_text_file(path)
                if text:
                    yield text


def iter_secret_free_corpus(
    *,
    pump_mcp_root: Path | None = None,
    repo_root: Path | None = None,
    extra_tokens: Iterable[str] | None = None,
    max_docs: int | None = None,
) -> Iterator[str]:
    """Yield corpus strings: tool names first, then harvested docs."""
    count = 0
    for name in all_trading_tool_tokens(extra_tokens):
        yield name
        count += 1
        if max_docs is not None and count >= max_docs:
            return
    if pump_mcp_root is not None:
        for doc in iter_pump_mcp_docs(pump_mcp_root):
            yield doc
            count += 1
            if max_docs is not None and count >= max_docs:
                return
    if repo_root is not None:
        for doc in iter_local_training_docs(repo_root):
            yield doc
            count += 1
            if max_docs is not None and count >= max_docs:
                return

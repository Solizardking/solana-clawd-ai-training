# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load, extend, save, and push the Nemotron Solana trading tokenizer."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .trading_tokens import (
    DEFAULT_HUB_REPO,
    NEMOTRON_TOKENIZER_ID,
    all_trading_tool_tokens,
)


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token or None


def load_nemotron_tokenizer(
    pretrained: str = NEMOTRON_TOKENIZER_ID,
    *,
    token: str | None = None,
):
    """Tokenizer files only — never download 30B NVFP4 weights."""
    from transformers import AutoTokenizer

    auth = token if token is not None else hf_token()
    kwargs: dict[str, Any] = {}
    if auth:
        kwargs["token"] = auth
    return AutoTokenizer.from_pretrained(pretrained, **kwargs)


def _added_token(name: str, *, special: bool):
    from transformers import AddedToken

    return AddedToken(
        name,
        single_word=True,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=special,
    )


def encode_text(tokenizer, text: str, *, add_special_tokens: bool = False) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return [int(i) for i in ids]


def decode_ids(tokenizer, ids: Iterable[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def tool_token_id(tokenizer, name: str) -> int:
    unk = tokenizer.unk_token_id
    tid = tokenizer.convert_tokens_to_ids(name)
    if tid is None or (unk is not None and tid == unk):
        raise KeyError(name)
    return int(tid)


def is_atomic_tool_token(tokenizer, name: str) -> bool:
    try:
        tid = tool_token_id(tokenizer, name)
    except KeyError:
        return False
    ids = encode_text(tokenizer, name, add_special_tokens=False)
    if len(ids) != 1:
        return False
    return ids[0] == tid and decode_ids(tokenizer, ids) == name


def apply_user_chat_template(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("apply_chat_template returned an empty value")
    return rendered


def extend_trading_tokenizer(tokenizer, extra_tokens: Iterable[str] | None = None):
    """Add PUMP-MCP + SOL GPT names so they encode as a single id."""
    names = all_trading_tool_tokens(extra_tokens)
    missing: list[str] = []
    split: list[str] = []
    for name in names:
        try:
            atomic = is_atomic_tool_token(tokenizer, name)
        except Exception:
            atomic = False
        if atomic:
            continue
        split.append(name)
        vocab = tokenizer.get_vocab()
        if name not in vocab:
            missing.append(name)
    if missing:
        tokenizer.add_tokens([_added_token(n, special=False) for n in missing])
    still_split = [n for n in names if not is_atomic_tool_token(tokenizer, n)]
    if still_split:
        existing = set(tokenizer.additional_special_tokens or [])
        to_add = [n for n in still_split if n not in existing]
        if to_add:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [_added_token(n, special=True) for n in to_add]}
            )
    return tokenizer


def train_or_extend_tokenizer(
    *,
    output_dir: str | Path,
    corpus: Iterable[str] | None = None,
    extra_tokens: Iterable[str] | None = None,
    pretrained: str = NEMOTRON_TOKENIZER_ID,
    token: str | None = None,
    train_new: bool = False,
    vocab_size: int | None = None,
):
    """Load Nemotron tokenizer, optionally train_new_from_iterator, then add tool tokens."""
    tokenizer = load_nemotron_tokenizer(pretrained, token=token)
    chat_template = getattr(tokenizer, "chat_template", None)
    docs = list(corpus or [])
    if train_new:
        if not docs:
            raise ValueError("train_new_from_iterator requires a non-empty corpus")
        size = int(vocab_size or tokenizer.vocab_size)
        tokenizer = tokenizer.train_new_from_iterator(docs, vocab_size=size)
        if chat_template:
            tokenizer.chat_template = chat_template
    tokenizer = extend_trading_tokenizer(tokenizer, extra_tokens=extra_tokens)
    save_tokenizer(tokenizer, output_dir)
    return tokenizer


def save_tokenizer(tokenizer, output_dir: str | Path) -> Path:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(dest))
    return dest


def push_tokenizer(
    tokenizer,
    repo_id: str = DEFAULT_HUB_REPO,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    auth = token if token is not None else hf_token()
    if not auth:
        raise RuntimeError("HF_TOKEN is required to push to the Hub")
    tokenizer.push_to_hub(repo_id, token=auth, private=private)
    return repo_id


def reload_tokenizer(source: str | Path, *, token: str | None = None):
    return load_nemotron_tokenizer(str(source), token=token)

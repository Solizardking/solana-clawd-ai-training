# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-vocab Solana agent tokenizer for chat / tool-use strings.

Encode/decode is lossless for ASCII (character fallback) and treats TRADE.md
plus SOL GPT tool names as first-class multi-character tokens. Does not load
model weights and does not require cudf.
"""

from __future__ import annotations

from typing import Iterable

from .trading_tokens import PUMP_MCP_TOOLS, SOL_GPT_TOOLS

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

# Longest-match phrases: pump.fun TRADE.md + SOL GPT catalog + Solana mechanics.
AGENT_PHRASES: tuple[str, ...] = (
    "permissionMode",
    "prepare_user_swap",
    "prepare_user_transfer",
    "prepare_phoenix_order",
    "prepare_phoenix_deposit",
    "prepare_phoenix_withdraw",
    "st_get_graduating_tokens",
    "scan_pump_token",
    "solana_token_info",
    "solana_top_traders",
    "solana_trending",
    "solana_price",
    "trade_execute",
    "memory_recall",
    "memory_write",
    "quote-api.jup.ag",
    "price.jup.ag",
    "slippageBps",
    "bonding %",
    "bonding%",
    "Market Cap",
    "pump.fun",
    "Raydium",
    "Jupiter",
    "Phoenix",
    "Imperial",
    "OODA",
    "OBSERVE",
    "ORIENT",
    "DECIDE",
    "LEARN",
    "INFERRED",
    "KNOWN",
    "observe",
    "orient",
    "decide",
    "act",
    "mint",
    "bonding",
    "graduation",
    "snipe",
    "ask",
    "PDA",
    "ALT",
    *PUMP_MCP_TOOLS,
    *SOL_GPT_TOOLS,
)

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class SolanaAgentTokenizer:
    """Pure encode/decode over a Solana + TRADE.md + SOL GPT vocab."""

    def __init__(self, extra_phrases: Iterable[str] | None = None) -> None:
        phrases = list(AGENT_PHRASES)
        if extra_phrases:
            phrases.extend(extra_phrases)
        # Longest first so `prepare_user_swap` wins over `prepare` / `user`.
        self._phrases = tuple(sorted(set(phrases), key=lambda p: (-len(p), p)))
        self._vocab: dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            self._add(token)
        for phrase in self._phrases:
            self._add(phrase)
        for ch in BASE58_ALPHABET:
            self._add(ch)
        for code in range(32, 127):
            self._add(chr(code))
        self._inv = {idx: token for token, idx in self._vocab.items()}

    def _add(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def lookup(self, token: str) -> int:
        if token not in self._vocab:
            raise KeyError(token)
        return self._vocab[token]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for phrase in self._phrases:
                if text.startswith(phrase, i):
                    matched = phrase
                    break
            if matched is not None:
                ids.append(self._vocab[matched])
                i += len(matched)
                continue
            ch = text[i]
            ids.append(self._vocab.get(ch, self._vocab["<unk>"]))
            i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._inv.get(int(i), "<unk>") for i in ids)

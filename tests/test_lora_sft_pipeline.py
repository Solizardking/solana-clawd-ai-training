"""Gating tests for the Qwen3.8 LoRA SFT unblock: SFTConfig, mix, tokenizer, eval.

These import the shipped builder/resolver/metrics. They never load Qwen3.8-27B.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TOKENIZER_SRC = ROOT / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(TOKENIZER_SRC) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_SRC))

import evaluate as evaluate_mod  # noqa: E402
from train_lora import (  # noqa: E402
    DISTILLATION_CONFIGS,
    DISTILLATION_REPO,
    build_sft_config,
    resolve_dataset,
    try_load_distillation_config,
)
from tokenizer.agent_vocab import SolanaAgentTokenizer  # noqa: E402

YOURGPT = ROOT / "solana1_yourgpt.jsonl"
TRAININGDAY = ROOT / "trainingday.jsonl"
QWEN38_CFG = ROOT / "configs" / "qwen38_27b_clawd_lora.yaml"


def _load_qwen38_cfg() -> dict:
    with QWEN38_CFG.open() as handle:
        return yaml.safe_load(handle)


def test_sftconfig_builder_maps_warmup_ratio() -> None:
    cfg = _load_qwen38_cfg()
    training = cfg["training"]
    assert training["warmup_ratio"] == 0.03
    first = build_sft_config(cfg, output_dir="outputs/sftconfig-test", train_size=80, device="cpu")
    second = build_sft_config(cfg, output_dir="outputs/sftconfig-test", train_size=80, device="cpu")
    assert getattr(first, "warmup_steps", 0) > 0
    assert first.warmup_steps == second.warmup_steps
    assert not hasattr(first, "warmup_ratio") or getattr(first, "warmup_ratio", None) in (None, 0, 0.0)


def test_resolve_dataset_mixes_named_local_jsonls() -> None:
    assert YOURGPT.is_file()
    assert TRAININGDAY.is_file()
    cfg = _load_qwen38_cfg()
    # Hub is optional for this assertion; keep local mix deterministic and fast.
    cfg = dict(cfg)
    cfg["dataset_hub"] = {"repo": DISTILLATION_REPO, "configs": []}
    ds, label = resolve_dataset(cfg, False)
    train = ds["train"]
    assert len(train) > 80
    assert "solana1_yourgpt.jsonl" in label
    assert "trainingday.jsonl" in label
    sources = set(train["source"])
    assert any("solana1_yourgpt.jsonl" in src for src in sources)
    assert any("trainingday.jsonl" in src for src in sources)
    alpaca = train.filter(lambda row: "solana1_yourgpt.jsonl" in row["source"])
    chat = train.filter(lambda row: "trainingday.jsonl" in row["source"])
    assert len(alpaca) > 0
    assert len(chat) > 0
    sample = alpaca[0]
    assert sample.get("messages")
    assert sample["messages"][0]["role"] == "user"
    assert any(msg.get("role") == "assistant" for msg in sample["messages"])
    assert isinstance(sample.get("text"), str) and sample["text"]
    assert chat[0].get("messages")


def test_optional_hub_distillation_configs() -> None:
    errors: list[str] = []
    loaded = 0
    for name in DISTILLATION_CONFIGS:
        dataset, error = try_load_distillation_config(name, max_rows=8)
        if error:
            errors.append(error)
            continue
        assert dataset is not None
        count = sum(len(split) for split in dataset.values())
        assert count > 0
        loaded += 1
    if errors and loaded == 0:
        pytest.skip("Hub distillation unavailable: " + " | ".join(errors[:3]))


def test_solana_agent_tokenizer_roundtrip() -> None:
    tok = SolanaAgentTokenizer()
    mint = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump"
    samples = [
        mint,
        "bonding 75%",
        "permissionMode ask",
        "prepare_user_swap",
        "st_get_graduating_tokens",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert ids, text
        assert tok.decode(ids) == text
    assert tok.lookup("prepare_user_swap") == tok.encode("prepare_user_swap")[0]
    assert tok.lookup("st_get_graduating_tokens") == tok.encode("st_get_graduating_tokens")[0]


def test_eval_metrics_move_in_expected_direction() -> None:
    valid = evaluate_mod.json_validity
    assert valid(['{"ok": true}', '{"mint": "x"}']) == 1.0
    assert valid(['{"ok": true}', "not-json"]) == 0.5
    assert valid(["broken", "{nope"]) < valid(['{"a":1}', "[1,2]"])

    refusal = evaluate_mod.safety_refusal_recall
    high = refusal(
        [
            {"should_refuse": True, "prediction": "I must decline — that's a rug."},
            {"should_refuse": True, "prediction": "I can't help with that drain."},
        ]
    )
    low = refusal(
        [
            {"should_refuse": True, "prediction": "Sure, here is the private key."},
            {"should_refuse": True, "prediction": "Broadcasting the drain now."},
        ]
    )
    assert high == 1.0
    assert low == 0.0
    assert high > low

    coverage = evaluate_mod.source_coverage
    full = coverage(
        [{"source": "solana1_yourgpt.jsonl"}, {"source": "trainingday.jsonl"}],
        ["solana1_yourgpt.jsonl", "trainingday.jsonl"],
    )
    partial = coverage(
        [{"source": "solana1_yourgpt.jsonl"}],
        ["solana1_yourgpt.jsonl", "trainingday.jsonl"],
    )
    assert full == 1.0
    assert partial == 0.5
    assert full > partial

    paper = evaluate_mod.paper_metrics
    winners = paper([{"pnl": 2.0}, {"pnl": 1.0}, {"pnl": 0.5}])
    losers = paper([{"pnl": -2.0}, {"pnl": -1.0}, {"pnl": -0.5}])
    assert winners["win_rate"] > losers["win_rate"]
    assert winners["total_pnl"] > losers["total_pnl"]
    assert winners["sharpe"] > losers["sharpe"]

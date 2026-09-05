"""Gating tests for the Nemotron Solana trading tokenizer.

Drives the shipped harvest / train-or-extend / encode / decode / chat-template
and clawd-ws parser. Does not load 30B NVFP4 weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TOKENIZER_SRC = ROOT / "nvidia" / "blueprints" / "transaction-foundation-model" / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(TOKENIZER_SRC) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_SRC))

from tokenizer.agent_vocab import SolanaAgentTokenizer  # noqa: E402
from tokenizer.clawd_ws import frame_to_text, parse_pump_frame  # noqa: E402
from tokenizer.corpus import is_secret_path, iter_secret_free_corpus  # noqa: E402
from tokenizer.hf_trading import (  # noqa: E402
    apply_user_chat_template,
    decode_ids,
    encode_text,
    is_atomic_tool_token,
    tool_token_id,
    train_or_extend_tokenizer,
)
from tokenizer.trading_tokens import (  # noqa: E402
    NEMOTRON_TOKENIZER_ID,
    PUMP_MCP_TOOLS,
    SOL_GPT_TOOLS,
    all_trading_tool_tokens,
)

FIXTURES = ROOT / "tests" / "fixtures"
PUMP_FIXTURE = FIXTURES / "pump_mcp"
WS_FIXTURES = FIXTURES / "clawd_ws"
MINT = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump"
ATOMIC_NAMES = (
    "get-token-info",
    "create-token",
    "buy-token",
    "sell-token",
    "list-accounts",
    "get-account-balance",
    "generate-image",
    "get-fee-tier",
    "list-free-models",
    "free-router-chat",
    "list-skills",
    "get-skill",
    "rerank-docs",
    "prepare_user_swap",
    "get_price",
    "list_phoenix_markets",
)


def test_tool_catalog_contains_required_names() -> None:
    names = all_trading_tool_tokens()
    for required in ATOMIC_NAMES:
        assert required in names
    assert len(PUMP_MCP_TOOLS) == 13
    assert len(SOL_GPT_TOOLS) == 72
    assert "prepare_user_swap" in SOL_GPT_TOOLS
    assert "get_price" in SOL_GPT_TOOLS
    assert "list_phoenix_markets" in SOL_GPT_TOOLS


def test_corpus_skips_secrets_and_reads_pump_mcp(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("SECRET=1\n")
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    (keys_dir / "default.json").write_text("{}")
    assert is_secret_path(env_file)
    assert is_secret_path(keys_dir / "default.json")
    docs = list(iter_secret_free_corpus(pump_mcp_root=PUMP_FIXTURE, max_docs=40))
    blob = "\n".join(docs)
    assert "get-token-info" in blob
    assert "SECRET=1" not in blob
    assert "node_modules" not in blob


def test_parse_pump_ws_fixtures() -> None:
    for name in ("status.json", "token-launch.json", "token-enriched.json"):
        payload = json.loads((WS_FIXTURES / name).read_text())
        parsed = parse_pump_frame(payload)
        assert parsed is not None
        assert parsed["type"] in {"status", "token-launch", "token-enriched"}
    assert parse_pump_frame('{"type":"nope"}') is None
    assert parse_pump_frame("not-json") is None


def test_solana_agent_tokenizer_includes_pump_mcp_tools() -> None:
    tok = SolanaAgentTokenizer()
    for name in ("get-token-info", "prepare_user_swap", "list_phoenix_markets"):
        ids = tok.encode(name)
        assert len(ids) == 1
        assert tok.decode(ids) == name
        assert tok.lookup(name) == ids[0]


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("solana-trading-tokenizer")
    corpus = list(
        iter_secret_free_corpus(
            pump_mcp_root=PUMP_FIXTURE,
            repo_root=ROOT,
            max_docs=48,
        )
    )
    tokenizer = train_or_extend_tokenizer(
        output_dir=output,
        corpus=corpus,
        pretrained=NEMOTRON_TOKENIZER_ID,
        train_new=False,
    )
    return tokenizer, output, corpus


def test_train_extend_atomic_roundtrip_and_chat_template(trained_tokenizer) -> None:
    tokenizer, output, _corpus = trained_tokenizer
    assert (output / "tokenizer.json").is_file() or (output / "tokenizer_config.json").is_file()
    for name in ATOMIC_NAMES:
        assert is_atomic_tool_token(tokenizer, name), name
        ids = encode_text(tokenizer, name, add_special_tokens=False)
        assert len(ids) == 1, (name, ids)
        assert decode_ids(tokenizer, ids) == name
        assert tool_token_id(tokenizer, name) == ids[0]
    mint_ids = encode_text(tokenizer, MINT, add_special_tokens=False)
    assert mint_ids
    assert decode_ids(tokenizer, mint_ids) == MINT
    rendered = apply_user_chat_template(tokenizer, "Who are you?")
    assert isinstance(rendered, str) and rendered.strip()
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        add_generation_prompt=True,
        tokenize=True,
    )
    assert tokenized


def test_tokenize_recorded_pump_frames(trained_tokenizer) -> None:
    tokenizer, _output, _corpus = trained_tokenizer
    raw = (WS_FIXTURES / "token-launch.json").read_text()
    frame = parse_pump_frame(raw)
    assert frame is not None
    ids = encode_text(tokenizer, frame_to_text(frame), add_special_tokens=False)
    assert ids


def test_hauhau_sibling_embeds_pump_ws_and_tokenizer() -> None:
    sibling = Path("/Users/8bit/sol-gpt/hauhau/nemotron-trading")
    html = (sibling / "www" / "index.html").read_text()
    assert 'data-pump-ws-url="wss://clawd-ws.fly.dev/ws"' in html
    assert "ordlibrary/solana-clawd-nemotron-trading-tokenizer" in html
    assert "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" in html
    original = Path("/Users/8bit/sol-gpt/hauhau/www/index.html").read_text()
    assert 'data-pump-ws-url="wss://clawd-ws.fly.dev/ws"' in original
    fly = (sibling / "fly.toml").read_text()
    assert "hauhau-nemotron-trading" in fly
    assert "hauhau-qwen36" not in fly.split("app", 1)[1][:80]

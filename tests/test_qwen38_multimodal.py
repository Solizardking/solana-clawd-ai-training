"""Unit tests for the Qwen/Qwen3.8-27B Clawd load, chat, and generate path.

These tests import the shipped helpers and spy the official transformers
factories. They never download Qwen/Qwen3.8-27B weights.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen38_multimodal as qwen38  # noqa: E402


class FakeIds:
    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self._tokens))


class FakeBatch(dict):
    def to(self, device):  # noqa: ANN001
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.template_kwargs: dict | None = None
        self.decoded = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
        self.messages = messages
        self.template_kwargs = kwargs
        return FakeBatch(input_ids=FakeIds([11, 22, 33]))

    def decode(self, tokens):  # noqa: ANN001
        self.decoded = list(tokens)
        return "a frog"


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.generate_kwargs: dict | None = None

    def generate(self, **kwargs):  # noqa: ANN003
        self.generate_kwargs = kwargs
        return [[11, 22, 33, 44, 55]]


@pytest.fixture
def fake_transformers(monkeypatch):
    recorded: dict = {"pipeline_calls": [], "causal_calls": []}

    auto_processor = MagicMock(name="AutoProcessor")
    auto_processor.from_pretrained.return_value = "processor"
    auto_mm = MagicMock(name="AutoModelForMultimodalLM")
    auto_mm.from_pretrained.return_value = "multimodal-model"
    auto_causal = MagicMock(name="AutoModelForCausalLM")

    def fake_pipeline(task, model=None, **kwargs):
        recorded["pipeline_calls"].append({"task": task, "model": model, "kwargs": kwargs})
        return "image-text-pipe"

    def fake_causal_from_pretrained(*args, **kwargs):
        recorded["causal_calls"].append((args, kwargs))
        return "causal-model"

    auto_causal.from_pretrained.side_effect = fake_causal_from_pretrained

    module = types.ModuleType("transformers")
    module.pipeline = fake_pipeline
    module.AutoProcessor = auto_processor
    module.AutoModelForMultimodalLM = auto_mm
    module.AutoModelForCausalLM = auto_causal
    monkeypatch.setitem(sys.modules, "transformers", module)
    return {
        "module": module,
        "recorded": recorded,
        "AutoProcessor": auto_processor,
        "AutoModelForMultimodalLM": auto_mm,
        "AutoModelForCausalLM": auto_causal,
    }


def test_pipeline_factory_uses_official_task_and_model(fake_transformers):
    pipe = qwen38.construct_qwen38_pipeline()
    assert pipe == "image-text-pipe"
    calls = fake_transformers["recorded"]["pipeline_calls"]
    assert len(calls) == 1
    assert calls[0]["task"] == "image-text-to-text"
    assert calls[0]["model"] == "Qwen/Qwen3.8-27B"


def test_load_uses_processor_and_multimodal_not_causal(fake_transformers):
    processor, model = qwen38.load_qwen38_processor_and_model()
    assert processor == "processor"
    assert model == "multimodal-model"
    fake_transformers["AutoProcessor"].from_pretrained.assert_called_once_with("Qwen/Qwen3.8-27B")
    fake_transformers["AutoModelForMultimodalLM"].from_pretrained.assert_called_once()
    _args, kwargs = fake_transformers["AutoModelForMultimodalLM"].from_pretrained.call_args
    assert _args[0] == "Qwen/Qwen3.8-27B"
    assert kwargs["device_map"] == "auto"
    fake_transformers["AutoModelForCausalLM"].from_pretrained.assert_not_called()
    assert fake_transformers["recorded"]["causal_calls"] == []


def test_official_candy_messages_match_hf_card():
    messages = qwen38.official_candy_messages()
    assert len(messages) == 1
    turn = messages[0]
    assert turn["role"] == "user"
    types_seen = {item["type"] for item in turn["content"]}
    assert types_seen == {"image", "text"}
    assert turn["content"][0] == {
        "type": "image",
        "url": qwen38.CANDY_IMAGE_URL,
    }
    assert turn["content"][1] == {
        "type": "text",
        "text": "What animal is on the candy?",
    }
    assert "candy.JPG" in turn["content"][0]["url"]


def test_text_only_clawd_rows_become_text_content():
    rows = [
        {"role": "system", "content": "You are Clawd."},
        {"role": "user", "content": "What is a PDA?"},
    ]
    normalized = qwen38.normalize_clawd_messages(rows)
    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"] == [{"type": "text", "text": "You are Clawd."}]
    assert normalized[1]["content"] == [{"type": "text", "text": "What is a PDA?"}]


def test_official_generate_path_records_template_generate_and_suffix_decode():
    processor = FakeProcessor()
    model = FakeModel()
    text = qwen38.official_generate(
        processor,
        model,
        qwen38.official_candy_messages(),
        max_new_tokens=40,
    )
    assert processor.template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == 40
    assert "input_ids" in model.generate_kwargs
    assert model.generate_kwargs["input_ids"].shape[-1] == 3
    assert processor.decoded == [44, 55]
    assert text == "a frog"


def test_official_generate_moves_inputs_to_model_device():
    tracked = FakeBatch(input_ids=FakeIds([1, 2]))
    processor = FakeProcessor()
    processor.apply_chat_template = lambda messages, **kwargs: tracked  # type: ignore[method-assign]
    model = FakeModel()
    qwen38.official_generate(
        processor,
        model,
        qwen38.official_candy_messages(),
        max_new_tokens=8,
    )
    assert tracked.moved_to == "cuda:0"
    assert model.generate_kwargs["max_new_tokens"] == 8


def test_train_config_base_model_is_exact_qwen38_id():
    from train_lora import load_config

    cfg = load_config(str(ROOT / "configs" / "qwen38_27b_clawd_lora.yaml"))
    assert cfg["base_model"] == "Qwen/Qwen3.8-27B"
    assert cfg["dataset_repo"] == "solanaclawd/solana-clawd-instruct"
    assert Path(cfg["dataset_path"]).exists() or (ROOT / cfg["dataset_path"]).exists()
    assert cfg["pipeline_task"] == "image-text-to-text"
    assert qwen38.is_qwen38_multimodal_base(cfg["base_model"])


def test_is_qwen38_base_is_exact():
    assert qwen38.is_qwen38_multimodal_base("Qwen/Qwen3.8-27B")
    assert not qwen38.is_qwen38_multimodal_base("Qwen/Qwen2.5-1.5B-Instruct")
    assert not qwen38.is_qwen38_multimodal_base("Qwen/Qwen3.8-27B-Instruct")

"""Data isolation and genuine Clef-head/LoRA gradient integration checks."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clef_research_data import build_decisions, digest
from clef_research_training import attach_lora, encode_rows, evaluate, load_upstream, supervised_loss


def raw_row(split, index, prompt=None):
    messages = [
        {"role": "system", "content": "Use supplied Solana research evidence."},
        {"role": "user", "content": prompt or f"{split}: Explain Solana account concept {index}."},
        {"role": "assistant", "content": f"Research response {split} {index}: Solana accounts carry address-specific data and owner information."},
    ]
    return {"messages": messages, "source_type": "parquet", "source": f"{split}.parquet",
            "record_id": str(index), "example_sha256": digest(messages)}


def fixture_splits():
    return {split: [raw_row(split, index) for index in range(16)] for split in ("train", "eval", "test")}


def test_data_labels_are_original_answers_and_candidates_stay_in_split():
    raw = fixture_splits()
    decisions, audit = build_decisions(raw)
    repeated, _ = build_decisions(raw)
    assert decisions == repeated
    all_labels = set()
    for split, rows in decisions.items():
        hashes = {row["example_sha256"] for row in raw[split]}
        originals = {row["record_id"]: row["messages"][-1]["content"] for row in raw[split]}
        assert audit[f"{split}_output"] == len(raw[split])
        for row in rows:
            question = row["questions"]["research_answer"]
            gold = row["labels"]["research_answer"]
            all_labels.add(gold)
            assert question["criteria"][gold] == originals[row["provenance"]["record_id"]]
            assert len(set(question["criteria"].values())) == 4
            assert set(row["candidate_example_sha256"].values()) <= hashes
            assert all(message["role"] != "assistant" for message in row["state"]["conversation"])
            assert "labels" not in row["state"] and "provenance" not in row["state"]
    assert all_labels == {f"option_{i}" for i in range(4)}


def test_duplicate_prompt_is_removed_from_training_with_heldout_priority():
    raw = fixture_splits()
    raw["train"].append(raw_row("train", 30, "Shared question"))
    raw["eval"].append(raw_row("eval", 30, "Shared question"))
    raw["test"].append(raw_row("test", 30, "Shared question"))
    decisions, audit = build_decisions(raw)
    assert audit["train_duplicate_prompt_removed"] == 1
    assert audit["eval_duplicate_prompt_removed"] == 1
    sets = [{row["provenance"]["prompt_hash"] for row in decisions[split]} for split in ("train", "eval", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_multiturn_or_nontext_examples_are_rejected():
    raw = fixture_splits()
    malformed = raw_row("train", 100)
    malformed["messages"].insert(2, {"role": "assistant", "content": "An earlier answer would leak the target."})
    raw["train"].append(malformed)
    nontext = raw_row("train", 101)
    nontext["messages"][-1]["content"] = [{"type": "text", "text": "not supported"}]
    raw["train"].append(nontext)
    _, audit = build_decisions(raw)
    assert audit["train_invalid"] == 2


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[1 + byte % 126 for byte in text.encode()])


@pytest.fixture(scope="module")
def upstream():
    return load_upstream()


def test_encoding_does_not_include_labels_or_provenance_and_reports_context_exclusion(upstream):
    from types import SimpleNamespace
    rows = build_decisions(fixture_splits())[0]["train"][:1]
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    altered = copy.deepcopy(rows[0])
    altered["labels"] = {"research_answer": "not_an_option"}
    altered["provenance"] = {"secret": "must never enter model input"}
    assert upstream.encode_record(processor.tokenizer, rows[0]) == upstream.encode_record(processor.tokenizer, altered)
    encoded, kept, errors = encode_rows(upstream, processor, rows, max_length=10)
    assert not encoded and not kept
    assert len(errors) == 1 and "exceeds" in errors[0]["reason"]


def test_native_clef_head_and_lora_train_and_saved_weights_reload(upstream, tmp_path):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from types import SimpleNamespace
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "vocab_size": 128, "layer_types": ["full_attention", "full_attention"],
                     "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "patch_size": 2,
                       "spatial_merge_size": 1, "num_position_embeddings": 16},
    )
    backbone = Qwen3_5ForConditionalGeneration(config)
    base_state = copy.deepcopy(backbone.state_dict())
    head_config = dict(hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    model = upstream.ClefModel(backbone, upstream.JointSchemaHead(**head_config))
    targets = attach_lora(model, rank=2)
    assert all(".language_model.layers." in target for target in targets)
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if ".visual." in name)
    rows = build_decisions(fixture_splits())[0]["train"][:1]
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    encoded, kept, excluded = encode_rows(upstream, processor, rows, max_length=4096)
    assert not excluded
    batch = upstream.collate_records(encoded, 0, torch.device("cpu"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    loss = supervised_loss(model(batch), encoded, rows)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if "lora_" in name)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if name.startswith("head."))
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if "lora_" in name)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name.startswith("head."))
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters() if not value.requires_grad)
    metrics, predictions = evaluate(model, upstream, processor, encoded, rows, torch.device("cpu"))
    assert metrics["questions"] == 1 and 0 <= metrics["accuracy"] <= 1
    model.language_model.save_pretrained(tmp_path)
    save_file(model.head.state_dict(), str(tmp_path / "joint_head.safetensors"))
    restored_base = Qwen3_5ForConditionalGeneration(config)
    restored_base.load_state_dict(base_state)
    restored = upstream.ClefModel(PeftModel.from_pretrained(restored_base, tmp_path), upstream.JointSchemaHead(**head_config))
    restored.head.load_state_dict(load_file(str(tmp_path / "joint_head.safetensors")))
    _, restored_predictions = evaluate(restored, upstream, processor, encoded, rows, torch.device("cpu"))
    assert restored_predictions == predictions

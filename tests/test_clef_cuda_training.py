"""CUDA continuation contracts plus real native-head CPU optimizer arithmetic.

These tests load no production model weights and do not substitute CPU for a
CUDA run. The paid-machine benchmark must prove actual NF4 CUDA kernels,
migration probabilities, memory, native gradients, and fresh saved-base reload.
The small constructed CPU model exercises exact loss and optimizer resume.
"""
from array import array
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_clef_cuda as cuda
from clef_research_data import decision_prompt_hash
from test_clef_mps_quantized import native_module, processor_fixture


def decision(identifier, source="fixture", source_type="research", state="Liquidity is stale."):
    row = {"id": identifier, "state": state,
        "questions": {"answer": {"type": "choice", "instructions": "Select the grounded response.",
            "criteria": {"buy": "Trade.", "wait": "Wait."}}},
        "labels": {"answer": "wait"}, "provenance": {"source": source, "source_type": source_type}}
    row["provenance"]["prompt_hash"] = decision_prompt_hash(row)
    return row


def predictions():
    return [{"id": f"pilot-{index}", "answers": {"answer": {
        "choice": "wait", "label": "wait", "probabilities": {"buy": 0.25, "wait": 0.75}}}} for index in range(16)]


def contract():
    return {"run_id": "owned-run", "run_mode": "full", "stage_manifest_sha256": "1" * 64,
        "model_revision": cuda.MODEL_REVISION, "dataset_revision": cuda.DATASET_REVISION,
        "max_length": 2048, "microbatch_size": 4, "gradient_accumulation": 1,
        "seed": 42, "learning_rate": 2e-5, "training_order_sha256": "2" * 64,
        "planned_steps": 3, "optimizer_steps": 1, "attention_implementation": "eager", "use_kernels": True,
        "runtime_source_sha256": {"train_clef_cuda.py": "3" * 64}}


def test_order_has_all_live_papers_and_longest_before_benchmark_and_exact_epoch_tail():
    rows = [decision("live-a", "https://clawd-ws.fly.dev/", "live_tape_observation"),
            decision("red-a", "2605.12151v2.pdf"), decision("red-b", "2605.12151v2.pdf"),
            decision("hour", "2606.08232v1.pdf")] + [decision(f"other-{index}") for index in range(101)]
    encoded = [SimpleNamespace(input_ids=[1] * (2048 if index == 80 else 30)) for index in range(len(rows))]
    order, required, longest = cuda.epoch_order(rows, encoded, 42)
    assert required == [0, 1, 2, 3] and longest == 80
    assert order[:5] == [0, 1, 2, 3, 80]
    assert order == cuda.epoch_order(rows, encoded, 42)[0]
    assert sorted(order) == list(range(len(rows)))
    benchmark = cuda.step_groups(order, 4, 1, 16)
    assert len(benchmark) == 16 and {0, 1, 2, 3, 80} <= set(sum(benchmark, []))
    full = cuda.step_groups(order, 4, 2)
    assert len(full) == 14 and len(full[-1]) == 1 and sum(full, []) == order
    cuda.verify_coverage(order, sum(full, []), len(order), True)
    for trained in (order[:-1], [*order[:-1], order[0]], [*order[:2], order[3], order[2], *order[4:]]):
        with pytest.raises(ValueError, match="coverage|cover"):
            cuda.verify_coverage(order, trained, len(trained), True)


@pytest.mark.parametrize("mutation", [
    lambda values: values.pop(),
    lambda values: values[0].update(id="pilot-1"),
    lambda values: values[0]["answers"]["answer"]["probabilities"].update(buy=0.3),
    lambda values: values[0]["answers"]["answer"].update(choice="buy"),
    lambda values: values[0]["answers"]["answer"].update(label="buy"),
    lambda values: values[0]["answers"]["answer"]["probabilities"].update(buy=float("nan")),
])
def test_migration_rejects_incomplete_wrong_identity_choice_label_or_drift(mutation):
    expected, actual = predictions(), predictions()
    mutation(actual)
    with pytest.raises(ValueError):
        cuda.check_migration(expected, actual)


def test_migration_declares_measured_strict_tolerance_and_sixteen_real_records():
    expected, actual = predictions(), predictions()
    actual[0]["answers"]["answer"]["probabilities"].update(buy=0.2505, wait=0.7495)
    evidence = cuda.check_migration(expected, actual)
    assert evidence["matched"] and evidence["records"] == 16 and evidence["tolerance"] == 1e-3
    assert evidence["max_probability_difference"] == pytest.approx(0.0005)


def test_encoding_preserves_complete_native_tokens_and_refuses_length_or_secret_exclusions(tmp_path):
    module, processor = native_module(), processor_fixture()
    rows = [decision("safe")]
    expected = module.encode_record(processor.tokenizer, rows[0], max_length=1_000_000, processor=processor)
    encoded = cuda.encode_complete(module, processor, rows, 2048)
    assert isinstance(encoded[0].input_ids, array) and encoded[0].input_ids.itemsize == 4
    assert list(encoded[0].input_ids) == list(expected.input_ids) and encoded[0].questions == expected.questions
    with pytest.raises(ValueError, match="context cap"):
        cuda.encode_complete(module, processor, rows, len(expected.input_ids) - 1)
    unsafe = decision("secret", state="HF_TOKEN=hf_" + "x" * 36)
    with pytest.raises(ValueError, match="privacy"):
        cuda.encode_complete(module, processor, [unsafe], 2048)
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="unique"):
        cuda.read_rows(path)


@pytest.mark.parametrize("field,value", [("dataset_revision", "0" * 40), ("learning_rate", 1e-4),
    ("microbatch_size", 1), ("training_order_sha256", "0" * 64), ("use_kernels", False),
    ("runtime_source_sha256", {"train_clef_cuda.py": "0" * 64})])
def test_true_resume_refuses_source_optimizer_order_kernel_or_code_changes(field, value):
    saved, requested = contract(), contract()
    requested[field] = value
    with pytest.raises(ValueError, match="contract"):
        cuda.validate_resume(saved, requested, {"cursor": 4, "optimizer_steps": 1, "trained_indices": [0, 1, 2, 3]}, list(range(10)))


def test_resume_exact_cursor_and_partial_final_update_are_validated():
    saved, requested = contract(), contract()
    cuda.validate_resume(saved, requested, {"cursor": 4, "optimizer_steps": 1, "trained_indices": [0, 1, 2, 3]}, list(range(10)))
    saved["optimizer_steps"] = 3
    cuda.validate_resume(saved, requested, {"cursor": 10, "optimizer_steps": 3, "trained_indices": list(range(10))}, list(range(10)))
    for state in ({"cursor": 4, "optimizer_steps": 2, "trained_indices": [0, 1, 2, 3]},
                  {"cursor": 4, "optimizer_steps": 1, "trained_indices": [0, 1, 2, 2]}):
        with pytest.raises(ValueError):
            cuda.validate_resume(contract(), requested, state, list(range(10)))


def test_cpu_machine_cannot_claim_cuda_execution(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back"):
        cuda.require_cuda()


def test_staged_reference_cannot_escape_or_follow_symlink(tmp_path):
    inside, outside = tmp_path / "stage", tmp_path / "outside"
    inside.mkdir()
    outside.write_text("protected")
    (inside / "safe").write_text("fixture")
    assert cuda.stage_path(inside, "safe") == inside / "safe"
    (inside / "escape").symlink_to(outside)
    for value in ("../outside", str(outside), "escape", "absent"):
        with pytest.raises(ValueError):
            cuda.stage_path(inside, value)


def toy_model(module):
    import torch
    from peft import LoraConfig, get_peft_model
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(32, 32, bias=False).to(dtype=torch.bfloat16)
        def forward(self, value):
            return self.projection(value)
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = get_peft_model(Text(), LoraConfig(r=8, lora_alpha=16, target_modules=["projection"], lora_dropout=0))
            self.embedding = torch.nn.Embedding(256, 32).to(dtype=torch.bfloat16).requires_grad_(False)
            self.output_embedding = torch.nn.Embedding(256, 32).to(dtype=torch.bfloat16).requires_grad_(False)
            self.head = module.JointSchemaHead(hidden_size=32, width=16, routing_layers=1,
                layers=1, heads=2, feedforward=32).to(dtype=torch.bfloat16)
        def forward(self, batch):
            hidden = self.language_model(self.embedding(batch["input_ids"]))
            return self.head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], self.output_embedding.weight)
    return Native()


def test_real_native_head_lora_question_weighted_accumulation_and_adamw_resume_cpu(tmp_path):
    """No CUDA claim: actual native BF16 head and rank8 PEFT arithmetic on CPU."""
    import torch
    from clef_research_training import trainable_fingerprints
    torch.manual_seed(91)
    module, processor = native_module(), processor_fixture()
    rows = [decision(f"native-{index}", state=f"Source {index} reports stale liquidity.") for index in range(3)]
    rows[0]["questions"]["second"] = deepcopy(rows[0]["questions"]["answer"])
    rows[0]["labels"]["second"] = "buy"
    rows[0]["provenance"]["prompt_hash"] = decision_prompt_hash(rows[0])
    encoded = cuda.encode_complete(module, processor, rows, 2048)
    model = toy_model(module)
    identical = deepcopy(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01, foreach=False)
    other_optimizer = torch.optim.AdamW([parameter for parameter in identical.parameters() if parameter.requires_grad], lr=0.01, foreach=False)
    before = trainable_fingerprints(model)
    loss, gradients, padded = cuda.batch_update(model, module, processor, encoded, rows, [0, 1, 2], optimizer, 1, torch.device("cpu"))
    other_loss, other_gradients, _ = cuda.batch_update(identical, module, processor, encoded, rows, [0, 1, 2], other_optimizer, 3, torch.device("cpu"))
    assert gradients == other_gradients == {"lora": True, "head": True}
    assert loss == pytest.approx(other_loss, abs=0.01) and padded == sum(len(item.input_ids) for item in encoded)
    after = trainable_fingerprints(model)
    assert all(before[kind]["sha256"] != after[kind]["sha256"] for kind in ("lora", "head"))
    # Serialize actual optimizer moments and model bytes; then resume a second
    # native update and require identical complete parameter bytes.
    state = {"model": model.state_dict(), "optimizer": cuda._cpu_state(optimizer.state_dict()), "rng": torch.get_rng_state()}
    torch.save(state, tmp_path / "resume.pt")
    restored = toy_model(module)
    loaded = torch.load(tmp_path / "resume.pt", weights_only=True)
    restored.load_state_dict(loaded["model"], strict=True)
    restored_optimizer = torch.optim.AdamW([parameter for parameter in restored.parameters() if parameter.requires_grad], lr=0.01, foreach=False)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["rng"])
    cuda.batch_update(model, module, processor, encoded, rows, [2, 1], optimizer, 1, torch.device("cpu"))
    torch.set_rng_state(loaded["rng"])
    cuda.batch_update(restored, module, processor, encoded, rows, [2, 1], restored_optimizer, 1, torch.device("cpu"))
    assert trainable_fingerprints(model) == trainable_fingerprints(restored)
    assert all(not parameter.requires_grad for parameter in restored.embedding.parameters())


def test_checkpoint_completion_is_last_atomic_manifest_and_pruning_preserves_unowned(tmp_path, monkeypatch):
    import torch
    required = ("adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors",
                "joint_head_config.json", "joint_schema_model.py", "training.json",
                "tokenizer.json", "tokenizer_config.json", "processor_config.json")
    metadata = contract()
    metadata["optimizer_steps"] = 2
    def save(*args):
        output, value = args[4], args[5]
        output.mkdir(parents=True)
        for name in required:
            (output / name).write_text(json.dumps(value) if name == "training.json" else "CPU completion fixture")
        cuda.refresh_adapter_manifest(output)
        assert cuda.verify_adapter_manifest(output).get("checkpoint_complete") is None
        return value
    monkeypatch.setattr(cuda, "save_artifact", save)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    path = cuda.checkpoint(None, None, None, None, tmp_path, metadata, None, optimizer, 8, list(range(8)))
    manifest = cuda.verify_adapter_manifest(path)
    assert manifest["checkpoint_complete"] is True and manifest["cursor"] == 8 and manifest["optimizer_steps"] == 2
    assert {"optimizer.pt", "resume.json"} <= manifest["files"].keys()
    with pytest.raises(ValueError, match="overwritten"):
        cuda.checkpoint(None, None, None, None, tmp_path, metadata, None, optimizer, 8, list(range(8)))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("protected")
    with pytest.raises(ValueError, match="outside"):
        cuda.prune_checkpoints(tmp_path, [unrelated, path], 1)
    assert (unrelated / "keep").is_file() and path.exists()
    cuda.prune_checkpoints(tmp_path, [path], 0)
    assert not path.exists()

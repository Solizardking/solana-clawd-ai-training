"""Local trainer guards and real tiny MPS native-head/subset-LoRA reload.

The actual MPS test uses a six-layer Qwen3.5 hybrid and serializable native
processor. It proves this workflow on tiny weights, not 27B memory or quality.
Unsupported MPS operations fail; no xfail, fallback, or mocked backend is used.
"""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clef_research_data import decision_prompt_hash, MODEL_ID, MODEL_REVISION
from train_clef_local import (DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, attach_local_lora, finite_trainables, guard_model_input,
    load_local_adapter, load_local_base, memory_guard, prune_owned_checkpoints,
    refresh_adapter_manifest, resident_tensor_bytes, save_local_adapter,
    select_encoded_cohort, source_requirements, training_order, require_training_budget,
    verify_adapter_manifest, verify_prepared_source, write_json)
from test_clef_mps_quantized import native_module, processor_fixture, quantization_config, require_mps


def decision(identifier, source, source_type="research", state="Source reports stale liquidity."):
    row = {
        "id": identifier, "state": state,
        "questions": {"answer": {"type": "choice", "instructions": "Select the source-grounded response.",
            "criteria": {"buy": "Trade immediately.", "wait": "Wait for fresh liquidity evidence."}}},
        "labels": {"answer": "wait"},
        "provenance": {"source": source, "source_type": source_type},
    }
    row["provenance"]["prompt_hash"] = decision_prompt_hash(row)
    return row


def training_metadata():
    return {"status": "trained_pending_reload_verification", "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "optimizer_steps": 1, "run_mode": "tiny_test",
        "security_filter": {"version": 1, "applied": True, "exclusion_counts": {}, "published_source_modified": False},
        "input_parquet_sha256": deepcopy(INPUT_PARQUET_SHA256),
        "input_raw_row_counts": deepcopy(INPUT_RAW_ROW_COUNTS),
        "verification_scope": "Tiny random six-layer architecture; no production weights or dataset training claim."}


def source_manifest():
    return {"dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "input_parquet_sha256": deepcopy(INPUT_PARQUET_SHA256),
        "input_source": {"input_rows": deepcopy(INPUT_RAW_ROW_COUNTS)},
        "outputs": {split: {"rows": count - 1} for split, count in INPUT_RAW_ROW_COUNTS.items()},
        "security_filter": {"applied": True, "version": 1, "exclusion_counts": {"train_excluded_signing_byte_array": 11}}}


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(dataset_revision="0" * 40),
    lambda m: m["input_source"]["input_rows"].update(train=10),
    lambda m: m["input_parquet_sha256"].update(train="1" * 64),
    lambda m: m["security_filter"].update(applied=False),
    lambda m: m.update(model_revision="2" * 40),
])
def test_prepared_source_fails_on_wrong_commit_counts_hashes_or_security(mutation):
    manifest = source_manifest()
    mutation(manifest)
    with pytest.raises(ValueError):
        verify_prepared_source(manifest)


def test_prepared_source_preserves_full_unfiltered_counts_and_security_evidence():
    evidence = verify_prepared_source(source_manifest())
    assert sum(evidence["input_raw_row_counts"].values()) == 83662
    assert evidence["security_filter"]["exclusion_counts"]["train_excluded_signing_byte_array"] == 11
    assert evidence["input_parquet_sha256"] == INPUT_PARQUET_SHA256


def test_cohort_backfills_complete_safe_records_and_pilot_actually_sees_papers_and_live():
    module, processor = native_module(), processor_fixture()
    rows = [decision("live", "https://clawd-ws.fly.dev/", "live_tape_observation"),
            decision("red", "https://arxiv.org/abs/2605.12151"),
            decision("hour", "https://arxiv.org/abs/2606.08232"),
            decision("too-long", "length fixture", state="Observation " * 3000),
            decision("secret", "security fixture", state="HF_TOKEN=hf_" + "a" * 36),
            decision("safe", "other source")]
    encoded, selected, audit = select_encoded_cohort(rows, module, processor, 0, 42, 1024, True)
    assert {row["id"] for row in selected} == {"live", "red", "hour", "safe"}
    assert len(encoded) == 4 and audit["prepared"] == 6 and audit["selected"] == 4
    assert {entry["reason"] for entry in audit["excluded"]} == {"credential_payload", "context_exceeds_limit"}
    assert all(len(record.input_ids) <= 1024 for record in encoded)
    again = select_encoded_cohort(rows, module, processor, 0, 42, 1024, True)
    assert [row["id"] for row in again[1]] == [row["id"] for row in selected]
    required = source_requirements(selected)
    order = training_order(selected, 42, 0)
    assert order[:len(required)] == required
    assert {selected[index]["id"] for index in required} == {"live", "red", "hour"}
    with pytest.raises(ValueError, match="required live and paper"):
        select_encoded_cohort(rows, module, processor, 2, 42, 1024, True)
    with pytest.raises(ValueError, match="privacy"):
        guard_model_input(rows[4])


def test_memory_guard_counts_external_tensor_bytes_and_driver_not_only_allocator(monkeypatch):
    import torch
    model = torch.nn.Linear(16, 16, bias=False)
    tensor_bytes = resident_tensor_bytes(model)
    assert tensor_bytes == 16 * 16 * 4
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 2048)
    with pytest.raises(RuntimeError, match="working-set"):
        memory_guard({"allocator_limit_bytes": 4096}, additional_bytes=2049, model=model)
    observed = memory_guard({"allocator_limit_bytes": 4096}, additional_bytes=100, model=model)
    assert observed["allocated_bytes"] == 0 and observed["guarded_working_bytes"] == 2048
    assert observed["model_tensor_bytes"] == tensor_bytes
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0)
    with pytest.raises(RuntimeError, match="working-set"):
        memory_guard({"allocator_limit_bytes": 1023}, model=model)


def test_retention_removes_only_completed_owned_checkpoint(tmp_path):
    completed = []
    for step in (8, 16, 24):
        path = tmp_path / "checkpoints" / f"step-{step:06d}"
        path.mkdir(parents=True)
        write_json(path / "adapter_artifact_manifest.json", {"files": {}})
        completed.append(path)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep").write_text("unrelated")
    prune_owned_checkpoints(tmp_path, completed, 2)
    assert not (tmp_path / "checkpoints/step-000008").exists()
    assert all(path.is_dir() for path in completed) and (outside / "keep").is_file()
    with pytest.raises(ValueError, match="outside"):
        prune_owned_checkpoints(tmp_path, [outside, *completed], 1)
    assert (outside / "keep").is_file()


def test_full_adapter_inventory_tracks_metadata_and_processor_tampering(tmp_path):
    required = ("adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors",
                "joint_head_config.json", "joint_schema_model.py", "training.json",
                "tokenizer.json", "tokenizer_config.json", "processor_config.json")
    for name in required:
        (tmp_path / name).write_text(name)
    refresh_adapter_manifest(tmp_path)
    assert set(verify_adapter_manifest(tmp_path)["files"]) == set(required)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "joint_schema_model.cpython-test.pyc").write_bytes(b"operational import cache")
    assert set(verify_adapter_manifest(tmp_path)["files"]) == set(required)
    assert set(refresh_adapter_manifest(tmp_path)) == set(required)
    (tmp_path / "tokenizer.json").write_text("changed tokenizer")
    with pytest.raises(ValueError, match="tokenizer.json"):
        verify_adapter_manifest(tmp_path)
    refresh_adapter_manifest(tmp_path)
    (tmp_path / "training.json").write_text("changed metadata")
    with pytest.raises(ValueError, match="training.json"):
        verify_adapter_manifest(tmp_path)
    refresh_adapter_manifest(tmp_path)
    (tmp_path / "extra.json").write_text("untracked")
    with pytest.raises(ValueError, match="actual files"):
        verify_adapter_manifest(tmp_path)


def test_actual_mps_last_two_layers_native_head_training_and_prequantized_base_adapter_reload(tmp_path):
    torch = require_mps()
    import bitsandbytes as bnb
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from clef_research_training import supervised_loss, trainable_fingerprints

    torch.manual_seed(23)
    module, processor = native_module(), processor_fixture()
    config = Qwen3_5Config(
        text_config={"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 6,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 256,
            "layer_types": ["linear_attention", "full_attention"] * 3,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 64, "linear_value_head_dim": 64,
            "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 128, "intermediate_size": 256,
            "num_heads": 2, "out_hidden_size": 128, "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "num_position_embeddings": 16},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4)
    original = tmp_path / "tiny-original-bf16"
    backbone = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    original_vision = {name: value.detach().cpu().clone() for name, value in backbone.state_dict().items() if ".visual." in name}
    backbone.save_pretrained(original)
    processor.save_pretrained(original)
    head_config = {"hidden_size": 128, "width": 64, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 128}
    head = module.JointSchemaHead(**head_config).to(dtype=torch.bfloat16)
    save_file(head.state_dict(), str(original / "joint_head.safetensors"))
    write_json(original / "joint_head_config.json", head_config)
    del backbone, head

    quantized, processor = module.load_release_model(original, device="mps", dtype=torch.bfloat16,
        quantization_config=quantization_config(), attn_implementation="eager", local_files_only=True)
    base = tmp_path / "tiny-prequantized-standalone"
    quantized.language_model.save_pretrained(base, max_shard_size="1MB")
    processor.save_pretrained(base)
    for name in ("joint_head.safetensors", "joint_head_config.json"):
        shutil.copyfile(original / name, base / name)
    shutil.copyfile(Path(module.__file__), base / "joint_schema_model.py")
    del quantized
    gc.collect()
    torch.mps.empty_cache()

    # Production helper must reload the saved NF4 base with no new quant config.
    model, processor = load_local_base(module, base)
    scope = attach_local_lora(model, rank=2, last_layers=2)
    assert scope["text_layers"] == [4, 5]
    assert all(".layers.4." in name or ".layers.5." in name for name in scope["target_modules"])
    assert not any("visual" in name or "lm_head" in name for name in scope["target_modules"])
    assert not hasattr(model.language_model, "_require_grads_hook")
    native_backbone = model.language_model.get_base_model()
    assert all(not parameter.requires_grad for parameter in native_backbone.model.language_model.layers[:4].parameters())
    embedded = native_backbone.get_input_embeddings()(torch.tensor([[10, 11]], device="mps"))
    assert embedded.requires_grad is False
    row = decision("actual-local-mps", "tiny architecture fixture")
    record = module.encode_record(processor.tokenizer, row, processor=processor)
    batch = module.collate_records([record], processor.tokenizer.pad_token_id, torch.device("mps"))
    requires_grad_at_boundary = []
    hook = native_backbone.model.language_model.layers[3].register_forward_hook(
        lambda layer, inputs, result: requires_grad_at_boundary.append(result.requires_grad))
    initial = trainable_fingerprints(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, foreach=False)
    model.train()
    logits = model(batch)
    loss = supervised_loss(logits, [record], [row])
    assert loss.device.type == "mps" and bool(torch.isfinite(loss))
    loss.backward()
    hook.remove()
    assert requires_grad_at_boundary and not any(requires_grad_at_boundary)
    assert any("lora_" in name and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for name, parameter in model.named_parameters())
    assert any(name.startswith("head.") and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for name, parameter in model.named_parameters())
    optimizer.step()
    updated = trainable_fingerprints(model)
    assert initial["lora"]["sha256"] != updated["lora"]["sha256"]
    assert initial["head"]["sha256"] != updated["head"]["sha256"]
    assert all(entry["finite"] for entry in finite_trainables(model).values())
    model.eval()
    with torch.inference_mode():
        expected = model(batch)[0][0].float().cpu()
    assert bool(torch.isfinite(expected).all())
    for name, value in native_backbone.state_dict().items():
        if name in original_vision:
            torch.testing.assert_close(value.cpu(), original_vision[name], rtol=0, atol=0)
    adapter = tmp_path / "real-tiny-trained-adapter"
    metadata = save_local_adapter(model, processor, module, base, adapter, training_metadata())
    assert metadata["trainable_fingerprints"] == updated
    verify_adapter_manifest(adapter)
    del model, native_backbone, parameters, optimizer, logits, loss, embedded
    gc.collect()
    torch.mps.empty_cache()
    restored, restored_processor = load_local_adapter(module, base, adapter, allow_unverified=True)
    restored_record = module.encode_record(restored_processor.tokenizer, row, processor=restored_processor)
    assert restored_record.input_ids == record.input_ids
    restored_batch = module.collate_records([restored_record], restored_processor.tokenizer.pad_token_id, torch.device("mps"))
    with torch.inference_mode():
        observed = restored(restored_batch)[0][0].float().cpu()
    torch.testing.assert_close(observed, expected, rtol=1e-3, atol=1e-3)
    assert int(observed.argmax()) == int(expected.argmax())
    assert any(isinstance(child, bnb.nn.Linear4bit) for child in restored.language_model.modules())
    for name, value in restored.language_model.get_base_model().state_dict().items():
        if name in original_vision:
            torch.testing.assert_close(value.cpu(), original_vision[name], rtol=0, atol=0)
    # Plain inference is allowed only after verification metadata and its hash
    # have been refreshed, while changed native head files fail before loading.
    metadata["status"] = "trained_and_reload_verified"
    write_json(adapter / "training.json", metadata)
    refresh_adapter_manifest(adapter)
    verify_adapter_manifest(adapter)
    with (adapter / "joint_head.safetensors").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="joint_head.safetensors"):
        load_local_adapter(module, base, adapter)
    del restored, restored_batch, batch
    gc.collect()
    torch.mps.empty_cache()



def test_compact_native_ids_preserve_real_mps_collation_gradients_and_adapter_reload(tmp_path):
    from array import array
    torch = require_mps()
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from clef_research_training import supervised_loss, trainable_fingerprints
    module, processor = native_module(), processor_fixture()
    torch.manual_seed(91)
    config = Qwen3_5Config(
        text_config={"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 64,
            "vocab_size": 256, "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 64, "linear_value_head_dim": 64,
            "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 128, "intermediate_size": 256,
            "num_heads": 2, "out_hidden_size": 128, "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "num_position_embeddings": 16},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4)
    original = tmp_path / "tiny-compact-original"
    backbone = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    backbone.save_pretrained(original)
    processor.save_pretrained(original)
    head_config = {"hidden_size": 128, "width": 64, "routing_layers": 1,
                   "layers": 1, "heads": 2, "feedforward": 128}
    head = module.JointSchemaHead(**head_config).to(dtype=torch.bfloat16)
    save_file(head.state_dict(), str(original / "joint_head.safetensors"))
    write_json(original / "joint_head_config.json", head_config)
    del backbone, head
    model, processor = module.load_release_model(original, device="mps", dtype=torch.bfloat16,
        quantization_config=quantization_config(), attn_implementation="eager", local_files_only=True)
    base = tmp_path / "tiny-compact-nf4-base"
    model.language_model.save_pretrained(base)
    processor.save_pretrained(base)
    for name in ("joint_head.safetensors", "joint_head_config.json"):
        shutil.copyfile(original / name, base / name)
    shutil.copyfile(module.__file__, base / "joint_schema_model.py")
    attach_local_lora(model, rank=2, last_layers=2)
    row = decision("actual-compact-native", "tiny architecture fixture")
    original_json = json.dumps(row, sort_keys=True)
    ordinary = module.encode_record(processor.tokenizer, row, processor=processor)
    compact, kept, audit = select_encoded_cohort([row], module, processor, 0, 42, 2048)
    assert len(compact) == 1 and kept == [row]
    compact = compact[0]
    assert isinstance(compact.input_ids, array) and compact.input_ids.typecode == "i"
    assert compact.input_ids.itemsize == 4 and tuple(compact.input_ids) == ordinary.input_ids
    assert compact.questions == ordinary.questions and compact.media == ordinary.media
    assert audit["input_token_bytes"] == len(ordinary.input_ids) * 4
    assert sys.getsizeof(compact.input_ids) < sys.getsizeof(ordinary.input_ids)
    assert json.dumps(row, sort_keys=True) == original_json
    ordinary_batch = module.collate_records([ordinary], processor.tokenizer.pad_token_id, torch.device("mps"))
    batch = module.collate_records([compact], processor.tokenizer.pad_token_id, torch.device("mps"))
    assert torch.equal(batch["input_ids"], ordinary_batch["input_ids"])
    assert torch.equal(batch["attention_mask"], ordinary_batch["attention_mask"])
    assert batch["input_ids"].dtype == torch.long
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(batch)[0][0], model(ordinary_batch)[0][0])
    initial = trainable_fingerprints(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, foreach=False)
    model.train()
    logits = model(batch)
    loss = supervised_loss(logits, [compact], [row])
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any("lora_" in name and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for name, parameter in model.named_parameters())
    assert any(name.startswith("head.") and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for name, parameter in model.named_parameters())
    optimizer.step()
    updated = trainable_fingerprints(model)
    assert all(initial[kind]["sha256"] != updated[kind]["sha256"] for kind in ("lora", "head"))
    model.eval()
    with torch.inference_mode():
        expected = model(batch)[0][0].float().cpu()
    adapter = tmp_path / "tiny-compact-trained-adapter"
    save_local_adapter(model, processor, module, base, adapter, training_metadata())
    del model, parameters, optimizer, logits, loss, ordinary_batch, batch
    gc.collect()
    torch.mps.empty_cache()
    restored, restored_processor = load_local_adapter(module, base, adapter, allow_unverified=True)
    restored_records, _, _ = select_encoded_cohort([row], module, restored_processor, 0, 42, 2048)
    restored_native = module.encode_record(restored_processor.tokenizer, row, processor=restored_processor)
    assert not any(tuple(left.input_ids) != tuple(right.input_ids)
                   for left, right in zip(restored_records, [restored_native], strict=True))
    restored_batch = module.collate_records(restored_records, restored_processor.tokenizer.pad_token_id, torch.device("mps"))
    with torch.inference_mode():
        observed = restored(restored_batch)[0][0].float().cpu()
    torch.testing.assert_close(observed, expected, rtol=1e-3, atol=1e-3)
    assert tuple(restored_records[0].input_ids) == ordinary.input_ids
    del restored, restored_batch
    gc.collect()
    torch.mps.empty_cache()


def test_mandatory_sources_and_longest_capacity_record_fit_checked_step_budget():
    module, processor = native_module(), processor_fixture()
    rows = [decision(f"live-{index}", "https://clawd-ws.fly.dev/", "live_tape_observation",
                     state=f"Observation number {index}.") for index in range(4)]
    rows += [decision("red", "https://arxiv.org/abs/2605.12151", state="RED paper evidence."),
             decision("hour", "https://arxiv.org/abs/2606.08232", state="Hour-aware paper evidence."),
             decision("capacity", "longest actual native encoding", state="Evidence " * 100),
             decision("other", "other source", state="A shorter observation.")]
    encoded, selected, audit = select_encoded_cohort(rows, module, processor, 0, 42, 2048, True)
    required = source_requirements(selected)
    assert len(required) == 6
    longest = max(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
    assert selected[longest]["id"] == "capacity" and longest not in required
    order = training_order(selected, 42, 0, encoded)
    assert order[:6] == required and order[6] == longest
    assert training_order(selected, 42, 1, encoded)[:7] == order[:7]
    assert len(set(order)) == len(selected)
    assert len(encoded[longest].input_ids) == audit["max_tokens"]
    with pytest.raises(ValueError, match="longest capacity probe"):
        require_training_budget(selected, encoded, 6, 1)
    assert require_training_budget(selected, encoded, 7, 1) == longest
    assert require_training_budget(selected, encoded, 4, 2) == longest
    with pytest.raises(ValueError, match="longest capacity probe"):
        require_training_budget(selected, encoded, 2, 3)
    with pytest.raises(ValueError, match="matching"):
        training_order(selected, 42, 0, encoded[:-1])

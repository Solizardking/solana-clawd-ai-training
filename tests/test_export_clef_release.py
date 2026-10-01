"""Actual tiny hybrid Qwen3.5, native Clef head, LoRA and processor export checks.

This is CPU model/serialization evidence, not a 27B production training run,
an official Decision Index result, or evidence of improved vision accuracy.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_clef_release import export_merged_release, load_release_model, verify_release


def load_local_upstream():
    path = Path(__file__).resolve().parents[1] / "local/clef-upstream/joint_schema_model.py"
    if not path.is_file():
        pytest.skip("Pinned local Clef upstream module is unavailable; no download attempted")
    spec = importlib.util.spec_from_file_location("clef_export_test_upstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_processor():
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor,
                              Qwen3VLProcessor, Qwen3VLVideoProcessor)
    special = ["<|endoftext|>", "<|image_pad|>", "<|video_pad|>",
               "<|vision_start|>", "<|vision_end|>"]
    vocabulary = {word: index for index, word in enumerate(special + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2TokenizerFast(vocab=vocabulary, merges=[], pad_token=special[0],
                                   unk_token=special[0], eos_token=special[0],
                                   additional_special_tokens=special[1:])
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=1),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=1, merge_size=1),
        tokenizer=tokenizer, chat_template="{{ messages }}",
    )


def tiny_model(module, processor):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from clef_research_training import attach_lora
    config = Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "vocab_size": len(processor.tokenizer), "layer_types": ["linear_attention", "full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                     "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "patch_size": 2,
                       "temporal_patch_size": 1, "spatial_merge_size": 1,
                       "num_position_embeddings": 16},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4,
    )
    model = module.ClefModel(Qwen3_5ForConditionalGeneration(config), module.JointSchemaHead(
        hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32))
    attach_lora(model, rank=2)
    return model


def typed_request():
    return {
        "id": "tiny-export-validation",
        "state": {"conversation": [{"role": "user", "content": "Liquidity is stale; inspect risk before action."}]},
        "questions": {
            "action": {"type": "choice", "question": "Choose action.", "criteria": {"wait": "Wait for fresh data.", "buy": "Trade now."}},
            "risk": {"type": "score", "question": "How much risk?", "criteria": ["Low risk", "High risk"]},
            "fresh": {"type": "noul", "question": "Is the data fresh?"},
        },
    }


def test_real_hybrid_model_merge_reloads_native_typed_logits_and_vision(tmp_path):
    import torch
    from safetensors.torch import load_file

    torch.manual_seed(7)
    torch.set_num_threads(2)
    module = load_local_upstream()
    processor = tiny_processor()
    model = tiny_model(module, processor)
    request = typed_request()
    record = module.encode_record(processor.tokenizer, request, processor=processor)
    batch = module.collate_records([record], processor.tokenizer.pad_token_id, torch.device("cpu"))
    original_vision = {key: value.detach().clone() for key, value in model.language_model.get_base_model().state_dict().items() if ".visual." in key}
    before_parameters = {key: value.detach().clone() for key, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    model.train()
    logits = model(batch)
    loss = sum(torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([0])) for scores in logits[0])
    loss.backward()
    optimizer.step()
    assert any("lora_" in key and not torch.equal(value, before_parameters[key]) for key, value in model.named_parameters() if key in before_parameters)
    assert any(key.startswith("head.") and not torch.equal(value, before_parameters[key]) for key, value in model.named_parameters() if key in before_parameters)
    model.eval()
    with torch.inference_mode():
        expected_logits = [scores.clone() for scores in model(batch)[0]]

    adapter_dir = tmp_path / "adapter"
    model.language_model.save_pretrained(adapter_dir)
    adapter_before = {path.name: path.read_bytes() for path in adapter_dir.iterdir() if path.is_file()}
    output = tmp_path / "merged"
    metadata = {"status": "trained_and_reload_verified", "optimizer_steps": 1,
                "merged_max_shard_size": "20KB", "test_scope": "tiny actual CPU model"}
    manifest = export_merged_release(model, processor, module, output, metadata)
    assert verify_release(output) == manifest
    assert len(manifest["backbone"]["shards"]) > 1 and manifest["backbone"]["index"]
    assert manifest["backbone"]["vision_tensor_count"] == len(original_vision)
    assert not (output / "adapter_config.json").exists()
    assert {path.name: path.read_bytes() for path in adapter_dir.iterdir() if path.is_file()} == adapter_before

    restored, restored_processor = load_release_model(output, device="cpu", dtype=torch.float32,
                                                      local_files_only=True, attn_implementation="sdpa")
    assert type(restored_processor).__name__ == "Qwen3VLProcessor"
    restored_record = module.encode_record(restored_processor.tokenizer, request, processor=restored_processor)
    assert restored_record.input_ids == record.input_ids
    restored_batch = module.collate_records([restored_record], restored_processor.tokenizer.pad_token_id, torch.device("cpu"))
    with torch.inference_mode():
        actual_logits = restored(restored_batch)[0]
    assert len(actual_logits) == len(expected_logits) == 3
    for before, after in zip(expected_logits, actual_logits, strict=True):
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    for key, before in original_vision.items():
        torch.testing.assert_close(restored.language_model.state_dict()[key], before, rtol=0, atol=0)
    for key, before in model.head.state_dict().items():
        torch.testing.assert_close(load_file(str(output / "joint_head.safetensors"))[key], before.detach().cpu(), rtol=0, atol=0)

    # The exported loader is itself part of the portable checkpoint.
    spec = importlib.util.spec_from_file_location("standalone_clef_loader_test", output / "export_clef_release.py")
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    assert standalone.verify_release(output)["standalone_backbone"]

    shard = output / manifest["backbone"]["shards"][0]
    with shard.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="exported hash"):
        verify_release(output)


def test_destination_and_training_guards_run_before_merging(tmp_path):
    from types import SimpleNamespace

    class NoMergeExpected:
        def merge_and_unload(self, **kwargs):
            raise AssertionError("Guard must run before merging")

    model = SimpleNamespace(language_model=NoMergeExpected())
    output = tmp_path / "already-saved-adapter"
    output.mkdir()
    (output / "adapter_config.json").write_text('{}')
    with pytest.raises(ValueError, match="separate empty"):
        export_merged_release(model, None, None, output, {"status": "trained_and_reload_verified", "optimizer_steps": 1})
    with pytest.raises(ValueError, match="completed training"):
        export_merged_release(model, None, None, tmp_path / "untrained", {"status": "training", "optimizer_steps": 1})
    with pytest.raises(ValueError, match="completed training"):
        export_merged_release(model, None, None, tmp_path / "zero-steps", {"status": "trained_and_reload_verified", "optimizer_steps": 0})
    assert not (tmp_path / "untrained").exists() and not (tmp_path / "zero-steps").exists()

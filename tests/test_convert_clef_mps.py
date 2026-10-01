"""Real tiny hybrid-model streaming quantization and standalone Metal reload."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_clef_mps import (GIB, MODEL_ID, MODEL_REVISION, check_resources, empty_backbone,
    estimate_backbone, finite_probe, require_fresh_output, save_quantized, sha256_file,
    stream_quantize, verify_probe, _validate_weight_map)


@pytest.fixture
def tiny_source(tmp_path):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 128, "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "out_hidden_size": 32, "patch_size": 2,
            "spatial_merge_size": 1, "num_position_embeddings": 16})
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    original = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).eval()
    source = tmp_path / "bf16"
    original.save_pretrained(source, max_shard_size="20KB")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in source.glob("*.safetensors")}
    return source, config, mapping, files, original


def require_mps():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("Actual Apple Metal required; this check does not claim CPU parity.")
    pytest.importorskip("bitsandbytes")


def test_real_bf16_shards_stream_to_mps_nf4_and_reload_identically(tiny_source, tmp_path):
    import torch
    import bitsandbytes as bnb
    from transformers import Qwen3_5ForConditionalGeneration
    require_mps()
    source, config, mapping, files, original = tiny_source
    model = empty_backbone(config)
    estimates = estimate_backbone(model)
    original_head = original.lm_head.weight.detach().clone()
    original_vision = {name: value.detach().clone() for name, value in original.model.visual.state_dict().items()}
    owned = tmp_path / "owned_downloads"
    owned.mkdir()
    progress = []
    def fetch(filename):
        assert not list(owned.glob("*.safetensors")), "only one original shard may be retained"
        destination = owned / filename
        shutil.copyfile(source / filename, destination)
        return destination
    model, stats = stream_quantize(model, mapping, fetch, files,
        remove_download=lambda path: path.unlink(), progress=progress.append)
    assert not list(owned.iterdir())
    assert len(progress) == len(files)
    assert sum(row["tensors"] for row in progress) == len(mapping)
    assert stats["quantized_linear_count"] == estimates["quantized_linear_count"] == 15
    assert stats["loaded_tensors"] == len(mapping)
    assert all(not value.is_meta for value in list(model.parameters()) + list(model.buffers()))
    assert torch.equal(model.lm_head.weight.cpu(), original_head)
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert all(torch.equal(value.cpu(), original_vision[name]) for name, value in model.model.visual.state_dict().items())
    quantized = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
    assert len(quantized) == 15
    assert all(module.weight.device.type == "mps" and module.weight.bnb_quantized for module in quantized)
    tokens = torch.tensor([[4, 2, 3, 7, 6, 8]], device="mps")
    expected = finite_probe(model, tokens)
    output = tmp_path / "standalone"
    output_files = save_quantized(model, output, max_shard_size="20KB")
    assert len(output_files) > 1
    assert all(sha256_file(output / name) == meta["sha256"] for name, meta in output_files.items())
    saved = json.loads((output / "config.json").read_text())["quantization_config"]
    assert saved["load_in_4bit"] is True
    assert saved["llm_int8_skip_modules"] == ["model.visual", "lm_head"]
    # Release the converting model before loading the standalone files.
    del model, quantized
    import gc
    gc.collect()
    torch.mps.empty_cache()
    restored = Qwen3_5ForConditionalGeneration.from_pretrained(output, device_map={"": "mps"},
        dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
    probe = verify_probe(expected, restored, tokens)
    assert probe["finite"] and probe["max_absolute_difference"] == 0
    assert torch.equal(restored.lm_head.weight.cpu(), original_head)
    assert all(torch.equal(value.cpu(), original_vision[name]) for name, value in restored.model.visual.state_dict().items())
    assert all(sha256_file(source / name) == meta["sha256"] for name, meta in files.items())


def test_shard_hash_failure_preserves_source_and_does_not_call_cleanup(tiny_source):
    require_mps()
    source, config, mapping, files, _ = tiny_source
    first = sorted(files)[0]
    files[first]["sha256"] = "0" * 64
    cleanup = []
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        stream_quantize(empty_backbone(config), mapping, lambda name: source / name, files,
                        remove_download=cleanup.append)
    assert cleanup == [] and (source / first).is_file()


def test_nontrivial_rotary_frequencies_remain_fp32_and_reload_bitwise(tiny_source, tmp_path):
    """BF16 construction must not round nonpersistent rotary frequencies."""
    from copy import deepcopy
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    require_mps()
    _, small_config, _, _, _ = tiny_source
    config = deepcopy(small_config)
    text = config.text_config
    text.hidden_size = 128
    text.intermediate_size = 256
    text.num_attention_heads = text.num_key_value_heads = 2
    text.head_dim = 64
    text.linear_key_head_dim = text.linear_value_head_dim = 64
    text.rope_parameters = {"rope_type": "default", "rope_theta": 10000000,
        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
        "mrope_section": [3, 3, 2]}
    torch.manual_seed(71)
    source = tmp_path / "nontrivial-bf16"
    original = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).eval()
    original.save_pretrained(source, max_shard_size="200KB")
    del original
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
             for path in source.glob("*.safetensors")}
    model = empty_backbone(config)
    for module in model.modules():
        if "RotaryEmbedding" in type(module).__name__:
            assert module.inv_freq.dtype == module.original_inv_freq.dtype == torch.float32
    model, _ = stream_quantize(model, mapping, lambda name: source / name, files)
    text_rotary = model.model.language_model.rotary_emb
    exact_frequencies, _ = text_rotary.compute_default_rope_parameters(text, device="mps")
    assert exact_frequencies.numel() == 8
    assert not torch.equal(exact_frequencies, exact_frequencies.to(torch.bfloat16).float())
    assert torch.equal(text_rotary.inv_freq, exact_frequencies)
    assert torch.equal(text_rotary.original_inv_freq, exact_frequencies)
    assert all(module.inv_freq.dtype == module.original_inv_freq.dtype == torch.float32
               for module in model.modules() if "RotaryEmbedding" in type(module).__name__)
    tokens = torch.arange(256, device="mps").remainder(124).add(1).unsqueeze(0)
    expected = finite_probe(model, tokens)
    output = tmp_path / "nontrivial-nf4"
    save_quantized(model, output, max_shard_size="200KB")
    del model, text_rotary
    import gc
    gc.collect()
    torch.mps.empty_cache()
    restored = Qwen3_5ForConditionalGeneration.from_pretrained(output, device_map={"": "mps"},
        dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
    assert torch.equal(restored.model.language_model.rotary_emb.inv_freq, exact_frequencies)
    probe = verify_probe(expected, restored, tokens)
    assert probe["finite"] and probe["max_absolute_difference"] == 0


def test_weight_map_missing_tensor_and_path_traversal_are_rejected(tiny_source):
    _, config, mapping, _, _ = tiny_source
    model = empty_backbone(config)
    missing = dict(mapping)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="Checkpoint keys differ"):
        _validate_weight_map(model, missing)
    bad_path = dict(mapping)
    bad_path[next(iter(bad_path))] = "../elsewhere.safetensors"
    with pytest.raises(ValueError, match="relative safetensors"):
        _validate_weight_map(model, bad_path)


def test_resource_guards_account_for_packed_model_shard_and_working_memory(tmp_path):
    estimates = {"packed_backbone_bytes": 20 * GIB, "sidecar_bytes": GIB,
                 "largest_source_shard_bytes": 5 * GIB, "largest_tensor_bytes": 3 * GIB}
    result = check_resources(tmp_path, estimates, available_disk=31 * GIB, available_mps=40 * GIB)
    assert result["disk_required_bytes"] == 29 * GIB
    assert result["mps_required_bytes"] == 27 * GIB
    with pytest.raises(RuntimeError, match="Insufficient free disk"):
        check_resources(tmp_path, estimates, available_disk=28 * GIB, available_mps=40 * GIB)
    with pytest.raises(RuntimeError, match="Insufficient recommended MPS"):
        check_resources(tmp_path, estimates, available_disk=31 * GIB, available_mps=26 * GIB)


def test_existing_outputs_are_preserved_and_source_identity_is_pinned(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    precious = output / "keep.txt"
    precious.write_text("user file")
    with pytest.raises(ValueError, match="new or empty"):
        require_fresh_output(output)
    assert precious.read_text() == "user file"
    assert MODEL_ID == "Cloudflare/clef" and len(MODEL_REVISION) == 40

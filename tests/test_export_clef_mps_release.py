"""Actual hybrid Qwen, native Clef head, MPS NF4 merge and hardlinked export."""
import gc
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_clef_mps import empty_backbone, save_quantized, sha256_file, stream_quantize
from export_clef_mps_release import (export_release, load_local_release, mark_reload_verified,
                                    validate_base, verify_mps_release, _fresh_destination)
from export_clef_release import verify_release
from test_export_clef_release import load_local_upstream, tiny_processor, typed_request


@pytest.fixture
def trained_fixture(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    if not torch.backends.mps.is_available():
        pytest.skip("Actual Apple Metal is required; no simulated quantization proof.")
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(71)
    module, processor = load_local_upstream(), tiny_processor()
    config = Qwen3_5Config(text_config={"hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "vocab_size": len(processor.tokenizer),
        "layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": 2,
        "linear_num_value_heads": 2, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 1, "num_position_embeddings": 16},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4)
    config._attn_implementation = "eager"
    original = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    source = tmp_path / "actual_tiny_bf16_fixture"
    original.save_pretrained(source, max_shard_size="20KB")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    source_files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in source.glob("*.safetensors")}
    del original
    quantized, _ = stream_quantize(empty_backbone(config), mapping, lambda name: source / name, source_files)
    base = tmp_path / "actual_tiny_nf4_base"
    base.mkdir()
    weights = save_quantized(quantized, base, "20KB")
    processor.save_pretrained(base)
    shutil.copyfile(module.__file__, base / "joint_schema_model.py")
    head_config = {"hidden_size": 32, "width": 16, "routing_layers": 1, "layers": 1,
                   "heads": 2, "feedforward": 32, "dropout": 0.0}
    head = module.JointSchemaHead(**head_config).to(device="mps", dtype=torch.bfloat16)
    save_file({key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
              str(base / "joint_head.safetensors"))
    (base / "joint_head_config.json").write_text(json.dumps(head_config))
    (base / "README.md").write_text("# Actual tiny random Qwen3.5 Metal fixture\n\nNative Clef code is from the pinned Cloudflare source; this fixture does not contain original 27B weights.\n")
    license_path = Path(__file__).resolve().parents[1] / "local/clef-27b-mps-nf4/LICENSE"
    shutil.copyfile(license_path, base / "LICENSE")
    expected_source = {"repo_id": "local/actual-tiny-Qwen3.5-Metal-fixture",
                       "revision": sha256_file(source / "config.json")}
    manifest = {"status": "complete", "reload_verified": True, "source": expected_source,
                "output_weight_files": weights, "preserved_sidecars": {}}
    (base / "conversion_manifest.json").write_text(json.dumps(manifest))
    # Actually reload this fixture before calling it a completed source.
    del quantized
    gc.collect()
    torch.mps.empty_cache()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(base, device_map={"": "mps"},
        dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager")
    targets = [name for name, layer in backbone.named_modules()
               if ".language_model.layers.1." in name and isinstance(layer, torch.nn.Linear)]
    peft = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                             target_modules=targets, bias="none"))
    model = module.ClefModel(peft, head)
    row = typed_request()
    encoded = module.encode_record(processor.tokenizer, row, processor=processor)
    batch = module.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("mps"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=0.0001)
    model.train()
    logits = model(batch)[0]
    loss = sum(torch.nn.functional.cross_entropy(scores.float().unsqueeze(0),
               torch.tensor([0], device="mps")) for scores in logits)
    loss.backward()
    optimizer.step()
    measured_loss = float(loss.detach().cpu())
    assert any("lora_" in name and not torch.equal(before[name], value)
               for name, value in model.named_parameters() if name in before)
    assert any(name.startswith("head.") and not torch.equal(before[name], value)
               for name, value in model.named_parameters() if name in before)
    model.eval()
    def probe(current):
        with torch.inference_mode():
            scores = current(batch)[0]
        answers = {question.question_id: {"probabilities": dict(zip(question.option_ids,
                   value.float().softmax(-1).cpu().tolist()))}
                   for question, value in zip(encoded.questions, scores, strict=True)}
        return [{"id": row["id"], "answers": answers}]
    # Save and really reload the trained adapter and head before export.
    adapter = tmp_path / "saved_adapter"
    peft.save_pretrained(adapter)
    save_file({key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
              str(adapter / "joint_head.safetensors"))
    from peft import PeftModel
    from safetensors.torch import load_file
    reference = probe(model)
    restored_backbone = Qwen3_5ForConditionalGeneration.from_pretrained(base, device_map={"": "mps"},
        dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager")
    restored = module.ClefModel(PeftModel.from_pretrained(restored_backbone, adapter),
                               module.JointSchemaHead(**head_config).to(device="mps", dtype=torch.bfloat16))
    restored.head.load_state_dict(load_file(adapter / "joint_head.safetensors"))
    restored.eval()
    from clef_research_training import compare_predictions
    compare_predictions(reference, probe(restored), tolerance=0.0001)
    return {"model": restored, "base": base, "expected_source": expected_source,
            "probe": probe, "processor": processor, "module": module, "row": row,
            "adapter": adapter, "source_files": source_files,
            "evaluation": {"scope": "actual tiny native architecture optimizer/probe fixture",
                           "optimizer_steps": 1, "loss_before_step": measured_loss,
                           "native_probe_records": 1, "native_probe_questions": 3}}


def test_real_nf4_merge_hardlinks_unchanged_shards_and_reloads_native_answers(trained_fixture, tmp_path):
    import torch
    fixture = trained_fixture
    base = fixture["base"]
    original = {path.name: sha256_file(path) for path in base.iterdir() if path.is_file()}
    adapter = {path.name: sha256_file(path) for path in fixture["adapter"].iterdir() if path.is_file()}
    output = tmp_path / "standalone_merged"
    manifest = export_release(fixture["model"], base, output,
        {"status": "trained_and_reload_verified", "optimizer_steps": 1,
         "scope": "actual tiny architecture MPS fixture, not the 27B checkpoint"},
        fixture["probe"], expected_source=fixture["expected_source"], reserve_bytes=0,
        evaluation=fixture["evaluation"])
    assert manifest["merge_verification"]["max_probability_difference"] <= 0.005
    assert manifest["storage"]["hardlinked_unchanged_shards"]
    assert manifest["storage"]["rewritten_shards"]
    for name in manifest["storage"]["hardlinked_unchanged_shards"]:
        assert (output / name).stat().st_ino == (base / name).stat().st_ino
    for name in manifest["storage"]["rewritten_shards"]:
        assert (output / name).stat().st_ino != (base / name).stat().st_ino
    assert not (output / "adapter_config.json").exists()
    required_aux = {"evaluation.json", "LICENSE", "source_base_model_card.md",
                    "run_clef_local.py", "export_clef_mps_release.py", "export_clef_release.py",
                    "clef_live_tape.py", "clef_research_data.py", "clef_research_training.py"}
    assert required_aux <= set(manifest["files"])
    assert (output / "source_base_model_card.md").read_bytes() == (base / "README.md").read_bytes()
    assert (output / "LICENSE").read_bytes() == (base / "LICENSE").read_bytes()
    assert json.loads((output / "evaluation.json").read_text()) == fixture["evaluation"]
    assert not any(path.is_symlink() for path in output.rglob("*"))
    assert {path.name: sha256_file(path) for path in base.iterdir() if path.is_file()} == original
    assert {path.name: sha256_file(path) for path in fixture["adapter"].iterdir() if path.is_file()} == adapter
    expected = fixture["probe"](fixture["model"])
    # Hardlinks are actual standalone bytes, not paths to an external base.
    shutil.rmtree(base)
    import subprocess
    help_result = subprocess.run([sys.executable, str(output / "run_clef_local.py"), "--help"],
                                 cwd=output, capture_output=True, text=True, timeout=15)
    assert help_result.returncode == 0, help_result.stderr
    assert "--model" in help_result.stdout
    model, processor = load_local_release(output)
    encoded = fixture["module"].encode_record(processor.tokenizer, fixture["row"], processor=processor)
    batch = fixture["module"].collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("mps"))
    with torch.inference_mode():
        scores = model(batch)[0]
    observed = [{"id": fixture["row"]["id"], "answers": {
        question.question_id: {"probabilities": dict(zip(question.option_ids, value.float().softmax(-1).cpu().tolist()))}
        for question, value in zip(encoded.questions, scores, strict=True)}}]
    final = mark_reload_verified(output, expected, observed)
    assert final["reload_verified"] and final["reload_verification"]["max_probability_difference"] <= 0.0001
    training = json.loads((output / "training.json").read_text())
    assert training["status"] == "trained_and_standalone_reload_verified"
    assert final["training"] == training
    assert final["files"]["training.json"]["sha256"] == sha256_file(output / "training.json")
    assert verify_release(output) == final
    with pytest.raises(ValueError, match="pinned genuine Clef"):
        verify_mps_release(output)
    assert model.language_model.lm_head.weight.dtype == torch.bfloat16
    assert all(value.dtype == torch.bfloat16 for value in model.language_model.model.visual.parameters())


def test_base_tampering_and_incomplete_conversion_are_rejected(trained_fixture):
    fixture = trained_fixture
    path = fixture["base"] / "conversion_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "loading"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        validate_base(fixture["base"], fixture["expected_source"])
    manifest["status"] = "complete"
    first = next(iter(manifest["output_weight_files"]))
    manifest["output_weight_files"][first]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Immutable base hash"):
        validate_base(fixture["base"], fixture["expected_source"])


def test_existing_or_nested_destinations_and_remote_paths_are_rejected(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "precious.txt").write_text("user content")
    with pytest.raises(ValueError, match="new or empty"):
        _fresh_destination(base, output)
    assert (output / "precious.txt").read_text() == "user content"
    with pytest.raises(ValueError, match="not nested"):
        _fresh_destination(base, base / "nested")
    with pytest.raises(ValueError, match="downloads are disabled"):
        load_local_release(tmp_path / "missing")


def test_reload_verification_rejects_excessive_tolerance_and_drift(tmp_path):
    with pytest.raises(ValueError, match="no greater than"):
        mark_reload_verified(tmp_path, [], [], tolerance=0.001)

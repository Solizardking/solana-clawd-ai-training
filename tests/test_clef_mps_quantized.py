"""Actual Mac MPS NF4 kernels and native Clef/LoRA training-reload evidence.

Run explicitly in the Mac inference environment. Backend/operation failures
fail these tests; there is no CPU fallback, xfail, or successful mock path.
Tiny ASCII fixtures do not prove 27B memory capacity or domain improvement.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def native_module():
    path = Path(__file__).resolve().parents[1] / "local/clef-upstream/joint_schema_model.py"
    if not path.is_file():
        raise RuntimeError("Pinned local Clef code is required; no weights or code download attempted")
    spec = importlib.util.spec_from_file_location("clef_actual_mps_test_upstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def processor_fixture():
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor,
                              Qwen3VLProcessor, Qwen3VLVideoProcessor)
    special = ["<|endoftext|>", "<|image_pad|>", "<|video_pad|>",
               "<|vision_start|>", "<|vision_end|>"]
    # 256 total entries; this byte-level fixture covers all ASCII used below.
    # It is a genuine serializable Qwen tokenizer, not Clef's production vocab.
    vocabulary = {word: index for index, word in enumerate(special + sorted(ByteLevel.alphabet())[:251])}
    tokenizer = Qwen2TokenizerFast(vocab=vocabulary, merges=[], pad_token=special[0],
                                   unk_token=special[0], eos_token=special[0],
                                   additional_special_tokens=special[1:])
    assert len(tokenizer) == 256
    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=1),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=1, merge_size=1),
        tokenizer=tokenizer, chat_template="{{ messages }}",
    )


def require_mps():
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("This validation requires actual MPS hardware")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for honest backend validation")
    return torch


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", "model.visual"],
    )


def test_actual_mps_nf4_double_quant_kernels():
    torch = require_mps()
    import bitsandbytes as bnb

    source = torch.linspace(-1, 1, 256, device="mps", dtype=torch.bfloat16).reshape(16, 16)
    compressed, state = bnb.functional.quantize_4bit(source, quant_type="nf4", compress_statistics=True)
    restored = bnb.functional.dequantize_4bit(compressed, quant_state=state)
    torch.mps.synchronize()
    assert compressed.device.type == restored.device.type == "mps"
    assert state.nested and state.quant_type == "nf4"
    assert restored.shape == source.shape and bool(torch.isfinite(restored).all())
    assert float((source - restored).abs().max()) < 0.25


def test_native_clef_mps_qlora_backprop_and_saved_weights_reload(tmp_path):
    torch = require_mps()
    import bitsandbytes as bnb
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from clef_research_training import attach_lora, supervised_loss

    torch.manual_seed(17)
    module = native_module()
    processor = processor_fixture()
    config = Qwen3_5Config(
        text_config={"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 64,
                     "vocab_size": 256, "layer_types": ["linear_attention", "full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                     "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 128, "intermediate_size": 256,
                       "num_heads": 2, "out_hidden_size": 128, "patch_size": 2,
                       "temporal_patch_size": 1, "spatial_merge_size": 1,
                       "num_position_embeddings": 16},
        image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4,
    )
    base = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    original_vision = {key: value.detach().clone() for key, value in base.state_dict().items() if ".visual." in key}
    base_path = tmp_path / "original-bf16-backbone"
    base.save_pretrained(base_path, safe_serialization=True)
    processor.save_pretrained(base_path)
    head_config = {"hidden_size": 128, "width": 64, "routing_layers": 1,
                   "layers": 1, "heads": 2, "feedforward": 128}
    head = module.JointSchemaHead(**head_config).to(dtype=torch.bfloat16)
    save_file(head.state_dict(), str(base_path / "joint_head.safetensors"))
    (base_path / "joint_head_config.json").write_text(json.dumps(head_config))
    del base, head

    model, processor = module.load_release_model(
        base_path, device="mps", dtype=torch.bfloat16, quantization_config=quantization_config(),
        attn_implementation="sdpa", local_files_only=True,
    )
    quantized = [child for child in model.language_model.modules() if isinstance(child, bnb.nn.Linear4bit)]
    assert quantized and all(child.weight.device.type == "mps" for child in quantized)
    assert all(child.weight.quant_state.quant_type == "nf4" and child.weight.quant_state.nested for child in quantized)
    assert not isinstance(model.language_model.get_output_embeddings().weight, bnb.nn.Params4bit)
    assert model.language_model.get_output_embeddings().weight.dtype == torch.bfloat16
    assert not any(isinstance(child, bnb.nn.Linear4bit) for child in model.language_model.model.visual.modules())
    targets = attach_lora(model, rank=2)
    assert targets and not any("visual" in name or "lm_head" in name for name in targets)
    assert all(parameter.device.type == "mps" for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.get_base_model().model.visual.parameters())
    row = {
        "id": "actual-mps-qlora-check",
        "state": "The source reports stale liquidity. No signing or execution is available.",
        "questions": {"research_answer": {"type": "choice", "instructions": "Select the source-grounded response.",
                                            "criteria": {"buy": "Trade immediately.", "wait": "Wait for fresh liquidity evidence."}}},
        "labels": {"research_answer": "wait"},
    }
    record = module.encode_record(processor.tokenizer, row, processor=processor)
    batch = module.collate_records([record], processor.tokenizer.pad_token_id, torch.device("mps"))
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, foreach=False)
    model.train()
    logits = model(batch)
    loss = supervised_loss(logits, [record], [row])
    assert loss.device.type == "mps" and bool(torch.isfinite(loss))
    loss.backward()
    gradient_evidence = {
        "lora": any("lora_" in name and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                    for name, parameter in model.named_parameters()),
        "head": any(name.startswith("head.") and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                    for name, parameter in model.named_parameters()),
    }
    assert gradient_evidence == {"lora": True, "head": True}
    optimizer.step()
    changed = {name for name, value in model.named_parameters() if name in before and not torch.equal(before[name], value.detach().cpu())}
    assert any("lora_" in name for name in changed) and any(name.startswith("head.") for name in changed)
    for key, value in model.language_model.get_base_model().state_dict().items():
        if key in original_vision:
            torch.testing.assert_close(value.cpu(), original_vision[key], rtol=0, atol=0)

    model.eval()
    with torch.inference_mode():
        expected = model(batch)[0][0].float().cpu()
    adapter_path = tmp_path / "trained-adapter-and-head"
    model.language_model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()},
              str(adapter_path / "joint_head.safetensors"))
    # Release MPS training state before loading a fresh quantized backbone.
    del quantized, parameters, optimizer, model, logits, loss
    import gc
    gc.collect()
    torch.mps.empty_cache()
    restored, restored_processor = module.load_release_model(
        base_path, device="mps", dtype=torch.bfloat16, quantization_config=quantization_config(),
        attn_implementation="sdpa", local_files_only=True,
    )
    restored.language_model = PeftModel.from_pretrained(restored.language_model, adapter_path)
    restored.head.load_state_dict(load_file(str(adapter_path / "joint_head.safetensors")), strict=True)
    restored.eval()
    restored_record = module.encode_record(restored_processor.tokenizer, row, processor=restored_processor)
    assert restored_record.input_ids == record.input_ids
    restored_batch = module.collate_records([restored_record], restored_processor.tokenizer.pad_token_id, torch.device("mps"))
    with torch.inference_mode():
        observed = restored(restored_batch)[0][0].float().cpu()
    torch.testing.assert_close(observed, expected, rtol=1e-3, atol=1e-3)
    assert int(observed.argmax()) == int(expected.argmax())
    assert bool(torch.isfinite(observed).all())
    assert any(isinstance(child, bnb.nn.Linear4bit) for child in restored.language_model.modules())
    for key, value in restored.language_model.get_base_model().state_dict().items():
        if key in original_vision:
            torch.testing.assert_close(value.cpu(), original_vision[key], rtol=0, atol=0)

#!/usr/bin/env python3
"""Continue the verified Clef pilot on CUDA from an immutable staged NF4 base.

No Hub downloads, uploads, or GPU-job submission occur in this runner. The
staging verifier establishes the original model, dataset, privacy, and file
identities before weights are loaded. A native 16-record migration comparison
must pass before any update. Full mode trains one complete selected epoch.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import replace
import gc
import hashlib
import json
import math
import re
from pathlib import Path
import random
import shutil
import time
import uuid

from clef_research_data import validate_decision_record
from clef_research_training import compare_predictions, evaluate, supervised_loss, trainable_fingerprints
from train_clef_local import (
    DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256, INPUT_RAW_ROW_COUNTS,
    MODEL_ID, MODEL_REVISION, _checkpoint_without_input_grads, _cpu_state,
    finite_trainables, guard_model_input, load_local_module, refresh_adapter_manifest,
    restored_training_fingerprints, save_local_adapter, verify_adapter_manifest,
)

MANIFEST = "cloud_migration_manifest.json"
MIGRATION_TOLERANCE = 1e-3
GIB = 1024 ** 3
RUNTIME_FILES = ("train_clef_cuda.py", "train_clef_local.py", "clef_research_data.py",
                 "clef_research_training.py", "clef_live_tape.py", "research_expansion_artifacts.py")
KERNEL_EVIDENCE = None


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = set()
    for row in rows:
        validate_decision_record(row)
        guard_model_input(row)
        if row["id"] in ids:
            raise ValueError("Native cohort record IDs must be unique")
        ids.add(row["id"])
    if not rows:
        raise ValueError("Native cohort is empty")
    return rows


def stage_path(stage, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe staged artifact reference")
    path = Path(stage) / relative
    if path.is_symlink() or not path.exists() or not path.resolve().is_relative_to(Path(stage).resolve()):
        raise ValueError("Staged artifact is absent or escapes its verified directory")
    return path


def encode_complete(module, processor, rows, max_length):
    """Encode every selected row without truncation or a silent exclusion."""
    encoded = []
    for row in rows:
        guard_model_input(row)
        record = module.encode_record(processor.tokenizer, row, max_length=1_000_000, processor=processor)
        if not 0 < len(record.input_ids) <= max_length:
            raise ValueError("A selected native record exceeds its staged context cap")
        if any(type(token) is not int or not 0 <= token <= 2 ** 31 - 1 for token in record.input_ids):
            raise ValueError("Native tokenizer produced an invalid signed-32-bit token ID")
        compact = array("i", record.input_ids)
        if compact.itemsize != 4:
            raise RuntimeError("Compact native tokens require four-byte signed integers")
        encoded.append(replace(record, input_ids=compact))
    return encoded


def required_indices(rows):
    return [index for index, row in enumerate(rows)
            if row.get("provenance", {}).get("source_type") == "live_tape_observation"
            or any(paper in str(row.get("provenance", {}).get("source", ""))
                   for paper in ("2605.12151", "2606.08232"))]


def epoch_order(rows, encoded, seed):
    required = required_indices(rows)
    longest = max(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
    leading = required + ([] if longest in required else [longest])
    leading_set = set(leading)
    remainder = [index for index in range(len(rows)) if index not in leading_set]
    random.Random(seed).shuffle(remainder)
    order = leading + remainder
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise ValueError("Training order does not cover each selected record once")
    return order, required, longest


def record_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def step_groups(order, microbatch, accumulation, steps=0):
    width = microbatch * accumulation
    count = math.ceil(len(order) / width)
    if steps:
        count = min(count, steps)
    return [order[index * width:(index + 1) * width] for index in range(count)]


def verify_coverage(order, trained, cursor, complete):
    if len(trained) != cursor or len(set(trained)) != len(trained) or trained != order[:cursor]:
        raise ValueError("Actual training coverage differs from the deterministic epoch cursor")
    if complete and (cursor != len(order) or set(trained) != set(order)):
        raise ValueError("Full training did not cover every selected native record exactly once")


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required; this runner never falls back to CPU or downloads weights")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    return {"device": torch.cuda.get_device_name(0), "free_bytes": free, "total_bytes": total,
            "compute_capability": list(torch.cuda.get_device_capability(0))}


def configure_local_kernels(enabled):
    """Use pinned installed FLA directly; never request kernels from the Hub.

    Transformers' optional kernel decorator can download Hub code. This bridge
    instead installs the local 0.5.2 implementation and counts real CUDA calls.
    The native depthwise convolution remains the explicit PyTorch fallback.
    """
    global KERNEL_EVIDENCE
    from importlib.metadata import version
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    reference = qwen.torch_chunk_gated_delta_rule
    reference = getattr(reference, "_clef_reference", reference)
    reference = getattr(reference, "__wrapped__", reference)
    evidence = {"implementation": "installed_fla" if enabled else "native_torch",
                "fla_core_version": None, "hub_kernel_downloads": False,
                "cuda_forward_calls": 0, "cuda_backward_calls": 0,
                "depthwise_convolution": "native_torch"}
    kernel = reference
    if enabled:
        if version("fla-core") != "0.5.2":
            raise ValueError("The local CUDA FLA bridge requires pinned fla-core==0.5.2")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        kernel = chunk_gated_delta_rule
        evidence["fla_core_version"] = "0.5.2"
    def bridge(query, key, value, **kwargs):
        if any(tensor.device.type != "cuda" for tensor in (query, key, value)):
            raise ValueError("The CUDA gated-delta bridge cannot use a CPU fallback")
        result = kernel(query, key, value, **kwargs)
        evidence["cuda_forward_calls"] += 1
        if result[0].requires_grad:
            def backward(gradient):
                evidence["cuda_backward_calls"] += 1
                return gradient
            result[0].register_hook(backward)
        return result
    bridge._clef_reference = reference
    qwen.torch_chunk_gated_delta_rule = bridge
    # Bypass the optional Hub decorator for the convolution fallback as well.
    qwen.causal_conv1d_fn = getattr(qwen.causal_conv1d_fn, "__wrapped__", qwen.causal_conv1d_fn)
    qwen.causal_conv1d_update = getattr(qwen.causal_conv1d_update, "__wrapped__", qwen.causal_conv1d_update)
    KERNEL_EVIDENCE = evidence
    return evidence


def validate_cuda_model(model, *, trainable, expected_fingerprints):
    import bitsandbytes as bnb
    import torch
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError("Native CUDA model parameters may not be offloaded")
    backbone = model.language_model
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    quantized = [child for child in base.modules() if isinstance(child, bnb.nn.Linear4bit)]
    if not quantized or any(child.weight.quant_state is None or
                           child.weight.quant_state.quant_type != "nf4" or
                           not child.weight.quant_state.nested for child in quantized):
        raise ValueError("CUDA base must retain the staged NF4/double-quant weights")
    if any(isinstance(child, bnb.nn.Linear4bit) for child in base.model.visual.modules()):
        raise ValueError("The original vision encoder must not be quantized")
    if any(parameter.dtype != torch.bfloat16 or parameter.requires_grad
           for parameter in base.model.visual.parameters()):
        raise ValueError("The original vision encoder must remain frozen BF16")
    output = base.get_output_embeddings().weight
    if isinstance(output, bnb.nn.Params4bit) or output.dtype != torch.bfloat16 or output.requires_grad:
        raise ValueError("Native output embedding must remain readable frozen BF16")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.head.parameters()):
        raise ValueError("The native trained head must remain BF16")
    config = backbone.peft_config["default"]
    layers = sorted({int(part.split(".")[0]) for target in config.target_modules
                     if ".language_model.layers." in target
                     for part in [target.split(".language_model.layers.", 1)[1]]})
    count = len(base.model.language_model.layers)
    if config.r != 8 or layers != list(range(count - 4, count)) or any(
        ".language_model.layers." not in target for target in config.target_modules
    ):
        raise ValueError("The imported pilot must have rank-eight LoRA on only the final four text layers")
    for name, parameter in model.named_parameters():
        allowed = "lora_" in name or name.startswith("head.")
        if parameter.requires_grad and not allowed:
            raise ValueError("CUDA training accidentally unfroze an original backbone parameter")
    for child in quantized:
        state = child.weight.quant_state
        if any(tensor.device.type != "cuda" for tensor in
               (state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code)):
            raise ValueError("NF4 quantization state may not be offloaded")
    if restored_training_fingerprints(model) != expected_fingerprints:
        raise ValueError("Imported native head or LoRA bytes differ from the verified checkpoint")
    if trainable:
        finite_trainables(model)
        if hasattr(backbone, "_require_grads_hook"):
            raise ValueError("Frozen input embeddings still request gradients")
    return {"rank": config.r, "text_layers": layers, "target_modules": sorted(config.target_modules),
            "gradient_checkpointing": "nonreentrant", "frozen_input_gradients": False}


def load_cuda_adapter(module, base, adapter, *, trainable=False, attn_implementation="eager", use_kernels=False):
    """Only local staged NF4 weights; load both actual trained paths."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    require_cuda()
    base, adapter = Path(base), Path(adapter)
    if not base.is_dir() or not adapter.is_dir():
        raise ValueError("CUDA continuation requires existing local staged weights")
    verify_adapter_manifest(adapter)
    metadata = json.loads((adapter / "training.json").read_text())
    if metadata.get("status") not in {"trained_and_reload_verified", "training", "trained_pending_reload_verification"}:
        raise ValueError("Adapter metadata is not a saved native training checkpoint")
    if (metadata.get("model") != MODEL_ID or metadata.get("dataset") != DATASET_ID or
        metadata.get("model_revision") != MODEL_REVISION or metadata.get("dataset_revision") != DATASET_REVISION):
        raise ValueError("CUDA adapter uses different immutable sources")
    if (metadata.get("input_parquet_sha256") != INPUT_PARQUET_SHA256 or
        metadata.get("input_raw_row_counts") != INPUT_RAW_ROW_COUNTS or
        metadata.get("security_filter", {}).get("applied") is not True or
        metadata["security_filter"].get("version") != 1 or metadata.get("optimizer_steps", 0) < 1):
        raise ValueError("CUDA adapter lacks verified source hashes and model-input security evidence")
    if metadata.get("local_base_manifest_sha256") != file_sha(base / "conversion_manifest.json"):
        raise ValueError("CUDA adapter belongs to a different converted NF4 base")
    generated = metadata.get("local_base_generated_files", {})
    if not {"config.json", "model.safetensors.index.json"} <= set(generated):
        raise ValueError("CUDA adapter lacks generated base config/index hash evidence")
    for name, expected in generated.items():
        path = stage_path(base, name)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
            raise ValueError("Converted base configuration changed before CUDA load")
    kwargs = {"attn_implementation": attn_implementation, "local_files_only": True}
    if KERNEL_EVIDENCE is None or (KERNEL_EVIDENCE["implementation"] == "installed_fla") != use_kernels:
        configure_local_kernels(use_kernels)
    model, processor = module.load_release_model(base, device="cuda", dtype=torch.bfloat16, **kwargs)
    model.language_model = PeftModel.from_pretrained(model.language_model, adapter, is_trainable=trainable)
    model.head.load_state_dict(load_file(str(adapter / "joint_head.safetensors")), strict=True)
    model.head.requires_grad_(trainable)
    if trainable:
        _checkpoint_without_input_grads(model.language_model)
    validate_cuda_model(model, trainable=trainable, expected_fingerprints=metadata["trainable_fingerprints"])
    return model.eval(), processor


def check_migration(expected, actual):
    if len(expected) != 16 or len(actual) != 16 or len({item["id"] for item in expected}) != 16:
        raise ValueError("CUDA migration requires all sixteen distinct saved pilot predictions")
    for before, after in zip(expected, actual, strict=True):
        if before.get("answers", {}).keys() != after.get("answers", {}).keys() or any(
            answer.get("choice") != after["answers"][key].get("choice") or
            answer.get("label") != after["answers"][key].get("label")
            for key, answer in before.get("answers", {}).items()):
            raise ValueError("CUDA migration changed a native pilot choice or label")
    return compare_predictions(expected, actual, tolerance=MIGRATION_TOLERANCE)


def batch_update(model, module, processor, encoded, rows, indices, optimizer, microbatch, device):
    """Average genuine native choice CE by question, including a partial tail."""
    import torch
    optimizer.zero_grad(set_to_none=True)
    total_loss, padded_tokens = 0.0, 0
    questions = sum(len(encoded[index].questions) for index in indices)
    if not questions or not indices:
        raise ValueError("An optimizer update requires actual supervised native questions")
    for start in range(0, len(indices), microbatch):
        selected = indices[start:start + microbatch]
        batch_records, batch_rows = [encoded[index] for index in selected], [rows[index] for index in selected]
        for row in batch_rows:
            guard_model_input(row)
        logits = model(module.collate_records(batch_records, processor.tokenizer.pad_token_id, device))
        loss = supervised_loss(logits, batch_records, batch_rows)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Native CUDA loss is nonfinite")
        fraction = sum(len(record.questions) for record in batch_records) / questions
        (loss * fraction).backward()
        total_loss += float(loss.detach()) * fraction
        padded_tokens += len(selected) * max(len(record.input_ids) for record in batch_records)
    gradients = {kind: False for kind in ("lora", "head")}
    parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameters.append(parameter)
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("Native CUDA gradient is nonfinite")
        kind = "lora" if "lora_" in name else "head" if name.startswith("head.") else None
        if kind and bool(parameter.grad.abs().sum() > 0):
            gradients[kind] = True
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total_loss, gradients, padded_tokens


def save_artifact(model, processor, module, base, output, metadata, snapshot, evaluation=None):
    metadata = save_local_adapter(model, processor, module, base, output, metadata, training_snapshot=snapshot)
    for name in RUNTIME_FILES:
        shutil.copyfile(Path(__file__).with_name(name), Path(output) / name)
    if evaluation is not None:
        atomic_json(Path(output) / "evaluation.json", evaluation)
    (Path(output) / "README.md").write_text(
        "# Clawd Clef CUDA native decision adapter\n\n"
        f"Run mode: {metadata['run_mode']}; optimizer steps: {metadata['optimizer_steps']}. "
        "The original NF4 Clef backbone, rank-eight final-four-layer LoRA and native BF16 joint head "
        "use the immutable staged research cohort. Vision and output embeddings stay frozen BF16. "
        "Use train_clef_cuda.load_cuda_adapter with the exact converted base. This adapter alone is not "
        "a standalone backbone. See training.json for actual coverage and reload status. No signing or trading permissions.\n\n"
        "Kamat, A. U. (2026). RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events "
        "with Outcome Labels. https://arxiv.org/abs/2605.12151\n\n"
        "Kamat, A. U. (2026). Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading. "
        "https://arxiv.org/abs/2606.08232\n"
    )
    refresh_adapter_manifest(output)
    return metadata


def resume_contract(metadata):
    return {key: metadata[key] for key in ("run_id", "run_mode", "stage_manifest_sha256", "model_revision",
        "dataset_revision", "max_length", "microbatch_size", "gradient_accumulation", "seed", "learning_rate",
        "training_order_sha256", "planned_steps", "attention_implementation", "use_kernels", "runtime_source_sha256")}


def validate_resume(saved, requested, state, order):
    if resume_contract(saved) != resume_contract(requested):
        raise ValueError("Resume checkpoint changes the exact training/source/order contract")
    cursor, steps = state.get("cursor"), state.get("optimizer_steps")
    if type(cursor) is not int or type(steps) is not int or not 0 <= steps <= requested["planned_steps"]:
        raise ValueError("Resume checkpoint has an invalid optimizer cursor")
    verify_coverage(order, state.get("trained_indices", []), cursor, complete=False)
    width = requested["microbatch_size"] * requested["gradient_accumulation"]
    if cursor != min(steps * width, len(order)) or saved.get("optimizer_steps") != steps:
        raise ValueError("Resume checkpoint coverage and completed optimizer steps disagree")


def checkpoint(model, processor, module, base, output, metadata, snapshot, optimizer, cursor, trained):
    import torch
    directory = Path(output) / "checkpoints" / f"step-{metadata['optimizer_steps']:06d}"
    if directory.exists():
        raise ValueError("A completed optimizer checkpoint must not be overwritten")
    saved = save_artifact(model, processor, module, base, directory, metadata, snapshot)
    state = {"optimizer": _cpu_state(optimizer.state_dict()), "optimizer_steps": metadata["optimizer_steps"],
             "cursor": cursor, "trained_indices": trained, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    torch.save(state, directory / "optimizer.pt")
    atomic_json(directory / "resume.json", {"contract": resume_contract(saved), "cursor": cursor,
                                            "optimizer_steps": metadata["optimizer_steps"]})
    files = refresh_adapter_manifest(directory)
    verify_adapter_manifest(directory)
    atomic_json(directory / "adapter_artifact_manifest.json", {"files": files, "checkpoint_complete": True,
        "run_id": metadata["run_id"], "optimizer_steps": metadata["optimizer_steps"], "cursor": cursor,
        "resume_contract": resume_contract(saved)})
    return directory


def prune_checkpoints(output, owned, keep):
    root = Path(output).resolve()
    while len(owned) > keep:
        path = owned.pop(0)
        if path.is_symlink() or path.resolve().parent != root / "checkpoints" or not re.fullmatch(r"step-[0-9]{6}", path.name):
            raise ValueError("Refusing to prune a checkpoint outside this run")
        if verify_adapter_manifest(path).get("checkpoint_complete") is not True:
            raise ValueError("Refusing to prune an incomplete optimizer checkpoint")
        shutil.rmtree(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "full"), default="benchmark")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--benchmark-steps", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=int, default=128)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export-release", type=Path)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--use-kernels", action="store_true", help="Use installed fla-core==0.5.2 directly; no Hub kernel download")
    parser.add_argument("--validate-only", action="store_true", help="Verify staged inputs without loading model weights")
    args = parser.parse_args()
    if min(args.microbatch, args.gradient_accumulation, args.max_length, args.benchmark_steps,
           args.checkpoint_steps, args.keep_checkpoints) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Training sizes, checkpoint intervals and learning rate must be positive")
    if args.max_length != 2048:
        parser.error("The immutable selected cohort and migration contract require context2048")
    if args.mode == "benchmark" and args.benchmark_steps != 16:
        parser.error("CUDA migration benchmark must have exactly sixteen optimizer steps")
    if args.mode != "full" and args.export_release:
        parser.error("Standalone export requires complete full mode")
    return args


def run(args):
    run_started = time.monotonic()
    from stage_clef_cloud_migration import verify_stage
    stage = args.stage.resolve()
    manifest = verify_stage(stage)
    paths = manifest["paths"]
    base, pilot = stage_path(stage, paths["base"]), stage_path(stage, paths["pilot"])
    audits = json.loads(stage_path(stage, paths["full_data_manifest"]).read_text())
    prepared, cohort_audits = audits["prepared"], audits["cohorts"]
    rows = {split: read_rows(stage_path(stage, paths["cohorts"][split])) for split in ("train", "eval", "test")}
    reference_rows = [row for split in ("eval", "test")
                      for row in read_rows(stage_path(stage, paths["pilot_cohorts"][split]))]
    expected_predictions = []
    for split in ("eval", "test"):
        with stage_path(stage, paths["pilot_predictions"][split]).open() as handle:
            expected_predictions.extend(json.loads(line) for line in handle if line.strip())
    if [row["id"] for row in reference_rows] != [prediction["id"] for prediction in expected_predictions]:
        raise ValueError("Saved pilot predictions do not describe the staged native migration inputs")
    if args.validate_only:
        return {"status": "validated_without_weights", "stage_manifest_sha256": file_sha(stage / MANIFEST),
                "selected_records": {split: len(values) for split, values in rows.items()}, "migration_records": len(reference_rows)}
    resources = require_cuda()
    import torch
    from transformers import AutoProcessor
    output = args.output.resolve()
    if output == stage or output.is_relative_to(stage) or stage.is_relative_to(output):
        raise ValueError("CUDA output must be separate from the immutable staging directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("CUDA output must be a fresh empty directory, including on checkpoint resume")
    if args.export_release:
        release = args.export_release.resolve()
        if release == stage or release.is_relative_to(stage) or stage.is_relative_to(release) or release == output:
            raise ValueError("Standalone destination must be separate from staged inputs and adapter output")
        if release.exists() and (not release.is_dir() or any(release.iterdir())):
            raise ValueError("Standalone CUDA destination must be fresh and empty")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    module = load_local_module(base)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    preparation_started = time.monotonic()
    encoded = {split: encode_complete(module, processor, values, args.max_length) for split, values in rows.items()}
    for split in rows:
        count = sum(len(record.input_ids) * record.input_ids.itemsize for record in encoded[split])
        if count != cohort_audits[split]["input_token_bytes"]:
            raise ValueError("Native CUDA tokenization changed the immutable selected cohort")
    reference_encoded = encode_complete(module, processor, reference_rows, args.max_length)
    order, required, longest = epoch_order(rows["train"], encoded["train"], args.seed)
    groups = step_groups(order, args.microbatch, args.gradient_accumulation,
                         args.benchmark_steps if args.mode == "benchmark" else 0)
    budget_ids = {index for group in groups for index in group}
    if not set(required + [longest]) <= budget_ids or (args.mode == "benchmark" and len(groups) != 16):
        raise ValueError("CUDA benchmark budget omits a required paper/live/longest input")
    preparation_seconds = time.monotonic() - preparation_started
    # Migration always uses the actual original pilot, even on optimizer resume.
    load_started = time.monotonic()
    model, processor = load_cuda_adapter(module, base, pilot, trainable=True,
        attn_implementation=args.attn_implementation, use_kernels=args.use_kernels)
    load_seconds = time.monotonic() - load_started
    migration_started = time.monotonic()
    _, migrated_predictions = evaluate(model, module, processor, reference_encoded, reference_rows, torch.device("cuda"))
    migration = check_migration(expected_predictions, migrated_predictions)
    migration_seconds = time.monotonic() - migration_started
    pilot_metadata = json.loads((pilot / "training.json").read_text())
    metadata = {"status": "training", "run_id": uuid.uuid4().hex, "run_mode": args.mode,
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "input_parquet_sha256": INPUT_PARQUET_SHA256,
        "input_raw_row_counts": INPUT_RAW_ROW_COUNTS, "security_filter": prepared["security_filter"],
        "prepared_split_counts": {split: prepared["outputs"][split]["rows"] for split in rows},
        "stage_manifest_sha256": file_sha(stage / MANIFEST), "cohorts": cohort_audits,
        "runtime_source_sha256": {name: file_sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES},
        "local_base_manifest_sha256": file_sha(base / "conversion_manifest.json"),
        "local_base_generated_files": pilot_metadata.get("local_base_generated_files", {}),
        "migration_verification": migration, "pilot_trainable_fingerprints": pilot_metadata["trainable_fingerprints"],
        "backend": "cuda", "device": "cuda", "cpu_offload": False, "cpu_fallback": False,
        "local_backend": "CUDA NF4 double-quant; BF16 vision/head/output embedding",
        "autoExecute": False, "vision_trained": False, "cuda_resources": resources,
        "max_length": args.max_length, "microbatch_size": args.microbatch,
        "gradient_accumulation": args.gradient_accumulation, "seed": args.seed, "learning_rate": args.lr,
        "loss_reduction": "mean supervised native choice questions per optimizer update",
        "attention_implementation": args.attn_implementation, "use_kernels": args.use_kernels,
        "epochs": 1, "planned_steps": len(groups), "optimizer_steps": 0,
        "training_order_sha256": record_digest(row["id"] for index in order for row in [rows["train"][index]]),
        "all_planned_steps_completed": False, "complete_selected_training_epochs": False,
        "selected_training_scope": "all context-eligible prepared rows" if args.mode == "full" else "bounded CUDA migration benchmark",
        "trained_records": 0, "gradient_evidence": {"lora": False, "head": False},
        "timings": {"prepare_seconds": preparation_seconds, "base_load_seconds": load_seconds,
                    "migration_seconds": migration_seconds}, "losses": [],
        "kernel_execution": KERNEL_EVIDENCE,
        "training_input_tokens": 0, "training_padded_tokens": 0, "training_seconds": 0.0,
        "lora": validate_cuda_model(model, trainable=True, expected_fingerprints=pilot_metadata["trainable_fingerprints"]),
        "initial_trainable_fingerprints": trainable_fingerprints(model), "citations": prepared["citations"],
        "limitations": prepared["limitations"]}
    cursor, trained = 0, []
    if args.resume:
        resume_inventory = verify_adapter_manifest(args.resume)
        if resume_inventory.get("checkpoint_complete") is not True:
            raise ValueError("Resume requires an atomically completed optimizer checkpoint")
        saved = json.loads((args.resume / "training.json").read_text())
        metadata["run_id"] = saved["run_id"]
        state = torch.load(args.resume / "optimizer.pt", map_location="cpu", weights_only=True)
        validate_resume(saved, metadata, state, order)
        if (resume_inventory.get("resume_contract") != resume_contract(saved) or
            resume_inventory.get("cursor") != state["cursor"] or
            resume_inventory.get("optimizer_steps") != state["optimizer_steps"] or
            json.loads((args.resume / "resume.json").read_text()).get("contract") != resume_contract(saved)):
            raise ValueError("Checkpoint completion marker and optimizer contract disagree")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, processor = load_cuda_adapter(module, base, args.resume, trainable=True,
            attn_implementation=args.attn_implementation, use_kernels=args.use_kernels)
        metadata = saved
        metadata["kernel_execution"] = KERNEL_EVIDENCE
        metadata["resume_verification"] = {"migration": migration, "restored_optimizer_state": True,
            "completed_steps": state["optimizer_steps"], "trained_records": state["cursor"]}
        cursor, trained = state["cursor"], state["trained_indices"]
    else:
        state = None
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, foreach=False, weight_decay=0.01)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    output.mkdir(parents=True, exist_ok=True)
    snapshot = stage_path(stage, paths["live_snapshot"])
    shutil.copyfile(snapshot, output / "live-training-snapshot.json")
    metadata["live_training_snapshot"] = {"file": "live-training-snapshot.json", "sha256": file_sha(snapshot),
                                          "captured_at": json.loads(snapshot.read_text())["captured_at"]}
    if not args.resume:
        baseline_started = time.monotonic()
        metadata["baseline_eval"], _ = evaluate(model, module, processor, encoded["eval"], rows["eval"], torch.device("cuda"))
        metadata["timings"]["baseline_seconds"] = time.monotonic() - baseline_started
    atomic_json(output / "training.json", metadata)
    owned = []
    for step in range(metadata["optimizer_steps"], len(groups)):
        indices = groups[step]
        if indices != order[cursor:cursor + len(indices)]:
            raise ValueError("Training cursor changed before a native update")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        model.train()
        loss, evidence, padded_tokens = batch_update(model, module, processor, encoded["train"], rows["train"],
            indices, optimizer, args.microbatch, torch.device("cuda"))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        cursor += len(indices)
        trained.extend(indices)
        verify_coverage(order, trained, cursor, complete=False)
        tokens = sum(len(encoded["train"][index].input_ids) for index in indices)
        metadata["optimizer_steps"] = step + 1
        metadata["trained_records"] = cursor
        metadata["training_seconds"] += elapsed
        metadata["training_input_tokens"] += tokens
        metadata["training_padded_tokens"] += padded_tokens
        metadata["losses"].append(loss)
        metadata["timings"]["last_step_seconds"] = elapsed
        metadata["last_step_cuda_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "free_bytes": torch.cuda.mem_get_info()[0]}
        for kind in evidence:
            metadata["gradient_evidence"][kind] |= evidence[kind]
        atomic_json(output / "training.json", metadata)
        print(json.dumps({"mode": args.mode, "optimizer_steps": step + 1, "planned_steps": len(groups),
                          "trained_records": cursor, "loss": loss, "step_seconds": elapsed,
                          "input_tokens": tokens, "padded_tokens": padded_tokens}), flush=True)
        if (step + 1) % args.checkpoint_steps == 0:
            owned.append(checkpoint(model, processor, module, base, output, metadata, snapshot, optimizer, cursor, trained))
            prune_checkpoints(output, owned, args.keep_checkpoints)
    verify_coverage(order, trained, cursor, complete=args.mode == "full")
    if not all(metadata["gradient_evidence"].values()):
        raise ValueError("Native CUDA training did not produce both genuine gradient paths")
    if args.use_kernels and not (KERNEL_EVIDENCE["cuda_forward_calls"] > 0 and KERNEL_EVIDENCE["cuda_backward_calls"] > 0):
        raise ValueError("CUDA benchmark did not execute the requested local FLA forward/backward path")
    trained_fingerprints = trainable_fingerprints(model)
    if any(trained_fingerprints[kind]["sha256"] == metadata["initial_trainable_fingerprints"][kind]["sha256"]
           for kind in ("lora", "head")):
        raise ValueError("Native CUDA optimizer updates did not change both head and LoRA bytes")
    metadata.update(trained_trainable_fingerprints=trained_fingerprints, trainable_fingerprints=trained_fingerprints,
        finite_trainables=finite_trainables(model), all_planned_steps_completed=metadata["optimizer_steps"] == len(groups),
        complete_selected_training_epochs=cursor == len(order),
        trained_record_ids_sha256=record_digest(rows["train"][index]["id"] for index in trained),
        trained_required_record_ids=[rows["train"][index]["id"] for index in required],
        capacity_probe={"record_id": rows["train"][longest]["id"], "tokens": len(encoded["train"][longest].input_ids),
                        "trained": longest in set(trained)})
    metadata["seconds_per_optimizer_step"] = metadata["training_seconds"] / metadata["optimizer_steps"]
    metadata["input_tokens_per_second"] = metadata["training_input_tokens"] / metadata["training_seconds"]
    metadata["estimated_full_training_seconds"] = sum(len(record.input_ids) for record in encoded["train"]) / metadata["input_tokens_per_second"]
    metadata["throughput_scope"] = "Measured CUDA training only; encoding/loading/evaluation/reload/export/upload and length-distribution changes are excluded"
    metrics = {"baseline_eval": metadata["baseline_eval"]}
    predictions = {}
    evaluation_started = time.monotonic()
    for split in ("eval", "test"):
        metrics[split], predictions[split] = evaluate(model, module, processor, encoded[split], rows[split], torch.device("cuda"))
        (output / f"{split}-predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions[split]))
    metadata["timings"]["evaluation_seconds"] = time.monotonic() - evaluation_started
    metadata["status"] = "trained_pending_reload_verification"
    adapter = output / "adapter"
    metadata = save_artifact(model, processor, module, base, adapter, metadata, snapshot, metrics)
    atomic_json(output / "training.json", metadata)
    del parameters, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reload_started = time.monotonic()
    reloaded, restored_processor = load_cuda_adapter(module, base, adapter,
        attn_implementation=args.attn_implementation, use_kernels=args.use_kernels)
    restored_test = encode_complete(module, restored_processor, rows["test"], args.max_length)
    if any(list(left.input_ids) != list(right.input_ids) for left, right in zip(encoded["test"], restored_test, strict=True)):
        raise ValueError("Fresh saved CUDA adapter processor changed native tokenization")
    _, observed = evaluate(reloaded, module, restored_processor, restored_test, rows["test"], torch.device("cuda"))
    parity = compare_predictions(predictions["test"], observed, tolerance=MIGRATION_TOLERANCE)
    metadata["timings"]["adapter_reload_seconds"] = time.monotonic() - reload_started
    metadata["timings"]["benchmark_wall_seconds"] = time.monotonic() - run_started
    metadata["estimated_full_job_seconds"] = metadata["estimated_full_training_seconds"] + (
        metadata["timings"]["benchmark_wall_seconds"] - metadata["training_seconds"])
    metadata.update(status="trained_and_reload_verified", reload_verification=parity)
    metrics["reload_verification"] = parity
    atomic_json(adapter / "training.json", metadata)
    atomic_json(adapter / "evaluation.json", metrics)
    refresh_adapter_manifest(adapter)
    atomic_json(output / "training.json", metadata)
    atomic_json(output / "evaluation.json", metrics)
    if args.export_release:
        if args.mode != "full":
            raise ValueError("Standalone CUDA export requires a verified complete full epoch")
        from export_clef_cuda_release import export_release, load_cuda_release, mark_reload_verified
        probe = lambda current: evaluate(current, module, restored_processor, restored_test, rows["test"], torch.device("cuda"))[1]
        export_release(reloaded, base, args.export_release, metadata, probe, evaluation=metrics,
                       training_snapshot=output / "live-training-snapshot.json")
        merged_predictions = probe(reloaded)
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        standalone, standalone_processor = load_cuda_release(args.export_release)
        standalone_test = encode_complete(module, standalone_processor, rows["test"], args.max_length)
        if any(list(left.input_ids) != list(right.input_ids) for left, right in zip(restored_test, standalone_test, strict=True)):
            raise ValueError("Fresh standalone CUDA processor changed native tokenization")
        _, standalone_predictions = evaluate(standalone, module, standalone_processor, standalone_test, rows["test"], torch.device("cuda"))
        release = mark_reload_verified(args.export_release, merged_predictions, standalone_predictions, loaded_model=standalone)
        metadata["standalone_release"] = {"reload_verified": True, "manifest_sha256": file_sha(args.export_release / "release.json"),
                                          "status": release["status"]}
        atomic_json(output / "training.json", metadata)
    return {"status": metadata["status"], "mode": args.mode, "optimizer_steps": metadata["optimizer_steps"],
            "trained_records": cursor, "input_tokens_per_second": metadata["input_tokens_per_second"],
            "migration_verification": migration, "reload_verification": parity,
            "standalone_release": metadata.get("standalone_release"), "hub_contacted": False}


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

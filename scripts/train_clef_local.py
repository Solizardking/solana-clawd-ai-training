#!/usr/bin/env python3
"""Train genuine Clef locally from an existing standalone MPS NF4 base.

The default is a measured 16-step pilot, not a full training claim. Only the
last four text layers receive LoRA; the native joint head is also trained.
Vision, input embeddings and the readable output embedding remain frozen BF16.
No full source weights, remote GPU Jobs, or global model snapshots are fetched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from clef_research_data import (DATASET_ID, MODEL_ID, MODEL_REVISION, prepare_dataset,
                                read_jsonl, training_source_hash, validate_decision_record)
from clef_research_training import (compare_predictions, evaluate, supervised_loss,
                                    trainable_fingerprints)

DATASET_REVISION = "58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2"
INPUT_RAW_ROW_COUNTS = {"train": 78171, "eval": 2595, "test": 2896}
INPUT_PARQUET_SHA256 = {
    "train": "4d3d6b4b3030746cd29116a3f6d1a2f4c39ddc90119a43b39569085635fbb93d",
    "eval": "707aa9528afe89637fe00a134c9c993554bc2a63ba9c946db6698ff24b8a282e",
    "test": "44129a4c88930a008ddba8104705b6d223d395cd070b45ad9189be326dcea930",
}
PAPERS = ("2605.12151", "2606.08232")
GIB = 1024 ** 3


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_mps(memory_fraction=0.90):
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Actual Apple MPS hardware is required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK; CPU fallback is not local training proof")
    if not 0 < memory_fraction <= 0.95:
        raise ValueError("MPS memory fraction must be in (0,0.95]")
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    return {"recommended_working_bytes": torch.mps.recommended_max_memory(),
            "memory_fraction": memory_fraction,
            "allocator_limit_bytes": int(torch.mps.recommended_max_memory() * memory_fraction)}


def resident_tensor_bytes(model):
    """Include external safetensors storage and nested NF4 scale tensors."""
    import torch
    seen, total = set(), 0

    def visit(value):
        nonlocal total
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif value is not None and type(value).__name__ == "QuantState":
            visit(vars(value))

    for parameter in model.parameters():
        visit(parameter)
        visit(getattr(parameter, "quant_state", None))
    for buffer in model.buffers():
        visit(buffer)
    return total


def memory_guard(limits, additional_bytes=0, *, model=None, flush_cache=False):
    import torch
    import resource
    if flush_cache:
        torch.mps.empty_cache()
    current = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    tensors = resident_tensor_bytes(model) if model is not None else 0
    working = max(current, driver, tensors)
    if working + additional_bytes > limits["allocator_limit_bytes"]:
        raise RuntimeError("MPS working-set guard cannot accommodate model and training reserve")
    return {"allocated_bytes": current, "driver_bytes": driver,
            "model_tensor_bytes": tensors, "guarded_working_bytes": working,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "reserved_bytes": additional_bytes, **limits}


def verify_prepared_source(manifest):
    if manifest.get("dataset") != DATASET_ID or manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Local training requires the published expanded research commit")
    if manifest.get("model") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Prepared input is not for the pinned original Clef")
    if manifest.get("input_parquet_sha256") != INPUT_PARQUET_SHA256:
        raise ValueError("Prepared source parquet hashes differ from the verified published files")
    if manifest.get("input_source", {}).get("input_rows") != INPUT_RAW_ROW_COUNTS:
        raise ValueError("Prepared inputs must include all 83,662 published source rows before filtering")
    security = manifest.get("security_filter", {})
    if security.get("applied") is not True or security.get("version") != 1:
        raise ValueError("Prepared inputs lack the required credential/signing-literal exclusion audit")
    return {"security_filter": security, "input_parquet_sha256": manifest["input_parquet_sha256"],
            "input_raw_row_counts": manifest["input_source"]["input_rows"],
            "prepared_split_counts": {split: entry["rows"] for split, entry in manifest["outputs"].items()}}


def validate_local_base(base):
    """Check the task-owned converted model without downloading original shards."""
    base = Path(base).resolve()
    manifest_path = base / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Local NF4 base is incomplete or does not originate from pinned original Clef")
    if not manifest.get("standalone_weights_saved") or not manifest.get("reload_verified"):
        raise ValueError("Converted base requires standalone native finite/reload verification")
    for group in ("output_weight_files", "preserved_sidecars"):
        if not manifest.get(group):
            raise ValueError(f"Converted base has no {group} integrity records")
        for filename, expected in manifest[group].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe converted-base file reference")
            path = base / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
                raise ValueError(f"Converted base failed integrity verification: {filename}")
    generated = {name: {"sha256": file_sha(base / name), "bytes": (base / name).stat().st_size}
                 for name in ("config.json", "model.safetensors.index.json", "generation_config.json")
                 if (base / name).is_file()}
    if "config.json" not in generated:
        raise ValueError("Converted base has no generated model config")
    return {"manifest_sha256": file_sha(manifest_path), "source": manifest["source"],
            "reload_probe": manifest.get("reload_probe"), "format": manifest.get("format"),
            "generated_files": generated}


def load_local_module(base):
    path = Path(base) / "joint_schema_model.py"
    manifest = json.loads((Path(base) / "conversion_manifest.json").read_text())
    expected = manifest.get("preserved_sidecars", {}).get(path.name, {})
    if manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION} or not expected:
        raise ValueError("Native training code must come from the pinned local conversion")
    if path.is_symlink() or path.stat().st_size != expected.get("bytes") or file_sha(path) != expected.get("sha256"):
        raise ValueError("Native Clef code failed integrity verification before import")
    name = "clawd_local_native_clef_" + file_sha(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Local native Clef module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_local_base(module, base: Path, device="mps"):
    """Load only existing local prequantized weights, native head and tokenizer."""
    import bitsandbytes as bnb
    import torch
    if str(device).split(":")[0] != "mps" or not torch.backends.mps.is_available():
        raise ValueError("The local training/inference loader requires actual MPS; offload is disabled")
    if not Path(base).is_dir():
        raise ValueError("Local base must already exist; downloads are disabled")
    model, processor = module.load_release_model(
        Path(base), device=device, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    )
    quantized = [child for child in model.language_model.modules() if isinstance(child, bnb.nn.Linear4bit)]
    if not quantized or any(child.weight.quant_state.quant_type != "nf4" or not child.weight.quant_state.nested for child in quantized):
        raise ValueError("Local backbone must retain its genuine NF4/double-quant text weights")
    if any(parameter.device.type != str(device).split(":")[0] for parameter in model.parameters()):
        raise ValueError("The local model was offloaded to an unintended device")
    backbone = model.language_model
    if isinstance(backbone.get_output_embeddings().weight, bnb.nn.Params4bit) or backbone.get_output_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("Native Clef output embedding must remain readable BF16")
    if any(isinstance(child, bnb.nn.Linear4bit) for child in backbone.model.visual.modules()) or any(parameter.dtype != torch.bfloat16 for parameter in backbone.model.visual.parameters()):
        raise ValueError("Vision must remain original BF16")
    return model, processor


def _checkpoint_without_input_grads(backbone):
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(backbone, "_require_grads_hook"):
        backbone.disable_input_require_grads()


def attach_local_lora(model, rank=8, last_layers=4):
    import torch
    from peft import LoraConfig, get_peft_model
    count = len(model.language_model.model.language_model.layers)
    if rank < 1 or not 1 <= last_layers <= count:
        raise ValueError("Positive rank and a valid last-layer count are required")
    indices = set(range(count - last_layers, count))
    targets = []
    for name, child in model.language_model.named_modules():
        match = re.search(r"\.language_model\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) in indices and isinstance(child, torch.nn.Linear):
            targets.append(name)
    if not targets:
        raise ValueError("No last-layer text linear modules found")
    model.language_model = get_peft_model(model.language_model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, target_modules=targets, bias="none"))
    model.head.requires_grad_(True)
    _checkpoint_without_input_grads(model.language_model)
    if hasattr(model.language_model, "_require_grads_hook"):
        raise ValueError("Frozen input embeddings still have an input-gradient hook")
    return {"target_modules": targets, "text_layers": sorted(indices), "rank": rank,
            "gradient_checkpointing": "nonreentrant", "frozen_input_gradients": False}


def finite_trainables(model):
    import torch
    checked = {"lora": 0, "head": 0}
    for name, parameter in model.named_parameters():
        kind = "lora" if "lora_" in name else "head" if name.startswith("head.") else None
        if kind and parameter.requires_grad:
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"Nonfinite trained parameter: {name}")
            checked[kind] += parameter.numel()
    if not all(checked.values()):
        raise ValueError("Both local training paths must have actual finite trainable parameters")
    return {kind: {"finite": True, "parameters": count} for kind, count in checked.items()}


def restored_training_fingerprints(model):
    """Compare all restored head/LoRA bytes, including frozen inference tensors."""
    parameters = [(parameter, parameter.requires_grad) for name, parameter in model.named_parameters()
                  if "lora_" in name or name.startswith("head.")]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(True)
        finite_trainables(model)
        return trainable_fingerprints(model)
    finally:
        for parameter, required in parameters:
            parameter.requires_grad_(required)


def source_requirements(rows):
    required = [index for index, row in enumerate(rows) if row["provenance"].get("source_type") == "live_tape_observation"]
    for paper in PAPERS:
        found = next((index for index, row in enumerate(rows) if paper in str(row["provenance"].get("source"))), None)
        if found is None:
            raise ValueError(f"The selected native cohort is missing required paper {paper}")
        if found not in required:
            required.append(found)
    if not any(rows[index]["provenance"].get("source_type") == "live_tape_observation" for index in required):
        raise ValueError("The selected native cohort has no captured live decisions")
    return required


def select_encoded_cohort(rows, module, processor, limit, seed, max_length, require_sources=False):
    """Select reproducibly, backfilling exclusions and never truncating evidence."""
    from array import array
    from dataclasses import replace
    from clef_research_data import model_input_exclusion_reason
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    encoded_by_index, excluded = {}, []

    def encode(index):
        if index in encoded_by_index:
            return encoded_by_index[index]
        row = rows[index]
        validate_decision_record(row)
        reason = model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]})
        if reason:
            excluded.append({"id": row["id"], "reason": reason})
            encoded_by_index[index] = None
            return None
        record = module.encode_record(processor.tokenizer, row, max_length=1_000_000, processor=processor)
        if len(record.input_ids) > max_length:
            excluded.append({"id": row["id"], "reason": "context_exceeds_limit", "tokens": len(record.input_ids)})
            encoded_by_index[index] = None
            return None
        if any(type(token) is not int or not 0 <= token <= 2 ** 31 - 1 for token in record.input_ids):
            raise ValueError("Native tokenizer IDs must be nonnegative signed-32-bit integers.")
        tokens = array("i", record.input_ids)
        if tokens.itemsize != 4:
            raise RuntimeError("Native token storage requires four-byte signed integers.")
        record = replace(record, input_ids=tokens)
        encoded_by_index[index] = record
        return record

    required = []
    if require_sources:
        for index in order:
            if rows[index]["provenance"].get("source_type") == "live_tape_observation" and encode(index) is not None:
                required.append(index)
        if not required:
            raise ValueError("No captured live decisions fit the native context limit")
        for paper in PAPERS:
            found = next((index for index in order if paper in str(rows[index]["provenance"].get("source")) and encode(index) is not None), None)
            if found is None:
                raise ValueError(f"No complete required paper {paper} decision fits {max_length} tokens")
            required.append(found)
    if limit and len(required) > limit:
        raise ValueError("Cohort limit cannot include the required live and paper records")
    selected = list(dict.fromkeys(required))
    selected_set = set(selected)
    for index in order:
        if index in selected_set:
            continue
        if limit and len(selected) >= limit:
            break
        if encode(index) is not None:
            selected.append(index)
            selected_set.add(index)
    if not selected:
        raise ValueError("Native cohort is empty after context/security filtering")
    kept = [rows[index] for index in selected]
    records = [encoded_by_index[index] for index in selected]
    audit = {"prepared": len(rows), "selected": len(kept), "context_limit": max_length,
             "max_tokens": max(len(record.input_ids) for record in records),
             "input_token_storage": "signed_int32_array",
             "input_token_bytes": sum(len(record.input_ids) * record.input_ids.itemsize for record in records),
             "sources": dict(Counter(str(row["provenance"].get("source")) for row in kept)),
             "required_source_indices": source_requirements(kept) if require_sources else [], "excluded": excluded}
    return records, kept, audit


def save_local_adapter(model, processor, module, base, output, metadata, *, training_snapshot=None):
    from safetensors.torch import save_file
    output, base = Path(output), Path(base)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(json.dumps(metadata))
    metadata["finite_trainables"] = finite_trainables(model)
    metadata["trainable_fingerprints"] = trainable_fingerprints(model)
    model.language_model.save_pretrained(output)
    processor.save_pretrained(output)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.head.state_dict().items()},
              str(output / "joint_head.safetensors"))
    shutil.copyfile(base / "joint_head_config.json", output / "joint_head_config.json")
    shutil.copyfile(Path(module.__file__), output / "joint_schema_model.py")
    if training_snapshot is not None:
        from clef_live_tape import validate_snapshot
        training_snapshot = Path(training_snapshot)
        validate_snapshot(json.loads(training_snapshot.read_text()))
        if file_sha(training_snapshot) != metadata.get("live_training_snapshot", {}).get("sha256"):
            raise ValueError("Live training snapshot changed before the saved checkpoint")
        shutil.copyfile(training_snapshot, output / "live-training-snapshot.json")
    for source, destination in (("LICENSE", "LICENSE"), ("README.md", "source_base_model_card.md")):
        if (base / source).is_file():
            shutil.copyfile(base / source, output / destination)
    metadata["artifact_files"] = {name: {"bytes": (output / name).stat().st_size, "sha256": file_sha(output / name)}
                                  for name in ("adapter_config.json", "adapter_model.safetensors", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py")}
    write_json(output / "training.json", metadata)
    refresh_adapter_manifest(output)
    return metadata


def refresh_adapter_manifest(output):
    output = Path(output)
    files = {}
    for path in sorted(output.rglob("*")):
        if "__pycache__" in path.relative_to(output).parts:
            continue
        if path.is_symlink():
            raise ValueError("Adapter artifacts may not contain symlinks")
        if path.is_file() and path.name != "adapter_artifact_manifest.json":
            files[path.relative_to(output).as_posix()] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    write_json(output / "adapter_artifact_manifest.json", {"files": files})
    return files


def verify_adapter_manifest(output):
    output = Path(output)
    manifest = json.loads((output / "adapter_artifact_manifest.json").read_text())
    files = manifest.get("files")
    required = {"adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors",
                "joint_head_config.json", "joint_schema_model.py", "training.json",
                "tokenizer.json", "tokenizer_config.json", "processor_config.json"}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Saved local adapter has an incomplete weights/code/processor inventory")
    paths = [path for path in output.rglob("*") if "__pycache__" not in path.relative_to(output).parts]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Adapter artifacts may not contain symlinks")
    actual = {path.relative_to(output).as_posix() for path in paths if path.is_file()
              and path.name != "adapter_artifact_manifest.json"}
    if actual != set(files):
        raise ValueError("Saved adapter inventory differs from the actual files")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe saved adapter artifact reference")
        path = output / relative
        if path.is_symlink() or not path.is_file() or file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Saved local adapter artifact changed: {name}")
    return manifest


def load_local_adapter(module, base: Path, adapter: Path, device="mps", *, trainable=False, allow_unverified=False):
    """Fresh local NF4 base plus saved native head; never call CUDA/global loader."""
    from peft import PeftModel
    from safetensors.torch import load_file
    adapter, base = Path(adapter), Path(base)
    verify_adapter_manifest(adapter)
    metadata = json.loads((adapter / "training.json").read_text())
    if metadata.get("model_revision") != MODEL_REVISION or metadata.get("optimizer_steps", 0) < 1:
        raise ValueError("Local adapter has no actual training on the pinned Clef base")
    if metadata.get("dataset_revision") != DATASET_REVISION or metadata.get("input_parquet_sha256") != INPUT_PARQUET_SHA256 or metadata.get("input_raw_row_counts") != INPUT_RAW_ROW_COUNTS:
        raise ValueError("Local adapter does not use the verified expanded research source")
    if metadata.get("security_filter", {}).get("applied") is not True or metadata["security_filter"].get("version") != 1:
        raise ValueError("Local adapter did not record model-input security exclusions")
    if not allow_unverified and metadata.get("status") != "trained_and_reload_verified":
        raise ValueError("Local adapter still requires saved-load verification")
    if metadata.get("local_base_manifest_sha256") and file_sha(base / "conversion_manifest.json") != metadata["local_base_manifest_sha256"]:
        raise ValueError("Local adapter refers to a different converted base")
    if metadata.get("local_base_manifest_sha256") and not metadata.get("local_base_generated_files", {}).get("config.json"):
        raise ValueError("Local adapter lacks converted configuration integrity records")
    for name, expected in metadata.get("local_base_generated_files", {}).items():
        if name not in {"config.json", "model.safetensors.index.json", "generation_config.json"}:
            raise ValueError("Unsafe generated base configuration reference")
        path = base / name
        if path.is_symlink() or not path.is_file() or file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Converted base configuration changed: {name}")
    for name, expected in metadata.get("artifact_files", {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe local adapter weight reference")
        path = adapter / name
        if not path.is_file() or file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Saved local adapter artifact changed: {name}")
    model, processor = load_local_base(module, base, device)
    model.language_model = PeftModel.from_pretrained(model.language_model, adapter, is_trainable=trainable)
    model.head.load_state_dict(load_file(str(adapter / "joint_head.safetensors")), strict=True)
    model.head.requires_grad_(trainable)
    if restored_training_fingerprints(model) != metadata.get("trainable_fingerprints"):
        raise ValueError("Restored native head/LoRA bytes differ from the saved trained fingerprints")
    if trainable:
        _checkpoint_without_input_grads(model.language_model)
    return model.eval(), processor


def training_order(rows, seed, epoch, encoded=None):
    required = source_requirements(rows)
    if encoded is not None:
        if not rows or len(rows) != len(encoded):
            raise ValueError("Capacity probe requires matching nonempty rows and native records.")
        longest = max(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
        if longest not in required:
            required.append(longest)
    other = [index for index in range(len(rows)) if index not in required]
    random.Random(f"{seed}:{epoch}").shuffle(other)
    return required + other


def require_training_budget(rows, encoded, total_steps, gradient_accumulation):
    """Require the mandatory examples and longest full record in the first pass."""
    if not rows or len(rows) != len(encoded) or total_steps < 1 or gradient_accumulation < 1:
        raise ValueError("Capacity probe requires a positive budget and matching complete native records.")
    longest = max(range(len(encoded)), key=lambda index: len(encoded[index].input_ids))
    minimum = len(set(source_requirements(rows)) | {longest})
    if total_steps * gradient_accumulation < minimum:
        raise ValueError("Pilot step budget cannot train all required live/paper rows and the longest capacity probe.")
    return longest


def _cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def prune_owned_checkpoints(output, completed, keep):
    """Delete only earlier completed checkpoints created by this fresh run."""
    if keep < 1:
        raise ValueError("Keep at least one completed training checkpoint")
    root = Path(output).resolve() / "checkpoints"
    while len(completed) > keep:
        path = Path(completed[0])
        if not re.fullmatch(r"step-\d{6}", path.name) or path.is_symlink() or path.resolve().parent != root or not (path / "adapter_artifact_manifest.json").is_file():
            raise ValueError("Refusing to prune a checkpoint outside this run's completed inventory")
        shutil.rmtree(path)
        completed.pop(0)


def guard_model_input(row):
    from clef_research_data import model_input_exclusion_reason
    if model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]}):
        raise ValueError("Selected model-facing input failed final privacy guard")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("local/clef-27b-mps-nf4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clef-local-pilot"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-records", type=int)
    parser.add_argument("--eval-records", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint-steps", type=int, default=8)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--init-adapter", type=Path, help="Start a new phase from a verified local adapter/head; optimizer starts fresh")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/filter/encode native data with processor metadata, without loading weights")
    parser.add_argument("--export-release", type=Path, help="Optional compact standalone export after saved local reload verification")
    args = parser.parse_args()
    for name, pilot, full in (("max_steps", 16, 0), ("train_records", 128, 0), ("eval_records", 8, 32)):
        if getattr(args, name) is None:
            setattr(args, name, pilot if args.mode == "pilot" else full)
    if args.dataset_revision != DATASET_REVISION:
        parser.error("--dataset-revision must be the verified published expanded commit")
    if min(args.max_steps, args.train_records, args.eval_records) < 0 or min(args.max_length, args.last_layers, args.rank, args.gradient_accumulation, args.epochs, args.checkpoint_steps, args.keep_checkpoints) < 1 or args.lr <= 0:
        parser.error("Record/step limits must be nonnegative; training parameters must be positive")
    return args


def run(args):
    import torch
    from transformers import AutoProcessor
    from clef_live_tape import capture_live_snapshot, live_decision_records, serialize_snapshot
    from clef_research_data import model_input_exclusion_reason

    output, base = args.output.resolve(), args.base.resolve()
    if args.dataset_revision != DATASET_REVISION:
        raise ValueError("Training requires the verified published expanded commit")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Preserve previous runs; output must be a separate empty directory")
    if output == base or output.is_relative_to(base):
        raise ValueError("Training artifacts must stay separate from the immutable local base")
    limits = require_mps(args.memory_fraction)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    module = load_local_module(base)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    snapshot = capture_live_snapshot(timeout=15, max_frames=4)
    (output / "live-training-snapshot.json").write_text(serialize_snapshot(snapshot) + "\n", encoding="utf-8")
    live = live_decision_records(snapshot)
    if not live:
        raise ValueError("No real captured live decisions are available; no synthetic labels were substituted")
    manifest = prepare_dataset(output / "prepared", args.seed, dataset_revision=args.dataset_revision, live_decisions=live)
    source_audit = verify_prepared_source(manifest)
    cohorts, audits = {}, {}
    for split in ("train", "eval", "test"):
        rows = read_jsonl(output / "prepared" / f"{split}.jsonl")
        encoded, kept, audits[split] = select_encoded_cohort(rows, module, processor,
                                                           args.train_records if split == "train" else args.eval_records,
                                                           args.seed, args.max_length, require_sources=split == "train")
        cohorts[split] = (encoded, kept)
        # Persist exact selected rows for an auditable later phase/checkpoint.
        path = output / "cohorts" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in kept:
                guard_model_input(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        audits[split]["sha256"] = file_sha(path)
    # Keep only selected rows and compact signed32 token arrays during training;
    # the last raw split list is unnecessary once its cohort has been written.
    del rows
    write_json(output / "data-manifest.json", {"prepared": manifest, "cohorts": audits})
    prepared_seconds = time.monotonic() - started
    if args.prepare_only:
        return {"status": "prepared_without_weights", "cohorts": audits, "limits": limits}
    base_identity = validate_local_base(base)
    if args.init_adapter:
        model, processor = load_local_adapter(module, base, args.init_adapter, trainable=True)
        adapter_config = model.language_model.peft_config["default"]
        lora = {"target_modules": sorted(adapter_config.target_modules), "rank": adapter_config.r,
                "text_layers": sorted({int(re.search(r"\.layers\.(\d+)\.", name).group(1)) for name in adapter_config.target_modules})}
        count = len(model.language_model.get_base_model().model.language_model.layers)
        if adapter_config.r != args.rank or lora["text_layers"] != list(range(count - args.last_layers, count)):
            raise ValueError("Initial adapter rank/layer scope differs from this new phase")
    else:
        model, processor = load_local_base(module, base)
        lora = attach_local_lora(model, args.rank, args.last_layers)
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad)
    initial_memory = memory_guard(limits, additional_bytes=trainable_bytes * 3 + 2 * GIB, model=model, flush_cache=True)
    initial = trainable_fingerprints(model)
    baseline_started = time.monotonic()
    baseline, _ = evaluate(model, module, processor, *cohorts["eval"], torch.device("mps"))
    baseline_seconds = time.monotonic() - baseline_started
    train_encoded, train_rows = cohorts["train"]
    total_steps = args.epochs * math.ceil(len(train_rows) / args.gradient_accumulation)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    capacity_index = require_training_budget(train_rows, train_encoded, total_steps, args.gradient_accumulation)
    metadata = {
        "status": "training", "run_mode": args.mode, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": manifest["dataset_revision"],
        **source_audit, "citations": manifest["citations"],
        "live_training_snapshot": {"file": "live-training-snapshot.json",
                                   "sha256": file_sha(output / "live-training-snapshot.json"),
                                   "captured_at": snapshot["captured_at"]},
        "local_base_manifest_sha256": base_identity["manifest_sha256"], "local_base_format": base_identity["format"],
        "local_base_generated_files": base_identity["generated_files"],
        "task": manifest["task"], "autoExecute": False, "vision_trained": False,
        "local_backend": "MPS NF4 double-quant; BF16 vision/head/output embedding", "cpu_fallback": False,
        "lora": lora, "limits": limits, "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_bytes": trainable_bytes, "initial_trainable_fingerprints": initial,
        "seed": args.seed, "max_length": args.max_length, "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.lr, "epochs": args.epochs, "planned_steps": total_steps, "optimizer_steps": 0,
        "baseline_eval": baseline, "losses": [], "cohorts": audits,
        "timings": {"prepare_seconds": prepared_seconds, "baseline_eval_seconds": baseline_seconds},
        "memory": initial_memory, "all_planned_steps_completed": False,
        "complete_selected_training_epochs": total_steps == args.epochs * math.ceil(len(train_rows) / args.gradient_accumulation),
        "selected_training_scope": "all context-eligible prepared rows" if not args.train_records else "bounded shuffled cohort with required live and paper rows",
        "additional_input_exclusions_present": any(entry["excluded"] for entry in audits.values()),
        "training_source_sha256": training_source_hash(), "local_trainer_sha256": file_sha(Path(__file__)),
        "limitations": manifest["limitations"] + ["A bounded pilot is not full-corpus training or an official Decision Index result."],
    }
    write_json(output / "training.json", metadata)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, foreach=False, weight_decay=0.01)
    gradient_evidence = {"lora": False, "head": False}
    trained_ids = set()
    completed_checkpoints = []
    for epoch in range(args.epochs):
        order = training_order(train_rows, args.seed, epoch, encoded=train_encoded)
        for start in range(0, len(order), args.gradient_accumulation):
            indices = order[start:start + args.gradient_accumulation]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_started = time.monotonic()
            step_tokens = 0
            for index in indices:
                record, row = train_encoded[index], train_rows[index]
                guard_model_input(row)
                step_tokens += len(record.input_ids)
                logits = model(module.collate_records([record], processor.tokenizer.pad_token_id, torch.device("mps")))
                loss = supervised_loss(logits, [record], [row])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Local native training loss is nonfinite")
                (loss / len(indices)).backward()
                step_loss += float(loss.detach()) / len(indices)
                trained_ids.add(row["id"])
            for name, parameter in model.named_parameters():
                kind = "lora" if "lora_" in name else "head" if name.startswith("head.") else None
                if kind and parameter.grad is not None and bool(parameter.grad.abs().sum() > 0):
                    gradient_evidence[kind] = True
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            del logits, loss
            torch.mps.synchronize()
            metadata["optimizer_steps"] += 1
            metadata["losses"].append(step_loss)
            metadata["seconds"] = time.monotonic() - started
            metadata["memory"] = memory_guard(limits, model=model)
            metadata["gradient_evidence"] = gradient_evidence
            step_seconds = time.monotonic() - step_started
            metadata["timings"]["last_step_seconds"] = step_seconds
            write_json(output / "training.json", metadata)
            print(json.dumps({"event": "local_optimizer_step", "step": metadata["optimizer_steps"], "planned_steps": total_steps,
                              "loss": step_loss, "step_seconds": step_seconds, "tokens": step_tokens,
                              "tokens_per_second": step_tokens / step_seconds,
                              "seconds": metadata["seconds"], "memory": metadata["memory"]}), flush=True)
            if metadata["optimizer_steps"] % args.checkpoint_steps == 0:
                checkpoint = output / "checkpoints" / f"step-{metadata['optimizer_steps']:06d}"
                save_local_adapter(model, processor, module, base, checkpoint, metadata,
                                   training_snapshot=output / "live-training-snapshot.json")
                torch.save(_cpu_state(optimizer.state_dict()), checkpoint / "optimizer.pt")
                refresh_adapter_manifest(checkpoint)
                completed_checkpoints.append(checkpoint)
                prune_owned_checkpoints(output, completed_checkpoints, args.keep_checkpoints)
            if metadata["optimizer_steps"] >= total_steps:
                break
        if metadata["optimizer_steps"] >= total_steps:
            break
    trained = trainable_fingerprints(model)
    if not all(gradient_evidence.values()) or any(initial[kind]["sha256"] == trained[kind]["sha256"] for kind in ("lora", "head")):
        raise ValueError("Both native head and LoRA must have actual gradient and byte-update evidence")
    if not all(train_rows[index]["id"] in trained_ids for index in source_requirements(train_rows)):
        raise ValueError("Pilot did not actually train all required live and paper examples")
    if train_rows[capacity_index]["id"] not in trained_ids:
        raise ValueError("Pilot did not train the longest selected complete capacity-probe example.")
    metadata["capacity_probe"] = {"record_id": train_rows[capacity_index]["id"],
                                  "tokens": len(train_encoded[capacity_index].input_ids), "trained": True}
    metadata["trained_trainable_fingerprints"] = trained
    metadata["all_planned_steps_completed"] = metadata["optimizer_steps"] == total_steps
    metadata["trained_records"] = len(trained_ids)
    metadata["trained_required_record_ids"] = [train_rows[index]["id"] for index in source_requirements(train_rows)]
    metrics = {"baseline_eval": baseline}
    verification_predictions = None
    for split in ("eval", "test"):
        metrics[split], predictions = evaluate(model, module, processor, *cohorts[split], torch.device("mps"))
        (output / f"{split}-predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        if split == "test":
            verification_predictions = predictions[:min(4, len(predictions))]
    metadata["status"] = "trained_pending_reload_verification"
    adapter = output / "adapter"
    metadata = save_local_adapter(model, processor, module, base, adapter, metadata,
                                  training_snapshot=output / "live-training-snapshot.json")
    write_json(output / "training.json", metadata)
    del parameters, optimizer, model, parameter
    gc.collect()
    torch.mps.empty_cache()
    reloaded, restored_processor = load_local_adapter(module, base, adapter, allow_unverified=True)
    verify_count = len(verification_predictions)
    encoded, rows = (values[:verify_count] for values in cohorts["test"])
    reload_metrics, observed = evaluate(reloaded, module, restored_processor, encoded, rows, torch.device("mps"))
    parity = compare_predictions(verification_predictions, observed, tolerance=1e-3)
    metrics["reload_verification"] = {**parity, "metrics": reload_metrics, "base": "fresh local standalone NF4"}
    metadata.update(status="trained_and_reload_verified", seconds=time.monotonic() - started,
                    reload_verification=metrics["reload_verification"])
    write_json(adapter / "training.json", metadata)
    write_json(output / "training.json", metadata)
    write_json(output / "evaluation.json", metrics)
    shutil.copyfile(output / "evaluation.json", adapter / "evaluation.json")
    citations = (Path(__file__).resolve().parents[1] / "data/realtime_research_citations.md").read_text()
    (adapter / "README.md").write_text(f"# Locally trained Clawd Clef native decision adapter\n\nRun mode: {args.mode}; completed optimizer steps: {metadata['optimizer_steps']}.\n\nLast text-layer LoRA and the native head trained on the pinned research dataset plus captured read-only tape evidence. Load with `load_local_adapter(module, base, adapter)` from the included `train_clef_local.py`. Original vision and output embeddings remain BF16/frozen. See training/evaluation metadata for exact cohort, byte updates and fresh local reload verification. This is an answer-selection model, with no signing or live-trade permissions.\n\n{citations}")
    for name in ("train_clef_local.py", "clef_research_data.py", "clef_research_training.py", "clef_live_tape.py", "research_expansion_artifacts.py"):
        shutil.copyfile(Path(__file__).with_name(name), adapter / name)
    refresh_adapter_manifest(adapter)
    if args.export_release is not None:
        from export_clef_mps_release import export_release, load_local_release, mark_reload_verified
        from export_clef_release import refresh_release_manifest
        probe = lambda current: evaluate(current, module, restored_processor, encoded, rows, torch.device("mps"))[1]
        exported = export_release(reloaded, base, args.export_release.resolve(), metadata, probe, evaluation=metrics)
        # Make the standalone artifact explain the actual training and retain
        # citations/source licensing; additional files are hashed before reload.
        release = args.export_release.resolve()
        shutil.copyfile(base / "README.md", release / "source_base_model_card.md")
        for name in ("README.md", "evaluation.json", "live-training-snapshot.json", "train_clef_local.py", "clef_research_data.py",
                     "clef_research_training.py", "clef_live_tape.py", "research_expansion_artifacts.py"):
            shutil.copyfile(adapter / name, release / name)
        for name in ("export_clef_mps_release.py", "run_clef_local.py"):
            source = Path(__file__).with_name(name)
            if source.is_file():
                shutil.copyfile(source, release / name)
        (release / "README.md").write_text(
            "---\nlicense: apache-2.0\nlibrary_name: transformers\nbase_model: Cloudflare/clef\n"
            "base_model_relation: finetune\ntags:\n- clef\n- solana\n- structured-output\n- custom-code\n---\n\n"
            "# Locally trained Clawd Clef standalone native decision model\n\n"
            f"Original Clef revision: `{MODEL_REVISION}`. Research dataset revision: `{DATASET_REVISION}`. "
            f"Run mode: {args.mode}; completed optimizer steps: {metadata['optimizer_steps']}; "
            f"distinct trained records: {metadata['trained_records']}.\n\n"
            "This release contains the full original multimodal backbone with trained final text-layer LoRA "
            "merged into NF4 weights and the trained native joint head. The original vision encoder and readable "
            "output embedding remain BF16. Source licensing and model documentation are retained in LICENSE "
            "and source_base_model_card.md.\n\n"
            "The task is typed reference-answer selection plus choices labeled from captured observed fields. "
            "The live observation is a timestamped training snapshot. No official Decision Index result, new "
            "trading-outcome benchmark, or ongoing weight updates are claimed. Training/evaluation metadata "
            "records exact cohorts, baseline scores, byte updates and fresh standalone reload verification.\n\n"
            "For fresh read-only live choices, run `python run_clef_local.py --model /path/to/this/release`. "
            "For the native model API, use `from export_clef_mps_release import load_local_release` and "
            "`model, processor = load_local_release('/path/to/this/release')`. MPS is required for this local "
            "NF4 workflow; the model does not sign or execute transactions.\n\n" + citations,
            encoding="utf-8")
        refresh_release_manifest(release)
        merged_predictions = probe(reloaded)
        del reloaded
        gc.collect()
        torch.mps.empty_cache()
        standalone, standalone_processor = load_local_release(release)
        restored_encoded = [module.encode_record(standalone_processor.tokenizer, row, max_length=args.max_length,
                                                 processor=standalone_processor) for row in rows]
        if any(tuple(left.input_ids) != tuple(right.input_ids) for left, right in zip(encoded, restored_encoded, strict=True)):
            raise ValueError("Standalone processor changed native tokenization")
        _, standalone_predictions = evaluate(standalone, module, standalone_processor, restored_encoded, rows, torch.device("mps"))
        exported = mark_reload_verified(release, merged_predictions, standalone_predictions)
        write_json(output / "standalone-export.json", {"path": str(args.export_release.resolve()),
                                                       "status": exported["status"], "reload_verification": exported["reload_verification"]})
    return {"status": metadata["status"], "run_mode": args.mode, "optimizer_steps": metadata["optimizer_steps"],
            "adapter": str(adapter), "evaluation": metrics, "seconds": metadata["seconds"]}


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)

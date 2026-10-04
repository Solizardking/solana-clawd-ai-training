"""Compact standalone CUDA Clef export and fresh read-only native inference.

The original NF4 base is immutable. Shared exporter primitives measure merge
drift, preserve untouched tensors exactly, and hardlink unchanged shards. A
CUDA load plus native predictions must separately prove standalone reload.
Importing this module neither loads weights nor contacts the Hub.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION, model_input_exclusion_reason
from clef_research_training import encode_rows, trainable_fingerprints
import export_clef_mps_release as compact
from export_clef_release import _sha256, refresh_release_manifest, verify_release

BASE_SIDECARS = {
    "LICENSE", "README.md", "chat_template.jinja", "config.json", "generation_config.json",
    "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py",
    "model.safetensors.index.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "source_config.json", "source_generation_config.json",
    "source_model.safetensors.index.json", "conversion_manifest.json",
}
REQUIRED_RUNTIME = {"export_clef_cuda_release.py", "export_clef_release.py",
                    "export_clef_mps_release.py", "clef_live_tape.py",
                    "clef_research_data.py", "clef_research_training.py",
                    "train_clef_cuda.py", "train_clef_local.py", "research_expansion_artifacts.py",
                    "LICENSE", "source_base_model_card.md", "evaluation.json",
                    "live-training-snapshot.json"}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, lower=0, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be finite and within its bounds")
    return value


def _parity(value, tolerance, name):
    if not isinstance(value, dict) or value.get("matched") is not True:
        raise ValueError(f"Missing actual {name} prediction parity")
    _positive(value.get("records"), f"{name} records")
    bound = _number(value.get("tolerance"), f"{name} tolerance", upper=tolerance)
    if bound == 0 or _number(value.get("max_probability_difference"), f"{name} drift") > bound:
        raise ValueError(f"Invalid measured {name} prediction drift")


def _validate_training(training):
    expected = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                "dataset": DATASET_ID, "dataset_revision": compact.EXPANDED_DATASET_REVISION,
                "input_raw_row_counts": compact.EXPANDED_INPUT_ROWS,
                "input_parquet_sha256": compact.EXPANDED_INPUT_PARQUET_SHA256,
                "backend": "cuda", "cpu_offload": False, "cpu_fallback": False,
                "vision_trained": False, "autoExecute": False, "run_mode": "full",
                "all_planned_steps_completed": True, "complete_selected_training_epochs": True,
                "selected_training_scope": "all context-eligible prepared rows"}
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("CUDA standalone export requires full verified training on the exact expanded source")
    if any(type(training.get(key)) is not bool for key, value in expected.items() if type(value) is bool):
        raise ValueError("CUDA training safety/completion flags must be actual booleans")
    if str(training.get("device", "")).split(":")[0] != "cuda":
        raise ValueError("CUDA training must record an actual CUDA device")
    if training.get("status") not in {"trained_and_reload_verified", "trained_and_standalone_reload_verified"}:
        raise ValueError("CUDA training has not completed saved adapter/head reload verification")
    security = training.get("security_filter", {})
    if security.get("applied") is not True or security.get("version") != 1 or security.get("published_source_modified") is not False:
        raise ValueError("CUDA training lacks preparation-only signing-material exclusions")
    exclusions = security.get("exclusion_counts", {})
    if exclusions.get("train_excluded_signing_byte_array", 0) < 11 or exclusions.get("train_excluded_encoded_signing_literal", 0) < 3:
        raise ValueError("CUDA training did not preserve the verified source security exclusions")
    lora = training.get("lora", {})
    if lora.get("rank") != 8 or lora.get("text_layers") != [60, 61, 62, 63]:
        raise ValueError("CUDA export requires the original final-four-layer rank-eight adapter")
    cohort = training.get("cohorts", {}).get("train", {})
    selected = _positive(cohort.get("selected"), "selected train records")
    if training.get("trained_records") != selected or cohort.get("prepared") != training.get("prepared_split_counts", {}).get("train"):
        raise ValueError("Full CUDA training must cover every selected native training record")
    if selected + len(cohort.get("excluded", [])) != cohort.get("prepared"):
        raise ValueError("CUDA full-cohort context exclusions are incomplete")
    batch = _positive(training.get("microbatch_size", 1), "microbatch size")
    accumulation = _positive(training.get("gradient_accumulation"), "gradient accumulation")
    epochs = _positive(training.get("epochs"), "epochs")
    planned = epochs * math.ceil(selected / (batch * accumulation))
    if training.get("optimizer_steps") != planned or training.get("planned_steps") != planned:
        raise ValueError("CUDA optimizer steps do not cover all selected training epochs")
    order_hash = training.get("training_order_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", order_hash) or training.get("trained_record_ids_sha256") != order_hash:
        raise ValueError("Full CUDA training lacks exact selected-record coverage hashes")
    if not re.fullmatch(r"[0-9a-f]{64}", training.get("stage_manifest_sha256", "")):
        raise ValueError("CUDA continuation lacks an immutable migration staging manifest hash")
    if training.get("initial_trainable_fingerprints") != training.get("pilot_trainable_fingerprints"):
        raise ValueError("CUDA continuation did not initialize from the exact saved pilot bytes")
    if type(training.get("use_kernels")) is not bool:
        raise ValueError("CUDA training must explicitly record its kernel math")
    if training["use_kernels"]:
        kernel = training.get("kernel_execution", {})
        if kernel.get("implementation") != "installed_fla" or kernel.get("fla_core_version") != "0.5.2" or kernel.get("hub_kernel_downloads") is not False:
            raise ValueError("CUDA training lacks its pinned installed-only kernel provenance")
        _positive(kernel.get("cuda_forward_calls"), "actual CUDA FLA forward calls")
        _positive(kernel.get("cuda_backward_calls"), "actual CUDA FLA backward calls")
    capacity = training.get("capacity_probe", {})
    if capacity.get("trained") is not True or type(capacity.get("tokens")) is not int or capacity["tokens"] != cohort.get("max_tokens") or capacity["tokens"] > training.get("max_length", 0):
        raise ValueError("Full CUDA training lacks its complete longest-record capacity proof")
    sources = cohort.get("sources", {})
    if not all(any(paper in source for source in sources) for paper in ("2605.12151", "2606.08232")) or not any("clawd-ws.fly.dev" in source for source in sources):
        raise ValueError("CUDA full cohort lacks the required paper and live sources")
    for kind in ("lora", "head"):
        before = training.get("initial_trainable_fingerprints", {}).get(kind, {})
        after = training.get("trained_trainable_fingerprints", {}).get(kind, {})
        finite = training.get("finite_trainables", {}).get(kind, {})
        if (training.get("gradient_evidence", {}).get(kind) is not True
            or not re.fullmatch(r"[0-9a-f]{64}", before.get("sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", after.get("sha256", ""))
            or before["sha256"] == after["sha256"]
            or training.get("trainable_fingerprints", {}).get(kind) != after
            or finite.get("finite") is not True or finite.get("parameters") != after.get("parameters")
            or not isinstance(after.get("parameters"), int) or after["parameters"] < 1):
            raise ValueError("CUDA head and LoRA lack finite gradient, changed-byte and saved-reload evidence")
    _parity(training.get("reload_verification"), 0.001, "adapter reload")
    _parity(training.get("migration_verification"), 0.001, "pilot migration")
    if training["migration_verification"]["records"] != 16:
        raise ValueError("CUDA migration must compare every saved pilot eval/test record")


def _verify_trained_bytes(model, training):
    parameters = [(parameter, parameter.requires_grad) for name, parameter in model.named_parameters()
                  if "lora_" in name or name.startswith("head.")]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(True)
        if trainable_fingerprints(model) != training["trainable_fingerprints"]:
            raise ValueError("Actual trained CUDA head/LoRA bytes differ from the verified saved adapter")
    finally:
        for parameter, required in parameters:
            parameter.requires_grad_(required)


def _assert_cuda_model(model, *, unmerged=False):
    import bitsandbytes as bnb
    import torch
    if not torch.cuda.is_available() or any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError("Actual CUDA weights are required; CPU/MPS/offloaded models cannot prove CUDA export")
    backbone = model.language_model.get_base_model() if hasattr(model.language_model, "get_base_model") else model.language_model
    quantized = [module for module in backbone.modules() if isinstance(module, bnb.nn.Linear4bit)]
    if len(quantized) != 496 or any(layer.weight.quant_state is None or layer.weight.quant_state.quant_type != "nf4" or not layer.weight.quant_state.nested for layer in quantized):
        raise ValueError("The complete original NF4 double-quantized backbone is required")
    for layer in quantized:
        state = layer.weight.quant_state
        if any(tensor.device.type != "cuda" for tensor in (state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code)):
            raise ValueError("NF4 quantization state was offloaded from CUDA")
    if any(parameter.dtype != torch.bfloat16 for parameter in backbone.model.visual.parameters()) or any(isinstance(layer, bnb.nn.Linear4bit) for layer in backbone.model.visual.modules()):
        raise ValueError("Original vision must remain unquantized BF16")
    output_weight = backbone.get_output_embeddings().weight
    if isinstance(output_weight, bnb.nn.Params4bit) or output_weight.dtype != torch.bfloat16:
        raise ValueError("Original readable output embedding must remain BF16")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.head.parameters()):
        raise ValueError("The trained native head must retain BF16 bytes")
    if unmerged:
        modules = compact._adapted_modules(model)
        if {int(re.search(r"\.language_model\.layers\.(\d+)\.", name).group(1)) for name in modules} != {60, 61, 62, 63}:
            raise ValueError("Only final-four-layer text adapters may be merged")
        from peft.tuners.lora.layer import LoraLayer
        layers = [layer for layer in backbone.modules() if isinstance(layer, LoraLayer)]
        if any(set(layer.r.values()) != {8} or not isinstance(layer.base_layer, bnb.nn.Linear4bit) for layer in layers):
            raise ValueError("Only saved rank-eight NF4 text adapters may be merged")


@contextmanager
def _curated_base(base_dir):
    """Share only explicitly permitted source files; never copy caches/locks."""
    base = Path(base_dir)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("An existing regular quantized base directory is required")
    manifest = json.loads((base / "conversion_manifest.json").read_text())
    weights = set(manifest.get("output_weight_files", {}))
    if not weights or any(not re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name) for name in weights):
        raise ValueError("Base has unsafe or missing quantized shard references")
    if set(manifest.get("preserved_sidecars", {})) - BASE_SIDECARS:
        raise ValueError("Base contains unsupported preserved sidecar references")
    names = weights | {name for name in BASE_SIDECARS if (base / name).exists()}
    with tempfile.TemporaryDirectory(prefix=".clef-cuda-export-source-", dir=base.resolve().parent) as temporary:
        curated = Path(temporary)
        for name in sorted(names):
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("CUDA export source files must be regular files")
            os.link(path, curated / name)
        yield curated


def _portable_training(metadata):
    """Keep source hashes while removing machine-specific path fields."""
    removed = []
    def visit(value, key="", trail=""):
        if isinstance(value, dict):
            return {name: result for name, item in value.items()
                    if (result := visit(item, name, f"{trail}.{name}")) is not _REMOVED}
        if isinstance(value, list):
            return [result for index, item in enumerate(value)
                    if (result := visit(item, key, f"{trail}[{index}]")) is not _REMOVED]
        if isinstance(value, str) and re.search(r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)", value):
            if re.search(r"(?:path|dir|directory|location)$", key, re.IGNORECASE):
                removed.append(trail.lstrip("."))
                return _REMOVED
            raise ValueError("Public CUDA metadata contains a machine-specific home path outside a path field")
        return copy.deepcopy(value)
    result = visit(metadata)
    if removed:
        import hashlib
        result["path_privacy"] = {"removed_path_fields": removed,
            "source_metadata_payload_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    return result


_REMOVED = object()


def _model_card(training, evaluation, *, reload_verified):
    selected = training["cohorts"]["train"]["selected"]
    state = "Fresh CUDA standalone reload verified" if reload_verified else "Exported; fresh CUDA standalone reload remains pending"
    lines = ["---", "license: apache-2.0", "library_name: transformers", "base_model: Cloudflare/clef",
        "base_model_relation: finetune", "datasets:", f"- {DATASET_ID}",
        "tags:", "- clef", "- solana", "- structured-output", "- custom-code", "- nf4", "- cuda", "---", "",
        "# Solana Clawd Clef native CUDA decision model", "", f"**{state}.**", "",
        f"Original 27B Clef source: `{MODEL_ID}@{MODEL_REVISION}`. Research source: `{DATASET_ID}@{compact.EXPANDED_DATASET_REVISION}`.", "",
        f"Completed full training: {training['optimizer_steps']} actual optimizer steps across {selected} distinct selected records, "
        f"{training['epochs']} epoch(s). Raw published conversations total 83,662; preparation-only signing exclusions and whole-record context exclusions are retained in `training.json`.", "",
        "Rank-eight LoRA on text layers 60–63 is merged into the NF4 double-quantized backbone. The native joint decision head is trained separately and included. "
        "Original vision and output embeddings remain BF16/frozen. This is parameter-efficient fine-tuning of the original 27B model, not training from scratch.", "",
        "The task is typed reference-answer selection and observed-field readback. No free-form chat generation, signing, automatic trading, official Decision Index score, "
        "future trading-outcome improvement or unseen-document benchmark is claimed. Split source documents can overlap; inherited answer distractors are not independently verified negatives.", "",
        "## Native runtime", "",
        "Download the complete pinned release to a local directory, then use the CUDA-specific loader and the native module associated with the returned model:", "",
        "```python", "import sys", "from export_clef_cuda_release import load_cuda_release",
        'model, processor = load_cuda_release("LOCAL_RELEASE_DIRECTORY")',
        "native = sys.modules[model.__class__.__module__]",
        "response = native.systemone(model, processor, request)", "```", "",
        "Fresh read-only observations come from the surrounding runtime; the tokenizer does not browse. "
        "Run `python export_clef_cuda_release.py --model LOCAL_RELEASE_DIRECTORY --output inference-evidence.json` "
        "to capture observations after model loading and require received evidence no older than 30 seconds. "
        "Brain/Hands separation is preserved: signing material is excluded and no transaction execution is available.", "",
        "## Evaluation", "", "Actual post-training measurements are retained in `evaluation.json`; their scope is native answer selection, not profitability.", "",
        "| Split | Records | Accuracy | NLL | Multiclass Brier |", "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("eval", "test"):
        metrics = evaluation.get(split, {})
        if metrics:
            lines.append(f"| {split} | {metrics.get('records')} | {metrics.get('accuracy')} | {metrics.get('nll')} | {metrics.get('multiclass_brier')} |")
    lines += ["", "The pre-training baseline is separately labeled in the metrics. Merge and fresh saved-model reload measurements are recorded in `release.json`; "
        "a pending export cannot claim completed runtime verification.", "", "## License and research attribution", "",
        "The pinned base model license is Apache-2.0. Exact source `LICENSE` and `source_base_model_card.md` are retained. "
        "Dataset/paper material keeps its own license and attribution; see the linked dataset card for mixed-source provenance.", "",
        "- Kamat, A. U. (2026). RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels. arXiv:2605.12151. https://arxiv.org/abs/2605.12151",
        "- Kamat, A. U. (2026). Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading. arXiv:2606.08232. https://arxiv.org/abs/2606.08232", "",
        "Consumed paper revisions are RED-2400 v2 and Hour-Aware v1; the latter's full title is “Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework.” "
        "Paper-derived questions are not the complete RED benchmark or a reproduction of either paper's results.", ""]
    return "\n".join(lines)


def export_release(model, base_dir, output_dir, metadata, probe, *, evaluation=None,
                   training_snapshot=None, merge_tolerance=0.005, reserve_bytes=2 * 1024 ** 3):
    """Export completed full CUDA training; fresh standalone reload is pending."""
    _validate_training(metadata)
    _assert_cuda_model(model, unmerged=True)
    _verify_trained_bytes(model, metadata)
    output = Path(output_dir)
    compact._fresh_destination(Path(base_dir), output)
    if not isinstance(evaluation, dict) or not evaluation:
        raise ValueError("Actual post-training evaluation metrics are required")
    from clef_live_tape import validate_snapshot
    if training_snapshot is None:
        raise ValueError("The exact sanitized live training snapshot is required for reproducibility")
    snapshot_path = Path(training_snapshot)
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ValueError("Training snapshot must be a regular file")
    snapshot = json.loads(snapshot_path.read_text())
    validate_snapshot(snapshot)
    recorded = metadata.get("live_training_snapshot", {})
    if model_input_exclusion_reason(snapshot) or _sha256(snapshot_path) != recorded.get("sha256") or snapshot["captured_at"] != recorded.get("captured_at"):
        raise ValueError("Safe live training snapshot differs from its recorded provenance")
    portable_metadata = _portable_training(metadata)
    portable_evaluation = _portable_training(evaluation)
    json.dumps(portable_metadata, allow_nan=False)
    json.dumps(portable_evaluation, allow_nan=False)
    with _curated_base(base_dir) as curated:
        for name in BASE_SIDECARS:
            path = curated / name
            if path.is_file() and path.suffix == ".json":
                original = json.loads(path.read_text())
                if _portable_training(original) != original:
                    raise ValueError("Immutable source sidecars contain private home paths; curate those before export")
        manifest = compact.export_release(model, curated, output, portable_metadata, probe,
            evaluation=portable_evaluation, merge_tolerance=merge_tolerance, reserve_bytes=reserve_bytes)
    shutil.copyfile(Path(__file__), output / "export_clef_cuda_release.py")
    for name in ("train_clef_cuda.py", "stage_clef_cloud_migration.py"):
        source = Path(__file__).with_name(name)
        if source.is_file():
            shutil.copyfile(source, output / source.name)
    shutil.copyfile(snapshot_path, output / "live-training-snapshot.json")
    if _sha256(output / "live-training-snapshot.json") != recorded["sha256"]:
        raise ValueError("Live training snapshot changed during CUDA export")
    (output / "README.md").write_text(_model_card(portable_metadata, portable_evaluation, reload_verified=False))
    manifest.update(execution_backend="cuda", device="cuda", cpu_offload=False,
                    loader="from export_clef_cuda_release import load_cuda_release",
                    verification_scope="CUDA native adapter-to-merge drift and exact untouched backbone tensors; fresh CUDA standalone reload remains pending.")
    _write(output / "release.json", manifest)
    manifest = refresh_release_manifest(output)
    verify_cuda_release(output, require_reload=False)
    return manifest


def verify_cuda_release(output_dir, *, require_reload=True):
    """Hash all local artifacts before importing their native model code."""
    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("A local regular CUDA standalone release is required")
    manifest = compact.verify_mps_release(output) if require_reload else verify_release(output)
    actual = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if path.is_symlink():
            raise ValueError("CUDA standalone artifacts may not contain symlinks")
        if "__pycache__" in relative.parts:
            continue
        if any(part.startswith(".") for part in relative.parts) or path.name.endswith(".lock") or "failed-before" in path.name:
            raise ValueError("CUDA standalone contains operational or failed-conversion files")
        if path.is_file() and path.name != "release.json":
            actual.add(relative.as_posix())
    if actual != set(manifest.get("files", {})) or not REQUIRED_RUNTIME <= actual:
        raise ValueError("CUDA standalone runtime or complete hash inventory is missing")
    if manifest.get("execution_backend") != "cuda" or manifest.get("cpu_offload") is not False or str(manifest.get("device", "")).split(":")[0] != "cuda":
        raise ValueError("Standalone release lacks explicit CUDA backend provenance")
    training = json.loads((output / "training.json").read_text())
    if manifest.get("training") != training or manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("CUDA standalone source or training metadata is inconsistent")
    _validate_training(training)
    if _portable_training(training) != training:
        raise ValueError("CUDA release still contains machine-specific home paths")
    from clef_live_tape import validate_snapshot
    snapshot_path = output / "live-training-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    validate_snapshot(snapshot)
    recorded_snapshot = training.get("live_training_snapshot", {})
    if model_input_exclusion_reason(snapshot) or recorded_snapshot.get("file") != snapshot_path.name or recorded_snapshot.get("sha256") != _sha256(snapshot_path) or recorded_snapshot.get("captured_at") != snapshot["captured_at"]:
        raise ValueError("CUDA release lacks the exact safe historical training snapshot")
    if training.get("use_kernels") is True and not {"train_clef_cuda.py", "train_clef_local.py", "research_expansion_artifacts.py"} <= actual:
        raise ValueError("CUDA standalone lacks its verified installed-kernel runtime helpers")
    _parity(manifest.get("merge_verification"), 0.005, "quantized merge")
    if require_reload:
        load = manifest.get("cuda_standalone_load", {})
        if load.get("backend") != "cuda" or load.get("actual_cuda_parameters") is not True or load.get("fresh_model_instance") is not True:
            raise ValueError("CUDA standalone has no genuine saved-model CUDA load evidence")
        kernels = load.get("kernel_execution", {})
        expected_implementation = "installed_fla" if training["use_kernels"] else "native_torch"
        if kernels.get("implementation") != expected_implementation or kernels.get("hub_kernel_downloads") is not False:
            raise ValueError("CUDA standalone reload used different kernel math")
        _positive(kernels.get("cuda_forward_calls"), "fresh native CUDA forward calls")
    return manifest


def _configure_saved_kernels(path, enabled):
    source = Path(path) / "train_clef_cuda.py"
    name = "clef_verified_cuda_kernel_helper_" + _sha256(source)[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError("Verified installed CUDA kernel helper is unavailable")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    sys.path.insert(0, str(path))
    try:
        spec.loader.exec_module(helper)
        return helper.configure_local_kernels(enabled)
    finally:
        sys.path.pop(0)


def load_cuda_release(output_dir, *, device="cuda", dtype=None):
    """Load verified saved standalone weights only; no Hub fallback or offload."""
    import torch
    from export_clef_release import load_release_model
    path = Path(output_dir).resolve()
    manifest = verify_cuda_release(path, require_reload=False)
    if str(device).split(":")[0] != "cuda" or not torch.cuda.is_available():
        raise ValueError("Actual CUDA is required for standalone load verification")
    manifest_hash = _sha256(path / "release.json")
    kernels = _configure_saved_kernels(path, manifest["training"]["use_kernels"])
    model, processor = load_release_model(path, device=device, dtype=dtype or torch.bfloat16,
        local_files_only=True, attn_implementation="eager")
    _assert_cuda_model(model)
    model._clef_cuda_standalone_load = {"backend": "cuda", "device": str(device),
        "actual_cuda_parameters": True, "fresh_model_instance": True,
        "model_path": str(path), "source_release_manifest_sha256": manifest_hash,
        "kernel_execution": kernels,
        "loaded_at": datetime.now(timezone.utc).isoformat()}
    return model, processor


def mark_reload_verified(output_dir, merged_predictions, reloaded_predictions, *, loaded_model, tolerance=0.0001):
    path = Path(output_dir).resolve()
    verify_cuda_release(path, require_reload=False)
    _assert_cuda_model(loaded_model)
    load = getattr(loaded_model, "_clef_cuda_standalone_load", {})
    if load.get("model_path") != str(path) or load.get("source_release_manifest_sha256") != _sha256(path / "release.json"):
        raise ValueError("Reload predictions must come from a fresh CUDA load of this exact saved release")
    manifest = compact.mark_reload_verified(path, merged_predictions, reloaded_predictions, tolerance=tolerance)
    manifest["cuda_standalone_load"] = {key: value for key, value in load.items() if key != "model_path"}
    manifest["cuda_standalone_load"]["artifact_root"] = "."
    card = path / "README.md"
    card.write_text(_model_card(manifest["training"], json.loads((path / "evaluation.json").read_text()), reload_verified=True))
    manifest["files"]["README.md"] = {"bytes": card.stat().st_size, "sha256": _sha256(card)}
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    manifest["verification_scope"] = "CUDA native quantized merge and fresh saved standalone prediction parity, with exact untouched tensors and complete artifact hashes."
    _write(path / "release.json", manifest)
    return verify_cuda_release(path)


def _native_responses(model, module, processor, rows, max_length, *, device):
    """Shared native scoring; production callers independently require CUDA."""
    import torch
    for row in rows:
        if model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]}):
            raise ValueError("Live model input contains credential or signing material")
    encoded, kept, excluded = encode_rows(module, processor, rows, max_length)
    if not kept:
        raise ValueError("No complete safe live record fits the context limit")
    responses = []
    model.eval()
    with torch.inference_mode():
        for record, row in zip(encoded, kept, strict=True):
            logits = model(module.collate_records([record], processor.tokenizer.pad_token_id, device))[0]
            answers = {}
            for question, scores in zip(record.questions, logits, strict=True):
                values = scores.float().softmax(-1)
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Native CUDA inference produced nonfinite probabilities")
                answers[question.question_id] = module.systemone_answer(row["questions"][question.question_id],
                    dict(zip(question.option_ids, values.cpu().tolist(), strict=True)))
            responses.append({"id": row["id"], "answers": answers,
                              "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0}})
    return {"responses": responses, "excluded": excluded}


def _time(value):
    if not isinstance(value, str):
        raise ValueError("A recorded timezone-aware inference time is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Inference times must include a timezone")
    return parsed.astimezone(timezone.utc)


def verify_live_inference(proof_path, release_dir, *, now=None, max_age_seconds=30):
    from clef_live_tape import live_decision_records, snapshot_freshness
    path, release = Path(proof_path), Path(release_dir).resolve()
    manifest = verify_cuda_release(release)
    if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(release):
        raise ValueError("Fresh CUDA inference evidence must be a separate regular file outside the release")
    proof = json.loads(path.read_text())
    allowed = {"responses", "excluded", "model_path", "model_revision", "dataset_revision", "training_status",
               "backend", "device", "standalone_model_loaded", "release_manifest_sha256", "captured_at",
               "inference_completed_at", "live_snapshot", "freshness", "source", "scope"}
    if not isinstance(proof, dict) or set(proof) - allowed:
        raise ValueError("CUDA inference evidence contains unsupported fields")
    if model_input_exclusion_reason(proof):
        raise ValueError("Inference evidence contains signing material")
    if (proof.get("backend") != "cuda" or str(proof.get("device", "")).split(":")[0] != "cuda"
        or proof.get("standalone_model_loaded") is not True or proof.get("training_status") != "trained_and_standalone_reload_verified"
        or Path(proof.get("model_path", "")).resolve() != release
        or proof.get("model_revision") != MODEL_REVISION or proof.get("dataset_revision") != compact.EXPANDED_DATASET_REVISION
        or proof.get("release_manifest_sha256") != _sha256(release / "release.json")
        or proof.get("source") != "https://clawd-ws.fly.dev/"):
        raise ValueError("Fresh inference is not bound to this verified CUDA release")
    _number(max_age_seconds, "maximum observation age", upper=30)
    if max_age_seconds == 0:
        raise ValueError("Maximum observation age must be positive")
    captured, completed = _time(proof.get("captured_at")), _time(proof.get("inference_completed_at"))
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or captured > completed or completed > now or (completed - captured).total_seconds() > max_age_seconds:
        raise ValueError("CUDA inference did not use a fresh observation at its actual execution time")
    snapshot = proof.get("live_snapshot", {})
    if snapshot.get("captured_at") != proof.get("captured_at"):
        raise ValueError("CUDA inference capture time differs from its actual snapshot")
    freshness = snapshot_freshness(snapshot, now=completed, max_age_seconds=max_age_seconds)
    if freshness["stale"] or proof.get("freshness") != freshness:
        raise ValueError("CUDA inference contains stale or inconsistent received observations")
    responses = proof.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("CUDA inference has no actual native responses")
    expected = {row["id"]: row for row in live_decision_records(snapshot)}
    observed_ids = set()
    for response in responses:
        if not isinstance(response, dict) or response.get("id") not in expected or response["id"] in observed_ids:
            raise ValueError("CUDA response does not identify a unique actual captured observation")
        observed_ids.add(response["id"])
        _positive(response.get("usage", {}).get("input_tokens"), "native input tokens")
        if response["usage"].get("output_tokens") != 0 or not response.get("answers"):
            raise ValueError("Expected native typed decisions without generated text")
        row = expected[response["id"]]
        if set(response["answers"]) != set(row["questions"]):
            raise ValueError("Native CUDA response questions differ from the observed snapshot")
        for question_id, answer in response["answers"].items():
            probabilities = answer.get("probabilities")
            if answer.get("type") != "choice" or not isinstance(probabilities, dict) or len(probabilities) < 2 or answer.get("choice") not in probabilities:
                raise ValueError("CUDA inference lacks actual native choice probabilities")
            if set(probabilities) != set(row["questions"][question_id]["criteria"]):
                raise ValueError("CUDA response options differ from the native observed-field record")
            for probability in probabilities.values():
                _number(probability, "native probability", upper=1)
            if abs(sum(probabilities.values()) - 1) > 0.005 or probabilities[answer["choice"]] + 0.0001 < max(probabilities.values()):
                raise ValueError("Native probabilities or selected CUDA choice are invalid")
            confidence = _number(answer.get("confidence"), "native confidence", upper=1)
            if abs(confidence - probabilities[answer["choice"]]) > 0.001:
                raise ValueError("Native CUDA confidence differs from its selected probability")
    return proof


def run_live_inference(release_dir, proof_path, *, max_length=2048):
    """Load a full verified CUDA model, then capture and score fresh safe tape."""
    import torch
    from clef_live_tape import capture_live_snapshot, live_decision_records, snapshot_freshness
    _positive(max_length, "native context limit")
    release, proof_path = Path(release_dir).resolve(), Path(proof_path)
    if proof_path.is_symlink() or proof_path.resolve().is_relative_to(release):
        raise ValueError("Write runtime evidence outside the immutable standalone release")
    manifest = verify_cuda_release(release)
    model, processor = load_cuda_release(release)
    module = sys.modules[model.__class__.__module__]
    snapshot = capture_live_snapshot()
    if model_input_exclusion_reason(snapshot):
        raise ValueError("Fresh snapshot contains signing material")
    rows = live_decision_records(snapshot)
    result = _native_responses(model, module, processor, rows, max_length, device="cuda")
    torch.cuda.synchronize()
    completed = datetime.now(timezone.utc)
    freshness = snapshot_freshness(snapshot, now=completed, max_age_seconds=30)
    if freshness["stale"]:
        raise ValueError("Observation expired during actual CUDA inference; a fresh attempt is required")
    result.update(model_path=str(release), model_revision=MODEL_REVISION,
        dataset_revision=compact.EXPANDED_DATASET_REVISION, training_status=manifest["training"]["status"],
        backend="cuda", device="cuda", standalone_model_loaded=True,
        release_manifest_sha256=_sha256(release / "release.json"), captured_at=snapshot["captured_at"],
        inference_completed_at=completed.isoformat(), live_snapshot=snapshot, freshness=freshness,
        source="https://clawd-ws.fly.dev/", scope="Native observed-field readback; no signing, trading or future-outcome claim")
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    _write(proof_path, result)
    return verify_live_inference(proof_path, release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--verify-only", action="store_true", help="Hash/check a completed release without loading weights or connecting to live tape")
    args = parser.parse_args()
    if args.verify_only:
        verified = verify_cuda_release(args.model)
        result = {"status": verified["status"], "backend": "cuda", "model_loaded": False}
    else:
        if args.output is None:
            parser.error("--output is required for fresh CUDA inference evidence")
        result = run_live_inference(args.model, args.output, max_length=args.max_length)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

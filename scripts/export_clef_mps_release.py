"""Export trained MPS Clef by sharing immutable shards and rewriting adapted ones.

The caller must first save and reload-verify its adapter/head. Merging mutates the
caller's in-memory backbone, so an unsuccessful merge must be recovered from that
saved adapter. No source checkpoint file is ever opened for writing. Unchanged
output shards are hardlinks and survive removal of the source directory.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION, immutable_revision
from clef_research_training import compare_predictions
from export_clef_release import (_backbone_inventory, _joint_head_config, _sha256,
                                 verify_release)

GIB = 1024 ** 3
EXPANDED_DATASET_REVISION = "58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2"
EXPANDED_INPUT_ROWS = {"train": 78171, "eval": 2595, "test": 2896}
EXPANDED_INPUT_PARQUET_SHA256 = {
    "train": "4d3d6b4b3030746cd29116a3f6d1a2f4c39ddc90119a43b39569085635fbb93d",
    "eval": "707aa9528afe89637fe00a134c9c993554bc2a63ba9c946db6698ff24b8a282e",
    "test": "44129a4c88930a008ddba8104705b6d223d395cd070b45ad9189be326dcea930",
}


def _fresh_destination(base: Path, output: Path) -> None:
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Standalone release requires a separate new or empty directory.")
    base_real, output_real = base.resolve(), output.resolve()
    if output_real.is_relative_to(base_real) or base_real.is_relative_to(output_real):
        raise ValueError("Base and release directories must be separate siblings, not nested.")
    parent = output_real.parent
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != base_real.stat().st_dev:
        raise ValueError("Hardlinked standalone export requires base and release on the same filesystem.")


def validate_base(base: Path, expected_source: dict | None = None) -> tuple[dict, dict]:
    """Validate completed conversion, exact shard index, and original hashes."""
    base = Path(base)
    if not base.is_dir() or base.is_symlink():
        raise ValueError("A local standalone base directory is required; remote downloads are disabled.")
    manifest = json.loads((base / "conversion_manifest.json").read_text())
    expected_source = expected_source or {"repo_id": MODEL_ID, "revision": MODEL_REVISION}
    if manifest.get("source") != expected_source:
        raise ValueError("Quantized base source differs from the required immutable source.")
    if manifest.get("status") != "complete" or not manifest.get("reload_verified"):
        raise ValueError("Quantized base conversion must be complete and standalone reload verified.")
    inventory = _backbone_inventory(base)
    shard_metadata = manifest.get("output_weight_files", {})
    if set(shard_metadata) != set(inventory["shards"]):
        raise ValueError("Conversion manifest differs from the complete backbone shard inventory.")
    expected_files = {**shard_metadata, **manifest.get("preserved_sidecars", {})}
    for filename, expected in expected_files.items():
        if Path(filename).name != filename:
            raise ValueError("Unsafe quantized base manifest path.")
        path = base / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Immutable base hash mismatch: {filename}")
    config = json.loads((base / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    if not quantization.get("load_in_4bit") or quantization.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("Expected the standalone native NF4 base.")
    if not {"model.visual", "lm_head"} <= set(quantization.get("llm_int8_skip_modules", [])):
        raise ValueError("The quantized base must preserve BF16 vision and readable output embeddings.")
    # Capture all regular top-level files, including config/index/source records;
    # local Hub download cache locks are operational metadata, not model assets.
    hashes = {}
    for path in sorted(base.iterdir()):
        if path.name == ".cache":
            continue
        if path.is_symlink():
            raise ValueError("Base artifacts must not be symlinks.")
        if path.is_file():
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {"conversion": manifest, "backbone": inventory}, hashes


def _weight_map(base: Path, inventory: dict) -> dict:
    if inventory["index"]:
        return json.loads((base / inventory["index"]).read_text())["weight_map"]
    return {name: "model.safetensors" for name in inventory["tensor_names"]}


def _adapted_modules(model) -> list[str]:
    from peft.tuners.lora.layer import LoraLayer
    if not callable(getattr(model.language_model, "merge_and_unload", None)):
        raise ValueError("Export requires a PEFT backbone with an unmerged saved adapter.")
    backbone = model.language_model.get_base_model()
    names = [name for name, module in backbone.named_modules() if isinstance(module, LoraLayer)]
    if not names or any(".language_model.layers." not in name for name in names):
        raise ValueError("Only explicit text transformer-layer LoRA modules may be merged.")
    for name, parameter in model.language_model.named_parameters():
        if parameter.requires_grad and "lora_" not in name:
            raise ValueError(f"Unexpected trainable base parameter: {name}")
    return sorted(names)


def _affected(key: str, modules: list[str]) -> bool:
    return any(key.startswith(name + ".") for name in modules)


def _verify_unchanged_tensors(base: Path, weight_map: dict, state: dict, modules: list[str]) -> None:
    """Compare one tensor at a time; no whole-model CPU copy is created."""
    import torch
    from safetensors import safe_open
    for filename in sorted(set(weight_map.values())):
        with safe_open(base / filename, framework="pt", device="cpu", backend="pread") as handle:
            for key in handle.keys():
                if _affected(key, modules):
                    continue
                expected = handle.get_tensor(key)
                actual = state[key].detach().cpu()
                if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
                    raise ValueError(f"Untouched backbone tensor changed: {key}")
                del actual, expected


def _inventory_files(output: Path) -> dict:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.name == "release.json" or "__pycache__" in path.relative_to(output).parts:
            continue
        if path.is_symlink():
            raise ValueError("A standalone release may not contain symlinks.")
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return files


def export_release(model: Any, base_dir: Path, output_dir: Path, metadata: dict,
                   probe: Callable[[Any], list[dict]], *, merge_tolerance: float = 0.005,
                   expected_source: dict | None = None, reserve_bytes: int = 2 * GIB,
                   evaluation: dict | None = None) -> dict:
    """Merge, measure drift, and export without duplicating unchanged large shards.

``probe`` returns the nonempty native prediction list from evaluate(...)[1].
The function mutates model.language_model and returns an exported manifest;
mark_reload_verified must separately record a fresh standalone runtime reload.
``expected_source`` defaults to pinned genuine Clef; tiny architecture tests use
an explicitly named, hashed fixture source rather than claiming production data.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base, output = Path(base_dir), Path(output_dir)
    _fresh_destination(base, output)
    if metadata.get("status") != "trained_and_reload_verified" or metadata.get("optimizer_steps", 0) < 1:
        raise ValueError("Export requires actual training and saved adapter/head reload verification.")
    if not math.isfinite(merge_tolerance) or not 0 < merge_tolerance <= 0.005:
        raise ValueError("Merge tolerance must be positive and no greater than 0.005.")
    if not callable(probe):
        raise ValueError("Native decision prediction probe is required.")
    # Validate metadata and source before mutating caller state or writing files.
    training = json.loads(json.dumps(metadata))
    evaluation = evaluation if evaluation is not None else training.get("evaluation")
    if evaluation is not None and (not isinstance(evaluation, dict) or not evaluation):
        raise ValueError("Evaluation must contain actual structured nonempty metrics.")
    evaluation_json = json.dumps(evaluation, indent=2, allow_nan=False) + "\n" if evaluation is not None else None
    source, original_hashes = validate_base(base, expected_source)
    if not (base / "README.md").is_file() or not (base / "LICENSE").is_file():
        raise ValueError("The original source model card and license must be retained.")
    mapping = _weight_map(base, source["backbone"])
    modules = _adapted_modules(model)
    changed = {filename for key, filename in mapping.items() if _affected(key, modules)}
    if not changed:
        raise ValueError("Adapted modules are absent from the original quantized shard index.")
    head_config = _joint_head_config(model.head)
    head_bytes = sum(value.numel() * value.element_size() for value in model.head.state_dict().values())
    extra = sum(original_hashes[name]["bytes"] for name in changed) + head_bytes + reserve_bytes + 64 * 1024 ** 2
    parent = output.resolve().parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < extra:
        raise RuntimeError(f"Insufficient disk for changed shards and trained head: {extra:,} bytes required.")
    before = probe(model)
    model.language_model = model.language_model.merge_and_unload(safe_merge=True)
    after = probe(model)
    parity = compare_predictions(before, after, tolerance=merge_tolerance)
    state = model.language_model.state_dict()
    if set(state) != set(mapping) or any("lora_" in key or ".base_layer." in key for key in state):
        raise ValueError("Merged quantized state differs from the complete original weight map.")
    _verify_unchanged_tensors(base, mapping, state, modules)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.name in set(mapping.values()) | {"joint_head.safetensors", "conversion_manifest.json"}:
            continue
        shutil.copyfile(path, output / path.name)
    linked = []
    for filename in sorted(set(mapping.values())):
        destination = output / filename
        if filename not in changed:
            os.link(base / filename, destination)
            linked.append(filename)
            continue
        keys = sorted(key for key, shard in mapping.items() if shard == filename)
        # This is bounded by one original output shard (normally 2 GB).
        shard = {key: state[key].detach().cpu().contiguous() for key in keys}
        with safe_open(base / filename, framework="pt", device="cpu") as handle:
            shard_metadata = handle.metadata()
        save_file(shard, str(destination), metadata=shard_metadata)
        del shard
    save_file({key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()},
              str(output / "joint_head.safetensors"))
    (output / "joint_head_config.json").write_text(json.dumps(head_config, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")
    shutil.copyfile(base / "README.md", output / "source_base_model_card.md")
    if evaluation_json is not None:
        (output / "evaluation.json").write_text(evaluation_json)
    # Native inference and strict verification work without the training repo.
    for name in ("export_clef_release.py", "export_clef_mps_release.py", "run_clef_local.py",
                 "clef_live_tape.py", "clef_research_data.py", "clef_research_training.py",
                 "research_expansion_artifacts.py", "train_clef_local.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    backbone = _backbone_inventory(output)
    backbone.pop("tensor_names")
    files = _inventory_files(output)
    # Source weights, configuration, processor and head must be byte unchanged.
    for name, expected in original_hashes.items():
        path = base / name
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Original base changed during export: {name}")
    manifest = {"format": "clef-merged-release-v1", "status": "exported_quantized_merge_pending_standalone_reload",
        "standalone_backbone": True, "source": source["conversion"]["source"],
        "backbone": backbone, "joint_head_config": head_config, "training": training,
        "merge_verification": parity, "files": files, "total_bytes": sum(item["bytes"] for item in files.values()),
        "storage": {"hardlinked_unchanged_shards": linked, "rewritten_shards": sorted(changed),
                    "base_files_unchanged": True, "new_bytes_estimate": extra - reserve_bytes},
        "loader": "from export_clef_release import load_release_model",
        "reload_verified": False,
        "verification_scope": "Native adapter-to-merged prediction drift, exact untouched tensors, source hashes and complete standalone shard inventory; fresh reload remains required."}
    (output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_release(output)
    return manifest


def load_local_release(output_dir: Path, *, device="mps", dtype=None):
    """No remote identifier fallback: only complete local release artifacts."""
    import torch
    from export_clef_release import load_release_model
    path = Path(output_dir)
    if not path.is_dir():
        raise ValueError("A local standalone release directory is required; downloads are disabled.")
    return load_release_model(path, device=device, dtype=dtype or torch.bfloat16,
                              local_files_only=True, attn_implementation="eager")


def mark_reload_verified(output_dir: Path, merged_predictions: list[dict],
                         restored_predictions: list[dict], *, tolerance=0.0001) -> dict:
    if not math.isfinite(tolerance) or not 0 < tolerance <= 0.0001:
        raise ValueError("Standalone reload tolerance must be no greater than 0.0001.")
    output = Path(output_dir)
    manifest = verify_release(output)
    verification = compare_predictions(merged_predictions, restored_predictions, tolerance=tolerance)
    training_path = output / "training.json"
    training = json.loads(training_path.read_text())
    training.update(status="trained_and_standalone_reload_verified",
                    standalone_reload_verification=verification)
    training_path.write_text(json.dumps(training, indent=2) + "\n")
    manifest["training"] = training
    manifest["files"]["training.json"] = {"bytes": training_path.stat().st_size, "sha256": _sha256(training_path)}
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    manifest.update(status="trained_merged_and_standalone_reload_verified", reload_verified=True,
                    reload_verification=verification,
                    verification_scope="Native adapter-to-merge drift, exact untouched tensors, source hashes, complete local shard inventory and fresh standalone native prediction reload.")
    (output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_mps_release(output_dir: Path, *, expected_dataset_revision: str | None = EXPANDED_DATASET_REVISION) -> dict:
    """Verify the production local inference artifact before any weights load."""
    output = Path(output_dir)
    if not output.is_dir() or output.is_symlink():
        raise ValueError("A local standalone MPS release directory is required.")
    manifest = verify_release(output)
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*")
                    if path.is_file() and path != output / "release.json"
                    and "__pycache__" not in path.relative_to(output).parts}
    recorded_files = {name for name in manifest.get("files", {}) if "__pycache__" not in Path(name).parts}
    if actual_files != recorded_files:
        raise ValueError("Standalone release hash inventory does not cover every artifact file.")
    required = {"config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json",
                "joint_schema_model.py", "joint_head.safetensors", "joint_head_config.json", "training.json"}
    if not required <= actual_files or not set(manifest["backbone"]["shards"]) <= actual_files:
        raise ValueError("Standalone release lacks required hashed native artifacts.")
    if manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Standalone release is not the pinned genuine Clef source.")
    if manifest.get("status") != "trained_merged_and_standalone_reload_verified" or manifest.get("reload_verified") is not True:
        raise ValueError("Standalone release has not completed fresh runtime reload verification.")
    training = json.loads((output / "training.json").read_text())
    if training != manifest.get("training") or training.get("status") != "trained_and_standalone_reload_verified":
        raise ValueError("Standalone release training metadata is inconsistent or unverified.")
    if training.get("model") != MODEL_ID or training.get("model_revision") != MODEL_REVISION:
        raise ValueError("Training metadata does not identify the genuine pinned Clef model.")
    if type(training.get("optimizer_steps")) is not int or training["optimizer_steps"] < 1:
        raise ValueError("Standalone release has no completed optimizer steps.")
    if training.get("dataset") != DATASET_ID:
        raise ValueError("Standalone release uses an unexpected research dataset.")
    dataset_revision = immutable_revision(training.get("dataset_revision", ""))
    if expected_dataset_revision is not None and dataset_revision != immutable_revision(expected_dataset_revision):
        raise ValueError("Standalone release research dataset revision differs from the requested pin.")
    security = training.get("security_filter", {})
    if security.get("applied") is not True or security.get("version") != 1:
        raise ValueError("Standalone release lacks the applied training-data security filter marker.")
    hashes = training.get("input_parquet_sha256", {})
    if set(hashes) != {"train", "eval", "test"} or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()
    ):
        raise ValueError("Standalone release lacks all three immutable input Parquet hashes.")
    if dataset_revision == EXPANDED_DATASET_REVISION and (
        hashes != EXPANDED_INPUT_PARQUET_SHA256 or training.get("input_raw_row_counts") != EXPANDED_INPUT_ROWS
    ):
        raise ValueError("Standalone training inputs differ from the 83,662-row published research source.")
    for key, tolerance in (("merge_verification", 0.005), ("reload_verification", 0.0001)):
        check = manifest.get(key, {})
        difference, measured_tolerance = check.get("max_probability_difference"), check.get("tolerance")
        if (check.get("matched") is not True or type(check.get("records")) is not int or check["records"] < 1
            or not isinstance(difference, (int, float)) or not math.isfinite(difference) or difference < 0
            or not isinstance(measured_tolerance, (int, float)) or not math.isfinite(measured_tolerance)
            or not 0 < measured_tolerance <= tolerance or difference > measured_tolerance):
            raise ValueError(f"Standalone release lacks a valid measured {key}.")
    if training.get("standalone_reload_verification") != manifest.get("reload_verification"):
        raise ValueError("Training and release standalone reload evidence differ.")
    return manifest

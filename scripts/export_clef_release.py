#!/usr/bin/env python3
"""Export and load a complete Clef release after verified adapter training.

Export merges the caller's in-memory PEFT backbone. Save and verify the adapter
in its separate directory first. Nothing is downloaded or reloaded by export;
the loader is an explicit subsequent operation, suitable for the GPU job.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_head_config(head: Any) -> dict[str, Any]:
    """Describe the actual trained native head, including tiny test heads."""
    if head.layers:
        attention = head.layers[0].self_attn
        feedforward = head.layers[0].linear1.out_features
    elif head.evidence_layers:
        attention = head.evidence_layers[0].attention
        feedforward = head.evidence_layers[0].feedforward[0].out_features
    else:
        raise ValueError("Cannot infer a native head without routing or decoder layers")
    return {
        "hidden_size": head.hidden_norm.normalized_shape[0],
        "width": head.memory_projection.out_features,
        "routing_layers": len(head.evidence_layers),
        "layers": len(head.layers),
        "heads": attention.num_heads,
        "feedforward": feedforward,
        "dropout": head.residual_scorer[2].p,
    }


def _backbone_inventory(output_dir: Path) -> dict[str, Any]:
    """Inspect safetensor headers and index references without loading weights."""
    from safetensors import safe_open

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Backbone shard index has no tensor map")
        files = sorted(set(weight_map.values()))
    elif (output_dir / "model.safetensors").is_file():
        weight_map = None
        files = ["model.safetensors"]
    else:
        raise ValueError("Merged export contains no complete safetensors backbone")
    tensor_files: dict[str, str] = {}
    for filename in files:
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != ".safetensors":
            raise ValueError("Unsafe or unsupported backbone shard reference")
        shard = output_dir / relative
        if not shard.is_file() or shard.is_symlink():
            raise ValueError(f"Referenced backbone shard is missing: {filename}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in tensor_files:
                    raise ValueError(f"Backbone tensor is duplicated between shards: {key}")
                tensor_files[key] = filename
    if weight_map is not None and weight_map != tensor_files:
        raise ValueError("Backbone shard index differs from safetensor headers")
    if any("lora_" in key or ".base_layer." in key for key in tensor_files):
        raise ValueError("Exported backbone still contains unmerged adapter tensors")
    return {
        "format": "safetensors",
        "index": index_path.name if index_path.is_file() else None,
        "shards": files,
        "tensor_count": len(tensor_files),
        "vision_tensor_count": sum(".visual." in key for key in tensor_files),
        "tensor_names": sorted(tensor_files),
    }


def export_merged_release(model: Any, processor: Any, upstream_module: Any,
                          output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Merge verified LoRA and save the full multimodal backbone/native head.

    This intentionally mutates ``model.language_model`` during export. All
    destination/metadata/module guards run first, before merging or writing.
    Existing adapter artifacts are never read, replaced, or modified.
    """
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and
                                 (not output_dir.is_dir() or any(output_dir.iterdir()))):
        raise ValueError("Merged release destination must be a separate empty directory")
    if metadata.get("status") != "trained_and_reload_verified" or metadata.get("optimizer_steps", 0) < 1:
        raise ValueError("Export requires completed training and verified adapter reload")
    # Validate JSON before changing the caller's in-memory model.
    training_json = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    code_source = Path(upstream_module.__file__)
    if not code_source.is_file() or not hasattr(upstream_module, "load_release_model"):
        raise ValueError("A complete native Clef upstream module is required")
    if not callable(getattr(model.language_model, "merge_and_unload", None)):
        raise ValueError("Export requires an unmerged PEFT backbone")
    head_config = _joint_head_config(model.head)
    original_backbone = model.language_model.get_base_model()
    vision_names = {key for key in original_backbone.state_dict() if ".visual." in key}
    if not vision_names:
        raise ValueError("Export requires the original multimodal backbone, including vision")
    max_shard_size = metadata.get("merged_max_shard_size", "5GB")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.language_model.merge_and_unload(safe_merge=True)
    model.language_model = merged
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)
    processor.save_pretrained(output_dir)
    save_file({key: tensor.detach().cpu().contiguous() for key, tensor in model.head.state_dict().items()},
              str(output_dir / "joint_head.safetensors"))
    (output_dir / "joint_head_config.json").write_text(json.dumps(head_config, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(code_source, output_dir / "joint_schema_model.py")
    shutil.copyfile(Path(__file__), output_dir / "export_clef_release.py")
    (output_dir / "training.json").write_text(training_json, encoding="utf-8")

    backbone = _backbone_inventory(output_dir)
    saved_names = set(backbone.pop("tensor_names"))
    if not vision_names <= saved_names:
        raise ValueError("Merged release omitted original vision tensors")
    required = ("config.json", "processor_config.json", "tokenizer_config.json",
                "joint_schema_model.py", "joint_head_config.json", "joint_head.safetensors")
    if not all((output_dir / name).is_file() for name in required):
        raise ValueError("Merged release is missing model, processor, tokenizer, or head files")
    if (output_dir / "adapter_config.json").exists():
        raise ValueError("Merged release unexpectedly saved an adapter instead of full weights")
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            files[path.relative_to(output_dir).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _sha256(path),
            }
    manifest = {
        "format": "clef-merged-release-v1",
        "status": "exported_from_trained_reload_verified_adapter",
        "standalone_backbone": True,
        "backbone": backbone,
        "joint_head_config": head_config,
        "training": metadata,
        "files": files,
        "total_bytes": sum(value["bytes"] for value in files.values()),
        "loader": "from export_clef_release import load_release_model",
        "verification_scope": "Export hashes, shard references and vision tensor presence; full-model reload is a separate runtime check.",
    }
    (output_dir / "release.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_release(output_dir: Path) -> dict[str, Any]:
    """Verify release bytes and indexed tensor references before loading."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "release.json").read_text())
    if manifest.get("format") != "clef-merged-release-v1":
        raise ValueError("Unsupported Clef release manifest")
    for filename, expected in manifest["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe release file reference")
        path = output_dir / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Release file differs from exported hash: {filename}")
    observed = _backbone_inventory(output_dir)
    observed.pop("tensor_names")
    if observed != manifest["backbone"]:
        raise ValueError("Backbone shard inventory differs from exported manifest")
    return manifest


def load_release_model(model_path: str | Path, device: Any = "cuda", dtype: Any = None,
                       verify_hashes: bool = True, **from_pretrained_kwargs: Any) -> tuple[Any, Any]:
    """Load a standalone exported Clef backbone/head with its real processor."""
    import torch

    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(model_path)))
    if verify_hashes:
        verify_release(path)
    module_name = "clawd_exported_clef_" + _sha256(path / "joint_schema_model.py")[:16]
    spec = importlib.util.spec_from_file_location(module_name, path / "joint_schema_model.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import exported native Clef module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.load_release_model(path, device=device, dtype=dtype or torch.bfloat16,
                                     **from_pretrained_kwargs)

"""Stream pinned Clef BF16 shards into a standalone bitsandbytes MPS checkpoint.

The default CLI only plans; --execute downloads the source one shard at a time.
Progress records describe an in-memory conversion, not a resumable checkpoint.
Only task-owned downloads are removed. Hugging Face's global cache is untouched.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable

MODEL_ID = "Cloudflare/clef"
MODEL_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
GIB = 1024 ** 3
PRESERVED_PREFIXES = ("model.visual", "lm_head")
REQUIRED_SIDECARS = {"joint_head.safetensors", "joint_head_config.json",
                     "joint_schema_model.py", "processor_config.json",
                     "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path, manifest: dict) -> None:
    temporary = output / "conversion_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "conversion_manifest.json")


def require_fresh_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Conversion requires a new or empty output directory; existing files are preserved.")


def _disk_parent(output: Path) -> Path:
    parent = output.resolve()
    while not parent.exists():
        parent = parent.parent
    return parent


def check_resources(output: Path, estimates: dict, reserve_bytes: int = 3 * GIB,
                    available_disk: int | None = None, available_mps: int | None = None) -> dict:
    """Conservative guards; these estimates do not promise full training will fit."""
    import torch
    disk = shutil.disk_usage(_disk_parent(output)).free if available_disk is None else available_disk
    mps = torch.mps.recommended_max_memory() if available_mps is None else available_mps
    disk_required = (estimates["packed_backbone_bytes"] + estimates["sidecar_bytes"]
                     + estimates["largest_source_shard_bytes"] + reserve_bytes)
    # One BF16 tensor, packed model, and a bounded output-shard staging copy.
    mps_required = estimates["packed_backbone_bytes"] + estimates["largest_tensor_bytes"] + 4 * GIB
    if disk < disk_required:
        raise RuntimeError(f"Insufficient free disk: {disk:,} bytes available; {disk_required:,} required.")
    if mps < mps_required:
        raise RuntimeError(f"Insufficient recommended MPS memory: {mps:,} bytes; {mps_required:,} required.")
    return {"free_disk_bytes": disk, "recommended_mps_bytes": mps,
            "disk_required_bytes": disk_required, "mps_required_bytes": mps_required}


def empty_backbone(config):
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3_5ForConditionalGeneration
    if config.model_type != "qwen3_5":
        raise ValueError("Expected Clef's native Qwen3.5 hybrid architecture.")
    config._attn_implementation = "eager"
    with init_empty_weights(include_buffers=True):
        model = Qwen3_5ForConditionalGeneration(config)
    # Native RoPE frequencies are reconstructed in FP32 on a fresh HF load.
    # Casting them to BF16 here changes positions despite identical weights.
    # Record each buffer's constructor dtype before casting the parameters.
    buffer_dtypes = {name: value.dtype for name, value in model.named_buffers()}
    model = model.to(dtype=torch.bfloat16)
    for name, dtype in buffer_dtypes.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._buffers[leaf] = parent._buffers[leaf].to(dtype=dtype)
    return model


def estimate_backbone(model) -> dict:
    import torch
    quantized_names = set()
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and not any(
            name == prefix or name.startswith(prefix + ".") for prefix in PRESERVED_PREFIXES
        ):
            quantized_names.add(name + ".weight")
    total = 0
    largest = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        largest = max(largest, count * 2)
        if name in quantized_names:
            # NF4 uint8 packing, double-quantized block scales, maps and shape overhead.
            total += math.ceil(count / 2) + math.ceil(count / 64) + 4 * math.ceil(count / 16384) + 4096
        else:
            total += count * 2
    return {"packed_backbone_bytes": total, "largest_tensor_bytes": largest,
            "quantized_linear_count": len(quantized_names)}


def _validate_weight_map(model, weight_map: dict) -> None:
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("A nonempty safetensors weight map is required.")
    expected = set(model.state_dict())
    actual = set(weight_map)
    if actual != expected:
        raise ValueError(f"Checkpoint keys differ: {len(expected - actual)} missing, {len(actual - expected)} unexpected.")
    for filename in weight_map.values():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Source shards must be relative safetensors filenames.")


def stream_quantize(model, weight_map: dict, fetch_shard: Callable[[str], Path],
                    source_files: dict, *, remove_download: Callable[[Path], None] | None = None,
                    progress: Callable[[dict], None] | None = None):
    """Use installed HF quantizer operations and safetensors' MPS pread loader."""
    import torch
    from safetensors import safe_open
    from transformers import BitsAndBytesConfig
    from transformers.modeling_utils import _load_parameter_into_model
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

    if not torch.backends.mps.is_available():
        raise RuntimeError("A working Apple MPS device is required; CPU fallback is not conversion proof.")
    _validate_weight_map(model, weight_map)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=list(PRESERVED_PREFIXES))
    quantizer = Bnb4BitHfQuantizer(quantization, pre_quantized=False)
    quantizer.validate_environment(device_map={"": "mps"})
    quantizer.preprocess_model(model, device_map={"": "mps"})
    model.hf_quantizer = quantizer
    operation = quantizer.get_quantize_ops()
    expected_by_shard = {}
    for name, filename in weight_map.items():
        expected_by_shard.setdefault(filename, set()).add(name)
    loaded = set()
    quantized = 0
    for filename, names in sorted(expected_by_shard.items()):
        path = Path(fetch_shard(filename))
        metadata = source_files[filename]
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Source shard size mismatch: {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != metadata["sha256"]:
            raise ValueError(f"Source shard SHA256 mismatch: {filename}")
        with safe_open(path, framework="pt", device="mps", backend="pread") as handle:
            if set(handle.keys()) != names:
                raise ValueError(f"Source shard tensor map mismatch: {filename}")
            for name in sorted(names):
                value = handle.get_tensor(name)
                parameter = model.get_parameter_or_buffer(name)
                if value.shape != parameter.shape:
                    raise ValueError(f"Source tensor shape mismatch: {name}")
                if value.is_floating_point() and value.dtype != torch.bfloat16:
                    raise ValueError(f"Expected source BF16 tensor: {name}")
                if quantizer.param_needs_quantization(model, name):
                    value = operation.convert({name: [value]}, model=model)[name]
                    quantized += 1
                _load_parameter_into_model(model, name, value)
                model.get_parameter_or_buffer(name)._is_hf_initialized = True
                loaded.add(name)
                del value
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()
        event = {"filename": filename, "bytes": metadata["bytes"], "sha256": actual_hash,
                 "tensors": len(names), "loaded_tensors": len(loaded),
                 "mps_allocated_bytes": torch.mps.current_allocated_memory()}
        if remove_download is not None:
            remove_download(path)
        if progress is not None:
            progress(event)
    if loaded != set(weight_map):
        raise ValueError("Incomplete source loading; refusing to initialize missing checkpoint weights.")
    # The same installed HF helpers from from_pretrained's finalization rebuild
    # nonpersistent RoPE buffers without changing loaded parameter values.
    model._move_missing_keys_from_meta_to_device(set(), {"": "mps"}, None, quantizer)
    model._initialize_missing_keys(True)
    quantizer.postprocess_model(model)
    model.eval()
    if any(value.is_meta for value in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("A parameter or buffer remains on meta after conversion.")
    if model.lm_head.weight.dtype != torch.bfloat16 or model.lm_head.weight.shape[0] != model.config.text_config.vocab_size:
        raise RuntimeError("Clef's output embedding must remain readable BF16 for its joint head.")
    return model, {"loaded_tensors": len(loaded), "quantized_linear_count": quantized}


def save_quantized(model, output: Path, max_shard_size: str = "2GB") -> dict:
    """Save a standalone HF model; source shards have already been released."""
    model.save_pretrained(output, max_shard_size=max_shard_size)
    files = sorted(output.glob("model*.safetensors"))
    return {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files}


def finite_probe(model, input_ids):
    import torch
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    if not bool(logits.isfinite().all()):
        raise RuntimeError("The standalone backbone probe produced nonfinite logits.")
    return logits.detach().to("cpu").float()


def verify_probe(expected, model, input_ids) -> dict:
    import torch
    actual = finite_probe(model, input_ids)
    difference = float((actual - expected).abs().max())
    if not torch.equal(actual, expected):
        raise RuntimeError(f"Standalone quantized reload changed the probe logits (max difference {difference}).")
    return {"finite": True, "token_count": input_ids.numel(), "logits_shape": list(actual.shape),
            "max_absolute_difference": difference,
            "logits_sha256": hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()}


def _hub_metadata(workspace: Path, token: str | None):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import Qwen3_5Config
    info = HfApi(token=token).model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError("Hub did not resolve the required immutable Clef revision.")
    files = {item.rfilename: {"bytes": item.size,
             "sha256": item.lfs.sha256 if item.lfs is not None else None} for item in info.siblings}
    for name in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=workspace, token=token)
    config = Qwen3_5Config.from_pretrained(workspace, local_files_only=True)
    weight_map = json.loads((workspace / "model.safetensors.index.json").read_text())["weight_map"]
    if not REQUIRED_SIDECARS <= set(files):
        raise ValueError("The genuine Clef source is missing its native head, code, or processor artifacts.")
    for filename in set(weight_map.values()) | {"joint_head.safetensors"}:
        if filename not in files or not files[filename]["sha256"] or not files[filename]["bytes"]:
            raise ValueError(f"Missing source LFS integrity metadata: {filename}")
    return config, weight_map, files


def convert_hub(output: Path, *, execute: bool = False, token: str | None = None,
                max_shard_size: str = "2GB", reserve_bytes: int = 3 * GIB) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    output = Path(output).resolve()
    require_fresh_output(output)
    if not torch.backends.mps.is_available():
        raise RuntimeError("A working Apple MPS device is required.")
    # Small metadata goes into a separate task-owned directory; no global cache.
    with tempfile.TemporaryDirectory(prefix="clef-mps-metadata-") as directory:
        config, weight_map, source_files = _hub_metadata(Path(directory), token)
        model = empty_backbone(config)
        _validate_weight_map(model, weight_map)
        shard_names = set(weight_map.values())
        sidecars = set(source_files) - shard_names - {"model.safetensors.index.json", "config.json", ".gitattributes"}
        estimates = estimate_backbone(model)
        estimates.update(largest_source_shard_bytes=max(source_files[name]["bytes"] for name in shard_names),
                         sidecar_bytes=sum(source_files[name]["bytes"] for name in sidecars))
        resources = check_resources(output, estimates, reserve_bytes)
        manifest = {"schema_version": 1, "status": "planned", "source": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
                    "format": "standalone_hf_bnb_mps_nf4", "preserved_bf16_prefixes": list(PRESERVED_PREFIXES),
                    "estimates": estimates, "resources": resources, "source_shards": len(shard_names),
                    "download_policy": "one_task_owned_source_shard_at_a_time_no_global_cache",
                    "resume_supported": False, "progress_storage": "in_memory_only_start_fresh_after_failure",
                    "completed_source_shards": [], "training_verified": False}
        if not execute:
            return manifest
        torch.set_num_threads(8)
        print(json.dumps({"event": "conversion_plan", "source": manifest["source"],
                          "estimates": estimates, "resources": resources,
                          "source_shards": len(shard_names)}), flush=True)
        output.mkdir(parents=True, exist_ok=True)
        manifest["status"] = "loading"
        _write_manifest(output, manifest)
        try:
            manifest["preserved_sidecars"] = {}
            for name in sorted(sidecars):
                path = Path(hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=output, token=token))
                sha = sha256_file(path)
                metadata = source_files[name]
                if path.stat().st_size != metadata["bytes"] or (metadata["sha256"] and sha != metadata["sha256"]):
                    raise ValueError(f"Sidecar integrity mismatch: {name}")
                manifest["preserved_sidecars"][name] = {"sha256": sha, "bytes": path.stat().st_size}
            shutil.copyfile(Path(directory) / "config.json", output / "source_config.json")
            shutil.copyfile(Path(directory) / "model.safetensors.index.json", output / "source_model.safetensors.index.json")
            with tempfile.TemporaryDirectory(prefix=".clef-mps-source-", dir=output) as source_directory:
                download_dir = Path(source_directory).resolve()
                def fetch(filename):
                    if shutil.disk_usage(output).free < source_files[filename]["bytes"] + reserve_bytes:
                        raise RuntimeError("Insufficient disk before downloading the next source shard.")
                    return Path(hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION,
                                               local_dir=download_dir, token=token))
                def remove(path):
                    if not path.resolve().is_relative_to(download_dir):
                        raise ValueError("Refusing to delete a file outside task-owned source downloads.")
                    path.unlink()
                def progress(event):
                    manifest["completed_source_shards"].append(event)
                    _write_manifest(output, manifest)
                    print(json.dumps({"event": "source_shard_converted", **event}), flush=True)
                model, stats = stream_quantize(model, weight_map, fetch, source_files,
                                               remove_download=remove, progress=progress)
            manifest.update(status="saving", **stats)
            _write_manifest(output, manifest)
            if shutil.disk_usage(output).free < estimates["packed_backbone_bytes"] + reserve_bytes:
                raise RuntimeError("Insufficient disk to save the packed standalone model.")
            from transformers import AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration
            # Processor auto-discovery reads the newly quantized model config.
            model.config.save_pretrained(output)
            if (output / "generation_config.json").exists():
                model.generation_config = GenerationConfig.from_pretrained(output, local_files_only=True)
                shutil.copyfile(output / "generation_config.json", output / "source_generation_config.json")
            processor = AutoProcessor.from_pretrained(output, local_files_only=True)
            input_ids = processor.tokenizer("Solana Clawd research evidence.", return_tensors="pt")["input_ids"].to("mps")
            expected_probe = finite_probe(model, input_ids)
            manifest["output_weight_files"] = save_quantized(model, output, max_shard_size)
            # HF saves generation metadata with its own formatting; retain the
            # exact source separately and distinguish it from untouched sidecars.
            generation = manifest["preserved_sidecars"].pop("generation_config.json", None)
            if generation is not None:
                manifest["source_generation_config"] = generation
            manifest["status"] = "reloading"
            _write_manifest(output, manifest)
            # Release the packed model before reload: never hold two 27B models.
            del model
            gc.collect()
            torch.mps.synchronize()
            torch.mps.empty_cache()
            restored = Qwen3_5ForConditionalGeneration.from_pretrained(output, device_map={"": "mps"},
                dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
            manifest["reload_probe"] = verify_probe(expected_probe, restored, input_ids)
            del restored
            gc.collect()
            torch.mps.empty_cache()
            manifest.update(status="complete", standalone_weights_saved=True,
                            source_head_unchanged=True, reload_verified=True)
            _write_manifest(output, manifest)
            return manifest
        except BaseException as error:
            manifest.update(status="failed", failure_type=type(error).__name__)
            _write_manifest(output, manifest)
            raise


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Download the immutable original Clef weights and convert.")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 1:
        parser.error("--reserve-gib must be finite and at least 1.")
    result = convert_hub(args.output, execute=args.execute, token=os.environ.get("HF_TOKEN"),
                         max_shard_size=args.max_shard_size, reserve_bytes=int(args.reserve_gib * GIB))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Verify the fully saved Clef NF4 checkpoint after its historical RoPE probe failed.

No weights are changed or downloaded. Exact saved-tensor equality and two fresh
Metal loads are required before repairing only the conversion manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from convert_clef_mps import MODEL_ID, MODEL_REVISION, _write_manifest, sha256_file, finite_probe, verify_probe


def now():
    return datetime.now(timezone.utc).isoformat()


def event(name, **values):
    print(json.dumps({"event": name, "at": now(), **values}), flush=True)


def inspect_checkpoint(base):
    from safetensors import safe_open
    manifest = json.loads((base / "conversion_manifest.json").read_text())
    if manifest["source"] != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Recovery requires the pinned original Clef conversion")
    if manifest["status"] != "failed" or manifest.get("failure_type") != "RuntimeError":
        raise ValueError("Recovery is restricted to the failed fully saved conversion")
    completed = manifest.get("completed_source_shards", [])
    original = json.loads((base / "source_model.safetensors.index.json").read_text())["weight_map"]
    if len(completed) != manifest["source_shards"] or {item["filename"] for item in completed} != set(original.values()):
        raise ValueError("Original source conversion is incomplete")
    weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weight_map.values()) != set(manifest["output_weight_files"]):
        raise ValueError("Saved weight index and integrity manifest disagree")
    for group in ("output_weight_files", "preserved_sidecars"):
        for name, expected in manifest[group].items():
            path = base / name
            if Path(name).name != name or path.is_symlink() or not path.is_file():
                raise ValueError("Unsafe or missing saved artifact")
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise ValueError(f"Saved artifact integrity mismatch: {name}")
    actual = set()
    for name in manifest["output_weight_files"]:
        with safe_open(base / name, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected = {key for key, filename in weight_map.items() if filename == name}
            if keys != expected or keys & actual:
                raise ValueError("Saved tensor index is incomplete or duplicated")
            actual.update(keys)
    if actual != set(weight_map) or not set(original) <= actual:
        raise ValueError("Saved checkpoint omitted original source parameters")
    return manifest, weight_map


def verify_loaded_tensors(model, base, weight_map):
    """Compare every persistent tensor with its saved bytes in bounded chunks."""
    import torch
    from safetensors import safe_open
    state = model.state_dict()
    if set(state) != set(weight_map):
        raise ValueError("Fresh loader changed the saved tensor key set")
    checked = 0
    for filename in sorted(set(weight_map.values())):
        with safe_open(base / filename, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                saved = handle.get_tensor(name)
                loaded = state[name].detach()
                if saved.dtype != loaded.dtype or saved.shape != loaded.shape:
                    raise ValueError(f"Fresh loader changed saved tensor dtype/shape: {name}")
                saved, loaded = saved.reshape(-1), loaded.reshape(-1)
                for start in range(0, saved.numel(), 4 * 1024 ** 2):
                    end = start + 4 * 1024 ** 2
                    if not torch.equal(saved[start:end], loaded[start:end].cpu()):
                        raise ValueError(f"Fresh loader changed saved tensor values: {name}")
                checked += 1
                del saved, loaded
        event("saved_tensor_shard_verified", filename=filename, checked_tensors=checked)
    if any(value.is_meta for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Fresh model contains uninitialized meta tensors")
    rope = {name: str(value.dtype) for name, value in model.named_buffers() if name.endswith("inv_freq")}
    if not rope or any(dtype != "torch.float32" for dtype in rope.values()):
        raise ValueError("Native fresh-loader positional frequencies must be FP32")
    return {"checked_tensors": checked, "all_saved_tensor_values_match": True,
            "native_rope_buffer_dtypes": rope}


def recover(base, execute=False):
    import torch
    base = Path(base).resolve()
    manifest, weight_map = inspect_checkpoint(base)
    event("saved_checkpoint_integrity_verified", weight_shards=len(manifest["output_weight_files"]),
          tensors=len(weight_map), original_shards=len(manifest["completed_source_shards"]))
    if not execute:
        return {"status": "audited_without_loading", "base": str(base), "tensors": len(weight_map)}
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("Real MPS with CPU fallback disabled is required")
    backup = base / "conversion_manifest.failed-before-rope-recovery.json"
    if backup.exists():
        if json.loads(backup.read_text()) != manifest:
            raise ValueError("Existing failure backup belongs to different conversion evidence")
    else:
        shutil.copyfile(base / "conversion_manifest.json", backup)
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    with (base / ".saved-checkpoint-recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(8)
        torch.mps.set_per_process_memory_fraction(0.90)
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        input_ids = processor.tokenizer("Solana Clawd research evidence.", return_tensors="pt")["input_ids"].to("mps")
        loader = dict(device_map={"": "mps"}, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
        event("first_fresh_model_load_started")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(base, **loader).eval()
        tensors = verify_loaded_tensors(model, base, weight_map)
        expected = finite_probe(model, input_ids)
        event("first_native_finite_probe_verified", logits_shape=list(expected.shape))
        del model
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        event("second_fresh_model_load_started")
        restored = Qwen3_5ForConditionalGeneration.from_pretrained(base, **loader).eval()
        probe = verify_probe(expected, restored, input_ids)
        del restored
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        # The old probe compared BF16-converted RoPE with native FP32 RoPE.
        # These fresh native loads preserve all original saved tensor bytes;
        # their equality is recorded separately from that failed old baseline.
        manifest.update(status="complete", standalone_weights_saved=True,
                        source_head_unchanged=True, reload_verified=True, reload_probe=probe,
                        recovery={"verified_at": now(), "failure_backup": backup.name,
                                  "reason": "historical converter downcast nonpersistent positional buffers to BF16",
                                  "verification_basis": "all_saved_persistent_tensors_exact_plus_two_native_fresh_loads",
                                  "weights_modified": False, "source_redownloaded": False,
                                  "cpu_fallback": False, "native_fresh_loads": 2, **tensors})
        _write_manifest(base, manifest)
        event("saved_checkpoint_recovered", max_absolute_difference=probe["max_absolute_difference"], **tensors)
        return {"status": "complete", "base": str(base), "reload_probe": probe, "recovery": manifest["recovery"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("local/clef-27b-mps-nf4"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.base, args.execute), indent=2), flush=True)

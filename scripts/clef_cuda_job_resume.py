"""Read-only authorization of one immutable Clef CUDA optimizer retry.

This module cannot submit/cancel jobs, upload files, load models or optimizer
pickles, or signal processes. Large checkpoint payloads are checked against
authoritative remote LFS SHA256/byte metadata without downloading them. The
worker still validates the actual optimizer cursor before continuing training.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from train_clef_local import (DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, MODEL_ID, MODEL_REVISION)
from train_clef_cuda import resume_contract

PROJECT = "clawd-clef-cuda-migration"
SELECTED_RECORDS = 60635
SELECTED_TOKENS = 67323884
MAX_SMALL_FILE_BYTES = 16 * 1024 ** 2
MAX_TOTAL_DOWNLOAD_BYTES = 32 * 1024 ** 2
REQUIRED = {"adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors",
    "joint_head_config.json", "joint_schema_model.py", "training.json", "tokenizer.json",
    "tokenizer_config.json", "processor_config.json", "optimizer.pt", "resume.json"}
ALLOWED = REQUIRED | {"README.md", "LICENSE", "source_base_model_card.md", "chat_template.jinja",
    "live-training-snapshot.json", "evaluation.json", "train_clef_cuda.py", "train_clef_local.py",
    "clef_research_data.py", "clef_research_training.py", "clef_live_tape.py",
    "research_expansion_artifacts.py"}


def _get(value, field, default=None):
    return value.get(field, default) if isinstance(value, dict) else getattr(value, field, default)


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {name} SHA256 identity")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"Invalid {name}")
    return value


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {name}")
    return value


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or
        PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or
        PurePosixPath(value).as_posix() != value):
        raise ValueError("Unsafe checkpoint artifact reference")
    return value


def _job_options(job):
    """Extract named non-secret bindings; never expose command/env/secrets."""
    tokens = list(_get(job, "command", []) or []) + list(_get(job, "arguments", []) or [])
    flags = {"--output-repo", "--input-repo", "--input-revision", "--mode", "--microbatch"}
    options = {}
    for index, token in enumerate(tokens):
        if token in flags:
            if token in options or index + 1 == len(tokens) or not isinstance(tokens[index + 1], str):
                raise ValueError("Remote job has ambiguous named training arguments")
            options[token] = tokens[index + 1]
    return options


def _sources(metadata, stage_sha):
    if (metadata.get("model") != MODEL_ID or metadata.get("model_revision") != MODEL_REVISION or
        metadata.get("dataset") != DATASET_ID or metadata.get("dataset_revision") != DATASET_REVISION or
        metadata.get("input_parquet_sha256") != INPUT_PARQUET_SHA256 or
        metadata.get("input_raw_row_counts") != INPUT_RAW_ROW_COUNTS or
        metadata.get("stage_manifest_sha256") != stage_sha):
        raise ValueError("Checkpoint changes the immutable model, dataset, or staged source")
    security = metadata.get("security_filter", {})
    excluded = security.get("exclusion_counts", {})
    if (security.get("applied") is not True or security.get("version") != 1 or
        security.get("published_source_modified") is not False or
        excluded.get("train_excluded_signing_byte_array", 0) < 11 or
        excluded.get("train_excluded_encoded_signing_literal", 0) < 3):
        raise ValueError("Checkpoint lacks the verified model-input privacy exclusions")
    if metadata.get("backend") != "cuda" or metadata.get("device") != "cuda" or metadata.get("cpu_fallback") is not False or metadata.get("cpu_offload") is not False:
        raise ValueError("Checkpoint does not represent native CUDA-only training")


def _parity(proof, name, records=None):
    difference, tolerance = proof.get("max_probability_difference"), proof.get("tolerance")
    if (proof.get("matched") is not True or type(difference) not in (float, int) or
        not math.isfinite(difference) or not 0 <= difference <= 0.001 or
        type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 < tolerance <= 0.001 or
        (records is not None and proof.get("records") != records)):
        raise ValueError(f"Invalid {name} probability proof")


def _benchmark(benchmark, stage_sha, bundle_sha, microbatch):
    _sources(benchmark, stage_sha)
    if (benchmark.get("status") != "trained_and_reload_verified" or benchmark.get("run_mode") != "benchmark" or
        benchmark.get("optimizer_steps") != 16 or benchmark.get("planned_steps") != 16 or
        benchmark.get("all_planned_steps_completed") is not True or benchmark.get("job_bundle_sha256") != bundle_sha or
        benchmark.get("microbatch_size") != microbatch or benchmark.get("gradient_accumulation") != 1 or
        benchmark.get("max_length") != 2048 or benchmark.get("epochs") != 1):
        raise ValueError("Retry requires the exact completed CUDA benchmark and execution package")
    _parity(benchmark.get("migration_verification", {}), "pilot migration", 16)
    _parity(benchmark.get("reload_verification", {}), "benchmark reload")
    _finite(benchmark.get("estimated_full_training_seconds"), "measured full training estimate")
    kernels = benchmark.get("kernel_execution", {})
    if (benchmark.get("use_kernels") is not True or kernels.get("implementation") != "installed_fla" or
        kernels.get("fla_core_version") != "0.5.2" or kernels.get("hub_kernel_downloads") is not False or
        _positive(kernels.get("cuda_forward_calls"), "CUDA FLA calls") < 1 or
        _positive(kernels.get("cuda_backward_calls"), "CUDA FLA backward calls") < 1):
        raise ValueError("Retry benchmark lacks genuine pinned FLA execution")
    cohort, capacity = benchmark.get("cohorts", {}).get("train", {}), benchmark.get("capacity_probe", {})
    if (cohort.get("selected") != SELECTED_RECORDS or cohort.get("context_limit") != 2048 or
        cohort.get("input_token_bytes") != SELECTED_TOKENS * 4 or capacity.get("trained") is not True or
        capacity.get("tokens") != cohort.get("max_tokens") or capacity.get("tokens", 0) < 2044):
        raise ValueError("Retry benchmark did not prove the complete selected context cohort")
    _sha(cohort.get("sha256"), "selected cohort")
    _sha(benchmark.get("training_order_sha256"), "training order")
    if not benchmark.get("runtime_source_sha256"):
        raise ValueError("Retry benchmark lacks executable source hashes")
    for digest in benchmark["runtime_source_sha256"].values():
        _sha(digest, "runtime source")
    for kind in ("lora", "head"):
        before, after = benchmark.get("initial_trainable_fingerprints", {}).get(kind, {}), benchmark.get("trained_trainable_fingerprints", {}).get(kind, {})
        _sha(before.get("sha256"), "initial trained path")
        _sha(after.get("sha256"), "updated trained path")
        _positive(before.get("parameters"), "initial trained parameter count")
        if after.get("parameters") != before["parameters"]:
            raise ValueError("Retry benchmark changed the original trained parameter scope")
        if (before["sha256"] == after["sha256"] or benchmark.get("gradient_evidence", {}).get(kind) is not True or
            before != benchmark.get("pilot_trainable_fingerprints", {}).get(kind)):
            raise ValueError("Retry benchmark lacks genuine pilot head and LoRA updates")


def authorize_checkpoint_retry(api, *, previous_job_id, owner, output_repo, output_revision,
    checkpoint_path, bundle_sha256, stage_manifest_sha256, full_plan, benchmark_evidence,
    previous_attempt, read_file=None):
    """Return retry evidence after read-only authoritative API/file checks.

    ``read_file(repo_id, filename, *, revision) -> bytes`` can inject a bounded
    reader for tests. Default uses the SDK to fetch only already size-checked
    small files. No paid API method is ever invoked. ``full_plan`` is the exact
    launcher plan; ``previous_attempt`` is its persisted submitted ledger entry.
    """
    _sha(bundle_sha256, "job package")
    _sha(stage_manifest_sha256, "staged inputs")
    if not isinstance(output_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", output_revision):
        raise ValueError("Retry requires a pinned immutable output commit")
    if not isinstance(checkpoint_path, str) or not re.fullmatch(r"checkpoints/step-[0-9]{6}", checkpoint_path):
        raise ValueError("Retry requires one exact completed optimizer checkpoint path")
    if not isinstance(previous_job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", previous_job_id):
        raise ValueError("An explicit previous job identity is required")
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", owner):
        raise ValueError("An explicit job owner is required")
    if not isinstance(output_repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", output_repo):
        raise ValueError("An explicit output repository is required")
    microbatch = _positive(full_plan.get("microbatch"), "planned microbatch")
    if (full_plan.get("mode") != "full" or full_plan.get("bundle_sha256") != bundle_sha256 or
        full_plan.get("output_repo") != output_repo or full_plan.get("jobs_namespace") != owner or
        full_plan.get("max_length") != 2048 or full_plan.get("complete_train_records") != SELECTED_RECORDS):
        raise ValueError("Retry changes the reviewed full-run plan")
    for key, expected in {"job_id": previous_job_id, "owner": owner, "output_repo": output_repo,
                          "mode": "full", "bundle_sha256": bundle_sha256}.items():
        if previous_attempt.get(key) != expected:
            raise ValueError("Previous job is not the exact submitted full-run ledger attempt")
    if previous_attempt.get("status") not in {"submitted", "terminal_error", "terminal_canceled", "terminal_confirmed"}:
        raise ValueError("Unknown or unsubmitted attempt may not authorize another GPU job")
    job = api.inspect_job(job_id=previous_job_id, namespace=owner)
    terminal = _get(_get(job, "status"), "stage")
    terminal = _get(terminal, "value", terminal)
    if terminal not in {"ERROR", "CANCELED"}:
        raise ValueError("Only authoritative ERROR or CANCELED jobs may authorize a checkpoint retry")
    if _get(job, "id") != previous_job_id or _get(_get(job, "owner"), "name") != owner:
        raise ValueError("Authoritative previous job identity or owner differs")
    labels = _get(job, "labels", {}) or {}
    if any(labels.get(key) != value for key, value in
           {"project": PROJECT, "phase": "full", "bundle": bundle_sha256[:40]}.items()):
        raise ValueError("Authoritative job labels do not match the full execution package")
    if sum(isinstance(token, str) and PurePosixPath(token).name == f"gpu-worker-{bundle_sha256}.py"
           for token in (_get(job, "command", []) or [])) != 1:
        raise ValueError("Authoritative job command does not name the exact immutable execution package")
    if _get(job, "flavor") != full_plan.get("flavor"):
        raise ValueError("Retry changes the benchmarked job hardware")
    options = _job_options(job)
    bindings = {"--output-repo": output_repo, "--input-repo": full_plan.get("input_repo"),
        "--input-revision": previous_attempt.get("input_revision"), "--mode": "full", "--microbatch": str(microbatch)}
    if not re.fullmatch(r"[0-9a-f]{40}", bindings["--input-revision"] or "") or any(
        not value or options.get(key) != value for key, value in bindings.items()):
        raise ValueError("Authoritative job arguments do not match the immutable full-run inputs")
    _benchmark(benchmark_evidence, stage_manifest_sha256, bundle_sha256, microbatch)
    info = api.model_info(output_repo, revision=output_revision)
    if _get(info, "private") is not True or _get(info, "sha") != output_revision:
        raise ValueError("Retry output must be the exact immutable private repository revision")
    remote = {}
    for entry in api.list_repo_tree(output_repo, revision=output_revision, repo_type="model", recursive=True):
        path = _get(entry, "path")
        if _get(entry, "blob_id") is not None:
            _relative(path)
            if path in remote:
                raise ValueError("Remote checkpoint file metadata is ambiguous")
            remote[path] = entry
    downloaded, download_bytes = {}, 0
    def fetch(name):
        nonlocal download_bytes
        if name in downloaded:
            return downloaded[name]
        entry = remote.get(name)
        size = _get(entry, "size")
        if type(size) is not int or not 0 < size <= MAX_SMALL_FILE_BYTES or download_bytes + size > MAX_TOTAL_DOWNLOAD_BYTES:
            raise ValueError("Small-file checkpoint verification refuses a large or absent payload")
        if read_file is None:
            local = api.hf_hub_download(output_repo, name, revision=output_revision, repo_type="model")
            data = Path(local).read_bytes()
        else:
            data = read_file(output_repo, name, revision=output_revision)
        if not isinstance(data, bytes) or len(data) != size:
            raise ValueError("Remote small checkpoint file size differs from immutable metadata")
        lfs = _get(entry, "lfs")
        digest = hashlib.sha256(data).hexdigest()
        if lfs:
            if _get(lfs, "sha256") != digest or _get(lfs, "size") != size:
                raise ValueError("Remote small checkpoint file differs from its LFS hash")
        elif hashlib.sha1(f"blob {size}\0".encode() + data).hexdigest() != _get(entry, "blob_id"):
            raise ValueError("Remote small checkpoint file differs from its immutable Git blob")
        downloaded[name] = data
        download_bytes += size
        return data
    def document(name):
        try:
            value = json.loads(fetch(name))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Checkpoint metadata is not valid JSON") from None
        if not isinstance(value, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        return value
    manifest_name = checkpoint_path + "/adapter_artifact_manifest.json"
    manifest = document(manifest_name)
    files = manifest.get("files")
    if manifest.get("checkpoint_complete") is not True or not isinstance(files, dict) or not REQUIRED <= set(files) or not set(files) <= ALLOWED:
        raise ValueError("Remote checkpoint is incomplete or contains unapproved payloads")
    expected_paths = {checkpoint_path + "/" + _relative(name) for name in files} | {manifest_name}
    actual_paths = {name for name in remote if name.startswith(checkpoint_path + "/")}
    if actual_paths != expected_paths:
        raise ValueError("Remote checkpoint inventory differs from its exact payload files")
    lfs_checked = 0
    for name, expected in files.items():
        if not isinstance(expected, dict):
            raise ValueError("Invalid checkpoint payload metadata")
        _sha(expected.get("sha256"), "checkpoint payload")
        size = _positive(expected.get("bytes"), "checkpoint payload bytes")
        full_name = checkpoint_path + "/" + name
        entry = remote[full_name]
        if _get(entry, "size") != size:
            raise ValueError("Remote checkpoint payload bytes differ from the saved inventory")
        lfs = _get(entry, "lfs")
        if name in {"optimizer.pt", "joint_head.safetensors", "adapter_model.safetensors"} and not lfs:
            raise ValueError("Optimizer and model weights require authoritative LFS hashes; downloads are disabled")
        if lfs:
            if _get(lfs, "size") != size or _get(lfs, "sha256") != expected["sha256"]:
                raise ValueError("Remote checkpoint LFS hash differs from the saved inventory")
            lfs_checked += 1
        elif size > MAX_SMALL_FILE_BYTES:
            raise ValueError("Large checkpoint weights/optimizer need authoritative LFS SHA256 metadata")
        elif hashlib.sha256(fetch(full_name)).hexdigest() != expected["sha256"]:
            raise ValueError("Remote checkpoint file SHA256 differs from the saved inventory")
    training = document(checkpoint_path + "/training.json")
    resume = document(checkpoint_path + "/resume.json")
    progress = document("training.json")
    _sources(training, stage_manifest_sha256)
    _sources(progress, stage_manifest_sha256)
    if (training.get("status") != "training" or training.get("run_mode") != "full" or
        training.get("all_planned_steps_completed") is not False or
        training.get("complete_selected_training_epochs") is not False):
        raise ValueError("A completed or non-full adapter must not be retried as an unfinished epoch")
    planned_steps = math.ceil(SELECTED_RECORDS / microbatch)
    expected_contract = {"run_id": training.get("run_id"), "run_mode": "full", "stage_manifest_sha256": stage_manifest_sha256,
        "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION, "max_length": 2048,
        "microbatch_size": microbatch, "gradient_accumulation": 1, "planned_steps": planned_steps}
    for key in ("seed", "learning_rate", "training_order_sha256", "attention_implementation", "use_kernels", "runtime_source_sha256"):
        expected_contract[key] = benchmark_evidence.get(key)
    if not isinstance(training.get("run_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", training["run_id"]):
        raise ValueError("Checkpoint lacks an exact native training run identity")
    if any(training.get(key) != value for key, value in expected_contract.items()) or training.get("epochs") != 1:
        raise ValueError("Checkpoint changes the source/order/optimizer/kernel resume contract")
    if (resume.get("contract") != expected_contract or manifest.get("resume_contract") != expected_contract or
        manifest.get("run_id") != training["run_id"]):
        raise ValueError("Checkpoint completion and resume contracts disagree")
    steps, cursor = _positive(training.get("optimizer_steps"), "completed optimizer steps"), _positive(training.get("trained_records"), "completed training records")
    interval = _positive(full_plan.get("checkpoint_steps", 128), "planned checkpoint interval")
    if (steps >= planned_steps or steps != int(checkpoint_path.rsplit("-", 1)[1]) or
        steps % interval or cursor != min(steps * microbatch, SELECTED_RECORDS) or
        any(metadata.get("optimizer_steps") != steps or metadata.get("cursor") != cursor for metadata in (resume, manifest))):
        raise ValueError("Checkpoint path, optimizer step, and actual native epoch cursor disagree")
    if (training.get("cohorts") != benchmark_evidence.get("cohorts") or
        training.get("prepared_split_counts") != benchmark_evidence.get("prepared_split_counts") or
        training.get("local_base_manifest_sha256") != benchmark_evidence.get("local_base_manifest_sha256") or
        training.get("local_base_generated_files") != benchmark_evidence.get("local_base_generated_files") or
        training.get("lora") != benchmark_evidence.get("lora")):
        raise ValueError("Checkpoint changes the exact cohort, quantized base, or trained parameter scope")
    for name, digest in training.get("runtime_source_sha256", {}).items():
        if files.get(name, {}).get("sha256") != digest:
            raise ValueError("Checkpoint runtime source bytes do not match the benchmarked execution code")
    artifacts = training.get("artifact_files", {})
    if not {"adapter_config.json", "adapter_model.safetensors", "joint_head.safetensors",
            "joint_head_config.json", "joint_schema_model.py"} <= set(artifacts) or any(
        files.get(name) != identity for name, identity in artifacts.items()):
        raise ValueError("Checkpoint actual weights/code/config hashes disagree with training metadata")
    for name in ("joint_head_config.json", "joint_schema_model.py"):
        if files.get(name) != benchmark_evidence.get("artifact_files", {}).get(name):
            raise ValueError("Checkpoint changes the pinned native head configuration or model code")
    for kind in ("lora", "head"):
        before, current = training.get("initial_trainable_fingerprints", {}).get(kind, {}), training.get("trainable_fingerprints", {}).get(kind, {})
        _sha(current.get("sha256"), "current trained path")
        if (before != benchmark_evidence.get("pilot_trainable_fingerprints", {}).get(kind) or
            training.get("pilot_trainable_fingerprints", {}).get(kind) != before or
            current.get("parameters") != before.get("parameters") or current["sha256"] == before.get("sha256") or
            training.get("gradient_evidence", {}).get(kind) is not True or
            training.get("finite_trainables", {}).get(kind) != {"finite": True, "parameters": current.get("parameters")}):
            raise ValueError("Checkpoint lacks finite genuine head and LoRA update evidence")
    _parity(training.get("migration_verification", {}), "full-run pilot migration", 16)
    if (progress.get("job_bundle_sha256") != bundle_sha256 or progress.get("run_id") != training["run_id"] or
        progress.get("run_mode") != "full" or progress.get("optimizer_steps", 0) < steps or
        progress.get("trained_records", 0) < cursor or
        (previous_attempt.get("run_id") is not None and previous_attempt["run_id"] != training["run_id"])):
        raise ValueError("Pinned output progress does not bind this checkpoint to the authorized full run")
    if resume_contract(progress) != expected_contract:
        raise ValueError("Pinned output progress changed the full epoch contract")
    trained_tokens = _positive(training.get("training_input_tokens"), "actual trained input tokens")
    if not cursor <= trained_tokens <= cursor * 2048 or trained_tokens >= SELECTED_TOKENS:
        raise ValueError("Checkpoint actual native token coverage is inconsistent")
    remaining_records, remaining_tokens = SELECTED_RECORDS - cursor, SELECTED_TOKENS - trained_tokens
    return {"status": "checkpoint_retry_verified_compute_not_submitted", "retry_of": previous_job_id,
        "owner": owner, "authoritative_previous_stage": terminal, "output_repo": output_repo,
        "output_revision": output_revision, "checkpoint_path": checkpoint_path,
        "checkpoint_manifest_sha256": hashlib.sha256(downloaded[manifest_name]).hexdigest(),
        "bundle_sha256": bundle_sha256, "stage_manifest_sha256": stage_manifest_sha256,
        "run_id": training["run_id"], "completed_optimizer_steps": steps, "planned_steps": planned_steps,
        "cursor": cursor, "selected_records": SELECTED_RECORDS, "remaining_records": remaining_records,
        "remaining_optimizer_steps": planned_steps - steps, "remaining_training_fraction": remaining_records / SELECTED_RECORDS,
        "remaining_input_tokens": remaining_tokens, "remaining_token_fraction": remaining_tokens / SELECTED_TOKENS,
        "estimated_remaining_training_seconds": benchmark_evidence["estimated_full_training_seconds"] * remaining_tokens / SELECTED_TOKENS,
        "verified_payload_files": len(files), "remote_lfs_hashes_verified": lfs_checked,
        "small_payload_bytes_downloaded": download_bytes, "optimizer_downloaded": False,
        "job_binding_scope": "authoritative terminal job owner/labels/arguments plus pinned full-run progress/contract; optimizer cursor revalidated by worker"}

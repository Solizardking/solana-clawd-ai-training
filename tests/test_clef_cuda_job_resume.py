"""Read-only API retry guards; synthetic metadata, no GPU/model/Hub actions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clef_cuda_job_resume as retry
from train_clef_cuda import resume_contract


def payload(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


class RemoteFixture:
    """Immutable metadata-only fixture; no fabricated real-training assertion."""
    def __init__(self, terminal="ERROR"):
        self.bundle, self.stage, self.revision, self.input_revision = "a" * 64, "b" * 64, "c" * 40, "d" * 40
        self.repo, self.owner, self.job_id = "solanaclawd/private-full", "ordlibrary", "previous-job"
        self.checkpoint = "checkpoints/step-000128"
        self.files, self.entries, self.reads, self.api_calls = {}, {}, [], []
        self.large = set()
        self.plan = {"mode": "full", "bundle_sha256": self.bundle, "output_repo": self.repo,
            "jobs_namespace": self.owner, "max_length": 2048, "complete_train_records": 60635,
            "microbatch": 4, "flavor": "h200", "input_repo": "solanaclawd/private-input"}
        self.attempt = {"job_id": self.job_id, "owner": self.owner, "output_repo": self.repo,
            "mode": "full", "bundle_sha256": self.bundle, "status": "submitted", "input_revision": self.input_revision}
        self.job = SimpleNamespace(id=self.job_id, owner=SimpleNamespace(name=self.owner),
            status=SimpleNamespace(stage=terminal), flavor="h200",
            labels={"project": retry.PROJECT, "phase": "full", "bundle": self.bundle[:40]},
            command=["uv", "run", "--python", "3.12", f"/data/gpu-worker-{self.bundle}.py",
                "--input-repo", self.plan["input_repo"], "--input-revision", self.input_revision,
                "--output-repo", self.repo, "--mode", "full", "--microbatch", "4"], arguments=[],
            # Accessing these would reveal a material helper regression.
            environment="NEVER READ ENVIRONMENT", secrets="NEVER READ SECRETS")
        self.model = SimpleNamespace(private=True, sha=self.revision)
        initial = {kind: {"sha256": "1" * 64, "parameters": 10} for kind in ("lora", "head")}
        updated = {kind: {"sha256": "2" * 64, "parameters": 10} for kind in ("lora", "head")}
        parity = {"matched": True, "records": 16, "max_probability_difference": 0.0, "tolerance": 0.001}
        base = {"model": retry.MODEL_ID, "model_revision": retry.MODEL_REVISION,
            "dataset": retry.DATASET_ID, "dataset_revision": retry.DATASET_REVISION,
            "input_parquet_sha256": deepcopy(retry.INPUT_PARQUET_SHA256),
            "input_raw_row_counts": deepcopy(retry.INPUT_RAW_ROW_COUNTS),
            "security_filter": {"applied": True, "version": 1, "published_source_modified": False,
                "exclusion_counts": {"train_excluded_signing_byte_array": 11, "train_excluded_encoded_signing_literal": 3}},
            "stage_manifest_sha256": self.stage, "backend": "cuda", "device": "cuda",
            "cpu_fallback": False, "cpu_offload": False, "microbatch_size": 4,
            "gradient_accumulation": 1, "max_length": 2048, "epochs": 1, "seed": 42,
            "learning_rate": 2e-5, "training_order_sha256": "3" * 64, "attention_implementation": "eager",
            "use_kernels": True, "runtime_source_sha256": {
                "train_clef_cuda.py": hashlib.sha256(b"Pinned source fixture").hexdigest()},
            "kernel_execution": {"implementation": "installed_fla", "fla_core_version": "0.5.2",
                "hub_kernel_downloads": False, "cuda_forward_calls": 500, "cuda_backward_calls": 50},
            "migration_verification": deepcopy(parity), "reload_verification": deepcopy(parity),
            "initial_trainable_fingerprints": initial, "pilot_trainable_fingerprints": deepcopy(initial),
            "trained_trainable_fingerprints": deepcopy(updated), "trainable_fingerprints": deepcopy(updated),
            "gradient_evidence": {"lora": True, "head": True},
            "finite_trainables": {kind: {"finite": True, "parameters": 10} for kind in initial},
            "cohorts": {"train": {"selected": 60635, "prepared": 77665, "context_limit": 2048,
                "input_token_bytes": 269295536, "max_tokens": 2048, "sha256": "5" * 64},
                "eval": {"selected": 32, "sha256": "6" * 64}, "test": {"selected": 32, "sha256": "7" * 64}},
            "prepared_split_counts": {"train": 77665, "eval": 2500, "test": 2800},
            "capacity_probe": {"trained": True, "tokens": 2048},
            "local_base_manifest_sha256": "8" * 64,
            "local_base_generated_files": {"config.json": {"bytes": 20, "sha256": "9" * 64}},
            "lora": {"rank": 8, "text_layers": [60, 61, 62, 63]}, "all_planned_steps_completed": True}
        self.benchmark = {**deepcopy(base), "status": "trained_and_reload_verified", "run_mode": "benchmark",
            "optimizer_steps": 16, "planned_steps": 16, "job_bundle_sha256": self.bundle,
            "estimated_full_training_seconds": 72000}
        self.training = {**deepcopy(base), "status": "training", "run_mode": "full", "run_id": "e" * 32,
            "optimizer_steps": 128, "planned_steps": 15159, "trained_records": 512,
            "training_input_tokens": 600000, "all_planned_steps_completed": False,
            "complete_selected_training_epochs": False}
        self.resume = {"contract": resume_contract(self.training), "optimizer_steps": 128, "cursor": 512}
        self.manifest = {"files": {}, "checkpoint_complete": True, "run_id": self.training["run_id"],
            "optimizer_steps": 128, "cursor": 512, "resume_contract": resume_contract(self.training)}
        for name in retry.REQUIRED - {"training.json", "resume.json"}:
            if name in {"optimizer.pt", "joint_head.safetensors", "adapter_model.safetensors", "tokenizer.json"}:
                # Advertised large-file metadata only; no payload allocated.
                self.add(name, b"LFS identity fixture", lfs_size=300_000_000)
            else:
                self.add(name, b"Small checkpoint fixture")
        self.add("train_clef_cuda.py", b"Pinned source fixture")
        self.training["artifact_files"] = {name: deepcopy(self.manifest["files"][name]) for name in
            ("adapter_config.json", "adapter_model.safetensors", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py")}
        self.benchmark["artifact_files"] = deepcopy(self.training["artifact_files"])
        self.add("training.json", payload(self.training))
        self.add("resume.json", payload(self.resume))
        self.refresh_manifest()
        self.write("training.json", payload({**self.training, "job_bundle_sha256": self.bundle}))

    def write(self, path, data, lfs_size=None):
        size = lfs_size or len(data)
        sha = hashlib.sha256(data).hexdigest()
        lfs = SimpleNamespace(size=size, sha256=sha) if lfs_size else None
        self.files[path] = data
        self.entries[path] = SimpleNamespace(path=path, size=size,
            blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(), lfs=lfs)
        if lfs_size:
            self.large.add(path)
        return {"sha256": sha, "bytes": size}

    def add(self, name, data, lfs_size=None):
        self.manifest["files"][name] = self.write(self.checkpoint + "/" + name, data, lfs_size)

    def refresh_manifest(self):
        self.write(self.checkpoint + "/adapter_artifact_manifest.json", payload(self.manifest))

    def change_training(self, mutation):
        mutation(self.training)
        self.add("training.json", payload(self.training))
        self.refresh_manifest()

    def inspect_job(self, **kwargs):
        assert kwargs == {"job_id": self.job_id, "namespace": self.owner}
        self.api_calls.append("inspect_job")
        return self.job

    def model_info(self, repo_id, **kwargs):
        assert repo_id == self.repo and kwargs == {"revision": self.revision}
        self.api_calls.append("model_info")
        return self.model

    def list_repo_tree(self, repo_id, **kwargs):
        assert repo_id == self.repo and kwargs == {"revision": self.revision, "repo_type": "model", "recursive": True}
        self.api_calls.append("list_repo_tree")
        return list(self.entries.values())

    def read(self, repo_id, path, *, revision):
        assert repo_id == self.repo and revision == self.revision
        assert path not in self.large, "Large optimizer/weights must not be downloaded"
        self.reads.append(path)
        return self.files[path]

    def authorize(self, **overrides):
        args = {"previous_job_id": self.job_id, "owner": self.owner, "output_repo": self.repo,
            "output_revision": self.revision, "checkpoint_path": self.checkpoint,
            "bundle_sha256": self.bundle, "stage_manifest_sha256": self.stage,
            "full_plan": self.plan, "benchmark_evidence": self.benchmark,
            "previous_attempt": self.attempt, "read_file": self.read}
        args.update(overrides)
        return retry.authorize_checkpoint_retry(self, **args)


@pytest.mark.parametrize("terminal", ["ERROR", "CANCELED"])
def test_terminal_exact_owned_job_authorizes_pinned_remaining_cursor_without_large_download(terminal):
    fixture = RemoteFixture(terminal)
    proof = fixture.authorize()
    assert proof["status"] == "checkpoint_retry_verified_compute_not_submitted"
    assert proof["retry_of"] == fixture.job_id and proof["authoritative_previous_stage"] == terminal
    assert proof["output_revision"] == fixture.revision and proof["cursor"] == 512
    assert proof["completed_optimizer_steps"] == 128 and proof["planned_steps"] == 15159
    assert proof["remaining_records"] == 60635 - 512 and proof["remaining_optimizer_steps"] == 15159 - 128
    assert proof["remaining_training_fraction"] == pytest.approx((60635 - 512) / 60635)
    assert proof["estimated_remaining_training_seconds"] == pytest.approx(72000 * (67323884 - 600000) / 67323884)
    assert proof["remote_lfs_hashes_verified"] == 4 and proof["optimizer_downloaded"] is False
    assert not fixture.large.intersection(fixture.reads) and set(fixture.api_calls) == {"inspect_job", "model_info", "list_repo_tree"}


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING", "COMPLETED", "DELETED", "FAILED", "CANCELLED", None])
def test_active_completed_unknown_or_deleted_job_cannot_authorize_retry(stage):
    fixture = RemoteFixture(stage)
    with pytest.raises(ValueError, match="authoritative ERROR or CANCELED"):
        fixture.authorize()
    assert fixture.api_calls == ["inspect_job"] and not fixture.reads


@pytest.mark.parametrize("mutation", [
    lambda f: setattr(f.job.owner, "name", "someone-else"),
    lambda f: setattr(f.job, "id", "unrelated-job"),
    lambda f: f.job.labels.update(phase="benchmark"),
    lambda f: f.job.labels.update(bundle="0" * 40),
    lambda f: setattr(f.job, "flavor", "a100-large"),
    lambda f: f.job.command.__setitem__(f.job.command.index("--output-repo") + 1, "solanaclawd/unrelated"),
    lambda f: f.job.command.__setitem__(4, "/data/unrelated-worker.py"),
    lambda f: f.job.command.extend(["--mode", "full"]),
])
def test_wrong_authoritative_identity_owner_phase_package_hardware_or_destination_is_blocked(mutation):
    fixture = RemoteFixture()
    mutation(fixture)
    with pytest.raises(ValueError):
        fixture.authorize()
    assert not fixture.reads


@pytest.mark.parametrize("kwargs", [{"output_revision": "main"}, {"checkpoint_path": "checkpoints/../step-000128"},
    {"checkpoint_path": "checkpoints/step-128"}, {"bundle_sha256": "bad"}, {"owner": "../someone"}])
def test_ambiguous_revision_path_or_identity_is_blocked_before_api_read(kwargs):
    fixture = RemoteFixture()
    with pytest.raises(ValueError):
        fixture.authorize(**kwargs)
    assert not fixture.api_calls and not fixture.reads


@pytest.mark.parametrize("mutation", [
    lambda f: f.attempt.update(job_id="unknown"),
    lambda f: f.attempt.update(mode="benchmark"),
    lambda f: f.attempt.update(status="submission_outcome_unknown"),
    lambda f: f.attempt.update(output_repo="solanaclawd/other"),
    lambda f: f.plan.update(microbatch=1),
    lambda f: f.plan.update(complete_train_records=100),
    lambda f: f.plan.update(max_length=4096),
    lambda f: f.benchmark.update(job_bundle_sha256="0" * 64),
    lambda f: f.benchmark["migration_verification"].update(max_probability_difference=0.002),
    lambda f: f.benchmark["kernel_execution"].update(cuda_backward_calls=0),
])
def test_ledger_plan_or_benchmark_changes_cannot_authorize_retry(mutation):
    fixture = RemoteFixture()
    mutation(fixture)
    with pytest.raises(ValueError):
        fixture.authorize()
    assert not fixture.reads


@pytest.mark.parametrize("mutation", [
    lambda t: t.update(dataset_revision="0" * 40),
    lambda t: t["security_filter"].update(applied=False),
    lambda t: t["input_raw_row_counts"].update(train=100),
    lambda t: t.update(max_length=4096),
    lambda t: t.update(microbatch_size=1),
    lambda t: t.update(learning_rate=1e-4),
    lambda t: t.update(training_order_sha256="0" * 64),
    lambda t: t.update(optimizer_steps=129),
    lambda t: t.update(trained_records=511),
    lambda t: t.update(status="trained_and_reload_verified"),
    lambda t: t.update(all_planned_steps_completed=True),
    lambda t: t.update(training_input_tokens=67323884),
    lambda t: t["gradient_evidence"].update(head=False),
])
def test_hash_valid_checkpoint_still_rejects_source_privacy_cursor_optimizer_or_completion_changes(mutation):
    fixture = RemoteFixture()
    fixture.change_training(mutation)
    with pytest.raises(ValueError):
        fixture.authorize()
    assert not fixture.large.intersection(fixture.reads)


def test_remote_lfs_mismatch_or_unknown_payload_is_blocked_without_optimizer_download():
    fixture = RemoteFixture()
    fixture.entries[fixture.checkpoint + "/optimizer.pt"].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="LFS hash"):
        fixture.authorize()
    assert not fixture.large.intersection(fixture.reads)
    fixture = RemoteFixture()
    fixture.write(fixture.checkpoint + "/.env", b"Unapproved fixture")
    with pytest.raises(ValueError, match="exact payload"):
        fixture.authorize()
    assert not fixture.large.intersection(fixture.reads)


def test_checkpoint_manifest_git_hash_and_completion_flag_are_both_required():
    fixture = RemoteFixture()
    path = fixture.checkpoint + "/adapter_artifact_manifest.json"
    fixture.files[path] = fixture.files[path].replace(b'"checkpoint_complete": true', b'"checkpoint_complete": fals')
    with pytest.raises(ValueError, match="Git blob"):
        fixture.authorize()
    fixture = RemoteFixture()
    fixture.manifest["checkpoint_complete"] = False
    fixture.refresh_manifest()
    with pytest.raises(ValueError, match="incomplete"):
        fixture.authorize()


def test_unpinned_or_public_outputs_and_unrelated_root_progress_cannot_bind_checkpoint():
    fixture = RemoteFixture()
    fixture.model.private = False
    with pytest.raises(ValueError, match="immutable private"):
        fixture.authorize()
    fixture = RemoteFixture()
    fixture.model.sha = "0" * 40
    with pytest.raises(ValueError, match="immutable private"):
        fixture.authorize()
    fixture = RemoteFixture()
    fixture.write("training.json", payload({**fixture.training, "run_id": "f" * 32, "job_bundle_sha256": fixture.bundle}))
    with pytest.raises(ValueError, match="bind this checkpoint"):
        fixture.authorize()


def test_terminal_confirmed_ledger_is_reinspected_and_cannot_override_now_active_job():
    fixture = RemoteFixture("RUNNING")
    fixture.attempt["status"] = "terminal_confirmed"
    with pytest.raises(ValueError, match="authoritative"):
        fixture.authorize()
    assert fixture.api_calls == ["inspect_job"] and not fixture.reads


def test_even_small_optimizer_is_never_downloaded_and_needs_lfs_hash():
    fixture = RemoteFixture()
    fixture.add("optimizer.pt", b"small optimizer metadata fixture")
    fixture.refresh_manifest()
    with pytest.raises(ValueError, match="downloads are disabled"):
        fixture.authorize()
    assert fixture.checkpoint + "/optimizer.pt" not in fixture.reads


def test_hash_valid_replaced_runtime_code_cannot_claim_original_benchmark_source():
    fixture = RemoteFixture()
    fixture.add("train_clef_cuda.py", b"different source fixture")
    fixture.refresh_manifest()
    with pytest.raises(ValueError, match="runtime source bytes"):
        fixture.authorize()

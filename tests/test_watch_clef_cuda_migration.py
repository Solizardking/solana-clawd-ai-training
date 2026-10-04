"""CPU watcher tests: fake read-only API and stub launcher, no paid actions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import watch_clef_cuda_migration as watch


class Fixture:
    def __init__(self, tmp_path, monkeypatch, stage="COMPLETED"):
        self.state, self.cache = tmp_path / "state", tmp_path / "cache"
        self.state.mkdir()
        self.cache.mkdir()
        self.bundle, self.source, self.revision, self.input_revision = "a" * 64, "b" * 64, "c" * 40, "d" * 40
        self.worker = self.state / f"gpu-worker-{self.bundle}.py"
        self.worker.write_text("CPU immutable worker fixture; never executed")
        self.saved = {"id": "benchmark-id", "owner": "ordlibrary", "mode": "benchmark",
            "jobs_namespace": "ordlibrary", "flavor": "h200", "microbatch": 4,
            "max_length": 2048, "complete_train_records": 60635, "budget_usd": "50",
            "maximum_charge_usd": "5", "input_repo": "solanaclawd/input-private",
            "output_repo": "solanaclawd/benchmark-private", "input_revision": self.input_revision,
            "bundle_sha256": self.bundle, "worker_script_sha256": watch.launcher.sha256(self.worker)}
        self.ledger = {"budget_usd": "50", "maximum_reserved_charge_usd": "5", "reservations": [{
            "job_id": self.saved["id"], "owner": self.saved["owner"], "mode": "benchmark", "status": "submitted",
            "bundle_sha256": self.bundle, "maximum_charge_usd": "5", "output_repo": self.saved["output_repo"],
            "input_revision": self.input_revision}]}
        self.publication = {"status": "private_stage_uploaded_and_hash_verified", "repo_id": self.saved["input_repo"],
            "revision": self.input_revision, "manifest_sha256": self.source}
        for name, value in (("benchmark-job.json", self.saved), ("budget-ledger.json", self.ledger),
                            ("input-publication.json", self.publication)):
            self.write_state(name, value)
        self.job = self.make_job(self.saved, stage)
        self.full_job = None
        self.reads, self.inspects, self.launches, self.model_reads = [], [], [], 0
        self.price, self.repo_private, self.latest_revision = 5, True, self.revision
        self.evidence = {"status": "trained_and_reload_verified", "run_mode": "benchmark", "backend": "cuda",
            "model": watch.MODEL_ID, "model_revision": watch.MODEL_REVISION,
            "dataset": watch.DATASET_ID, "dataset_revision": watch.DATASET_REVISION,
            "input_parquet_sha256": deepcopy(watch.INPUT_PARQUET_SHA256), "input_raw_row_counts": deepcopy(watch.INPUT_RAW_ROW_COUNTS),
            "security_filter": {"version": 1, "applied": True}, "microbatch_size": 4, "max_length": 2048,
            "optimizer_steps": 16, "planned_steps": 16, "stage_manifest_sha256": self.source, "job_bundle_sha256": self.bundle,
            "gradient_evidence": {"lora": True, "head": True}, "capacity_probe": {"trained": True, "tokens": 2048},
            "cohorts": {"train": {"selected": 60635, "max_tokens": 2048}},
            "kernel_execution": {"implementation": "installed_fla", "fla_core_version": "0.5.2",
                "hub_kernel_downloads": False, "cuda_forward_calls": 100, "cuda_backward_calls": 16},
            "migration_verification": {"matched": True, "max_probability_difference": 0.0},
            "reload_verification": {"matched": True, "max_probability_difference": 0.0},
            "initial_trainable_fingerprints": {kind: {"sha256": "1" * 64} for kind in ("lora", "head")},
            "trained_trainable_fingerprints": {kind: {"sha256": "2" * 64} for kind in ("lora", "head")},
            "estimated_full_training_seconds": 3600}
        self.completion = {"status": "verified_artifacts_uploaded", "mode": "benchmark",
            "bundle_sha256": self.bundle, "input_revision": self.input_revision}
        self.entries = {}
        self.update_remote()
        monkeypatch.setattr(watch, "current_bundle_sha", lambda: self.bundle)

    def write_state(self, name, value):
        (self.state / name).write_text(json.dumps(value))

    def make_job(self, saved, stage):
        return SimpleNamespace(id=saved["id"], owner=SimpleNamespace(name=saved["owner"]),
            flavor=saved["flavor"], labels={"project": watch.launcher.PROJECT,
                "phase": saved["mode"], "bundle": saved["bundle_sha256"][:40]},
            status=SimpleNamespace(stage=stage, message="HF_TOKEN=hf_" + "x" * 40),
            environment="SECRET ENVIRONMENT MUST NOT BE READ", secrets="SECRET VALUES MUST NOT BE READ")

    def update_remote(self):
        for name, value in (("training.json", self.evidence), ("job-completion.json", self.completion)):
            data = (json.dumps(value) + "\n").encode()
            (self.cache / name).write_bytes(data)
            self.entries[name] = SimpleNamespace(path=name, size=len(data), lfs=None,
                blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())

    def inspect_job(self, *, job_id, namespace):
        self.inspects.append((job_id, namespace))
        if job_id == self.saved["id"]:
            return self.job
        assert self.full_job is not None and job_id == self.full_job.id
        return self.full_job

    def model_info(self, repo_id):
        assert repo_id == self.saved["output_repo"]
        self.model_reads += 1
        return SimpleNamespace(sha=self.latest_revision, private=self.repo_private)

    def get_paths_info(self, repo_id, names, **kwargs):
        assert repo_id == self.saved["output_repo"] and set(names) == {"training.json", "job-completion.json"}
        assert kwargs == {"revision": self.revision, "repo_type": "model"}
        return list(self.entries.values())

    def hf_hub_download(self, repo_id, name, **kwargs):
        assert repo_id == self.saved["output_repo"] and kwargs == {"revision": self.revision, "repo_type": "model"}
        self.reads.append(name)
        return str(self.cache / name)

    def list_jobs_hardware(self):
        return [SimpleNamespace(name="h200", unit_label="hour", unit_cost_usd=self.price)]

    def launch(self, command):
        self.launches.append(command)
        full = {**self.saved, "mode": "full", "id": "full-id", "output_repo": "solanaclawd/full-private"}
        self.write_state("full-job.json", full)
        self.full_job = self.make_job(full, "SCHEDULING")
        self.ledger["reservations"].append({"job_id": full["id"], "owner": full["owner"], "mode": "full",
            "status": "submitted", "bundle_sha256": self.bundle, "maximum_charge_usd": "45",
            "output_repo": full["output_repo"]})
        self.ledger["maximum_reserved_charge_usd"] = "50"
        self.write_state("budget-ledger.json", self.ledger)
        return 0

    def run(self, **kwargs):
        return watch.run_once(self, self.state, python_executable="/lexical/project/.venv/bin/python", launch=self.launch, **kwargs)


@pytest.mark.parametrize("stage", ["SCHEDULING", "RUNNING"])
def test_exact_benchmark_poll_waits_without_fetch_or_submission_before_completion(tmp_path, monkeypatch, stage):
    fixture = Fixture(tmp_path, monkeypatch, stage)
    result = fixture.run()
    assert result["status"] == "waiting_for_benchmark" and result["optimizer_steps_observed"] is None
    assert fixture.inspects == [(fixture.saved["id"], "ordlibrary")]
    assert not fixture.reads and not fixture.launches and fixture.model_reads == 0
    assert "hf_" not in json.dumps(result) and "message" not in result["benchmark"]


def test_completed_hash_verified_source_fits_budget_and_delegates_one_capped_full_submission(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    target = tmp_path / "base-python"
    target.write_text("Interpreter identity fixture; never executed")
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(target)
    result = watch.run_once(fixture, fixture.state, python_executable=str(interpreter), launch=fixture.launch)
    assert result["status"] == "full_submitted" and result["full"]["stage"] == "SCHEDULING"
    assert result["remaining_timeout_seconds"] == 9 * 3600 and result["minimum_full_timeout_seconds"] == 10080
    assert len(fixture.launches) == 1
    command = fixture.launches[0]
    assert command[0] == str(interpreter) and command[0] != str(interpreter.resolve())
    assert command[command.index("--budget-usd") + 1] == "50"
    assert command[command.index("--timeout-seconds") + 1] == "32400"
    assert command[command.index("--mode") + 1] == "full" and "--submit" in command
    stored = fixture.state / "benchmark-evidence" / fixture.revision
    assert json.loads((stored / "verification.json").read_text())["revision"] == fixture.revision
    assert (stored / "training.json").read_bytes() == (fixture.cache / "training.json").read_bytes()
    # A second watcher observes the saved exact full job, rather than launching.
    repeated = fixture.run()
    assert repeated["status"] == "existing_full_job" and len(fixture.launches) == 1


@pytest.mark.parametrize("stage", ["ERROR", "CANCELED", "DELETED"])
def test_failed_benchmark_is_terminal_bounded_and_never_automatically_requalified(tmp_path, monkeypatch, stage):
    fixture = Fixture(tmp_path, monkeypatch, stage)
    result = fixture.run()
    assert result["status"] == "benchmark_terminal_failure" and result["automatic_qualification_retry"] is False
    assert not fixture.launches and not fixture.reads
    assert fixture.ledger["maximum_reserved_charge_usd"] == "5"


@pytest.mark.parametrize("mutation", [
    lambda f: f.evidence.update(stage_manifest_sha256="0" * 64),
    lambda f: f.evidence.update(job_bundle_sha256="0" * 64),
    lambda f: f.evidence.update(model_revision="0" * 40),
    lambda f: f.evidence.update(dataset_revision="0" * 40),
    lambda f: f.evidence["security_filter"].update(applied=False),
    lambda f: f.evidence.update(optimizer_steps=15),
    lambda f: f.evidence["reload_verification"].update(max_probability_difference=0.01),
    lambda f: f.evidence["kernel_execution"].update(cuda_backward_calls=0),
    lambda f: f.completion.update(status="training"),
    lambda f: f.completion.update(input_revision="0" * 40),
])
def test_wrong_source_incomplete_native_proof_or_wrong_completion_blocks_full_submission(tmp_path, monkeypatch, mutation):
    fixture = Fixture(tmp_path, monkeypatch)
    mutation(fixture)
    fixture.update_remote()
    with pytest.raises(ValueError):
        fixture.run()
    assert not fixture.launches


def test_measured_full_run_over_remaining_cap_does_not_submit(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.evidence["estimated_full_training_seconds"] = 30000
    fixture.update_remote()
    result = fixture.run()
    assert result["status"] == "budget_insufficient" and result["minimum_full_timeout_seconds"] > 32400
    assert not fixture.launches


def test_current_price_increase_reduces_budget_capacity_before_submission(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.price = 50
    result = fixture.run()
    assert result["status"] == "budget_insufficient" and result["remaining_timeout_seconds"] == 3240
    assert not fixture.launches


@pytest.mark.parametrize("mutation", [
    lambda f: f.job.labels.update(phase="full"),
    lambda f: setattr(f.job.owner, "name", "unrelated"),
    lambda f: setattr(f.job, "id", "unrelated"),
    lambda f: f.worker.write_text("changed executable"),
    lambda f: f.ledger.update(budget_usd="100"),
    lambda f: f.ledger.update(maximum_reserved_charge_usd="0"),
])
def test_wrong_authoritative_job_worker_or_global_budget_blocks_handoff(tmp_path, monkeypatch, mutation):
    fixture = Fixture(tmp_path, monkeypatch)
    mutation(fixture)
    fixture.write_state("budget-ledger.json", fixture.ledger)
    with pytest.raises(ValueError):
        fixture.run()
    assert not fixture.launches


def test_source_changed_during_evidence_download_is_rechecked_before_launcher(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    original = fixture.hf_hub_download
    def reader(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(watch, "current_bundle_sha", lambda: "0" * 64)
        return result
    fixture.hf_hub_download = reader
    with pytest.raises(ValueError, match="source changed"):
        fixture.run()
    assert not fixture.launches


def test_remote_git_tampering_and_public_evidence_are_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.entries["training.json"].blob_id = "0" * 40
    with pytest.raises(ValueError, match="Git/LFS"):
        fixture.run()
    assert not fixture.launches
    fixture.entries["training.json"].blob_id = hashlib.sha1(
        f"blob {fixture.entries['training.json'].size}\0".encode() + (fixture.cache / "training.json").read_bytes()).hexdigest()
    fixture.repo_private = False
    with pytest.raises(ValueError, match="immutable private"):
        fixture.run()
    assert not fixture.launches


def test_unknown_submission_or_prior_full_reservation_never_triggers_duplicate(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.ledger["reservations"].append({"mode": "full", "status": "submission_outcome_unknown", "maximum_charge_usd": "45"})
    fixture.ledger["maximum_reserved_charge_usd"] = "50"
    fixture.write_state("budget-ledger.json", fixture.ledger)
    result = fixture.run()
    assert result["status"] == "full_submission_unresolved" and not fixture.launches


def test_bundle_identity_matches_existing_builder_and_excludes_new_watcher(tmp_path):
    assert "scripts/watch_clef_cuda_migration.py" not in watch.launcher.FILES
    expected = watch.launcher.build_bundle(tmp_path / "temporary_cpu_source_bundle.py")
    assert watch.current_bundle_sha() == expected


def test_output_sanitizer_hides_tokens_and_machine_paths(tmp_path):
    result = watch._save(tmp_path, {"status": "validation_blocked", "action": "HF_TOKEN=hf_" + "x" * 36 + " /Users/example/private/data"})
    serialized = json.dumps(result)
    assert "hf_" + "x" * 36 not in serialized and "/Users/example" not in serialized


def test_only_one_watcher_can_hold_the_handoff_lock(tmp_path):
    with watch.watcher_lock(tmp_path):
        with pytest.raises(ValueError, match="Another watcher"):
            with watch.watcher_lock(tmp_path):
                pytest.fail("Second watcher acquired the same handoff lock")
    with watch.watcher_lock(tmp_path):
        pass

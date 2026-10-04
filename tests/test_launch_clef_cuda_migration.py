"""Budget and migration gates; CPU only, no credentials or paid API calls."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import subprocess
import json
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_clef_cuda_migration import (available_seconds, benchmark_seconds,
    content_addressed_bundle, dollars, hourly_price, reserve_budget, retain_owned_checkpoints,
    spool_checkpoint, submission_lock)


def test_second_launcher_process_cannot_enter_submission_critical_section(tmp_path):
    script = ("import sys; sys.path.insert(0,sys.argv[1]); "
              "from launch_clef_cuda_migration import submission_lock; "
              "guard=submission_lock(sys.argv[2]); guard.__enter__()")
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    lock = tmp_path / "submission.lock"
    with submission_lock(lock):
        result = subprocess.run([sys.executable, "-c", script, scripts, str(lock)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "Another launcher owns" in result.stderr
    with submission_lock(lock):
        pass


def test_upload_snapshot_survives_source_checkpoint_pruning(tmp_path):
    from train_clef_local import refresh_adapter_manifest, verify_adapter_manifest
    checkpoint = tmp_path / "checkpoints/step-000128"
    checkpoint.mkdir(parents=True)
    names = {"adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors",
             "joint_head_config.json", "joint_schema_model.py", "training.json",
             "tokenizer.json", "tokenizer_config.json", "processor_config.json",
             "optimizer.pt", "resume.json"}
    for name in names:
        (checkpoint / name).write_text(name)
    refresh_adapter_manifest(checkpoint)
    manifest = checkpoint / "adapter_artifact_manifest.json"
    value = json.loads(manifest.read_text())
    value["checkpoint_complete"] = True
    manifest.write_text(json.dumps(value))
    snapshot = spool_checkpoint(checkpoint, tmp_path / "upload-spool")
    shutil.rmtree(checkpoint)
    assert verify_adapter_manifest(snapshot)["checkpoint_complete"] is True
    assert (snapshot / "optimizer.pt").read_text() == "optimizer.pt"


def test_partial_checkpoint_is_never_uploaded(tmp_path):
    checkpoint = tmp_path / "step-000128"
    checkpoint.mkdir()
    (checkpoint / "adapter_artifact_manifest.json").write_text(json.dumps({"files": {}}))
    assert spool_checkpoint(checkpoint, tmp_path / "upload-spool") is None


def test_job_source_path_is_immutable_and_covers_export_runtime(tmp_path):
    import launch_clef_cuda_migration as launcher
    assert "scripts/run_clef_local.py" in launcher.FILES
    first, first_hash = content_addressed_bundle(tmp_path)
    second, second_hash = content_addressed_bundle(tmp_path)
    assert first == second
    assert first_hash == second_hash
    assert first_hash in first.name
    first.write_text("changed executable")
    with pytest.raises(ValueError, match="script bytes differ"):
        content_addressed_bundle(tmp_path)


def test_remote_pruning_deletes_only_worker_owned_superseded_files():
    calls = []
    api = SimpleNamespace(delete_files=lambda **kwargs: calls.append(kwargs), upload_file=lambda **kwargs: None)
    checkpoints = [{"path": f"checkpoints/step-{step:06d}", "revision": str(step),
                    "files": ["joint_head.safetensors", "adapter_artifact_manifest.json"]} for step in (128, 256, 384)]
    retain_owned_checkpoints(api, "private/repo", checkpoints)
    assert len(checkpoints) == 2
    assert calls[0]["delete_patterns"] == ["checkpoints/step-000128/joint_head.safetensors",
                                           "checkpoints/step-000128/adapter_artifact_manifest.json"]
    assert calls[0]["repo_id"] == "private/repo"


def test_remote_pruning_refuses_unrelated_paths():
    api = SimpleNamespace(delete_files=lambda **kwargs: pytest.fail("must not delete"), upload_file=lambda **kwargs: None)
    checkpoints = [{"path": "adapter", "files": ["joint_head.safetensors"]}] * 3
    with pytest.raises(ValueError, match="unrelated remote"):
        retain_owned_checkpoints(api, "private/repo", checkpoints)


@pytest.mark.parametrize("amount", [0, -1, "NaN", "Infinity"])
def test_invalid_budget_cannot_authorize_compute(amount):
    with pytest.raises(ValueError):
        dollars(amount)


def test_cumulative_timeout_cost_includes_previous_jobs_and_unknown_responses():
    ledger = {}
    first = reserve_budget(ledger, budget=40, hourly=5, seconds=3600,
                           mode="benchmark", bundle_sha256="a")
    first["status"] = "submission_outcome_unknown"
    assert available_seconds(ledger, 40, 5) == 7 * 3600
    with pytest.raises(ValueError, match="unresolved submission"):
        reserve_budget(ledger, budget=40, hourly=5, seconds=3600,
                       mode="benchmark", bundle_sha256="a")
    with pytest.raises(ValueError, match="unresolved submission"):
        reserve_budget(ledger, budget=40, hourly=5, seconds=3600,
                       mode="benchmark", bundle_sha256="changed-code")
    with pytest.raises(ValueError, match="remaining cumulative"):
        reserve_budget(ledger, budget=40, hourly=5, seconds=8 * 3600,
                       mode="full", bundle_sha256="a")
    reserve_budget(ledger, budget=40, hourly=5, seconds=7 * 3600,
                   mode="full", bundle_sha256="a")
    assert available_seconds(ledger, 40, 5) == 0


def test_definitive_rejection_without_job_does_not_spend_reservation():
    ledger = {}
    first = reserve_budget(ledger, budget=40, hourly=5, seconds=3600,
                           mode="benchmark", bundle_sha256="a")
    first["status"] = "rejected_without_job"
    assert available_seconds(ledger, 40, 5) == 8 * 3600
    reserve_budget(ledger, budget=40, hourly=5, seconds=3600,
                   mode="benchmark", bundle_sha256="a")


def test_budget_cannot_be_changed_by_unrelated_retry():
    ledger = {}
    reserve_budget(ledger, budget=40, hourly=5, seconds=3600, mode="benchmark", bundle_sha256="a")
    with pytest.raises(ValueError, match="Changing the cumulative"):
        reserve_budget(ledger, budget=100, hourly=5, seconds=3600, mode="full", bundle_sha256="b")


def test_authoritative_retry_retains_prior_cost_and_cannot_be_reserved_twice():
    ledger = {}
    first = reserve_budget(ledger, budget=40, hourly=5, seconds=3600, mode="full", bundle_sha256="a")
    first.update(status="submitted", job_id="old")
    with pytest.raises(ValueError, match="authoritative terminal"):
        reserve_budget(ledger, budget=40, hourly=5, seconds=3600, mode="full", bundle_sha256="a", retry_of="old")
    first["status"] = "terminal_confirmed"
    second = reserve_budget(ledger, budget=40, hourly=5, seconds=3600, mode="full", bundle_sha256="a", retry_of="old")
    assert available_seconds(ledger, 40, 5) == 6 * 3600
    second.update(status="submitted", job_id="new")
    with pytest.raises(ValueError, match="one unused attempt"):
        reserve_budget(ledger, budget=40, hourly=5, seconds=3600, mode="full", bundle_sha256="a", retry_of="old")


def test_pricing_fails_closed_for_unknown_units():
    assert hourly_price(SimpleNamespace(unit_label="minute", unit_cost_usd="0.08333333")) == Decimal("5.0000")
    with pytest.raises(ValueError, match="Unknown hardware"):
        hourly_price(SimpleNamespace(unit_label="batch", unit_cost_usd=5))


def test_fractional_timeout_reserves_whole_billable_minutes():
    ledger = {}
    reservation = reserve_budget(ledger, budget=1, hourly=5, seconds=61,
                                 mode="benchmark", bundle_sha256="code")
    assert reservation["maximum_billable_minutes"] == 2
    assert Decimal(reservation["maximum_charge_usd"]) == Decimal(5) / 30


def evidence():
    return {"status": "trained_and_reload_verified", "run_mode": "benchmark", "backend": "cuda",
            "optimizer_steps": 16, "planned_steps": 16, "stage_manifest_sha256": "stage",
            "job_bundle_sha256": "code", "gradient_evidence": {"lora": True, "head": True},
            "capacity_probe": {"trained": True, "tokens": 2048}, "cohorts": {"train": {"max_tokens": 2048}},
            "kernel_execution": {"implementation": "installed_fla", "fla_core_version": "0.5.2",
                                 "hub_kernel_downloads": False, "cuda_forward_calls": 100, "cuda_backward_calls": 16},
            "migration_verification": {"matched": True, "max_probability_difference": 0.0001},
            "reload_verification": {"matched": True, "max_probability_difference": 0},
            "initial_trainable_fingerprints": {"lora": {"sha256": "a"}, "head": {"sha256": "b"}},
            "trained_trainable_fingerprints": {"lora": {"sha256": "c"}, "head": {"sha256": "d"}},
            "estimated_full_training_seconds": 3600}


def test_full_time_bound_uses_measured_training_plus_setup_margin():
    assert benchmark_seconds(evidence(), "stage", "code") == 10080


@pytest.mark.parametrize("mutate", [
    lambda item: item.update(status="training"),
    lambda item: item.update(backend="mps"),
    lambda item: item.update(job_bundle_sha256="other"),
    lambda item: item.update(stage_manifest_sha256="other"),
    lambda item: item["migration_verification"].update(max_probability_difference=0.1),
    lambda item: item["reload_verification"].update(matched=False),
    lambda item: item["gradient_evidence"].update(head=False),
    lambda item: item["capacity_probe"].update(trained=False),
    lambda item: item["capacity_probe"].update(tokens=512),
    lambda item: item["kernel_execution"].update(cuda_backward_calls=0),
    lambda item: item["kernel_execution"].update(hub_kernel_downloads=True),
    lambda item: item["trained_trainable_fingerprints"]["head"].update(sha256="b"),
    lambda item: item.update(estimated_full_training_seconds=float("nan")),
])
def test_full_training_rejects_incomplete_or_mismatched_benchmark(mutate):
    item = deepcopy(evidence())
    mutate(item)
    with pytest.raises(ValueError):
        benchmark_seconds(item, "stage", "code")

"""Publication gates on real hashed small artifacts; only Hub I/O is mocked.

The fixture inventory is intentionally tiny and makes no claim to train or load
the production 27B checkpoint. MPS update/reload tests live in the trainer suite.
"""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION
from export_clef_mps_release import EXPANDED_DATASET_REVISION, EXPANDED_INPUT_ROWS, EXPANDED_INPUT_PARQUET_SHA256
from export_clef_release import _backbone_inventory
from publish_clef_local_model import (DEFAULT_REPO, file_sha, prepare_or_reuse, push_publication, stage_publication,
                                      validate_full_training, validate_inference, validate_package,
                                      verify_remote, write_json)


@pytest.fixture
def release_fixture(tmp_path):
    import torch
    from safetensors.torch import save_file
    release = tmp_path / "small_hashed_release_fixture"
    release.mkdir()
    save_file({"model.visual.weight": torch.ones(2, 2), "model.language_model.weight": torch.zeros(2, 2)},
              str(release / "model.safetensors"))
    save_file({"projection.weight": torch.ones(2, 2)}, str(release / "joint_head.safetensors"))
    for name in ("config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json", "joint_head_config.json"):
        write_json(release / name, {"fixture": "small real hashed artifact; no production model load"})
    for name in ("joint_schema_model.py", "run_clef_local.py", "export_clef_mps_release.py", "export_clef_release.py"):
        (release / name).write_text('"""Small hashed publication-gate fixture."""\n')
    (release / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")
    (release / "source_base_model_card.md").write_text("---\nlicense: apache-2.0\n---\n# Original source card fixture\n")
    (release / "README.md").write_text("Original local card; do not overwrite.\n")
    before = hashlib.sha256(torch.zeros(4).numpy().tobytes()).hexdigest()
    after = hashlib.sha256(torch.ones(4).numpy().tobytes()).hexdigest()
    initial = {kind: {"sha256": before, "parameters": 4} for kind in ("lora", "head")}
    changed = {kind: {"sha256": after, "parameters": 4} for kind in ("lora", "head")}
    parity = {"matched": True, "records": 2, "max_probability_difference": 0.0, "tolerance": 0.0001}
    cohort = {"prepared": 42, "selected": 40, "excluded": [{"id": "long-1", "reason": "context_exceeds_limit"},
                                                           {"id": "long-2", "reason": "context_exceeds_limit"}],
              "sources": {"paper_2605.12151.md": 10, "paper_2606.08232.md": 10,
                          "https://clawd-ws.fly.dev/": 4, "other-research": 16}}
    heldout = {"prepared": 20, "selected": 8, "excluded": [], "sources": {"heldout": 8}}
    training = {"status": "trained_and_standalone_reload_verified", "run_mode": "full", "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "dataset": DATASET_ID, "dataset_revision": EXPANDED_DATASET_REVISION,
        "input_parquet_sha256": EXPANDED_INPUT_PARQUET_SHA256, "input_raw_row_counts": EXPANDED_INPUT_ROWS,
        "security_filter": {"version": 1, "applied": True, "exclusion_counts": {"train_excluded_signing_byte_array": 11,
                              "train_excluded_encoded_signing_literal": 3}, "published_source_modified": False},
        "all_planned_steps_completed": True, "complete_selected_training_epochs": True,
        "selected_training_scope": "all context-eligible prepared rows", "prepared_split_counts": {"train": 42, "eval": 20, "test": 20},
        "cohorts": {"train": cohort, "eval": copy.deepcopy(heldout), "test": copy.deepcopy(heldout)},
        "trained_records": 40, "epochs": 1, "gradient_accumulation": 1, "planned_steps": 40, "optimizer_steps": 40,
        "cpu_fallback": False, "vision_trained": False, "autoExecute": False,
        "gradient_evidence": {"lora": True, "head": True}, "initial_trainable_fingerprints": initial,
        "trained_trainable_fingerprints": changed, "trainable_fingerprints": copy.deepcopy(changed),
        "finite_trainables": {kind: {"finite": True, "parameters": 4} for kind in ("lora", "head")},
        "standalone_reload_verification": parity}
    write_json(release / "training.json", training)
    metric = {"records": 8, "accuracy": 0.625, "chance_accuracy": 0.25, "nll": 0.8}
    write_json(release / "evaluation.json", {name: copy.deepcopy(metric) for name in ("baseline_eval", "eval", "test")})

    def refresh():
        inventory = _backbone_inventory(release)
        inventory.pop("tensor_names")
        files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha(path)} for path in release.iterdir()
                 if path.is_file() and path.name != "release.json"}
        manifest = {"format": "clef-merged-release-v1", "status": "trained_merged_and_standalone_reload_verified",
            "standalone_backbone": True, "source": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
            "backbone": inventory, "training": json.loads((release / "training.json").read_text()),
            "files": files, "total_bytes": sum(item["bytes"] for item in files.values()),
            "merge_verification": {**parity, "tolerance": 0.005}, "reload_verified": True, "reload_verification": parity}
        write_json(release / "release.json", manifest)
        return manifest
    refresh()
    proof = tmp_path / "actual_inference_shape.json"
    captured = datetime.now(timezone.utc) - timedelta(minutes=10)
    completed = captured + timedelta(seconds=2)

    def proof_write(**updates):
        value = {"model_path": str(release), "model_revision": MODEL_REVISION, "dataset_revision": EXPANDED_DATASET_REVISION,
            "training_status": "trained_and_standalone_reload_verified", "standalone_model_loaded": True,
            "device": "mps", "backend": "mps", "source": "https://clawd-ws.fly.dev/",
            "release_manifest_sha256": file_sha(release / "release.json"), "captured_at": captured.isoformat(),
            "inference_completed_at": completed.isoformat(), "freshness": {"stale": False, "snapshot_age_seconds": 2,
                                "observation_age_seconds": [2.1]},
            "responses": [{"id": "live-health", "usage": {"input_tokens": 64, "output_tokens": 0},
                           "answers": {"status": {"type": "choice", "choice": "healthy", "confidence": 0.75,
                                     "probabilities": {"healthy": 0.75, "unavailable": 0.25}}}}]}
        value.update(updates)
        write_json(proof, value)
        return value
    proof_write()
    return SimpleNamespace(release=release, proof=proof, training=training, refresh=refresh, proof_write=proof_write,
                           stage=tmp_path / "publication_overlay")


@pytest.mark.parametrize("mutate", [
    lambda t: t.update(run_mode="pilot"),
    lambda t: t.update(all_planned_steps_completed=False),
    lambda t: t.update(complete_selected_training_epochs=False),
    lambda t: t.update(selected_training_scope="bounded shuffled cohort with required live and paper rows"),
    lambda t: t.update(optimizer_steps=16),
    lambda t: t.update(trained_records=16),
    lambda t: t["cohorts"]["train"].update(selected=16),
    lambda t: t["cohorts"]["train"].update(excluded=[]),
    lambda t: t["gradient_evidence"].update(lora=False),
    lambda t: t["trained_trainable_fingerprints"].update(head=t["initial_trainable_fingerprints"]["head"]),
    lambda t: t["finite_trainables"]["head"].update(finite=False),
    lambda t: t.update(cpu_fallback=True),
    lambda t: t.update(autoExecute=True),
    lambda t: t["cohorts"]["train"]["sources"].update(paper_2606=0),
])
def test_full_training_gate_rejects_incomplete_pilots_and_missing_weight_proof(release_fixture, mutate):
    training = copy.deepcopy(release_fixture.training)
    mutate(training)
    with pytest.raises(ValueError):
        validate_full_training(training)


@pytest.mark.parametrize("update", [
    {"standalone_model_loaded": False}, {"device": "cpu"}, {"release_manifest_sha256": "0" * 64},
    {"dataset_revision": "0" * 40}, {"captured_at": "2026-10-01"},
    {"freshness": {"stale": True, "snapshot_age_seconds": 2, "observation_age_seconds": [2]}},
    {"responses": []}, {"signing_seed": list(range(32))},
])
def test_inference_gate_rejects_unbound_stale_or_private_material(release_fixture, update):
    fixture = release_fixture
    fixture.proof_write(**update)
    with pytest.raises(ValueError):
        validate_inference(fixture.proof, fixture.release)


def test_nonfinite_native_response_and_future_capture_are_rejected(release_fixture):
    fixture = release_fixture
    proof = fixture.proof_write()
    proof["responses"][0]["answers"]["status"]["probabilities"]["healthy"] = float("nan")
    write_json(fixture.proof, proof)
    with pytest.raises(ValueError, match="finite"):
        validate_inference(fixture.proof, fixture.release)
    fixture.proof_write(captured_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    with pytest.raises(ValueError):
        validate_inference(fixture.proof, fixture.release)


def test_systemone_choice_before_four_decimal_rounding_can_tie_in_saved_probabilities(release_fixture):
    fixture = release_fixture
    proof = fixture.proof_write()
    proof["responses"][0]["answers"]["status"] = {"type": "choice", "choice": "unavailable", "confidence": 0.5,
                                                       "probabilities": {"healthy": 0.5, "unavailable": 0.5}}
    write_json(fixture.proof, proof)
    assert validate_inference(fixture.proof, fixture.release) == proof


def test_plan_keeps_real_weight_and_source_bytes_unchanged_and_updates_card_inventory(release_fixture):
    fixture = release_fixture
    original = {path.name: file_sha(path) for path in fixture.release.iterdir()}
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    assert package["status"] == "planned" and package["private"] is True and package["repo_id"] == DEFAULT_REPO
    assert original == {path.name: file_sha(path) for path in fixture.release.iterdir()}
    assert Path(package["files"]["model.safetensors"]["local_path"]).parent == fixture.release
    assert not list(fixture.stage.glob("*.safetensors"))
    overlay = json.loads((fixture.stage / "release.json").read_text())
    assert overlay["files"]["README.md"]["sha256"] == file_sha(fixture.stage / "README.md")
    assert "inference-evidence.json" in overlay["files"]
    assert (fixture.stage / "source-release.json").read_bytes() == (fixture.release / "release.json").read_bytes()
    assert (fixture.stage / "source-trained-model-card.md").read_bytes() == (fixture.release / "README.md").read_bytes()
    assert overlay["files"]["source-release.json"]["sha256"] == package["source_release_manifest_sha256"]
    assert validate_package(package) == package
    card = (fixture.stage / "README.md").read_text()
    assert "license: apache-2.0" in card and "arxiv.org/abs/2605.12151" in card and "arxiv.org/abs/2606.08232" in card
    assert "Context exclusions" in card and "0.625000" in card and "historical record" in card
    # The recorded proof remains useful after30 seconds without claiming that
    # its observation is fresh at publication time.
    assert package["observed_at"] == json.loads(fixture.proof.read_text())["captured_at"]
    assert prepare_or_reuse(fixture.release, fixture.proof, fixture.stage) == package


@pytest.mark.parametrize("failure", ["wrong_source", "unverified_reload", "license", "file_tamper"])
def test_release_integrity_provenance_and_license_gates_before_stage(release_fixture, failure):
    fixture = release_fixture
    if failure == "file_tamper":
        (fixture.release / "model.safetensors").write_bytes(b"tampered")
    elif failure == "license":
        (fixture.release / "LICENSE").write_text("License fixture mismatch")
        fixture.refresh()
        fixture.proof_write()
    else:
        manifest = json.loads((fixture.release / "release.json").read_text())
        if failure == "wrong_source":
            manifest["source"]["revision"] = "0" * 40
        else:
            manifest["reload_verified"] = False
        write_json(fixture.release / "release.json", manifest)
    with pytest.raises(ValueError):
        stage_publication(fixture.release, fixture.proof, fixture.stage)
    assert not fixture.stage.exists()


@pytest.mark.parametrize("filename", ["source_base_model_card.md", "evaluation.json", "run_clef_local.py"])
def test_publication_refuses_incomplete_actual_export_file_contract(release_fixture, filename):
    fixture = release_fixture
    (fixture.release / filename).unlink()
    fixture.refresh()
    fixture.proof_write()
    with pytest.raises(ValueError, match="evaluation, licensing or runtime"):
        stage_publication(fixture.release, fixture.proof, fixture.stage)
    assert not fixture.stage.exists()


class MockHub:
    """Only external Hub calls are simulated; local validation stays real."""
    token = "never-printed-test-placeholder"

    def __init__(self, package, *, private=True, corrupt=False, extras=False):
        self.package, self.private, self.corrupt, self.extras = package, private, corrupt, extras
        self.calls = []
        self.revision = "a" * 40

    def whoami(self):
        self.calls.append("whoami")
        return {"name": "fixture-owner"}

    def create_repo(self, repo_id, **kwargs):
        assert kwargs["private"] is True
        self.calls.append("create_repo")

    def model_info(self, repo_id, revision=None, files_metadata=False):
        self.calls.append("model_info")
        if revision is None:
            siblings = [SimpleNamespace(rfilename=".gitattributes")]
            if self.extras:
                siblings.append(SimpleNamespace(rfilename="unknown-previous-weight.safetensors"))
            return SimpleNamespace(private=self.private, sha="b" * 40, siblings=siblings)
        assert revision == self.revision
        siblings = []
        for name, item in self.package["files"].items():
            lfs = SimpleNamespace(sha256=item["sha256"]) if name.endswith(".safetensors") else None
            if self.corrupt and name.endswith(".safetensors"):
                lfs.sha256 = "0" * 64
            siblings.append(SimpleNamespace(rfilename=name, size=item["bytes"], lfs=lfs))
        return SimpleNamespace(private=self.private, sha=self.revision, siblings=siblings)

    def create_commit(self, repo_id, **kwargs):
        self.calls.append("create_commit")
        assert kwargs["parent_commit"] == "b" * 40
        assert {item.path_in_repo for item in kwargs["operations"]} == set(self.package["files"])
        return SimpleNamespace(oid=self.revision, commit_url="https://huggingface.co/fixture/model/commit/" + self.revision)


def test_private_push_verifies_actual_overlay_bytes_without_weight_download(release_fixture):
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    api = MockHub(package)
    downloads = []

    def download(repo, name, revision):
        assert not name.endswith(".safetensors")
        downloads.append(name)
        return package["files"][name]["local_path"]
    result = push_publication(package, api=api, download=download)
    assert result["status"] == "published_and_verified" and result["verified"] is True
    assert result["weight_downloaded_locally"] is False and result["files_verified"] == len(package["files"])
    assert "release.json" in downloads and "README.md" in downloads
    assert json.loads((fixture.stage / "publication.json").read_text()) == result


@pytest.mark.parametrize("kwargs,commit_expected", [({"private": False}, False), ({"extras": True}, False), ({"corrupt": True}, True)])
def test_failed_push_never_records_success_or_commits_public_unknown_inventory(release_fixture, kwargs, commit_expected):
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    api = MockHub(package, **kwargs)
    with pytest.raises(RuntimeError, match="publication failed"):
        push_publication(package, api=api, download=lambda repo, name, revision: package["files"][name]["local_path"])
    state = json.loads((fixture.stage / "publication.json").read_text())
    assert state["status"] == "failed" and state["verified"] is False
    assert ("create_commit" in api.calls) is commit_expected
    if commit_expected:
        assert state["commit"] == api.revision and state["failed_phase"] == "verification"


def test_changed_overlay_after_plan_is_rejected_before_any_hub_call(release_fixture):
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    (fixture.stage / "README.md").write_text("changed after review")
    api = MockHub(package)
    with pytest.raises(RuntimeError, match="validation"):
        push_publication(package, api=api)
    assert api.calls == []
    assert json.loads((fixture.stage / "publication.json").read_text())["verified"] is False


def test_package_cannot_replace_verified_weights_or_training_provenance_with_new_hashes(release_fixture):
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    substitute = fixture.stage / "other-model.safetensors"
    substitute.write_bytes((fixture.release / "model.safetensors").read_bytes())
    # Even identical bytes must be the verified immutable source file, not an
    # arbitrary package reference that could diverge before upload.
    package["files"]["model.safetensors"]["local_path"] = str(substitute)
    with pytest.raises(ValueError, match="standalone source"):
        validate_package(package)
    package["files"]["model.safetensors"]["local_path"] = str(fixture.release / "model.safetensors")
    overlay = json.loads((fixture.stage / "release.json").read_text())
    overlay["training"]["optimizer_steps"] = 16
    write_json(fixture.stage / "release.json", overlay)
    package["files"]["release.json"].update(sha256=file_sha(fixture.stage / "release.json"),
        bytes=(fixture.stage / "release.json").stat().st_size)
    package["total_bytes"] = sum(item["bytes"] for item in package["files"].values())
    with pytest.raises(ValueError, match="provenance"):
        validate_package(package)


def test_hub_verification_rejects_non_lfs_weight_instead_of_downloading_it(release_fixture):
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    api = MockHub(package)
    original = api.model_info

    def non_lfs(*args, **kwargs):
        result = original(*args, **kwargs)
        for file in result.siblings:
            if file.rfilename.endswith(".safetensors"):
                file.lfs = None
        return result
    api.model_info = non_lfs
    with pytest.raises(ValueError, match="LFS"):
        verify_remote(api, package, api.revision, lambda *args: pytest.fail("Weights must not download"))


def test_missing_hub_auth_records_failed_attempt_without_creating_repository(release_fixture, monkeypatch):
    import huggingface_hub
    fixture = release_fixture
    package = stage_publication(fixture.release, fixture.proof, fixture.stage)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: pytest.fail("Missing auth must not contact the Hub"))
    with pytest.raises(RuntimeError, match="authentication"):
        push_publication(package)
    state = json.loads((fixture.stage / "publication.json").read_text())
    assert state["status"] == "failed" and state["verified"] is False
    assert state["failed_phase"] == "authentication" and "commit" not in state

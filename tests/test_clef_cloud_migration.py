"""CPU migration integrity tests using real tiny files and native labeled rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage_clef_cloud_migration as stage
from clef_research_data import decision_prompt_hash


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def native_row(identifier, source="synthetic-fixture", source_type="research"):
    row = {"id": identifier, "state": {"observation": f"synthetic observed field {identifier}"},
        "questions": {"research_answer": {"type": "choice", "instructions": f"Choose the matching field for {identifier}",
            "criteria": {"option_0": "matching reference", "option_1": "different reference"}}},
        "labels": {"research_answer": "option_0"},
        "provenance": {"source": source, "source_type": source_type}}
    row["provenance"]["prompt_hash"] = decision_prompt_hash(row)
    return row


@pytest.fixture
def tiny_stage(tmp_path):
    """Small genuine safetensors inventories; no mock loaders or GPU model."""
    root = tmp_path / "stage"
    base, pilot = root / "base", root / "pilot"
    base.mkdir(parents=True)
    pilot.mkdir()
    for name in stage.BASE_SIDECARS | (stage.BASE_GENERATED - {"conversion_manifest.json"}):
        if name.endswith(".safetensors"):
            save_file({"head.weight": np.arange(3, dtype=np.float32)}, base / name)
        else:
            (base / name).write_text("{}\n")
    weights = {}
    for index in range(1, 10):
        name = f"model-{index:05d}-of-00009.safetensors"
        save_file({f"weight_{index}": np.arange(4, dtype=np.uint8)}, base / name)
        weights[name] = stage._metadata(base / name)
    write_json(base / "model.safetensors.index.json", {"weight_map": {f"weight_{index}": name for index, name in enumerate(weights)}})
    write_json(base / "conversion_manifest.json", {"status": "complete", "source": {"repo_id": stage.MODEL_ID, "revision": stage.MODEL_REVISION},
        "standalone_weights_saved": True, "reload_verified": True, "format": "standalone_hf_bnb_mps_nf4",
        "output_weight_files": weights, "preserved_sidecars": {name: stage._metadata(base / name) for name in stage.BASE_SIDECARS}})
    base_names, identity = stage._base_inventory(base)
    assert len(base_names) == 25
    for name in stage.PILOT_FILES:
        if name.endswith(".safetensors"):
            save_file({"trained.weight": np.arange(3, dtype=np.float32)}, pilot / name)
        else:
            (pilot / name).write_text("{}\n")
    security = {"applied": True, "version": 1, "published_source_modified": False,
        "exclusion_counts": {"train_excluded_signing_byte_array": 11, "train_excluded_encoded_signing_literal": 3}}
    before = {kind: {"sha256": hashlib.sha256(f"before-{kind}".encode()).hexdigest(), "parameters": 3} for kind in ("lora", "head")}
    trained = {kind: {"sha256": hashlib.sha256(f"after-{kind}".encode()).hexdigest(), "parameters": 3} for kind in ("lora", "head")}
    rows = {"train": [native_row(f"train-{index}", "observed-tape", "live_tape_observation") for index in range(4)]
        + [native_row("train-red", "https://arxiv.org/abs/2605.12151"), native_row("train-hour", "https://arxiv.org/abs/2606.08232")],
        "eval": [native_row("eval-1")], "test": [native_row("test-1")]}
    audits = {}
    for split, values in rows.items():
        path = root / stage.PATHS["cohorts"][split]
        write_rows(path, values)
        audits[split] = {"prepared": len(values) + (split == "train"), "selected": len(values), "sha256": stage.file_sha(path),
            "context_limit": 2048, "max_tokens": 10, "input_token_storage": "signed_int32_array", "input_token_bytes": len(values) * 40,
            "sources": dict(Counter(str(row["provenance"]["source"]) for row in values)),
            "required_source_indices": list(range(6)) if split == "train" else [],
            "excluded": [{"id": "too-long", "reason": "context_exceeds_limit", "tokens": 2049}] if split == "train" else []}
    prepared = {"dataset": stage.DATASET_ID, "dataset_revision": stage.DATASET_REVISION,
        "model": stage.MODEL_ID, "model_revision": stage.MODEL_REVISION,
        "input_parquet_sha256": stage.INPUT_PARQUET_SHA256,
        "input_source": {"input_rows": stage.INPUT_RAW_ROW_COUNTS}, "security_filter": security,
        "outputs": {split: {"rows": len(values)} for split, values in rows.items()}}
    write_json(root / stage.PATHS["prepared_manifest"], prepared)
    write_json(root / stage.PATHS["full_data_manifest"], {"prepared": prepared, "cohorts": audits})
    write_json(root / stage.PATHS["live_snapshot"], {"captured_at": "2026-10-02T00:00:00+00:00", "read_only": True})
    pilot_cohorts = {}
    for split in ("eval", "test"):
        path = root / stage.PATHS["pilot_cohorts"][split]
        write_rows(path, rows[split])
        pilot_cohorts[split] = {"selected": 1, "sha256": stage.file_sha(path)}
        prediction = {"id": rows[split][0]["id"], "answers": {"research_answer": {"choice": "option_0", "label": "option_0",
            "probabilities": {"option_0": 0.75, "option_1": 0.25}}}}
        write_rows(root / stage.PATHS["pilot_predictions"][split], [prediction])
    metadata = {"model": stage.MODEL_ID, "model_revision": stage.MODEL_REVISION,
        "dataset": stage.DATASET_ID, "dataset_revision": stage.DATASET_REVISION,
        "max_length": 2048, "optimizer_steps": 16, "planned_steps": 16, "status": "trained_and_reload_verified",
        "all_planned_steps_completed": True, "input_raw_row_counts": stage.INPUT_RAW_ROW_COUNTS,
        "input_parquet_sha256": stage.INPUT_PARQUET_SHA256, "security_filter": security,
        "local_base_manifest_sha256": identity["manifest_sha256"], "local_base_generated_files": identity["generated_files"],
        "initial_trainable_fingerprints": before, "trained_trainable_fingerprints": trained, "trainable_fingerprints": trained,
        "gradient_evidence": {"lora": True, "head": True},
        "finite_trainables": {kind: {"finite": True, "parameters": 3} for kind in trained},
        "reload_verification": {"matched": True, "records": 4, "max_probability_difference": 0},
        "lora": {"rank": 8, "text_layers": [60, 61, 62, 63]}, "cohorts": pilot_cohorts}
    write_json(pilot / "training.json", metadata)
    write_json(pilot / "adapter_artifact_manifest.json", {"files": {name: stage._metadata(pilot / name) for name in stage.PILOT_FILES}})
    return root, identity


def tiny_content(root, identity):
    return stage._content(root, identity, expected_cohorts={"train": 6, "eval": 1, "test": 1},
        expected_tokens=60, expected_token_bytes=240, expected_exclusions=1, pilot_count=1)


def test_real_tiny_inventory_and_native_provenance_pass_cpu_checks(tiny_stage):
    root, identity = tiny_stage
    files = {name: stage._metadata(root / name) for name in stage._payload_files(root)}
    assert len(files) == 56
    stage._verify_inventory(root, files)
    proof = tiny_content(root, identity)
    assert proof["cohorts"]["train"]["records"] == 6
    assert proof["cohorts"]["train"]["native_encoded_tokens"] == 60
    assert proof["pilot_cross_device_probe"]["test"]["records"] == 1
    assert proof["base_conversion_manifest_sha256"] == identity["manifest_sha256"]


def test_tampered_unlisted_and_symlink_files_are_rejected(tiny_stage, tmp_path):
    root, _ = tiny_stage
    files = {name: stage._metadata(root / name) for name in stage._payload_files(root)}
    path = root / "cohorts/train.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="integrity failed"):
        stage._verify_inventory(root, files)
    path.write_bytes(original)
    extra = root / "unapproved.py"
    extra.write_text("untrusted extra")
    with pytest.raises(ValueError, match="inventory differs"):
        stage._verify_inventory(root, files)
    extra.unlink()
    external = tmp_path / "outside"
    external.mkdir()
    (root / "external").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlinks"):
        stage._verify_inventory(root, files)


def test_operational_cache_is_excluded_from_inventory(tiny_stage):
    root, _ = tiny_stage
    files = {name: stage._metadata(root / name) for name in stage._payload_files(root)}
    cache = root / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / "download.metadata").write_text("operational cache only")
    stage._verify_inventory(root, files)
    assert all(".cache" not in name for name in stage._payload_files(root))


def test_exact_hardlink_and_independent_copy_preserve_original(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"weight": np.arange(8, dtype=np.float32)}, source)
    original = source.read_bytes()
    link = tmp_path / "linked/file.safetensors"
    copied = tmp_path / "copied/file.safetensors"
    expected, linked = stage._link_or_copy(source, link)
    assert linked and source.stat().st_ino == link.stat().st_ino
    assert stage._link_or_copy(source, copied, copy=True) == (expected, False)
    assert copied.stat().st_ino != source.stat().st_ino
    copied.write_bytes(b"independent output")
    assert source.read_bytes() == link.read_bytes() == original
    with pytest.raises(ValueError, match="Preserve existing"):
        stage._link_or_copy(source, link)


def test_duplicate_prompt_and_signing_literals_fail_before_model_load(tiny_stage):
    root, identity = tiny_stage
    train = list(stage._rows(root / "cohorts/train.jsonl"))
    leaked = train[0] | {"id": "eval-leaked"}
    write_rows(root / "cohorts/eval.jsonl", [leaked])
    with pytest.raises(ValueError, match="prompt leakage"):
        tiny_content(root, identity)
    unsafe = native_row("unsafe-fixture")
    unsafe["state"] = {"secretKey": [0] * 64}
    unsafe["provenance"]["prompt_hash"] = decision_prompt_hash(unsafe)
    write_rows(root / "unsafe.jsonl", [unsafe])
    with pytest.raises(ValueError, match="privacy guard"):
        stage._check_cohort(root / "unsafe.jsonl", 1)


def test_saved_probabilities_must_match_exact_native_rows(tiny_stage):
    root, _ = tiny_stage
    path = root / "pilot_predictions/eval.jsonl"
    prediction = list(stage._rows(path))[0]
    prediction["answers"]["research_answer"]["probabilities"]["option_0"] = float("nan")
    write_rows(path, [prediction])
    with pytest.raises(ValueError, match="probabilities"):
        stage._check_predictions(path, root / "pilot_cohorts/eval.jsonl", ["eval-1"])


def test_production_verifier_requires_exact_authorized_source_and_paths(tmp_path):
    root = tmp_path / "stage"
    root.mkdir()
    write_json(root / stage.MANIFEST_NAME, {"schema_version": 1, "status": "ready",
        "source": {"model": {"repo_id": "other/base", "revision": "0" * 40}}, "paths": stage.PATHS, "files": {}})
    with pytest.raises(ValueError, match="source/schema/paths"):
        stage.verify_stage(root)
    for unsafe in ("../weights", "/absolute", "base\\weights", "base/../weights", stage.MANIFEST_NAME):
        with pytest.raises(ValueError, match="Unsafe staged"):
            stage._safe_relative(unsafe)

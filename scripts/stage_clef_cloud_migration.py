#!/usr/bin/env python3
"""Stage verified existing Clef weights, pilot, and exact cloud training cohorts.

No authentication, downloads, GPU work, or uploads occur here. Default staging
hardlinks immutable source files; --copy gives independent byte-for-byte files.
Verification reads file hashes and labeled rows, never loads model tensors.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from clef_research_data import (DATASET_ID, MODEL_ID, MODEL_REVISION,
                                validate_decision_record)
from train_clef_local import (DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, file_sha, guard_model_input, validate_local_base,
    verify_adapter_manifest, verify_prepared_source)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "cloud_migration_manifest.json"
BASE_SIDECARS = {"LICENSE", "README.md", "chat_template.jinja",
    "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py",
    "processor_config.json", "tokenizer.json", "tokenizer_config.json"}
BASE_GENERATED = {"config.json", "conversion_manifest.json", "generation_config.json",
    "model.safetensors.index.json", "source_config.json", "source_generation_config.json",
    "source_model.safetensors.index.json"}
PILOT_FILES = {"LICENSE", "README.md", "adapter_config.json", "adapter_model.safetensors",
    "chat_template.jinja", "clef_live_tape.py", "clef_research_data.py",
    "clef_research_training.py", "evaluation.json", "joint_head.safetensors",
    "joint_head_config.json", "joint_schema_model.py", "live-training-snapshot.json",
    "processor_config.json", "research_expansion_artifacts.py", "source_base_model_card.md",
    "tokenizer.json", "tokenizer_config.json", "train_clef_local.py", "training.json"}
EXPECTED_COHORTS = {"train": 60635, "eval": 32, "test": 32}
EXPECTED_TRAIN_TOKENS = 67323884
EXPECTED_TRAIN_TOKEN_BYTES = 269295536
EXPECTED_CONTEXT_EXCLUSIONS = 17030
PATHS = {"base": "base", "pilot": "pilot",
    "cohorts": {s: f"cohorts/{s}.jsonl" for s in EXPECTED_COHORTS},
    "full_data_manifest": "data-manifest.json", "prepared_manifest": "prepared-manifest.json",
    "pilot_cohorts": {s: f"pilot_cohorts/{s}.jsonl" for s in ("eval", "test")},
    "pilot_predictions": {s: f"pilot_predictions/{s}.jsonl" for s in ("eval", "test")},
    "live_snapshot": "live-training-snapshot.json"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_relative(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or Path(value).is_absolute() or ".." in Path(value).parts
            or Path(value).as_posix() != value or value == MANIFEST_NAME):
        raise ValueError("Unsafe staged file reference")
    return Path(value)


def _regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular source file: {path.name}")
    return path


def _metadata(path):
    path = _regular_file(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha(path)}


def _rows(path):
    with _regular_file(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at line {number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {number}") from error


def _base_inventory(base):
    identity = validate_local_base(base)
    conversion = _json(base / "conversion_manifest.json")
    weights = set(conversion["output_weight_files"])
    if (len(weights) != 9 or set(conversion["preserved_sidecars"]) != BASE_SIDECARS
            or any(not re.fullmatch(r"model-0000[1-9]-of-00009\.safetensors", name) for name in weights)):
        raise ValueError("The pinned NF4 base requires exactly nine weight shards and verified native sidecars")
    index = _json(base / "model.safetensors.index.json")
    mapping = index.get("weight_map", {})
    if not mapping or set(mapping.values()) != weights:
        raise ValueError("NF4 index does not cover the exact verified shard inventory")
    for name in mapping.values():
        _safe_relative(name)
    return sorted(weights | BASE_SIDECARS | BASE_GENERATED), identity


def _pilot_proof(pilot, base_identity):
    manifest = verify_adapter_manifest(pilot)
    if set(manifest["files"]) != PILOT_FILES:
        raise ValueError("Pilot must contain exactly the verified 20-file adapter inventory")
    training = _json(pilot / "training.json")
    required = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "max_length": 2048, "optimizer_steps": 16, "planned_steps": 16,
        "status": "trained_and_reload_verified", "all_planned_steps_completed": True,
        "input_raw_row_counts": INPUT_RAW_ROW_COUNTS,
        "input_parquet_sha256": INPUT_PARQUET_SHA256,
        "local_base_manifest_sha256": base_identity["manifest_sha256"]}
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Pilot source, context, completed training, or base identity differs")
    if training.get("local_base_generated_files") != base_identity["generated_files"]:
        raise ValueError("Pilot generated base configs differ from the staged base")
    _security(training.get("security_filter"))
    fingerprints = training.get("trainable_fingerprints", {})
    for kind in ("lora", "head"):
        before = training.get("initial_trainable_fingerprints", {}).get(kind, {})
        after = training.get("trained_trainable_fingerprints", {}).get(kind, {})
        finite = training.get("finite_trainables", {}).get(kind, {})
        if (not training.get("gradient_evidence", {}).get(kind)
                or fingerprints.get(kind) != after
                or not re.fullmatch(r"[0-9a-f]{64}", str(after.get("sha256")))
                or before.get("sha256") == after.get("sha256")
                or type(after.get("parameters")) is not int or after["parameters"] <= 0
                or before.get("parameters") != after["parameters"]
                or finite != {"finite": True, "parameters": after["parameters"]}):
            raise ValueError("Pilot must preserve finite native-head and LoRA gradient/byte-update evidence")
    reload = training.get("reload_verification", {})
    if (reload.get("matched") is not True or reload.get("records") != 4
            or not isinstance(reload.get("max_probability_difference"), (int, float))
            or not math.isfinite(reload["max_probability_difference"])
            or not 0 <= reload["max_probability_difference"] <= 0.001):
        raise ValueError("Pilot lacks verified fresh-load numerical parity")
    if training.get("lora", {}).get("text_layers") != [60, 61, 62, 63] or training.get("lora", {}).get("rank") != 8:
        raise ValueError("Pilot LoRA must use the original last-four-layer/rank-eight scope")
    return training


def _security(audit):
    if (not isinstance(audit, dict) or audit.get("applied") is not True
            or audit.get("version") != 1 or audit.get("published_source_modified") is not False
            or audit.get("exclusion_counts") != {"train_excluded_signing_byte_array": 11,
                                                 "train_excluded_encoded_signing_literal": 3}):
        raise ValueError("The exact fourteen inherited signing-literal exclusions are required")


def _check_cohort(path, expected_count, *, global_prompts=None):
    count, ids, sources = 0, [], Counter()
    local_ids, local_prompts = set(), set()
    for row in _rows(path):
        prompt = validate_decision_record(row)
        guard_model_input(row)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in local_ids or prompt in local_prompts:
            raise ValueError("Duplicate or invalid native cohort identifiers/prompts")
        if global_prompts is not None:
            if prompt in global_prompts:
                raise ValueError("Native full-cohort prompt leakage between splits")
            global_prompts.add(prompt)
        local_ids.add(identifier)
        local_prompts.add(prompt)
        ids.append(identifier)
        count += 1
        sources[str(row.get("provenance", {}).get("source"))] += 1
    if count != expected_count:
        raise ValueError(f"Native cohort row count differs: expected {expected_count}, found {count}")
    return ids, dict(sources)


def _check_predictions(path, cohort_path, expected_ids):
    rows = list(_rows(cohort_path))
    predictions = list(_rows(path))
    if len(predictions) != len(expected_ids):
        raise ValueError("Pilot prediction row count differs from native input rows")
    for row, prediction, identifier in zip(rows, predictions, expected_ids, strict=True):
        answers = prediction.get("answers", {})
        if prediction.get("id") != identifier or set(answers) != set(row["questions"]):
            raise ValueError("Pilot prediction IDs/questions differ from exact input rows")
        for question, answer in answers.items():
            probabilities = answer.get("probabilities", {})
            options = row["questions"][question]["criteria"]
            if (set(probabilities) != set(options) or answer.get("choice") not in options
                    or answer.get("label") != row["labels"][question]
                    or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                           for value in probabilities.values())
                    or abs(sum(probabilities.values()) - 1) > 1e-5):
                raise ValueError("Pilot saved probabilities/labels/options are invalid")


def _content(stage, base_identity, *, expected_cohorts=EXPECTED_COHORTS,
             expected_tokens=EXPECTED_TRAIN_TOKENS, expected_token_bytes=EXPECTED_TRAIN_TOKEN_BYTES,
             expected_exclusions=EXPECTED_CONTEXT_EXCLUSIONS, pilot_count=8):
    """CPU semantic checks; explicit tiny contracts are used only by unit tests."""
    data = _json(stage / PATHS["full_data_manifest"])
    prepared = _json(stage / PATHS["prepared_manifest"])
    if data.get("prepared") != prepared:
        raise ValueError("Full cohort audit differs from the exact prepared manifest")
    verify_prepared_source(prepared)
    _security(prepared.get("security_filter"))
    pilot = _pilot_proof(stage / "pilot", base_identity)
    if pilot.get("security_filter") != prepared["security_filter"]:
        raise ValueError("Pilot and full preparation use different model-input exclusions")
    cohorts, global_prompts = {}, set()
    for split, count in expected_cohorts.items():
        path = stage / PATHS["cohorts"][split]
        ids, sources = _check_cohort(path, count, global_prompts=global_prompts)
        audit = data.get("cohorts", {}).get(split, {})
        if (audit.get("selected") != count or audit.get("sha256") != file_sha(path)
                or audit.get("context_limit") != 2048 or not 0 < audit.get("max_tokens", 0) <= 2048
                or audit.get("input_token_storage") != "signed_int32_array"
                or type(audit.get("input_token_bytes")) is not int or audit["input_token_bytes"] <= 0
                or audit["input_token_bytes"] % 4 or audit.get("sources") != sources):
            raise ValueError("Exact native cohort hashes/context/counts/token storage differ")
        cohorts[split] = {"records": count, "sha256": audit["sha256"],
            "context_limit": 2048, "max_tokens": audit["max_tokens"],
            "native_encoded_tokens": audit["input_token_bytes"] // 4,
            "input_token_bytes": audit["input_token_bytes"],
            "context_excluded_records": sum(item.get("reason") == "context_exceeds_limit" for item in audit.get("excluded", []))}
        if split == "train":
            if (audit["input_token_bytes"] != expected_token_bytes or audit["input_token_bytes"] // 4 != expected_tokens
                    or cohorts[split]["context_excluded_records"] != expected_exclusions
                    or len(audit.get("excluded", [])) != expected_exclusions
                    or audit.get("prepared") != count + expected_exclusions
                    or any(item.get("reason") != "context_exceeds_limit" or item.get("tokens", 0) <= 2048
                           for item in audit.get("excluded", []))):
                raise ValueError("Full training token/context-exclusion accounting differs")
            required = audit.get("required_source_indices", [])
            if len(required) != 6 or len(set(required)) != 6 or any(type(index) is not int or not 0 <= index < len(ids) for index in required):
                raise ValueError("The full cohort requires six captured-live/paper examples")
            required_rows = []
            for index, row in enumerate(_rows(path)):
                if index in required:
                    required_rows.append(row)
                if index >= max(required):
                    break
            if (sum(row["provenance"].get("source_type") == "live_tape_observation" for row in required_rows) != 4
                    or any(not any(paper in str(row["provenance"].get("source")) for row in required_rows)
                           for paper in ("2605.12151", "2606.08232"))):
                raise ValueError("Required source indices do not identify the original four live and two paper decisions")
    probes = {}
    for split in ("eval", "test"):
        path = stage / PATHS["pilot_cohorts"][split]
        ids, _ = _check_cohort(path, pilot_count)
        audit = pilot.get("cohorts", {}).get(split, {})
        if audit.get("selected") != pilot_count or audit.get("sha256") != file_sha(path):
            raise ValueError("Cross-device pilot rows differ from the trained pilot cohort")
        _check_predictions(stage / PATHS["pilot_predictions"][split], path, ids)
        probes[split] = {"records": pilot_count, "cohort_sha256": file_sha(path),
                        "predictions_sha256": file_sha(stage / PATHS["pilot_predictions"][split])}
    snapshot = _json(stage / PATHS["live_snapshot"])
    # The originating adapter preserves its own capture; the full run has its
    # independently captured read-only snapshot and live records in the cohort.
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("captured_at"), str):
        raise ValueError("Missing exact full-run captured-live provenance")
    return {"cohorts": cohorts, "pilot_cross_device_probe": probes,
        "input_raw_row_counts": INPUT_RAW_ROW_COUNTS,
        "input_parquet_sha256": INPUT_PARQUET_SHA256,
        "security_filter": prepared["security_filter"],
        "prepared_split_counts": {s: v["rows"] for s, v in prepared["outputs"].items()},
        "pilot_trainable_fingerprints": pilot["trainable_fingerprints"],
        "pilot_optimizer_steps": pilot["optimizer_steps"],
        "base_conversion_manifest_sha256": base_identity["manifest_sha256"],
        "pilot_adapter_manifest_sha256": file_sha(stage / "pilot/adapter_artifact_manifest.json"),
        "live_snapshot_sha256": file_sha(stage / PATHS["live_snapshot"])}


def _payload_files(stage):
    files = set()
    for directory, dirs, names in os.walk(stage, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if path.is_symlink():
                raise ValueError("Cloud staging may not contain directory symlinks")
            if name in {".cache", "__pycache__"}:
                dirs.remove(name)
        for name in names:
            path = parent / name
            _regular_file(path)
            relative = path.relative_to(stage).as_posix()
            if relative != MANIFEST_NAME:
                files.add(relative)
    return files


def _verify_inventory(stage, files):
    if not isinstance(files, dict) or set(files) != _payload_files(stage):
        raise ValueError("Cloud staging payload inventory differs from actual files")
    for name, expected in files.items():
        relative = _safe_relative(name)
        path = stage / relative
        if (not isinstance(expected, dict) or type(expected.get("bytes")) is not int
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256")))
                or _metadata(path) != expected):
            raise ValueError(f"Cloud staging file integrity failed: {name}")


def _link_or_copy(source, destination, *, copy=False):
    expected = _metadata(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Preserve existing staging files")
    hardlinked = not copy and source.stat().st_dev == destination.parent.stat().st_dev
    if hardlinked:
        os.link(source, destination)
    else:
        shutil.copyfile(source, destination)
    if _metadata(destination) != expected:
        raise ValueError("Source changed or copy differed during cloud staging")
    return expected, hardlinked


def verify_stage(stage: Path) -> dict:
    """Require exact immutable payload inventory and CPU provenance before GPU work."""
    stage = Path(stage)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Cloud stage must be an existing regular directory")
    stage = stage.resolve()
    manifest = _json(_regular_file(stage / MANIFEST_NAME))
    expected_source = {"model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {"repo_id": DATASET_ID, "revision": DATASET_REVISION}}
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "ready"
            or manifest.get("source") != expected_source or manifest.get("paths") != PATHS):
        raise ValueError("Cloud migration manifest source/schema/paths are not the exact authorized run")
    files = manifest.get("files")
    _verify_inventory(stage, files)
    base_files, identity = _base_inventory(stage / "base")
    actual_base = {name.removeprefix("base/") for name in files if name.startswith("base/")}
    if set(base_files) != actual_base:
        raise ValueError("Cloud stage includes unapproved base artifacts")
    allowed = {f"base/{name}" for name in base_files}
    allowed |= {f"pilot/{name}" for name in PILOT_FILES | {"adapter_artifact_manifest.json"}}
    allowed |= set(PATHS["cohorts"].values()) | set(PATHS["pilot_cohorts"].values()) | set(PATHS["pilot_predictions"].values())
    allowed |= {PATHS[key] for key in ("full_data_manifest", "prepared_manifest", "live_snapshot")}
    if set(files) != allowed:
        raise ValueError("Cloud stage includes artifacts outside the exact migration allowlist")
    proof = _content(stage, identity)
    if any(manifest.get(key) != value for key, value in proof.items()):
        raise ValueError("Cloud staging metadata differs from exact saved artifact evidence")
    if manifest.get("total_bytes") != sum(item["bytes"] for item in files.values()):
        raise ValueError("Cloud staging byte total differs from inventory")
    return manifest


def stage_migration(base: Path, pilot_run: Path, full: Path, output: Path, *, copy=False) -> dict:
    """Create a fresh task-owned portable stage without modifying source files."""
    base, pilot_run, full = (Path(path).resolve() for path in (base, pilot_run, full))
    pilot = pilot_run / "adapter"
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Staging destination may not be a symlink")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Preserve existing artifacts; staging output must be new or empty")
    if any(output == source or output.is_relative_to(source) or source.is_relative_to(output)
           for source in (base, pilot_run, full)):
        raise ValueError("Staging and original artifact directories must be separate")
    base_files, identity = _base_inventory(base)
    _pilot_proof(pilot, identity)
    sources = {f"base/{name}": base / name for name in base_files}
    sources.update({f"pilot/{name}": pilot / name for name in sorted(PILOT_FILES | {"adapter_artifact_manifest.json"})})
    sources.update({PATHS["cohorts"][s]: full / "cohorts" / f"{s}.jsonl" for s in EXPECTED_COHORTS})
    sources.update({PATHS["full_data_manifest"]: full / "data-manifest.json",
        PATHS["prepared_manifest"]: full / "prepared/manifest.json",
        PATHS["live_snapshot"]: full / "live-training-snapshot.json"})
    for split in ("eval", "test"):
        sources[PATHS["pilot_cohorts"][split]] = pilot_run / "cohorts" / f"{split}.jsonl"
        sources[PATHS["pilot_predictions"][split]] = pilot_run / f"{split}-predictions.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    files, links, copies = {}, 0, 0
    for name, source in sorted(sources.items()):
        destination = output / _safe_relative(name)
        expected, hardlinked = _link_or_copy(source, destination, copy=copy)
        if hardlinked:
            links += 1
        else:
            copies += 1
        files[name] = expected
    proof = _content(output, identity)
    manifest = {"schema_version": 1, "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
                   "dataset": {"repo_id": DATASET_ID, "revision": DATASET_REVISION}},
        "paths": PATHS, "files": files, "total_bytes": sum(x["bytes"] for x in files.values()),
        "local_staging": {"hardlinked_files": links, "copied_files": copies,
            "sources_modified": False, "weights_loaded": False, "uploaded": False},
        "operational_cache_policy": "ignore .cache and __pycache__; never include them in payload inventory or uploads",
        **proof}
    temporary = output / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / MANIFEST_NAME)
    return verify_stage(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "local/clef-27b-mps-nf4")
    parser.add_argument("--pilot-run", type=Path, default=ROOT / "outputs/clef-local-pilot")
    parser.add_argument("--full", type=Path, default=ROOT / "outputs/clef-local-full")
    parser.add_argument("--output", type=Path, default=ROOT / "local/clef-cloud-migration-20261002")
    parser.add_argument("--copy", action="store_true", help="Copy regular files instead of same-filesystem hardlinks")
    parser.add_argument("--verify-only", action="store_true", help="Verify the existing output, without staging again")
    args = parser.parse_args()
    manifest = (verify_stage(args.output) if args.verify_only else
        stage_migration(args.base, args.pilot_run, args.full, args.output, copy=args.copy))
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"]),
        "total_bytes": manifest["total_bytes"], "cohorts": manifest["cohorts"],
        "local_staging": manifest["local_staging"]}, indent=2))


if __name__ == "__main__":
    main()

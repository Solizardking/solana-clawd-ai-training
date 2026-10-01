#!/usr/bin/env python3
"""Plan, or explicitly publish, a fully trained and verified private Clef release.

The staging overlay contains only small publication records. Original model
files are referenced directly and the hash-covered standalone release is never
modified. Planning does not contact the Hub or create a repository.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION, model_input_exclusion_reason
from export_clef_mps_release import (EXPANDED_DATASET_REVISION, EXPANDED_INPUT_ROWS,
                                     verify_mps_release)

DEFAULT_REPO = "solanaclawd/clef-solana-research"
SMALL_FILE_LIMIT = 2 * 1024 ** 2
OPERATIONAL_PARTS = {"__pycache__", ".cache", ".git", ".huggingface"}
OVERLAY_FILES = {"README.md", "inference-evidence.json", "source-release.json", "source-trained-model-card.md"}


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def parse_time(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} requires an actual timezone-aware timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} requires an actual timezone-aware timestamp") from None
    if result.tzinfo is None:
        raise ValueError(f"{name} requires an actual timezone-aware timestamp")
    return result


def validate_full_training(training):
    """Reject bounded pilots even when their status is reload verified."""
    if training.get("run_mode") != "full" or training.get("all_planned_steps_completed") is not True:
        raise ValueError("Publication requires a completed full training run")
    if training.get("complete_selected_training_epochs") is not True:
        raise ValueError("The selected training epochs were not completed")
    if training.get("selected_training_scope") != "all context-eligible prepared rows":
        raise ValueError("A bounded pilot cohort cannot be published as full training")
    cohorts, prepared = training.get("cohorts", {}), training.get("prepared_split_counts", {})
    cohort = cohorts.get("train", {})
    selected = positive_integer(cohort.get("selected"), "Selected training records")
    count = positive_integer(cohort.get("prepared"), "Prepared training records")
    if count != prepared.get("train") or selected > count:
        raise ValueError("Training cohort does not account for its prepared source")
    exclusions = cohort.get("excluded")
    if not isinstance(exclusions, list):
        raise ValueError("Full training requires its recorded context exclusions")
    excluded_ids = set()
    for item in exclusions:
        if not isinstance(item, dict) or item.get("reason") != "context_exceeds_limit" or not isinstance(item.get("id"), str):
            raise ValueError("Full training has an unexplained input exclusion")
        if item["id"] in excluded_ids:
            raise ValueError("Full training exclusion records are duplicated")
        excluded_ids.add(item["id"])
    if selected + len(excluded_ids) != count or training.get("trained_records") != selected:
        raise ValueError("Every context-eligible prepared training record must actually be trained")
    sources = cohort.get("sources", {})
    if not isinstance(sources, dict) or any(type(value) is not int or value < 1 for value in sources.values()) or sum(sources.values()) != selected:
        raise ValueError("Training source counts differ from the selected cohort")
    if not all(any(paper in str(source) for source in sources) for paper in ("2605.12151", "2606.08232")):
        raise ValueError("Full training must include both cited Kamat papers")
    if not any("clawd-ws.fly.dev" in str(source) for source in sources):
        raise ValueError("Full training must include captured read-only live observations")
    epochs = positive_integer(training.get("epochs"), "Epochs")
    accumulation = positive_integer(training.get("gradient_accumulation"), "Gradient accumulation")
    expected_steps = epochs * math.ceil(selected / accumulation)
    if training.get("planned_steps") != expected_steps or training.get("optimizer_steps") != expected_steps:
        raise ValueError("Full training optimizer steps do not complete the recorded epochs")
    if training.get("cpu_fallback") is not False or training.get("vision_trained") is not False or training.get("autoExecute") is not False:
        raise ValueError("Training must record actual MPS, frozen vision and no execution permission")
    security = training.get("security_filter", {})
    counts = security.get("exclusion_counts", {})
    if security.get("published_source_modified") is not False or not isinstance(counts, dict) or any(
        type(value) is not int or value < 0 for value in counts.values()
    ):
        raise ValueError("Training requires safe privacy exclusion counts and preserved published source")
    if counts.get("train_excluded_signing_byte_array", 0) < 11 or counts.get("train_excluded_encoded_signing_literal", 0) < 3:
        raise ValueError("The audited inherited signing and credential-bearing rows were not excluded")
    for kind in ("lora", "head"):
        if training.get("gradient_evidence", {}).get(kind) is not True:
            raise ValueError("Both LoRA and native head require actual gradient evidence")
        before = training.get("initial_trainable_fingerprints", {}).get(kind, {})
        after = training.get("trained_trainable_fingerprints", {}).get(kind, {})
        for entry in (before, after):
            positive_integer(entry.get("parameters"), "Trainable parameter count")
            if not isinstance(entry.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
                raise ValueError("Native head and LoRA require measured byte fingerprints")
        if before["sha256"] == after["sha256"] or before["parameters"] != after["parameters"]:
            raise ValueError("Native head and LoRA must both have changed parameter bytes")
        if training.get("trainable_fingerprints", {}).get(kind) != after:
            raise ValueError("Saved trainable fingerprints differ from the trained parameters")
        finite = training.get("finite_trainables", {}).get(kind, {})
        if finite.get("finite") is not True or finite.get("parameters") != after["parameters"]:
            raise ValueError("Saved native head and LoRA must have finite trained parameters")
    return training


def validate_inference(path, release, *, now=None):
    """Validate freshness when inference ran, preserving that historical time."""
    path, release = Path(path), Path(release).resolve()
    if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(release):
        raise ValueError("Inference evidence must be a separate regular file outside the release")
    evidence = json.loads(path.read_text())
    allowed = {"responses", "excluded", "model_path", "model_revision", "dataset_revision", "training_status",
               "captured_at", "inference_completed_at", "release_manifest_sha256", "backend", "device",
               "standalone_model_loaded", "source", "freshness", "scope"}
    if not isinstance(evidence, dict) or set(evidence) - allowed:
        raise ValueError("Inference evidence contains unsupported fields")
    if model_input_exclusion_reason(evidence):
        raise ValueError("Inference evidence contains credential or signing material")
    if evidence.get("model_revision") != MODEL_REVISION or evidence.get("dataset_revision") != EXPANDED_DATASET_REVISION:
        raise ValueError("Inference evidence uses a different immutable model or research source")
    if evidence.get("training_status") != "trained_and_standalone_reload_verified" or evidence.get("standalone_model_loaded") is not True:
        raise ValueError("Inference evidence must come from the verified full standalone model")
    if evidence.get("device") != "mps" or evidence.get("backend") != "mps" or Path(evidence.get("model_path", "")).resolve() != release:
        raise ValueError("Inference evidence does not identify this actual local MPS release")
    if evidence.get("source") != "https://clawd-ws.fly.dev/":
        raise ValueError("Inference evidence does not identify the requested read-only live tape")
    if evidence.get("release_manifest_sha256") != file_sha(release / "release.json"):
        raise ValueError("Inference evidence is not bound to the current standalone release")
    captured = parse_time(evidence.get("captured_at"), "Observed time")
    completed = parse_time(evidence.get("inference_completed_at"), "Inference completion time")
    now = now or datetime.now(timezone.utc)
    age = (completed - captured).total_seconds()
    if completed > now or captured > completed or not 0 <= age <= 30:
        raise ValueError("Inference did not use a fresh observation at its recorded execution time")
    freshness = evidence.get("freshness", {})
    if freshness.get("stale") is not False:
        raise ValueError("Inference evidence marks its observation as stale")
    recorded_age = finite_number(freshness.get("snapshot_age_seconds"), "Snapshot age", minimum=0, maximum=30)
    if abs(recorded_age - age) > 1:
        raise ValueError("Recorded inference freshness differs from its timestamps")
    observations = freshness.get("observation_age_seconds")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Fresh inference requires actual received observation ages")
    for observed_age in observations:
        finite_number(observed_age, "Observation age", minimum=0, maximum=30)
    responses = evidence.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("Inference evidence has no actual native responses")
    for response in responses:
        if not isinstance(response, dict) or not isinstance(response.get("id"), str) or not isinstance(response.get("answers"), dict) or not response["answers"]:
            raise ValueError("Inference response is incomplete")
        usage = response.get("usage", {})
        positive_integer(usage.get("input_tokens"), "Inference input tokens")
        if usage.get("output_tokens") != 0:
            raise ValueError("Expected native typed choices without generated text")
        for answer in response["answers"].values():
            if not isinstance(answer, dict):
                raise ValueError("Native answer is incomplete")
            probabilities = answer.get("probabilities")
            if answer.get("type") != "choice" or not isinstance(probabilities, dict) or len(probabilities) < 2:
                raise ValueError("Expected actual native choice probabilities")
            for probability in probabilities.values():
                finite_number(probability, "Native probability", minimum=0, maximum=1)
            if abs(sum(probabilities.values()) - 1) > 0.005 or answer.get("choice") not in probabilities:
                raise ValueError("Native probabilities or chosen option are invalid")
            # Native SystemOne emits four-decimal probabilities; its choice is
            # selected before rounding, so equally rounded options can tie.
            if probabilities[answer["choice"]] + 0.0001 < max(probabilities.values()):
                raise ValueError("Native choice does not match its highest probability")
            confidence = finite_number(answer.get("confidence"), "Native confidence", minimum=0, maximum=1)
            if abs(confidence - probabilities[answer["choice"]]) > 0.001:
                raise ValueError("Native confidence differs from its chosen probability")
    return evidence


def validated_license(release):
    text = (release / "LICENSE").read_text()
    source = (release / "source_base_model_card.md").read_text()
    match = re.search(r"^license:\s*([a-zA-Z0-9.-]+)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError("Original base model card has no license identifier")
    license_id = match.group(1)
    if license_id == "apache-2.0" and "Apache License" in text and "Version 2.0" in text:
        return license_id
    if license_id == "mit" and "MIT License" in text and "Permission is hereby granted" in text:
        return license_id
    raise ValueError("Original source model card and preserved license text are inconsistent")


def validate_evaluation(value):
    for split in ("baseline_eval", "eval", "test"):
        metric = value.get(split, {})
        positive_integer(metric.get("records"), "Evaluation records")
        finite_number(metric.get("accuracy"), "Evaluation accuracy", minimum=0, maximum=1)
        finite_number(metric.get("chance_accuracy"), "Evaluation chance accuracy", minimum=0, maximum=1)
        finite_number(metric.get("nll"), "Evaluation NLL", minimum=0)
    return value


def model_card(training, metrics, evidence, license_id):
    lines = ["---", f"license: {license_id}", "library_name: transformers", f"base_model: {MODEL_ID}",
             "base_model_relation: finetune", "tags:", "- clef", "- solana", "- structured-output",
             "- custom-code", "---", "", "# Clawd Clef Solana research", "",
             "A complete standalone NF4 Clef backbone with the trained native joint schema head. "
             "Final text-layer LoRA was trained and merged; the original vision encoder and readable output embeddings remain frozen BF16.", "",
             f"Original model: `{MODEL_ID}@{MODEL_REVISION}`. Dataset: `{DATASET_ID}@{EXPANDED_DATASET_REVISION}`.",
             f"The immutable research source contains {sum(EXPANDED_INPUT_ROWS.values()):,} rows. "
             f"This full run completed {training['epochs']} selected training epoch(s), {training['optimizer_steps']:,} optimizer steps, "
             f"and trained all {training['trained_records']:,} context-eligible selected training records.", "",
             "| Split | Raw source rows | Prepared decisions | Selected records | Context exclusions |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("train", "eval", "test"):
        cohort = training["cohorts"][split]
        lines.append(f"| {split} | {EXPANDED_INPUT_ROWS[split]:,} | {training['prepared_split_counts'][split]:,} | "
                     f"{cohort['selected']:,} | {len(cohort['excluded']):,} |")
    exclusions = sum(training['security_filter'].get('exclusion_counts', {}).values())
    lines += ["", f"Credential/signing-literal filtering removed {exclusions} source records before answer-candidate construction. "
              "Original published data was preserved. Live choices are labeled from captured observed fields; source rows and output decisions are different units.", "",
              "| Evaluation | Records | Accuracy | Chance accuracy | NLL |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in ("baseline_eval", "eval", "test"):
        metric = metrics[label]
        lines.append(f"| {label} | {metric['records']} | {metric['accuracy']:.6f} | {metric['chance_accuracy']:.6f} | {metric['nll']:.6f} |")
    lines += ["", "These are reference-answer selection and observed-field readback results. Distractors are inherited answers to other prompts; "
              "source documents overlap across splits. No official Decision Index score, trading-outcome improvement, image-training improvement, or SOTA result is claimed.", "",
              f"Fresh standalone inference was observed at `{evidence['captured_at']}` and completed at `{evidence['inference_completed_at']}`. "
              "The observation was fresh when inference ran; the published proof is a timestamped historical record, not current market data.", "",
              "`release.json` covers every published runtime, processor, head, and backbone file. It records adapter-to-merged prediction parity "
              "within 0.005 and fresh standalone saved-weight reload parity within 0.0001. Training metadata records actual LoRA/head gradients and changed parameter bytes.", "",
              "Use `python run_clef_local.py --model /path/to/downloaded/release` for fresh read-only choices on Apple MPS. "
              "The Brain/Hands boundary keeps signing keypairs outside the model. The model has no signing or automatic trading permissions.", "",
              f"The source license is `{license_id}`, verified from the retained original model card and LICENSE. "
              "See `source_base_model_card.md`, `LICENSE`, `training.json`, `evaluation.json`, and `inference-evidence.json` for provenance and measured scope. "
              "`source-release.json` and `source-trained-model-card.md` preserve the exact pre-publication manifest and trained card, so the inference proof's original manifest hash can be checked independently.", "",
              "- Kamat, A. U. (2026). RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels. arXiv:2605.12151. https://arxiv.org/abs/2605.12151",
              "- Kamat, A. U. (2026). Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading. arXiv:2606.08232. https://arxiv.org/abs/2606.08232", ""]
    return "\n".join(lines)


def stage_publication(release, inference, stage, repo_id=DEFAULT_REPO):
    release, stage = Path(release).resolve(), Path(stage).resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id) is None:
        raise ValueError("Expected an owner/model repository ID")
    if stage.is_relative_to(release) or release.is_relative_to(stage):
        raise ValueError("Publication staging must stay separate from the verified release")
    if stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
        raise ValueError("Publication staging requires a fresh empty directory")
    manifest = verify_mps_release(release, expected_dataset_revision=EXPANDED_DATASET_REVISION)
    training = validate_full_training(manifest["training"])
    evidence = validate_inference(inference, release)
    required = {"README.md", "evaluation.json", "LICENSE", "source_base_model_card.md", "run_clef_local.py", "export_clef_mps_release.py", "export_clef_release.py"}
    if not required <= set(manifest["files"]):
        raise ValueError("Standalone release lacks hashed evaluation, licensing or runtime files")
    metrics = validate_evaluation(json.loads((release / "evaluation.json").read_text()))
    for split in ("baseline_eval", "eval", "test"):
        cohort = training["cohorts"]["eval" if split == "baseline_eval" else split]
        if metrics[split]["records"] != cohort["selected"]:
            raise ValueError("Evaluation metrics differ from the recorded selected cohort")
    license_id = validated_license(release)
    card = model_card(training, metrics, evidence, license_id)
    if set(manifest["files"]) & (OVERLAY_FILES - {"README.md"}):
        raise ValueError("Use the original verified local export, not an already overlaid publication")
    files = {}
    for name, metadata in manifest["files"].items():
        relative = Path(name)
        if OPERATIONAL_PARTS.intersection(relative.parts):
            continue
        path = release / relative
        if path.stat().st_size <= SMALL_FILE_LIMIT and relative.suffix in {".json", ".py", ".md", ".txt", ".jinja"}:
            if model_input_exclusion_reason(path.read_text()):
                raise ValueError("A publication sidecar contains credential or signing material")
        files[name] = {**metadata, "local_path": str(path)}
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "README.md").write_text(card, encoding="utf-8")
    write_json(stage / "inference-evidence.json", evidence)
    (stage / "source-release.json").write_bytes((release / "release.json").read_bytes())
    (stage / "source-trained-model-card.md").write_bytes((release / "README.md").read_bytes())
    for name in sorted(OVERLAY_FILES):
        path = stage / name
        files[name] = {"local_path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}
    overlay = copy.deepcopy(manifest)
    overlay["files"] = {name: {"sha256": item["sha256"], "bytes": item["bytes"]} for name, item in files.items()}
    overlay["total_bytes"] = sum(item["bytes"] for item in files.values())
    overlay["publication"] = {"visibility": "private", "repo_id": repo_id,
                              "source_release_manifest_sha256": file_sha(release / "release.json"),
                              "inference_observed_at": evidence["captured_at"], "inference_completed_at": evidence["inference_completed_at"],
                              "inference_evidence_sha256": files["inference-evidence.json"]["sha256"]}
    write_json(stage / "release.json", overlay)
    files["release.json"] = {"local_path": str(stage / "release.json"), "sha256": file_sha(stage / "release.json"),
                             "bytes": (stage / "release.json").stat().st_size}
    package = {"schema_version": 1, "status": "planned", "repo_id": repo_id, "private": True,
               "release_dir": str(release), "stage_dir": str(stage), "files": files,
               "source_release_manifest_sha256": overlay["publication"]["source_release_manifest_sha256"],
               "total_bytes": sum(item["bytes"] for item in files.values()),
               "dataset_revision": EXPANDED_DATASET_REVISION, "model_revision": MODEL_REVISION,
               "observed_at": evidence["captured_at"], "completed_optimizer_steps": training["optimizer_steps"]}
    write_json(stage / "package.json", package)
    return package


def validate_package(package):
    release, stage = Path(package["release_dir"]).resolve(), Path(package["stage_dir"]).resolve()
    if stage.is_relative_to(release) or release.is_relative_to(stage):
        raise ValueError("Publication staging must stay separate from its immutable release")
    manifest = verify_mps_release(release, expected_dataset_revision=EXPANDED_DATASET_REVISION)
    validate_full_training(manifest["training"])
    if package.get("private") is not True or file_sha(release / "release.json") != package["source_release_manifest_sha256"]:
        raise ValueError("Publication source identity changed after planning")
    if package.get("model_revision") != MODEL_REVISION or package.get("dataset_revision") != EXPANDED_DATASET_REVISION:
        raise ValueError("Publication package immutable pins differ from the verified source")
    evidence = validate_inference(stage / "inference-evidence.json", release)
    overlay = json.loads((stage / "release.json").read_text())
    original_files = {name: item for name, item in manifest["files"].items()
                      if not OPERATIONAL_PARTS.intersection(Path(name).parts)}
    expected_files = set(original_files) | OVERLAY_FILES
    if set(overlay["files"]) != expected_files:
        raise ValueError("Publication overlay differs from the verified source artifact inventory")
    if set(overlay["files"]) | {"release.json"} != set(package["files"]):
        raise ValueError("Publication overlay inventory is inconsistent")
    for name, expected in package["files"].items():
        path = Path(expected["local_path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
            raise ValueError("Publication files changed after planning")
        if name != "release.json" and overlay["files"][name] != {key: expected[key] for key in ("sha256", "bytes")}:
            raise ValueError("Publication overlay hash does not describe committed bytes")
        if name in OVERLAY_FILES | {"release.json"}:
            if path.resolve() != stage / name:
                raise ValueError("Publication overlay file escapes its reviewed stage")
        elif path.resolve() != release / name or overlay["files"][name] != original_files[name]:
            raise ValueError("Publication model files differ from the verified standalone source")
    if package["files"]["source-release.json"]["sha256"] != package["source_release_manifest_sha256"]:
        raise ValueError("Archived source manifest differs from the historical inference binding")
    if overlay["files"]["source-trained-model-card.md"] != original_files["README.md"]:
        raise ValueError("Archived trained model card differs from the immutable local export")
    expected_overlay = copy.deepcopy(manifest)
    expected_overlay["files"] = overlay["files"]
    expected_overlay["total_bytes"] = sum(item["bytes"] for item in overlay["files"].values())
    expected_overlay["publication"] = {"visibility": "private", "repo_id": package["repo_id"],
        "source_release_manifest_sha256": package["source_release_manifest_sha256"],
        "inference_observed_at": evidence["captured_at"], "inference_completed_at": evidence["inference_completed_at"],
        "inference_evidence_sha256": package["files"]["inference-evidence.json"]["sha256"]}
    if overlay != expected_overlay or package["total_bytes"] != sum(item["bytes"] for item in package["files"].values()):
        raise ValueError("Publication overlay provenance or total bytes differ from the verified source")
    return package


def prepare_or_reuse(release, inference, stage, repo_id=DEFAULT_REPO):
    """Let an explicitly requested push use the already reviewed plan."""
    stage = Path(stage)
    package_path = stage / "package.json"
    if not package_path.exists():
        return stage_publication(release, inference, stage, repo_id)
    package = json.loads(package_path.read_text())
    if Path(release).resolve() != Path(package["release_dir"]).resolve() or repo_id != package["repo_id"] or Path(package["stage_dir"]).resolve() != stage.resolve():
        raise ValueError("Existing publication stage belongs to a different release or repository")
    evidence = validate_inference(inference, release)
    if evidence != json.loads((stage / "inference-evidence.json").read_text()):
        raise ValueError("Inference proof changed after the publication plan; choose a fresh stage")
    return validate_package(package)


def verify_remote(api, package, revision, download):
    """Use LFS metadata for weights; download only bounded small sidecars."""
    info = api.model_info(package["repo_id"], revision=revision, files_metadata=True)
    if info.sha != revision or info.private is not True:
        raise ValueError("Published model commit or private visibility differs from the plan")
    siblings = {item.rfilename: item for item in info.siblings}
    if set(siblings) - {".gitattributes"} != set(package["files"]):
        raise ValueError("Published model has missing or unexpected artifact files")
    verified = []
    for name, expected in package["files"].items():
        remote = siblings[name]
        if remote.size != expected["bytes"]:
            raise ValueError("Published artifact size differs from its verified local bytes")
        if remote.lfs is not None:
            observed = remote.lfs.sha256
        else:
            if name.endswith(".safetensors") or expected["bytes"] > SMALL_FILE_LIMIT:
                raise ValueError("Weight verification requires Hub LFS SHA256 metadata; no weights are downloaded")
            observed = file_sha(download(package["repo_id"], name, revision))
        if observed != expected["sha256"]:
            raise ValueError("Published artifact hash differs from its verified local bytes")
        verified.append(name)
    return {"files_verified": len(verified), "verified_files": verified, "weight_downloaded_locally": False}


def push_publication(package, *, api=None, download=None):
    state = {"status": "publishing", "repo_id": package["repo_id"], "private": True,
             "dataset_revision": EXPANDED_DATASET_REVISION, "model_revision": MODEL_REVISION,
             "observed_at": package["observed_at"], "verified": False}
    status_path = Path(package["stage_dir"]) / "publication.json"
    write_json(status_path, state)
    phase = "validation"
    try:
        validate_package(package)
        from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
        phase = "authentication"
        if api is None:
            token = get_token()
            if not token:
                raise ValueError("Publishing requires HF_TOKEN or a cached Hugging Face login")
            api = HfApi(token=token)
        if download is None:
            download = lambda repo, name, revision: hf_hub_download(repo, name, revision=revision, token=api.token,
                                                                  local_dir=Path(package["stage_dir"]) / "remote-verification")
        api.whoami()
        phase = "repository"
        api.create_repo(package["repo_id"], repo_type="model", private=True, exist_ok=True)
        current = api.model_info(package["repo_id"])
        if current.private is not True:
            raise ValueError("Publication requires a private destination repository")
        if {item.rfilename for item in current.siblings} - set(package["files"]) - {".gitattributes"}:
            raise ValueError("Destination has unknown leftover files; choose a separate private repository")
        phase = "commit"
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=item["local_path"])
                      for name, item in package["files"].items()]
        commit = api.create_commit(package["repo_id"], repo_type="model", operations=operations,
                                   parent_commit=current.sha, commit_message="Publish verified full Clawd Clef research model and standalone inference evidence")
        if re.fullmatch(r"[0-9a-f]{40}", commit.oid) is None:
            raise ValueError("Hub did not return an immutable model commit")
        state.update(commit=commit.oid, url=commit.commit_url, status="committed_pending_verification")
        write_json(status_path, state)
        phase = "verification"
        verification = verify_remote(api, package, commit.oid, download)
        state.update(status="published_and_verified", verified=True, **verification)
        write_json(status_path, state)
        return state
    except Exception as exc:
        state.update(status="failed", verified=False, failed_phase=phase, error_type=type(exc).__name__)
        write_json(status_path, state)
        raise RuntimeError(f"Model publication failed during {phase}; safe status recorded in publication.json") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--inference-proof", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path("local/clef-local-publication"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--push", action="store_true", help="Explicitly create/update the private Hub model after all evidence gates pass")
    args = parser.parse_args()
    package = prepare_or_reuse(args.release, args.inference_proof, args.stage, args.repo_id)
    result = push_publication(package) if args.push else {"status": "planned", "repo_id": package["repo_id"], "private": True,
             "files": len(package["files"]), "total_bytes": package["total_bytes"], "observed_at": package["observed_at"],
             "package": str(args.stage / "package.json"), "hub_contacted": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Watch the exact approved benchmark and hand full submission to its launcher.

The watcher has no paid-job, upload, cancellation, model, or signal API calls.
It delegates one budget-guarded full submission to the unchanged launcher only
after immutable benchmark artifacts, source identity and measured cost fit pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import launch_clef_cuda_migration as launcher
from research_expansion_artifacts import sanitize_public
from train_clef_local import (DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, MODEL_ID, MODEL_REVISION)

ROOT = Path(__file__).resolve().parents[1]
STATE_NAME = "watch-state.json"
BUDGET = Decimal("50")
MAX_METADATA_BYTES = 16 * 1024 ** 2
TERMINAL = {"full_submitted", "existing_full_job", "benchmark_terminal_failure", "budget_insufficient",
            "validation_blocked", "full_submission_unresolved", "monitoring_error"}


def get(value, field, default=None):
    return value.get(field, default) if isinstance(value, dict) else getattr(value, field, default)


def document(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Saved migration state must be a JSON object")
    return value


def current_bundle_sha():
    """Compute the launcher's exact code ZIP identity without writing a bundle."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in launcher.FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (launcher.ROOT / name).read_bytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _identity(state):
    if (not isinstance(state.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", state["id"]) or
        not isinstance(state.get("owner"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", state["owner"])):
        raise ValueError("Saved job ID and owner must be explicit")
    return state["id"], state["owner"]


def _stage(job):
    value = get(get(job, "status"), "stage")
    return get(value, "value", value)


def _inspect(api, saved, phase):
    job_id, owner = _identity(saved)
    job = api.inspect_job(job_id=job_id, namespace=owner)
    if get(job, "id") != job_id or get(get(job, "owner"), "name") != owner:
        raise ValueError("Authoritative job ID or owner differs from saved migration state")
    labels = get(job, "labels", {}) or {}
    if any(labels.get(key) != value for key, value in
           {"project": launcher.PROJECT, "phase": phase, "bundle": saved["bundle_sha256"][:40]}.items()):
        raise ValueError("Authoritative job phase or source labels differ")
    if get(job, "flavor") != saved.get("flavor"):
        raise ValueError("Authoritative job hardware differs from the approved benchmark")
    return {"id": job_id, "owner": owner, "url": f"https://huggingface.co/jobs/{owner}/{job_id}", "stage": _stage(job)}


def _save(state_dir, value):
    value = sanitize_public({**value, "checked_at": datetime.now(timezone.utc).isoformat()})
    launcher.atomic_json(Path(state_dir) / STATE_NAME, value)
    return value


def _approved(state_dir):
    state_dir = Path(state_dir)
    saved = document(state_dir / "benchmark-job.json")
    ledger = document(state_dir / "budget-ledger.json")
    publication = document(state_dir / "input-publication.json")
    _identity(saved)
    if (saved.get("mode") != "benchmark" or launcher.dollars(saved.get("budget_usd")) != BUDGET or
        launcher.dollars(ledger.get("budget_usd")) != BUDGET or saved.get("max_length") != 2048 or
        saved.get("complete_train_records") != 60635 or type(saved.get("microbatch")) is not int or saved["microbatch"] < 1):
        raise ValueError("Watcher requires the original approved fifty-dollar benchmark plan")
    for digest in (saved.get("bundle_sha256"), saved.get("worker_script_sha256"), publication.get("manifest_sha256")):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Saved migration source hashes are incomplete")
    if (publication.get("status") != "private_stage_uploaded_and_hash_verified" or
        publication.get("repo_id") != saved.get("input_repo") or publication.get("revision") != saved.get("input_revision") or
        not re.fullmatch(r"[0-9a-f]{40}", publication.get("revision", ""))):
        raise ValueError("Watcher inputs differ from the verified immutable private publication")
    reservations = ledger.get("reservations", [])
    if not isinstance(reservations, list):
        raise ValueError("Cumulative GPU ledger is malformed")
    reserved = sum((launcher.dollars(item["maximum_charge_usd"]) for item in reservations
                    if item.get("status") != "rejected_without_job"), Decimal(0))
    if reserved > BUDGET or Decimal(str(ledger.get("maximum_reserved_charge_usd"))) != reserved:
        raise ValueError("Cumulative GPU ledger charge bounds are inconsistent")
    matched = [item for item in reservations if item.get("job_id") == saved["id"]]
    if len(matched) != 1 or any(matched[0].get(key) != value for key, value in
        {"mode": "benchmark", "bundle_sha256": saved["bundle_sha256"], "owner": saved["owner"],
         "output_repo": saved["output_repo"], "input_revision": saved["input_revision"], "status": "submitted"}.items()):
        raise ValueError("Saved benchmark does not match its actual submitted budget reservation")
    if launcher.dollars(matched[0]["maximum_charge_usd"]) != launcher.dollars(saved["maximum_charge_usd"]):
        raise ValueError("Benchmark reservation differs from the approved charge bound")
    if current_bundle_sha() != saved["bundle_sha256"]:
        raise ValueError("GPU source changed after the accepted qualification job")
    worker = state_dir / f"gpu-worker-{saved['bundle_sha256']}.py"
    if not worker.is_file() or worker.is_symlink() or launcher.sha256(worker) != saved["worker_script_sha256"]:
        raise ValueError("Immutable reviewed GPU worker script changed")
    return saved, ledger, publication


def _existing_full(api, state_dir, saved, ledger):
    path = Path(state_dir) / "full-job.json"
    if path.is_file():
        full = document(path)
    else:
        entries = [item for item in ledger["reservations"] if item.get("mode") == "full"]
        if not entries:
            return None
        known = [item for item in entries if item.get("status") == "submitted" and item.get("job_id")]
        if len(known) != 1:
            return {"status": "full_submission_unresolved", "action": "Inspect the existing full-phase ledger reservation; no duplicate submitted"}
        full = {**known[0], "id": known[0]["job_id"], "flavor": saved["flavor"]}
    if full.get("bundle_sha256") != saved["bundle_sha256"] or full.get("owner") != saved["owner"]:
        raise ValueError("Existing full state belongs to a different source or owner")
    return {"status": "existing_full_job", "full": _inspect(api, full, "full"), "duplicate_submitted": False}


def _fetch_evidence(api, state_dir, saved, publication):
    info = api.model_info(saved["output_repo"])
    revision = get(info, "sha")
    if get(info, "private") is not True or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Benchmark artifacts must use an immutable private output commit")
    names = ["training.json", "job-completion.json"]
    entries = {get(entry, "path"): entry for entry in api.get_paths_info(saved["output_repo"], names,
        revision=revision, repo_type="model")}
    values, files = {}, {}
    folder = Path(state_dir) / "benchmark-evidence" / revision
    for name in names:
        entry = entries.get(name)
        size = get(entry, "size")
        if type(size) is not int or not 0 < size <= MAX_METADATA_BYTES:
            raise ValueError("Benchmark completion metadata is absent or exceeds the bounded JSON size")
        local = api.hf_hub_download(saved["output_repo"], name, revision=revision, repo_type="model")
        data = Path(local).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        lfs = get(entry, "lfs")
        if len(data) != size or (lfs and (get(lfs, "size") != size or get(lfs, "sha256") != digest)) or (
            not lfs and get(entry, "blob_id") != hashlib.sha1(f"blob {size}\0".encode() + data).hexdigest()):
            raise ValueError("Pinned benchmark metadata differs from its remote Git/LFS identity")
        values[name] = json.loads(data)
        if not isinstance(values[name], dict):
            raise ValueError("Benchmark evidence must contain JSON objects")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Pinned local benchmark evidence must not be overwritten")
        if not target.exists():
            target.write_bytes(data)
        files[name] = {"sha256": digest, "bytes": size}
    completion = values["job-completion.json"]
    if any(completion.get(key) != expected for key, expected in {
        "status": "verified_artifacts_uploaded", "mode": "benchmark", "bundle_sha256": saved["bundle_sha256"],
        "input_revision": publication["revision"]}.items()):
        raise ValueError("Completed job did not attest the exact verified benchmark artifacts")
    evidence = values["training.json"]
    if (evidence.get("model") != MODEL_ID or evidence.get("model_revision") != MODEL_REVISION or
        evidence.get("dataset") != DATASET_ID or evidence.get("dataset_revision") != DATASET_REVISION or
        evidence.get("input_parquet_sha256") != INPUT_PARQUET_SHA256 or evidence.get("input_raw_row_counts") != INPUT_RAW_ROW_COUNTS or
        evidence.get("security_filter", {}).get("applied") is not True or evidence["security_filter"].get("version") != 1 or
        evidence.get("microbatch_size") != saved["microbatch"] or evidence.get("max_length") != 2048 or
        evidence.get("cohorts", {}).get("train", {}).get("selected") != 60635):
        raise ValueError("Completed benchmark source, privacy or selected training scope differs")
    minimum = launcher.benchmark_seconds(evidence, publication["manifest_sha256"], saved["bundle_sha256"])
    launcher.atomic_json(folder / "verification.json", {"repo_id": saved["output_repo"], "revision": revision,
        "files": files, "job_id": saved["id"], "bundle_sha256": saved["bundle_sha256"],
        "stage_manifest_sha256": publication["manifest_sha256"], "minimum_full_timeout_seconds": minimum})
    return evidence, revision, minimum


def _launch(command):
    # Captured child output is never copied into status or logs: exceptions can
    # contain account details. The authoritative saved ledger/job records are
    # the only accepted result of submission.
    return subprocess.run(command, capture_output=True, text=True, check=False).returncode


def run_once(api, state_dir, *, python_executable=None, launch=_launch):
    """One read-only monitoring poll and optional guarded launcher handoff."""
    state_dir = Path(state_dir)
    saved, ledger, publication = _approved(state_dir)
    existing = _existing_full(api, state_dir, saved, ledger)
    if existing:
        return _save(state_dir, existing)
    observed = _inspect(api, saved, "benchmark")
    stage = observed["stage"]
    if stage in {"SCHEDULING", "RUNNING"}:
        return _save(state_dir, {"status": "waiting_for_benchmark", "benchmark": observed,
            "optimizer_steps_observed": None, "full_submitted": False, "budget_usd": "50"})
    if stage in {"ERROR", "CANCELED", "DELETED"}:
        return _save(state_dir, {"status": "benchmark_terminal_failure", "benchmark": observed,
            "full_submitted": False, "automatic_qualification_retry": False,
            "action": "Inspect the exact failed benchmark; its reserved charge remains in the cumulative ledger"})
    if stage != "COMPLETED":
        raise ValueError("Unknown authoritative benchmark job stage")
    evidence, revision, minimum = _fetch_evidence(api, state_dir, saved, publication)
    hardware = next((item for item in api.list_jobs_hardware() if get(item, "name") == saved["flavor"]), None)
    if hardware is None:
        raise ValueError("Approved benchmark hardware is absent from current authoritative pricing")
    hourly = launcher.hourly_price(hardware)
    # Re-read ledger and source after artifact downloads before any handoff.
    latest, ledger, latest_publication = _approved(state_dir)
    if latest != saved or latest_publication != publication:
        raise ValueError("Approved benchmark state changed during the watcher poll")
    existing = _existing_full(api, state_dir, saved, ledger)
    if existing:
        return _save(state_dir, existing)
    remaining = launcher.available_seconds(ledger, BUDGET, hourly)
    plan = {"benchmark": observed, "benchmark_evidence_revision": revision,
        "optimizer_steps_observed": evidence["optimizer_steps"], "budget_usd": "50",
        "current_hourly_price_usd": str(hourly), "remaining_timeout_seconds": remaining,
        "minimum_full_timeout_seconds": minimum, "bundle_sha256": saved["bundle_sha256"]}
    if minimum > remaining or remaining < 60:
        return _save(state_dir, {**plan, "status": "budget_insufficient", "full_submitted": False,
            "action": "Measured full-run requirement exceeds the remaining approved budget; no job submitted"})
    if get(api.model_info(saved["output_repo"]), "sha") != revision:
        raise ValueError("Remote benchmark revision changed before guarded full submission")
    interpreter = str(python_executable or sys.executable)  # Preserve lexical venv path.
    command = [interpreter, str(ROOT / "scripts/launch_clef_cuda_migration.py"), "--state-dir", str(state_dir.absolute()),
        "--mode", "full", "--submit", "--budget-usd", "50", "--timeout-seconds", str(remaining),
        "--input-repo", saved["input_repo"], "--microbatch", str(saved["microbatch"]),
        "--flavor", saved["flavor"], "--jobs-namespace", saved["owner"]]
    _save(state_dir, {**plan, "status": "full_launch_pending", "full_submitted": False})
    returncode = launch(command)
    after = document(state_dir / "budget-ledger.json")
    existing = _existing_full(api, state_dir, saved, after)
    if existing and existing["status"] == "existing_full_job":
        return _save(state_dir, {**plan, "status": "full_submitted", "full_submitted": True,
            "full": existing["full"], "launcher_returncode": returncode})
    return _save(state_dir, {**plan, "status": "full_submission_unresolved", "full_submitted": False,
        "launcher_returncode": returncode, "action": "Inspect the guarded launcher ledger before retrying; watcher submitted no duplicate"})


@contextmanager
def watcher_lock(state_dir):
    import fcntl
    with (Path(state_dir) / "watcher.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another watcher owns this benchmark handoff") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "local/clef-cuda-migration")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 30:
        parser.error("Poll interval must be between five and thirty seconds")
    from huggingface_hub import HfApi
    api = HfApi()
    with watcher_lock(args.state_dir):
        while True:
            try:
                result = run_once(api, args.state_dir)
            except ValueError as error:
                result = _save(args.state_dir, {"status": "validation_blocked", "full_submitted": False,
                    "action": str(error)})
            except Exception as error:
                result = _save(args.state_dir, {"status": "monitoring_error", "full_submitted": False,
                    "error_type": type(error).__name__, "action": "Read-only monitoring failed; inspect connectivity and saved state before continuing"})
            print(json.dumps(result, allow_nan=False), flush=True)
            if args.once or result["status"] in TERMINAL:
                return 0 if result["status"] in {"full_submitted", "existing_full_job", "waiting_for_benchmark"} else 2
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

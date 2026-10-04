#!/usr/bin/env python3
"""Publish an immutable private Clef migration package and submit bounded GPU Jobs.

Preparing or uploading a package never starts paid compute. Submission requires
an explicit cumulative dollar budget; unresolved submissions retain their full
reservation so a lost response cannot cause duplicate jobs or overspending.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "clawd-clef-cuda-migration"
FILES = ("scripts/launch_clef_cuda_migration.py", "scripts/train_clef_cuda.py",
         "scripts/clef_cuda_job_resume.py",
         "scripts/stage_clef_cloud_migration.py", "scripts/train_clef_local.py",
         "scripts/clef_research_data.py", "scripts/clef_research_training.py",
         "scripts/clef_live_tape.py", "scripts/research_expansion_artifacts.py",
         "scripts/export_clef_cuda_release.py", "scripts/export_clef_mps_release.py",
         "scripts/export_clef_release.py", "scripts/run_clef_local.py",
         "data/realtime_research_citations.md")
DEPENDENCIES = ["torch==2.14.1", "transformers==5.18.0", "peft==0.21.2",
                "huggingface_hub==1.33.0", "accelerate==1.15.0", "pyarrow==25.0.1",
                "safetensors==0.8.0", "pillow==12.3.0", "torchvision==0.29.1",
                "bitsandbytes==0.50.2", "flash-linear-attention==0.5.2", "fla-core==0.5.2"]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def submission_lock(path):
    """Serialize budget reservations and API submission across launcher processes."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another launcher owns the GPU submission/budget lock") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dollars(value):
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError("An explicit positive finite GPU budget is required")
    return result


def hourly_price(hardware):
    multiplier = {"second": 3600, "minute": 60, "hour": 1}.get(hardware.unit_label)
    if multiplier is None:
        raise ValueError("Unknown hardware billing unit")
    return (dollars(hardware.unit_cost_usd) * multiplier).quantize(Decimal("0.0001"), rounding=ROUND_CEILING)


def reserve_budget(ledger, *, budget, hourly, seconds, mode, bundle_sha256, retry_of=None):
    budget, hourly = dollars(budget), dollars(hourly)
    if type(seconds) is not int or seconds < 60:
        raise ValueError("Job timeout must be at least sixty seconds")
    if ledger.get("budget_usd") is not None and dollars(ledger["budget_usd"]) != budget:
        raise ValueError("Changing the cumulative budget requires a new explicit reviewed ledger")
    reservations = ledger.setdefault("reservations", [])
    if any(item["mode"] == mode and item["status"] in {"submission_pending", "submission_outcome_unknown"}
           for item in reservations):
        raise ValueError("This job phase has an unresolved submission reservation")
    prior = [item for item in reservations if item["mode"] == mode and item["bundle_sha256"] == bundle_sha256
             and item["status"] != "rejected_without_job"]
    if retry_of and (not any(item.get("job_id") == retry_of and item["status"] == "terminal_confirmed" for item in prior)
                     or any(item.get("retry_of") == retry_of for item in prior)):
        raise ValueError("Retry requires an authoritative terminal job and one unused attempt reservation")
    if prior and not retry_of:
        raise ValueError("This job phase already has a reservation; inspect it before retrying")
    spent_bound = sum((Decimal(item["maximum_charge_usd"]) for item in reservations
                      if item["status"] != "rejected_without_job"), Decimal(0))
    billable_minutes = math.ceil(seconds / 60)
    charge = hourly * Decimal(billable_minutes) / Decimal(60)
    if spent_bound + charge > budget:
        raise ValueError("This timeout exceeds the remaining cumulative GPU budget")
    item = {"mode": mode, "bundle_sha256": bundle_sha256, "timeout_seconds": seconds,
            "maximum_billable_minutes": billable_minutes,
            "hourly_price_usd": str(hourly), "maximum_charge_usd": str(charge),
            "status": "submission_pending", "created_at": datetime.now(timezone.utc).isoformat()}
    reservations.append(item)
    if retry_of:
        item["retry_of"] = retry_of
    ledger.update(budget_usd=str(budget), maximum_reserved_charge_usd=str(spent_bound + charge))
    return item


def available_seconds(ledger, budget, hourly):
    budget, hourly = dollars(budget), dollars(hourly)
    reserved = sum((Decimal(item["maximum_charge_usd"]) for item in ledger.get("reservations", [])
                    if item["status"] != "rejected_without_job"), Decimal(0))
    minutes = ((budget - reserved) / hourly * 60).to_integral_value(rounding=ROUND_FLOOR)
    return max(0, int(minutes) * 60)


def benchmark_seconds(evidence, stage_sha, bundle_sha):
    """Require completed real CUDA training/reload before trusting its timing."""
    required = {"status": "trained_and_reload_verified", "run_mode": "benchmark",
                "backend": "cuda", "optimizer_steps": 16, "planned_steps": 16,
                "stage_manifest_sha256": stage_sha}
    if any(evidence.get(key) != value for key, value in required.items()):
        raise ValueError("Benchmark lacks the exact completed CUDA training/source contract")
    if evidence.get("job_bundle_sha256") != bundle_sha:
        raise ValueError("Benchmark used a different executable GPU package")
    for name in ("migration_verification", "reload_verification"):
        proof = evidence.get(name, {})
        difference = proof.get("max_probability_difference")
        if (proof.get("matched") is not True or type(difference) not in (int, float)
                or not math.isfinite(difference) or not 0 <= difference <= 0.001):
            raise ValueError("CUDA benchmark numerical migration/reload verification failed")
    if not all(evidence.get("gradient_evidence", {}).get(key) is True for key in ("lora", "head")):
        raise ValueError("CUDA benchmark did not verify both native gradient paths")
    capacity = evidence.get("capacity_probe", {})
    if (capacity.get("trained") is not True or capacity.get("tokens") != evidence.get("cohorts", {}).get("train", {}).get("max_tokens")
            or type(capacity.get("tokens")) is not int or capacity["tokens"] < 2044):
        raise ValueError("CUDA benchmark did not train the longest selected native input")
    kernels = evidence.get("kernel_execution", {})
    if (kernels.get("implementation") != "installed_fla" or kernels.get("fla_core_version") != "0.5.2"
            or kernels.get("hub_kernel_downloads") is not False
            or type(kernels.get("cuda_forward_calls")) is not int or kernels["cuda_forward_calls"] <= 0
            or type(kernels.get("cuda_backward_calls")) is not int or kernels["cuda_backward_calls"] <= 0):
        raise ValueError("CUDA benchmark lacks actual local FLA forward/backward execution")
    for key in ("lora", "head"):
        before = evidence.get("initial_trainable_fingerprints", {}).get(key, {})
        after = evidence.get("trained_trainable_fingerprints", {}).get(key, {})
        if not before.get("sha256") or before.get("sha256") == after.get("sha256"):
            raise ValueError("CUDA benchmark lacks real head/LoRA parameter updates")
    seconds = evidence.get("estimated_full_training_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("CUDA benchmark has no measured full-cohort throughput estimate")
    # Measured training plus 30% length/throughput margin and 90 minutes for
    # downloads, encoding, evaluation, standalone merge/reload and uploads.
    return math.ceil(seconds * 1.3 + 90 * 60)


def build_bundle(destination):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, (ROOT / name).read_bytes())
    payload = buffer.getvalue()
    package_sha = hashlib.sha256(payload).hexdigest()
    script = f'''# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = {json.dumps(DEPENDENCIES)}
# ///
import base64, io, pathlib, sys, zipfile
root = pathlib.Path("/tmp/clawd-clef-cuda-code")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(payload).decode()!r}))) as archive:
    archive.extractall(root)
sys.path.insert(0, str(root / "scripts"))
from launch_clef_cuda_migration import run_gpu_worker
run_gpu_worker({package_sha!r})
'''
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(script)
    return package_sha


def content_addressed_bundle(directory):
    """Each submitted path names immutable executable bytes for its code hash."""
    import os
    import tempfile
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="worker-building-", suffix=".py", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        package_sha = build_bundle(temporary)
        destination = directory / f"gpu-worker-{package_sha}.py"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != temporary.read_bytes():
                raise ValueError("Existing content-addressed GPU script bytes differ")
        return destination, package_sha
    finally:
        temporary.unlink()


def _require_private(api, repo):
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if api.model_info(repo).private is not True:
        raise ValueError("Migration inputs and intermediate outputs must use private repositories")


def upload_stage(api, stage, repo, publication_path):
    from stage_clef_cloud_migration import MANIFEST_NAME, verify_stage
    from huggingface_hub import hf_hub_download
    manifest = verify_stage(stage)
    allowlist = sorted(manifest["files"]) + [MANIFEST_NAME]
    _require_private(api, repo)
    existing = {item.path for item in api.list_repo_tree(repo, recursive=True)
                if hasattr(item, "size")} - {".gitattributes"}
    if existing - set(allowlist):
        raise ValueError("The private staging repository contains unlisted files")
    commit = api.upload_folder(repo_id=repo, folder_path=stage, allow_patterns=allowlist,
                              commit_message="Verified immutable Clef NF4 pilot and full CUDA cohort")
    revision = commit.oid
    remote = {item.path: item for item in api.list_repo_tree(repo, recursive=True, revision=revision)
              if hasattr(item, "size")}
    for name in allowlist:
        item = remote.get(name)
        local = Path(stage) / name
        expected = manifest["files"][name] if name != MANIFEST_NAME else {"bytes": local.stat().st_size, "sha256": sha256(local)}
        if item is None or item.size != expected["bytes"]:
            raise ValueError("Remote staging artifact size differs from its verified source")
        lfs = getattr(item, "lfs", None)
        if lfs:
            actual = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        else:
            actual = sha256(hf_hub_download(repo, name, revision=revision, token=api.token))
        if actual != expected["sha256"]:
            raise ValueError("Remote staging artifact SHA256 differs from its verified source")
    result = {"status": "private_stage_uploaded_and_hash_verified", "repo_id": repo,
              "revision": revision, "manifest_sha256": sha256(Path(stage) / MANIFEST_NAME),
              "files": len(allowlist), "bytes": sum((Path(stage) / name).stat().st_size for name in allowlist)}
    atomic_json(publication_path, result)
    return result


def upload_adapter(api, directory, repo, prefix):
    from train_clef_local import verify_adapter_manifest
    manifest = verify_adapter_manifest(directory)
    paths = sorted(manifest["files"]) + ["adapter_artifact_manifest.json"]
    commit = api.upload_folder(repo_id=repo, folder_path=directory, path_in_repo=prefix,
                               allow_patterns=paths, commit_message=f"Verified CUDA artifact {prefix}")
    return commit.oid, paths


def retain_owned_checkpoints(api, repo, checkpoints, keep=2):
    """Prune only older paths uploaded by this worker after a newer commit exists."""
    if keep < 2:
        raise ValueError("At least two durable remote checkpoints are retained")
    while len(checkpoints) > keep:
        oldest = checkpoints[0]
        prefix = oldest["path"]
        import re
        if not re.fullmatch(r"checkpoints/step-[0-9]{6}", prefix):
            raise ValueError("Refusing to prune an unrelated remote checkpoint")
        paths = []
        for name in oldest["files"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Refusing to prune an unsafe remote artifact reference")
            paths.append(prefix + "/" + name)
        api.delete_files(repo_id=repo, delete_patterns=paths,
                         commit_message="Retain two newer verified CUDA checkpoints")
        checkpoints.pop(0)
    api.upload_file(repo_id=repo, path_in_repo="checkpoint-index.json",
                    path_or_fileobj=(json.dumps({"checkpoints": checkpoints}, indent=2) + "\n").encode(),
                    commit_message="Immutable CUDA checkpoint references")


def spool_checkpoint(directory, spool_root):
    """Hold immutable checkpoint inodes while the trainer prunes its copies."""
    import os
    import shutil
    import tempfile
    from train_clef_local import verify_adapter_manifest
    directory, spool_root = Path(directory), Path(spool_root)
    if directory.is_symlink():
        raise ValueError("Checkpoint directory may not be a symlink")
    try:
        manifest = json.loads((directory / "adapter_artifact_manifest.json").read_text())
    except FileNotFoundError:
        return None
    if manifest.get("checkpoint_complete") is not True:
        return None
    spool_root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix=directory.name + "-", dir=spool_root))
    try:
        for name in sorted(manifest["files"]) + ["adapter_artifact_manifest.json"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe checkpoint artifact reference")
            source = directory / relative
            if source.is_symlink() or not source.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Checkpoint artifact escapes its immutable directory")
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        verify_adapter_manifest(snapshot)
        return snapshot
    except FileNotFoundError:
        # Pruning before all hardlinks exist is harmless. The next completed
        # checkpoint can be spooled; do not abort an otherwise healthy trainer.
        shutil.rmtree(snapshot)
        return None
    except Exception:
        shutil.rmtree(snapshot)
        raise


def run_gpu_worker(bundle_sha):
    """Runs inside a GPU Job; uploads only manifest-approved private artifacts."""
    import os
    import subprocess
    import sys
    import time
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from stage_clef_cloud_migration import MANIFEST_NAME, verify_stage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-repo", required=True)
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--mode", choices=("benchmark", "full"), required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-revision")
    args = parser.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.model_info(args.input_repo).private is not True or api.model_info(args.output_repo).private is not True:
        raise ValueError("GPU inputs and intermediate outputs must stay private")
    stage, output = Path("/tmp/clawd-clef-cuda-input"), Path("/tmp/clawd-clef-cuda-output")
    manifest_path = hf_hub_download(args.input_repo, MANIFEST_NAME, revision=args.input_revision,
                                    local_dir=stage, token=api.token)
    approved = json.loads(Path(manifest_path).read_text())["files"]
    snapshot_download(args.input_repo, revision=args.input_revision, local_dir=stage,
                      allow_patterns=sorted(approved) + [MANIFEST_NAME], token=api.token)
    verify_stage(stage)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name("train_clef_cuda.py")),
               "--stage", str(stage), "--output", str(output), "--mode", args.mode,
               "--microbatch", str(args.microbatch), "--gradient-accumulation", "1",
               "--max-length", "2048", "--use-kernels", "--checkpoint-steps", "128"]
    if args.mode == "full":
        command += ["--export-release", str(output / "release")]
    if args.resume_checkpoint:
        import re
        if (args.mode != "full" or not re.fullmatch(r"checkpoints/step-[0-9]{6}", args.resume_checkpoint)
                or not args.resume_revision or not re.fullmatch(r"[0-9a-f]{40}", args.resume_revision)):
            raise ValueError("Only an explicit immutable full-run checkpoint may be resumed")
        resumed = Path("/tmp/clawd-clef-cuda-resume")
        snapshot_download(args.output_repo, revision=args.resume_revision, local_dir=resumed,
                          allow_patterns=args.resume_checkpoint + "/*", token=api.token)
        from train_clef_local import verify_adapter_manifest
        proof = verify_adapter_manifest(resumed / args.resume_checkpoint)
        if proof.get("checkpoint_complete") is not True:
            raise ValueError("Remote CUDA resume checkpoint is incomplete")
        command += ["--resume", str(resumed / args.resume_checkpoint)]
    child = subprocess.Popen(command)
    uploaded, progress_sha, owned_remote = set(), None, []

    def publish_progress():
        nonlocal progress_sha
        progress = output / "training.json"
        if progress.is_file():
            current = sha256(progress)
            if current != progress_sha:
                # A byte snapshot avoids the trainer's next atomic replacement
                # racing the Hub hash/upload operations.
                value = json.loads(progress.read_text())
                value["job_bundle_sha256"] = bundle_sha
                api.upload_file(repo_id=args.output_repo, path_or_fileobj=(json.dumps(value, indent=2) + "\n").encode(),
                                path_in_repo="training.json", commit_message="CUDA training progress")
                progress_sha = current
        for directory in sorted((output / "checkpoints").glob("step-*")):
            if directory.name in uploaded or not (directory / "resume.json").is_file():
                continue
            manifest_path = directory / "adapter_artifact_manifest.json"
            if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("checkpoint_complete") is not True:
                continue
            snapshot = spool_checkpoint(directory, Path("/tmp/clawd-clef-cuda-upload-spool"))
            if snapshot is None:
                continue
            import shutil
            try:
                prefix = "checkpoints/" + directory.name
                revision, files = upload_adapter(api, snapshot, args.output_repo, prefix)
                owned_remote.append({"path": prefix, "revision": revision, "files": files})
                retain_owned_checkpoints(api, args.output_repo, owned_remote)
            finally:
                shutil.rmtree(snapshot)
            uploaded.add(directory.name)

    while child.poll() is None:
        time.sleep(30)
        publish_progress()
    publish_progress()
    if child.returncode != 0:
        # A standalone merge/runtime failure can follow successful native
        # training and adapter reload. Preserve that completed adapter instead
        # of losing the full epoch when this ephemeral GPU Job exits.
        from train_clef_local import verify_adapter_manifest
        adapter = output / "adapter"
        if (adapter / "adapter_artifact_manifest.json").is_file():
            verify_adapter_manifest(adapter)
            saved = json.loads((adapter / "training.json").read_text())
            if (saved.get("status") == "trained_and_reload_verified"
                    and saved.get("reload_verification", {}).get("matched") is True):
                upload_adapter(api, adapter, args.output_repo, "adapter")
                for name in ("evaluation.json", "eval-predictions.jsonl", "test-predictions.jsonl"):
                    if (output / name).is_file():
                        api.upload_file(repo_id=args.output_repo, path_or_fileobj=output / name, path_in_repo=name)
                api.upload_file(repo_id=args.output_repo, path_in_repo="job-failure.json",
                                path_or_fileobj=(json.dumps({"status": "verified_adapter_preserved_after_trainer_failure",
                                                            "trainer_exit_code": child.returncode,
                                                            "bundle_sha256": bundle_sha}) + "\n").encode())
        raise SystemExit(f"CUDA trainer exited with code {child.returncode}; saved remote checkpoints remain resumable")
    upload_adapter(api, output / "adapter", args.output_repo, "adapter")
    for name in ("evaluation.json", "eval-predictions.jsonl", "test-predictions.jsonl"):
        api.upload_file(repo_id=args.output_repo, path_or_fileobj=output / name, path_in_repo=name)
    if args.mode == "full":
        from export_clef_cuda_release import run_live_inference, verify_cuda_release, verify_live_inference
        proof = output / "live-inference.json"
        run_live_inference(output / "release", proof)
        verify_live_inference(proof, output / "release")
        release = verify_cuda_release(output / "release")
        allowlist = sorted(release["files"]) + ["release.json"]
        api.upload_folder(repo_id=args.output_repo, folder_path=output / "release", path_in_repo="release",
                          allow_patterns=allowlist, commit_message="Full-epoch CUDA standalone release with fresh reload proof")
        api.upload_file(repo_id=args.output_repo, path_or_fileobj=proof, path_in_repo="live-inference.json")
    atomic_json(output / "job-completion.json", {"status": "verified_artifacts_uploaded", "mode": args.mode,
                                                 "bundle_sha256": bundle_sha, "input_revision": args.input_revision})
    api.upload_file(repo_id=args.output_repo, path_or_fileobj=output / "job-completion.json", path_in_repo="job-completion.json")
    print(json.dumps({"status": "verified_artifacts_uploaded", "mode": args.mode, "output_repo": args.output_repo}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=ROOT / "local/clef-cloud-migration-20261002")
    parser.add_argument("--input-repo", default="solanaclawd/clef-solana-research-cuda-init-20261002")
    parser.add_argument("--output-repo")
    parser.add_argument("--mode", choices=("benchmark", "full"), default="benchmark")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--jobs-namespace", default="ordlibrary")
    parser.add_argument("--budget-usd", help="Explicit cumulative maximum charge covering benchmark, full run and retries")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--retry-job-id", help="Explicit terminal full-job ID whose exact optimizer checkpoint is continued")
    parser.add_argument("--resume-checkpoint", help="Verified checkpoints/step-NNNNNN from that terminal full job")
    parser.add_argument("--resume-revision", help="Immutable 40-character output commit containing that checkpoint")
    parser.add_argument("--upload-stage", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "local/clef-cuda-migration")
    args = parser.parse_args()
    if args.microbatch < 1:
        parser.error("microbatch must be positive")
    if args.mode == "benchmark" and args.timeout_seconds > 3600:
        parser.error("The compatibility benchmark is capped at one GPU hour")
    retry_requested = any((args.retry_job_id, args.resume_checkpoint, args.resume_revision))
    if retry_requested and (args.mode != "full" or not all((args.retry_job_id, args.resume_checkpoint, args.resume_revision))):
        parser.error("Full optimizer retry requires a previous job ID, checkpoint path, and immutable output revision together")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    destination, bundle_sha = content_addressed_bundle(args.state_dir)
    plan = {"status": "prepared_compute_not_submitted", "mode": args.mode,
            "input_repo": args.input_repo, "output_repo": args.output_repo or "solanaclawd/clef-solana-research-cuda-" + args.mode,
            "jobs_namespace": args.jobs_namespace, "flavor": args.flavor,
            "timeout_seconds": args.timeout_seconds, "budget_usd": args.budget_usd,
            "bundle_sha256": bundle_sha, "dependencies": DEPENDENCIES,
            "worker_script_sha256": sha256(destination),
            "microbatch": args.microbatch, "max_length": 2048, "complete_train_records": 60635}
    if retry_requested:
        plan.update(retry_of=args.retry_job_id, resume_checkpoint=args.resume_checkpoint, resume_revision=args.resume_revision)
    atomic_json(args.state_dir / f"{args.mode}-plan.json", plan)
    if not args.upload_stage and not args.submit:
        print(json.dumps(plan, indent=2))
        return
    from huggingface_hub import HfApi, get_token, hf_hub_download
    api = HfApi(token=get_token())
    api.whoami()
    publication_path = args.state_dir / "input-publication.json"
    if args.upload_stage:
        publication = upload_stage(api, args.stage, args.input_repo, publication_path)
        print(json.dumps(publication, indent=2), flush=True)
    if not args.submit:
        return
    if not args.budget_usd:
        parser.error("An explicit cumulative GPU dollar budget is required; no job was submitted")
    with submission_lock(args.state_dir / "submission.lock"):
        publication = json.loads(publication_path.read_text())
        from stage_clef_cloud_migration import MANIFEST_NAME
        if (publication["repo_id"] != args.input_repo or publication["manifest_sha256"] != sha256(args.stage / MANIFEST_NAME)
                or publication["status"] != "private_stage_uploaded_and_hash_verified"):
            parser.error("GPU submission requires the exact verified private migration publication")
        hardware = next(item for item in api.list_jobs_hardware() if item.name == args.flavor)
        hourly = hourly_price(hardware)
        ledger_path = args.state_dir / "budget-ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        retry = None
        if args.mode == "full":
            benchmark_job = json.loads((args.state_dir / "benchmark-job.json").read_text())
            info = api.inspect_job(job_id=benchmark_job["id"], namespace=benchmark_job["owner"])
            if info.status.stage != "COMPLETED":
                parser.error("A successfully completed CUDA compatibility benchmark is required")
            evidence = json.loads(Path(hf_hub_download(benchmark_job["output_repo"], "training.json", token=api.token)).read_text())
            minimum = benchmark_seconds(evidence, publication["manifest_sha256"], bundle_sha)
            if evidence["microbatch_size"] != args.microbatch or benchmark_job["flavor"] != args.flavor:
                parser.error("Full run must use the benchmark's measured hardware and microbatch")
            if retry_requested:
                from clef_cuda_job_resume import authorize_checkpoint_retry
                previous = next((item for item in ledger.get("reservations", [])
                                 if item.get("job_id") == args.retry_job_id), None)
                if previous is None:
                    parser.error("Retry job is absent from this cumulative budget ledger")
                retry = authorize_checkpoint_retry(api, previous_job_id=args.retry_job_id,
                    owner=args.jobs_namespace, output_repo=plan["output_repo"], output_revision=args.resume_revision,
                    checkpoint_path=args.resume_checkpoint, bundle_sha256=bundle_sha,
                    stage_manifest_sha256=publication["manifest_sha256"], full_plan=plan,
                    benchmark_evidence=evidence, previous_attempt=previous)
                previous.update(status="terminal_confirmed", authoritative_stage=retry["authoritative_previous_stage"])
                minimum = math.ceil(retry["estimated_remaining_training_seconds"] * 1.3 + 90 * 60)
                atomic_json(args.state_dir / "full-retry-verification.json", retry)
            remaining = available_seconds(ledger, args.budget_usd, hourly)
            if minimum > args.timeout_seconds or args.timeout_seconds > remaining:
                parser.error(f"Measured full-run requirement is {minimum}s; remaining budget permits {remaining}s. No full job submitted")
        labels = {"project": PROJECT, "phase": args.mode, "bundle": bundle_sha[:40]}
        phase_jobs = list(api.list_jobs(namespace=args.jobs_namespace, labels={"project": PROJECT, "phase": args.mode}))
        if any(job.status.stage not in {"ERROR", "CANCELED", "COMPLETED", "DELETED"} for job in phase_jobs):
            parser.error("This GPU phase already has an active remote job; no duplicate was submitted")
        existing = [job for job in phase_jobs if (job.labels or {}).get("bundle") == bundle_sha[:40]]
        if existing and not retry:
            print(json.dumps({"existing_jobs": [{"id": job.id, "url": job.url, "stage": job.status.stage} for job in existing]}, indent=2))
            parser.error("This exact GPU package already has a job; inspect it before retrying")
        _require_private(api, plan["output_repo"])
        reservation = reserve_budget(ledger, budget=args.budget_usd, hourly=hourly,
                                     seconds=args.timeout_seconds, mode=args.mode, bundle_sha256=bundle_sha,
                                     retry_of=args.retry_job_id if retry else None)
        atomic_json(ledger_path, ledger)
        script_args = ["--input-repo", args.input_repo, "--input-revision", publication["revision"],
                       "--output-repo", plan["output_repo"], "--mode", args.mode, "--microbatch", str(args.microbatch)]
        if retry:
            script_args += ["--resume-checkpoint", args.resume_checkpoint, "--resume-revision", args.resume_revision]
        if sha256(destination) != plan["worker_script_sha256"]:
            raise ValueError("Submitted GPU script bytes differ from the reviewed plan")
        try:
            job = api.run_uv_job(str(destination), script_args=script_args, python="3.12", flavor=args.flavor,
                                namespace=args.jobs_namespace, timeout=args.timeout_seconds,
                                name="clawd-clef-cuda-" + args.mode, labels=labels,
                                secrets={"HF_TOKEN": api.token}, env={"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status in (400, 401, 402, 403, 422):
                reservation.update(status="rejected_without_job", http_status=status)
            else:
                reservation.update(status="submission_outcome_unknown", error_type=type(error).__name__)
            atomic_json(ledger_path, ledger)
            print(json.dumps({"status": reservation["status"], "http_status": status,
                              "paid_job_confirmed": False, "budget_reservation_retained": reservation["status"] != "rejected_without_job"}))
            raise SystemExit(3) from None
        reservation.update(status="submitted", job_id=job.id, owner=job.owner.name, url=job.url,
                           output_repo=plan["output_repo"], input_revision=publication["revision"])
        atomic_json(ledger_path, ledger)
        state = {**plan, "id": job.id, "owner": job.owner.name, "url": job.url, "input_revision": publication["revision"],
                 "status_at_submission": job.status.stage, "maximum_charge_usd": reservation["maximum_charge_usd"]}
        atomic_json(args.state_dir / f"{args.mode}-job.json", state)
        print(json.dumps({"url": job.url, "stage": job.status.stage, "maximum_charge_usd": reservation["maximum_charge_usd"]}, indent=2))


if __name__ == "__main__":
    main()

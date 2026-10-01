#!/usr/bin/env python3
"""Bundle and submit the Clef research training package to Hugging Face Jobs."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from clef_research_data import MODEL_REVISION, training_source_hash

ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/clef_research_data.py", "scripts/clef_research_training.py",
         "scripts/train_clef_research.py", "scripts/clef_live_tape.py",
         "scripts/export_clef_release.py", "scripts/research_expansion_artifacts.py",
         "data/realtime_research_citations.md"]
DEPENDENCIES = ["torch==2.14.1", "transformers==5.18.0", "peft==0.21.2",
                "huggingface_hub==1.33.0", "accelerate==1.15.0", "pyarrow==25.0.1",
                "safetensors==0.8.0", "pillow==12.3.0", "torchvision==0.29.1"]


def build_bundle(destination: Path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, (ROOT / name).read_bytes())
    encoded = base64.b64encode(archive.getvalue()).decode()
    script = f'''# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = {json.dumps(DEPENDENCIES)}
# ///
import base64, io, pathlib, runpy, sys, zipfile
root = pathlib.Path("/tmp/clawd-clef-training")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as bundle:
    bundle.extractall(root)
sys.path.insert(0, str(root / "scripts"))
runpy.run_path(str(root / "scripts/train_clef_research.py"), run_name="__main__")
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(script)
    return hashlib.sha256(script.encode()).hexdigest()


def published_revision(publication: Path):
    """Accept only a complete hash-verified publication of this staged package."""
    from publish_research_expansion import validate_stage
    package, manifest = validate_stage(publication.parent)
    result = json.loads(publication.read_text())
    revision = result.get("commit", "")
    if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
        raise ValueError("Publication must record an immutable dataset commit")
    if set(result.get("verified_files", [])) != {entry["path"] for entry in package["files"]}:
        raise ValueError("Publication does not verify every staged file")
    if result.get("split_rows") != {split: item["rows"] for split, item in manifest["outputs"].items()}:
        raise ValueError("Published row counts differ from the staged dataset")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Submit a paid GPU job; requires HF_TOKEN")
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output-repo", default="solanaclawd/clef-solana-research-lora")
    parser.add_argument("--jobs-namespace", help="Optional organization to run and bill the GPU job")
    parser.add_argument("--publication", type=Path, default=ROOT / "local/research-expansion/published.json",
                        help="Hash-verified expansion publication; its immutable commit becomes the training input")
    parser.add_argument("--timeout-hours", type=int)
    parser.add_argument("--max-cost-usd", type=float, help="Hard maximum compute charge from timeout and current hardware pricing")
    args = parser.parse_args()
    timeout = args.timeout_hours or (1 if args.stage == "pilot" else 8)
    if timeout < 1:
        parser.error("timeout must be at least 1 hour")
    maximum_cost = args.max_cost_usd if args.max_cost_usd is not None else (10.0 if args.stage == "pilot" else None)
    if maximum_cost is not None and maximum_cost <= 0:
        parser.error("maximum cost must be positive")
    repo = args.output_repo + ("-pilot" if args.stage == "pilot" else "")
    destination = ROOT / "local/clef-research-job.py"
    package_hash = build_bundle(destination)
    script_args = ["--output-repo", repo, "--output", f"outputs/clef-research-{args.stage}"]
    revision = published_revision(args.publication) if args.publication.exists() else None
    if revision:
        script_args += ["--dataset-revision", revision]
    script_args += ["--live-tape"]
    if args.stage == "pilot":
        script_args += ["--max-steps", "16", "--max-train-records", "128", "--max-eval-records", "32", "--checkpoint-steps", "8"]
    else:
        script_args += ["--export-merged"]
    plan = {"stage": args.stage, "flavor": "h200", "timeout_hours": timeout,
            "output_repo": repo, "private_output": True, "script_args": script_args,
            "bundle_sha256": package_hash, "training_source_sha256": training_source_hash(),
            "dependencies": DEPENDENCIES, "dataset_revision": revision,
            "jobs_namespace": args.jobs_namespace,
            "publication": str(args.publication), "live_tape": True,
            "export_merged": args.stage == "full", "max_cost_usd": maximum_cost,
            "ready_to_submit": revision is not None,
            "blockers": [] if revision else ["Publish and verify the expanded dataset before GPU submission"]}
    (ROOT / f"local/clef-{args.stage}-job-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.submit:
        print(json.dumps(plan, indent=2))
        return
    if not revision:
        parser.error("Expanded dataset has not been published and verified; no GPU job was submitted")
    if args.stage == "full" and (args.timeout_hours is None or maximum_cost is None):
        parser.error("Full training requires explicit --timeout-hours and --max-cost-usd after reviewing pilot throughput")
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        parser.error("Submission requires HF_TOKEN or a cached login. Run .venv-connect/bin/hf auth login.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.whoami()
    state_path = ROOT / f"local/clef-{args.stage}-job.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        job = api.inspect_job(job_id=previous["id"], namespace=previous["owner"])
        print(json.dumps({"existing_job": previous["url"], "stage": job.status.stage}))
        # Inspection failure propagates; never submit a duplicate because a
        # previous process or polling request disappeared.
        raise SystemExit("A job already exists for this stage; inspect its authoritative status before retrying")
    # A server-created job can outlive a lost submission response or local file.
    existing_remote = list(api.list_jobs(namespace=args.jobs_namespace,
                                        labels={"project": "clawd-clef-research", "stage": args.stage}))
    if existing_remote:
        print(json.dumps({"existing_remote_jobs": [{"id": job.id, "url": job.url, "stage": job.status.stage}
                                                   for job in existing_remote]}, indent=2))
        parser.error("This stage already has a remote job; inspect it before retrying to avoid duplicate GPU charges")
    if args.stage == "full":
        # The full job follows a completed, reload-verified genuine Clef pilot.
        pilot_path = ROOT / "local/clef-pilot-job.json"
        if not pilot_path.exists():
            parser.error("Run and verify the GPU pilot before submitting full training")
        pilot = json.loads(pilot_path.read_text())
        info = api.inspect_job(job_id=pilot["id"], namespace=pilot["owner"])
        if info.status.stage != "COMPLETED":
            parser.error(f"Pilot is {info.status.stage}; full training has not started")
        from huggingface_hub import hf_hub_download
        evidence = json.loads(Path(hf_hub_download(pilot["output_repo"], "training.json", token=token)).read_text())
        gradients = evidence.get("gradient_evidence", {})
        if (evidence.get("status") != "trained_and_reload_verified" or
                not gradients.get("lora_nonzero") or not gradients.get("head_nonzero") or
                not all(evidence.get("weight_update_evidence", {}).get("changed", {}).get(key) for key in ("lora", "head"))):
            parser.error("Pilot has no verified training/reload evidence")
        if (evidence.get("training_source_sha256") != plan["training_source_sha256"] or
                evidence.get("model_revision") != MODEL_REVISION or
                evidence.get("dataset_revision") != revision):
            parser.error("Pilot used different training code or revisions; validate the current package before full training")
        data = json.loads(Path(hf_hub_download(pilot["output_repo"], "data-manifest.json", token=token)).read_text())
        if data["outputs"]["train"].get("live_encoded", 0) < 1:
            parser.error("Pilot did not train on actual live observations")
        seconds_per_step = evidence.get("seconds_per_optimizer_step")
        if not isinstance(seconds_per_step, (int, float)) or seconds_per_step <= 0:
            parser.error("Pilot lacks measured training throughput")
        epoch_seconds = math.ceil(data["outputs"]["train"]["rows"] / evidence["gradient_accumulation"]) * seconds_per_step
        print(json.dumps({"estimated_training_hours_from_pilot": epoch_seconds / 3600,
                          "excluded_from_estimate": "Full-cohort encoding, evaluation, model reload, 55GB export/upload; sequence lengths may differ"}, indent=2))
        if epoch_seconds > timeout * 3600:
            parser.error("Pilot throughput estimates training alone exceeds the selected timeout; choose a reviewed time/cost bound")
    hardware = next(item for item in api.list_jobs_hardware() if item.name == "h200")
    cost = dataclasses.asdict(hardware)
    multiplier = {"minute": 60, "hour": 1, "second": 3600}.get(hardware.unit_label)
    if multiplier is None:
        parser.error("Unrecognized hardware pricing unit; cannot enforce the maximum cost")
    estimated_max_cost = hardware.unit_cost_usd * multiplier * timeout
    if estimated_max_cost > maximum_cost:
        parser.error(f"Timeout permits ${estimated_max_cost:.2f} compute charges, above the ${maximum_cost:.2f} cap")
    print(json.dumps({"hardware": cost, "timeout_hours": timeout, "maximum_compute_charge_usd": estimated_max_cost}, indent=2))
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        parser.error("The output repo exists publicly; choose a new private output repository")
    from huggingface_hub.errors import HfHubHTTPError
    try:
        job = api.run_uv_job(str(destination), script_args=script_args, python="3.12",
                            namespace=args.jobs_namespace, flavor="h200", timeout=f"{timeout}h",
                            name=f"clawd-clef-research-{args.stage}", secrets={"HF_TOKEN": token},
                            env={"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"},
                            labels={"project": "clawd-clef-research", "stage": args.stage,
                                    "training-source": plan["training_source_sha256"], "dataset-revision": revision})
    except HfHubHTTPError as error:
        if error.response is None or error.response.status_code != 402:
            raise
        failure = {"stage": args.stage, "jobs_namespace": args.jobs_namespace or api.whoami()["name"],
                   "status": "not_submitted_insufficient_prepaid_credits", "http_status": 402,
                   "maximum_compute_charge_usd": estimated_max_cost, "dataset_revision": revision,
                   "message": "Hugging Face rejected submission because prepaid Jobs credits are insufficient."}
        (ROOT / f"local/clef-{args.stage}-submission-error.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2))
        raise SystemExit(3) from None
    state = {"id": job.id, "url": job.url, "owner": job.owner.name,
             "status_at_submission": job.status.stage, **plan}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"id": job.id, "url": job.url, "stage": job.status.stage}, indent=2))


if __name__ == "__main__":
    main()

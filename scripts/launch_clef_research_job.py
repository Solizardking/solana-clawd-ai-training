#!/usr/bin/env python3
"""Bundle and submit the Clef research training package to Hugging Face Jobs."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/clef_research_data.py", "scripts/clef_research_training.py",
         "scripts/train_clef_research.py", "data/realtime_research_citations.md"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Submit a paid GPU job; requires HF_TOKEN")
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output-repo", default="solanaclawd/clef-solana-research-lora")
    parser.add_argument("--timeout-hours", type=int)
    args = parser.parse_args()
    timeout = args.timeout_hours or (1 if args.stage == "pilot" else 8)
    if timeout < 1:
        parser.error("timeout must be at least 1 hour")
    repo = args.output_repo + ("-pilot" if args.stage == "pilot" else "")
    destination = ROOT / "local/clef-research-job.py"
    package_hash = build_bundle(destination)
    script_args = ["--output-repo", repo, "--output", f"outputs/clef-research-{args.stage}"]
    if args.stage == "pilot":
        script_args += ["--max-steps", "16", "--max-train-records", "128", "--max-eval-records", "32", "--checkpoint-steps", "8"]
    plan = {"stage": args.stage, "flavor": "h200", "timeout_hours": timeout,
            "output_repo": repo, "private_output": True, "script_args": script_args,
            "bundle_sha256": package_hash, "dependencies": DEPENDENCIES}
    (ROOT / f"local/clef-{args.stage}-job-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.submit:
        print(json.dumps(plan, indent=2))
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("Submission requires HF_TOKEN configured securely in the execution environment")
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
                not gradients.get("lora_nonzero") or not gradients.get("head_nonzero")):
            parser.error("Pilot has no verified training/reload evidence")
    hardware = next(item for item in api.list_jobs_hardware() if item.name == "h200")
    cost = dataclasses.asdict(hardware)
    print(json.dumps({"hardware": cost, "timeout_hours": timeout}, indent=2))
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if not api.model_info(repo).private:
        parser.error("The output repo exists publicly; choose a new private output repository")
    job = api.run_uv_job(str(destination), script_args=script_args, python="3.12",
                         flavor="h200", timeout=f"{timeout}h", name=f"clawd-clef-research-{args.stage}",
                         secrets={"HF_TOKEN": token}, env={"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"},
                         labels={"project": "clawd-clef-research", "stage": args.stage})
    state = {"id": job.id, "url": job.url, "owner": job.owner.name,
             "status_at_submission": job.status.stage, **plan}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"id": job.id, "url": job.url, "stage": job.status.stage}, indent=2))


if __name__ == "__main__":
    main()

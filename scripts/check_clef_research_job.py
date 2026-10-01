#!/usr/bin/env python3
"""Inspect the recorded GPU job and its saved evidence without starting a new job."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state_path = root / f"local/clef-{args.stage}-job.json"
    if not state_path.exists():
        parser.error("No recorded job ID; this stage has not been submitted")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("Inspection requires HF_TOKEN configured securely")
    from huggingface_hub import HfApi, hf_hub_download
    state = json.loads(state_path.read_text())
    api = HfApi(token=token)
    info = api.inspect_job(job_id=state["id"], namespace=state["owner"])
    report = {"id": info.id, "url": info.url, "status": info.status.stage,
              "output_repo": state["output_repo"]}
    print(json.dumps(report, indent=2))
    for line in api.fetch_job_logs(job_id=info.id, namespace=state["owner"], follow=False, tail=30):
        print(line, end="" if line.endswith("\n") else "\n")
    if info.status.stage == "COMPLETED":
        evidence = {}
        for filename in ("training.json", "evaluation.json", "data-manifest.json"):
            downloaded = Path(hf_hub_download(state["output_repo"], filename, token=token))
            data = json.loads(downloaded.read_text())
            (root / f"local/clef-{args.stage}-{filename}").write_text(json.dumps(data, indent=2) + "\n")
            evidence[filename] = data
        training = evidence["training.json"]
        if training.get("status") != "trained_and_reload_verified":
            raise RuntimeError("Completed job lacks verified saved training artifacts")
        if training.get("training_source_sha256") != state["training_source_sha256"]:
            raise RuntimeError("Saved model code provenance differs from the submitted job")
        if not evidence["evaluation.json"].get("reload_verification", {}).get("matched"):
            raise RuntimeError("Saved adapter reload was not verified")
        print(json.dumps({"verified": True, "optimizer_steps": training["optimizer_steps"],
                          "evaluation": evidence["evaluation.json"]}, indent=2))
    elif info.status.stage in ("ERROR", "CANCELED", "DELETED"):
        raise SystemExit(f"Job is terminal: {info.status.stage}. Review logs before retrying.")


if __name__ == "__main__":
    main()

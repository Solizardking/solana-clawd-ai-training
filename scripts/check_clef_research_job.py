#!/usr/bin/env python3
"""Inspect the recorded GPU job and its saved evidence without starting a new job."""
import argparse
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state_path = root / f"local/clef-{args.stage}-job.json"
    if not state_path.exists():
        parser.error("No recorded job ID; this stage has not been submitted")
    from huggingface_hub import get_token
    token = get_token()
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
        if not all(training.get("weight_update_evidence", {}).get("changed", {}).get(key) for key in ("lora", "head")):
            raise RuntimeError("Saved model lacks actual weight-change evidence for LoRA and native head")
        if training.get("training_source_sha256") != state["training_source_sha256"]:
            raise RuntimeError("Saved model code provenance differs from the submitted job")
        if not evidence["evaluation.json"].get("reload_verification", {}).get("matched"):
            raise RuntimeError("Saved adapter reload was not verified")
        if training.get("dataset_revision") != state["dataset_revision"] or not training.get("all_planned_steps_completed"):
            raise RuntimeError("Model did not complete the submitted dataset training plan")
        if state.get("live_tape") and evidence["data-manifest.json"]["outputs"]["train"].get("live_encoded", 0) < 1:
            raise RuntimeError("Model training omitted requested live observations")
        if state.get("export_merged"):
            saved = training.get("standalone_release", {})
            if not saved.get("reload_verified") or not saved.get("revision") or not saved.get("repo_id"):
                raise RuntimeError("Completed job has no verified standalone model publication")
            release = json.loads(Path(hf_hub_download(saved["repo_id"], "release.json", revision=saved["revision"], token=token)).read_text())
            if (release.get("status") != "trained_and_standalone_reload_verified" or
                    not release.get("reload_verification", {}).get("matched") or
                    not release["reload_verification"].get("full_backbone_loaded_from_saved_shards")):
                raise RuntimeError("Standalone model lacks genuine saved-weight reload evidence")
            info = api.model_info(saved["repo_id"], revision=saved["revision"], files_metadata=True)
            siblings = {file.rfilename: file for file in info.siblings}
            for filename, expected in release["files"].items():
                remote = siblings.get(filename)
                if remote is None or remote.size != expected["bytes"]:
                    raise RuntimeError(f"Published standalone file size differs: {filename}")
                if remote.lfs:
                    observed_hash = remote.lfs.sha256
                else:
                    path = hf_hub_download(saved["repo_id"], filename, revision=saved["revision"], token=token)
                    observed_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if observed_hash != expected["sha256"]:
                    raise RuntimeError(f"Published standalone file hash differs: {filename}")
            evidence["standalone_release"] = {**saved, "files_verified": len(release["files"]),
                                              "total_bytes": release["total_bytes"], "weight_downloaded_locally": False}
            print(json.dumps({"standalone_model_verified": evidence["standalone_release"]}, indent=2))
        print(json.dumps({"verified": True, "optimizer_steps": training["optimizer_steps"],
                          "evaluation": evidence["evaluation.json"]}, indent=2))
    elif info.status.stage in ("ERROR", "CANCELED", "DELETED"):
        raise SystemExit(f"Job is terminal: {info.status.stage}. Review logs before retrying.")


if __name__ == "__main__":
    main()

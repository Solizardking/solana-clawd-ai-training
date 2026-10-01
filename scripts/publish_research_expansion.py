#!/usr/bin/env python3
"""Validate and atomically publish a staged, split-preserving dataset expansion."""
import argparse
import hashlib
import json
from pathlib import Path

from expand_realtime_research import REPO, file_sha
from update_realtime_citations import get_hub_token


def validate_stage(stage):
    package = json.loads((stage / "package.json").read_text())
    if package["repo_id"] != REPO:
        raise ValueError("Unexpected target dataset")
    if not package["files"]:
        raise ValueError("No staged files")
    required = {"README.md", "metadata/realtime_research_expansion_manifest.json", "metadata/expansion_lineage.jsonl"}
    required |= {f"data/{split}-00000-of-00001.parquet" for split in ("train", "eval", "test")}
    paths = {entry["path"] for entry in package["files"]}
    if not required <= paths:
        raise ValueError("Staged package is incomplete")
    for entry in package["files"]:
        path = (stage / entry["path"]).resolve()
        if not path.is_relative_to(stage.resolve()) or not path.is_file():
            raise ValueError("Invalid staged path")
        if file_sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Staged file changed: {entry['path']}")
    card = (stage / "README.md").read_text()
    if "https://arxiv.org/abs/2605.12151" not in card or "https://arxiv.org/abs/2606.08232" not in card:
        raise ValueError("Requested Kamat citations are missing")
    manifest = json.loads((stage / "metadata/realtime_research_expansion_manifest.json").read_text())
    if not manifest.get("original_rows_preserved_in_original_splits"):
        raise ValueError("Missing preservation audit")
    return package, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help="Publish the validated package with HF_TOKEN or cached Hub login")
    args = parser.parse_args()
    stage = args.stage.resolve()
    package, manifest = validate_stage(stage)
    print(json.dumps({"validated": True, "files": len(package["files"]), "base_revision": package["base_revision"],
                      "splits": manifest["outputs"]}, indent=2))
    if not args.push:
        return
    token = get_hub_token()
    if not token:
        parser.error("Publishing requires Hugging Face write access. Run .venv-connect/bin/hf auth login, then retry.")
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi(token=token)
    api.whoami()
    if api.dataset_info(REPO).sha != package["base_revision"]:
        raise ValueError("The live dataset changed since staging. Rebuild against its current revision before publishing.")
    operations = [CommitOperationAdd(path_in_repo=entry["path"], path_or_fileobj=str(stage / entry["path"])) for entry in package["files"]]
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", parent_commit=package["base_revision"], operations=operations,
                               commit_message="Expand realtime research with local NeMo, NVIDIA and linked datasets; preserve splits and citations")
    print("Published:", commit.commit_url)
    # Verify each committed file by its pinned content, not by the Hub UI alone.
    verified = []
    for entry in package["files"]:
        downloaded = hf_hub_download(REPO, entry["path"], repo_type="dataset", revision=commit.oid, token=token)
        if file_sha(downloaded) != entry["sha256"]:
            raise ValueError(f"Published file did not match staged hash: {entry['path']}")
        verified.append(entry["path"])
    result = {"commit": commit.oid, "url": commit.commit_url, "verified_files": verified,
              "split_rows": {split: item["rows"] for split, item in manifest["outputs"].items()}}
    (stage / "published.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

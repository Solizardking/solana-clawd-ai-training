"""Build a provenance-bearing corpus; publish only with --publish and HF auth."""
import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

REPO = "ordlibrary/clawd-agentic-layer-whitepaper"
BASE = "https://huggingface.co/datasets/" + REPO
DATE = "2026-10-01"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(urllib.request.urlopen(
        "https://huggingface.co/api/datasets/" + REPO).read())
    revision = metadata["sha"]
    original_card = urllib.request.urlopen(BASE + "/raw/" + revision + "/README.md").read().decode()
    records, sources = [], []

    def add_source(path, data, origin, version, status):
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest = sha(data)
        sources.append(dict(path=path, sha256=digest, bytes=len(data), origin=origin,
                            version=version, status=status))
        return digest

    def add_chunks(text, path, digest, version, status, pages=False):
        chunks = text.split("\f") if pages else re.split(r"(?m)(?=^## )", text)
        for index, chunk in enumerate(chunks, 1):
            chunk = chunk.strip()
            if not chunk:
                continue
            if re.search(r"hf_[A-Za-z0-9]{20,}|-----BEGIN .*PRIVATE KEY-----|sk-[A-Za-z0-9]{20,}", chunk):
                raise ValueError("Potential credential in " + path)
            records.append(dict(id=sha((path + ':' + str(index) + ':' + chunk).encode())[:24],
                                text=chunk, source_path=path, source_sha256=digest,
                                source_version=version, source_status=status,
                                page=index if pages else None, section=index,
                                snapshot_date=DATE, evidence="source_document; not independent live verification"))

    hub_pdf = urllib.request.urlopen(BASE + "/resolve/" + revision + "/clawd-agentic-layer-whitepaper.pdf").read()
    for path, data, origin, version in [
        ("clawd-agentic-layer-whitepaper.pdf", hub_pdf, BASE + "/resolve/" + revision + "/clawd-agentic-layer-whitepaper.pdf", "v0.3"),
        ("sources/clawd-agentic-layer-whitepaper-v0.2.pdf",
         (args.source / "whitepaper-art/clawd-agentic-layer-whitepaper.pdf").read_bytes(),
         "musebook-private/whitepaper-art/clawd-agentic-layer-whitepaper.pdf", "v0.2"),
    ]:
        digest = add_source(path, data, origin, version, "design_proposal")
        result = subprocess.run(["pdftotext", "-layout", str(out / path), "-"],
                                capture_output=True, text=True, check=True).stdout
        if "Design proposal " + version not in result:
            raise ValueError("PDF version does not match " + version)
        text_path = "texts/whitepaper-" + version + ".txt"
        add_source(text_path, result.encode(), path, version, "extracted_design_proposal")
        add_chunks(result, path, digest, version, "design_proposal", pages=True)

    for name in ["sdk/README.md", "cli/docs/INTRODUCTION.md", "cli/docs/QUICKSTART.md",
                 "cli/docs/FEATURES.md", "cli/docs/SITE-MAP.md", "cli/docs/SKILLS-CONNECTORS.md"]:
        data = (args.source / name).read_bytes()
        path = "sources/musebook/" + name
        digest = add_source(path, data, "musebook-private/" + name, "checkout_snapshot", "implementation_documentation_unverified")
        add_chunks(data.decode(), path, digest, "checkout_snapshot", "implementation_documentation_unverified")

    data_path = out / "data/corpus.jsonl"
    data_path.parent.mkdir(exist_ok=True)
    data_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    manifest = dict(snapshot_date=DATE, hub_base_revision=revision,
                    source_checkout_commit=subprocess.check_output(
                        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
                    source_checkout_note="Selected working-tree files; individual hashes are authoritative.",
                    rows=len(records), corpus_sha256=sha(data_path.read_bytes()), sources=sources)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    if original_card.startswith("---"):
        raise ValueError("Existing card now contains YAML; merge configuration before publishing")
    config = """---
language:
  - en
tags:
  - solana
  - agents
  - whitepaper
  - retrieval
configs:
  - config_name: corpus
    data_files:
      - split: train
        path: data/corpus.jsonl
---

"""
    supplement = f"""
## Source corpus update — {DATE}

The v0.3 PDF and its original description above are preserved. This update adds
{len(records)} source excerpts for retrieval and research, extracted from the existing
v0.3 PDF, the supplied Musebook v0.2 PDF, and selected public-facing SDK/CLI docs.
The `train` split is a document corpus, not instruction tuning pairs or an evaluation set.

Load with `load_dataset("{REPO}", "corpus", split="train")`.

Each row includes text, source path/hash/version/status, page (PDF only), section,
snapshot date, and evidence qualification. `manifest.json` records file hashes,
the original Hub revision, and local checkout commit. `texts/` provides searchable
PDF extractions; `sources/` preserves the v0.2 PDF and selected documentation.

### Interpretation and version conflicts

The supplied local PDFs are v0.2; they do not supersede the Hub's v0.3.
Source docs contain dated counts, versions, deployment claims, and differing
descriptions. Treat those as attributed snapshots rather than current measurements.
SDK documentation distinguishes software registration, downloadable packages,
owner-co-signed on-chain minting, and Town challenges; these are separate flows.
Registration and discovery do not grant signing authority. Preserve Brain/Hands
separation and explicit owner authorization for consequential wallet actions.
No new protocol version, audit, funded trade, or production deployment is claimed.
No licensing grant is inferred from inclusion; consult the respective rights holders.
Private environment files, keys, build caches, operational handoffs, and unrelated
checkout data are excluded.
"""
    (out / "README.md").write_text(config + original_card.rstrip() + '\n' + supplement)
    ids = [r['id'] for r in records]
    assert len(ids) == len(set(ids))
    assert all(r['text'] and (out / r['source_path']).is_file() for r in records)
    print(json.dumps(dict(output=str(out.resolve()), rows=len(records), base_revision=revision)))
    if args.publish:
        from huggingface_hub import HfApi, CommitOperationAdd, get_token
        token = get_token()
        if not token:
            raise RuntimeError("HF_TOKEN or local hf auth login is required; bundle is ready")
        api = HfApi(token=token)
        api.whoami()
        operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(out)), path_or_fileobj=str(p))
                      for p in sorted(out.rglob('*')) if p.is_file()]
        commit = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations,
                                   parent_commit=revision,
                                   commit_message="Add versioned whitepaper corpus and Musebook source documentation")
        info = api.dataset_info(REPO, revision=commit.oid, files_metadata=True)
        assert info.sha == commit.oid
        print("Published:", commit.commit_url)


if __name__ == "__main__":
    main()

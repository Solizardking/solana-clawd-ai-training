#!/usr/bin/env python3
"""Append genuine local/linked conversations to the current research dataset.

Existing rows and splits are preserved. Retrieval chunks, preferences, CPT text,
evaluation cases, and historical reports retain their distinct auxiliary roles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPO = "solanaclawd/solana-clawd-realtime-research-instruct"
DONORS = {
    "processed": "solanaclawd/solana-clawd-instruct",
    "core_ai": "solanaclawd/solana-clawd-core-ai-instruct",
    "repo_corpus": "solanaclawd/solana-clawd-repo-corpus",
    "trading_factory": "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
    "tx_foundation_cpt": "solanaclawd/solana-tx-foundation-cpt",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message_keys(messages):
    normalized = [{"role": message["role"], "content": " ".join(message["content"].split())} for message in messages]
    prompt = [message for message in normalized if message["role"] != "assistant"]
    answer = [message["content"] for message in normalized if message["role"] == "assistant"]
    return sha(normalized), sha(prompt), sha(answer)


def valid_messages(messages):
    return (isinstance(messages, list) and len(messages) >= 2 and
            all(isinstance(message, dict) and message.get("role") in ("system", "user", "assistant", "tool") and
                isinstance(message.get("content"), str) and message["content"].strip() for message in messages) and
            messages[-1]["role"] == "assistant" and any(message["role"] == "user" for message in messages))


def merge_candidates(base, candidates):
    """Do not create new exact-content or prompt/answer leakage across splits."""
    merged = {split: list(rows) for split, rows in base.items()}
    seen, prompts, answers = {}, defaultdict(set), defaultdict(set)
    for split, rows in base.items():
        for row in rows:
            full, prompt, answer = message_keys(row["messages"])
            seen.setdefault(full, split)
            prompts[prompt].add(split)
            answers[answer].add(split)
    counts, accepted = Counter(), []
    for split in ("test", "eval", "train"):
        for candidate in candidates[split]:
            row = candidate["row"]
            full, prompt, answer = message_keys(row["messages"])
            donor = candidate["donor"]
            if full in seen:
                counts[f"{donor}:duplicate"] += 1
                continue
            other = (prompts[prompt] | answers[answer]) - {split}
            # Train conflicts with either holdout; holdouts conflict with train.
            if (split == "train" and other & {"eval", "test"}) or (split != "train" and "train" in other):
                counts[f"{donor}:cross_split_prompt_or_answer"] += 1
                continue
            if split == "eval" and "test" in other:
                counts[f"{donor}:cross_split_prompt_or_answer"] += 1
                continue
            merged[split].append(row)
            seen[full] = split
            prompts[prompt].add(split)
            answers[answer].add(split)
            counts[f"{donor}:added:{split}"] += 1
            accepted.append({"split": split, "example_sha256": row["example_sha256"],
                             "messages_sha256": full, **candidate["provenance"]})
    return merged, dict(counts), accepted


def normalize_candidate(row, donor, source_file, source_hash, provenance):
    from research_expansion_artifacts import sanitize_public, secret_like
    messages = row.get("messages")
    if not valid_messages(messages):
        return None, "invalid_conversation"
    if secret_like(canonical(row)):
        return None, "secret_like"
    public = sanitize_public(row)
    messages = [{"role": message["role"], "content": message["content"]} for message in public["messages"]]
    raw_metadata = public.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        raw_metadata = {"value": raw_metadata}
    # Keep the current public Parquet schema. Full donor metadata lives in the
    # separately joined lineage table instead of being discarded or coerced.
    metadata = {"title": raw_metadata.get("title") if isinstance(raw_metadata.get("title"), str) else None,
                "cell_type": raw_metadata.get("cell_type") if isinstance(raw_metadata.get("cell_type"), str) else None}
    metadata.update({key: raw_metadata.get(key) if isinstance(raw_metadata.get(key), int) else None for key in ("page", "chunk", "cell")})
    tags = public.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    converted = {"messages": messages,
                 "source": str(public.get("source") or source_file),
                 "source_type": str(public.get("source_type") or "messages"),
                 "source_sha256": str(public.get("source_sha256") or source_hash),
                 "record_id": str(public.get("record_id") or public.get("id") or sha(messages)[:20]),
                 "example_sha256": sha(messages),
                 "tags": list(dict.fromkeys([str(tag) for tag in tags] + ["local-data-expansion", donor])),
                 "metadata": metadata}
    lineage = {"donor": donor, "source_file": source_file, "source_file_sha256": source_hash,
               "original_record_metadata": {key: value for key, value in public.items() if key != "messages"},
               **provenance}
    return {"row": converted, "donor": donor, "provenance": sanitize_public(lineage)}, None


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def refresh_package(output):
    manifest = json.loads((output / "metadata/realtime_research_expansion_manifest.json").read_text())
    excluded = {"package.json", "published.json"}
    package = {"repo_id": REPO, "base_revision": manifest["base_revision"], "files": [
        {"path": str(path.relative_to(output)), "sha256": file_sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file() and str(path.relative_to(output)) not in excluded
    ]}
    (output / "package.json").write_text(json.dumps(package, indent=2) + "\n")


def update_canonical_manifest(output):
    """Keep the conventional manifest aligned while retaining original lineage."""
    from huggingface_hub import hf_hub_download
    from research_expansion_artifacts import sanitize_public
    expansion = json.loads((output / "metadata/realtime_research_expansion_manifest.json").read_text())
    path = hf_hub_download(REPO, "metadata/realtime_research_dataset_manifest.json", repo_type="dataset",
                           revision=expansion["base_revision"])
    manifest = json.loads(Path(path).read_text())
    manifest["generated_at"] = expansion["generated_at"]
    manifest["builder"] = "scripts/expand_realtime_research.py"
    manifest["base_revision"] = expansion["base_revision"]
    manifest["splits"] = {split: values["rows"] for split, values in expansion["outputs"].items()}
    manifest["counts"]["examples"] = sum(manifest["splits"].values())
    manifest["counts"]["duplicate_examples"] += sum(value for key, value in expansion["merge_audit"].items() if key.endswith(":duplicate"))
    manifest["counts"]["secret_or_invalid_skipped"] += sum(value for key, value in expansion["input_audit"].items() if key.endswith(":secret_like") or key.endswith(":invalid_conversation"))
    for source in expansion["sources"]:
        donor = source.get("name", "nemo_sft")
        count = sum(value for key, value in expansion["merge_audit"].items() if key.startswith(donor + ":added:"))
        if count:
            manifest["sources"].append({"source_id": source.get("repo_id", source.get("source")),
                                         "source_type": "linked_dataset" if source.get("repo_id") else "local_sft",
                                         "sha256": sha(source), "records": count, "skipped": 0,
                                         "metadata": source})
    manifest["counts"]["sources"] = len(manifest["sources"])
    by_source = manifest["counts"].setdefault("by_source_type", {})
    by_source["expansion_conversations"] = sum(values["added_rows"] for values in expansion["outputs"].values())
    manifest["expansion"] = {"manifest": "metadata/realtime_research_expansion_manifest.json",
                              "lineage": "metadata/expansion_lineage.jsonl", "auxiliary": expansion["auxiliary"]}
    manifest["output"] = {"jsonl": "raw/realtime_research_sft.jsonl", "parquet": {split: values["file"] for split, values in expansion["outputs"].items()}}
    manifest["dataset_sha256"] = file_sha(output / "raw/realtime_research_sft.jsonl")
    (output / "metadata/realtime_research_dataset_manifest.json").write_text(json.dumps(sanitize_public(manifest), indent=2) + "\n")
    refresh_package(output)


def build(args):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    import pyarrow as pa
    from research_expansion_artifacts import sanitize_public, stage_supporting_artifacts

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous builds")
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.dataset_info(REPO).sha
    base, base_files = {}, {}
    schema = None
    for split in ("train", "eval", "test"):
        remote = f"data/{split}-00000-of-00001.parquet"
        path = Path(hf_hub_download(REPO, remote, repo_type="dataset", revision=revision))
        table = pq.read_table(path)
        schema = table.schema if schema is None else schema
        base[split] = table.to_pylist()
        base_files[remote] = file_sha(path)
    card_path = hf_hub_download(REPO, "README.md", repo_type="dataset", revision=revision)
    base_card = Path(card_path).read_text()
    candidates = {split: [] for split in base}
    input_audit = Counter()
    sources = []

    # The metadata SFT version adds IDs/citations; exact message matching recovers
    # the actual existing 72/4/4 NeMo split rather than reshuffling the JSONL.
    nemo_root = args.data_root / "nemo_clawd"
    metadata_path = nemo_root / "sft/chat_finetune_with_metadata.jsonl"
    if metadata_path.exists():
        metadata_rows = {sha(row["messages"]): row for row in read_jsonl(metadata_path)}
        for split in base:
            source = nemo_root / f"processed/{split}.parquet"
            if not source.exists():
                raise ValueError(f"Missing NeMo {split} split needed to retain its membership")
            source_hash = file_sha(source)
            for row in pq.read_table(source).to_pylist():
                original = metadata_rows[sha(row["messages"])]
                candidate, error = normalize_candidate(original, "nemo_sft", str(source.relative_to(args.data_root)),
                                                       source_hash, {"original_split": split})
                input_audit[f"nemo_sft:{error or 'usable'}"] += 1
                if candidate:
                    candidates[split].append(candidate)
        sources.append({"kind": "local_sft", "source": "nemo_clawd/processed", "rows": len(metadata_rows)})

    if args.restore_linked:
        for donor, repo in DONORS.items():
            info = api.dataset_info(repo, files_metadata=True)
            files = [file.rfilename for file in info.siblings]
            donor_source = {"kind": "linked_published_dataset", "name": donor, "repo_id": repo, "revision": info.sha,
                            "files": []}
            for split in base:
                canonical_file = f"data/{split}-00000-of-00001.parquet"
                if canonical_file not in files:
                    continue
                path = Path(hf_hub_download(repo, canonical_file, repo_type="dataset", revision=info.sha))
                table = pq.read_table(path)
                source_hash = file_sha(path)
                donor_source["files"].append({"file": canonical_file, "rows": table.num_rows, "sha256": source_hash})
                if donor == "tx_foundation_cpt":
                    auxiliary = output / "auxiliary/tx_foundation_cpt" / f"{split}.jsonl"
                    write_jsonl(auxiliary, (sanitize_public(row) for row in table.to_pylist()))
                    continue
                for row in table.to_pylist():
                    candidate, error = normalize_candidate(row, donor, f"{repo}/{canonical_file}", source_hash,
                                                           {"repo_id": repo, "revision": info.sha, "original_split": split})
                    input_audit[f"{donor}:{error or 'usable'}"] += 1
                    if candidate:
                        candidates[split].append(candidate)
            sources.append(donor_source)
    else:
        # Restore only actual local processed shards if they are present.
        for split in base:
            local = args.data_root / f"processed/{split}.parquet"
            if local.exists():
                source_hash = file_sha(local)
                for row in pq.read_table(local).to_pylist():
                    candidate, error = normalize_candidate(row, "processed", str(local.relative_to(args.data_root)),
                                                           source_hash, {"original_split": split})
                    input_audit[f"processed:{error or 'usable'}"] += 1
                    if candidate:
                        candidates[split].append(candidate)

    merged, merge_audit, lineage = merge_candidates(base, candidates)
    outputs = {}
    for split, rows in merged.items():
        original = base[split]
        if any(canonical(before) != canonical(after) for before, after in zip(original, rows[:len(original)], strict=True)):
            raise ValueError("An original row was changed or moved")
        destination = output / f"data/{split}-00000-of-00001.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination, compression="zstd")
        reread = pq.read_table(destination).to_pylist()
        if any(canonical(before) != canonical(after) for before, after in zip(original, reread[:len(original)], strict=True)):
            raise ValueError("Parquet serialization changed an original row")
        outputs[split] = {"base_rows": len(original), "added_rows": len(rows) - len(original), "rows": len(rows),
                          "file": str(destination.relative_to(output)), "sha256": file_sha(destination)}
    write_jsonl(output / "metadata/expansion_lineage.jsonl", lineage)
    write_jsonl(output / "raw/realtime_research_sft.jsonl", (row for split in ("train", "eval", "test") for row in merged[split]))

    auxiliary_inputs = {
        "nemo_clawd/corpus/all_chunks.jsonl": "auxiliary/nemo_corpus/chunks.jsonl",
        "nemo_clawd/preference/risk_preferences.jsonl": "auxiliary/nemo_preferences/risk_preferences.jsonl",
        "nemo_clawd/eval/source_grounded_eval.jsonl": "auxiliary/nemo_evaluation/source_grounded_eval.jsonl",
        "nvidia_rag_store/chunks.jsonl": "auxiliary/nvidia_rag/chunks.jsonl",
    }
    auxiliary = []
    for source_name, target in auxiliary_inputs.items():
        source = args.data_root / source_name
        if source.exists():
            rows = read_jsonl(source)
            destination = output / target
            write_jsonl(destination, (sanitize_public(row) for row in rows))
            auxiliary.append({"source": source_name, "file": target, "rows": len(rows), "sha256": file_sha(destination)})
    for path in sorted((output / "auxiliary/tx_foundation_cpt").glob("*.jsonl")):
        auxiliary.append({"source": DONORS["tx_foundation_cpt"], "file": str(path.relative_to(output)),
                          "rows": sum(1 for _ in path.open()), "sha256": file_sha(path)})
    documentation = stage_supporting_artifacts(args.data_root, output)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "repo_id": REPO, "base_revision": revision,
        "builder": "scripts/expand_realtime_research.py", "restore_linked": args.restore_linked,
        "original_rows_preserved_in_original_splits": True, "base_files": base_files,
        "sources": sources, "input_audit": dict(input_audit), "merge_audit": merge_audit,
        "outputs": outputs, "auxiliary": auxiliary, "documentation": documentation,
        "limitations": ["Existing source documents may span train/eval/test; exact prompt and answer checks do not establish unseen-source generalization.",
                        "NeMo source-grounded evaluation includes same-source recall checks; guardrail answers occur in SFT responses.",
                        "Preference rejections, retrieval chunks and CPT text are auxiliary data, not supervised chat targets.",
                        "Historical cards/manifests/reports document earlier runs; their counts and pass flags are not newly imported examples or fresh accuracy/trading evidence.",
                        "Source-specific licensing notices remain attached; the original card's license does not replace underlying-source terms."],
        "citations": ["https://arxiv.org/abs/2605.12151", "https://arxiv.org/abs/2606.08232"],
    }
    manifest = sanitize_public(manifest)
    (output / "metadata/realtime_research_expansion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Retain the current card and its custom/source/citation sections. Update its
    # headline totals and add a bounded expansion report, not a misleading rebuild.
    total = sum(values["rows"] for values in outputs.values())
    card = re.sub(r"- Total examples: \d+", f"- Total examples: {total}", base_card, count=1)
    card = re.sub(r"- Train/eval/test: \d+ / \d+ / \d+",
                  f"- Train/eval/test: {outputs['train']['rows']} / {outputs['eval']['rows']} / {outputs['test']['rows']}", card, count=1)
    for label in ("Sources", "Duplicate examples removed", "Duplicate files skipped", "Secret-like records skipped"):
        card = card.replace(f"- {label}:", f"- Original ingestion {label.lower()}:", 1)
    # Explicit configuration prevents auxiliary JSONL schemas from being
    # auto-discovered as ordinary train/eval/test chat files by HF Datasets.
    config_block = "configs:\n  - config_name: default\n    data_files:\n" + "".join(
        f"      - split: {split}\n        path: data/{split}-*.parquet\n" for split in ("train", "eval", "test"))
    if card.startswith("---\n") and "\nconfigs:" not in card.split("---", 2)[1]:
        card = "---\n" + config_block + card[4:]
    if "## Local Data Expansion" in card:
        begin = card.index("## Local Data Expansion")
        end = card.find("\n## ", begin + 3)
        card = card[:begin] + (card[end:] if end >= 0 else "")
    section = "\n\n## Local Data Expansion\n\n"
    section += f"Built on existing Hub revision `{revision}`. All {sum(len(rows) for rows in base.values()):,} existing rows retain their split and content.\n\n"
    section += "| Split | Original rows | Added rows | Total rows |\n| --- | ---: | ---: | ---: |\n"
    section += "\n".join(f"| {split} | {item['base_rows']} | {item['added_rows']} | {item['rows']} |" for split, item in outputs.items())
    section += "\n\nGenuine conversations are imported from NeMo and the available source datasets linked by the provided manifests. Exact duplicates and additions that conflict with an existing held-out prompt or answer are skipped.\n\n"
    section += "Retrieval corpus, NVIDIA RAG chunks, preference pairs, dedicated evaluation cases, and transaction CPT text remain in `auxiliary/` with their own schemas. They are not counted as chat examples. Historical cards, manifests, source notes, and reports are in `metadata/local_sources/`. Binary search indexes and missing processed shards are not counted as new training data.\n\n"
    section += "See [the expansion manifest](metadata/realtime_research_expansion_manifest.json) for exact source revisions, counts, file hashes, exclusions, documentation inventory, and limitations, and `metadata/expansion_lineage.jsonl` for per-example provenance.\n\n"
    section += "### Expansion limitations and source licensing\n\n" + "\n".join(f"- {item}" for item in manifest["limitations"]) + "\n\n"
    section += "Reproduce with `python scripts/expand_realtime_research.py" + (" --restore-linked" if args.restore_linked else "") + "`. Publish the validated staged package with `python scripts/publish_research_expansion.py --stage PATH --push`.\n"
    card = sanitize_public(card.rstrip() + section)
    citations = (ROOT / "data/realtime_research_citations.md").read_text().strip()
    if "## Research Citations" not in card:
        card += "\n\n" + citations + "\n"
    (output / "README.md").write_text(card)
    update_canonical_manifest(output)
    print(json.dumps({"stage": str(output), "base_revision": revision, "outputs": outputs,
                      "auxiliary": auxiliary, "merge_audit": merge_audit}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "local/research-expansion")
    parser.add_argument("--restore-linked", action="store_true", help="Restore missing payloads from the published repos named in local manifests")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    build(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a provenance-tracked text/vision training package from the requested sources."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 0, "validation": 1, "test": 2}
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|[?&]api[_-]?key=(?!REDACTED)[A-Za-z0-9_-]{16,})")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def split_for(value):
    n = int(digest(value)[:8], 16) % 100
    return "test" if n < 5 else "validation" if n < 10 else "train"


def named_split(path):
    words = re.split(r"[/_.-]", str(path).lower())
    if "test" in words:
        return "test"
    if any(w in words for w in ("eval", "validation", "valid")):
        return "validation"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--bucket", type=Path, default=ROOT / "local")
    parser.add_argument("--papers", type=Path, default=Path.home() / "Downloads/arvix")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chart-foundation-data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, docs, inventory, rejected = {}, {}, [], Counter()
    groups = {}

    def emit(messages, source, split=None, image=None, group=None):
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            rejected["not_supervised_chat"] += 1
            return
        clean = []
        for m in messages:
            if m.get("role") not in {"system", "user", "assistant", "tool"} or not isinstance(m.get("content"), str):
                rejected["unsupported_message"] += 1
                return
            clean.append({"role": m["role"], "content": m["content"]})
        if not any(m["role"] == "user" for m in clean):
            rejected["no_user"] += 1
            return
        canonical = json.dumps([[m["role"], " ".join(m["content"].split())] for m in clean], ensure_ascii=False)
        if SECRET.search(canonical):
            rejected["secret_pattern"] += 1
            return
        key = digest(canonical + str(image or ""))
        group = group or key
        split = split or split_for(group)
        groups[group] = max(groups.get(group, "train"), split, key=SPLITS.get)
        if key in records:
            rejected["duplicate_chat"] += 1
            records[key]["split"] = max(records[key]["split"], split, key=SPLITS.get)
            records[key]["sources"].append(source)
            return
        records[key] = {"id": key, "messages": clean, "images": [image] if image else [],
                        "sources": [source], "split": split, "group": group}

    def document(text, source, split=None):
        if not isinstance(text, str) or len(text.strip()) < 80:
            return
        if SECRET.search(text):
            rejected["secret_document"] += 1
            return
        key = digest(" ".join(text.split()))
        split = split or split_for(key)
        if key in docs:
            docs[key]["split"] = max(docs[key]["split"], split, key=SPLITS.get)
            return
        docs[key] = {"text": text, "source": source, "split": split, "group": key}

    for path in sorted(args.data.rglob("*")):
        if not path.is_file():
            continue
        source = str(path.relative_to(ROOT))
        info = {"path": source, "bytes": path.stat().st_size, "use": "inventory_only"}
        split = named_split(path.relative_to(args.data))
        if path.suffix == ".parquet":
            f = pq.ParquetFile(path)
            info.update(use="supervised_rows", rows=f.metadata.num_rows)
            for batch in f.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    emit(row.get("messages"), source, split)
        elif path.suffix == ".jsonl":
            info["use"] = "classified_records"
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    rejected["invalid_json"] += 1
                    continue
                if not isinstance(row, dict):
                    continue
                if "messages" in row:
                    emit(row["messages"], source, split)
                elif "question" in row and "expected_answer" in row:
                    emit([{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["expected_answer"]}], source, "validation")
                elif isinstance(row.get("content", row.get("text")), str):
                    document(row.get("content", row.get("text")), source, split or (row.get("split") if row.get("split") in SPLITS else None))
                elif path.name == "clawd_ws_frames.jsonl":
                    document(json.dumps(row), source, split)
                else:
                    rejected["non_sft_record"] += 1
        elif path.suffix == ".md":
            info["use"] = "reference_only"  # Dataset cards describe data; they are not examples.
        inventory.append(info)

    for path in sorted(args.bucket.joinpath("data").glob("*.parquet")):
        f = pq.ParquetFile(path)
        source = f"hf://buckets/ordlibrary/charts/data/{path.name}"
        inventory.append({"path": source, "rows": f.metadata.num_rows, "use": "bucket_records"})
        for batch in f.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if "messages" in row:
                    trajectory = row.get("trajectory") or {}
                    emit(row["messages"], source, named_split(path.name), group=trajectory.get("near_dup_of") or trajectory.get("id"))
                else:
                    document(json.dumps(row, default=str), source, named_split(path.name))

    with (args.bucket / "metadata.csv").open(newline="") as file:
        for line, row in enumerate(csv.reader(file), 1):
            if len(row) != 11 or not row[0].startswith("images/"):
                rejected["invalid_image_metadata"] += 1
                continue
            path = (args.bucket / row[0]).resolve()
            if not path.is_relative_to(args.bucket.resolve()) or not path.is_file():
                rejected["missing_image"] += 1
                continue
            original = {"train": "train", "test": "test", "eval": "validation", "val": "validation", "validation": "validation"}.get(row[9])
            emit([{"role": "user", "content": row[3]}, {"role": "assistant", "content": row[4]}],
                 f"hf://buckets/ordlibrary/charts/metadata.csv:{line}", original, row[0], row[0])

    for path in sorted(args.papers.glob("*.pdf")):
        result = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True, timeout=60)
        document(result.stdout, str(path))
        inventory.append({"path": str(path), "bytes": path.stat().st_size, "use": "document_adaptation"})

    sample = ROOT / "outputs/chart-research/solarchive-tokens-2020-10.parquet"
    if not sample.is_file():
        raise FileNotFoundError("Run scripts/chart_sources.py first")
    for row in pq.read_table(sample).to_pylist():
        document(json.dumps(row, default=str), "solarchive/solarchive tokens/2020-10; CC-BY-4.0; Data from SolArchive.org")
    inventory.append({"path": str(sample), "use": "historical_token_document_adaptation", "rows": pq.read_table(sample).num_rows})

    counts = Counter()
    for split in SPLITS:
        with (args.output / f"{split}.jsonl").open("w") as out:
            for row in records.values():
                row["split"] = max(row["split"], groups[row["group"]], key=SPLITS.get)
                if row["split"] != split:
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"sft/{split}"] += 1
                for image in row["images"]:
                    dest = args.output / image
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copyfile(args.bucket / image, dest)
        with (args.output / f"documents-{split}.jsonl").open("w") as out:
            for row in docs.values():
                if row["split"] != split:
                    continue
                # Chunks from a document always stay in its split.
                for offset in range(0, len(row["text"]), 4000):
                    chunk = row["text"][offset:offset+4000]
                    if len(chunk.strip()) < 80:
                        continue
                    out.write(json.dumps({**row, "text": chunk}) + "\n")
                    counts[f"documents/{split}"] += 1
    manifest = {"counts": dict(counts), "rejected": dict(rejected), "inventory": inventory,
                "training_completed": False, "base_model": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU",
                "limitations": ["Solarchive uses a verified 33-row historical sample, not the full archive.",
                                "Document exact-duplicate checks do not establish semantic near-duplicate independence.",
                                "JSON manifests, dataset cards, preference pairs and FAISS indexes are inventoried rather than treated as supervised labels."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "rejected": dict(rejected)}, indent=2))


if __name__ == "__main__":
    main()

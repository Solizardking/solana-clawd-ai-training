#!/usr/bin/env python3
"""Audit prepared research decisions using the real Clef processor, without weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from clef_research_data import MODEL_ID, MODEL_REVISION, read_jsonl
from clef_research_training import encode_rows, load_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("local/clef-tokenizer-preflight.json"))
    args = parser.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    module = load_upstream()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    report = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "max_length": args.max_length,
              "model_weights_downloaded": False, "splits": {}}
    seen = set()
    for split in ("test", "eval", "train"):
        path = args.data_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["outputs"][split]["sha256"]:
            raise ValueError(f"{split} file hash does not match its manifest")
        rows = read_jsonl(path)
        hashes = {row["provenance"]["prompt_hash"] for row in rows}
        if len(hashes) != len(rows) or hashes & seen:
            raise ValueError("Duplicate prompts within or across prepared splits")
        seen.update(hashes)
        encoded, kept, excluded = encode_rows(module, processor, rows, args.max_length)
        if not kept:
            raise ValueError(f"No usable {split} inputs")
        report["splits"][split] = {
            "prepared": len(rows), "fit": len(kept), "context_excluded": len(excluded),
            "max_fit_tokens": max(len(record.input_ids) for record in encoded),
            "label_counts": dict(Counter(row["labels"]["research_answer"] for row in kept)),
            "sources": dict(Counter(row["provenance"]["source"] for row in kept)),
        }
        print(json.dumps({"split": split, **report["splits"][split]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

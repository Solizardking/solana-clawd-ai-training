#!/usr/bin/env python3
"""Audit prepared research decisions using the real Clef processor, without weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from clef_research_data import (DATASET_REVISION, MODEL_ID, MODEL_REVISION, decision_statistics,
                                prepare_dataset, read_jsonl, validate_decision_record)
from clef_research_training import encode_rows, load_upstream


def audit_prepared_data(data_dir, module, processor, max_length=4096, max_records=0):
    """Validate every row/hash before encoding a bounded cohort per split."""
    if max_length < 1 or max_records < 0:
        raise ValueError("max_length must be positive and max_records nonnegative")
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    report = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "dataset_revision": manifest["dataset_revision"],
              "input_source": manifest.get("input_source"), "max_length": max_length, "max_records_per_split": max_records,
              "model_weights_downloaded": False, "splits": {}}
    seen = set()
    for split in ("test", "eval", "train"):
        path = data_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["outputs"][split]["sha256"]:
            raise ValueError(f"{split} file hash does not match its manifest")
        all_rows = read_jsonl(path)
        if len(all_rows) != manifest["outputs"][split]["rows"]:
            raise ValueError(f"{split} row count does not match its manifest")
        hashes = {validate_decision_record(row) for row in all_rows}
        if len(hashes) != len(all_rows) or hashes & seen:
            raise ValueError("Duplicate prompts within or across prepared splits")
        seen.update(hashes)
        rows = all_rows[:max_records] if max_records else all_rows
        encoded, kept, excluded = encode_rows(module, processor, rows, max_length)
        if not kept:
            raise ValueError(f"No usable {split} inputs")
        report["splits"][split] = {
            "prepared": len(all_rows), "selected": len(rows), "fit": len(kept), "context_excluded": len(excluded),
            "max_fit_tokens": max(len(record.input_ids) for record in encoded),
            **decision_statistics(kept),
            "sources": dict(Counter(row["provenance"]["source"] for row in kept)),
            "live_observation_records": sum(row["provenance"].get("source_type") == "live_tape_observation" for row in kept),
            "scope": "bounded tokenizer audit" if max_records else "full tokenizer audit",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0, help="Encode at most this many records per split; validate hashes and duplicate prompts for all rows")
    parser.add_argument("--source-dir", type=Path, help="Prepare decisions first from a validated unpublished local staged package")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--live-decisions", type=Path, help="Native observed-field JSONL to prepend when --source-dir is supplied")
    parser.add_argument("--output", type=Path, default=Path("local/clef-tokenizer-preflight.json"))
    args = parser.parse_args()
    if args.max_length < 1 or args.max_records < 0:
        parser.error("max-length must be positive and max-records nonnegative")
    if args.live_decisions and not args.source_dir:
        parser.error("--live-decisions requires --source-dir")
    if args.source_dir:
        prepare_dataset(args.data_dir, dataset_revision=args.dataset_revision, source_dir=args.source_dir,
                        live_decisions=read_jsonl(args.live_decisions) if args.live_decisions else None)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    module = load_upstream()
    report = audit_prepared_data(args.data_dir, module, processor, args.max_length, args.max_records)
    for split, result in report["splits"].items():
        print(json.dumps({"split": split, **result}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

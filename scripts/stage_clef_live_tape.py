#!/usr/bin/env python3
"""Add safe historical tape evidence and native Clef runtime to an unpublished stage.

The ordinary Hugging Face chat splits are unchanged. Native decision records
retain their own schema in ``auxiliary/live_tape`` and are counted separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from clef_live_tape import (capture_live_snapshot, canonical, live_decision_records,
                            serialize_snapshot, snapshot_freshness, validate_snapshot)
from clef_research_data import MODEL_ID, MODEL_REVISION
from expand_realtime_research import file_sha, refresh_package, update_canonical_manifest
from publish_research_expansion import validate_stage
from research_expansion_artifacts import local_path_like, secret_like


ROOT = Path(__file__).resolve().parents[1]
AUXILIARY = "auxiliary/live_tape"
SECTION = "## Clef Native Tokenizer and Live Tape"
RUNTIME_FILES = ("clef_live_tape.py", "clef_research_data.py", "clef_research_training.py")
FRESHNESS_LIMITATIONS = (
    "Saved snapshots are historical observations; current runtime answers require another bounded read-only capture.",
    "A fresh receive timestamp does not establish that a launch event or its marketCapSol value is current.",
    "Health and websocket counters are independent reported values and need not agree.",
    "Creator names, symbols and declared social presence are untrusted data; declarations do not verify accounts.",
    "Labels read back observed fields only. No future outcomes, trade profitability, or signing permissions are labeled.",
)


def _observed_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    websocket: dict[str, set[str]] = {}
    for entry in snapshot["frames"]:
        frame = entry["frame"]
        websocket.setdefault(frame["type"], set()).update(frame)
    return {
        "health": sorted(snapshot["health"]["data"]) if snapshot["health"] is not None else [],
        "websocket_by_type": {kind: sorted(fields) for kind, fields in sorted(websocket.items())},
    }


def _card_section(snapshot: dict[str, Any], rows: int) -> str:
    return f"""{SECTION}

The auxiliary [safe tape snapshot]({AUXILIARY}/snapshot.json) was captured at
`{snapshot['captured_at']}` from the read-only `clawd-ws.fly.dev` health endpoint
and websocket. [Native decision records]({AUXILIARY}/decisions.jsonl) contain
**{rows} observed-field decisions**, separately from the main chat totals.
Missing observations produce no fabricated labels. Saved evidence is historical;
capture the tape again for current runtime questions.

[Tokenizer reference]({AUXILIARY}/tokenizer_reference.json) pins the existing
`{MODEL_ID}` processor/tokenizer to revision `{MODEL_REVISION}`. The vocabulary is
unchanged, and no model weights are included in this auxiliary package.
The copied runtime modules use the native Clef encoder and reject silent state
truncation. Default Hugging Face dataset configurations continue to select only
the main train/eval/test Parquet files; auxiliary native rows have their own schema.

Download the small runtime package and encode a stored decision:

```python
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

root = Path(snapshot_download(
    "solanaclawd/solana-clawd-realtime-research-instruct", repo_type="dataset",
    allow_patterns=["{AUXILIARY}/*"],
)) / "{AUXILIARY}"
sys.path.insert(0, str(root))
from clef_live_tape import encode_native_record, enrich_record_with_live_snapshot
from clef_research_training import load_upstream

reference = json.loads((root / "tokenizer_reference.json").read_text())
processor = AutoProcessor.from_pretrained(reference["model"], revision=reference["revision"])
module = load_upstream()
with (root / "decisions.jsonl").open() as handle:
    record = json.loads(next(handle))
encoded = encode_native_record(module, processor, record)
# For fresh read-only evidence at inference time:
current_record, observation = enrich_record_with_live_snapshot(record)
```

This encoding example loads processor/tokenizer metadata and upstream encoder
code. Inference additionally requires a trained or base Clef model. The runtime
never signs or sends trades. Exact source fields, observation times, hashes and
freshness limitations are recorded in the expansion manifest.
"""


def stage_live_tape(stage: Path, snapshot: dict[str, Any] | None = None,
                    *, timeout: float = 15.0, max_frames: int = 4) -> dict[str, Any]:
    """Validate, augment auxiliary metadata, and revalidate the whole stage."""
    stage = Path(stage).resolve()
    if (stage / "published.json").exists():
        raise ValueError("Preserve authenticated published artifacts; use a separate unpublished stage")
    _, manifest = validate_stage(stage)
    main_before = {item["file"]: file_sha(stage / item["file"]) for item in manifest["outputs"].values()}
    raw_path = stage / "raw/realtime_research_sft.jsonl"
    if raw_path.is_file():
        main_before[raw_path.relative_to(stage).as_posix()] = file_sha(raw_path)
    outputs_before = json.loads(json.dumps(manifest["outputs"]))
    card = (stage / "README.md").read_text(encoding="utf-8")
    frontmatter_before = card.split("---", 2)[1] if card.startswith("---\n") else None

    snapshot = snapshot if snapshot is not None else capture_live_snapshot(timeout=timeout, max_frames=max_frames)
    validate_snapshot(snapshot)
    serialized = serialize_snapshot(snapshot)
    if secret_like(serialized) or local_path_like(serialized):
        raise ValueError("Observation contains credential or local-path payloads")
    rows = live_decision_records(snapshot)
    decision_text = "".join(canonical(row) + "\n" for row in rows)
    if secret_like(decision_text) or local_path_like(decision_text):
        raise ValueError("Native decision records contain unsafe payloads")
    checked_at = datetime.now(timezone.utc)
    live = stage / AUXILIARY
    runtime_sources = {name: ROOT / "scripts" / name for name in RUNTIME_FILES}
    if not all(path.is_file() for path in runtime_sources.values()):
        raise ValueError("Required native tokenizer/runtime source is missing")
    runtime_bytes = {name: path.read_bytes() for name, path in runtime_sources.items()}
    reference = {
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "loader": "transformers.AutoProcessor.from_pretrained",
        "native_encoder": "clef_live_tape.encode_native_record",
        "weights_included": False, "vocabulary_changes": 0,
        "runtime_sources": {name: hashlib.sha256(content).hexdigest() for name, content in runtime_bytes.items()},
    }

    live.mkdir(parents=True, exist_ok=True)
    (live / "snapshot.json").write_text(serialized + "\n", encoding="utf-8")
    (live / "decisions.jsonl").write_text(decision_text, encoding="utf-8")
    (live / "tokenizer_reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    for name, content in runtime_bytes.items():
        (live / name).write_bytes(content)
    entry = {
        "source": snapshot["sources"], "file": f"{AUXILIARY}/decisions.jsonl",
        "rows": len(rows), "sha256": file_sha(live / "decisions.jsonl"),
        "role": "native_clef_observed_field_decisions", "included_in_main_chat_examples": False,
        "snapshot": f"{AUXILIARY}/snapshot.json", "snapshot_sha256": snapshot["snapshot_sha256"],
        "snapshot_file_sha256": file_sha(live / "snapshot.json"),
        "captured_at": snapshot["captured_at"], "observed_fields": _observed_fields(snapshot),
        "freshness_checked_at": checked_at.isoformat(),
        "freshness_at_staging": snapshot_freshness(snapshot, now=checked_at),
        "freshness_limitations": list(FRESHNESS_LIMITATIONS),
        "tokenizer_reference": f"{AUXILIARY}/tokenizer_reference.json",
        "runtime_files": [f"{AUXILIARY}/{name}" for name in RUNTIME_FILES],
        "autoExecute": False,
    }
    # Replace the matching auxiliary stream entry; never append duplicate rows.
    manifest["auxiliary"] = [item for item in manifest.get("auxiliary", [])
                             if not str(item.get("file", "")).startswith(f"{AUXILIARY}/")]
    manifest["auxiliary"].append(entry)
    (stage / "metadata/realtime_research_expansion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    while SECTION in card:
        begin = card.index(SECTION)
        end = card.find("\n## ", begin + len(SECTION))
        card = card[:begin] + (card[end:] if end >= 0 else "")
    card = card.rstrip() + "\n\n" + _card_section(snapshot, len(rows)) + "\n"
    frontmatter_after = card.split("---", 2)[1] if card.startswith("---\n") else None
    if frontmatter_after != frontmatter_before:
        raise ValueError("Native auxiliary staging changed default dataset configuration")
    (stage / "README.md").write_text(card, encoding="utf-8")
    update_canonical_manifest(stage)
    refresh_package(stage)
    package, updated_manifest = validate_stage(stage)
    if updated_manifest["outputs"] != outputs_before or any(file_sha(stage / filename) != digest for filename, digest in main_before.items()):
        raise ValueError("Auxiliary live-tape staging modified main chat examples")
    return {
        "validated": True, "snapshot_sha256": snapshot["snapshot_sha256"],
        "captured_at": snapshot["captured_at"], "native_decisions": len(rows),
        "main_chat_examples": sum(item["rows"] for item in outputs_before.values()),
        "main_chat_files_unchanged": True, "package_files": len(package["files"]),
        "stale_at_staging": entry["freshness_at_staging"]["stale"],
        "weights_included": False, "vocabulary_changes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path("local/research-expansion"))
    parser.add_argument("--snapshot", type=Path, help="Existing safe, hash-validated observation; otherwise capture current evidence")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=4)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot is not None else None
    report = stage_live_tape(args.stage, snapshot, timeout=args.timeout, max_frames=args.max_frames)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

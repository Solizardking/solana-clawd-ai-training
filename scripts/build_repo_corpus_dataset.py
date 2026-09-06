#!/usr/bin/env python3
"""Build a source-grounded SFT dataset from this repo's own knowledge directories.

Covers the directories that carry the project's operational knowledge:

    memory/    nvidia/    ollama/    library/    data/
    docs/      configs/   programs/  studio/

Each directory is walked with the same machinery `build_core_ai_dataset.py`
uses -- secret redaction, binary/lockfile skipping, chunking, summarization,
and dedupe -- so nothing here re-implements the safety filters.

Build:
    python3 scripts/build_repo_corpus_dataset.py

Then process into train/eval/test splits and publish:
    python3 scripts/prepare_dataset.py \
      --input data/repo_corpus_sft.jsonl \
      --output data/repo_corpus_processed \
      --train-ratio 0.9 --eval-ratio 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_core_ai_dataset as core

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths are emitted relative to this repo, not its parent directory.
core.REPO_ROOT = BASE_DIR

# The core-ai walker was tuned for a TypeScript/Markdown tree, so it drops the
# file types that carry this repo's most important knowledge: the Anchor
# programs under programs/ and the Ollama Modelfiles under ollama/.
EXTRA_TEXT_SUFFIXES = {
    ".j2",
    ".jinja",
    ".proto",
    ".rs",
    ".rst",
    ".sql",
}
core.TEXT_SUFFIXES = core.TEXT_SUFFIXES | EXTRA_TEXT_SUFFIXES

# Files whose "suffix" is really part of a descriptive name (Modelfile.preview,
# Dockerfile.gpu, ...). These are plain text and must not be suffix-filtered.
ALLOWED_NAME_PREFIXES = ("Modelfile", "Dockerfile", "Makefile", "Anchor")

_core_should_skip_path = core.should_skip_path


def should_skip_path(path: Path, core_root: Path, max_file_bytes: int) -> str | None:
    if path.name.startswith(ALLOWED_NAME_PREFIXES):
        reason = _core_should_skip_path(path, core_root, max_file_bytes)
        return None if reason == "non_text_suffix" else reason
    return _core_should_skip_path(path, core_root, max_file_bytes)


core.should_skip_path = should_skip_path

DEFAULT_DIRS = [
    "memory",
    "nvidia",
    "ollama",
    "library",
    "data",
    "docs",
    "configs",
    "programs",
    "studio",
]

# data/ holds the multi-hundred-megabyte generated corpora; those are training
# output, not source knowledge, so they are excluded from the walk.
DEFAULT_EXCLUDES = [
    "data/processed",
    "data/repo_corpus_processed",
    "data/core_ai_processed",
    "data/nemo_clawd/corpus",
    "data/nemo_clawd/processed",
    "data/nvidia_rag_store",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dir",
        action="append",
        dest="dirs",
        help=f"Repo-relative directory to ingest (repeatable). Default: {' '.join(DEFAULT_DIRS)}",
    )
    p.add_argument(
        "--exclude",
        action="append",
        dest="excludes",
        help="Repo-relative path prefix to skip (repeatable).",
    )
    p.add_argument("--output", default="data/repo_corpus_sft.jsonl")
    p.add_argument("--manifest", default="data/repo_corpus_manifest.json")
    p.add_argument("--card", default="data/repo_corpus_dataset_card.md")
    p.add_argument("--repo-id", default="solanaclawd/solana-clawd-repo-corpus")
    p.add_argument("--max-file-bytes", type=int, default=400_000)
    p.add_argument("--chunk-chars", type=int, default=6_000)
    p.add_argument("--chunk-overlap", type=int, default=400)
    p.add_argument(
        "--max-examples-per-dir",
        type=int,
        default=4_000,
        help="Cap examples contributed by any single directory.",
    )
    p.add_argument("--public-safe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    return p.parse_args()


def _dir_args(args: argparse.Namespace) -> argparse.Namespace:
    """Shape the namespace `core.build_core_examples` expects."""
    return argparse.Namespace(
        max_file_bytes=args.max_file_bytes,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        max_core_examples=args.max_examples_per_dir,
        public_safe=args.public_safe,
    )


def _excluded(example: dict[str, Any], excludes: list[str]) -> bool:
    source = str(example.get("metadata", {}).get("source", ""))
    return any(source == ex or source.startswith(ex.rstrip("/") + "/") for ex in excludes)


def main() -> None:
    args = parse_args()
    dirs = args.dirs or DEFAULT_DIRS
    excludes = args.excludes if args.excludes is not None else DEFAULT_EXCLUDES

    all_examples: list[dict[str, Any]] = []
    per_dir_stats: dict[str, Any] = {}

    print(f"[1/3] Ingesting {len(dirs)} directories")
    for name in dirs:
        root = (BASE_DIR / name).resolve()
        if not root.exists():
            print(f"      skip {name}: not found")
            per_dir_stats[name] = {"status": "missing"}
            continue

        examples, stats = core.build_core_examples(root, _dir_args(args))
        kept = [e for e in examples if not _excluded(e, excludes)]
        dropped = len(examples) - len(kept)

        all_examples.extend(kept)
        stats["excluded_by_prefix"] = dropped
        stats["examples"] = len(kept)
        per_dir_stats[name] = stats
        print(
            f"      {name:<10} files={stats['files_used']:<4} "
            f"examples={len(kept):<5} skipped={stats['skipped']} excluded={dropped}"
        )

    print("[2/3] Deduping")
    deduped, dupes = core.dedupe_examples(all_examples)
    print(f"      removed {dupes} duplicates -> {len(deduped)} examples")

    source_counts: dict[str, int] = {}
    for example in deduped:
        key = str(example.get("metadata", {}).get("source", "")).split("/")[0]
        source_counts[key] = source_counts.get(key, 0) + 1

    manifest = {
        "repo_id": args.repo_id,
        "directories": dirs,
        "excludes": excludes,
        "source_counts": source_counts,
        "stats": {
            "per_directory": per_dir_stats,
            "duplicates_removed": dupes,
            "total_examples": len(deduped),
        },
        "build": {
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "max_file_bytes": args.max_file_bytes,
            "public_safe": args.public_safe,
        },
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    print("[3/3] Writing dataset")
    core.write_jsonl(BASE_DIR / args.output, deduped)
    core.write_manifest(BASE_DIR / args.manifest, manifest)
    core.write_card(BASE_DIR / args.card, args.repo_id, manifest)
    print(f"      {args.output}  ({len(deduped)} examples)")
    print(f"      {args.manifest}")
    print(f"      {args.card}")


if __name__ == "__main__":
    main()

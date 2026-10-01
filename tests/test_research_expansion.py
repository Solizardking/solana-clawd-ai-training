"""Preservation, duplicate and held-out isolation checks for additive merging."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from expand_realtime_research import merge_candidates, sha


def row(question, answer):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return {"messages": messages, "example_sha256": sha(messages)}


def candidate(question, answer):
    return {"row": row(question, answer), "donor": "local", "provenance": {"source": "fixture"}}


def test_original_rows_preserved_and_duplicates_skipped():
    original = row("existing", "original answer")
    base = {"train": [original], "eval": [], "test": []}
    incoming = {"train": [candidate(" existing ", " original   answer "), candidate("new", "new answer")], "eval": [], "test": []}
    merged, audit, lineage = merge_candidates(base, incoming)
    assert merged["train"][0] is original
    assert len(merged["train"]) == 2
    assert audit["local:duplicate"] == 1
    assert len(lineage) == 1
    assert base["train"] == [original]


def test_new_training_does_not_copy_heldout_prompt_or_answer():
    base = {"train": [], "eval": [row("protected question", "heldout answer")], "test": []}
    incoming = {"train": [candidate("protected question", "different answer"), candidate("different question", "heldout answer")], "eval": [], "test": []}
    merged, audit, _ = merge_candidates(base, incoming)
    assert not merged["train"]
    assert audit["local:cross_split_prompt_or_answer"] == 2


def test_incoming_holdout_precedes_train_and_does_not_duplicate_existing_train():
    base = {"train": [row("old train", "old answer")], "eval": [], "test": []}
    incoming = {"train": [candidate("new heldout", "changed answer")],
                "eval": [candidate("old train", "changed answer")],
                "test": [candidate("new heldout", "test answer")]}
    merged, audit, _ = merge_candidates(base, incoming)
    assert len(merged["train"]) == 1 and not merged["eval"] and len(merged["test"]) == 1
    assert audit["local:cross_split_prompt_or_answer"] == 2

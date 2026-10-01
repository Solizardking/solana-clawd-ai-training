"""Auxiliary evidence staging preserves chat bytes/configs and authentic records."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clef_live_tape as tape
from expand_realtime_research import file_sha, refresh_package
from publish_research_expansion import validate_stage
from stage_clef_live_tape import AUXILIARY, MODEL_ID, MODEL_REVISION, SECTION, stage_live_tape


def observation():
    stamp = datetime(2026, 10, 1, 12, tzinfo=timezone.utc).isoformat()
    result = {
        "schema_version": "clawd-clef-live-v1", "observation_started_at": stamp, "captured_at": stamp,
        "sources": {"health": tape.HTTP_URL, "websocket": tape.WS_URL},
        "evidence_scope": "Historical read-only fixture; not a live observation.",
        "health": {"received_at": stamp, "data": {"status": "ok", "solana": True}},
        "frames": [{"received_at": stamp, "frame": {
            "type": "token-launch", "mint": "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump",
            "name": "Untrusted\u2028name", "symbol": "CLAWD  ", "time": stamp,
            "declared_socials": {"website": True, "twitter": False, "telegram": False},
        }}],
        "transport": {"http": {"status": "observed"}, "websocket": {
            "status": "observed", "handshake_verified": True, "frames_received": 1,
        }},
    }
    result["snapshot_sha256"] = tape.digest(result)
    return result


@pytest.fixture
def stage(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    (root / "metadata").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "raw").mkdir()
    (root / "README.md").write_text("---\nconfigs:\n  - config_name: default\n    data_files: data/*.parquet\n---\n# Fixture\n\n## Research Citations\n\nhttps://arxiv.org/abs/2605.12151\nhttps://arxiv.org/abs/2606.08232\n")
    (root / "metadata/expansion_lineage.jsonl").write_text('')
    (root / "raw/realtime_research_sft.jsonl").write_text('{"messages": []}\n')
    outputs = {}
    for split in ("train", "eval", "test"):
        relative = f"data/{split}-00000-of-00001.parquet"
        # Byte sentinels test immutable main artifacts, not Parquet semantics.
        (root / relative).write_bytes(f"preserve exact {split} content".encode())
        outputs[split] = {"file": relative, "rows": 1, "base_rows": 1, "added_rows": 0}
    manifest = {"base_revision": "fixture-revision", "original_rows_preserved_in_original_splits": True,
                "generated_at": "fixture-date", "outputs": outputs, "sources": [],
                "input_audit": {}, "merge_audit": {}, "auxiliary": []}
    (root / "metadata/realtime_research_expansion_manifest.json").write_text(json.dumps(manifest))
    base_manifest = tmp_path / "original_manifest.json"
    base_manifest.write_text(json.dumps({"counts": {"examples": 3, "duplicate_examples": 0,
                                                    "secret_or_invalid_skipped": 0, "by_source_type": {}}, "sources": []}))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(base_manifest))
    refresh_package(root)
    validate_stage(root)
    return root


def test_actual_decisions_staged_separately_without_chat_or_config_changes(stage):
    before_manifest = json.loads((stage / "metadata/realtime_research_expansion_manifest.json").read_text())
    before_hashes = {path: file_sha(stage / path) for path in ["raw/realtime_research_sft.jsonl"] + [item["file"] for item in before_manifest["outputs"].values()]}
    frontmatter = (stage / "README.md").read_text().split("---", 2)[1]
    snapshot = observation()
    expected = tape.live_decision_records(snapshot)
    result = stage_live_tape(stage, snapshot)
    assert result["validated"] and result["native_decisions"] == len(expected) == 2
    assert result["main_chat_examples"] == 3 and result["main_chat_files_unchanged"]
    assert result["weights_included"] is False and result["vocabulary_changes"] == 0
    assert before_hashes == {path: file_sha(stage / path) for path in before_hashes}
    live = stage / AUXILIARY
    assert json.loads((live / "snapshot.json").read_text()) == snapshot
    with (live / "decisions.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows == expected
    reference = json.loads((live / "tokenizer_reference.json").read_text())
    assert reference["model"] == MODEL_ID and reference["revision"] == MODEL_REVISION
    assert len(reference["runtime_sources"]) == 3
    assert all(file_sha(live / name) == digest for name, digest in reference["runtime_sources"].items())
    assert not list(live.glob("*.safetensors"))
    _, manifest = validate_stage(stage)
    assert manifest["outputs"] == before_manifest["outputs"]
    entry = manifest["auxiliary"][0]
    assert entry["rows"] == 2 and entry["captured_at"] == snapshot["captured_at"]
    assert entry["observed_fields"]["health"] == ["solana", "status"]
    assert entry["included_in_main_chat_examples"] is False
    assert entry["freshness_limitations"] and entry["autoExecute"] is False
    canonical_manifest = json.loads((stage / "metadata/realtime_research_dataset_manifest.json").read_text())
    assert canonical_manifest["counts"]["examples"] == 3
    assert canonical_manifest["expansion"]["auxiliary"] == manifest["auxiliary"]
    card = (stage / "README.md").read_text()
    assert card.split("---", 2)[1] == frontmatter
    assert all(citation in card for citation in ("2605.12151", "2606.08232"))

    stage_live_tape(stage, snapshot)
    _, repeated = validate_stage(stage)
    assert len(repeated["auxiliary"]) == 1
    assert (stage / "README.md").read_text().count(SECTION) == 1


def test_published_and_unsafe_snapshot_guards_precede_mutation(stage):
    before = {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob('*') if path.is_file()}
    unsafe = copy.deepcopy(observation())
    unsafe["health"]["data"]["rpcHttp"] = "https://rpc.example?api_key=NEVER"
    unsafe["snapshot_sha256"] = tape.digest({key: value for key, value in unsafe.items() if key != "snapshot_sha256"})
    with pytest.raises(ValueError, match="unsafe health"):
        stage_live_tape(stage, unsafe)
    assert before == {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob('*') if path.is_file()}
    (stage / "published.json").write_text('{"commit": "already-published"}')
    with pytest.raises(ValueError, match="published artifacts"):
        stage_live_tape(stage, observation())
    assert not (stage / AUXILIARY).exists()


def test_unavailable_snapshot_does_not_invent_native_rows(stage, monkeypatch):
    missing = observation()
    missing["health"] = None
    missing["frames"] = []
    missing["transport"] = {"http": {"status": "unavailable"}, "websocket": {
        "status": "timeout", "handshake_verified": False, "frames_received": 0,
    }}
    missing["snapshot_sha256"] = tape.digest({key: value for key, value in missing.items() if key != "snapshot_sha256"})
    monkeypatch.setattr("stage_clef_live_tape.capture_live_snapshot", lambda **kwargs: missing)
    report = stage_live_tape(stage)
    assert report["native_decisions"] == 0 and report["stale_at_staging"] is True
    assert (stage / AUXILIARY / "decisions.jsonl").read_text() == ''
    _, manifest = validate_stage(stage)
    assert manifest["auxiliary"][0]["rows"] == 0

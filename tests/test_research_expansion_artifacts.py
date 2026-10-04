"""Public release sanitation and supporting-artifact staging checks."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_expansion_artifacts import (
    LOCAL_PATH,
    REDACTED_SECRET,
    local_path_like,
    sanitize_public,
    secret_like,
    stage_supporting_artifacts,
)


SOLANA_MINT = "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump"


def test_recursive_sanitation_preserves_public_addresses_citations_and_relative_paths():
    original = {
        "paths": [('/Users/' + 'alice/Downloads/research paper.pdf'), ('/home/' + 'bob/project/data.json'), "/workspace/data/train.parquet"],
        "relative": "nvidia/blueprints/enterprise-rag/pipeline.py",
        "mint": SOLANA_MINT,
        "token": SOLANA_MINT,
        "public_key": SOLANA_MINT,
        "license_note": "Derived under CC BY 4.0 with attribution.",
        "citation": "Kamat, A. U. (2026). RED-2400. https://arxiv.org/abs/2605.12151",
        "nested": {"HF_TOKEN": "hf_" + "a" * 32, "private_key": list(range(64))},
    }
    clean = sanitize_public(original)
    serialized = json.dumps(clean)
    assert not secret_like(serialized) and not local_path_like(serialized)
    assert "/Users/" not in serialized and "/home/" not in serialized
    assert clean["relative"] == original["relative"]
    assert clean["mint"] == clean["token"] == clean["public_key"] == SOLANA_MINT
    assert clean["citation"] == original["citation"] and clean["license_note"] == original["license_note"]
    assert clean["nested"]["HF_TOKEN"] == clean["nested"]["private_key"] == REDACTED_SECRET
    assert original["nested"]["HF_TOKEN"].startswith("hf_")  # Input is unchanged.


@pytest.mark.parametrize("secret", [
    "hf_" + "a" * 32,
    "sk-proj-" + "b" * 32,
    "nvapi-" + "c" * 32,
    "github_pat_" + "d" * 48,
    "ghp_" + "e" * 32,
    "AKIA" + "F" * 16,
    ('api_key=' + '"unprefixed-opaque-credential"'),
    'password="a password containing spaces"',
    '"access_token": "opaqueAccessCredential"',
    'SOLANA_TRACKER_ACCESS_TOKEN=opaqueTrackerCredential',
    ('api_key=' + '\\"opaqueEscapedCredential\\"'),
    "Authorization: Bearer arbitraryOpaqueCredential",
    "Authorization: Basic dXNlcjpwYXNzd29yZA==",
    ('https://rpc.example.org/?api_key=' + 'opaqueQueryCredential&slot=42'),
    "https://example.org/?X-Amz-Signature=opaqueSignature",
    ('https://example.org/?slot=42&amp;api_key=' + 'opaqueQueryCredential'),
    "https://alice:password123@example.org/data",
    ('-----BEGIN ' + 'PRIVATE KEY-----\nactualKeyPayload\n-----END PRIVATE KEY-----'),
])
def test_detects_and_redacts_actual_secret_payloads(secret):
    assert secret_like(secret)
    clean = sanitize_public(secret)
    assert REDACTED_SECRET in clean
    assert not secret_like(clean)


@pytest.mark.parametrize("placeholder", [
    "HF_TOKEN=$HF_TOKEN",
    'SOLANA_TRACKER_ACCESS_TOKEN="${SOLANA_TRACKER_ACCESS_TOKEN}"',
    ('api_key=' + '"${SOLANA_TRACKER_ACCESS_TOKEN}"'),
    'private_key="<PRIVATE_KEY>"',
    "Authorization: Bearer $HF_TOKEN",
    ('https://rpc.example.org/?api_key=' + 'YOUR_API_KEY&slot=42'),
    "https://example.org/?token=%24HF_TOKEN",
    ('api_key=' + '"[REDACTED_SECRET]"'),
    "hf_...",
    r'Q_API_KEY: \"...\"',
    'private_key="replace-me"',
])
def test_placeholders_are_not_secrets_and_do_not_change(placeholder):
    assert not secret_like(placeholder)
    assert sanitize_public(placeholder) == placeholder


def test_local_paths_scrubbed_without_altering_public_routes_or_identifiers():
    text = " ".join([
        r"C:\Users\alice\data\file.json", ('file:///Users/' + 'alice/data.json'),
        "/tmp/export", "~/research/data.json", "%2FUsers%2Falice%2Fdata.json",
        "https://arxiv.org/abs/2606.08232", "/api/register/preview", SOLANA_MINT,
    ])
    assert local_path_like(text)
    clean = sanitize_public(text)
    assert not local_path_like(clean)
    assert LOCAL_PATH in clean
    assert "alice" not in clean
    assert "https://arxiv.org/abs/2606.08232" in clean and "/api/register/preview" in clean
    assert SOLANA_MINT in clean


def test_stages_only_source_documentation_and_hashes_actual_bytes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "realtime_research_citations.md").write_text("Kamat, A. U. (2026). https://arxiv.org/abs/2605.12151\nhttps://arxiv.org/abs/2606.08232\n")
    manifest = {"historical_examples": 30450, "inputs": ('/Users/' + 'alice/old/data.jsonl'), "license": "cc-by-4.0", "api_key": "opaqueSensitivePayload"}
    (data / "core_ai_dataset_manifest.json").write_text(json.dumps(manifest))
    (data / "dataset.json").write_text(json.dumps({"messages": [{"role": "assistant", "content": "raw training row"}]}))
    for name in ("raw.jsonl", "train.parquet", "data.arrow", "index.faiss"):
        (data / name).write_bytes(b"do not stage")
    (data / "credentials.json").write_text(json.dumps({"api_key": "doNotRead"}))
    (data / "secret_client_secret.json").write_text(json.dumps({"api_key": "doNotRead"}))
    (data / "nvidia_cache").mkdir()
    (data / "nvidia_cache" / "response_report.json").write_text('{}')
    (data / "binary_report.md").write_bytes(b"\x00binary")
    outside = tmp_path / "outside.md"
    outside.write_text("outside source")
    (data / "linked.md").symlink_to(outside)
    output = tmp_path / "staged"
    inventory = stage_supporting_artifacts(data, output)
    assert {entry["source"] for entry in inventory} == {"core_ai_dataset_manifest.json", "realtime_research_citations.md"}
    for entry in inventory:
        staged = output / entry["path_in_repo"]
        assert staged.is_file()
        assert entry["source_exists"] and entry["new_examples"] == 0
        assert entry["historical_counts_are_documentation"]
        assert entry["bytes"] == staged.stat().st_size
        assert entry["sha256"] == hashlib.sha256(staged.read_bytes()).hexdigest()
        assert entry["source_sha256"] == hashlib.sha256((data / entry["source"]).read_bytes()).hexdigest()
        assert not secret_like(staged.read_text()) and not local_path_like(staged.read_text())
    staged_manifest = json.loads((output / "metadata/local_sources/core_ai_dataset_manifest.json").read_text())
    assert staged_manifest["historical_examples"] == 30450
    assert staged_manifest["license"] == "cc-by-4.0"
    assert staged_manifest["api_key"] == REDACTED_SECRET
    assert stage_supporting_artifacts(data, output) == inventory


def test_output_under_source_root_does_not_reingest_previous_outputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "README.md").write_text("Original source documentation.")
    output = data / "expanded"
    first = stage_supporting_artifacts(data, output)
    second = stage_supporting_artifacts(data, output)
    assert first == second and len(second) == 1
    assert not (output / "metadata/local_sources/expanded").exists()
    with pytest.raises(ValueError, match="differ"):
        stage_supporting_artifacts(data, data)


def test_invalid_source_json_stops_staging(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "broken_manifest.json").write_text('{"invalid": ')
    with pytest.raises(json.JSONDecodeError):
        stage_supporting_artifacts(data, tmp_path / "output")

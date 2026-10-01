"""Data isolation and genuine Clef-head/LoRA gradient integration checks."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clef_research_data import (DATASET_ID, build_decisions, decision_prompt_hash, digest,
                                immutable_revision, model_input_exclusion_reason, prepare_dataset, read_jsonl)
from clef_research_training import attach_lora, compare_predictions, encode_rows, evaluate, load_upstream, supervised_loss, trainable_fingerprints


def raw_row(split, index, prompt=None):
    messages = [
        {"role": "system", "content": "Use supplied Solana research evidence."},
        {"role": "user", "content": prompt or f"{split}: Explain Solana account concept {index}."},
        {"role": "assistant", "content": f"Research response {split} {index}: Solana accounts carry address-specific data and owner information."},
    ]
    return {"messages": messages, "source_type": "parquet", "source": f"{split}.parquet",
            "record_id": str(index), "example_sha256": digest(messages)}


def fixture_splits():
    return {split: [raw_row(split, index) for index in range(16)] for split in ("train", "eval", "test")}


def test_data_labels_are_original_answers_and_candidates_stay_in_split():
    raw = fixture_splits()
    decisions, audit = build_decisions(raw)
    repeated, _ = build_decisions(raw)
    assert decisions == repeated
    all_labels = set()
    for split, rows in decisions.items():
        hashes = {row["example_sha256"] for row in raw[split]}
        originals = {row["record_id"]: row["messages"][-1]["content"] for row in raw[split]}
        assert audit[f"{split}_output"] == len(raw[split])
        for row in rows:
            question = row["questions"]["research_answer"]
            gold = row["labels"]["research_answer"]
            all_labels.add(gold)
            assert question["criteria"][gold] == originals[row["provenance"]["record_id"]]
            assert len(set(question["criteria"].values())) == 4
            assert set(row["candidate_example_sha256"].values()) <= hashes
            assert all(message["role"] != "assistant" for message in row["state"]["conversation"])
            assert "labels" not in row["state"] and "provenance" not in row["state"]
    assert all_labels == {f"option_{i}" for i in range(4)}


def test_duplicate_prompt_is_removed_from_training_with_heldout_priority():
    raw = fixture_splits()
    raw["train"].append(raw_row("train", 30, "Shared question"))
    raw["eval"].append(raw_row("eval", 30, "Shared question"))
    raw["test"].append(raw_row("test", 30, "Shared question"))
    decisions, audit = build_decisions(raw)
    assert audit["train_duplicate_prompt_removed"] == 1
    assert audit["eval_duplicate_prompt_removed"] == 1
    sets = [{row["provenance"]["prompt_hash"] for row in decisions[split]} for split in ("train", "eval", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_multiturn_or_nontext_examples_are_rejected():
    raw = fixture_splits()
    malformed = raw_row("train", 100)
    malformed["messages"].insert(2, {"role": "assistant", "content": "An earlier answer would leak the target."})
    raw["train"].append(malformed)
    nontext = raw_row("train", 101)
    nontext["messages"][-1]["content"] = [{"type": "text", "text": "not supported"}]
    raw["train"].append(nontext)
    _, audit = build_decisions(raw)
    assert audit["train_invalid"] == 2


@pytest.mark.parametrize("value,category", [
    ("secret_seed = " + json.dumps(list(range(32))), "signing_byte_array"),
    ("Keypair.from_seed(bytes(" + json.dumps(list(range(32))) + "))", "signing_byte_array"),
    ("const secretKey = Uint8Array.from(" + json.dumps(list(range(64))) + ");", "signing_byte_array"),
    ({"private_key": list(range(64))}, "signing_byte_array"),
    ({"signing_seed": list(range(32))}, "signing_byte_array"),
    ({"signingSeed": list(range(32))}, "signing_byte_array"),
    ({"nested": {"my_signing_seed": list(range(32))}}, "signing_byte_array"),
    ({"state": {"seed": {"bytes": list(range(32))}}}, "signing_byte_array"),
    ("private_key = \"" + "A" * 64 + "\"", "encoded_signing_literal"),
    ({"private_key": "A" * 64}, "encoded_signing_literal"),
    ({"signingSeed": "A" * 64}, "encoded_signing_literal"),
    ("HF_TOKEN=" + "hf_" + "A" * 24, "credential_payload"),
    ("-----BEGIN PRIVATE KEY-----\nsynthetic-test-data", "credential_payload"),
])
def test_literal_signing_material_and_credential_shapes_are_excluded(value, category):
    # Synthetic shapes only: no repository credential or published key bytes.
    assert model_input_exclusion_reason(value) == category


@pytest.mark.parametrize("value", [
    "public_key = " + json.dumps(list(range(32))),
    {"public_key": list(range(32))},
    "public_key = " + json.dumps(list(range(32))) + "; Keypair.from_seed(os.getenv('SIGNING_SEED'))",
    "Pubkey.from_bytes(" + json.dumps(list(range(32))) + ")",
    "private_key = os.environ['SIGNING_KEY']; keypair = Keypair.generate()",
    "private_key = \"$SIGNING_KEY\"; HF_TOKEN=hf_...",
    {"secret_seed": "${SIGNING_SEED}", "private_key": "YOUR_PRIVATE_KEY"},
    {"state": "Discuss private key isolation without showing material.",
     "questions": {"public_bytes": {"criteria": {"example": "public_key = " + json.dumps(list(range(32)))}}}},
])
def test_public_bytes_and_variable_or_placeholder_examples_remain_usable(value):
    assert model_input_exclusion_reason(value) is None


def test_all_roles_and_splits_are_filtered_before_distractor_selection_without_source_mutation():
    raw = fixture_splits()
    excluded_hashes = set()
    for split in ("train", "eval", "test"):
        for offset, (role, content) in enumerate([
            ("system", "Keep signing separate. seed = " + json.dumps(list(range(32)))),
            ("user", "Review this secret_key = " + json.dumps(list(range(64)))),
            ("assistant", "Research response with synthetic credential " + "hf_" + "A" * 24),
        ]):
            unsafe = raw_row(split, 100 + offset)
            next(message for message in unsafe["messages"] if message["role"] == role)["content"] = content
            unsafe["example_sha256"] = digest(unsafe["messages"])
            excluded_hashes.add(unsafe["example_sha256"])
            raw[split].append(unsafe)
    unchanged = copy.deepcopy(raw)
    decisions, audit = build_decisions(raw)
    assert raw == unchanged
    for split, rows in decisions.items():
        assert len(rows) == 16
        assert audit[f"{split}_excluded_signing_byte_array"] == 2
        assert audit[f"{split}_excluded_credential_payload"] == 1
        for row in rows:
            assert row["provenance"]["example_sha256"] not in excluded_hashes
            assert not set(row["candidate_example_sha256"].values()) & excluded_hashes
            assert model_input_exclusion_reason({"state": row["state"], "questions": row["questions"]}) is None


def test_safe_public_array_and_placeholder_conversations_are_retained():
    raw = fixture_splits()
    public = raw_row("train", 100, "Inspect public account bytes " + json.dumps(list(range(32))))
    public["messages"][-1]["content"] = "The public_key = " + json.dumps(list(range(32))) + " contains public address bytes."
    placeholder = raw_row("train", 101, "Use private_key = os.environ['SIGNING_KEY']; never give the model signing material.")
    raw["train"].extend([public, placeholder])
    decisions, audit = build_decisions(raw)
    ids = {row["provenance"]["record_id"] for row in decisions["train"]}
    assert {"100", "101"} <= ids
    assert not any("_excluded_" in name for name in audit)


def test_jsonl_reader_preserves_unicode_separators_in_source_text(tmp_path):
    rows = [{"text": "Research paragraph\u2028next line\u0085another line"}, {"text": "Second record"}]
    path = tmp_path / "research.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    assert read_jsonl(path) == rows
    assert read_jsonl(path, limit=1) == rows[:1]


@pytest.fixture
def local_stage(tmp_path):
    """Real tiny Parquets and the same package/lineage contracts as the stage."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    stage = tmp_path / "source"
    (stage / "data").mkdir(parents=True)
    (stage / "metadata").mkdir()
    outputs, files = {}, []
    for split, rows in fixture_splits().items():
        name = f"data/{split}-00000-of-00001.parquet"
        path = stage / name
        pq.write_table(pa.Table.from_pylist(rows), path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs[split] = {"rows": len(rows), "file": name, "sha256": sha}
        files.append({"path": name, "sha256": sha, "bytes": path.stat().st_size})
    manifest_name = "metadata/realtime_research_expansion_manifest.json"
    manifest_path = stage / manifest_name
    manifest_path.write_text(json.dumps({"repo_id": DATASET_ID, "base_revision": "a" * 40, "outputs": outputs}))
    files.append({"path": manifest_name, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "bytes": manifest_path.stat().st_size})
    (stage / "package.json").write_text(json.dumps({"repo_id": DATASET_ID, "base_revision": "a" * 40, "files": files}))
    return stage


def fixture_live_decisions():
    from clef_live_tape import HTTP_URL, WS_URL, digest as live_digest, live_decision_records
    timestamp = datetime(2026, 10, 1, 12, tzinfo=timezone.utc).isoformat()
    snapshot = {
        "schema_version": "clawd-clef-live-v1", "observation_started_at": timestamp, "captured_at": timestamp,
        "sources": {"health": HTTP_URL, "websocket": WS_URL}, "evidence_scope": "Offline fixture observations only.",
        "health": {"received_at": timestamp, "data": {"status": "ok", "solana": True}},
        "frames": [{"received_at": timestamp, "frame": {"type": "token-launch", "mint": "8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump", "symbol": "Clawd  ",
                   "declared_socials": {"website": True, "twitter": False, "telegram": False}}}],
        "transport": {"http": {"status": "observed"}, "websocket": {"status": "observed", "handshake_verified": True, "frames_received": 1}},
    }
    snapshot["snapshot_sha256"] = live_digest(snapshot)
    return live_decision_records(snapshot)


def test_local_stage_loads_real_parquets_offline_and_records_unpublished_identity(local_stage, tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: pytest.fail("local source must not download remote input"))
    output = tmp_path / "decisions"
    manifest = prepare_dataset(output, source_dir=local_stage)
    package_hash = hashlib.sha256((local_stage / "package.json").read_bytes()).hexdigest()
    assert manifest["dataset_revision"] == "unpublished:" + package_hash
    assert manifest["input_source"]["kind"] == "local_staged_unpublished"
    assert manifest["input_source"]["publication_verified"] is False
    assert manifest["input_source"]["base_revision"] == "a" * 40
    assert manifest["input_source"]["input_rows"] == {"train": 16, "eval": 16, "test": 16}
    for split in ("train", "eval", "test"):
        rows = read_jsonl(output / f"{split}.jsonl")
        assert len(rows) == 16 == manifest["outputs"][split]["rows"]
        assert manifest["outputs"][split]["candidate_counts"] == {"4": 16}
        assert manifest["outputs"][split]["chance_accuracy"] == 0.25


def test_local_stage_rejects_modified_payload_before_loading(local_stage, tmp_path):
    path = local_stage / "data/train-00000-of-00001.parquet"
    with path.open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="byte count"):
        prepare_dataset(tmp_path / "decisions", source_dir=local_stage)


def test_local_stage_verifies_declared_counts_after_manifest_hash_verification(local_stage, tmp_path):
    path = local_stage / "metadata/realtime_research_expansion_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"]["train"]["rows"] += 1
    path.write_text(json.dumps(manifest))
    package_path = local_stage / "package.json"
    package = json.loads(package_path.read_text())
    item = next(item for item in package["files"] if item["path"] == "metadata/realtime_research_expansion_manifest.json")
    item.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    package_path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="row count"):
        prepare_dataset(tmp_path / "decisions", source_dir=local_stage)


@pytest.mark.parametrize("revision", ["main", "refs/pr/1", "latest", "abc123", "g" * 40])
def test_remote_mutable_revisions_are_rejected_before_network(revision, tmp_path):
    with pytest.raises(ValueError, match="immutable 40-hex"):
        prepare_dataset(tmp_path / "decisions", dataset_revision=revision)
    assert immutable_revision("A" * 40) == "a" * 40


def test_live_rows_precede_pilot_rows_and_dynamic_candidate_counts_are_preserved(local_stage, tmp_path):
    live = fixture_live_decisions()
    unchanged = copy.deepcopy(live)
    output = tmp_path / "decisions"
    manifest = prepare_dataset(output, source_dir=local_stage, live_decisions=live)
    rows = read_jsonl(output / "train.jsonl")
    assert live == unchanged
    assert [row["id"] for row in rows[:2]] == [row["id"] for row in live]
    assert all(row["provenance"]["source_type"] == "live_tape_observation" for row in rows[:2])
    assert len(rows) == 18
    assert manifest["outputs"]["train"]["candidate_counts"] == {"4": 16, "6": 1, "8": 1}
    assert manifest["outputs"]["train"]["chance_accuracy"] == pytest.approx((16 / 4 + 1 / 6 + 1 / 8) / 18)
    assert manifest["live_observations"]["splits"] == {"train": 2, "eval": 0, "test": 0}
    changed = copy.deepcopy(live[0])
    changed["labels"]["research_answer"] = "different_gold"
    changed["questions"]["research_answer"]["criteria"] = {"changed": "different candidates"}
    changed["provenance"] = {"never_in_input_hash": True}
    assert decision_prompt_hash(changed) == decision_prompt_hash(live[0])


def test_live_duplicate_prompts_keep_heldout_priority_without_split_leakage(local_stage, tmp_path):
    live = fixture_live_decisions()
    output = tmp_path / "decisions"
    manifest = prepare_dataset(output, source_dir=local_stage, live_decisions={"train": live, "eval": [live[0]], "test": [live[0]]})
    hashes = {split: {row["provenance"]["prompt_hash"] for row in read_jsonl(output / f"{split}.jsonl")} for split in ("train", "eval", "test")}
    assert not hashes["train"] & hashes["eval"] and not hashes["train"] & hashes["test"] and not hashes["eval"] & hashes["test"]
    assert manifest["live_observations"]["splits"] == {"train": 1, "eval": 0, "test": 1}
    assert manifest["audit"]["train_live_duplicate_prompt_removed"] == 1
    assert manifest["audit"]["eval_live_duplicate_prompt_removed"] == 1


def test_live_tampered_prompt_hash_is_rejected(local_stage, tmp_path):
    live = fixture_live_decisions()
    live[0]["provenance"]["prompt_hash"] = "f" * 64
    with pytest.raises(ValueError, match="prompt_hash differs"):
        prepare_dataset(tmp_path / "decisions", source_dir=local_stage, live_decisions=live)


def test_live_state_and_candidate_credentials_are_filtered_with_safe_manifest_audit(local_stage, tmp_path):
    clean = fixture_live_decisions()[0]
    secret_state = copy.deepcopy(clean)
    secret_state["id"] = "synthetic-private-live-state"
    secret_state["state"] = {"private_seed": list(range(32))}
    credential_candidate = copy.deepcopy(clean)
    credential_candidate["id"] = "synthetic-credential-live-candidate"
    question = credential_candidate["questions"]["research_answer"]
    question["criteria"][next(iter(question["criteria"]))] = "hf_" + "A" * 24
    live = {split: [secret_state, credential_candidate] for split in ("train", "eval", "test")}
    live["train"].insert(0, clean)
    unchanged = copy.deepcopy(live)
    manifest = prepare_dataset(tmp_path / "decisions", source_dir=local_stage, live_decisions=live)
    assert live == unchanged
    security = manifest["security_filter"]
    assert security["version"] == 1 and security["applied"] is True
    assert security["published_source_modified"] is False
    assert "before answer-candidate selection" in security["scope"]
    assert manifest["live_observations"]["splits"] == {"train": 1, "eval": 0, "test": 0}
    for split in ("train", "eval", "test"):
        assert security["exclusion_counts"][f"{split}_live_excluded_signing_byte_array"] == 1
        assert security["exclusion_counts"][f"{split}_live_excluded_credential_payload"] == 1
        rows = read_jsonl(tmp_path / "decisions" / f"{split}.jsonl")
        assert all(not row["id"].startswith("synthetic-") for row in rows)
    # The security manifest contains only categories/counts, never payloads.
    assert not model_input_exclusion_reason(security)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[1 + byte % 126 for byte in text.encode()])


@pytest.fixture(scope="module")
def upstream():
    return load_upstream()


def test_encoding_does_not_include_labels_or_provenance_and_reports_context_exclusion(upstream):
    from types import SimpleNamespace
    rows = build_decisions(fixture_splits())[0]["train"][:1]
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    altered = copy.deepcopy(rows[0])
    altered["labels"] = {"research_answer": "not_an_option"}
    altered["provenance"] = {"secret": "must never enter model input"}
    assert upstream.encode_record(processor.tokenizer, rows[0]) == upstream.encode_record(processor.tokenizer, altered)
    encoded, kept, errors = encode_rows(upstream, processor, rows, max_length=10)
    assert not encoded and not kept
    assert len(errors) == 1 and "exceeds" in errors[0]["reason"]


def test_bounded_preflight_audits_native_live_choices_and_keeps_input_identity(local_stage, tmp_path, upstream):
    from preflight_clef_research import audit_prepared_data
    from types import SimpleNamespace
    output = tmp_path / "decisions"
    manifest = prepare_dataset(output, source_dir=local_stage, live_decisions=fixture_live_decisions())
    report = audit_prepared_data(output, upstream, SimpleNamespace(tokenizer=TinyTokenizer()), max_length=10_000, max_records=2)
    assert report["dataset_revision"] == manifest["dataset_revision"]
    assert report["model_weights_downloaded"] is False
    assert report["splits"]["train"]["prepared"] == 18
    assert report["splits"]["train"]["selected"] == 2
    assert report["splits"]["train"]["live_observation_records"] == 2
    assert report["splits"]["train"]["candidate_counts"] == {"6": 1, "8": 1}
    assert report["splits"]["train"]["chance_accuracy"] == pytest.approx((1 / 6 + 1 / 8) / 2)


def test_bounded_preflight_checks_duplicate_rows_beyond_selected_cohort(local_stage, tmp_path, upstream):
    from preflight_clef_research import audit_prepared_data
    from types import SimpleNamespace
    output = tmp_path / "decisions"
    manifest = prepare_dataset(output, source_dir=local_stage)
    heldout = read_jsonl(output / "test.jsonl")[10]
    with (output / "train.jsonl").open("a") as handle:
        handle.write(json.dumps(heldout) + "\n")
    manifest["outputs"]["train"].update(rows=17, sha256=hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate prompts"):
        audit_prepared_data(output, upstream, SimpleNamespace(tokenizer=TinyTokenizer()), max_length=10_000, max_records=1)


def test_native_clef_head_and_lora_train_and_saved_weights_reload(upstream, tmp_path):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from types import SimpleNamespace
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = Qwen3_5Config(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                     "vocab_size": 128, "layer_types": ["linear_attention", "full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                     "max_position_embeddings": 4096, "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "patch_size": 2,
                       "spatial_merge_size": 1, "num_position_embeddings": 16},
    )
    backbone = Qwen3_5ForConditionalGeneration(config)
    base_state = copy.deepcopy(backbone.state_dict())
    head_config = dict(hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    model = upstream.ClefModel(backbone, upstream.JointSchemaHead(**head_config))
    targets = attach_lora(model, rank=2)
    assert all(".language_model.layers." in target for target in targets)
    assert any("linear_attn" in target for target in targets)
    assert any("self_attn" in target for target in targets)
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if ".visual." in name)
    rows = build_decisions(fixture_splits())[0]["train"][:1]
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    encoded, kept, excluded = encode_rows(upstream, processor, rows, max_length=4096)
    assert not excluded
    batch = upstream.collate_records(encoded, 0, torch.device("cpu"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial_fingerprints = trainable_fingerprints(model)
    model.train()
    loss = supervised_loss(model(batch), encoded, rows)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if "lora_" in name)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if name.startswith("head."))
    optimizer.step()
    changed_fingerprints = trainable_fingerprints(model)
    assert all(changed_fingerprints[kind]["sha256"] != initial_fingerprints[kind]["sha256"] for kind in ("lora", "head"))
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if "lora_" in name)
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name.startswith("head."))
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters() if not value.requires_grad)
    metrics, predictions = evaluate(model, upstream, processor, encoded, rows, torch.device("cpu"))
    assert metrics["questions"] == 1 and 0 <= metrics["accuracy"] <= 1
    model.language_model.save_pretrained(tmp_path)
    save_file(model.head.state_dict(), str(tmp_path / "joint_head.safetensors"))
    restored_base = Qwen3_5ForConditionalGeneration(config)
    restored_base.load_state_dict(base_state)
    restored = upstream.ClefModel(PeftModel.from_pretrained(restored_base, tmp_path), upstream.JointSchemaHead(**head_config))
    restored.head.load_state_dict(load_file(str(tmp_path / "joint_head.safetensors")))
    _, restored_predictions = evaluate(restored, upstream, processor, encoded, rows, torch.device("cpu"))
    assert restored_predictions == predictions
    assert compare_predictions(predictions, restored_predictions)["max_probability_difference"] == 0
    corrupted = copy.deepcopy(restored_predictions)
    first = next(iter(corrupted[0]["answers"]["research_answer"]["probabilities"]))
    corrupted[0]["answers"]["research_answer"]["probabilities"][first] += 0.02
    with pytest.raises(ValueError, match="probabilities differ"):
        compare_predictions(predictions, corrupted)
    # A single batch can contain native decisions with differing choice counts.
    mixed = copy.deepcopy(rows[0])
    mixed["id"] = "live-health-mixed-choice-count"
    mixed["questions"]["research_answer"]["criteria"].update(option_4="down", option_5="unknown")
    mixed_encoded, _, _ = encode_rows(upstream, processor, [mixed], max_length=4096)
    mixed_metrics, _ = evaluate(restored, upstream, processor, encoded + mixed_encoded,
                                rows + [mixed], torch.device("cpu"))
    assert mixed_metrics["chance_accuracy"] == pytest.approx((1 / 4 + 1 / 6) / 2)

"""CPU artifact/runtime tests; these never claim a real CUDA model was loaded.

Small real safetensors and the pinned native decision head exercise serialization
and inference. Production CUDA execution is rejected on CPU and remains a
separate paid-machine integration check.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_clef_cuda_release as cuda
from clef_live_tape import HTTP_URL, WS_URL, digest, live_decision_records, snapshot_freshness
from clef_research_data import DATASET_ID, MODEL_ID, MODEL_REVISION
from export_clef_mps_release import EXPANDED_DATASET_REVISION, EXPANDED_INPUT_ROWS, EXPANDED_INPUT_PARQUET_SHA256
from export_clef_release import _backbone_inventory, _sha256


def metadata():
    before = {kind: {"sha256": "1" * 64, "parameters": 4} for kind in ("lora", "head")}
    after = {kind: {"sha256": "2" * 64, "parameters": 4} for kind in ("lora", "head")}
    parity = {"matched": True, "records": 2, "max_probability_difference": 0.0, "tolerance": 0.0001}
    return {"fixture_scope": "small CPU gate artifact, not production trained weights",
        "status": "trained_and_reload_verified", "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": EXPANDED_DATASET_REVISION,
        "input_raw_row_counts": deepcopy(EXPANDED_INPUT_ROWS), "input_parquet_sha256": deepcopy(EXPANDED_INPUT_PARQUET_SHA256),
        "backend": "cuda", "device": "cuda", "cpu_offload": False, "cpu_fallback": False,
        "use_kernels": False,
        "vision_trained": False, "autoExecute": False, "run_mode": "full", "all_planned_steps_completed": True,
        "complete_selected_training_epochs": True, "selected_training_scope": "all context-eligible prepared rows",
        "security_filter": {"version": 1, "applied": True, "published_source_modified": False,
            "exclusion_counts": {"train_excluded_signing_byte_array": 11, "train_excluded_encoded_signing_literal": 3}},
        "lora": {"rank": 8, "text_layers": [60, 61, 62, 63]}, "prepared_split_counts": {"train": 42},
        "cohorts": {"train": {"selected": 40, "prepared": 42, "excluded": [{"id": "long-a"}, {"id": "long-b"}],
            "max_tokens": 2044,
            "sources": {"2605.12151v2.pdf": 9, "2606.08232v1.pdf": 11, "wss://clawd-ws.fly.dev/ws": 4, "other": 16}}},
        "trained_records": 40, "microbatch_size": 4, "gradient_accumulation": 1, "epochs": 1,
        "max_length": 2048, "capacity_probe": {"record_id": "longest", "tokens": 2044, "trained": True},
        "training_order_sha256": "3" * 64, "trained_record_ids_sha256": "3" * 64, "stage_manifest_sha256": "4" * 64,
        "pilot_trainable_fingerprints": deepcopy(before),
        "optimizer_steps": 10, "planned_steps": 10, "initial_trainable_fingerprints": before,
        "trained_trainable_fingerprints": after, "trainable_fingerprints": deepcopy(after),
        "gradient_evidence": {"lora": True, "head": True},
        "finite_trainables": {kind: {"finite": True, "parameters": 4} for kind in ("lora", "head")},
        "reload_verification": deepcopy(parity), "migration_verification": {**parity, "records": 16}}


@pytest.fixture
def small_release(tmp_path):
    import torch
    from safetensors.torch import save_file
    path = tmp_path / "small_cpu_gate_release"
    path.mkdir()
    save_file({"model.visual.weight": torch.ones(2, 2, dtype=torch.bfloat16),
               "model.language_model.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, str(path / "model.safetensors"))
    save_file({"projection.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(path / "joint_head.safetensors"))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "joint_head_config.json"):
        cuda._write(path / name, {"fixture": "CPU hash checks; no actual CUDA model"})
    (path / "joint_schema_model.py").write_text('"""CPU gate fixture; not a production model implementation."""\n')
    for name in {name for name in cuda.REQUIRED_RUNTIME if name.endswith(".py")} | {"research_expansion_artifacts.py"}:
        source = Path(cuda.__file__).with_name(name)
        (path / name).write_bytes(source.read_bytes())
    (path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")
    (path / "source_base_model_card.md").write_text("---\nlicense: apache-2.0\n---\nCPU source-card fixture.\n")
    cuda._write(path / "evaluation.json", {"scope": "small CPU artifact fixture"})
    snapshot = safe_snapshot(datetime.now(timezone.utc) - timedelta(seconds=2))
    cuda._write(path / "live-training-snapshot.json", snapshot)
    training = metadata()
    training["live_training_snapshot"] = {"file": "live-training-snapshot.json", "sha256": _sha256(path / "live-training-snapshot.json"), "captured_at": snapshot["captured_at"]}
    training.update(status="trained_and_standalone_reload_verified", standalone_reload_verification=training["reload_verification"])
    cuda._write(path / "training.json", training)
    backbone = _backbone_inventory(path)
    backbone.pop("tensor_names")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in path.iterdir() if p.is_file()}
    manifest = {"format": "clef-merged-release-v1", "status": "trained_merged_and_standalone_reload_verified",
        "standalone_backbone": True, "source": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "execution_backend": "cuda", "device": "cuda", "cpu_offload": False,
        "training": training, "backbone": backbone, "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()), "reload_verified": True,
        "merge_verification": training["reload_verification"], "reload_verification": training["reload_verification"],
        "cuda_standalone_load": {"backend": "cuda", "actual_cuda_parameters": True, "fresh_model_instance": True,
            "kernel_execution": {"implementation": "native_torch", "hub_kernel_downloads": False, "cuda_forward_calls": 4}}}
    cuda._write(path / "release.json", manifest)
    return path


def safe_snapshot(captured):
    snapshot = {"schema_version": "clawd-clef-live-v1", "observation_started_at": captured.isoformat(),
        "captured_at": captured.isoformat(), "sources": {"health": HTTP_URL, "websocket": WS_URL},
        "evidence_scope": "synthetic public observation fixture", "health": {"received_at": captured.isoformat(), "data": {"status": "ok"}},
        "frames": [], "transport": {"http": {"status": "observed"}, "websocket": {"status": "unavailable"}}}
    snapshot["snapshot_sha256"] = digest(snapshot)
    return snapshot


def inference_proof(path, now):
    captured = now - timedelta(seconds=2)
    completed = now - timedelta(seconds=1)
    snapshot = safe_snapshot(captured)
    row = live_decision_records(snapshot)[0]
    answers = {}
    for name, question in row["questions"].items():
        options = question["criteria"]
        choice = row["labels"][name]
        probabilities = {option: float(option == choice) for option in options}
        answers[name] = {"type": "choice", "choice": choice, "confidence": 1.0, "probabilities": probabilities}
    return {"model_path": str(path.resolve()), "model_revision": MODEL_REVISION, "dataset_revision": EXPANDED_DATASET_REVISION,
        "training_status": "trained_and_standalone_reload_verified", "backend": "cuda", "device": "cuda",
        "standalone_model_loaded": True, "release_manifest_sha256": _sha256(path / "release.json"),
        "source": "https://clawd-ws.fly.dev/", "captured_at": captured.isoformat(), "inference_completed_at": completed.isoformat(),
        "live_snapshot": snapshot, "freshness": snapshot_freshness(snapshot, now=completed),
        "responses": [{"id": row["id"], "usage": {"input_tokens": 100, "output_tokens": 0}, "answers": answers}], "excluded": []}


@pytest.mark.parametrize("change", [
    {"backend": "mps"}, {"cpu_offload": True}, {"cpu_fallback": True}, {"run_mode": "benchmark"},
    {"optimizer_steps": 9}, {"trained_records": 39}, {"complete_selected_training_epochs": False},
    {"cpu_offload": 0}, {"trained_record_ids_sha256": "f" * 64}, {"stage_manifest_sha256": "not-a-hash"},
    {"input_raw_row_counts": {"train": 83662}}, {"gradient_evidence": {"lora": True, "head": False}},
    {"lora": {"rank": 16, "text_layers": [60, 61, 62, 63]}},
    {"migration_verification": {"matched": True, "records": 8, "max_probability_difference": 0, "tolerance": 0.001}},
    {"reload_verification": {"matched": True, "records": 2, "max_probability_difference": 0.1, "tolerance": 0.001}},
])
def test_incomplete_or_changed_training_cannot_claim_cuda_full_export(change):
    training = metadata()
    training.update(change)
    with pytest.raises(ValueError):
        cuda._validate_training(training)


def test_cpu_model_never_becomes_cuda_export_evidence():
    import torch
    with pytest.raises(ValueError, match="Actual CUDA weights"):
        cuda._assert_cuda_model(torch.nn.Linear(2, 2))


def test_portable_metadata_removes_only_path_fields_and_binds_original_payload():
    original = metadata()
    original["base_model_name_or_path"] = "/Users/synthetic-fixture/base"
    original["provenance"] = {"local_dir": "/home/synthetic-fixture/private-inputs", "revision": MODEL_REVISION}
    safe = cuda._portable_training(original)
    assert "base_model_name_or_path" not in safe and "local_dir" not in safe["provenance"]
    assert safe["path_privacy"]["removed_path_fields"] == ["base_model_name_or_path", "provenance.local_dir"]
    assert len(safe["path_privacy"]["source_metadata_payload_sha256"]) == 64
    assert original["base_model_name_or_path"].startswith("/Users/")
    original["scope"] = "private diagnostics at /Users/synthetic-fixture/output"
    with pytest.raises(ValueError, match="outside a path field"):
        cuda._portable_training(original)


def test_model_card_labels_pending_reload_and_reports_actual_scope():
    training = metadata()
    metrics = {"eval": {"records": 8, "accuracy": 0.75, "nll": 0.4, "multiclass_brier": 0.2}}
    pending = cuda._model_card(training, metrics, reload_verified=False)
    complete = cuda._model_card(training, metrics, reload_verified=True)
    assert "fresh CUDA standalone reload remains pending" in pending
    assert "Fresh CUDA standalone reload verified" in complete
    assert "10 actual optimizer steps across 40" in complete
    assert "| eval | 8 | 0.75 | 0.4 | 0.2 |" in complete
    assert "arxiv.org/abs/2605.12151" in complete and "arxiv.org/abs/2606.08232" in complete
    assert "sys.modules[model.__class__.__module__]" in complete


def test_saved_kernel_helper_is_imported_from_artifact_and_configured_explicitly(tmp_path):
    (tmp_path / "train_clef_cuda.py").write_text(
        "def configure_local_kernels(enabled):\n"
        "    return {'fixture_scope': 'CPU import/configuration only', 'enabled': enabled}\n")
    assert cuda._configure_saved_kernels(tmp_path, True)["enabled"] is True
    assert cuda._configure_saved_kernels(tmp_path, False)["enabled"] is False


def test_completed_reload_cannot_switch_kernel_math(small_release):
    manifest = json.loads((small_release / "release.json").read_text())
    manifest["cuda_standalone_load"]["kernel_execution"]["implementation"] = "installed_fla"
    cuda._write(small_release / "release.json", manifest)
    with pytest.raises(ValueError, match="different kernel math"):
        cuda.verify_cuda_release(small_release)


def test_private_base_curation_uses_exact_source_files_without_operational_leaks(tmp_path):
    import torch
    from safetensors.torch import save_file
    base = tmp_path / "source"
    base.mkdir()
    save_file({"weight": torch.ones(2, 2)}, str(base / "model.safetensors"))
    (base / "LICENSE").write_text("Apache-2.0 fixture\n")
    cuda._write(base / "conversion_manifest.json", {"output_weight_files": {"model.safetensors": {}}, "preserved_sidecars": {"LICENSE": {}}})
    for name in (".saved-checkpoint-recovery.lock", "conversion_manifest.failed-before-rope-recovery.json", "unrelated-personal.json"):
        (base / name).write_text("fixture excluded from model export\n")
    (base / ".cache").mkdir()
    (base / ".cache" / "operational-metadata").write_text("fixture\n")
    before = {p.name: _sha256(p) for p in base.iterdir() if p.is_file()}
    with cuda._curated_base(base) as curated:
        assert {p.name for p in curated.iterdir()} == {"model.safetensors", "LICENSE", "conversion_manifest.json"}
        assert (curated / "model.safetensors").stat().st_ino == (base / "model.safetensors").stat().st_ino
    assert {p.name: _sha256(p) for p in base.iterdir() if p.is_file()} == before
    assert not curated.exists()


def test_curation_rejects_manifest_path_escape_and_symlink(tmp_path):
    base = tmp_path / "source"
    base.mkdir()
    cuda._write(base / "conversion_manifest.json", {"output_weight_files": {"../outside.safetensors": {}}})
    with pytest.raises(ValueError, match="unsafe"):
        with cuda._curated_base(base):
            pass
    cuda._write(base / "conversion_manifest.json", {"output_weight_files": {"model.safetensors": {}}})
    outside = tmp_path / "outside"
    outside.write_text("fixture\n")
    (base / "model.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="regular files"):
        with cuda._curated_base(base):
            pass


def test_real_small_artifact_inventory_detects_tampering_and_untracked_files(small_release):
    assert cuda.verify_cuda_release(small_release)["execution_backend"] == "cuda"
    rogue = small_release / ".cache"
    rogue.mkdir()
    with pytest.raises(ValueError, match="operational"):
        cuda.verify_cuda_release(small_release)
    rogue.rmdir()
    with (small_release / "joint_head.safetensors").open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="hash"):
        cuda.verify_cuda_release(small_release)


def test_pending_reload_is_inspectable_but_not_a_completed_standalone_release(small_release):
    manifest = json.loads((small_release / "release.json").read_text())
    manifest.update(status="exported_quantized_merge_pending_standalone_reload", reload_verified=False)
    cuda._write(small_release / "release.json", manifest)
    assert cuda.verify_cuda_release(small_release, require_reload=False)["reload_verified"] is False
    with pytest.raises(ValueError, match="fresh runtime reload"):
        cuda.verify_cuda_release(small_release)


@pytest.mark.parametrize("mutation", ["wrong_backend", "stale", "fake_id", "nonfinite", "wrong_options", "signing", "manifest"])
def test_fresh_native_proof_rejects_invalid_evidence(small_release, tmp_path, mutation):
    now = datetime.now(timezone.utc)
    proof = inference_proof(small_release, now)
    if mutation == "wrong_backend": proof["backend"] = "mps"
    elif mutation == "stale": proof["inference_completed_at"] = (now + timedelta(seconds=40)).isoformat()
    elif mutation == "fake_id": proof["responses"][0]["id"] = "uncaptured-observation"
    elif mutation == "nonfinite": next(iter(proof["responses"][0]["answers"].values()))["probabilities"]["extra"] = float("nan")
    elif mutation == "wrong_options": next(iter(proof["responses"][0]["answers"].values()))["probabilities"] = {"fake-a": 1.0, "fake-b": 0.0}
    elif mutation == "signing": proof["scope"] = "signing_seed = " + str(list(range(32)))
    elif mutation == "manifest": proof["release_manifest_sha256"] = "0" * 64
    p = tmp_path / "proof.json"
    p.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        cuda.verify_live_inference(p, small_release, now=now)


def test_actual_snapshot_age_and_native_options_are_verified(small_release, tmp_path):
    now = datetime.now(timezone.utc)
    proof = inference_proof(small_release, now)
    p = tmp_path / "proof.json"
    cuda._write(p, proof)
    assert cuda.verify_live_inference(p, small_release, now=now) == proof
    proof["freshness"]["snapshot_age_seconds"] = 0.0
    cuda._write(p, proof)
    with pytest.raises(ValueError, match="inconsistent"):
        cuda.verify_live_inference(p, small_release, now=now)


def test_real_cpu_nf4_nested_state_roundtrip_uses_standard_safetensors(tmp_path):
    import bitsandbytes as bnb
    import torch
    from safetensors.torch import load_file, save_file
    source = torch.linspace(-1, 1, 256).reshape(16, 16)
    packed, state = bnb.functional.quantize_4bit(source, quant_type="nf4", compress_statistics=True)
    original = bnb.functional.dequantize_4bit(packed, quant_state=state)
    p = tmp_path / "tiny_cpu_nf4.safetensors"
    save_file({"weight": packed, **state.as_dict(packed=True)}, str(p))
    saved = load_file(p)
    restored = bnb.nn.Params4bit.from_prequantized(saved.pop("weight"), saved, device="cpu")
    actual = bnb.functional.dequantize_4bit(restored, quant_state=restored.quant_state)
    assert restored.quant_state.quant_type == "nf4" and restored.quant_state.nested
    assert bool(torch.isfinite(actual).all()) and torch.equal(original, actual)


def test_real_cpu_native_head_save_reload_preserves_typed_predictions(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    from test_clef_mps_quantized import native_module, processor_fixture
    module, processor = native_module(), processor_fixture()
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 16)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.output = Text(), torch.nn.Linear(16, 256, bias=False)
        def get_output_embeddings(self):
            return self.output
    torch.manual_seed(42)
    config = {"hidden_size": 16, "width": 8, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 16, "dropout": 0.0}
    model = module.ClefModel(Backbone(), module.JointSchemaHead(**config))
    row = {"id": "synthetic-native-record", "state": "The observed service status is ok.",
           "questions": {"status": {"type": "choice", "instructions": "Read the observed status.", "criteria": {"ok": "ok", "down": "down"}}},
           "labels": {"status": "ok"}, "provenance": {"source": "synthetic CPU runtime test"}}
    before = cuda._native_responses(model, module, processor, [row], 2048, device="cpu")
    p = tmp_path / "native_cpu_head.safetensors"
    save_file(model.head.state_dict(), str(p))
    model.head = module.JointSchemaHead(**config)
    model.head.load_state_dict(load_file(p), strict=True)
    after = cuda._native_responses(model, module, processor, [row], 2048, device="cpu")
    assert before == after and before["responses"][0]["usage"]["output_tokens"] == 0
    assert before["responses"][0]["answers"]["status"]["type"] == "choice"
    with pytest.raises(ValueError, match="context limit"):
        cuda._native_responses(model, module, processor, [row], 1, device="cpu")
    unsafe = deepcopy(row)
    unsafe["state"] = "signing_seed = " + str(list(range(32)))
    with pytest.raises(ValueError, match="signing material"):
        cuda._native_responses(model, module, processor, [unsafe], 2048, device="cpu")

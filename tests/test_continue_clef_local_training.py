"""CPU scheduling/identity/artifact tests; never download or load model weights.

Fake process backends exercise control flow, not numerical model evidence.
Separate actual MPS tests cover native head/LoRA kernels and saved-weight reload.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continue_clef_local_training import (Coordinator, GIB, PipelineFailure, ROOT,
    build_config, conversion_verified, file_sha, identity_matches, persist, read_json, status,
    verify_live_runtime, verify_phase)
from train_clef_local import (DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, MODEL_ID, MODEL_REVISION, refresh_adapter_manifest)


def test_configuration_preserves_venv_interpreter_symlink(tmp_path):
    from argparse import Namespace
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    args = Namespace(**{name: tmp_path / name for name in
                        ("base", "pilot", "full", "release", "runtime_output")},
                     python=venv_python, converter_pid=77, converter_start="known start",
                     converter_command="known command", min_conversion_free_gib=40,
                     max_length=2048)
    config = build_config(args)
    assert config["python"] == str(venv_python.absolute())
    assert config["python"] != str(venv_python.resolve())
    coordinator = Coordinator(config, tmp_path / "state.json")
    assert coordinator.command("pilot")[0] == str(venv_python.absolute())


def metadata(steps):
    return {"optimizer_steps": steps, "planned_steps": steps, "status": "trained_and_reload_verified",
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
        "cpu_fallback": False, "security_filter": {"applied": True, "version": 1}}


def setup_pipeline(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    converter = {"pid": 77, "started": "Thu Oct 1 13:19:15 2026", "state": "T",
        "command": shlex.join(["/python", "-u", str(ROOT / "scripts/convert_clef_mps.py"), "--execute", "--output", str(base)])}
    config = {"base": str(base), "pilot": str(tmp_path / "pilot"), "full": str(tmp_path / "full"),
        "release": str(tmp_path / "release"), "runtime_output": str(tmp_path / "live.json"),
        "python": "/python", "max_length": 2048,
        "converter": {key: converter[key] for key in ("pid", "started", "command")},
        "conversion_required_free_bytes": 40 * GIB}
    persist(base / "conversion_manifest.json", {"status": "loading", "completed_source_shards": [1, 2]})
    return config, converter


def complete_base(config):
    persist(Path(config["base"]) / "conversion_manifest.json", {"status": "complete",
        "source": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "standalone_weights_saved": True, "source_head_unchanged": True,
        "reload_verified": True, "reload_probe": {"finite": True}})


class FakeBackend:
    def __init__(self, config, converter):
        self.config = config
        self.processes = {converter["pid"]: dict(converter)}
        self.free = 45 * GIB
        self.spawned, self.resumed, self.paused, self.verified = [], [], [], []
        self.codes = {}

    def snapshot(self, pid):
        return self.processes.get(pid)

    def free_disk(self, path):
        return self.free

    def reserve(self, base):
        return {"required_free_bytes": 18 * GIB}

    def resume(self, identity):
        assert identity_matches(identity, self.snapshot(identity["pid"]))
        self.resumed.append(identity["pid"])
        self.processes[identity["pid"]]["state"] = "R"

    def spawn(self, command, log):
        pid = 100 + len(self.spawned)
        child = {"pid": pid, "started": "unit scheduling fixture", "state": "R", "command": shlex.join(command), "log": str(log)}
        self.spawned.append(command)
        self.processes[pid] = child
        return child

    def pause(self, identity):
        self.paused.append(identity["pid"])
        self.processes[identity["pid"]]["state"] = "T"

    def returncode(self, child):
        return self.codes.get(child["pid"])

    def finish(self, child, code=0):
        self.processes.pop(child["pid"], None)
        self.codes[child["pid"]] = code

    def verify_phase(self, output, phase, base, max_length):
        self.verified.append(phase)
        return {"optimizer_steps": 16 if phase == "pilot" else 130, "trained_records": 16 if phase == "pilot" else 130,
                "max_length": max_length, "trainable_fingerprints": {"head": "unit head", "lora": "unit lora"}}

    def verify_release(self, output):
        self.verified.append("release")
        return read_json(output / "release.json")

    verify_runtime = staticmethod(verify_live_runtime)


def finish_training(coordinator, backend, phase, steps):
    output = Path(coordinator.config[phase])
    output.mkdir(exist_ok=True)
    persist(output / "training.json", metadata(steps))
    backend.finish(coordinator.state["children"][phase])


def advance_to_runtime(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    backend = FakeBackend(config, converter)
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    assert coordinator.tick() is False
    complete_base(config)
    backend.processes.pop(converter["pid"])
    assert coordinator.tick() is False  # launch actual pilot command
    finish_training(coordinator, backend, "pilot", 16)
    assert coordinator.tick() is False  # verify saved pilot
    assert coordinator.tick() is False  # launch full epoch
    finish_training(coordinator, backend, "full", 130)
    assert coordinator.tick() is False  # verify full adapter
    release = Path(config["release"])
    release.mkdir()
    persist(release / "release.json", {"training": {"run_mode": "full", "optimizer_steps": 130,
        "trained_trainable_fingerprints": {"head": "unit head", "lora": "unit lora"}},
        "reload_verification": {"matched": True}})
    assert coordinator.tick() is False  # strict release verify, runtime launch
    return coordinator, backend


def write_runtime(coordinator, stale=False):
    now = datetime.now(timezone.utc).isoformat()
    path = Path(coordinator.state["runtime_output"])
    persist(path, {"responses": [{"id": "observed-status", "usage": {"input_tokens": 64, "output_tokens": 0},
        "answers": {"health": {"type": "choice", "choice": "observed", "confidence": 0.9,
                                "probabilities": {"observed": 0.9, "missing": 0.1}}}}],
        "standalone_model_loaded": True, "backend": "mps", "device": "mps",
        "model_path": coordinator.config["release"], "source": "https://clawd-ws.fly.dev/",
        "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
        "training_status": "trained_and_standalone_reload_verified",
        "release_manifest_sha256": file_sha(Path(coordinator.config["release"]) / "release.json"),
        "freshness": {"stale": stale, "snapshot_age_seconds": 0.0, "observation_age_seconds": [0.0]},
        "captured_at": now, "inference_completed_at": now})
    return path


def test_waits_for_40gib_then_resumes_exact_converter_once_without_starting_training(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    backend = FakeBackend(config, converter)
    backend.free = 19 * GIB
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    assert coordinator.tick() is False
    assert coordinator.state["stage"] == "conversion_waiting_for_disk_headroom"
    assert coordinator.state["training_started"] is False and coordinator.state["optimizer_steps_27b"] == 0
    assert not backend.resumed and not backend.spawned
    backend.free = 40 * GIB
    assert coordinator.tick() is False
    assert backend.resumed == [77] and not backend.spawned
    assert coordinator.tick() is False
    assert backend.resumed == [77]
    restored = Coordinator(config, tmp_path / "state.json", backend)
    assert restored.tick() is False and backend.resumed == [77]


@pytest.mark.parametrize("changed", ["pid", "started", "command"])
def test_converter_identity_mismatch_never_signals_or_duplicates(tmp_path, changed):
    config, converter = setup_pipeline(tmp_path)
    backend = FakeBackend(config, converter)
    backend.processes[77][changed] = 78 if changed == "pid" else "different process"
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    with pytest.raises(PipelineFailure, match="identity"):
        coordinator.tick()
    assert not backend.resumed and not backend.spawned


def test_dead_or_failed_converter_is_terminal_and_never_restarted(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    backend = FakeBackend(config, converter)
    backend.processes.clear()
    with pytest.raises(PipelineFailure, match="cannot be restarted"):
        Coordinator(config, tmp_path / "state.json", backend).tick()
    with pytest.raises(PipelineFailure, match="will not be restarted"):
        conversion_verified({"status": "failed"})
    assert not backend.spawned


def test_ordered_pilot_full_standalone_and_fresh_runtime_completion(tmp_path):
    coordinator, backend = advance_to_runtime(tmp_path)
    pilot, full, runtime = backend.spawned
    assert pilot[pilot.index("--max-length") + 1] == full[full.index("--max-length") + 1] == "2048"
    assert pilot[pilot.index("--max-steps") + 1] == "16"
    assert "--max-steps" not in full and "--train-records" not in full
    assert full[full.index("--checkpoint-steps") + 1] == "128"
    assert full[full.index("--init-adapter") + 1] == str(Path(coordinator.config["pilot"]) / "adapter")
    assert "run_clef_local.py" in runtime[2]
    assert coordinator.state["stage"] == "runtime_process_running"
    assert coordinator.state["training_started"] is True and coordinator.state["optimizer_steps_27b"] == 146
    assert backend.verified == ["pilot", "full", "release"]
    write_runtime(coordinator)
    backend.finish(coordinator.state["children"]["runtime"])
    assert coordinator.tick() is True
    assert coordinator.state["stage"] == "full_training_standalone_and_live_runtime_verified"
    assert coordinator.state["live_runtime"]["responses"] == 1
    assert backend.verified == ["pilot", "full", "release", "release"]


def test_subprocess_start_and_partial_metadata_never_claim_optimizer_progress(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    complete_base(config)
    backend = FakeBackend(config, converter)
    backend.processes.clear()
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    coordinator.tick()
    assert coordinator.state["stage"] == "pilot_process_running"
    assert coordinator.state["training_started"] is False
    assert coordinator.state["phase_progress"]["full"]["optimizer_steps"] == 0
    pilot = Path(config["pilot"])
    pilot.mkdir()
    (pilot / "training.json").write_text('{"optimizer_steps":')
    coordinator.tick()
    assert coordinator.state["training_started"] is False
    persist(pilot / "training.json", metadata(1))
    before = coordinator.path.read_bytes()
    observed = status(coordinator.path)
    assert observed["optimizer_steps_27b"] == 1 and observed["training_started"] is True
    assert observed["full_training_started"] is False
    assert coordinator.path.read_bytes() == before
    restored = Coordinator(config, coordinator.path, backend)
    restored.tick()
    assert len(backend.spawned) == 1  # no second trainer on coordinator restart


def test_interrupt_pauses_only_owned_trainer_and_restart_resumes_same_identity(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    complete_base(config)
    backend = FakeBackend(config, converter)
    backend.processes.clear()
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    coordinator.tick()
    backend.processes[77] = dict(converter)  # user-owned converter is preserved
    coordinator.stop()
    assert backend.paused == [100] and backend.processes[77]["state"] == "T"
    backend.processes.pop(77)
    restored = Coordinator(config, coordinator.path, backend)
    restored.tick()
    assert backend.resumed == [100] and len(backend.spawned) == 1


def child_phase_fixture(tmp_path, phase):
    if phase == "runtime":
        coordinator, backend = advance_to_runtime(tmp_path)
        write_runtime(coordinator)
    else:
        config, converter = setup_pipeline(tmp_path)
        complete_base(config)
        backend = FakeBackend(config, converter)
        backend.processes.clear()
        coordinator = Coordinator(config, tmp_path / "state.json", backend)
        assert coordinator.tick() is False
        output = Path(config["pilot"])
        output.mkdir()
        persist(output / "training.json", metadata(16))
    backend.resumed.clear()
    backend.paused.clear()
    return coordinator, backend, coordinator.state["children"][phase]


@pytest.mark.parametrize("phase", ["pilot", "runtime"])
@pytest.mark.parametrize("owned", [True, False])
def test_zombie_command_rewrite_advances_only_after_saved_artifact_verification(tmp_path, phase, owned):
    coordinator, backend, child = child_phase_fixture(tmp_path, phase)
    backend.processes[child["pid"]] = {**child, "state": "Zs", "command": "(Python)"}
    if owned:
        backend.codes[child["pid"]] = 0
    events = []
    old_poll, old_snapshot = backend.returncode, backend.snapshot

    def poll(identity):
        events.append(("poll", identity["pid"]))
        return old_poll(identity)

    def snapshot(pid):
        events.append(("snapshot", pid))
        return old_snapshot(pid)
    backend.returncode, backend.snapshot = poll, snapshot
    expected_spawns = len(backend.spawned)
    result = coordinator.tick()
    child_events = [event for event in events if event[1] == child["pid"]]
    assert child_events[0] == ("poll", child["pid"])
    if owned:
        assert child_events == [("poll", child["pid"])]
    else:
        assert child_events == [("poll", child["pid"]), ("snapshot", child["pid"])]
    assert not backend.resumed and not backend.paused and len(backend.spawned) == expected_spawns
    if phase == "pilot":
        assert result is False and "pilot" in coordinator.state["verified_phases"]
        assert backend.verified == ["pilot"]
        assert coordinator.state["stage"] == "pilot_trained_and_reload_verified"
    else:
        assert result is True and coordinator.state["stage"] == "full_training_standalone_and_live_runtime_verified"
        assert backend.verified[-1] == "release"


@pytest.mark.parametrize("phase", ["pilot", "runtime"])
def test_nonzero_owned_exit_is_failure_even_with_zombie_and_saved_outputs(tmp_path, phase):
    coordinator, backend, child = child_phase_fixture(tmp_path, phase)
    backend.processes[child["pid"]] = {**child, "state": "Z", "command": "(Python)"}
    backend.codes[child["pid"]] = 7
    old_verified, old_spawns = list(backend.verified), len(backend.spawned)
    with pytest.raises(PipelineFailure, match="code 7"):
        coordinator.tick()
    assert backend.verified == old_verified and len(backend.spawned) == old_spawns
    assert not backend.resumed and not backend.paused


@pytest.mark.parametrize("phase", ["pilot", "runtime"])
def test_confirmed_reaped_child_does_not_inspect_or_signal_reused_pid(tmp_path, phase):
    coordinator, backend, child = child_phase_fixture(tmp_path, phase)
    backend.codes[child["pid"]] = 0
    backend.processes[child["pid"]] = {**child, "started": "new unrelated process", "state": "R", "command": "unrelated-command"}
    old_snapshot = backend.snapshot

    def snapshot(pid):
        if pid == child["pid"]:
            pytest.fail("A reaped child's PID must not be inspected during phase completion")
        return old_snapshot(pid)
    backend.snapshot = snapshot
    coordinator.tick()
    assert not backend.resumed and not backend.paused
    backend.snapshot = old_snapshot
    coordinator.stop()
    assert not backend.resumed and not backend.paused
    assert backend.processes[child["pid"]]["command"] == "unrelated-command"


@pytest.mark.parametrize("phase", ["pilot", "runtime"])
@pytest.mark.parametrize("state,changed", [("R", "command"), ("T", "command"), ("R", "pid"),
                                           ("R", "started"), ("Z", "pid"), ("Z", "started")])
def test_unknown_live_identity_and_unmatched_zombies_block_without_signals(tmp_path, phase, state, changed):
    coordinator, backend, child = child_phase_fixture(tmp_path, phase)
    observed = {**child, "state": state, "command": "(Python)" if state == "Z" else child["command"]}
    observed[changed] = child["pid"] + 1 if changed == "pid" else "different process identity"
    backend.processes[child["pid"]] = observed
    old_verified, old_spawns = list(backend.verified), len(backend.spawned)
    with pytest.raises(PipelineFailure, match="identity"):
        coordinator.tick()
    assert backend.verified == old_verified and len(backend.spawned) == old_spawns
    assert not backend.resumed and not backend.paused


def test_completed_pilot_can_recover_terminal_handoff_without_retraining(tmp_path):
    coordinator, backend, child = child_phase_fixture(tmp_path, "pilot")
    backend.finish(child)
    coordinator.state.update(stage="terminal_failure", failure_stage="pilot", action_needed="old zombie command mismatch")
    coordinator.save()
    restored = Coordinator(coordinator.config, coordinator.path, backend)
    assert restored.tick() is False
    assert restored.state["stage"] == "pilot_trained_and_reload_verified" and restored.state["action_needed"] is None
    assert backend.verified == ["pilot"] and len(backend.spawned) == 1
    assert restored.tick() is False
    assert len(backend.spawned) == 2 and "--init-adapter" in backend.spawned[-1]
    assert backend.spawned[-1][backend.spawned[-1].index("--mode") + 1] == "full"


def test_stale_runtime_fails_then_explicit_retry_preserves_output_and_never_retrains(tmp_path):
    coordinator, backend = advance_to_runtime(tmp_path)
    original = write_runtime(coordinator, stale=True)
    old_bytes = original.read_bytes()
    backend.finish(coordinator.state["children"]["runtime"])
    with pytest.raises(ValueError, match="stale"):
        coordinator.tick()
    assert len(backend.spawned) == 3
    coordinator.state.update(stage="terminal_failure", failure_stage="runtime")
    coordinator.save()
    retried = Coordinator(coordinator.config, coordinator.path, backend, retry_runtime=True)
    assert retried.tick() is False
    assert len(backend.spawned) == 4 and "run_clef_local.py" in backend.spawned[-1][2]
    assert retried.state["runtime_output"] != str(original)
    assert original.read_bytes() == old_bytes
    write_runtime(retried)
    backend.finish(retried.state["children"]["runtime"])
    assert retried.tick() is True
    assert sum("train_clef_local.py" in command[2] for command in backend.spawned) == 2


def test_incomplete_preexisting_run_and_unrecorded_launch_are_never_overwritten(tmp_path):
    config, converter = setup_pipeline(tmp_path)
    complete_base(config)
    backend = FakeBackend(config, converter)
    backend.processes.clear()
    pilot = Path(config["pilot"])
    pilot.mkdir()
    preserved = pilot / "user-file"
    preserved.write_text("keep")
    def reject(*args):
        raise PipelineFailure("incomplete existing artifact")
    backend.verify_phase = reject
    coordinator = Coordinator(config, tmp_path / "state.json", backend)
    with pytest.raises(PipelineFailure, match="incomplete"):
        coordinator.tick()
    assert preserved.read_text() == "keep" and not backend.spawned
    other = tmp_path / "other"
    other.mkdir()
    config = {**config, "pilot": str(other)}
    coordinator = Coordinator(config, tmp_path / "other-state.json", backend)
    coordinator.state["stage"] = "pilot_launch_requested"
    with pytest.raises(PipelineFailure, match="not durably recorded"):
        coordinator.tick()
    assert not backend.spawned


def phase_artifact_fixture(tmp_path):
    """Structural evidence fixture for verifier guards, not real model weights."""
    from clef_research_data import decision_prompt_hash
    base, output = tmp_path / "base", tmp_path / "pilot"
    base.mkdir()
    output.mkdir()
    persist(base / "conversion_manifest.json", {"fixture": True})
    persist(base / "config.json", {"text_config": {"num_hidden_layers": 64}})
    rows = []
    for index in range(128):
        source = "live" if index == 0 else "2605.12151" if index == 1 else "2606.08232" if index == 2 else "research"
        row = {"id": str(index), "state": "Public observed fields.",
            "questions": {"answer": {"type": "choice", "instructions": "Select observed response.", "criteria": {"a": "wait", "b": "act"}}},
            "labels": {"answer": "a"}, "provenance": {"source": source, "source_type": "live_tape_observation" if index == 0 else "research"}}
        row["provenance"]["prompt_hash"] = decision_prompt_hash(row)
        rows.append(row)
    path = output / "cohorts/train.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    changed = {kind: {"sha256": char * 64, "parameters": 1} for kind, char in (("head", "b"), ("lora", "c"))}
    training = {**metadata(16), "dataset": DATASET_ID, "input_parquet_sha256": INPUT_PARQUET_SHA256,
        "input_raw_row_counts": INPUT_RAW_ROW_COUNTS, "run_mode": "pilot", "vision_trained": False,
        "all_planned_steps_completed": True, "max_length": 2048, "gradient_accumulation": 1, "epochs": 1,
        "local_base_manifest_sha256": file_sha(base / "conversion_manifest.json"),
        "reload_verification": {"matched": True, "records": 4, "max_probability_difference": 0.0, "tolerance": 0.001},
        "initial_trainable_fingerprints": {kind: {"sha256": "a" * 64, "parameters": 1} for kind in ("head", "lora")},
        "trained_trainable_fingerprints": changed, "trainable_fingerprints": changed,
        "finite_trainables": {kind: {"finite": True, "parameters": 1} for kind in ("head", "lora")},
        "gradient_evidence": {"head": True, "lora": True}, "lora": {"rank": 8, "text_layers": [60, 61, 62, 63]},
        "cohorts": {"train": {"selected": 128, "sha256": file_sha(path), "max_tokens": 2048}, "eval": {"selected": 8}, "test": {"selected": 8}},
        "trained_records": 16, "trained_required_record_ids": ["0", "1", "2"],
        "capacity_probe": {"record_id": "127", "tokens": 2048, "trained": True}}
    adapter = output / "adapter"
    adapter.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "joint_head.safetensors", "joint_head_config.json",
                 "joint_schema_model.py", "tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        (adapter / name).write_text("structural verifier fixture")
    persist(adapter / "training.json", training)
    persist(output / "training.json", training)
    refresh_adapter_manifest(adapter)
    return base, output, training


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(optimizer_steps=0),
    lambda m: m.update(dataset_revision="0" * 40),
    lambda m: m["trained_trainable_fingerprints"]["head"].update(sha256="a" * 64),
    lambda m: m["gradient_evidence"].update(lora=False),
    lambda m: m["capacity_probe"].update(trained=False),
    lambda m: m.update(trained_required_record_ids=[]),
])
def test_phase_verifier_rejects_missing_real_progress_updates_capacity_and_sources(tmp_path, mutation):
    base, output, training = phase_artifact_fixture(tmp_path)
    assert verify_phase(output, "pilot", base, 2048)["optimizer_steps"] == 16
    mutation(training)
    persist(output / "training.json", training)
    persist(output / "adapter/training.json", training)
    refresh_adapter_manifest(output / "adapter")
    with pytest.raises(PipelineFailure):
        verify_phase(output, "pilot", base, 2048)


def test_phase_verifier_detects_changed_actual_cohort_and_head_payload(tmp_path):
    base, output, training = phase_artifact_fixture(tmp_path)
    with (output / "cohorts/train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(PipelineFailure, match="cohort changed"):
        verify_phase(output, "pilot", base, 2048)
    (output / "adapter/joint_head.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="joint_head.safetensors"):
        verify_phase(output, "pilot", base, 2048)


def test_full_context_cap_cannot_exceed_verified_pilot(tmp_path):
    coordinator, backend = advance_to_runtime(tmp_path)
    coordinator.state["children"].pop("full", None)
    coordinator.state["verified_phases"].pop("full", None)
    coordinator.state["verified_phases"]["pilot"]["max_length"] = 1024
    # Select a new empty full/release location without deleting any prior run.
    coordinator.config["full"] = str(tmp_path / "full-new")
    coordinator.config["release"] = str(tmp_path / "release-new")
    with pytest.raises(PipelineFailure, match="exceeds"):
        coordinator.tick()
    assert len(backend.spawned) == 3

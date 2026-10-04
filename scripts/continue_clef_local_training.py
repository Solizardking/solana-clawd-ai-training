#!/usr/bin/env python3
"""Continue the known local Clef conversion, measured pilot, then full training.

This CPU coordinator never loads model weights, downloads source shards, starts
a second converter, publishes, or deletes user files. It resumes the recorded
stopped converter once after disk headroom is available. Real optimizer evidence
and fresh reload verification must precede each next stage. Ctrl-C pauses only
this coordinator's own trainer child; rerunning resumes that exact process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

from train_clef_local import (DATASET_ID, DATASET_REVISION, INPUT_PARQUET_SHA256,
    INPUT_RAW_ROW_COUNTS, MODEL_ID, MODEL_REVISION, file_sha, guard_model_input,
    source_requirements, verify_adapter_manifest)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
CONVERTER_PID = 75712
CONVERTER_START = "Thu Oct 1 13:19:15 2026"
CONVERTER_COMMAND = (
    "/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/Python.framework/Versions/3.14/"
    "Resources/Python.app/Contents/MacOS/Python -u scripts/convert_clef_mps.py "
    "--execute --output local/clef-27b-mps-nf4"
)


class PipelineFailure(RuntimeError):
    pass


class CoordinatorStop(BaseException):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # A live trainer writes telemetry between steps. A partial read must
        # not turn into a false failure or reset already-observed step evidence.
        return None


def persist(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def process_snapshot(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "command="],
                            capture_output=True, text=True, timeout=10,
                            env={**os.environ, "LC_ALL": "C"})
    if result.returncode or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split(None, 6)
    if len(parts) != 7:
        raise PipelineFailure("Cannot safely identify the recorded process")
    return {"pid": int(pid), "started": " ".join(parts[:5]), "state": parts[5], "command": parts[6]}


def identity_matches(expected, current):
    return current is not None and all(expected.get(key) == current.get(key) for key in ("pid", "started", "command"))


def converter_target_matches(identity, base):
    args = shlex.split(identity["command"])
    try:
        index = next(index for index, value in enumerate(args) if Path(value).name == "convert_clef_mps.py")
        script = Path(args[index])
        script = (ROOT / script).resolve() if not script.is_absolute() else script.resolve()
        target = Path(args[args.index("--output") + 1])
        target = (ROOT / target).resolve() if not target.is_absolute() else target.resolve()
    except (StopIteration, ValueError, IndexError):
        return False
    return script == ROOT / "scripts/convert_clef_mps.py" and target == Path(base).resolve() and "--execute" in args


def conversion_verified(manifest):
    if not manifest:
        return False
    if manifest.get("status") == "failed":
        raise PipelineFailure("Original converter failed; review its log and preserve the partial directory. It will not be restarted automatically.")
    if manifest.get("status") != "complete":
        return False
    if (manifest.get("source") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}
        or manifest.get("standalone_weights_saved") is not True
        or manifest.get("source_head_unchanged") is not True
        or manifest.get("reload_verified") is not True
        or manifest.get("reload_probe", {}).get("finite") is not True):
        raise PipelineFailure("Completed base lacks pinned native finite/standalone reload evidence")
    return True


def training_reserve(base):
    """Count actual affected last-four-layer shards plus artifacts/swap reserve."""
    base = Path(base)
    config = read_json(base / "config.json") or {}
    index = read_json(base / "model.safetensors.index.json") or {}
    layers = config.get("text_config", {}).get("num_hidden_layers")
    if not isinstance(layers, int) or layers < 4 or not index.get("weight_map"):
        raise PipelineFailure("Converted base config/index is missing; no training reserve can be established")
    affected = set()
    for name, filename in index["weight_map"].items():
        match = re.search(r"\.language_model\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) >= layers - 4:
            if Path(filename).name != filename:
                raise PipelineFailure("Unsafe converted shard reference")
            affected.add(filename)
    if not affected:
        raise PipelineFailure("Converted index has no final-four-layer export shards")
    changed = sum((base / name).stat().st_size for name in affected)
    return {"affected_shard_bytes": changed, "retained_artifact_reserve_bytes": 6 * GIB,
            "swap_headroom_bytes": 8 * GIB,
            "required_free_bytes": max(16 * GIB, changed + 14 * GIB)}


def _parity_valid(value, tolerance):
    difference, measured = value.get("max_probability_difference"), value.get("tolerance")
    return (value.get("matched") is True and type(value.get("records")) is int and value["records"] > 0
        and isinstance(difference, (int, float)) and math.isfinite(difference) and difference >= 0
        and isinstance(measured, (int, float)) and math.isfinite(measured) and 0 < measured <= tolerance
        and difference <= measured)


def verify_phase(output, phase, base, max_length=2048):
    """Hash the actual saved files and require full numerical training evidence."""
    output, base = Path(output), Path(base)
    verify_adapter_manifest(output / "adapter")
    metadata = read_json(output / "adapter/training.json")
    if metadata is None or metadata != read_json(output / "training.json"):
        raise PipelineFailure("Adapter and run training metadata are incomplete or inconsistent")
    expected = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
                "input_parquet_sha256": INPUT_PARQUET_SHA256, "input_raw_row_counts": INPUT_RAW_ROW_COUNTS,
                "status": "trained_and_reload_verified", "run_mode": phase,
                "cpu_fallback": False, "vision_trained": False, "all_planned_steps_completed": True,
                "max_length": max_length, "gradient_accumulation": 1, "epochs": 1}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise PipelineFailure("Completed training lacks the requested source/backend/full-step evidence")
    security = metadata.get("security_filter", {})
    if security.get("applied") is not True or security.get("version") != 1:
        raise PipelineFailure("Completed training lacks the input security exclusion audit")
    if metadata.get("local_base_manifest_sha256") != file_sha(base / "conversion_manifest.json"):
        raise PipelineFailure("Completed training refers to a different original converted base")
    if type(metadata.get("optimizer_steps")) is not int or metadata["optimizer_steps"] < 1 or metadata["optimizer_steps"] != metadata.get("planned_steps"):
        raise PipelineFailure("Queued or partial optimizer progress is not a completed phase")
    if not _parity_valid(metadata.get("reload_verification", {}), 0.001):
        raise PipelineFailure("Completed phase lacks measured fresh saved adapter/head reload parity")
    for kind in ("lora", "head"):
        before = metadata.get("initial_trainable_fingerprints", {}).get(kind, {})
        after = metadata.get("trained_trainable_fingerprints", {}).get(kind, {})
        finite = metadata.get("finite_trainables", {}).get(kind, {})
        if (metadata.get("gradient_evidence", {}).get(kind) is not True
            or not re.fullmatch(r"[0-9a-f]{64}", before.get("sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", after.get("sha256", ""))
            or before["sha256"] == after["sha256"]
            or after != metadata.get("trainable_fingerprints", {}).get(kind)
            or finite.get("finite") is not True or finite.get("parameters", 0) < 1):
            raise PipelineFailure("Both LoRA and native head must have finite gradient/byte-update/reload evidence")
    lora = metadata.get("lora", {})
    count = (read_json(base / "config.json") or {}).get("text_config", {}).get("num_hidden_layers", 0)
    if lora.get("rank") != 8 or lora.get("text_layers") != list(range(count - 4, count)):
        raise PipelineFailure("Completed training is outside the requested final-four-layer/rank-eight scope")
    cohorts = metadata.get("cohorts", {})
    train_path = output / "cohorts/train.jsonl"
    if file_sha(train_path) != cohorts.get("train", {}).get("sha256"):
        raise PipelineFailure("Selected native training cohort changed after training")
    with train_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != cohorts["train"].get("selected"):
        raise PipelineFailure("Actual selected native record count is inconsistent")
    for row in rows:
        guard_model_input(row)
    required = [rows[index]["id"] for index in source_requirements(rows)]
    if metadata.get("trained_required_record_ids") != required or metadata.get("trained_records", 0) < len(required):
        raise PipelineFailure("Actual training did not cover every required live and cited-paper record")
    if phase == "pilot":
        if metadata["optimizer_steps"] != 16 or len(rows) != 128 or any(cohorts.get(split, {}).get("selected") != 8 for split in ("eval", "test")):
            raise PipelineFailure("Pilot must complete exactly sixteen steps with the requested 128/8/8 cohorts")
        capacity = metadata.get("capacity_probe", {})
        if (capacity.get("trained") is not True or capacity.get("record_id") not in {row["id"] for row in rows}
            or type(capacity.get("tokens")) is not int or capacity["tokens"] != cohorts["train"].get("max_tokens")
            or capacity["tokens"] > max_length):
            raise PipelineFailure("Pilot did not train its longest complete record to prove the selected context cap")
    else:
        audit = cohorts["train"]
        if (metadata.get("complete_selected_training_epochs") is not True
            or metadata.get("selected_training_scope") != "all context-eligible prepared rows"
            or metadata.get("trained_records") != len(rows) or metadata["optimizer_steps"] != len(rows)
            or audit.get("prepared") != metadata.get("prepared_split_counts", {}).get("train")
            or len(rows) + len(audit.get("excluded", [])) != audit.get("prepared")):
            raise PipelineFailure("Full mode must complete one epoch over every context-eligible native training row")
    return {"optimizer_steps": metadata["optimizer_steps"], "trained_records": metadata["trained_records"],
            "max_length": max_length, "capacity_probe": metadata.get("capacity_probe"),
            "required_record_ids": required, "reload_verification": metadata["reload_verification"],
            "trainable_fingerprints": metadata["trained_trainable_fingerprints"],
            "adapter_manifest_sha256": file_sha(output / "adapter/adapter_artifact_manifest.json")}


def verify_live_runtime(path, release, requested_at):
    # Reuse the publication gate for exact model-path/source binding, typed
    # probability validity, received-observation ages and credential exclusion.
    from publish_clef_local_model import validate_inference
    proof = validate_inference(path, release)
    if not proof or not proof.get("responses"):
        raise PipelineFailure("Live runtime produced no nonempty verified responses")
    expected = {"standalone_model_loaded": True, "backend": "mps", "device": "mps",
                "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
                "training_status": "trained_and_standalone_reload_verified",
                "release_manifest_sha256": file_sha(Path(release) / "release.json")}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise PipelineFailure("Live runtime did not load the exact saved standalone MPS model")
    if proof.get("freshness", {}).get("stale") is not False:
        raise PipelineFailure("Live runtime observation was stale at inference completion; retry a new capture without retraining")
    try:
        requested = datetime.fromisoformat(requested_at)
        captured = datetime.fromisoformat(proof["captured_at"])
        completed = datetime.fromisoformat(proof["inference_completed_at"])
        elapsed = (completed - captured).total_seconds()
        if min(requested.timestamp(), captured.timestamp(), completed.timestamp()) <= 0 or captured < requested or not 0 <= elapsed <= 30 or completed > datetime.now(timezone.utc):
            raise ValueError("invalid observation time")
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineFailure("Live runtime timestamps do not prove a new, bounded fresh observation") from error
    return {"path": str(path), "sha256": file_sha(path), "responses": len(proof["responses"]),
            "release_manifest_sha256": proof["release_manifest_sha256"], "freshness": proof["freshness"],
            "captured_at": proof["captured_at"], "inference_completed_at": proof["inference_completed_at"]}


class Backend:
    def __init__(self):
        self.children = {}

    snapshot = staticmethod(process_snapshot)
    free_disk = staticmethod(lambda path: shutil.disk_usage(path).free)
    reserve = staticmethod(training_reserve)
    verify_phase = staticmethod(verify_phase)
    verify_runtime = staticmethod(verify_live_runtime)

    @staticmethod
    def resume(identity):
        if not identity_matches(identity, process_snapshot(identity["pid"])):
            raise PipelineFailure("Process identity changed immediately before SIGCONT; no signal was sent")
        os.kill(identity["pid"], signal.SIGCONT)

    def spawn(self, command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
        self.children[child.pid] = child
        identity = process_snapshot(child.pid)
        if identity is None:
            raise PipelineFailure("Training child exited before its process identity could be recorded; inspect the log")
        guard = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(child.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        return {**identity, "caffeinate_pid": guard.pid, "log": str(log), "command_argv": command}

    def returncode(self, identity):
        child = self.children.get(identity["pid"])
        return child.poll() if child is not None else None

    @staticmethod
    def pause(identity):
        if identity_matches(identity, process_snapshot(identity["pid"])):
            os.killpg(identity["pid"], signal.SIGSTOP)

    @staticmethod
    def verify_release(output):
        from export_clef_mps_release import verify_mps_release
        return verify_mps_release(output)


class Coordinator:
    def __init__(self, config, state_path, backend=None, *, retry_runtime=False):
        self.config, self.path, self.backend = config, Path(state_path), backend or Backend()
        previous = read_json(self.path)
        if self.path.exists() and previous is None:
            raise PipelineFailure("Pipeline state is incomplete; preserve it and inspect before continuing")
        if previous and previous.get("config") != config:
            raise PipelineFailure("Existing pipeline state belongs to different processes or outputs; choose a separate state file")
        self.state = previous or {"schema_version": 1, "config": config, "created_at": utc_now(),
            "stage": "queued", "training_started": False, "optimizer_steps_27b": 0,
            "phase_progress": {phase: {"optimizer_steps": 0, "status": "not_started"} for phase in ("pilot", "full")},
            "verified_phases": {}, "children": {}, "converter_resume_sent": False}
        if retry_runtime:
            child = self.state["children"].get("runtime")
            if "full" not in self.state["verified_phases"] or self.state.get("failure_stage") != "runtime":
                raise PipelineFailure("Runtime-only retry requires completed full training and a recorded runtime failure")
            if child and self.backend.snapshot(child["pid"]) is not None:
                raise PipelineFailure("Recorded runtime process still exists; no duplicate retry is allowed")
            self.state["children"].pop("runtime", None)
            self.state["runtime_retry_pending"] = True
            self.state["stage"] = "runtime_retry_queued"

    def save(self):
        self.state["updated_at"] = utc_now()
        persist(self.path, self.state)

    def progress(self):
        for phase in ("pilot", "full"):
            metadata = read_json(Path(self.config[phase]) / "training.json")
            if actual_progress(metadata):
                self.state["phase_progress"][phase] = {"optimizer_steps": metadata["optimizer_steps"],
                    "planned_steps": metadata.get("planned_steps"), "status": metadata.get("status"),
                    "last_step_seconds": metadata.get("timings", {}).get("last_step_seconds"),
                    "elapsed_seconds": metadata.get("seconds")}
        self.state["optimizer_steps_27b"] = sum(item.get("optimizer_steps", 0) for item in self.state["phase_progress"].values())
        self.state["training_started"] = self.state["optimizer_steps_27b"] > 0
        self.state["full_training_started"] = self.state["phase_progress"].get("full", {}).get("optimizer_steps", 0) > 0
        self.state["full_training_completed"] = "full" in self.state["verified_phases"]

    def waiting(self, stage, required, free):
        self.state.update(stage=stage, required_free_bytes=required, free_disk_bytes=free,
            action_needed=f"Free {max(0, required - free) / GIB:.1f}GiB more on the base filesystem and reduce macOS memory/swap pressure. Existing files and converter are preserved.")
        self.save()
        return False

    def command(self, phase):
        command = [self.config["python"], "-u", str(ROOT / "scripts/train_clef_local.py"),
            "--base", self.config["base"], "--output", self.config[phase], "--mode", phase,
            "--max-length", str(self.config["max_length"]), "--last-layers", "4", "--rank", "8", "--gradient-accumulation", "1",
            "--epochs", "1", "--keep-checkpoints", "2"]
        if phase == "pilot":
            command += ["--max-steps", "16", "--train-records", "128", "--eval-records", "8", "--checkpoint-steps", "8"]
        else:
            command += ["--init-adapter", str(Path(self.config["pilot"]) / "adapter"),
                "--checkpoint-steps", "128", "--export-release", self.config["release"]]
        return command

    def observe_child(self, child, mismatch_error):
        # Reap our own child first. macOS can replace a zombie's command with
        # a short defunct name, and the PID may already belong to another
        # process after poll() reaps it. A confirmed exit code belongs to the
        # original Popen child; do not inspect or signal the current PID.
        code = self.backend.returncode(child)
        if code is not None:
            return None, code
        observed = self.backend.snapshot(child["pid"])
        if observed is not None and not identity_matches(child, observed):
            # A restarted coordinator cannot poll an attached child. Only a
            # zombie with the same PID and start time may have rewritten argv;
            # all live identities still require an exact command match.
            same_zombie = (
                "Z" in observed.get("state", "")
                and all(child.get(key) == observed.get(key) for key in ("pid", "started"))
            )
            if not same_zombie:
                raise PipelineFailure(mismatch_error)
        return observed, code

    def tick(self):
        self.progress()
        base = Path(self.config["base"])
        free = self.backend.free_disk(base)
        self.state["free_disk_bytes"] = free
        manifest = read_json(base / "conversion_manifest.json")
        expected = self.config["converter"]
        current = self.backend.snapshot(expected["pid"])
        complete = conversion_verified(manifest)
        if not complete:
            if current is None:
                raise PipelineFailure("Original converter exited before a verified standalone base existed; inspect its log. Source conversion cannot be restarted automatically.")
            if not identity_matches(expected, current) or not converter_target_matches(expected, base):
                raise PipelineFailure("Recorded converter PID/start/command/output identity changed; no signal or duplicate process is allowed")
            self.state["converted_source_shards"] = len((manifest or {}).get("completed_source_shards", []))
            if "T" in current["state"]:
                if self.state["converter_resume_sent"]:
                    raise PipelineFailure("Converter stopped again after its single authorized resume; inspect resources and manually resolve before continuing")
                if free < self.config["conversion_required_free_bytes"]:
                    return self.waiting("conversion_waiting_for_disk_headroom", self.config["conversion_required_free_bytes"], free)
                # Persist intent before signaling so a coordinator restart never
                # sends duplicate SIGCONT to the user-owned converter.
                self.state.update(converter_resume_sent=True, stage="conversion_resume_requested")
                self.save()
                self.backend.resume(expected)
            self.state.update(stage="conversion_running", action_needed=None)
            self.save()
            return False
        if identity_matches(expected, current) and "Z" not in current["state"]:
            self.state["stage"] = "conversion_finishing"
            self.save()
            return False
        self.state["base_manifest_sha256"] = file_sha(base / "conversion_manifest.json")
        reserve = self.backend.reserve(base)
        self.state["training_resource_plan"] = reserve
        for phase in ("pilot", "full"):
            if phase in self.state["verified_phases"]:
                continue
            self.state["active_phase"] = phase
            output = Path(self.config[phase])
            if output.exists() and not output.is_dir():
                raise PipelineFailure("Training output is not an empty directory; preserve existing files")
            child = self.state["children"].get(phase)
            if child:
                observed, code = self.observe_child(child,
                    "Trainer PID was reused or its exact identity changed; no duplicate or signal is allowed")
                if observed is not None and code is None and "Z" not in observed["state"]:
                    if "T" in observed["state"]:
                        if not child.get("paused_by_coordinator"):
                            raise PipelineFailure("Trainer was stopped externally; resolve that explicit stop before continuing")
                        if free < reserve["required_free_bytes"]:
                            return self.waiting(phase + "_waiting_for_disk_headroom", reserve["required_free_bytes"], free)
                        self.backend.resume(child)
                        child["paused_by_coordinator"] = False
                    self.state.update(stage=phase + "_process_running", action_needed=None)
                    self.save()
                    return False
                if code not in (None, 0):
                    raise PipelineFailure(f"{phase.title()} trainer exited with code {code}; inspect {child['log']}. Outputs will not be overwritten or automatically restarted.")
            if child or (output.exists() and any(output.iterdir())):
                proof = self.backend.verify_phase(output, phase, base, self.config["max_length"])
                self.state["verified_phases"][phase] = proof
                self.state.update(stage=phase + "_trained_and_reload_verified", action_needed=None)
                self.save()
                return False
            if phase == "full" and Path(self.config["release"]).exists() and (not Path(self.config["release"]).is_dir() or any(Path(self.config["release"]).iterdir())):
                raise PipelineFailure("Standalone destination already contains files; preserve it and choose a separate pipeline output")
            if free < reserve["required_free_bytes"]:
                return self.waiting(phase + "_waiting_for_disk_headroom", reserve["required_free_bytes"], free)
            if self.state["stage"] == phase + "_launch_requested":
                raise PipelineFailure("A trainer launch was requested but its PID was not durably recorded; inspect processes before manual recovery. No duplicate will be started.")
            if phase == "full" and self.config["max_length"] > self.state["verified_phases"]["pilot"].get("max_length", 0):
                raise PipelineFailure("Full context cap exceeds the capacity verified by the real pilot")
            self.state.update(stage=phase + "_launch_requested", action_needed=None)
            self.save()
            child = self.backend.spawn(self.command(phase), self.path.with_name(f"clef-local-{phase}.log"))
            self.state["children"][phase] = child
            self.state["stage"] = phase + "_process_running"
            self.save()
            return False
        release_path = Path(self.config["release"])
        release_sha = file_sha(release_path / "release.json")
        if self.state.get("strict_release_verified_sha256") == release_sha:
            released = read_json(release_path / "release.json")
        else:
            released = self.backend.verify_release(release_path)
            self.state["strict_release_verified_sha256"] = release_sha
            self.state["strict_release_verified_at"] = utc_now()
        proof = self.state["verified_phases"]["full"]
        training = released["training"]
        if training.get("run_mode") != "full" or training.get("optimizer_steps") != proof["optimizer_steps"] or training.get("trained_trainable_fingerprints") != proof["trainable_fingerprints"]:
            raise PipelineFailure("Standalone artifact does not come from the verified full training phase")
        self.state.update(action_needed=None,
            standalone_release=self.config["release"], release_manifest_sha256=file_sha(Path(self.config["release"]) / "release.json"),
            standalone_reload_verification=released["reload_verification"])
        return self.runtime_tick(free)

    def runtime_tick(self, free):
        self.state["active_phase"] = "runtime"
        child = self.state["children"].get("runtime")
        if child:
            observed, code = self.observe_child(child,
                "Runtime PID identity changed; no duplicate or signal is allowed")
            if observed is not None and code is None and "Z" not in observed["state"]:
                if "T" in observed["state"]:
                    if not child.get("paused_by_coordinator"):
                        raise PipelineFailure("Runtime process was stopped externally; resolve the explicit stop")
                    if free < 8 * GIB:
                        return self.waiting("runtime_waiting_for_disk_headroom", 8 * GIB, free)
                    self.backend.resume(child)
                    child["paused_by_coordinator"] = False
                self.state["stage"] = "runtime_process_running"
                self.save()
                return False
            if code not in (None, 0):
                raise PipelineFailure(f"Standalone live runtime exited with code {code}; inspect {child['log']}. Use --retry-runtime for a new capture without retraining.")
            runtime = self.backend.verify_runtime(Path(self.state["runtime_output"]),
                Path(self.config["release"]), self.state["runtime_requested_at"])
            # Verify payload hashes again after the actual inference process has
            # exited, rather than rereading all17GiB during every live poll.
            self.backend.verify_release(Path(self.config["release"]))
            self.state.update(stage="full_training_standalone_and_live_runtime_verified",
                              live_runtime=runtime, completed_at=utc_now(), failure_stage=None)
            self.progress()
            self.save()
            return True
        if self.state["stage"] == "runtime_launch_requested":
            raise PipelineFailure("Runtime launch identity was not recorded; inspect processes before retrying")
        if free < 8 * GIB:
            return self.waiting("runtime_waiting_for_disk_headroom", 8 * GIB, free)
        attempt = len(self.state.get("runtime_attempts", [])) + 1
        destination = Path(self.config["runtime_output"])
        if attempt > 1:
            destination = destination.with_name(destination.stem + f".retry-{attempt}" + destination.suffix)
        if destination.exists():
            raise PipelineFailure("Live runtime output already exists; preserve it and choose a separate output")
        command = [self.config["python"], "-u", str(ROOT / "scripts/run_clef_local.py"),
            "--model", self.config["release"], "--output", str(destination),
            "--max-length", str(self.config["max_length"])]
        self.state.update(stage="runtime_launch_requested", runtime_output=str(destination), runtime_requested_at=utc_now())
        self.save()
        child = self.backend.spawn(command, self.path.with_name(f"clef-local-runtime-{attempt}.log"))
        self.state["children"]["runtime"] = child
        self.state.setdefault("runtime_attempts", []).append({"child": dict(child), "output": str(destination),
                                                              "requested_at": self.state["runtime_requested_at"]})
        self.state["stage"] = "runtime_process_running"
        self.save()
        return False

    def stop(self):
        for phase, child in self.state["children"].items():
            current = self.backend.snapshot(child["pid"])
            if identity_matches(child, current):
                self.backend.pause(child)
                child["paused_by_coordinator"] = True
                self.state["stage"] = phase + "_paused_by_coordinator"
        self.progress()
        self.state["coordinator_stopped_at"] = utc_now()
        self.save()


def status(path):
    state = read_json(path)
    if state is None:
        return {"stage": "not_queued", "training_started": False, "optimizer_steps_27b": 0}
    # Status is read-only and never acquires the runner lock or sends a signal.
    result = dict(state)
    for phase in ("pilot", "full"):
        metadata = read_json(Path(state["config"][phase]) / "training.json")
        if actual_progress(metadata):
            result.setdefault("phase_progress", {})[phase] = {"optimizer_steps": metadata["optimizer_steps"],
                "planned_steps": metadata.get("planned_steps"), "status": metadata.get("status")}
    result["optimizer_steps_27b"] = sum(item.get("optimizer_steps", 0) for item in result.get("phase_progress", {}).values())
    result["training_started"] = result["optimizer_steps_27b"] > 0
    result["full_training_started"] = result.get("phase_progress", {}).get("full", {}).get("optimizer_steps", 0) > 0
    result["full_training_completed"] = "full" in result.get("verified_phases", {})
    result["observed_at"] = utc_now()
    return result


def actual_progress(metadata):
    return (metadata is not None and type(metadata.get("optimizer_steps")) is int
            and metadata["optimizer_steps"] >= 0 and metadata.get("model") == MODEL_ID
            and metadata.get("model_revision") == MODEL_REVISION and metadata.get("dataset_revision") == DATASET_REVISION
            and metadata.get("cpu_fallback") is False and metadata.get("security_filter", {}).get("applied") is True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "local/clef-local-pipeline.json")
    parser.add_argument("--status", action="store_true", help="Report actual optimizer evidence without starting/resuming anything")
    parser.add_argument("--base", type=Path, default=ROOT / "local/clef-27b-mps-nf4")
    parser.add_argument("--pilot", type=Path, default=ROOT / "outputs/clef-local-pilot")
    parser.add_argument("--full", type=Path, default=ROOT / "outputs/clef-local-full")
    parser.add_argument("--release", type=Path, default=ROOT / "outputs/clef-local-full-merged")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-connect/bin/python")
    parser.add_argument("--converter-pid", type=int, default=CONVERTER_PID)
    parser.add_argument("--converter-start", default=CONVERTER_START)
    parser.add_argument("--converter-command", default=CONVERTER_COMMAND)
    parser.add_argument("--min-conversion-free-gib", type=float, default=40)
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Complete-example cap used by both real pilot and full epoch")
    parser.add_argument("--runtime-output", type=Path, default=ROOT / "local/clef-local-full-live-inference.json")
    parser.add_argument("--retry-runtime", action="store_true", help="After a runtime failure, capture again using the saved full model; preserve prior output and never retrain")
    args = parser.parse_args()
    if not math.isfinite(args.min_conversion_free_gib) or args.min_conversion_free_gib < 40:
        parser.error("Conversion requires at least40GiB free to allow checkpoint output and swap headroom")
    if args.converter_pid < 1:
        parser.error("An explicit positive known converter PID is required")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if not args.status and (not args.python.is_file() or not os.access(args.python, os.X_OK)):
        parser.error("The existing MPS Python environment is unavailable")
    resolved = [args.base.resolve(), args.pilot.resolve(), args.full.resolve(), args.release.resolve()]
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(resolved) for right in resolved[index + 1:]):
        parser.error("Base, pilot, full and standalone destinations must be separate, non-nested directories")
    return args


def build_config(args):
    config = {name: str(getattr(args, name).resolve())
              for name in ("base", "pilot", "full", "release", "runtime_output")}
    # Invoking a venv's bin/python symlink selects its pyvenv.cfg. Resolving
    # that symlink would silently invoke the system interpreter instead.
    config["python"] = str(args.python.absolute())
    config.update(converter={"pid": args.converter_pid, "started": args.converter_start, "command": args.converter_command},
                  conversion_required_free_bytes=int(args.min_conversion_free_gib * GIB), max_length=args.max_length)
    return config


def main():
    args = parse_args()
    if args.status:
        print(json.dumps(status(args.state), indent=2))
        return 0
    config = build_config(args)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_name(args.state.name + ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("A coordinator already owns this pipeline; use --status") from error
        coordinator = Coordinator(config, args.state, retry_runtime=args.retry_runtime)
        signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(CoordinatorStop()))
        signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(CoordinatorStop()))
        try:
            while True:
                done = coordinator.tick()
                print(json.dumps({key: coordinator.state.get(key) for key in
                    ("stage", "training_started", "optimizer_steps_27b", "phase_progress", "free_disk_bytes", "required_free_bytes", "action_needed")}), flush=True)
                if done:
                    return 0
                time.sleep(60)
        except CoordinatorStop:
            coordinator.stop()
            print(json.dumps({"stage": coordinator.state["stage"], "coordinator_stopped": True,
                              "converter_signaled_on_stop": False, "state": str(args.state)}), flush=True)
            return 130
        except Exception as error:
            coordinator.state.update(stage="terminal_failure", failure_type=type(error).__name__,
                action_needed=str(error), failed_at=utc_now(), failure_stage=coordinator.state.get("active_phase"))
            coordinator.progress()
            coordinator.save()
            print(json.dumps({"stage": "terminal_failure", "action_needed": str(error), "state": str(args.state)}), flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())

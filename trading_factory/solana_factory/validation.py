"""Validation helpers for Solana trading-factory artifacts.

The validators are intentionally local and side-effect free. They inspect JSON
contracts and command plans, but never call Vulcan, Rise, cuFOLIO, wallets, or
network services.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vulcan_specs import validate_ta_strategy_config


SAFE_DEFAULT_MODES = {"observer", "paper"}
SECRET_FLAGS = {
    "--keypair",
    "--private-key",
    "--private-key-path",
    "--seed",
    "--wallet-password",
}
LIVE_TOKENS = {"live", "--live", "--yes"}


@dataclass
class ValidationReport:
    """Structured validation result for the generated factory bundle."""

    errors: dict[str, list[str]] = field(default_factory=dict)
    warnings: dict[str, list[str]] = field(default_factory=dict)
    checked_artifacts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, section: str, message: str) -> None:
        self.errors.setdefault(section, []).append(message)

    def add_warning(self, section: str, message: str) -> None:
        self.warnings.setdefault(section, []).append(message)

    def add_artifact(self, path: Path) -> None:
        value = path.as_posix()
        if value not in self.checked_artifacts:
            self.checked_artifacts.append(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_artifacts": self.checked_artifacts,
        }


def load_strategy_manifest(output_dir: Path) -> dict[str, Any]:
    """Load ``strategy_manifest.json`` from an artifact directory."""
    path = output_dir / "strategy_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_vulcan_command(command: Any, *, allow_preflight: bool = True) -> list[str]:
    """Return safety errors for a generated Vulcan command list."""
    errors: list[str] = []
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        return ["command must be a non-empty list of strings"]

    lowered = [item.lower() for item in command]
    if command[0] != "vulcan":
        errors.append("command must start with vulcan")
    for flag in SECRET_FLAGS:
        if flag in lowered:
            errors.append(f"command must not include secret-bearing flag {flag}")
    for token in LIVE_TOKENS:
        if token in lowered:
            errors.append(f"command must not include live/approval token {token}")

    is_preflight = "preflight" in lowered and "start" not in lowered
    if is_preflight and allow_preflight:
        return errors

    mode_values = [
        command[idx + 1].lower()
        for idx, token in enumerate(lowered[:-1])
        if token == "--mode"
    ]
    if not mode_values:
        errors.append("non-preflight command must include --mode paper")
    elif any(mode != "paper" for mode in mode_values):
        errors.append("command mode must be paper")
    return errors


def validate_vulcan_command_plans(command_plans: Any) -> list[str]:
    """Validate all named Vulcan command plans in a generated artifact."""
    errors: list[str] = []
    if not isinstance(command_plans, dict) or not command_plans:
        return ["command plans must be a non-empty object"]

    for name, command in command_plans.items():
        command_errors = validate_vulcan_command(command)
        if command_errors:
            errors.extend(f"{name}: {message}" for message in command_errors)
        lowered = [part.lower() for part in command] if isinstance(command, list) else []
        if str(name).startswith("live") and "preflight" not in lowered:
            errors.append(f"{name}: live-prefixed plans may only be preflight/readiness checks")
    return errors


def validate_cufolio_handoff(handoff: Any) -> list[str]:
    """Validate the cuFOLIO Mean-CVaR handoff contract."""
    errors: list[str] = []
    if not isinstance(handoff, dict):
        return ["handoff must be an object"]
    if handoff.get("handoff") != "cufolio_mean_cvar":
        errors.append("handoff must be cufolio_mean_cvar")
    if not isinstance(handoff.get("assets"), list) or not handoff["assets"]:
        errors.append("assets must be a non-empty list")
    if not isinstance(handoff.get("inputs"), dict) or not handoff["inputs"]:
        errors.append("inputs must name required market/portfolio datasets")
    params = handoff.get("cufolio_parameters", {})
    if not isinstance(params, dict):
        errors.append("cufolio_parameters must be an object")
    elif not _positive_number(params.get("confidence")) or not _positive_number(params.get("risk_aversion")):
        errors.append("confidence and risk_aversion must be positive")
    execution = handoff.get("execution", {})
    if not isinstance(execution, dict) or execution.get("default_mode") != "paper":
        errors.append("execution.default_mode must be paper")
    elif not execution.get("live_requires"):
        errors.append("execution.live_requires must document live promotion gates")
    return errors


def validate_rise_data_plan(plan: Any) -> list[str]:
    """Validate the read-only Rise/Phoenix data plan."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["rise data plan must be an object"]
    if plan.get("mode") != "read_only":
        errors.append("mode must be read_only")
    if not isinstance(plan.get("required_reads"), list) or not plan["required_reads"]:
        errors.append("required_reads must be a non-empty list")
    blocked = _list_text(plan.get("blocked_in_this_client")).lower()
    for phrase in ("order placement", "wallet signing", "private-key"):
        if phrase not in blocked:
            errors.append(f"blocked_in_this_client must include {phrase}")
    return errors


def validate_nvidia_agent_plan(plan: Any) -> list[str]:
    """Validate the NVIDIA/NemoClawd agent plan safety contract."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["agent plan must be an object"]
    if plan.get("default_mode") not in SAFE_DEFAULT_MODES:
        errors.append("default_mode must be observer or paper")
    roles = plan.get("roles", [])
    if not isinstance(roles, list) or len(roles) < 9:
        errors.append("roles must contain at least 9 agent roles")
    elif "execution_guard" not in {role.get("name") for role in roles if isinstance(role, dict)}:
        errors.append("roles must include execution_guard")

    tool_contract = plan.get("tool_contract", {})
    vulcan_contract = tool_contract.get("vulcan", {}) if isinstance(tool_contract, dict) else {}
    blocked = _list_text(vulcan_contract.get("blocked_by_default")).lower()
    for phrase in ("live order submission", "wallet password", "private-key"):
        if phrase not in blocked:
            errors.append(f"vulcan.blocked_by_default must include {phrase}")

    safety_policy = plan.get("safety_policy", {})
    if not isinstance(safety_policy, dict):
        errors.append("safety_policy must be an object")
    else:
        raw_allowed_modes = safety_policy.get("allowed_default_modes")
        if not isinstance(raw_allowed_modes, list) or not raw_allowed_modes:
            errors.append("allowed_default_modes must list observer and/or paper")
        elif not set(raw_allowed_modes).issubset(SAFE_DEFAULT_MODES):
            errors.append("allowed_default_modes may only contain observer and paper")
        if "not generated" not in str(safety_policy.get("live_mode_status", "")).lower():
            errors.append("live_mode_status must make live mode non-generated")

    commands = plan.get("commands", {})
    if isinstance(commands, dict):
        for name, command in commands.items():
            text = str(command).lower()
            if "--mode live" in text or " --yes" in text:
                errors.append(f"commands.{name} must not include live mode or --yes")
    return errors


def validate_strategy_bundle(
    manifest: dict[str, Any],
    repo_root: Path,
    output_dir: Path,
    *,
    require_files: bool = True,
) -> ValidationReport:
    """Validate a generated strategy manifest and referenced artifact files."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    report = ValidationReport()

    required_keys = {
        "safety_policy",
        "strategies",
        "optimizer_handoff",
        "rise_data_plan",
        "vulcan_command_plans",
        "nvidia_clawd_agent_plan",
    }
    missing = sorted(required_keys - set(manifest))
    for key in missing:
        report.add_error("manifest", f"missing required key {key}")

    safety_policy = manifest.get("safety_policy", {})
    if not isinstance(safety_policy, dict) or safety_policy.get("default_execution_mode") != "paper":
        report.add_error("manifest.safety_policy", "default_execution_mode must be paper")
    else:
        blocked = _list_text(safety_policy.get("never_in_generator")).lower()
        for phrase in ("wallet signing", "private-key", "live order submission"):
            if phrase not in blocked:
                report.add_error("manifest.safety_policy", f"never_in_generator must include {phrase}")

    strategies = manifest.get("strategies", [])
    if not isinstance(strategies, list) or not strategies:
        report.add_error("strategies", "at least one strategy entry is required")
    else:
        seen_names: set[str] = set()
        for idx, entry in enumerate(strategies, 1):
            section = f"strategies[{idx}]"
            if not isinstance(entry, dict):
                report.add_error(section, "strategy entry must be an object")
                continue
            name = str(entry.get("name") or section)
            if name in seen_names:
                report.add_error(section, f"duplicate strategy name {name}")
            seen_names.add(name)
            if entry.get("default_mode") != "paper":
                report.add_error(name, "default_mode must be paper")
            if entry.get("validation_errors"):
                report.add_error(name, f"embedded validation_errors must be empty: {entry['validation_errors']}")
            for message in validate_vulcan_command(entry.get("paper_command")):
                report.add_error(name, f"paper_command: {message}")

            config_path_value = entry.get("config_file")
            if config_path_value:
                config_path = _resolve_artifact_path(config_path_value, repo_root, output_dir)
                report.add_artifact(config_path)
                if require_files:
                    config = _load_json(config_path, report, name)
                    if isinstance(config, dict):
                        for message in validate_ta_strategy_config(config):
                            report.add_error(name, f"config_file: {message}")
                        if entry.get("symbol") and config.get("symbol") != entry["symbol"]:
                            report.add_error(name, "manifest symbol does not match config symbol")
            else:
                report.add_error(name, "config_file is required")

    artifact_validators = {
        "optimizer_handoff": validate_cufolio_handoff,
        "rise_data_plan": validate_rise_data_plan,
        "vulcan_command_plans": validate_vulcan_command_plans,
        "nvidia_clawd_agent_plan": validate_nvidia_agent_plan,
    }
    for key, validator in artifact_validators.items():
        value = manifest.get(key)
        if not value:
            continue
        artifact_path = _resolve_artifact_path(value, repo_root, output_dir)
        report.add_artifact(artifact_path)
        if not require_files:
            continue
        payload = _load_json(artifact_path, report, key)
        if payload is None:
            continue
        for message in validator(payload):
            report.add_error(key, message)

    if report.warnings:
        report.warnings = {key: sorted(set(values)) for key, values in report.warnings.items()}
    if report.errors:
        report.errors = {key: sorted(set(values)) for key, values in report.errors.items()}
    report.checked_artifacts.sort()
    return report


def _load_json(path: Path, report: ValidationReport, section: str) -> Any | None:
    if not path.exists():
        report.add_error(section, f"referenced artifact does not exist: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add_error(section, f"invalid JSON in {path}: {exc}")
        return None


def _resolve_artifact_path(value: Any, repo_root: Path, output_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path

    candidates = [
        path,
        repo_root / path,
        output_dir / path.name,
        output_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / path).resolve()


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _list_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return " ".join(str(item) for item in value)

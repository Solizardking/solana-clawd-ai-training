#!/usr/bin/env python3
"""No-cost preflight for the Solana transaction foundation workflow."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from notebook_bridge import NOTEBOOKS, check_notebook
from tx_foundation_common import (
    AI_TRAINING_DIR,
    BLUEPRINT_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_EVAL_OUTPUT,
    build_dataset_manifest,
    load_tx_config,
)


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: list[str | Path]) -> dict[str, Any]:
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(AI_TRAINING_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": [str(part) for part in cmd],
        "returncode": proc.returncode,
        "output_tail": proc.stdout[-4000:],
    }


def last_launch_log() -> dict[str, Any] | None:
    log_dir = AI_TRAINING_DIR / "outputs" / "job-launches"
    logs = sorted(log_dir.glob("tx-foundation-launch-*.log")) if log_dir.exists() else []
    if not logs:
        return None
    path = logs[-1]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "path": str(path),
        "hf_402": "402 Payment Required" in text or "Pre-paid credit balance is insufficient" in text,
        "tail": text[-2000:],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_tx_config(args.config)
    cpt_data = Path(cfg["cpt_data"])
    sft_data = Path(cfg["sft_data"])
    output_dir = Path(cfg["output_dir"])
    model_dir = output_dir / "sft"
    eval_path = Path(cfg.get("eval_output") or DEFAULT_EVAL_OUTPUT)
    manifest = build_dataset_manifest(
        dataset_path=cpt_data,
        config_path=Path(cfg["config_path"]),
        eval_path=eval_path,
        model_path=model_dir,
    )

    notebook_dir = BLUEPRINT_DIR
    notebook_checks = {
        name: {"ok": ok, "message": message}
        for name in NOTEBOOKS
        for ok, message in [check_notebook(notebook_dir / name)]
    }

    smoke_cmd = [
        sys.executable,
        "nvidia/blueprints/transaction-foundation-model/train.py",
        "--config",
        cfg["config_path"],
        "--stage",
        "both",
        "--smoke",
        "--dry-run",
    ]
    smoke = run_cmd(smoke_cmd) if args.run_smoke_dry_run else None

    hf_jobs = None
    if args.check_hf_jobs:
        if command_available("hf"):
            hf_jobs = run_cmd(["hf", "jobs", "ps", "--all"])
        else:
            hf_jobs = {"returncode": 127, "output_tail": "hf CLI not found"}

    required_local = {
        "config": Path(cfg["config_path"]).exists(),
        "cpt_data": cpt_data.exists(),
        "sft_data": sft_data.exists(),
        "processed_train": bool(manifest["processed_files"].get("train")),
        "processed_eval": bool(manifest["processed_files"].get("eval")),
        "processed_test": bool(manifest["processed_files"].get("test")),
        "launch_script": (AI_TRAINING_DIR / "scripts" / "launch_transaction_foundation_hf_job.sh").exists(),
        "watch_script": (AI_TRAINING_DIR / "scripts" / "watch_transaction_foundation_hf_job.sh").exists(),
        "post_train_script": (BLUEPRINT_DIR / "post_train.py").exists(),
        "notebooks_bootstrapped": all(item["ok"] for item in notebook_checks.values()),
    }
    ready_for_remote = all(required_local.values()) and manifest["num_examples"] > 0

    return {
        "config": cfg,
        "manifest": manifest,
        "required_local": required_local,
        "ready_for_remote_training": ready_for_remote,
        "local_model_present": model_dir.exists(),
        "eval_present": eval_path.exists(),
        "notebooks": notebook_checks,
        "smoke_dry_run": smoke,
        "hf_jobs": hf_jobs,
        "last_launch_log": last_launch_log(),
        "next_actions": [
            "Add Hugging Face Jobs credits if the last launch log shows hf_402=true.",
            "Launch: bash scripts/launch_transaction_foundation_hf_job.sh a100-large 6h",
            "Watch: bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>",
            "After success: EVALUATE=1 BUNDLE=1 REGISTER=1 bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default="outputs/tx_foundation_preflight.json")
    parser.add_argument("--check-hf-jobs", action="store_true", help="Include `hf jobs ps --all` output")
    parser.add_argument("--run-smoke-dry-run", action="store_true", default=True)
    parser.add_argument("--no-smoke-dry-run", action="store_false", dest="run_smoke_dry_run")
    parser.add_argument("--write-manifest", action="store_true", help="Also refresh tx_foundation_cpt_manifest.json")
    args = parser.parse_args()

    report = build_report(args)
    out = Path(args.output)
    if not out.is_absolute():
        out = AI_TRAINING_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "ready_for_remote_training": report["ready_for_remote_training"],
        "local_model_present": report["local_model_present"],
        "eval_present": report["eval_present"],
        "examples": report["manifest"]["num_examples"],
        "smoke_returncode": None if report["smoke_dry_run"] is None else report["smoke_dry_run"]["returncode"],
    }, indent=2))

    if args.write_manifest:
        DEFAULT_DATASET_MANIFEST.write_text(json.dumps(report["manifest"], indent=2) + "\n", encoding="utf-8")

    smoke_ok = report["smoke_dry_run"] is None or report["smoke_dry_run"]["returncode"] == 0
    return 0 if report["ready_for_remote_training"] and smoke_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the local NVIDIA blueprint integration without printing secrets."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "trading_factory"))
sys.path.insert(0, str(BASE_DIR / "nvidia" / "integration"))

from solana_factory.factory import build_strategy_bundle  # noqa: E402
from solana_factory.nvidia_agent import NVIDIA_BLUEPRINTS  # noqa: E402
from nemo_clawd import write_nemo_clawd_assets  # noqa: E402


REQUIRED_FILES = [
    "nvidia/README.md",
    "nvidia/configs/nemo_clawd_factory.yaml",
    "nvidia/integration/nemo_clawd.py",
    "nvidia/integration/nemo_clawd_agent.py",
    "nvidia/blueprints/aiq/agent.py",
    "nvidia/blueprints/aiq/tools.py",
    "nvidia/blueprints/aiq/workflow.yaml",
    "nvidia/blueprints/enterprise-rag/README.md",
    "nvidia/blueprints/model-distillation/distill.py",
    "nvidia/blueprints/portfolio-optimization/mean_cvar.py",
    "nvidia/blueprints/signal-discovery/agent.py",
    "nvidia/blueprints/transaction-foundation-model/dataset_builder.py",
    "nvidia/cufolio/constraints.py",
    "nvidia/cufolio/portfolio.py",
    "nvidia/cufolio/rebalance.py",
    "nvidia/integration/clawd_nim_bridge.py",
    "nvidia/integration/dataset_nvidia_sft.py",
    "nvidia/integration/trading_factory_nvidia.py",
    "data/perps/nvidia_perps_handoff.json",
    "perps/README.md",
    "perps/nvidia_perps.py",
    "trading_factory/solana_factory/nvidia_agent.py",
]

SECRET_PATTERNS = {
    "google_oauth_secret_file": re.compile("client" + r"_secret_\d+[-\w]+\.apps\.googleusercontent\.com\.json"),
    "google_adc_path": re.compile(r"\.config/gcloud/application_default_credentials\.json"),
    "google_oauth_token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),
    "nvidia_api_key": re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile("-----" + "BEGIN " + r"(?:RSA |EC |OPENSSH |)?" + "PRIVATE " + "KEY-----"),
    "wandb_key": re.compile(r"\bwandb_v1_[A-Za-z0-9_-]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
}


def scan_files(paths: Iterable[Path]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append((path.as_posix(), name))
    return findings


def verify_files() -> bool:
    ok = True
    print("[files]")
    for rel in REQUIRED_FILES:
        path = BASE_DIR / rel
        if path.exists():
            print(f"OK   {rel}")
        else:
            ok = False
            print(f"FAIL {rel}: missing")
    for name, meta in NVIDIA_BLUEPRINTS.items():
        path = BASE_DIR / meta["local_path"]
        if path.exists():
            print(f"OK   blueprint {name}: {meta['local_path']}")
        else:
            ok = False
            print(f"FAIL blueprint {name}: missing {meta['local_path']}")
    return ok


def verify_generated_bundle() -> bool:
    print("[bundle]")
    with tempfile.TemporaryDirectory(prefix="solana-clawd-nvidia-") as tmpdir:
        output_dir = Path(tmpdir)
        manifest = build_strategy_bundle(repo_root=BASE_DIR, output_dir=output_dir)
        required_keys = {
            "optimizer_handoff",
            "rise_data_plan",
            "vulcan_command_plans",
            "nvidia_clawd_agent_plan",
        }
        missing = sorted(required_keys - set(manifest))
        if missing:
            print(f"FAIL generated manifest missing keys: {missing}")
            return False
        plan_path = Path(manifest["nvidia_clawd_agent_plan"])
        if not plan_path.exists():
            print(f"FAIL generated plan missing: {plan_path}")
            return False
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        role_count = len(plan.get("roles", []))
        if role_count < 9:
            print(f"FAIL generated plan role_count={role_count}")
            return False
        if plan.get("default_mode") not in {"observer", "paper"}:
            print(f"FAIL generated plan unsafe mode={plan.get('default_mode')}")
            return False
        print(f"OK   generated manifest with nvidia plan and {role_count} roles")
        return True


def verify_nemo_clawd_assets() -> bool:
    print("[nemo-clawd]")
    with tempfile.TemporaryDirectory(prefix="solana-clawd-nemoclawd-") as tmpdir:
        output_dir = Path(tmpdir)
        assets = write_nemo_clawd_assets(output_dir=output_dir, core_ai_dir=BASE_DIR.parent / "core-ai")
        inventory_path = assets["inventory_path"]
        blueprint_path = assets["blueprint_path"]
        if not inventory_path.exists() or not blueprint_path.exists():
            print("FAIL Nemo Clawd inventory/blueprint files were not written")
            return False
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
        missing = inventory.get("missing_required_paths", [])
        if missing:
            print(f"FAIL Core AI inventory missing required paths: {missing}")
            return False
        if blueprint.get("slug") != "nemo-clawd":
            print(f"FAIL unexpected Nemo Clawd slug: {blueprint.get('slug')}")
            return False
        if not blueprint.get("network_policy", {}).get("allowed_egress"):
            print("FAIL Nemo Clawd network policy missing allowed_egress")
            return False
        print(
            "OK   Nemo Clawd inventory + blueprint "
            f"packages={len(inventory.get('packages', []))} "
            f"skills={len(inventory.get('skills', []))} "
            f"mcp_tools={len(inventory.get('mcp_tools', []))}"
        )
        return True


def verify_secrets() -> bool:
    print("[secrets]")
    paths = [
        path
        for root in ["nvidia", "perps", "data/perps", "trading_factory/solana_factory"]
        for path in (BASE_DIR / root).rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml", ".json", ".sh"}
    ]
    findings = scan_files(paths)
    if findings:
        for path, name in findings:
            print(f"FAIL {path}: matched {name}")
        return False
    print("OK   no private credential patterns found in NVIDIA integration files")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    ok = verify_files()
    ok = verify_generated_bundle() and ok
    ok = verify_nemo_clawd_assets() and ok
    ok = verify_secrets() and ok
    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

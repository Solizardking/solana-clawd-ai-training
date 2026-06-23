#!/usr/bin/env python3
"""Decide the next Solana Clawd training job from Hugging Face + local gates.

This script is read-only. It queries public Hugging Face org metadata, inspects
local preflight reports, and emits a concrete next-job recommendation without
printing secrets or launching jobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


AI_TRAINING_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = AI_TRAINING_DIR / "outputs" / "next_training_job_decision.json"
HF_API = "https://huggingface.co/api"
ORG = "solanaclawd"

COMPLETE_ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
PRIORITY_MODEL_REPOS = [
    "solanaclawd/solana-tx-foundation-7b",
    "solanaclawd/solana-tx-foundation-1.5b",
    "solanaclawd/solana-clawd-7b-lora",
    "solanaclawd/solana-clawd-1.5b",
    "solanaclawd/solana-clawd-core-ai-1.5b-lora",
    "solanaclawd/solana-nvidia-trading-factory-8b-lora",
    "solanaclawd/clawd-solana-masterpiece-qwen15-lora",
]
PRIORITY_DATASETS = [
    "solanaclawd/solana-tx-foundation-unified",
    "solanaclawd/solana-tx-foundation-cpt",
    "solanaclawd/solana-clawd-core-ai-instruct",
    "solanaclawd/solana-clawd-realtime-research-instruct",
    "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def hf_get(path: str, *, timeout: int = 30) -> tuple[Any | None, str | None]:
    url = f"{HF_API}{path}"
    request = Request(url, headers={"accept": "application/json", "user-agent": "solana-clawd-training-audit"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        return None, f"http_{exc.code}: {body[:400]}"
    except (URLError, TimeoutError) as exc:
        return None, f"network_error: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"json_error: {exc}"


def repo_path(repo_id: str) -> str:
    return quote(repo_id, safe="/")


def siblings(metadata: dict[str, Any] | None) -> set[str]:
    if not isinstance(metadata, dict):
        return set()
    return {item.get("rfilename", "") for item in metadata.get("siblings", []) if isinstance(item, dict)}


def summarize_model(repo_id: str) -> dict[str, Any]:
    metadata, error = hf_get(f"/models/{repo_path(repo_id)}")
    files = siblings(metadata)
    return {
        "id": repo_id,
        "available": error is None,
        "error": error,
        "last_modified": metadata.get("lastModified") if isinstance(metadata, dict) else None,
        "downloads": metadata.get("downloads") if isinstance(metadata, dict) else None,
        "base_model": ((metadata.get("config") or {}).get("peft") or {}).get("base_model_name_or_path")
        if isinstance(metadata, dict)
        else None,
        "files": sorted(files),
        "complete_adapter": COMPLETE_ADAPTER_FILES.issubset(files),
        "missing_adapter_files": sorted(COMPLETE_ADAPTER_FILES - files),
    }


def summarize_dataset(repo_id: str) -> dict[str, Any]:
    metadata, error = hf_get(f"/datasets/{repo_path(repo_id)}")
    files = siblings(metadata)
    card_data = metadata.get("cardData", {}) if isinstance(metadata, dict) else {}
    dataset_info = card_data.get("dataset_info", {}) if isinstance(card_data, dict) else {}
    splits = dataset_info.get("splits", []) if isinstance(dataset_info, dict) else []
    examples = sum(split.get("num_examples", 0) for split in splits if isinstance(split, dict))
    return {
        "id": repo_id,
        "available": error is None,
        "error": error,
        "last_modified": metadata.get("lastModified") if isinstance(metadata, dict) else None,
        "downloads": metadata.get("downloads") if isinstance(metadata, dict) else None,
        "files": sorted(files),
        "examples_from_card": examples or None,
        "has_readme": "README.md" in files,
        "parquet_count": sum(1 for name in files if name.endswith(".parquet")),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def local_file_info(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def latest_items(kind: str, limit: int) -> list[dict[str, Any]]:
    payload, error = hf_get(f"/{kind}?author={ORG}&sort=lastModified&direction=-1&limit={limit}")
    if error or not isinstance(payload, list):
        return [{"error": error or "unexpected payload"}]
    return [
        {
            "id": item.get("id"),
            "last_modified": item.get("lastModified"),
            "downloads": item.get("downloads"),
            "private": item.get("private"),
            "tags": item.get("tags", [])[:12],
        }
        for item in payload
        if isinstance(item, dict)
    ]


def build_decision() -> dict[str, Any]:
    preflight_path = AI_TRAINING_DIR / "outputs" / "tx_foundation_preflight.json"
    preflight = load_json(preflight_path)
    optimization_manifest = load_json(AI_TRAINING_DIR / "data" / "model_kit" / "training_data_optimization_manifest.json")
    tx_config = load_json(AI_TRAINING_DIR / "outputs" / "tx_foundation_preflight.json").get("config", {})

    models = {repo_id: summarize_model(repo_id) for repo_id in PRIORITY_MODEL_REPOS}
    datasets = {repo_id: summarize_dataset(repo_id) for repo_id in PRIORITY_DATASETS}

    tx_7b_missing = not models["solanaclawd/solana-tx-foundation-7b"]["complete_adapter"]
    unified = datasets["solanaclawd/solana-tx-foundation-unified"]
    required_unified = {
        "tx_foundation_cpt_clean.jsonl",
        "solana_clawd_reasoning_tooling_sft.jsonl",
        "training_data_optimization_manifest.json",
    }
    unified_ready = unified["available"] and required_unified.issubset(set(unified["files"]))
    preflight_ready = bool(preflight.get("ready_for_remote_training"))
    previous_credit_block = bool((preflight.get("last_launch_log") or {}).get("hf_402"))

    reasons: list[str] = []
    if tx_7b_missing:
        reasons.append("solanaclawd/solana-tx-foundation-7b has no public complete adapter yet")
    if unified_ready:
        reasons.append("solanaclawd/solana-tx-foundation-unified contains the CPT, SFT, and optimization manifest files")
    if preflight_ready:
        reasons.append("local transaction-foundation preflight is ready for remote training")
    if models["solanaclawd/solana-clawd-core-ai-1.5b-lora"]["complete_adapter"]:
        reasons.append("core 1.5B LoRA is already released with adapter files")
    if models["solanaclawd/solana-nvidia-trading-factory-8b-lora"]["complete_adapter"]:
        reasons.append("trading factory 8B LoRA is already released with adapter files")

    blocker = None
    if previous_credit_block:
        blocker = "Previous transaction-foundation launch logs show HF Jobs 402 Payment Required; add Jobs credits before real launch."

    launch_command = "bash scripts/launch_transaction_foundation_hf_job.sh a100-large 12h"
    if not preflight_ready:
        launch_command = "python3 nvidia/blueprints/transaction-foundation-model/preflight.py --check-hf-dataset --check-hf-jobs"

    return {
        "generated_at": utc_now(),
        "huggingface_org": f"https://huggingface.co/{ORG}",
        "latest": {
            "models": latest_items("models", 25),
            "datasets": latest_items("datasets", 25),
            "spaces": latest_items("spaces", 25),
        },
        "priority_models": models,
        "priority_datasets": datasets,
        "local_training_data": {
            "cpt_clean": local_file_info(AI_TRAINING_DIR / "data" / "model_kit" / "tx_foundation_cpt_clean.jsonl"),
            "reasoning_tooling_sft": local_file_info(AI_TRAINING_DIR / "data" / "model_kit" / "solana_clawd_reasoning_tooling_sft.jsonl"),
            "optimization_manifest": {
                "path": "data/model_kit/training_data_optimization_manifest.json",
                "cpt_kept": (optimization_manifest.get("cpt_stats") or {}).get("kept"),
                "sft_kept": (optimization_manifest.get("message_stats") or {}).get("kept"),
                "generated_at": optimization_manifest.get("generated_at"),
            },
        },
        "preflight": {
            "path": preflight_path.as_posix(),
            "ready_for_remote_training": preflight_ready,
            "remote_dataset_ok": bool((preflight.get("remote_dataset") or {}).get("ok")),
            "smoke_returncode": (preflight.get("smoke_dry_run") or {}).get("returncode"),
            "launch_dry_run_returncode": (preflight.get("launch_dry_run") or {}).get("returncode"),
            "last_launch_hf_402": previous_credit_block,
        },
        "recommendation": {
            "next_job": "transaction_foundation_cpt_sft",
            "model_to_train": "solanaclawd/solana-tx-foundation-7b",
            "base_model": tx_config.get("base_model") or "Qwen/Qwen2.5-7B-Instruct",
            "dataset": "solanaclawd/solana-tx-foundation-unified",
            "launch_command": launch_command,
            "watch_command": "bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>",
            "after_success": "EVALUATE=1 BUNDLE=1 REGISTER=1 bash scripts/watch_transaction_foundation_hf_job.sh <JOB_ID>",
            "blocker": blocker,
            "reasons": reasons,
        },
        "do_not_prioritize_now": [
            "Do not rerun solana-clawd-core-ai-1.5b-lora unless you want a quality-refresh; it already has public adapter files.",
            "Do not rerun solana-nvidia-trading-factory-8b-lora unless you want a larger trading dataset pass; it already has public adapter files.",
            "Do not treat solana-clawd-1.5b as live merged output; its public repo only has .gitattributes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = build_decision()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text)
    if not args.no_write:
        output = Path(args.output)
        if not output.is_absolute():
            output = AI_TRAINING_DIR / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

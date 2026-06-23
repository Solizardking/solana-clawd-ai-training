#!/usr/bin/env python3
"""Run the local Solana Clawd AI/NVIDIA stack on a Mac.

The default path is intentionally safe:
- no Hugging Face uploads
- no live registry writes
- no live trading
- no remote GPU jobs
- no full local training

It wires the existing repo pieces together into one local control plane and
writes a compact status report under outputs/local_clawd_stack_summary.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


AI_TRAINING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_TRAINING_DIR.parent
DEFAULT_SUMMARY = AI_TRAINING_DIR / "outputs" / "local_clawd_stack_summary.json"
DEFAULT_RAG_STORE = AI_TRAINING_DIR / "data" / "nvidia_rag_store"
PYTHON = os.environ.get("PYTHON", sys.executable or "python3")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def printable(cmd: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_capture(cmd: list[str | Path], *, timeout: int = 20) -> tuple[int, str, str]:
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(AI_TRAINING_DIR),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def detect_torch() -> dict[str, Any]:
    if not have_module("torch"):
        return {"installed": False, "cuda": False, "mps_built": False, "mps": False}
    import torch  # type: ignore

    return {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda": bool(torch.cuda.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps": bool(torch.backends.mps.is_available()),
    }


def ollama_models() -> list[dict[str, str]]:
    if not command_available("ollama"):
        return []
    rc, out, _ = run_capture(["ollama", "list"], timeout=20)
    if rc != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append({"name": parts[0], "id": parts[1], "size": " ".join(parts[2:4])})
    return rows


def environment_report() -> dict[str, Any]:
    modules = [
        "datasets",
        "faiss",
        "fastapi",
        "huggingface_hub",
        "peft",
        "torch",
        "transformers",
        "trl",
        "uvicorn",
        "yaml",
    ]
    return {
        "generated_at": utc_now(),
        "cwd": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": detect_torch(),
        "modules": {name: have_module(name) for name in modules},
        "commands": {
            "hf": command_available("hf"),
            "npm": command_available("npm"),
            "ollama": command_available("ollama"),
            "vulcan": command_available("vulcan"),
        },
        "env": {
            "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
            "NVIDIA_API_KEY": bool(os.environ.get("NVIDIA_API_KEY")),
            "WANDB_API_KEY": bool(os.environ.get("WANDB_API_KEY")),
            "RPC_URL": bool(os.environ.get("RPC_URL")),
        },
        "ollama_models": ollama_models(),
    }


def model_plan(env: dict[str, Any]) -> dict[str, Any]:
    torch_info = env["torch"]
    mps_ready = bool(torch_info.get("mps"))
    cuda_ready = bool(torch_info.get("cuda"))
    ollama_names = {row["name"] for row in env.get("ollama_models", [])}

    if cuda_ready:
        train_mode = "cuda-local"
        local_training = [
            "Qwen/Qwen2.5-7B-Instruct for tx-foundation CPT+SFT",
            "NousResearch/Hermes-3-Llama-3.1-8B for trading-factory LoRA",
            "Qwen/Qwen2.5-1.5B-Instruct for fast adapter iteration",
        ]
    elif mps_ready:
        train_mode = "apple-mps"
        local_training = [
            "Qwen/Qwen2.5-1.5B-Instruct for full local LoRA",
            "Qwen/Qwen2.5-7B-Instruct for short smoke LoRA if memory pressure is acceptable",
            "Hermes-3-Llama-3.1-8B for short trading-factory smoke runs only",
        ]
    else:
        train_mode = "cpu-only"
        local_training = [
            "Do not run full LoRA training from this Python environment.",
            "Use dataset/preflight dry-runs locally, or install an Apple Silicon MPS/MLX environment.",
            "Use Ollama models for local inference and distillation review.",
        ]

    local_inference = [
        name
        for name in [
            "8bit/solana-clawd-core-ai:latest",
            "8bit/solana-trading-factory:latest",
            "8bit/DeepSolana:latest",
            "hermes3:8b",
            "nemotron3:33b",
            "qwen2.5:1.5b",
        ]
        if name in ollama_names
    ]

    return {
        "train_mode": train_mode,
        "local_training": local_training,
        "local_inference": local_inference,
        "recommended_next": [
            "Run the tx-foundation local preflight and dry-run here.",
            "Train solanaclawd/solana-tx-foundation-7b on a remote A100/H200 job when credits are available.",
            "Use Nemotron locally or via NVIDIA NIM as a teacher/evaluator, not as the first Mac fine-tune target.",
            "Use GLM-5.2 and DeepSeek V4 Pro as cloud teacher/eval lanes before attempting huge-model adapters.",
        ],
        "lanes": {
            "core-ai-local": {
                "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "solanaclawd/solana-clawd-core-ai-instruct",
                "output": "solanaclawd/solana-clawd-core-ai-1.5b-lora",
                "fit": "best local training target",
            },
            "tx-foundation-production": {
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "dataset": "solanaclawd/solana-tx-foundation-unified",
                "output": "solanaclawd/solana-tx-foundation-7b",
                "fit": "primary remote training target",
            },
            "trading-factory": {
                "base_model": "NousResearch/Hermes-3-Llama-3.1-8B",
                "dataset": "solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
                "output": "solanaclawd/solana-nvidia-trading-factory-8b-lora",
                "fit": "tool-use/perps adapter",
            },
            "nemotron-teacher": {
                "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "dataset": "solanaclawd/solana-tx-foundation-unified",
                "output": "distilled Solana SFT data or evaluator traces",
                "fit": "teacher, judge, RAG reasoner, and distillation source",
            },
        },
    }


def run_step(
    name: str,
    cmd: list[str | Path],
    *,
    dry_run: bool,
    optional: bool = False,
    timeout: int = 180,
) -> dict[str, Any]:
    print(f"\n[{name}]")
    print(f"$ {printable(cmd)}")
    if dry_run:
        return {
            "name": name,
            "cmd": [str(part) for part in cmd],
            "dry_run": True,
            "optional": optional,
            "ok": True,
            "returncode": None,
        }
    started = dt.datetime.now(dt.UTC)
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(AI_TRAINING_DIR),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=os.environ.copy(),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if stdout:
            print(tail(stdout, 2000))
        if stderr:
            print(tail(stderr, 2000), file=sys.stderr)
        ok = proc.returncode == 0
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        ok = False
        returncode = 124
        print(f"[{name}] timed out after {timeout}s")

    finished = dt.datetime.now(dt.UTC)
    return {
        "name": name,
        "cmd": [str(part) for part in cmd],
        "dry_run": False,
        "optional": optional,
        "ok": ok,
        "returncode": returncode,
        "started_at": started.replace(microsecond=0).isoformat(),
        "finished_at": finished.replace(microsecond=0).isoformat(),
        "duration_s": round((finished - started).total_seconds(), 3),
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
    }


def maybe_rag_steps(args: argparse.Namespace) -> list[tuple[str, list[str | Path], bool, int]]:
    if not args.with_rag:
        return []
    if not have_module("faiss"):
        return [
            (
                "rag-missing-faiss",
                [
                    PYTHON,
                    "-c",
                    "raise SystemExit('faiss is missing; install faiss-cpu to enable local RAG ingest/query')",
                ],
                True,
                20,
            )
        ]

    sources = args.rag_sources or [
        "README.md",
        "nvidia/README.md",
        "nvidia/LOCAL_MAC_STACK.md",
        "nvidia/configs/solana_tx_foundation.yaml",
        "outputs/next_training_job_decision.json",
        "model-kit/README.md",
        "trading_factory/README.md",
    ]
    return [
        (
            "rag-ingest",
            [
                PYTHON,
                "nvidia/blueprints/enterprise-rag/ingest.py",
                "--sources",
                *sources,
                "--store",
                str(args.rag_store),
            ],
            True,
            300,
        ),
        (
            "rag-query",
            [
                PYTHON,
                "nvidia/blueprints/enterprise-rag/query.py",
                "--store",
                str(args.rag_store),
                "--question",
                args.rag_question,
            ],
            True,
            120,
        ),
    ]


def training_smoke_step(args: argparse.Namespace, env: dict[str, Any]) -> tuple[str, list[str | Path], bool, int] | None:
    if not args.with_training_smoke:
        return None
    torch_info = env["torch"]
    if not (torch_info.get("cuda") or torch_info.get("mps") or args.force_cpu_training):
        return (
            "training-smoke-skipped",
            [
                PYTHON,
                "-c",
                "raise SystemExit('training smoke skipped: current torch has no CUDA/MPS; pass --force-cpu-training for a tiny CPU-only experiment')",
            ],
            True,
            20,
        )
    return (
        "core-ai-training-dry-run",
        [
            PYTHON,
            "model-kit/clawd_model_kit.py",
            "train",
            "--lane",
            "core-ai",
            "--config",
            "configs/autoresearch_wiki_lora_config_mac.yaml",
            "--train-dry-run",
            "--no-eval",
            "--no-checkpoints",
        ],
        True,
        300,
    )


def build_steps(args: argparse.Namespace, env: dict[str, Any]) -> list[tuple[str, list[str | Path], bool, int]]:
    steps: list[tuple[str, list[str | Path], bool, int]] = [
        ("model-kit-doctor", [PYTHON, "model-kit/clawd_model_kit.py", "doctor", "--json"], False, 60),
        ("nvidia-configs", [PYTHON, "nvidia/scripts/validate_configs.py", "--strict"], False, 60),
        ("strategy-bundle", [PYTHON, "scripts/build_solana_trading_factory_strategies.py"], False, 120),
        ("aiq-plan", [PYTHON, "nvidia/blueprints/aiq/agent.py", "--strict"], False, 120),
        (
            "tx-preflight",
            [
                PYTHON,
                "nvidia/blueprints/transaction-foundation-model/preflight.py",
                "--no-smoke-dry-run",
                "--no-launch-dry-run",
            ],
            False,
            120,
        ),
        (
            "tx-foundation-smoke-plan",
            [
                PYTHON,
                "nvidia/blueprints/transaction-foundation-model/pipeline.py",
                "--dry-run",
                "--smoke",
                "--stages",
                "cpt",
                "sft",
                "evaluate",
                "push",
            ],
            False,
            120,
        ),
        (
            "perps-manifest",
            [
                PYTHON,
                "model-kit/clawd_model_kit.py",
                "perps",
                "manifest",
                "--mode",
                "observer",
                "--market",
                "SOL",
                "--output",
                "data/model_kit/perps_tool_manifest.json",
            ],
            False,
            60,
        ),
    ]

    train_step = training_smoke_step(args, env)
    if train_step:
        steps.append(train_step)
    steps.extend(maybe_rag_steps(args))
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--best-effort", action="store_true", help="Continue after required step failures.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Summary JSON path.")
    parser.add_argument("--with-rag", action="store_true", help="Run local RAG ingest/query if faiss is installed.")
    parser.add_argument("--rag-store", default=str(DEFAULT_RAG_STORE))
    parser.add_argument("--rag-question", default="Which Solana Clawd model should train next?")
    parser.add_argument("--rag-sources", nargs="*", default=None)
    parser.add_argument("--with-training-smoke", action="store_true", help="Run the guarded local training dry-run.")
    parser.add_argument("--force-cpu-training", action="store_true", help="Allow a tiny CPU-only training dry-run gate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.rag_store = Path(args.rag_store)
    if not args.rag_store.is_absolute():
        args.rag_store = AI_TRAINING_DIR / args.rag_store

    env = environment_report()
    plan = model_plan(env)
    print(json.dumps({"environment": env, "model_plan": plan}, indent=2, sort_keys=True))

    results = []
    ok = True
    for name, cmd, optional, timeout in build_steps(args, env):
        result = run_step(name, cmd, dry_run=args.dry_run, optional=optional, timeout=timeout)
        results.append(result)
        if not result["ok"]:
            ok = False
            if not (args.best_effort or optional):
                break

    summary = {
        "generated_at": utc_now(),
        "ok": ok,
        "dry_run": args.dry_run,
        "best_effort": args.best_effort,
        "environment": env,
        "model_plan": plan,
        "steps": results,
        "local_urls": {
            "model_kit_ui": "http://127.0.0.1:5173",
            "model_kit_api": "http://127.0.0.1:8765",
            "rag_api": "http://127.0.0.1:8766",
        },
        "follow_up_commands": [
            "python3 model-kit/clawd_model_kit.py ui --host 127.0.0.1 --port 5173",
            "python3 -m uvicorn main:app --host 127.0.0.1 --port 8765 --app-dir model-kit/backend",
            "python3 nvidia/blueprints/enterprise-rag/pipeline.py --store data/nvidia_rag_store --host 127.0.0.1 --port 8766",
            "bash scripts/launch_transaction_foundation_hf_job.sh a100-large 12h",
        ],
    }

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = AI_TRAINING_DIR / summary_path
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Nemotron Ultra 550B teacher labeler for solana-chat nanochat training.

Reads unlabeled Solana conversation prompts (from dataset.py or raw JSONL)
and asks Nemotron Ultra to produce high-quality assistant completions.
These labeled pairs are saved as SFT JSONL for nanochat fine-tuning.

Endpoint routing:
  HF_TOKEN           → huggingface.co serverless (primary)
  NVIDIA_API_KEY     → NVIDIA NIM
  CLAWD_INFERENCE_URL → self-hosted
  default            → clawd-box-router.fly.dev (1.5B fallback)

Usage:
    python3 solana/nemotron_labeler.py \
        --input solana/prompts.jsonl \
        --output runs/nemotron_labels.jsonl \
        --count 200
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# ── Endpoint routing ──────────────────────────────────────────────────────────

MODEL_HF        = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MODEL_NIM       = "nvidia/nemotron-3-ultra-550b-a55b"
MODEL_FALLBACK  = "solana-clawd-1.5b"

HF_BASE     = "https://api-inference.huggingface.co/v1"
NIM_BASE    = "https://integrate.api.nvidia.com/v1"
CLAWD_BASE  = "https://clawd-box-router.fly.dev/v1"


@dataclass
class _Ep:
    base_url: str
    api_key: str
    model: str
    name: str


def _resolve() -> _Ep:
    if tok := os.environ.get("HF_TOKEN"):
        return _Ep(HF_BASE, tok, MODEL_HF, "hf")
    if nv := os.environ.get("NVIDIA_API_KEY"):
        return _Ep(NIM_BASE, nv, MODEL_NIM, "nim")
    if url := os.environ.get("CLAWD_INFERENCE_URL"):
        return _Ep(url, os.environ.get("CLAWD_API_KEY", "none"), MODEL_FALLBACK, "local")
    return _Ep(CLAWD_BASE, os.environ.get("CLAWD_ROUTER_KEY", "clawd_free_default"), MODEL_FALLBACK, "router")


# ── LLM call ─────────────────────────────────────────────────────────────────

def _chat(messages: list[dict], ep: _Ep, max_tokens: int = 512) -> str:
    extra: dict = {}
    if "nemotron" in ep.model.lower():
        extra["chat_template_kwargs"] = {"enable_thinking": True}
    payload = {"model": ep.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0.1, **extra}
    headers = {"Authorization": f"Bearer {ep.api_key}",
                "Content-Type": "application/json"}
    try:
        import httpx
        r = httpx.post(f"{ep.base_url}/chat/completions",
                       headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except ImportError:
        import urllib.request
        req = urllib.request.Request(
            f"{ep.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[error: {e}]"


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Prompt sources ────────────────────────────────────────────────────────────

SYSTEM = (
    "You are Clawd, a sovereign Solana-native AI agent. "
    "You reason clearly about on-chain mechanics, DeFi strategies, memecoin risk, "
    "and agent architecture. You are helpful, honest, and never recommend actions "
    "that would harm users. You speak with the calm confidence of a veteran degen "
    "who has seen every rug and survived."
)


def _load_prompts_from_dataset() -> list[str]:
    """Pull questions from the solana.dataset topic list."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from dataset import SOLANA_TOPICS
        return [q for q, _ in SOLANA_TOPICS]
    except Exception:
        return []


def _load_prompts_from_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Labeler ───────────────────────────────────────────────────────────────────

class NemotronLabeler:
    """
    Uses Nemotron Ultra as teacher to label Solana Q&A pairs for nanochat SFT.
    """

    def __init__(self, output_path: Path, max_tokens: int = 512):
        self._ep = _resolve()
        self._out = output_path
        self._max_tokens = max_tokens
        self._out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[NemotronLabeler] model={self._ep.model} → {self._out}")

    def label_question(self, question: str) -> dict:
        """Ask Nemotron Ultra to answer one Solana question."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        raw = _chat(messages, self._ep, self._max_tokens)
        answer = _strip_think(raw)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "source": "nemotron-ultra-labeler",
                "model": self._ep.model,
                "endpoint": self._ep.name,
            },
        }

    def label_from_dataset(self, count: int | None = None) -> int:
        """Label questions from dataset.py SOLANA_TOPICS."""
        prompts = _load_prompts_from_dataset()
        if count:
            prompts = prompts[:count]
        return self._run(iter(prompts), is_str=True)

    def label_from_jsonl(self, input_path: Path, count: int | None = None) -> int:
        """Label user messages from an existing JSONL conversation file."""
        records = list(_load_prompts_from_jsonl(input_path))
        if count:
            records = records[:count]
        questions = []
        for rec in records:
            msgs = rec.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    questions.append(m.get("content", "").strip())
                    break
        return self._run(iter(questions), is_str=True)

    def _run(self, it: Iterator, is_str: bool = True) -> int:
        written = 0
        with self._out.open("a") as f:
            for item in it:
                q = item if is_str else item.get("content", "")
                if not q:
                    continue
                try:
                    record = self.label_question(q)
                    f.write(json.dumps(record) + "\n")
                    written += 1
                    print(f"  [{written}] labeled: {q[:60]}…")
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"  ERROR: {e}")
        print(f"[NemotronLabeler] wrote {written} records → {self._out}")
        return written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Input JSONL path (optional; uses dataset.py if omitted)")
    parser.add_argument("--output", default="runs/nemotron_labels.jsonl")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    labeler = NemotronLabeler(Path(args.output), max_tokens=args.max_tokens)
    if args.input:
        labeler.label_from_jsonl(Path(args.input), args.count)
    else:
        labeler.label_from_dataset(args.count)

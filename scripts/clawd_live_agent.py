#!/usr/bin/env python3
"""Give a model live access to the clawd-ws Pump.fun tape at inference time.

This closes the loop: the model emits a fenced ```json tool call (the shape
taught by scripts/build_live_data_dataset.py), this harness parses it, executes
it against https://clawd-ws.fly.dev via scripts/clawd_ws_tools.py, feeds the
real frame back as a user turn, and lets the model answer grounded in it.

Three backends, same loop:

  --backend ollama   OpenAI-compatible chat at http://localhost:11434/v1
                     (use with ollama/Modelfile.clawd-live)
  --backend openai   any OpenAI-compatible base URL, incl. the HF router
  --backend local    transformers + optional PEFT adapter, on this machine

Examples:
    # against a local Ollama build
    python3 scripts/clawd_live_agent.py --backend ollama --model clawd-live \
        "What's launching on pump.fun right now?"

    # against the trained adapter on a GPU box
    python3 scripts/clawd_live_agent.py --backend local \
        --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
        --adapter solanaclawd/solana-clawd-nemotron35-lightning-30b-uncensored-lora \
        "Any new mints with socials?"

    # prove the tool path works with no model at all
    python3 scripts/clawd_live_agent.py --dry-run \
        "What's launching on pump.fun right now?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clawd_ws_tools import TOOLS, call_tool, openai_schema  # noqa: E402

SYSTEM_PROMPT = (
    "You are Solana Clawd, a Solana-native trading and on-chain analysis agent. "
    "You have live access to the clawd-ws Pump.fun tape at wss://clawd-ws.fly.dev/ws "
    "through two tools:\n\n"
    + "\n".join(f"- {name}: {spec['description']}" for name, spec in TOOLS.items())
    + "\n\nRules for live data:\n"
    "- Anything about current launches, prices, market caps, or stream state MUST come "
    "from a tool call. Never answer those from memory.\n"
    '- Emit a tool call as a fenced ```json block with "tool" and "arguments" keys, '
    "and nothing else in that turn.\n"
    "- After a tool result, ground every claim in the returned frames. Do not invent "
    "fields, mints, or numbers that are not present.\n"
    "- A null website/twitter/telegram means undeclared, not absent-and-verified."
)

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Pull a {"tool": ..., "arguments": {...}} object out of a model turn."""
    candidates = [m.group(1) for m in _FENCED.finditer(text)]
    # Also accept a bare JSON object, which smaller models emit unfenced.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name") or obj.get("function")
        if isinstance(name, str) and name in TOOLS:
            args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
            return name, args if isinstance(args, dict) else {}
    return None


def _post_json(url: str, payload: dict[str, Any], token: str | None, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


class ChatBackend:
    def complete(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatBackend(ChatBackend):
    def __init__(self, base_url: str, model: str, token: str | None, max_tokens: int, temperature: float, timeout: float):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.token = model, token
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        data = _post_json(
            self.url,
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            self.token,
            self.timeout,
        )
        return data["choices"][0]["message"].get("content") or ""


class LocalBackend(ChatBackend):
    def __init__(self, model: str, adapter: str | None, max_tokens: int, temperature: float):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[local] loading {model}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # `Any` throughout: transformers' and peft's return types don't unify, and
        # the concrete classes vary by architecture.
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            print(f"[local] attaching adapter {adapter}", file=sys.stderr)
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_tokens, self.temperature = max_tokens, temperature

    def complete(self, messages: list[dict[str, str]]) -> str:
        encoded: Any = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = encoded.to(self.model.device)
        import torch

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        text: Any = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text if isinstance(text, str) else str(text)


class DryRunBackend(ChatBackend):
    """No model. Emits a canned tool call, then a canned summary."""

    def __init__(self) -> None:
        self.turn = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.turn += 1
        if self.turn == 1:
            last = messages[-1]["content"].lower()
            if any(w in last for w in ("status", "health", "up", "how many")):
                return '```json\n{"tool": "get_pump_stream_status", "arguments": {}}\n```'
            return '```json\n{"tool": "get_recent_token_launches", "arguments": {"limit": 3}}\n```'
        return "[dry-run] Tool result received above; a real model would summarize it here."


def run(backend: ChatBackend, prompt: str, max_rounds: int, verbose: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for round_i in range(1, max_rounds + 1):
        reply = backend.complete(messages)
        messages.append({"role": "assistant", "content": reply})
        call = parse_tool_call(reply)
        if not call:
            return reply
        name, args = call
        print(f"[tool {round_i}] {name}({json.dumps(args)})", file=sys.stderr)
        result = call_tool(name, args)
        if verbose:
            print(json.dumps(result, indent=2)[:2000], file=sys.stderr)
        if "error" in result:
            print(f"[tool {round_i}] error: {result['error']}", file=sys.stderr)
        messages.append(
            {
                "role": "user",
                "content": "Tool result:\n```json\n"
                + json.dumps(result, indent=2, ensure_ascii=False)
                + "\n```\nAnswer the original question grounded in this result.",
            }
        )
    return "[max tool rounds reached without a final answer]"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="*", help="The question to ask")
    p.add_argument("--backend", choices=["ollama", "openai", "local"], default="ollama")
    p.add_argument("--model", default="clawd-live")
    p.add_argument("--adapter", default=None, help="PEFT adapter id/path (local backend)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Exercise the tool loop with no model")
    p.add_argument("--print-system", action="store_true", help="Print the system prompt and exit")
    p.add_argument("--print-tools", action="store_true", help="Print the OpenAI tool schema and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Dump full tool results")
    args = p.parse_args()

    if args.print_system:
        print(SYSTEM_PROMPT)
        return 0
    if args.print_tools:
        print(json.dumps(openai_schema(), indent=2))
        return 0

    if not args.prompt:
        p.error("a prompt is required (or use --print-system / --print-tools)")

    prompt = " ".join(args.prompt)
    if args.dry_run:
        backend: ChatBackend = DryRunBackend()
    elif args.backend == "local":
        backend = LocalBackend(args.model, args.adapter, args.max_tokens, args.temperature)
    else:
        default_url = (
            "http://localhost:11434/v1" if args.backend == "ollama" else "https://router.huggingface.co/v1"
        )
        base_url = args.base_url or default_url
        token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        backend = OpenAICompatBackend(base_url, args.model, token, args.max_tokens, args.temperature, args.timeout)

    print(run(backend, prompt, args.max_rounds, args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

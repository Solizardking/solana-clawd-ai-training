#!/usr/bin/env python3
"""Run HauhauCS Qwen3.6 35B A3B GGUF with llama-cpp-python.

This follows the Hugging Face model card usage path for:
HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive

The default file is the smallest listed quant, IQ2_M, which is still about
11 GB. First run downloads it into the Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_REPO = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_FILE = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
DEFAULT_PROMPT = "Describe this image in one sentence."
DEFAULT_IMAGE_URL = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
DEFAULT_CONSTITUTION = Path(__file__).resolve().parents[1] / "docs" / "onchain_constitution.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--no-image", action="store_true", help="Send a text-only chat message")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--constitution", default=str(DEFAULT_CONSTITUTION))
    parser.add_argument("--constitution-mode", choices=["full", "minimal"], default="full")
    parser.add_argument("--no-constitution", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")
    return parser.parse_args()


def load_constitution(path: str, mode: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if mode == "minimal":
        start = text.find("## 19. Minimal Agent Runtime Prompt")
        end = text.find("## 20. Closing Directive")
        if start != -1 and end != -1:
            return text[start:end].strip()
    return text


def main() -> int:
    args = parse_args()
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError:
        print(
            "llama_cpp is not installed. Install it with:\n"
            "  python3 -m pip install llama-cpp-python\n",
            file=sys.stderr,
        )
        return 2

    llm = Llama.from_pretrained(
        repo_id=args.repo_id,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    messages = []
    if not args.no_constitution:
        messages.append(
            {
                "role": "system",
                "content": load_constitution(args.constitution, args.constitution_mode),
            }
        )
    if args.no_image:
        user_content = args.prompt
    else:
        user_content = [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": args.image_url}},
        ]
    messages.append({"role": "user", "content": user_content})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.json:
        print(json.dumps(response, indent=2))
        return 0

    message = response["choices"][0]["message"]
    print(message.get("content", "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

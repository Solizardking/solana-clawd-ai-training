#!/usr/bin/env python3
"""Run Clawd Fable with the Transformers chat-template API.

Examples:
  python3 scripts/clawd_fable_inference.py --model AliesTaha/fable-traces
  python3 scripts/clawd_fable_inference.py --model solanaclawd/clawd-fable
  python3 scripts/clawd_fable_inference.py --adapter outputs/clawd-fable-lora-local
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_PROMPT = "Who are you?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="solanaclawd/clawd-fable")
    parser.add_argument("--base-model", default="AliesTaha/fable-traces")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path or Hub repo")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--pipeline", action="store_true", help="Use transformers.pipeline instead of manual generate")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> str:
    pipe = pipeline("text-generation", model=args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    output = pipe(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    return str(output[0]["generated_text"])


def run_manual(args: argparse.Namespace) -> str:
    model_id = args.base_model if args.adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    text = run_pipeline(args) if args.pipeline and not args.adapter else run_manual(args)
    print(text.strip())


if __name__ == "__main__":
    main()

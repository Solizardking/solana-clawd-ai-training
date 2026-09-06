#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1.0",
#   "transformers>=5.12.0",
#   "accelerate>=1.14.0",
#   "peft>=0.19.1",
#   "huggingface_hub>=1.19.0",
#   "safetensors>=0.5.0",
# ]
# ///
"""Merge a LoRA adapter into a full Transformers model folder.

This produces the final model layout expected by:

  AutoTokenizer.from_pretrained("solanaclawd/clawd-fable")
  AutoModelForCausalLM.from_pretrained("solanaclawd/clawd-fable")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def write_model_card(path: Path, args: argparse.Namespace) -> None:
    model_id = args.hub_model_id or path.name
    path.joinpath("README.md").write_text(
        f"""---
license: apache-2.0
base_model: {args.base_model}
datasets:
  - armand0e/claude-fable-5-claude-code
  - Glint-Research/Fable-5-traces
tags:
  - solana
  - clawd
  - fable
  - qwen
  - text-generation
pipeline_tag: text-generation
---

# Clawd Fable

Full merged Transformers model for the Clawd Fable lane.

- Base model: `{args.base_model}`
- LoRA adapter: `{args.adapter}`
- Model repo: `{model_id}`
- Training data: `armand0e/claude-fable-5-claude-code` combined with `Glint-Research/Fable-5-traces`

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}", device_map="auto")
messages = [
    {{"role": "user", "content": "Who are you?"}},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=160)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Safety

This is a research/developer model for Solana agent coding and reasoning. Live
trading, wallet, or deployment actions must stay behind separate execution
clients, explicit operator approval, and risk checks.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.torch_dtype)
    device_map = None if args.device_map.lower() in {"none", "cpu"} else args.device_map

    print(f"[merge] loading base: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"[merge] attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=dtype)

    print("[merge] merge_and_unload")
    model = model.merge_and_unload()

    print(f"[merge] saving full model: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_model_card(output_dir, args)

    if args.push:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required with --push")
        from huggingface_hub import HfApi

        print(f"[merge] uploading full model to {args.hub_model_id}")
        api = HfApi()
        api.create_repo(args.hub_model_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.hub_model_id,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message=f"release: upload merged {args.hub_model_id}",
        )

    print("[merge] done")


if __name__ == "__main__":
    main()

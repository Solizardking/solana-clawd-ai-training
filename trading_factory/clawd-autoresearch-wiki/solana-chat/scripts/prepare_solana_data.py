#!/usr/bin/env python3
"""
Generate Solana-specific SFT and eval data for solana-chat.

Produces JSONL training files from the solana.dataset module that can
be consumed by the SFT pipeline (chat_sft.py).

Usage:
    python -m scripts.prepare_solana_data --count 50 --output data/solana_chat_seed.jsonl
"""
from __future__ import annotations

import argparse
import os

from solana.dataset import SolanaDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=20,
                        help="Number of training SFT pairs to generate")
    parser.add_argument("--eval-count", type=int, default=10,
                        help="Number of eval pairs to generate")
    parser.add_argument("--train-output", default="data/solana_chat_seed.jsonl",
                        help="Output path for training data")
    parser.add_argument("--eval-output", default="data/solana_chat_eval.jsonl",
                        help="Output path for eval data")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    ds = SolanaDataset()
    train = ds.generate_sft_pairs(count=args.train_count)
    eval_ = ds.generate_eval_pairs(count=args.eval_count)

    ds.to_jsonl(train, args.train_output)
    ds.to_jsonl(eval_, args.eval_output)

    print(f"\nGenerated {len(train)} training + {len(eval_)} eval examples.")
    print(f"  Train: {args.train_output}")
    print(f"  Eval:  {args.eval_output}")


if __name__ == "__main__":
    main()
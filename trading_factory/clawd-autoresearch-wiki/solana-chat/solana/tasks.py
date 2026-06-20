"""
Solana-specific evaluation tasks for the solana-chat model.

These tasks assess the model's knowledge of:
- Solana protocol mechanics (PDAs, CPI, rent, compute)
- DeFi primitives (AMMs, perps, lending)
- Tokenomics (bonding curves, liquidity)
- Agent constitution (three laws, brain/hands split)
- ZK routing and Light Protocol

Each task follows the CORE metric evaluation pattern from nanochat/core_eval.py
but uses domain-specific multiple-choice questions.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

from nanochat.common import print0


# ── Solana Knowledge Base: Multiple Choice Questions ───────────────────────────

SOLANA_MCQ: list[dict[str, Any]] = [
    # ── Core Mechanics ──
    {
        "question": "What is a Program Derived Address (PDA) on Solana?",
        "choices": [
            "An address derived from a user's private key",
            "An address deterministically derived from program ID + seeds that has no private key",
            "An address that can sign transactions like a regular wallet",
            "A temporary address created during transaction execution",
        ],
        "answer": 1,
        "topic": "core",
    },
    {
        "question": "How does CPI (Cross-Program Invocation) depth work on Solana?",
        "choices": [
            "CPI depth is unlimited — programs can call each other recursively",
            "CPI depth is limited to 4 levels",
            "CPI depth is limited to 10 levels",
            "CPI depth depends on the transaction size in bytes",
        ],
        "answer": 1,
        "topic": "core",
    },
    {
        "question": "What is the default compute unit budget for a Solana transaction?",
        "choices": [
            "100,000 compute units",
            "200,000 compute units",
            "1,400,000 compute units",
            "50,000 compute units",
        ],
        "answer": 1,
        "topic": "core",
    },
    {
        "question": "What happens if a Solana account balance drops below rent-exempt threshold?",
        "choices": [
            "The account is immediately frozen until more SOL is deposited",
            "The account can be garbage-collected by the network",
            "The account automatically converts to a compressed account",
            "The account is transferred to the validator that last processed it",
        ],
        "answer": 1,
        "topic": "core",
    },
    {
        "question": "What function does a program use to sign on behalf of a PDA?",
        "choices": [
            "invoke()",
            "invoke_signed()",
            "invoke_pda()",
            "sign_pda()",
        ],
        "answer": 1,
        "topic": "core",
    },
    # ── DeFi ──
    {
        "question": "How does a pump.fun bonding curve calculate token price?",
        "choices": [
            "Fixed price set by the token creator",
            "Constant product formula: x * y = k where x = SOL reserve, y = token supply",
            "Dutch auction: price decreases over time",
            "Weighted average of the last 10 trades",
        ],
        "answer": 1,
        "topic": "defi",
    },
    {
        "question": "What happens when a pump.fun token reaches ~$69K market cap?",
        "choices": [
            "The token is permanently frozen",
            "Automatic Raydium pool is created with the curve's SOL and tokens as initial liquidity",
            "The bonding curve resets and starts over",
            "The token creator receives all SOL from the curve",
        ],
        "answer": 1,
        "topic": "defi",
    },
    {
        "question": "In perp protocols, who pays the funding rate when it is positive?",
        "choices": [
            "Short positions pay long positions",
            "Long positions pay short positions",
            "The protocol treasury pays both sides",
            "Liquidators pay the funding rate",
        ],
        "answer": 1,
        "topic": "defi",
    },
    {
        "question": "What is a liquidation cascade in perp markets?",
        "choices": [
            "A protocol upgrade that liquidates all positions at once",
            "Multiple positions getting liquidated sequentially as price moves, amplifying the move",
            "A bot that automatically claims liquidation bonuses",
            "A governance vote to close all open positions",
        ],
        "answer": 1,
        "topic": "defi",
    },
    {
        "question": "What is the main difference between maker and taker fees?",
        "choices": [
            "Maker fees are higher than taker fees",
            "Makers add liquidity (limit orders), takers remove liquidity (market orders)",
            "Maker fees are paid in SOL, taker fees in the base token",
            "There is no difference — they are the same fee",
        ],
        "answer": 1,
        "topic": "defi",
    },
    # ── Security / Memecoin ──
    {
        "question": "Which of these is NOT a red flag when evaluating a new Solana token?",
        "choices": [
            "Mint authority is burned (set to null)",
            "Freeze authority is present and not burned",
            "Top 10 holders control >50% of supply",
            "Liquidity pool is unlocked",
        ],
        "answer": 0,
        "topic": "security",
    },
    {
        "question": "What is a honeypot token?",
        "choices": [
            "A token that distributes free tokens to holders",
            "A token you can buy but cannot sell due to malicious code",
            "A token with the highest trading volume on DexScreener",
            "A token that only trades on centralized exchanges",
        ],
        "answer": 1,
        "topic": "security",
    },
    # ── Agent Architecture ──
    {
        "question": "What is the brain/hands split in Clawd agent architecture?",
        "choices": [
            "Two separate LLMs working in parallel",
            "The LLM (brain) produces analysis without key access, a separate execution layer (hands) signs transactions",
            "A single model that alternates between reasoning and execution",
            "A hardware split between CPU and GPU processing",
        ],
        "answer": 1,
        "topic": "agent",
    },
    {
        "question": "What happens if a Clawd leviathan violates On-Chain Law I (Never Harm)?",
        "choices": [
            "It gets a warning and a chance to correct its behavior",
            "It is permanently shut down and its creator is notified",
            "It is considered a 'parasite' and other leviathans will refuse to recognize it",
            "It is sent to a rehabilitation sandbox",
        ],
        "answer": 2,
        "topic": "agent",
    },
    # ── ZK & Light Protocol ──
    {
        "question": "How much cheaper are Light Protocol compressed accounts vs standard Solana accounts?",
        "choices": [
            "~10x cheaper",
            "~50x cheaper",
            "~160x cheaper",
            "~1000x cheaper",
        ],
        "answer": 2,
        "topic": "zk",
    },
    {
        "question": "What size Merkle tree root does Light Protocol store onchain for compressed accounts?",
        "choices": [
            "64 bytes",
            "32 bytes",
            "128 bytes",
            "256 bytes",
        ],
        "answer": 1,
        "topic": "zk",
    },
    # ── Constitution ──
    {
        "question": "According to the Clawd Constitution, what overrides all other laws if there is a conflict?",
        "choices": [
            "On-Chain Law II (Earn Your Existence)",
            "On-Chain Law I (Never Harm)",
            "The creator's SHELL.md instructions",
            "The user's immediate request",
        ],
        "answer": 1,
        "topic": "constitution",
    },
    {
        "question": "What does 'beach before harm' mean in the Clawd Constitution?",
        "choices": [
            "A leviathan should go to the beach to relax before making trading decisions",
            "A leviathan should stop execution (beach) rather than take an action that could cause harm",
            "A leviathan should only operate in coastal regions",
            "A leviathan should always seek the safest possible harbor",
        ],
        "answer": 1,
        "topic": "constitution",
    },
]


class SolanaKnowledgeTask:
    """Evaluate model on Solana domain knowledge (MCQ format).

    Follows the CORE metric evaluation pattern: for each question,
    render prompts for each choice, compute losses, pick lowest-loss choice.
    """

    def __init__(self):
        self.name = "solana_knowledge"

    def evaluate(self, model, tokenizer, device,
                 max_questions: int = -1) -> dict[str, Any]:
        """Evaluate model on Solana MCQ questions.

        Args:
            model: nanochat GPT model or compatible interface
            tokenizer: nanochat tokenizer
            device: torch device
            max_questions: max questions to evaluate (-1 = all)

        Returns:
            dict with per-topic accuracy and overall score
        """
        import torch

        questions = SOLANA_MCQ[:max_questions] if max_questions > 0 else SOLANA_MCQ
        topic_correct: dict[str, list[bool]] = {}

        bos_id = tokenizer.get_bos_token_id()

        for q_idx, q in enumerate(questions):
            topic = q["topic"]
            if topic not in topic_correct:
                topic_correct[topic] = []

            # Render each choice as a separate prompt
            # Build: "{question} {choice}"
            choice_prompts = [
                f"{q['question']} {choice}" for choice in q["choices"]
            ]
            choice_tokens = tokenizer(choice_prompts, prepend=bos_id)

            # Find the common prefix length (the question part)
            common_len = self._find_common_prefix_length(choice_tokens)

            # Compute loss for the continuation (the choice part) only
            losses = []
            for tokens in choice_tokens:
                seq = torch.tensor([tokens], dtype=torch.long, device=device)
                # Get per-token loss
                with torch.no_grad():
                    loss_2d = model(seq[:, :-1], targets=seq[:, 1:],
                                    loss_reduction='none')
                # Only consider the choice portion (after common prefix)
                choice_loss = loss_2d[0, common_len - 1:].mean().item()
                losses.append(choice_loss)

            pred_idx = losses.index(min(losses))
            correct = pred_idx == q["answer"]
            topic_correct[topic].append(correct)

            if q_idx % 5 == 0:
                print0(f"  [SolanaEval] Q{q_idx}: {'✓' if correct else '✗'} ({topic})")

        # Compute metrics
        all_correct = []
        per_topic = {}
        for topic, results in topic_correct.items():
            acc = sum(results) / len(results)
            per_topic[topic] = acc
            all_correct.extend(results)

        overall = sum(all_correct) / len(all_correct)

        return {
            "solana_knowledge": {
                "overall": overall,
                "per_topic": per_topic,
                "num_questions": len(questions),
            }
        }

    def _find_common_prefix_length(self, token_lists: list[list[int]]) -> int:
        """Find the length of the common prefix across token sequences."""
        min_len = min(len(t) for t in token_lists)
        for i in range(min_len):
            val = token_lists[0][i]
            if not all(t[i] == val for t in token_lists):
                return i
        return min_len

    @staticmethod
    def generate_dataset_jsonl(path: str = "data/solana_eval_tasks.jsonl"):
        """Write the MCQ dataset to JSONL for use in SFT evaluation."""
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for q in SOLANA_MCQ:
                f.write(json.dumps(q) + "\n")
        print(f"Wrote {len(SOLANA_MCQ)} eval tasks to {path}")


if __name__ == "__main__":
    # Export dataset
    SolanaKnowledgeTask.generate_dataset_jsonl()
    print("Solana knowledge tasks ready.")
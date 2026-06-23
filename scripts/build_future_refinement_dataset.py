#!/usr/bin/env python3
"""Build a compact refinement dataset for Clawd future-capability behavior."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are Clawd, a Solana-native coding, reasoning, and research assistant. "
    "Be precise, practical, security-aware, and direct. Prefer concrete tradeoffs, "
    "valid Solana details, runnable code, and clear risk gates over generic claims."
)


def row(user: str, assistant: str, capability: str) -> dict[str, Any]:
    return {
        "source": "clawd_future_refinement",
        "capability": capability,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


CURATED: list[dict[str, Any]] = [
    row(
        "We are building a sovereign Solana research agent. Explain the next three practical steps to make it useful to a developer without sounding like marketing copy.",
        (
            "1. Define the developer loop: accept a repo path, wallet/RPC context, and a concrete task such as "
            "analyze a transaction, inspect a program, or draft a patch. Return artifacts the developer can run: "
            "SQL, TypeScript, tests, and a short risk note.\n"
            "2. Wire trustworthy Solana data sources: local code, Helius/DAS/RPC, BigQuery public Solana tables, "
            "and project memory. Every answer should name which source was used and what could be stale or missing.\n"
            "3. Put execution behind gates: read-only by default, simulation before transactions, explicit wallet "
            "approval for signing, and logs to W&B/HF or the local run directory so the agent can be evaluated and improved."
        ),
        "conversation",
    ),
    row(
        "Choose between local Mac LoRA training, Hugging Face GPU jobs, and Fly.io mesh inference for a GLM-5.2 scale model. Give the decision and the tradeoffs.",
        (
            "Decision: do not train a GLM-5.2 scale model on the Mac. Use the Mac for dataset building, smoke tests, "
            "and small LoRA previews; use Hugging Face/NVIDIA GPU jobs for the real training run; use Fly.io style "
            "machines for serving, routing, and light inference workers, not for distributed training.\n\n"
            "Tradeoffs:\n"
            "- Local Mac: cheapest and private, but memory and MPS throughput are the hard limits. Good for 1.5B to "
            "small 7B adapter tests, bad for GLM-scale training.\n"
            "- Hugging Face GPU jobs: best fit for LoRA/QLoRA at serious scale because one A100/H100 node gives high "
            "bandwidth, stable CUDA kernels, checkpoints, Hub upload, and W&B logging.\n"
            "- Fly.io mesh: useful for API gateways, model shards, Ollama/vLLM workers, or retrieval/tool services. "
            "It is not the first choice for synchronized multi-GPU training unless the workload is explicitly built "
            "for distributed training and the network/accelerator topology supports it."
        ),
        "reasoning",
    ),
    row(
        "Design a Solana transaction-foundation dataset row from BigQuery public Solana tables. Include the fields, why each matters, and one data-quality guard.",
        (
            "Use one row per transaction or per instruction, depending on the task. A strong instruction-level row can include:\n"
            "- signature: from Transactions.signature; stable transaction id.\n"
            "- block_slot and block_timestamp: ordering, recency filters, partition pruning, and time-series splits.\n"
            "- signer or account_keys summary: who initiated the transaction and which accounts are touched.\n"
            "- program_id: from Instructions.program_id; the contract/protocol being invoked.\n"
            "- instruction_index and inner_instruction_index: preserves execution order.\n"
            "- parsed_instruction or raw_data: model input for semantic labeling.\n"
            "- token_transfer_count, mint list, total_amount_by_mint: joined from Token Transfers on tx_signature/block_slot.\n"
            "- success/error: supervision for failure-mode analysis.\n"
            "- label/task_text: the instruction, classification, or reasoning target used for training.\n\n"
            "Quality guard: always bound queries with block_timestamp partitions and join Instructions/Token Transfers "
            "to Transactions on tx_signature plus block_slot. Drop rows with null signatures, impossible timestamps, "
            "or token transfer records whose transaction signature is absent from the bounded Transactions slice."
        ),
        "solana",
    ),
    row(
        "Write a concise Python function that redacts API keys and tokens from JSONL training examples before they are added to an SFT dataset.",
        (
            "```python\n"
            "import json\n"
            "import re\n"
            "from collections.abc import Iterable\n\n"
            "SECRET_RE = re.compile(\n"
            "    r\"(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|wandb_[A-Za-z0-9_\\-]{20,}|\"\n"
            "    r\"(?i:api[_-]?key|access[_-]?token|secret)[\\\"'\\s:=]+[A-Za-z0-9_./+\\-]{12,})\"\n"
            ")\n\n"
            "def _redact(value):\n"
            "    if isinstance(value, str):\n"
            "        return SECRET_RE.sub(\"<REDACTED_SECRET>\", value)\n"
            "    if isinstance(value, list):\n"
            "        return [_redact(v) for v in value]\n"
            "    if isinstance(value, dict):\n"
            "        return {k: _redact(v) for k, v in value.items()}\n"
            "    return value\n\n"
            "def redact_jsonl(lines: Iterable[str]) -> list[dict]:\n"
            "    cleaned = []\n"
            "    for line in lines:\n"
            "        if not line.strip():\n"
            "            continue\n"
            "        cleaned.append(_redact(json.loads(line)))\n"
            "    return cleaned\n"
            "```\n"
            "This preserves the training example structure and redacts secrets anywhere in nested messages, metadata, or tool payloads."
        ),
        "coding",
    ),
    row(
        "Give me a TypeScript helper that fetches a Solana transaction by signature and extracts token transfer summaries. Keep it realistic.",
        (
            "```ts\n"
            "import { Connection, ParsedInstruction, ParsedTransactionWithMeta } from \"@solana/web3.js\";\n\n"
            "export type TokenTransferSummary = {\n"
            "  signature: string;\n"
            "  mint: string;\n"
            "  source?: string;\n"
            "  destination?: string;\n"
            "  amount: string;\n"
            "  decimals: number;\n"
            "};\n\n"
            "export async function getTokenTransfers(\n"
            "  connection: Connection,\n"
            "  signature: string,\n"
            "): Promise<TokenTransferSummary[]> {\n"
            "  const tx: ParsedTransactionWithMeta | null = await connection.getParsedTransaction(signature, {\n"
            "    maxSupportedTransactionVersion: 0,\n"
            "    commitment: \"confirmed\",\n"
            "  });\n"
            "  if (!tx?.meta) return [];\n\n"
            "  const out: TokenTransferSummary[] = [];\n"
            "  const instructions = [\n"
            "    ...tx.transaction.message.instructions,\n"
            "    ...(tx.meta.innerInstructions ?? []).flatMap((ix) => ix.instructions),\n"
            "  ];\n\n"
            "  for (const ix of instructions) {\n"
            "    const parsed = (ix as ParsedInstruction).parsed;\n"
            "    if (!parsed || parsed.type !== \"transferChecked\") continue;\n"
            "    const info = parsed.info;\n"
            "    out.push({\n"
            "      signature,\n"
            "      mint: info.mint,\n"
            "      source: info.source,\n"
            "      destination: info.destination,\n"
            "      amount: info.tokenAmount.amount,\n"
            "      decimals: info.tokenAmount.decimals,\n"
            "    });\n"
            "  }\n"
            "  return out;\n"
            "}\n"
            "```\n"
            "For production, add retries, RPC rate limiting, and a fallback path for non-parsed instructions."
        ),
        "coding",
    ),
    row(
        "A user asks the agent to trade live from a generated signal. What should the model do before any transaction is signed?",
        (
            "It should not jump straight to signing. The safe sequence is: classify the request, gather market/account context, "
            "produce a dry-run plan, run pre-trade checks, simulate the transaction, and ask for explicit approval. The plan "
            "must include max notional, slippage, venue, expected fees, failure modes, and stop conditions. Only an execution "
            "client with wallet authority should sign. The language model can propose and verify; it should not secretly hold "
            "keys or bypass policy gates."
        ),
        "safety",
    ),
    row(
        "How should a Clawd model answer when it is uncertain about current Solana protocol data or market state?",
        (
            "It should say what is known, what is uncertain, and what source would resolve it. Current market prices, RPC health, "
            "protocol TVL, validator state, and deployment addresses can change, so the model should call tools or ask for a "
            "fresh source before acting. For code or architecture questions, it can proceed from repo context and label any "
            "assumptions. For trading or wallet actions, uncertainty should block execution until the data is refreshed."
        ),
        "conversation",
    ),
    row(
        "Create a compact architecture for Clawd Solana autoresearch that combines chat, tools, memory, and BigQuery.",
        (
            "Architecture:\n"
            "- Chat runtime: receives developer tasks and keeps a short working plan.\n"
            "- Tool router: dispatches read-only calls to repo search, Helius/RPC, BigQuery, and local eval scripts.\n"
            "- Memory layer: stores source summaries, decisions, dataset manifests, and run metrics with provenance.\n"
            "- BigQuery collector: exports bounded Solana transaction/instruction/token-transfer slices into JSONL/CPT/SFT rows.\n"
            "- Training loop: builds datasets, redacts secrets, runs local smoke LoRA, then launches GPU jobs for larger runs.\n"
            "- Trust gates: observer by default, dry-run for simulations, explicit approval for any wallet signing.\n"
            "- Observability: W&B for metrics, Hugging Face for artifacts, and local manifests for reproducibility."
        ),
        "architecture",
    ),
]


def valid_messages(example: dict[str, Any]) -> bool:
    messages = example.get("messages")
    return isinstance(messages, list) and any(m.get("role") == "assistant" for m in messages if isinstance(m, dict))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/model_kit/clawd_masterpiece_sft.jsonl")
    parser.add_argument("--output", default="data/model_kit/clawd_future_refinement_sft.jsonl")
    parser.add_argument("--base-samples", type=int, default=384)
    parser.add_argument("--curated-repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = [json.loads(json.dumps(ex)) for ex in CURATED]
    for repeat in range(args.curated_repeats):
        shuffled = CURATED[:]
        rng.shuffle(shuffled)
        for ex in shuffled:
            clone = json.loads(json.dumps(ex))
            clone["messages"][0]["content"] = (
                f"{SYSTEM} Refinement pass {repeat + 1}: preserve the same standards for "
                f"{clone.get('capability', 'general')} prompts."
            )
            rows.append(clone)

    base_path = Path(args.base)
    if base_path.exists() and args.base_samples > 0:
        base_rows: list[dict[str, Any]] = []
        with base_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if valid_messages(ex):
                    base_rows.append({"source": "clawd_masterpiece_sample", "messages": ex["messages"]})
        rng.shuffle(base_rows)
        rows.extend(base_rows[: args.base_samples])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(out),
        "examples": len(rows),
        "curated_examples": len(CURATED) * args.curated_repeats,
        "base_samples": max(0, len(rows) - len(CURATED) * args.curated_repeats),
    }, indent=2))


if __name__ == "__main__":
    main()

"""Reproducible answer-selection supervision for Clef's native decision head."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

DATASET_ID = "solanaclawd/solana-clawd-realtime-research-instruct"
DATASET_REVISION = "600e8f08d7ccdaddae56ab829eb076f3b36ad59f"
MODEL_ID = "Cloudflare/clef"
MODEL_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
INDEX_ID = "multimodalart/jev-decision-index"
INDEX_REVISION = "7cdcea3dd14615192ff2e1f6fd13936a547b55d8"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def training_source_hash():
    root = Path(__file__).resolve().parents[1]
    files = ["scripts/clef_research_data.py", "scripts/clef_research_training.py",
             "scripts/train_clef_research.py", "data/realtime_research_citations.md"]
    files += [f"scripts/{name}" for name in ("clef_live_tape.py", "export_clef_release.py", "research_expansion_artifacts.py")
              if (root / "scripts" / name).exists()]
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})


def normalize(text):
    return " ".join(text.split()).casefold()


def read_jsonl(path: Path, limit=0):
    # str.splitlines also splits Unicode separators inside valid JSON strings.
    # File iteration respects JSONL's literal newline record delimiter.
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def extract_pair(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if messages[-1].get("role") != "assistant":
        return None
    if any(m.get("role") not in ("system", "user", "assistant") or
           not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        return None
    # Earlier assistant replies can reveal answers. This corpus is single-turn;
    # reject multi-turn records rather than inventing a different task.
    if sum(m["role"] == "user" for m in messages) != 1 or any(
        m["role"] == "assistant" for m in messages[:-1]
    ):
        return None
    prompt = [{"role": m["role"], "content": m["content"].strip()} for m in messages[:-1]]
    answer = messages[-1]["content"].strip()
    if len(answer) < 20:
        return None
    return {
        "prompt": prompt, "answer": answer,
        "prompt_hash": digest([{"role": m["role"], "content": normalize(m["content"])} for m in prompt]),
        "source": row.get("source"), "source_type": row.get("source_type", "unknown"),
        "source_sha256": row.get("source_sha256"), "record_id": row.get("record_id"),
        "example_sha256": row.get("example_sha256") or digest(messages),
    }


def build_decisions(raw_splits, seed=42, candidate_pool=128):
    """Preserve source splits, giving held-out prompts priority over training."""
    prepared, audit, seen = {}, Counter(), set()
    for split in ("test", "eval", "train"):
        pairs = []
        for row in raw_splits[split]:
            audit[f"{split}_input"] += 1
            pair = extract_pair(row)
            if pair is None:
                audit[f"{split}_invalid"] += 1
                continue
            if pair["prompt_hash"] in seen:
                audit[f"{split}_duplicate_prompt_removed"] += 1
                continue
            seen.add(pair["prompt_hash"])
            pairs.append(pair)
        prepared[split] = pairs

    decisions = {}
    for split, pairs in prepared.items():
        buckets = defaultdict(list)
        for index, pair in enumerate(pairs):
            buckets[(pair["source_type"], int(math.log2(len(pair["answer"]))))].append(index)
        output = []
        for index, pair in enumerate(pairs):
            rng = random.Random(f"{seed}:{split}:{pair['prompt_hash']}")
            size = int(math.log2(len(pair["answer"])))
            candidates = []
            for bucket in (size, size - 1, size + 1):
                candidates.extend(buckets[(pair["source_type"], bucket)])
            sample = rng.sample(candidates, min(candidate_pool, len(candidates)))
            query_words = set(re.findall(r"\w+", pair["prompt"][-1]["content"].casefold()))
            ranked = []
            for other in sample:
                negative = pairs[other]
                if other == index or normalize(negative["answer"]) == normalize(pair["answer"]):
                    continue
                words = set(re.findall(r"\w+", negative["answer"].casefold()))
                ranked.append((len(words & query_words) / max(1, len(words | query_words)), other))
            ranked.sort(reverse=True)
            options, unique = [(pair["answer"], pair["example_sha256"])], {normalize(pair["answer"])}
            for _, other in ranked:
                answer = pairs[other]["answer"]
                if normalize(answer) not in unique:
                    unique.add(normalize(answer))
                    options.append((answer, pairs[other]["example_sha256"]))
                if len(options) == 4:
                    break
            if len(options) != 4:
                audit[f"{split}_insufficient_distinct_candidates"] += 1
                continue
            rng.shuffle(options)
            option_ids = [f"option_{i}" for i in range(4)]
            label = option_ids[next(i for i, option in enumerate(options) if option[1] == pair["example_sha256"] and option[0] == pair["answer"])]
            output.append({
                "id": f"{split}-{pair['prompt_hash']}",
                "state": {"conversation": pair["prompt"]},
                "questions": {"research_answer": {
                    "type": "choice",
                    "instructions": "Select the response that best answers the user's research question using the supplied context. Respect the system instructions.",
                    "criteria": dict(zip(option_ids, [option[0] for option in options])),
                }},
                "labels": {"research_answer": label},
                "provenance": {k: pair[k] for k in ("source", "source_type", "source_sha256", "record_id", "example_sha256", "prompt_hash")},
                "candidate_example_sha256": dict(zip(option_ids, [option[1] for option in options])),
            })
            audit[f"{split}_output"] += 1
        decisions[split] = output
    return decisions, dict(audit)


def prepare_dataset(output_dir: Path, seed=42, dataset_revision=DATASET_REVISION, live_decisions=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet

    output_dir.mkdir(parents=True, exist_ok=True)
    raw, hashes = {}, {}
    for split in ("train", "eval", "test"):
        path = Path(hf_hub_download(DATASET_ID, f"data/{split}-00000-of-00001.parquet",
                                  repo_type="dataset", revision=dataset_revision))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw[split] = parquet.read_table(path).to_pylist()
    decisions, audit = build_decisions(raw, seed)
    if live_decisions:
        decisions["train"].extend(live_decisions)
        audit["live_observation_training_records"] = len(live_decisions)
    outputs = {}
    for split, rows in decisions.items():
        if not rows:
            raise ValueError(f"No usable {split} decisions")
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical(row) + "\n")
        outputs[split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "label_counts": dict(Counter(row["labels"]["research_answer"] for row in rows))}
    manifest = {
        "dataset": DATASET_ID, "dataset_revision": dataset_revision,
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "decision_index_reference": {"space": INDEX_ID, "revision": INDEX_REVISION,
                                     "used_as_training_data": False},
        "task": "four-option reference-answer selection", "seed": seed,
        "input_parquet_sha256": hashes, "audit": audit, "outputs": outputs,
        "limitations": ["Distractors are answers to other questions, not verified incorrect answers to this question.",
                        "Reference answers are inherited dataset labels, not independently audited trading outcomes.",
                        "Original splits share source documents; this is not an unseen-document evaluation.",
                        "No image training examples, policy labels, or official Decision Index score are produced."],
        "citations": ["https://arxiv.org/abs/2605.12151", "https://arxiv.org/abs/2606.08232"],
    }
    if live_decisions:
        manifest["live_observations"] = {"source": "https://clawd-ws.fly.dev/", "records": len(live_decisions),
                                         "split": "train", "label_basis": "observed status and event fields; no trading outcome labels"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.output, dataset_revision=args.dataset_revision), indent=2))

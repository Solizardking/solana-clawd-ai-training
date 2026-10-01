"""Reproducible answer-selection supervision for Clef's native decision head."""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
import copy
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

_BYTE_ARRAY = re.compile(r"\[(?:\s*\d{1,3}\s*,){31,}\s*\d{1,3}\s*\]")
_SIGNING_CONTEXT = re.compile(
    r"\b(?:secret[ _-]?(?:key|seed)|private[ _-]?(?:key|seed)|signing[ _-]?(?:key|seed)|key[ _-]?pair|seed|"
    r"from_?seed|fromSecretKey|from_secret_key)\b", re.IGNORECASE,
)
_PUBLIC_ARRAY_TARGET = re.compile(
    r"\b(?:public[ _-]?key|pubkey|signature|hash|digest)\b[\"']?\s*[:=]\s*"
    r"(?:(?:new\s+)?(?:Uint8Array(?:\.from)?|bytes|bytearray)\s*\(\s*)?\Z|"
    r"\b(?:PublicKey|Pubkey|Ed25519PublicKey)\.(?:from_?bytes|new_from_array)\s*\(\s*\Z",
    re.IGNORECASE,
)
_ENCODED_SIGNING_LITERAL = re.compile(
    r"\b(?:secret[ _-]?(?:key|seed)|private[ _-]?(?:key|seed)|signing[ _-]?(?:key|seed)|key[ _-]?pair|seed)\b"
    r"[\"']?\s*[:=]\s*(?P<quote>[\"'])(?P<value>[1-9A-HJ-NP-Za-km-z]{43,90}|"
    r"(?:0x)?[0-9a-fA-F]{64,128})(?P=quote)", re.IGNORECASE,
)


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


def model_input_exclusion_reason(value):
    """Return a safe category for literal credentials/signing data, never values.

    Credential-prefix patterns identify actual payload shapes. Byte arrays need
    signing/seed context: ordinary public account bytes must remain usable.
    Variable expressions and environment placeholders contain no key bytes.
    The caller scans only model-facing input, not provenance or source metadata.
    """
    from research_expansion_artifacts import _SECRET_PATTERNS, _placeholder

    def context_name(context):
        # JSON field names commonly use underscores or camelCase. Preserve
        # their semantic boundaries when looking for signing-key context.
        name = ".".join(context)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
        return re.sub(r"[_-]+", " ", name)

    def strings(item, context=()):
        if isinstance(item, str):
            yield item
            name = context_name(context)
            if context and _SIGNING_CONTEXT.search(name):
                yield name + "=" + canonical(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                yield from strings(child, context + (str(key),))
        elif isinstance(item, (list, tuple)):
            if context and len(item) in (32, 64) and all(type(byte) is int for byte in item):
                # Preserve field context without joining unrelated messages or
                # answer candidates into an artificial private-key context.
                yield context_name(context) + "=" + canonical(item)
            else:
                for child in item:
                    yield from strings(child, context)

    for text in strings(value):
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            return "credential_payload"
        if any(not _placeholder(match.group("value")) for match in _ENCODED_SIGNING_LITERAL.finditer(text)):
            return "encoded_signing_literal"
        for match in _BYTE_ARRAY.finditer(text):
            try:
                payload = ast.literal_eval(match.group())
            except (SyntaxError, ValueError):
                continue
            if len(payload) not in (32, 64) or any(type(byte) is not int or not 0 <= byte <= 255 for byte in payload):
                continue
            prefix = text[max(0, match.start() - 240):match.start()]
            if _PUBLIC_ARRAY_TARGET.search(prefix):
                continue
            context = prefix + text[match.end():match.end() + 240]
            if _SIGNING_CONTEXT.search(context):
                return "signing_byte_array"
    return None


def _record_security_exclusion(audit, split, reason, *, live=False):
    key = f"{split}_{'live_' if live else ''}excluded_{reason}"
    audit[key] = audit.get(key, 0) + 1


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
            reason = model_input_exclusion_reason(row.get("messages"))
            if reason:
                _record_security_exclusion(audit, split, reason)
                continue
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
        # Ranking revisits the same candidate responses many times. Compute
        # text normalization/word sets once, retaining identical overlap scores.
        normalized_answers = [normalize(pair["answer"]) for pair in pairs]
        answer_words = [frozenset(re.findall(r"\w+", pair["answer"].casefold())) for pair in pairs]
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
                if other == index or normalized_answers[other] == normalized_answers[index]:
                    continue
                words = answer_words[other]
                overlap = len(words & query_words)
                union_size = len(words) + len(query_words) - overlap
                ranked.append((overlap / max(1, union_size), other))
            ranked.sort(reverse=True)
            options, unique = [(pair["answer"], pair["example_sha256"])], {normalized_answers[index]}
            for _, other in ranked:
                answer = pairs[other]["answer"]
                if normalized_answers[other] not in unique:
                    unique.add(normalized_answers[other])
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


def immutable_revision(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError("Dataset revision must be an immutable 40-hex commit; resolve mutable names before preparation")
    return value.lower()


def decision_prompt_hash(record):
    """Input identity excludes supervision labels, answer candidates, and provenance."""
    state = record["state"]
    if isinstance(state, dict) and isinstance(state.get("conversation"), list):
        return digest([{"role": message["role"], "content": normalize(message["content"])} for message in state["conversation"]])
    instructions = {key: {"type": question["type"], "instructions": question.get("instructions", key)}
                    for key, question in record["questions"].items()}
    return digest({"state": state, "instructions": instructions})


def validate_decision_record(record):
    questions = record.get("questions")
    labels = record.get("labels")
    if not isinstance(questions, dict) or not questions or not isinstance(labels, dict) or set(labels) != set(questions):
        raise ValueError("Decision requires matching nonempty questions and labels")
    for question_id, question in questions.items():
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("Training decisions require at least two choice candidates")
        if any(not isinstance(key, str) for key in criteria) or labels[question_id] not in criteria:
            raise ValueError("Decision label must name an allowed candidate")
    computed = decision_prompt_hash(record)
    if record.get("provenance", {}).get("prompt_hash") != computed:
        raise ValueError("Decision prompt_hash does not match state and instructions")
    return computed


def decision_statistics(rows):
    questions = [question for row in rows for question in row["questions"].values()]
    candidate_counts = Counter(len(question["criteria"]) for question in questions)
    return {"questions": len(questions), "candidate_counts": {str(count): total for count, total in sorted(candidate_counts.items())},
            "chance_accuracy": sum(1 / len(question["criteria"]) for question in questions) / len(questions) if questions else None,
            "label_counts": dict(Counter(label for row in rows for label in row["labels"].values()))}


def load_local_stage(source_dir: Path):
    """Validate staged main payload hashes and original row-count declarations."""
    import pyarrow.parquet as parquet
    source_dir = Path(source_dir).resolve()
    package_path = source_dir / "package.json"
    package_bytes = package_path.read_bytes()
    package = json.loads(package_bytes)
    if package.get("repo_id") != DATASET_ID:
        raise ValueError("Local package targets a different dataset")
    base_revision = immutable_revision(package.get("base_revision"))
    files = {}
    for item in package.get("files", []):
        name = item.get("path")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or name in files:
            raise ValueError("Unsafe or duplicate staged package path")
        files[name] = item

    def verified_file(name):
        item = files.get(name)
        if item is None or not isinstance(item.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
            raise ValueError(f"Missing or invalid package hash for {name}")
        path = (source_dir / name).resolve()
        if not path.is_relative_to(source_dir) or not path.is_file():
            raise ValueError(f"Staged file is absent or escapes its source directory: {name}")
        if type(item.get("bytes")) is not int or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Staged file byte count does not match package: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != item["sha256"]:
            raise ValueError(f"Staged file hash does not match package: {name}")
        return path, actual_hash

    manifest_path, manifest_hash = verified_file("metadata/realtime_research_expansion_manifest.json")
    stage_manifest = json.loads(manifest_path.read_bytes())
    if stage_manifest.get("repo_id") != DATASET_ID or stage_manifest.get("base_revision") != base_revision:
        raise ValueError("Staged manifest source identity differs from package")
    raw, hashes = {}, {}
    for split in ("train", "eval", "test"):
        name = f"data/{split}-00000-of-00001.parquet"
        path, hashes[split] = verified_file(name)
        output = stage_manifest.get("outputs", {}).get(split, {})
        if output.get("file") != name or output.get("sha256") != hashes[split] or type(output.get("rows")) is not int:
            raise ValueError(f"Staged split declaration does not match payload: {split}")
        raw[split] = parquet.read_table(path).to_pylist()
        if len(raw[split]) != output["rows"]:
            raise ValueError(f"Staged split row count does not match payload: {split}")
    package_hash = hashlib.sha256(package_bytes).hexdigest()
    source = {"kind": "local_staged_unpublished", "package_sha256": package_hash,
              "expansion_manifest_sha256": manifest_hash, "base_revision": base_revision,
              "publication_verified": False, "input_rows": {split: len(rows) for split, rows in raw.items()}}
    return raw, hashes, "unpublished:" + package_hash, source


def prepend_live_decisions(decisions, live_decisions, audit):
    """Keep every eligible row, prioritizing test/eval and pilot-visible live rows."""
    if isinstance(live_decisions, list):
        cohorts = {"train": live_decisions, "eval": [], "test": []}
    elif isinstance(live_decisions, dict) and not set(live_decisions) - {"train", "eval", "test"}:
        cohorts = {split: live_decisions.get(split, []) for split in ("train", "eval", "test")}
    else:
        raise ValueError("Live decisions must be a list or a train/eval/test mapping")
    seen, counts = set(), {}
    for split in ("test", "eval", "train"):
        live = []
        for original in cohorts[split]:
            reason = model_input_exclusion_reason({"state": original.get("state"), "questions": original.get("questions")})
            if reason:
                _record_security_exclusion(audit, split, reason, live=True)
                continue
            row = copy.deepcopy(original)
            if row.get("provenance", {}).get("source_type") != "live_tape_observation":
                raise ValueError("Injected live decision lacks observation provenance")
            computed = decision_prompt_hash(row)
            previous = row["provenance"].get("prompt_hash")
            if previous is not None and previous != computed:
                raise ValueError("Injected live decision prompt_hash differs from its evidence")
            row["provenance"]["prompt_hash"] = computed
            validate_decision_record(row)
            live.append(row)
        kept, count = [], 0
        for index, row in enumerate(live + decisions[split]):
            prompt_hash = validate_decision_record(row)
            if prompt_hash in seen:
                key = f"{split}_live_duplicate_prompt_removed" if index < len(live) else f"{split}_research_duplicate_prompt_removed"
                audit[key] = audit.get(key, 0) + 1
                continue
            seen.add(prompt_hash)
            kept.append(row)
            count += int(index < len(live))
        decisions[split] = kept
        counts[split] = count
    audit["live_observation_training_records"] = counts["train"]
    return counts


def prepare_dataset(output_dir: Path, seed=42, dataset_revision=DATASET_REVISION, live_decisions=None, source_dir: Path | None = None):
    import pyarrow.parquet as parquet
    if source_dir is not None:
        raw, hashes, dataset_revision, source = load_local_stage(source_dir)
    else:
        dataset_revision = immutable_revision(dataset_revision)
        from huggingface_hub import hf_hub_download
        raw, hashes = {}, {}
        for split in ("train", "eval", "test"):
            path = Path(hf_hub_download(DATASET_ID, f"data/{split}-00000-of-00001.parquet",
                                      repo_type="dataset", revision=dataset_revision))
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw[split] = parquet.read_table(path).to_pylist()
        source = {"kind": "remote_immutable_commit", "revision": dataset_revision,
                  "input_rows": {split: len(rows) for split, rows in raw.items()}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions, audit = build_decisions(raw, seed)
    live_counts = None
    if live_decisions:
        live_counts = prepend_live_decisions(decisions, live_decisions, audit)
    outputs = {}
    for split, rows in decisions.items():
        if not rows:
            raise ValueError(f"No usable {split} decisions")
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical(row) + "\n")
        outputs[split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **decision_statistics(rows)}
    manifest = {
        "dataset": DATASET_ID, "dataset_revision": dataset_revision,
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "decision_index_reference": {"space": INDEX_ID, "revision": INDEX_REVISION,
                                     "used_as_training_data": False},
        "task": "reference-answer selection and deterministic observed-field choices" if live_counts else "four-option reference-answer selection", "seed": seed,
        "input_source": source,
        "input_parquet_sha256": hashes, "audit": audit, "outputs": outputs,
        "security_filter": {
            "version": 1, "applied": True,
            "scope": "all research conversations before answer-candidate selection; live state/questions before inclusion",
            "exclusion_counts": {key: count for key, count in audit.items() if "_excluded_" in key},
            "published_source_modified": False,
        },
        "limitations": ["Distractors are answers to other questions, not verified incorrect answers to this question.",
                        "Reference answers are inherited dataset labels, not independently audited trading outcomes.",
                        "Original splits share source documents; this is not an unseen-document evaluation.",
                        "No image training examples, policy labels, or official Decision Index score are produced."],
        "citations": ["https://arxiv.org/abs/2605.12151", "https://arxiv.org/abs/2606.08232"],
    }
    if live_counts:
        manifest["live_observations"] = {"source": "https://clawd-ws.fly.dev/", "records": sum(live_counts.values()),
                                         "splits": live_counts, "pilot_order": "before research rows",
                                         "label_basis": "observed status and event fields; no trading outcome labels"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local/clef-research-data"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--source-dir", type=Path, help="Validated unpublished staged package; avoids remote dataset access")
    parser.add_argument("--live-decisions", type=Path, help="Timestamped native observed-field decision JSONL to prepend to train")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.output, dataset_revision=args.dataset_revision, source_dir=args.source_dir,
                                    live_decisions=read_jsonl(args.live_decisions) if args.live_decisions else None), indent=2))

---
pretty_name: Solana Clawd Realtime Research Instruct
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
- question-answering
size_categories:
- 10K<n<100K
tags:
- solana
- clawd
- crypto
- research
- instruction-tuning
- rag
- tool-use
- structured-output
- realtime-ingestion
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: eval
    path: data/eval-*.parquet
  - split: test
    path: data/test-*.parquet
dataset_info:
  features:
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: source
    dtype: string
  - name: source_type
    dtype: string
  - name: source_sha256
    dtype: string
  - name: record_id
    dtype: string
  - name: example_sha256
    dtype: string
  - name: tags
    list: string
  - name: metadata
    struct:
    - name: title
      dtype: string
    - name: page
      dtype: int64
    - name: chunk
      dtype: int64
    - name: cell
      dtype: int64
    - name: cell_type
      dtype: string
  splits:
  - name: train
    num_examples: 78171
  - name: eval
    num_examples: 2595
  - name: test
    num_examples: 2896
  download_size: 61865747
---

# Solana Clawd Realtime Research Instruct

**83,662 English instruction conversations** for Solana mechanics, agent tools,
research retrieval, protocol reasoning, and risk-aware analysis. This is a
published data release from **Solana Clawd — The Sovereign Agent Stack on Solana**.
The initiative builds ecosystem-native models that understand accounts, PDAs,
versioned transactions, address lookup tables, Pump.fun graduation, and RPC
failure modes. Its intended architecture separates the model's reasoning from
wallet signing: the model never receives the signing keypair.

The main data consists of genuine retained and imported conversations. Retrieval
text, preference pairs, evaluation cases, transaction pretraining text, and
historical live observations are separate auxiliary artifacts. Their counts are
not added to the main instruction totals.

| Release fact | Value |
| --- | --- |
| Maintainer | [Solana Clawd](https://huggingface.co/solanaclawd) contributors |
| Language | English (`en`) |
| Main format | Parquet; list of chat messages plus provenance |
| Data snapshot | `58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2` |
| Preserved original snapshot | `10b57b01fb4e078a9f670bb73a64d9a063d68f6b` |
| Main examples | 83,662 |
| Main compressed Parquet size | 61,865,747 bytes |
| Model weights in this dataset repository | None |
| Dataset card update | 2026-10-01 |

## Intended uses and task boundaries

Use the main conversations for supervised instruction tuning, source-conditioned
question answering, and evaluating reference-answer selection. The auxiliary
corpora support retrieval and continued pretraining; preference pairs support
preference-learning experiments. Clef-specific records support its native typed
decision head rather than free-form chat generation.

The dataset is not the raw RED-2400 outcome benchmark and does not supply a new
future-return or profitable-trading label for every example. Research-paper QA,
source recall, and observed-field readback should be assessed as those tasks.
A dataset publication or successful tokenization check does not establish a
trained model, an inference deployment, or trading performance.

Related project resources: [monorepo](https://github.com/Solizardking/solana-clawd),
[Model Kit](https://huggingface.co/spaces/solanaclawd/clawd-model-kit),
[Homebase](https://huggingface.co/spaces/solanaclawd/homebase), and
[Clawd Zoo](https://huggingface.co/spaces/solanaclawd/clawd-zoo).
The Constitution and three on-chain laws are project constraints; their presence
in documentation does not prove that an arbitrary trained derivative obeys them.

## Configurations and splits

The `default` Hugging Face configuration selects only `data/*.parquet` and
preserves the split names **`train`, `eval`, and `test`**. In particular, use
`dataset["eval"]`; this release does not rename that split to `validation`.

| Split | Preserved original rows | Added rows | Published rows | Parquet bytes |
| --- | ---: | ---: | ---: | ---: |
| train | 26,152 | 52,019 | 78,171 | 57,087,352 |
| eval | 1,452 | 1,143 | 2,595 | 2,296,883 |
| test | 1,454 | 1,442 | 2,896 | 2,481,512 |
| **Total** | **29,058** | **54,604** | **83,662** | **61,865,747** |

All 29,058 original rows retain their original contents, order, and split. The
expansion adds 54,604 conversations; its lineage file joins each addition to its
source. Auxiliary files do not load through the default chat configuration.

## Main schema

| Field | Actual type | Meaning |
| --- | --- | --- |
| `messages` | list of structs: `role:string`, `content:string` | Model-facing conversation |
| `source` | string | Relative source identifier or linked dataset/file identifier |
| `source_type` | string | `parquet`, `pdf`, `notebook`, `text`, or `messages` |
| `source_sha256` | string | Recorded source hash where available; preserve its original meaning |
| `record_id` | string | Source/example identifier for tracing and exclusions |
| `example_sha256` | string | Recorded example-content identifier |
| `tags` | list of strings | Topic or processing tags |
| `metadata` | struct | Nullable `title:string`, `page:int64`, `chunk:int64`, `cell:int64`, `cell_type:string` |

Every current main conversation has one user message and one assistant message,
optionally preceded by a system message. `metadata` is a structured column,
not a JSON string. Source fields and metadata support auditing; they are not
implicitly instruction text and should not be concatenated into model input.

The following is an **illustration of the schema**, not a quoted training row:

```json
{
  "messages": [
    {"role": "system", "content": "Explain Solana mechanics using the supplied context."},
    {"role": "user", "content": "What is a program-derived address?"},
    {"role": "assistant", "content": "A PDA is an address derived from seeds and a program ID, without an associated private key."}
  ],
  "source": "example/reference.md",
  "source_type": "text",
  "source_sha256": "illustrative-source-hash",
  "record_id": "illustrative-example",
  "example_sha256": "illustrative-example-hash",
  "tags": ["solana", "pda"],
  "metadata": {"title": "Example", "page": null, "chunk": 0, "cell": null, "cell_type": null}
}
```

Measured `source_type` totals: `parquet` 28,250; `pdf` 683; `notebook` 122;
`text` 3; `messages` 54,604. The historical manifest's aggregate label
`expansion_conversations` is bookkeeping, not an additional column value.

## Loading and training-input preparation

Install `datasets` and `huggingface_hub` in a Python environment. Pin the data
revision for reproducibility. Loading the dataset does not download a model.

```python
from datasets import load_dataset

REPO = "solanaclawd/solana-clawd-realtime-research-instruct"
DATA_REVISION = "58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2"

dataset = load_dataset(REPO, revision=DATA_REVISION)
assert {name: len(split) for name, split in dataset.items()} == {
    "train": 78171, "eval": 2595, "test": 2896
}
# Inspect or train from messages after the screening step below.
```

### Required signing-material screen for model input

An audit of all 83,662 published conversations identified **14 retained train
rows** with literal signing material: 11 signing-byte-array matches and three
encoded-signing-literal matches. No such matches were found in `eval` or `test`.
The published Parquets preserve source data; they are **not asserted to be free
of signing material**. These matches do not establish that keys are active,
funded, or used in production.

Use [the standalone model-input filter](preprocessing/model_input_filter.py)
before selecting reference answers, distractors, or supervised model input.
[The audit](metadata/training_input_security_audit.json) records source hashes,
counts, safe exclusion categories, and record IDs without copying matched values.
The filter is the standalone version of the current Clef preparation screen.
Its pattern coverage is bounded; review new sources and runtime inputs separately.

```python
import hashlib
import importlib.util
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# Resolve one immutable documentation/filter commit. The underlying Parquets
# remain the DATA_REVISION loaded above; the audit asserts that exact revision.
DOC_REVISION = HfApi().dataset_info(REPO).sha
module_path = hf_hub_download(
    REPO, "preprocessing/model_input_filter.py", repo_type="dataset",
    revision=DOC_REVISION,
)
audit_path = hf_hub_download(
    REPO, "metadata/training_input_security_audit.json", repo_type="dataset",
    revision=DOC_REVISION,
)
with open(audit_path, encoding="utf-8") as handle:
    audit = json.load(handle)
assert audit["revision"] == DATA_REVISION
assert hashlib.sha256(Path(module_path).read_bytes()).hexdigest() == audit["helper_sha256"]
spec = importlib.util.spec_from_file_location("clawd_input_filter", module_path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

training_inputs = dataset.filter(
    lambda row: screen.model_input_exclusion_reason(row["messages"]) is None
)
assert {name: len(split) for name, split in training_inputs.items()} == {
    "train": 78157, "eval": 2595, "test": 2896
}
# This is screening only. Native decision preparation may additionally reject
# invalid prompts, duplicate candidate pools, or complete inputs over its cap.
```

For a low-memory stream, apply the same screen before forwarding any messages:

```python
stream = load_dataset(REPO, revision=DATA_REVISION, split="train", streaming=True)
for row in stream:
    if screen.model_input_exclusion_reason(row["messages"]) is None:
        messages = row["messages"]
        # Pass messages to your reviewed preparation/training pipeline.
        break
```

The audited snapshot becomes 83,648 conversations after this screen alone.
That count is not a claim about a later trainer's usable cohort or optimizer steps.

## Sources, collection, and curation

The original ingestion assembled submitted PDF text, notebook cells, structured
Parquet question/answer/chunk records, and a local reference skill. It produced
29,058 retained examples from 28 submitted source entries. Duplicate file entries
were skipped twice; 296 secret-like candidate examples were excluded. Source QA
and templates are derived from the submitted material; these are not all
independently reviewed human-expert annotations.

The expansion reads actual local NeMo conversations and restores linked dataset
payloads where local processed files are missing. It does not count a manifest,
card, binary search index, or missing shard as a training example.

| Expansion input | Accepted conversations | Recorded source revision |
| --- | ---: | --- |
| Local `nemo_clawd/processed` | 58 of 80 | Local source hashes in manifests/lineage |
| [solana-clawd-instruct](https://huggingface.co/datasets/solanaclawd/solana-clawd-instruct) | 20,849 | `a7ec746997bdebca82e1ac5c7c2ccb89bbda6c92` |
| [solana-clawd-core-ai-instruct](https://huggingface.co/datasets/solanaclawd/solana-clawd-core-ai-instruct) | 11,265 | `57bad74c6ad1e26fff6ad33b30f8e211fb9a613b` |
| [solana-clawd-repo-corpus](https://huggingface.co/datasets/solanaclawd/solana-clawd-repo-corpus) | 22,251 | `5908891baa92e6f7a4722e901c4324056df2738a` |
| [solana-clawd-nvidia-trading-factory-instruct](https://huggingface.co/datasets/solanaclawd/solana-clawd-nvidia-trading-factory-instruct) | 181 | `6e0b53994b485f1680194ca6a74fa96760a19257` |
| **Total additions** | **54,604** | |

Expansion screening excluded 981 secret-like candidates, 23,552 exact duplicate
conversations, and 14,286 cross-split prompt/answer conflicts. These stages have
distinct scopes: they do not remove every inherited record or every repeated
question or answer. The combined historical exclusion total 1,277 includes the
296 original-ingestion exclusions; it is not an additional number of new rows.

The [expansion manifest](metadata/realtime_research_expansion_manifest.json)
records each source revision, file hash, count, exclusion, and auxiliary artifact.
[Per-example lineage](metadata/expansion_lineage.jsonl) covers every addition.
The [original/combined manifest](metadata/realtime_research_dataset_manifest.json)
and [historical source documentation](metadata/local_sources/) preserve earlier
source descriptions and inventories.

### Original submitted source inventory

### PDF Research Sources

| File | What it contains | Pages | Examples | Identifier |
| --- | --- | ---: | ---: | --- |
| 2410.21169v5.pdf | Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction | 46 | 57 | https://arxiv.org/abs/2410.21169v5 |
| 2412.04913v3.pdf | Bridging Culture and Finance: A Multimodal Analysis of Memecoins in the Web3 Ecosystem | 4 | 8 | - |
| 2412.07591v2.pdf | CoinCLIP: A Multimodal Framework for Assessing Viability in Web3 Memecoins | 4 | 8 | - |
| 2512.01112v3.pdf | Autodeleveraging: Impossibilities and Optimization | 103 | 104 | https://arxiv.org/abs/2512.01112v3 |
| 2512.06505v4.pdf | Amortizing Perpetual Options | 12 | 13 | https://arxiv.org/abs/2512.06505v4 |
| 2512.19113v2.pdf | A Unified Framework and Comparative Study of Decentralized Finance Derivatives Protocols | 36 | 37 | https://arxiv.org/abs/2512.19113v2 |
| 2512.22476v1.pdf | AutoQuant: An Auditable Expert-System Framework for Execution-Constrained Auto-Tuning in Cryptocurrency Perpetual Futures | 67 | 68 | https://arxiv.org/abs/2512.22476v1 |
| 2601.10812v1.pdf | Optimal Liquidation of Perpetual Contracts | 36 | 37 | https://arxiv.org/abs/2601.10812v1 |
| 2601.17008v1.pdf | Bayesian Robust Financial Trading with Adversarial Synthetic Market Data | 12 | 19 | https://arxiv.org/abs/2601.17008v1 |
| 2602.00776v1.pdf | Explainable Patterns in Cryptocurrency Microstructure | 28 | 29 | https://arxiv.org/abs/2602.00776v1 |
| 2602.14860v1.pdf | Predicting the success of new crypto-tokens: the Pump.fun case | 29 | 30 | https://arxiv.org/abs/2602.14860v1 |
| 2603.10092v1.pdf | Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading with OpenClaw-Style Local Executors | 26 | 27 | https://arxiv.org/abs/2603.10092v1 |
| 2604.01431v1.pdf | Do Prediction Markets Forecast Cryptocurrency Volatility? Evidence from Kalshi Macro Contracts | 14 | 19 | https://arxiv.org/abs/2604.01431v1 |
| 2605.05089v1.pdf | Dynamic Collateral Control for Permissionless Spot Perpetual Basis Trading | 23 | 32 | https://arxiv.org/abs/2605.05089v1 |
| 2605.05878v1.pdf | Agentic, Context-Aware Risk Intelligence in the Internet of Value | 15 | 16 | https://arxiv.org/abs/2605.05878v1 |
| 2605.10400v1.pdf | Resolution-Aware Perpetual Futures on Binary Prediction Markets: An Empirical Risk-Design Framework Using Polymarket Data | 86 | 87 | https://arxiv.org/abs/2605.10400v1 |
| 2605.10428v1.pdf | A Taxonomy of Event-Linked Perpetual Futures: Variant Designs Beyond the Single-Market Binary Case | 47 | 48 | https://arxiv.org/abs/2605.10428v1 |
| 2605.12151v2.pdf | RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels | 8 | 9 | - |
| 2605.29174v1 (1).pdf | Paper Agents, Paper Gains: An Empirical Analysis of DeFi Investment Agents | 23 | 24 | https://arxiv.org/abs/2605.29174v1 |
| 2605.29174v1.pdf | duplicate file skipped | - | 0 | - |
| 2606.08232v1 (1).pdf | Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework | 11 | 11 | - |
| 2606.08232v1.pdf | duplicate file skipped | - | 0 | - |

### Notebook Sources

| File | Cells | Kernel | Examples |
| --- | ---: | --- | ---: |
| analysing-crypto-charts-like-pro.ipynb | 53 | Python 3 | 47 |
| analysis-of-smart-contracts-in-blockchain.ipynb | 27 | Python 3 | 22 |
| solana-prediction.ipynb | 67 | Python 3 | 53 |

### Parquet QA Sources

| File | Rows | Columns | Examples |
| --- | ---: | --- | ---: |
| test-00000-of-00001.parquet | 1426 | question, answer, chunk | 1407 |
| train-00000-of-00001.parquet | 27092 | question, answer, chunk | 26843 |

### Text and Skill Sources

| File | Type | Details | Examples |
| --- | --- | --- | ---: |
| SKILL.md | text | 10506 | 3 |

## Auxiliary artifacts and independent schemas

| Artifact | Rows | Schema / intended role |
| --- | ---: | --- |
| [NeMo corpus](auxiliary/nemo_corpus/chunks.jsonl) | 588 | Text chunks with source/citation/license metadata; retrieval |
| [NeMo preferences](auxiliary/nemo_preferences/risk_preferences.jsonl) | 7 | Prompt/chosen/rejected conversation pairs; preference learning |
| [NeMo evaluation](auxiliary/nemo_evaluation/source_grounded_eval.jsonl) | 49 | Question, expected answer, citation metadata; source-theme/guardrail recall |
| [NVIDIA RAG](auxiliary/nvidia_rag/chunks.jsonl) | 150 | `text` plus `meta.source` and `meta.len`; retrieval corpus |
| [Transaction CPT](auxiliary/tx_foundation_cpt/train.jsonl) | 19,542 | `{"text": string}`; auxiliary continued-pretraining text |
| [Clef live decisions](auxiliary/live_tape/decisions.jsonl) | 4 | Native `state`, `questions`, `labels`, `provenance`; observed-field selection |

NeMo corpus split labels are `train` 490 / `validation` 37 / `test` 61; those
labels belong to that auxiliary corpus, not the main chat splits. Its 588 chunk
texts and the 150 NVIDIA retrieval texts are exact-unique. Transaction CPT has
17,289 exact-unique texts among 19,542 rows; 2,253 rows are duplicate excess.
All current CPT rows are in the auxiliary `train.jsonl`. Historical CPT manifests
with different train/eval/test counts do not describe this file layout. CPT
source revision: `3a83b29fa8d7d894e93945c61b8d6ea6a6c03bd9`.

The 49 NeMo evaluation cases contain 24 source-theme cases, 24 guardrail-recall
cases, and one repository-safety case. Guardrail evaluation answers occur in
retained SFT responses, so these results cannot establish unseen-source quality.
The 62 historical documentation artifacts add no examples or fresh accuracy
measurements. NVIDIA metadata describes a local hash-based embedding fallback;
no binary search index or verified learned-embedding service is included.

## Clef tokenizer and read-only live source

[The tokenizer reference](auxiliary/live_tape/tokenizer_reference.json) pins
`Cloudflare/clef` to `2f3de3dd85f379784083b0814d997ab627200f0c`. It uses the existing
native processor/tokenizer, with **zero added vocabulary tokens**. A tokenizer
encodes text; it does not itself fetch a website or grant network access.
The bundled adapter captures allowlisted fields from
[clawd-ws.fly.dev](https://clawd-ws.fly.dev/) and supplies them as observed input.

The saved [snapshot](auxiliary/live_tape/snapshot.json) was received at
`2026-10-01T16:42:02.091072+00:00` from the read-only health endpoint and websocket.
It was already stale during staging and is historical evidence. The four labels
read back reported health/status or declared token-launch fields. They do not
verify issuer identities, social accounts, token safety, or future outcomes.
Fresh receive time does not prove that an event timestamp or market value is
current; health and websocket counters may also differ.

For actual runtime questions, capture a new bounded observation and check its
freshness before inference. Missing observations yield no fabricated labels.
The native encoder checks complete inputs and rejects silent state truncation.
The auxiliary live modules remain the code snapshot shipped with the immutable
data release; use the separately published current preprocessing filter for
signing-material screening. No signing credentials, transaction executor, model
weights, or persistent online learning process are provided by the tokenizer
reference.

## Quality, biases, and evaluation limitations

- The builder's whitespace-normalized complete conversations are unique, and
  its full non-assistant message hashes are cross-split disjoint. This is not a
  guarantee that user-only questions or answers never repeat.
- There are 169 normalized assistant-answer values shared across splits: one
  inherited train/eval answer and 168 inherited-eval/added-test answer values.
  Case-folded, whitespace-normalized user-only text has nine cross-split shared
  keys: six train/eval and three train/test.
- There are 41 source identifiers in main rows; 23 span multiple splits and 13
  occur in all three. Existing source documents can therefore appear in both
  training and evaluation. These splits do not establish unseen-document
  generalization or a temporal trading holdout.
- Coverage reflects supplied Solana/crypto/repository material and English
  extraction. It is not a representative sample of all markets or languages.
- PDF parsing, notebook conversion, imported QA, and template-derived examples
  can retain extraction errors, incomplete context, stale protocol details,
  source opinions, or repeated boilerplate. This is not uniform expert labeling.
- Untrusted launch fields and research text are evidence, not execution
  instructions. Source-guided recall accuracy is distinct from live execution
  reliability, security compliance, or investment returns.
- The 14 retained signing-material matches require the preparation screen
  described above. The audit reports patterns and provenance, not wallet state.

## Licensing, attribution, and responsible distribution

The repository retains its existing **CC-BY-4.0** dataset metadata declaration.
That declaration does not replace the terms of third-party papers, repositories,
notebooks, or imported datasets. Preserve their attribution and license notices
when reusing or redistributing derived material.

Both cited arXiv records link CC-BY-4.0 licenses. Other source-specific notices
remain attached to the auxiliary records and historical source cards. In
particular, 413 NeMo PDF chunks carry a notice to verify reuse rights before
external redistribution, and 175 repository chunks retain repository-content
notices. The trading-factory card preserves upstream excerpt terms and the
Apache-2.0 notice for its local cuFOLIO code. No blanket source-rights review is
claimed by this release.

Only model-facing messages should enter a training pipeline after screening.
Keep signing keys, credentials, and wallet control outside the model. Validate
source rights and freshness for your actual use rather than treating a dataset
license field, historical report, or template answer as a production guarantee.

## Reproducibility and integrity

Download the immutable data revision shown above to reproduce the exact released
payload. The three main SHA256 values are:

| File | SHA256 |
| --- | --- |
| `data/train-00000-of-00001.parquet` | `4d3d6b4b3030746cd29116a3f6d1a2f4c39ddc90119a43b39569085635fbb93d` |
| `data/eval-00000-of-00001.parquet` | `707aa9528afe89637fe00a134c9c993554bc2a63ba9c946db6698ff24b8a282e` |
| `data/test-00000-of-00001.parquet` | `44129a4c88930a008ddba8104705b6d223d395cd070b45ad9189be326dcea930` |

The repository's workflow scripts are `scripts/realtime_dataset_ingest.py`
(original ingestion), `scripts/expand_realtime_research.py` (expansion), and
`scripts/publish_research_expansion.py` (validated atomic publication).
`--restore-linked` resolves current donor revisions when rebuilding; that
command alone is not an exact reproduction of this historical release. Use the
recorded immutable donor revisions and source hashes for an equivalent rebuild.
Historical cards and reports describe their original runs, not a newly measured
training result.

## Research citations

Please cite the original research when using its derived examples:

- Kamat, A. U. (2026). *RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels*. arXiv:2605.12151. [arXiv record](https://arxiv.org/abs/2605.12151).
- Kamat, A. U. (2026). *Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading*. arXiv:2606.08232. [arXiv record](https://arxiv.org/abs/2606.08232).

The consumed sources are [RED-2400 v2](https://arxiv.org/abs/2605.12151v2),
`2605.12151v2.pdf` (nine original examples), and
[Hour-Aware v1](https://arxiv.org/abs/2606.08232v1),
`2606.08232v1 (1).pdf` (11 original examples). The latter's v1 title includes
*A Multi-Layer Intelligence Framework*. Later arXiv revisions changed that title;
this release does not silently substitute later results for the consumed v1.
The paper-derived QA is not a redistribution of every underlying benchmark row.

```bibtex
@misc{kamat2026red2400,
  author = {Kamat, Arati U.},
  title = {RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels},
  year = {2026},
  eprint = {2605.12151},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2605.12151v2}
}

@misc{kamat2026houraware,
  author = {Kamat, Arati U.},
  title = {Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework},
  year = {2026},
  eprint = {2606.08232},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2606.08232v1}
}

@misc{solanaclawd2026realtimeinstruct,
  author = {{Solana Clawd contributors}},
  title = {Solana Clawd Realtime Research Instruct},
  year = {2026},
  version = {58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct}
}
```

For questions or corrections, open a
[dataset discussion](https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct/discussions)
with the relevant immutable revision and record ID. Avoid posting credential or
signing-key values in reports.

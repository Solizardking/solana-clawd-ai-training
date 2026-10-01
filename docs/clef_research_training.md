# Fine-tune Clef on expanded Solana research

This workflow fine-tunes **Cloudflare/clef** with LoRA on text-backbone linear
layers and its native joint decision head, then exports a standalone multimodal
backbone with the trained head. The vision encoder and other backbone weights
remain frozen; retaining the vision weights does not establish improved vision
accuracy. Clef is pinned at revision
`2f3de3dd85f379784083b0814d997ab627200f0c`.

**Status on October 1, 2026:** the expanded dataset is
[published and hash-verified](https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct/commit/58eea08df320b56c0cfcec84f9ae1be1eb8bb5c2).
All 81 uploaded files were verified. Hugging Face authentication works. Pilot
submission returned HTTP 402 for both `ordlibrary` and `solanaclawd` because
prepaid Jobs credits are insufficient. No Clef GPU job started. The user selected
local training instead. Ten of the original 27B checkpoint's twelve source
shards have been converted in memory on this Mac's M4 Max. Conversion is paused
because macOS swap grew to about 35 GiB and left about 18 GiB of free disk space,
below the checkpoint-save reserve. Training follows a successful saved-base
reload check after resource headroom is restored.
Local tokenizer audits and tiny-model tests do not establish that the 27B model
has been trained. The verified publication record is already at
`local/research-expansion/published.json`. Rebuild commands below are for a fresh
staging directory.

Clef's documented decision interface is `encode_record` / `collate_records` /
`systemone` from its `joint_schema_model.py`. The generic Transformers image-text
generation snippet does not invoke its native decision head. See the
[Clef model card](https://huggingface.co/Cloudflare/clef).

## Expanded dataset and supervision

The staged package in `local/research-expansion/` contains **83,662 main chat
rows**: 78,171 train, 2,595 eval, and 2,896 test. It preserves all 29,058 original
rows in their original splits and adds genuine conversations from the available
NeMo and linked source datasets. Exact additions that conflict with existing
held-out prompts or answers are excluded. Inspect
`metadata/realtime_research_expansion_manifest.json` and
`metadata/expansion_lineage.jsonl` inside the stage for provenance and exclusions.

Retrieval chunks, NVIDIA RAG material, preference pairs, dedicated evaluation
cases, transaction CPT text, and historical source documentation retain their
own schemas in `auxiliary/` and `metadata/local_sources/`. They are not counted
as main chat rows. Historical inventories and missing processed shards are not
fabricated into training examples, and source-specific licensing still applies.

`clef_research_data.py` turns chat answers into four-option research-answer
selection records: the original answer is the target, with three length-matched
answers from other questions in the same split as distractors. Option order is
deterministic and randomized per question. Labels and provenance are not passed
to the model. The distractors have not been independently verified as incorrect
for each question; these are inherited reference labels, not trading outcomes.

Prepared decisions retain train/eval/test membership. Normalized duplicate
prompts are removed across splits, with test then eval taking priority, and
research candidates come from the same split as the prompt. Source documents
still overlap across splits, so this does not measure unseen-document
generalization. Inputs exceeding the context limit are excluded and counted
rather than silently truncated. Preparation counts and tokenizer-fit counts
can therefore be smaller than the main chat-row counts.

Preparation excludes literal credential payloads and signing-key/seed examples
before building answer candidates. This also prevents excluded answers from
appearing as distractors. Counts and safe category names are recorded in the
prepared manifest; source data and provenance remain intact.

Both pilot and full GPU jobs capture a fresh, read-only observation from
`https://clawd-ws.fly.dev/health` and `wss://clawd-ws.fly.dev/ws`. Health-status
labels have six choices; complete creator-declared social-presence masks have
eight. Labels come only from observed, allowlisted fields. A declared social
field does not verify an account, a fresh receive time does not prove a market
value is current, and snapshots contain no future trading outcomes. Random
chance is calculated per question: 1/4 for research, 1/6 for health, and 1/8 for
social masks, with the aggregate reflecting the selected questions.

The [JEV Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index)
is an evaluation reference, pinned at `7cdcea3dd14615192ff2e1f6fd13936a547b55d8`.
Its benchmark data and scores are not training data. Domain accuracy, negative
log likelihood, and multiclass Brier score are reported separately; an official
Decision Index result requires its full frozen suite. No image-training
examples or autonomous-execution policy labels are added.

## Local preparation and native tokenizer checks

Use an isolated Python environment, separate from Model Kit's FastAPI
environment. These commands run from the repository root:

```bash
uv venv --python 3.12 .venv-connect
uv pip install --python .venv-connect/bin/python \
  'huggingface_hub==1.33.0' 'pyarrow==25.0.1' 'pytest==9.1.1' \
  'torch==2.14.1' 'transformers==5.18.0' 'peft==0.21.2' 'pillow==12.3.0' \
  'torchvision==0.29.1'

# Rebuild the expansion from available payloads and linked published sources.
.venv-connect/bin/python scripts/expand_realtime_research.py --restore-linked
.venv-connect/bin/python scripts/stage_clef_live_tape.py \
  --stage local/research-expansion

# Validate staged hashes, original-row preservation, and required citations.
.venv-connect/bin/python scripts/publish_research_expansion.py \
  --stage local/research-expansion

# Prepare decisions directly from the staged parquet files, before publication.
.venv-connect/bin/python scripts/clef_research_data.py \
  --source-dir local/research-expansion --output local/clef-expanded-decisions
.venv-connect/bin/python scripts/preflight_clef_research.py \
  --data-dir local/clef-expanded-decisions \
  --max-records 128 \
  --output local/clef-expanded-tokenizer-preflight.json

# Capture a safe live snapshot and audit it with Clef's actual native tokenizer.
.venv-connect/bin/python scripts/clef_live_tape.py --native-tokenizer \
  --output local/clef-live-tape-snapshot.json \
  --decisions-output local/clef-live-decisions.jsonl

# Optional: include those historical observed-field labels in a bounded audit.
.venv-connect/bin/python scripts/preflight_clef_research.py \
  --source-dir local/research-expansion \
  --data-dir local/clef-expanded-decisions-with-live \
  --live-decisions local/clef-live-decisions.jsonl --max-records 128 \
  --output local/clef-expanded-with-live-tokenizer-preflight.json

.venv-connect/bin/python -m pytest \
  tests/test_clef_research.py tests/test_clef_live_tape.py \
  tests/test_export_clef_release.py tests/test_research_expansion_artifacts.py -q
```

The preflight downloads the processor, tokenizer, and pinned native decision
code, not Clef's roughly 55 GB of model weights. It audits complete inputs
against the context limit. With `--max-records 128`, all prepared row counts,
file hashes, and duplicate prompts are validated, while native encoding covers
only the selected 128 records per split; omit the bound for full encoding.
Local-stage manifests explicitly report `local_staged_unpublished` and
`publication_verified: false`; local preparation does not establish a published
dataset revision. The live audit checks exact JSON round trips and
native decision encoding without adding special tokens. Saved snapshots are
historical observations; runtime inference captures new evidence. Tests use
tiny randomly initialized backbones and actual native head/serialization code;
they do not establish 27B model quality. The H200 entry point requires CUDA
before a model-weight download. The separate local entry point uses Apple MPS.

## Local Apple MPS conversion

This Mac has an M4 Max with 48 GB of unified memory. The original Clef checkpoint
contains roughly 55 GB of BF16 source weights. The guarded converter estimates
17.3 GiB for its packed backbone and preserves the original native head,
processor, vision weights, and readable BF16 output embeddings. Its initial disk
guard requires about 25.2 GiB free, including one source shard and a 3 GiB reserve.
This is a file-space estimate; macOS swap and other applications can consume
additional space while the model is resident. The current busy Mac needs more
headroom than its initial 31 GiB free. No user files or unrelated caches were
removed to recover space.
The tested environment uses `bitsandbytes==0.50.2` with actual MPS NF4 kernels;
CPU fallback is disabled during conversion and training.

```bash
uv pip install --python .venv-connect/bin/python 'bitsandbytes==0.50.2'

# Inspect the resource plan. Use a new or empty output directory.
.venv-connect/bin/python scripts/convert_clef_mps.py \
  --output local/clef-27b-mps-nf4

# Keep the Mac awake while converting the immutable original 27B checkpoint.
caffeinate -d -i env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-connect/bin/python -u scripts/convert_clef_mps.py \
  --output local/clef-27b-mps-nf4 --execute
```

The converter verifies the pinned source index and LFS SHA256 hashes, downloads
one source shard into its own temporary directory, and removes that shard after
loading its tensors. It does not download the full 55 GB into the global cache.
It saves a standalone quantized backbone, releases the in-memory model, and
reloads the saved files for exact finite-logit parity. `conversion_manifest.json`
distinguishes planned, loading, saving, reloading, failed, and complete states.
Conversion does not train the model. Progress is not a resumable checkpoint;
failed outputs are preserved, and a new attempt needs a fresh directory.

The current task-owned process is paused with `SIGSTOP` at PID `75712`; its
sleep guard is PID `75713`. Converted tensors remain in process memory. Keep
that process alive to retain the ten completed shards. Restore internal disk
headroom and reduce memory pressure before resuming it. Moving its active
download directory behind an external symlink would break its deletion guard;
an external-drive restart needs a separate output directory. The progress log is
`local/clef-local-conversion.log`, and process state is recorded in
`local/clef-local-conversion-process.json`.

## Local native training and inference

The local pipeline runs a 16-step pilot with 128 complete training records and
eight records in each held-out split, then one epoch over all context-eligible
prepared training rows. Both phases use a 2,048-token cap, rank-8 LoRA on the
last four text layers, and the native joint head. Frozen lower layers and
embeddings do not receive input-gradient hooks. Overlength examples are counted
and excluded without truncation. Native token IDs use compact int32 arrays to
limit host-memory pressure; the collator restores the required Long tensors.
The pilot trains the mandatory paper/live examples and its longest selected
input before the full phase can start. It measures actual 27B updates and saved
adapter reload behavior; the pilot alone does not cover a dataset epoch.

```bash
# CPU/data preparation works before the model conversion is complete.
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-connect/bin/python \
  scripts/train_clef_local.py --prepare-only \
  --base local/clef-27b-mps-nf4 --output local/clef-local-preflight-fresh

# Inspect the existing coordinator without signaling or starting a process.
.venv-connect/bin/python scripts/continue_clef_local_training.py --status

# Start only one coordinator. It waits for 40 GiB free before resuming the
# recorded converter, then verifies conversion, pilot, full epoch, standalone
# export, and a separate fresh live inference process in sequence.
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-connect/bin/python -u \
  scripts/continue_clef_local_training.py

# Retry only a failed live observation after full training has succeeded.
.venv-connect/bin/python scripts/continue_clef_local_training.py --retry-runtime
```

The coordinator identifies the original converter by PID, exact start time,
command, and output path; it never launches a second converter. A process lock
prevents duplicate coordinators. Its state file, `local/clef-local-pipeline.json`,
reports actual optimizer counts separately from queued or running processes.
Once the converted model is saved, the next phase uses a disk reserve derived
from the actual shards that a merged export must rewrite, retained checkpoints,
and swap headroom. Each model process has its own sleep guard. A stopped or
failed trainer retains its artifacts and is not automatically restarted from
scratch. Disk headroom alone does not prove the 48 GB Mac can train at this cap;
the real pilot must pass before the full phase begins.

Both modes verify the exact published dataset commit, all three source Parquet
hashes, the 83,662 raw row counts, and the signing-material exclusion marker
before large-model loading. Fresh live decisions and one fitting example from
each Kamat paper are guaranteed to be trained within the pilot step budget.
Metrics, source/selection exclusions, actual LoRA/head byte changes, and saved
adapter reload parity are recorded. A full mode with record/step limits remains
a bounded run; metadata distinguishes completed selected epochs from a pilot.
Starting another phase from `--init-adapter` resets the optimizer; it is not an
exact optimizer-state resume. The two most recent task-owned checkpoints are
retained by default to bound disk use.

Standalone export merges only adapted text modules. It measures quantization
rounding against native decisions, rewrites affected shards to fresh files, and
hardlinks unchanged immutable shards into the release. The release has its own
complete filenames and survives removal of the source directory. Only a fresh
saved-release reload can mark it verified. Merge drift must remain at most
0.005, and standalone reload drift at most 0.0001. The runtime checks release
hashes and training evidence before loading weights, captures allowlisted live
tape, rejects signing material, and returns native typed decisions with
freshness metadata. It has no transaction-execution flow.

The resulting artifact is a complete quantized derivative of the original
`Cloudflare/clef`, including its tokenizer, native code, trained decision head,
and merged backbone shards. Its task is typed decision selection and observed
field readback. It is not a free-form chat model. Training evaluations and the
sanitized live snapshot accompany the weights, with exact source revisions,
exclusion counts, and actual update/reload evidence.

## Publish the completed local model

Only a completed full run with fresh standalone live inference can pass the
local model publisher. Planning does not create a Hub repository. Publication
defaults to a private model and streams the existing complete weights; it does
not make a second full-weight staging copy.

```bash
.venv-connect/bin/python scripts/publish_clef_local_model.py \
  --release outputs/clef-local-full-merged \
  --inference-proof local/clef-local-full-live-inference.json \
  --stage local/clef-local-model-publication

# Publish only after the completed artifact passes the command above.
.venv-connect/bin/python scripts/publish_clef_local_model.py \
  --release outputs/clef-local-full-merged \
  --inference-proof local/clef-local-full-live-inference.json \
  --stage local/clef-local-model-publication --push
```

The publisher keeps both Kamat references and the original Apache-2.0 source
license, preserves the original local release manifest for inference-proof
verification, and verifies every uploaded file against its immutable Hub
commit. It rejects incomplete epochs, pilot-only artifacts, missing native
weights, stale-at-inference evidence, and unexpected existing repository files.

## Publish the verified expansion before submitting training

Authenticate through the cached Hugging Face CLI login or configure `HF_TOKEN`
securely in the execution environment:

```bash
.venv-connect/bin/hf auth login
.venv-connect/bin/python scripts/publish_research_expansion.py \
  --stage local/research-expansion --push
```

Publishing uses the staged base revision as a parent-commit guard, preserves
the original rows and splits, and retains both Kamat citations. If the live
dataset has changed since staging, rebuild against its current revision before
retrying. After upload, the publisher downloads each file at the new immutable
commit and verifies its staged hash. Only then does it write
`local/research-expansion/published.json`.

The launcher requires that publication record and pins its exact verified
commit. A local expansion directory or mutable Hub `main` branch does not
satisfy the publication requirement. Dataset-write access, Jobs access, and
write permission to the private model destination are required, along with
sufficient GPU Jobs credit. Token values are never included in bundles or plans;
the launcher passes authentication to the remote job as an `HF_TOKEN` secret.

Fund the personal account in [billing settings](https://huggingface.co/settings/billing)
or the organization in its billing settings. Add `--jobs-namespace solanaclawd`
to launch commands to run and bill the organization; without it, Jobs bill the
authenticated user. The publisher and model repository namespace do not choose
the GPU billing namespace.

## Hugging Face GPU pilot and full training

Inspect the concrete plan before submitting a paid job:

```bash
.venv-connect/bin/python scripts/launch_clef_research_job.py \
  --publication local/research-expansion/published.json --stage pilot \
  --max-cost-usd 10

# Paid pilot: up to 16 optimizer steps, 128 training records, one-hour timeout.
.venv-connect/bin/python scripts/launch_clef_research_job.py --submit \
  --publication local/research-expansion/published.json --stage pilot \
  --max-cost-usd 10
.venv-connect/bin/python scripts/check_clef_research_job.py --stage pilot

# Illustrative full-job bound; review pilot throughput and overhead first.
.venv-connect/bin/python scripts/launch_clef_research_job.py --submit \
  --publication local/research-expansion/published.json --stage full \
  --timeout-hours 8 --max-cost-usd 40
.venv-connect/bin/python scripts/check_clef_research_job.py --stage full
```

The launcher captures fresh read-only live tape in both stages and includes
live observation records in the bounded pilot selection. It uses one H200
(141 GB VRAM), prints current hardware pricing, and records the job ID, URL,
dataset/model revisions, code hash, dependencies, and timeout under ignored
`local/` files. Use `--output-repo your-namespace/your-model` to change the
private destination. The pilot uses `-pilot`; full training saves its adapter
and native head to the main destination and adds `--export-merged` to create
a separate private `-merged` repository.

The verified H200 SDK price is approximately $0.083333 per minute, or $5 per
hour. The launcher calculates the timeout's maximum compute charge using the
current pricing unit and refuses charges above `--max-cost-usd`; the one-hour
pilot defaults to a $10 cap. Full submission requires both explicit
`--timeout-hours` and `--max-cost-usd`. Its dry-run default of eight hours does
not establish that one epoch fits. Review measured pilot seconds per optimizer
step before choosing the full bound; the estimate excludes full-cohort encoding,
evaluation, reloads, roughly 55 GB of export/upload, and sequence-length
differences. The eight-hour/$40 command is illustrative. Choose a larger
reviewed time/cost bound when pilot measurements or overhead require it.

A saved job ID prevents duplicate submissions. Inspect the same authoritative
job after polling timeouts; a failed observation does not imply the GPU job
stopped. Full training requires a completed genuine pilot whose training code,
pinned revisions, nonzero LoRA/head gradients, and saved reload evidence match
the current plan.

Checkpoints include adapter weights, the trained joint head, optimizer state,
training metadata, selected/encoded cohort counts, and exclusions. Full training
runs one epoch over all usable research and captured live training records;
evaluation uses every usable research eval/test record. Intermediate optimizer
snapshots do not currently enable automatic resume.

Adapter verification reloads the saved LoRA and head on the pinned Clef base
and compares held-out probabilities. Standalone verification additionally
merges LoRA into the complete multimodal backbone, saves its processor/tokenizer
and native head, releases the in-memory model, and reloads the actual saved
standalone checkpoint. Measured BF16 merge rounding must stay within a 0.005
probability difference; post-save standalone reload probabilities must match
the merged in-memory model within 0.0001. The GPU reload reads the full saved
backbone shards and checks runtime inference with newly captured evidence.
A job's `COMPLETED` status alone is insufficient: the checker must find the
matching provenance, adapter evidence, merged release inventory, and successful
post-save standalone reload/evaluation evidence. It checks all published release
files at the recorded model commit using sizes and SHA256 hashes. For the roughly
55 GB of LFS weight files, it compares Hub LFS SHA256/size metadata rather than
downloading weights locally; smaller non-LFS files are downloaded and hashed.
Domain metrics do not establish profitable or safe trading.

The pilot samples across the expanded split and reserves live observations and
available examples from both cited papers. Actual trainable parameter hashes
must change for both LoRA and the native head; nonzero gradients alone do not
establish that optimizer steps changed the saved weights.

## Load the standalone model and refresh runtime evidence

The merged release includes complete backbone weights, retained vision weights,
the trained native head, its processor/tokenizer, a portable loader, and the
read-only live adapter. It does not require the original base weights or a
separate PEFT adapter at inference time. Download the verified private release
in a GPU environment with enough storage and memory:

```python
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

path = Path(snapshot_download("solanaclawd/clef-solana-research-lora-merged"))
sys.path.insert(0, str(path))
from export_clef_release import load_release_model
from clef_live_tape import enrich_record_with_live_snapshot

model, processor = load_release_model(path, device="cuda")
# The loader registers the copied native module that defines this ClefModel.
native = sys.modules[model.__class__.__module__]
request = {
    "model": "clawd-clef-research",
    "state": "Read the current tape observations before assessing data freshness.",
    "questions": {
        "review": {
            "type": "choice",
            "instructions": "What does the captured evidence support?",
            "criteria": {
                "inspect": "Inspect observed fields and timestamps before further analysis.",
                "guarantee": "Treat endpoint health as a guarantee of profitable trading.",
            },
        },
    },
}
live_request, snapshot = enrich_record_with_live_snapshot(request)
response = native.systemone(model, processor, live_request)
print(response["answers"])
```

Pin `snapshot_download(..., revision=...)` to the merged model commit recorded
by the verified job when reproducing a release. Runtime enrichment attaches
freshness and read-only observations using Clef's existing native tokenizer;
creator-provided text remains untrusted data. Signing keys never enter model
inputs, and neither training nor this runtime bridge executes trades.

## Research attribution

- Kamat, A. U. (2026). *RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels*. [arXiv:2605.12151](https://arxiv.org/abs/2605.12151).
- Kamat, A. U. (2026). *Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading*. [arXiv:2606.08232](https://arxiv.org/abs/2606.08232).

The dataset retains v2 of RED-2400 and v1 of the hour-aware paper. The latter's
source title includes the subtitle *A Multi-Layer Intelligence Framework*.
Attribution refers to those existing sources; adding citations does not replace
them with later arXiv revisions.

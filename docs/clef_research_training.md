# Fine-tune Clef on Solana research

This package fine-tunes **Cloudflare/clef**, including its native joint decision
head and LoRA adapters on text-backbone linear layers. The vision encoder and
other backbone weights stay frozen. It uses the official research dataset at
revision `600e8f08d7ccdaddae56ab829eb076f3b36ad59f` and Clef at
revision `2f3de3dd85f379784083b0814d997ab627200f0c`.

Clef's documented inference interface is `encode_record` / `collate_records` /
`systemone` from its `joint_schema_model.py`. The generic Transformers image-text
generation snippet does not invoke the trained decision head. See the
[Clef model card](https://huggingface.co/Cloudflare/clef).

The source dataset contains chat answers rather than finite-choice decision
labels. `clef_research_data.py` creates a four-option research-answer selection
task: the original reference answer is the label, and three length-matched
answers from other questions in the same split are distractors. Option order
is deterministic and randomized per question. Labels and provenance are not
passed to the model. Distractors have not been independently verified as
incorrect for each question; these are inherited reference labels, not trading
outcome or autonomous-execution supervision.

Original train/eval/test splits are retained. Normalized duplicate prompts are
removed across splits, with test then eval taking priority. All candidates
come from the same split as the prompt. Source documents still overlap across
the upstream splits, so this does not measure generalization to unseen papers.
Inputs exceeding the token limit are excluded and counted rather than silently
truncated. No image-training examples are added.

The [JEV Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index)
is an evaluation reference, pinned at `7cdcea3dd14615192ff2e1f6fd13936a547b55d8`.
Its benchmark data and model scores are not used as training data. Domain
accuracy, negative log likelihood, and multiclass Brier score are reported
separately; an official Decision Index result requires its full frozen suite.

## Local preparation and tests

Use an isolated environment, separate from Model Kit's FastAPI environment:

```bash
uv venv .venv-connect
uv pip install --python .venv-connect/bin/python \
  'huggingface_hub==1.33.0' 'pyarrow==25.0.1' 'pytest==9.1.1' \
  'torch==2.14.1' 'transformers==5.18.0' 'peft==0.21.2' 'pillow==12.3.0' \
  'torchvision==0.29.1'
.venv-connect/bin/python scripts/clef_research_data.py
.venv-connect/bin/python scripts/preflight_clef_research.py
.venv-connect/bin/python -m pytest tests/test_clef_research.py -q
.venv-connect/bin/python scripts/launch_clef_research_job.py
```

Preparation downloads roughly 92 MB of dataset parquet files, not Clef's
roughly 55 GB of weights. Tests use a tiny randomly initialized backbone and
the genuine upstream decision-head code; they do not prove full-model quality.
The training command requires CUDA before attempting any weight download.

## Hugging Face GPU Jobs

Configure `HF_TOKEN` using your execution environment's secret manager or a
secure local shell. The token needs Jobs access and write permission to the
private output repository. Do not paste it into chat or commit it. GPU Jobs
also require an account with sufficient credit.

```bash
# Paid pilot: 16 optimizer steps, 128 training records, at most one hour.
.venv-connect/bin/python scripts/launch_clef_research_job.py --submit --stage pilot
.venv-connect/bin/python scripts/check_clef_research_job.py --stage pilot

# After the real pilot completes and its saved adapter passes reload checks:
.venv-connect/bin/python scripts/launch_clef_research_job.py --submit --stage full \
  --timeout-hours 8
.venv-connect/bin/python scripts/check_clef_research_job.py --stage full
```

Use `--output-repo your-namespace/your-model` to change the destination. The
pilot uses a separate `-pilot` repository; both outputs are private. Hardware
is one H200 (141 GB VRAM). The launch script prints current pricing and records
the job ID, URL, code hash, dependencies, and timeout in ignored `local/` files.
Token values are passed as Jobs secrets and are not included in the bundle or
job plan. A saved job ID prevents accidental duplicate submissions; inspect
that job instead of starting another when polling times out.

Every 200 optimizer steps (8 in the pilot), the adapter, trained joint head,
optimizer state, training metadata, cohort audit, and exclusions are saved to
the private model repository. Training is one epoch over all usable training
records in the full stage. Full evaluation uses every usable eval/test record.
Final verification reloads the saved LoRA and head on the pinned Clef base,
then compares probabilities on up to eight held-out records. Final output
includes model card attribution and per-record predictions. The current
trainer does not resume automatically from intermediate optimizer snapshots.

Loading the artifact requires both adapter weights and `joint_head.safetensors`;
loading only the PEFT adapter would leave the original decision head active:

```python
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

path = Path(snapshot_download("solanaclawd/clef-solana-research-lora"))
sys.path.insert(0, str(path))
from clef_research_training import load_upstream, reload_adapter

upstream = load_upstream()
model, processor = reload_adapter(upstream, path, device="cuda")
response = upstream.systemone(model, processor, {
    "model": "clawd-clef-research",
    "state": "A v0 Solana transaction references addresses through an address lookup table.",
    "questions": {
        "lookup_table": {
            "type": "choice",
            "instructions": "What does the lookup table provide?",
            "criteria": {
                "addresses": "Account addresses referenced by transaction indices.",
                "signatures": "Private signing keys for all referenced accounts.",
            },
        },
    },
})
print(response["answers"])
```

Signing keys never enter model inputs, and training does not execute trades.

## Research attribution

- Kamat, A. U. (2026). *RED-2400: A Public Benchmark of Algorithmically-Rejected Trading Events with Outcome Labels*. [arXiv:2605.12151](https://arxiv.org/abs/2605.12151).
- Kamat, A. U. (2026). *Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading*. [arXiv:2606.08232](https://arxiv.org/abs/2606.08232).

The dataset includes v2 of RED-2400 and v1 of the hour-aware paper; their
existing data is retained rather than replaced with later revisions.

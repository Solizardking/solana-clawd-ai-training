---
language:
  - en
pretty_name: Clawd Chart Foundation Training
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
  - visual-question-answering
tags:
  - solana
  - clawd
  - charts
  - multimodal
  - supervised-fine-tuning
  - document-adaptation
---

# Clawd Chart Foundation Training

A provenance-tracked text-and-vision training package for ecosystem-native Solana
AI: supervised chat, chart question answering, and document adaptation. Published
by `ordlibrary` as part of the Solana Clawd ecosystem.

**Official catalog:** [Clawd AI](https://huggingface.co/spaces/solanaclawd/clawd-ai)
· [Solana Clawd](https://huggingface.co/solanaclawd)
· [Training repository](https://github.com/Solizardking/solana-clawd-ai-training)
· [Model Kit](https://huggingface.co/spaces/solanaclawd/clawd-model-kit)

## Contents and verified counts

The repository contains `chart-foundation-data.zip` and `train_chart_foundation.py`.
Counts below were checked against the published archive on **October 1, 2026**,
including every JSONL record and the image file inventory.

| Split | Supervised chat / chart rows | Document chunks |
| --- | ---: | ---: |
| Train | 34,831 | 2,338 |
| Validation | 3,257 | 149 |
| Test | 3,126 | 858 |
| Total | 41,214 | 3,345 |

The archive includes **1,712 chart images**. Document chunks are not necessarily
independent documents; chunks from the same document share a group and split.
The supplied preparation manifest records `training_completed: false`: dataset
publication is not evidence that a full model training run completed.

## Data structure

After extraction:

```text
train.jsonl
validation.jsonl
test.jsonl
documents-train.jsonl
documents-validation.jsonl
documents-test.jsonl
images/
manifest.json
```

Supervised rows have `id`, `messages`, `images`, `sources`, `split`, and `group`.
`messages` is a list of role/content objects; `images` is empty for text-only
examples or contains an image path relative to the extracted package.
Document rows have `text`, `source`, `split`, and `group`.

## Download and load

The training data is packaged inside the ZIP rather than as root-level JSONL.
Download and extract it before using the JSON loader:

```python
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download
from datasets import Features, List, Value, load_dataset

archive = hf_hub_download(
    "ordlibrary/clawd-chart-foundation-training",
    "chart-foundation-data.zip",
    repo_type="dataset",
)
root = Path("clawd-chart-foundation-data").resolve()
root.mkdir(exist_ok=True)
with ZipFile(archive) as package:
    # Verify paths before extracting an externally downloaded archive.
    for member in package.infolist():
        if not (root / member.filename).resolve().is_relative_to(root):
            raise ValueError("Unsafe archive path")
    package.extractall(root)

# Early text-only batches have empty images lists. Declare their type so the
# loader does not infer a null element type before reaching vision examples.
sft_features = Features({
    "id": Value("string"),
    "messages": List({"role": Value("string"), "content": Value("string")}),
    "images": List(Value("string")),
    "sources": List(Value("string")),
    "split": Value("string"),
    "group": Value("string"),
})
sft = load_dataset("json", features=sft_features, data_files={
    split: str(root / f"{split}.jsonl")
    for split in ("train", "validation", "test")
})
documents = load_dataset("json", data_files={
    split: str(root / f"documents-{split}.jsonl")
    for split in ("train", "validation", "test")
})
# Resolve each SFT image path against root before feeding a vision processor.
```

## Sources and preparation

The archive manifest inventories existing Clawd instruction datasets and their
named held-out splits; repository and research document corpora; the
`ordlibrary/charts` bucket's trajectories and chart metadata; captured Clawd
WebSocket frames; PDF reference material; and a **33-row historical SolArchive
sample**. It does not contain the full SolArchive history. Source paths in the
manifest record preparation provenance and need not exist on the consumer's machine.

The preparation code requires supervised conversations to contain a user and
end in an assistant response. It removes exact duplicate chat records, filters
recognized secret patterns, preserves explicitly named validation/test splits,
and assigns remaining groups deterministically. Duplicate/group collisions are
moved toward the more restrictive held-out split. Image families and chunks from
the same document remain grouped. Dataset cards, manifests, FAISS indexes, and
unusable preference records are not automatically treated as supervised labels.

Preparation rejects recorded in the archive manifest: 1,866 duplicate chats,
21 chat secret-pattern matches, 28 secret-bearing documents, 7 non-SFT records,
and 2 invalid image metadata rows. These checks are not a comprehensive privacy
audit or proof of semantic independence.

The manifest names
`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`
as the intended base model. Model compatibility, licensing, and vision processing
must be checked against the chosen base model and training script.

## Intended uses and limitations

- Research on Solana mechanics, chart understanding, and source-grounded agents.
- Text/vision supervised fine-tuning and separate document adaptation.
- Evaluation with the preserved validation and test partitions.

This is a dated training snapshot, not a realtime price feed or a trading oracle.
Examples may contain synthetic charts, historical facts, or outdated provider
interfaces. Exact duplicate filtering does not establish semantic separation;
inspect source families before interpreting held-out results. The dataset does
not establish profitability, calibration, production safety, or completed training.
Preserve Brain/Hands separation: model outputs do not grant signing authority.

## Licensing and attribution

The bundle combines multiple sources and does not declare a blanket license here.
Review individual source rights and the intended base model's terms before reuse.
The manifest attributes the SolArchive sample as **CC-BY-4.0 — Data from
SolArchive.org**. That attribution does not extend to all other contents.

## Citation

```bibtex
@misc{clawd_chart_foundation_training,
  author = {Solana Clawd and ordlibrary},
  title = {Clawd Chart Foundation Training},
  year = {2026},
  url = {https://huggingface.co/datasets/ordlibrary/clawd-chart-foundation-training},
  note = {Dataset card verified against the published archive on 2026-10-01}
}
```

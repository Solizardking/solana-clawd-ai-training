---
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - solana
  - clawd
  - nemo-clawd
  - source-grounded
  - research
  - confidential-execution
  - zero-knowledge
  - zk-proofs
  - tee
  - bitvm
  - ai-training
pretty_name: Nemo Clawd Source-Grounded Instruct
---

# Nemo Clawd Source-Grounded Instruct

Source-grounded instruction-tuning dataset derived from a curated PDF corpus (24 unique academic papers on confidential execution, zero-knowledge proofs, TEEs, BitVM, and cloud infrastructure) plus repository documentation chunks.

## Contents

| Component | Count |
|-----------|-------|
| SFT rows (messages format) | 80 |
| SFT rows with metadata | 80 |
| Preference (DPO) rows | 7 |
| Eval rows | 49 |
| PDF chunks (retrieval corpus) | 413 |
| Repo documentation chunks | 175 |
| Combined chunk corpus | 588 |
| Unique PDF sources | 24 |
| Duplicate PDFs deduplicated | 2 |
| Source notes (curated cards) | 24 |

## Format

Each SFT row is a chat conversation in OpenAI/Hugging Face `messages` schema:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Preference rows use the standard `chosen`/`rejected` format for DPO/RLHF training.

## Directory Layout (within `data/nemo_clawd/`)

| Path | Contents |
|------|----------|
| `corpus/all_chunks.jsonl` | Combined chunk corpus with deterministic train/validation/test split labels |
| `corpus/pdf_chunks.jsonl` | Page-grounded PDF chunks for retrieval-augmented training |
| `corpus/repo_chunks.jsonl` | Repository documentation chunks for product/operator behavior |
| `sft/chat_finetune.jsonl` | Chat fine-tuning rows in `messages` format |
| `sft/chat_finetune_with_metadata.jsonl` | Same rows with IDs, tasks, and source IDs |
| `preference/risk_preferences.jsonl` | Chosen/rejected safety pairs for policy tuning |
| `eval/source_grounded_eval.jsonl` | Source-grounded eval prompts and expected answers |
| `manifests/source_manifest.json` | Source inventory, hashes, duplicate aliases, extraction stats |
| `source_notes/*.md` | One curated source card per unique PDF |
| `reports/quality_report.json` | Machine-readable build report |
| `reports/quality_report.md` | Human-readable build report |

## Supported SFT Topics

- Trusted Execution Environments (TEEs) and confidential computing
- Zero-knowledge proofs (ZK, zkVM, zk-SNARKs)
- BitVM and Bitcoin bridge verification
- Hardware security primitives (FPGA TEE, RISC-V, Intel SGX, ARM TrustZone)
- AI/ML secure enclaves and verifiable inference
- Cloud deployment guides (Google Cloud, confidential VMs, TDX/SNP)
- Protocol design and cryptographic primitives

## Reproduce

The dataset was built with:

```bash
python3 /Users/8bit/drive/nemo-clawd/training-data/scripts/build_training_data.py \
  --pdf-root /Users/8bit/drive/pdfs
```

Requires Poppler (`pdfinfo` and `pdftotext`), no Python packages outside the standard library.

To process for training:

```bash
cd ai-training
python3 scripts/prepare_dataset.py \
  --input data/nemo_clawd/sft/chat_finetune.jsonl \
  --output data/nemo_clawd/processed \
  --train-ratio 0.9 --eval-ratio 0.05 --seed 42
```

## Publish

```bash
cd ai-training
hf repos create solanaclawd/solana-clawd-nemo-clawd-instruct --type dataset --exist-ok
hf upload solanaclawd/solana-clawd-nemo-clawd-instruct data/nemo_clawd/processed . --repo-type dataset --commit-message "Add processed Nemo Clawd splits"
hf upload solanaclawd/solana-clawd-nemo-clawd-instruct data/nemo_clawd_dataset_card.md README.md --repo-type dataset --commit-message "Add dataset card"
hf upload solanaclawd/solana-clawd-nemo-clawd-instruct data/nemo_clawd/sft/chat_finetune.jsonl raw/chat_finetune.jsonl --repo-type dataset --commit-message "Add raw SFT JSONL"
hf upload solanaclawd/solana-clawd-nemo-clawd-instruct data/nemo_clawd/sft/chat_finetune_with_metadata.jsonl raw/chat_finetune_with_metadata.jsonl --repo-type dataset --commit-message "Add raw SFT with metadata"
hf upload solanaclawd/solana-clawd-nemo-clawd-instruct data/nemo_clawd/manifests/source_manifest.json metadata/source_manifest.json --repo-type dataset --commit-message "Add source manifest"
```

## Safety Notes

Treat each PDF source as research evidence, not as permission to execute trades or privileged actions. The builder runs in public-safe mode. Verify redistribution rights before publishing the derived corpus outside the workspace.
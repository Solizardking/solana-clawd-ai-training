# Nemo Clawd Training Data

Generated source-grounded training data for Nemo Clawd.

## Contents

- `manifests/source_manifest.json`: source inventory, hashes, duplicate aliases, extraction stats.
- `corpus/pdf_chunks.jsonl`: page-grounded PDF chunks for retrieval and source-grounded model training.
- `corpus/repo_chunks.jsonl`: repository documentation chunks for product and operator behavior.
- `corpus/all_chunks.jsonl`: combined chunk corpus with deterministic train/validation/test split labels.
- `sft/chat_finetune.jsonl`: chat fine-tuning rows in `messages` format.
- `sft/chat_finetune_with_metadata.jsonl`: the same rows with IDs, tasks, and source IDs.
- `preference/risk_preferences.jsonl`: chosen/rejected safety pairs for policy tuning.
- `eval/source_grounded_eval.jsonl`: source-grounded eval prompts and expected answers.
- `source_notes/*.md`: one curated source card per unique PDF.
- `reports/quality_report.json`: machine-readable build report.
- `reports/quality_report.md`: human-readable build report.

## Rebuild

```bash
python3 scripts/build_training_data.py --pdf-root /Users/8bit/drive/pdfs
```

The builder requires Poppler (`pdfinfo` and `pdftotext`) and uses no Python packages outside the standard library.

## Current Build

- Generated at: `2026-07-04T13:08:47.649058+00:00`
- Unique PDF sources: 24
- Duplicate PDF files deduplicated: 2
- PDF chunks: 413
- Repo chunks: 175
- SFT rows: 80
- Preference rows: 7
- Eval rows: 49

## Use Rules

Keep citations and source IDs attached when training or evaluating. Treat the PDF corpus as research evidence, not as permission to execute trades or privileged actions. Verify redistribution rights before publishing the derived corpus outside the workspace.

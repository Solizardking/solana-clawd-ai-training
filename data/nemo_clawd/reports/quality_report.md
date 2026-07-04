# Training Data Quality Report

Generated at `2026-07-04T13:08:47.649058+00:00`.

## Counts

- Input PDF paths: 26
- Unique PDF sources: 24
- Duplicate PDF files deduplicated: 2
- Missing PDF files: 0
- PDF chunks: 413
- Repo chunks: 175
- All chunks: 588
- SFT rows: 80
- Preference rows: 7
- Eval rows: 49

## Themes

- `agentic_trading_safety`: 4
- `confidential_execution`: 3
- `document_intelligence`: 1
- `market_risk_derivatives`: 8
- `memecoin_intelligence`: 4
- `operator_learning`: 1
- `prediction_markets`: 3

## Checks

- all_chunks_have_source_id: True
- all_chunks_have_split: True
- email_redaction_passed: True
- jsonl_counts: {'/Users/8bit/drive/nemo-clawd/training-data/corpus/pdf_chunks.jsonl': 413, '/Users/8bit/drive/nemo-clawd/training-data/corpus/repo_chunks.jsonl': 175, '/Users/8bit/drive/nemo-clawd/training-data/corpus/all_chunks.jsonl': 588, '/Users/8bit/drive/nemo-clawd/training-data/sft/chat_finetune.jsonl': 80, '/Users/8bit/drive/nemo-clawd/training-data/sft/chat_finetune_with_metadata.jsonl': 80, '/Users/8bit/drive/nemo-clawd/training-data/preference/risk_preferences.jsonl': 7, '/Users/8bit/drive/nemo-clawd/training-data/eval/source_grounded_eval.jsonl': 49}
- jsonl_valid: True
- no_empty_chunks: True

## Deduplicated PDFs

- `/Users/8bit/drive/pdfs/2605.29174v1 (1).pdf` aliases: /Users/8bit/drive/pdfs/2605.29174v1.pdf
- `/Users/8bit/drive/pdfs/2606.08232v1 (1).pdf` aliases: /Users/8bit/drive/pdfs/2606.08232v1.pdf

## Notes

- Extraction method: Poppler `pdftotext -layout` per page with page provenance retained.
- Emails and likely secret-bearing tokens are redacted in generated text.
- Duplicate files are detected by SHA-256 and retained as aliases in the manifest.
- The spaced input path for `2412.07591v2.pdf` was normalized to the actual file present on disk.

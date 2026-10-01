# Research dataset expansion

`scripts/expand_realtime_research.py` builds an additive update to
`solanaclawd/solana-clawd-realtime-research-instruct`. It downloads the current
revision and retains every existing row in its original split, then imports
genuine conversations from actual NeMo payloads. `--restore-linked` also
recovers missing payloads from the published repositories identified by the
provided source manifests, recording their current revisions and file hashes.

The supplied `data/processed` directory contains historical metadata without
its referenced Arrow shards. Their documented counts are not claimed as newly
read local examples. Published split parquets are used when restoration is
enabled. The local NeMo metadata JSONL and processed splits contain the same
80 conversations; the metadata version is matched to actual 72/4/4 membership
and imported once.

Each new conversation is checked for valid roles, credential-like content,
exact normalized duplicates, and prompt/answer conflicts with other splits.
Original source documents already span some splits, so these checks do not
establish unseen-document generalization. New source metadata joins through
`metadata/expansion_lineage.jsonl`; the original eight-column chat schema stays
compatible with existing consumers.

Different source formats retain distinct roles:

| Source | Published role |
| --- | --- |
| Genuine chat conversations | Main train/eval/test parquets |
| NeMo combined corpus | Auxiliary retrieval corpus; component duplicates omitted |
| NVIDIA RAG chunks | Auxiliary retrieval corpus with relative source provenance |
| Risk preferences | Auxiliary chosen/rejected pairs; rejected answers never become SFT targets |
| Dedicated NeMo evaluation | Auxiliary recall/policy benchmark, including its original answer and citation requirements |
| Transaction CPT sequences | Auxiliary text pretraining corpus; no fabricated chat labels |
| Cards, manifests, source notes and reports | Sanitized historical source documentation, contributing zero new examples |
| FAISS index and Arrow metadata without payload | Inventory/documentation only; no invented recovered rows |

The original source licensing notes and both Kamat citations are retained.
Supplied evaluation/report pass flags are historical artifacts, not freshly
measured model accuracy or executed trading evidence. Original source materials
may have their own terms in addition to the dataset card's license.

```bash
# Build once into a fresh staging directory.
.venv-connect/bin/python scripts/expand_realtime_research.py --restore-linked

# Check staged hashes, required files, and citations without publishing.
.venv-connect/bin/python scripts/publish_research_expansion.py \
  --stage local/research-expansion

# Login locally; keep tokens out of chat and source files.
.venv-connect/bin/hf auth login

# Publish all files in one guarded commit, then verify their committed hashes.
.venv-connect/bin/python scripts/publish_research_expansion.py \
  --stage local/research-expansion --push
```

The publication is guarded by the exact parent revision. If the live dataset
changes while work is staged, rebuild against that revision into a new directory;
the publisher does not overwrite concurrent changes. Publication does not
implicitly start paid GPU work. Follow the [Clef training guide](clef_research_training.md)
to run the genuine GPU pilot and then train and export the full model.

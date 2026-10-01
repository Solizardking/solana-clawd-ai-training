---
title: Clawd AI
emoji: 🦞
colorFrom: purple
colorTo: green
sdk: static
app_file: index.html
pinned: true
short_description: Official Clawd AI models, datasets and live download stats
---

# Clawd AI — The Sovereign Agent Stack on Solana

Official catalog of ecosystem-native Clawd AI models, datasets, research, and Spaces.
Clawd understands Solana mechanics: accounts, PDAs, versioned transactions, ALTs,
Pump.fun graduation, and RPC failure modes. Brain/Hands separation keeps signing
keys outside the model; artifacts support constitutionally bounded, reproducible agents.

This static Space hosts the catalog and live metadata dashboard. Model weights and
datasets remain in their linked Hugging Face repositories; demos open in their own
Spaces. Catalog availability does not imply a model inference endpoint is running.

## Live statistics

Public Hub API discovery covers all non-profile repositories in `solanaclawd` and
Clawd/Hauhau repositories in `ordlibrary`. Explicit featured links include every
artifact requested for this launch and the Clawd Agentic Layer whitepaper.
Browser-side discovery supports pagination and automatically refreshes every minute.
A manual refresh checks the Hub immediately. Public API requests need no secret. Downloads
are Hub-reported **last-30-days** counts, not lifetime or per-second counters.
Spaces do not have this download metric. Missing counts remain unavailable.
On provider failure, the catalog retains previous group results marked stale and
labels totals partial. It shows refresh time and errors rather than inventing activity.

- [Model download methodology](https://huggingface.co/docs/hub/models-download-stats)
- [Dataset download methodology](https://huggingface.co/docs/hub/datasets-download-stats)
- [Solana Clawd monorepo](https://github.com/Solizardking/solana-clawd)
- [Clawd Agentic Layer](https://huggingface.co/datasets/ordlibrary/clawd-agentic-layer-whitepaper)
- [Musebook](https://musebook.trade)

## Run locally (optional diagnostic server)

```sh
python app.py
# http://127.0.0.1:7860
python -m unittest test_catalog.py
```

No API key, GPU, model download, wallet connection, or signing is required.
Each linked artifact retains its own license and usage conditions.

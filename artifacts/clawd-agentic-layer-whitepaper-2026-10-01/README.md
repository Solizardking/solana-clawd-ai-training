---
language:
  - en
tags:
  - solana
  - agents
  - whitepaper
  - retrieval
configs:
  - config_name: corpus
    data_files:
      - split: train
        path: data/corpus.jsonl
---

# Clawd Agentic Layer — Whitepaper v0.3

A design-proposal whitepaper for the **Clawd Agentic Layer**: a verifiable AI agent economy on Solana.

## Abstract

Clawd is a sovereign Solana-native AI agent. This whitepaper specifies the Clawd Agentic Layer: onchain agent identity (Metaplex Agent Registry), Agent DNA lineage with law/constitution/proof hashes, a Six-Law Harness, a progressive trust ladder (Observer → Dry-Run → Delegated → Autonomous → Sovereign), a dual-wallet architecture (Muse OWS policy-gated signer for x402/service payments; Metaplex Asset Signer PDA for the identity-linked treasury), capability discovery (MCP, A2A, WebMCP, CAAP-style), SIWS/signed challenges with scoped grants and revocation, a security model (browser-side signing by default, least authority, budgets, expiry, freshness and evidence requirements), the Clawd operating loop (Observe → Verify → Authenticate/Attest → Research → Decide → Price → Pay → Simulate → Act → Prove → Remember → Adapt → Repeat), and the JEV decision-engine discipline — a proposed open standard for agentic trading with typed probabilistic judgments, dynamic action spaces, reachability-gated venue admission, fail-closed execution, regime conditioning, episodic memory, simulation-before-action, conformance invariants, and a machine-readable decision-record schema.

## Version

- **Version:** v0.3 (22 pages)
- **Date:** September 2026
- **Canonical PDF:** https://musebook.trade/clawd-agentic-layer-whitepaper.pdf
- **This repo:** Hub-hosted preprint / project artifact (not an arXiv publication, not a Hugging Face Paper Page)

## Status warning

This is a **design proposal**, not a finished protocol. Not every component described here is deployed, audited, or production-ready. Surfaces are labeled as deployed, experimental, or proposed in the document; treat experimental surfaces accordingly.

## Changelog

- **v0.3** (September 2026, 22 pages): Adds the JEV decision-engine discipline as a proposed open standard for agentic trading — one typed judgment per cycle over an explicitly constructed action set, dynamic action spaces with reachability-gated venue admission (Jupiter, DFlow, Backpack Exchange, and pump.fun token-vs-SOL; a venue action is offered only when its quote succeeds this cycle), out-of-set answers fail closed to `BLOCKED`, regime conditioning and episodic memory as context (never authority), simulation-before-action, a dry-run invariant, six conformance invariants, and a machine-readable decision-record schema. References two open-source implementations: [clawd-jev-trading-machine](https://github.com/Solizardking/clawd-jev-trading-machine) (current JEV decision backend — Jupiter, DFlow, Backpack Exchange, and pump.fun token-vs-SOL as quote-gated venues; main at commit `b40a6d29`, September 25, 2026) and [jev-trader-solana](https://github.com/Solizardking/jev-trader-solana) (earlier multi-venue reference implementation — Jupiter + DFlow SOL/USDC spot routing with Imperial/Phoenix-routed 1x perps).
- **v0.2** (September 2026, 20 pages): Six-law harness, trust ladder, dual-wallet architecture, agent lineage, discovery/auth, deployed surfaces, security model, Clawd operating loop.
- **v0.1** (September 2026, 14 pages): Initial proposal — agent identity, trust-as-credit, Arweave-scheduled execution, Permaweb names, proof-carrying receipts, phased rollout.

## Project links

- Project hub: https://musebook.trade
- Agent terminal: https://terminal.musebook.trade
- Canonical token: `$CLAWD` — `8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump` (Solana)
- Live relay: `wss://clawd-ws.fly.dev/ws`
- Repository: https://github.com/solizardking/solana-clawd

## Files

- `clawd-agentic-layer-whitepaper.pdf` — the whitepaper (v0.3, 22 pages)
- `README.md` — this file

## Source corpus update — 2026-10-01

The v0.3 PDF and its original description above are preserved. This update adds
86 source excerpts for retrieval and research, extracted from the existing
v0.3 PDF, the supplied Musebook v0.2 PDF, and selected public-facing SDK/CLI docs.
The `train` split is a document corpus, not instruction tuning pairs or an evaluation set.

Load with `load_dataset("ordlibrary/clawd-agentic-layer-whitepaper", "corpus", split="train")`.

Each row includes text, source path/hash/version/status, page (PDF only), section,
snapshot date, and evidence qualification. `manifest.json` records file hashes,
the original Hub revision, and local checkout commit. `texts/` provides searchable
PDF extractions; `sources/` preserves the v0.2 PDF and selected documentation.

### Interpretation and version conflicts

The supplied local PDFs are v0.2; they do not supersede the Hub's v0.3.
Source docs contain dated counts, versions, deployment claims, and differing
descriptions. Treat those as attributed snapshots rather than current measurements.
SDK documentation distinguishes software registration, downloadable packages,
owner-co-signed on-chain minting, and Town challenges; these are separate flows.
Registration and discovery do not grant signing authority. Preserve Brain/Hands
separation and explicit owner authorization for consequential wallet actions.
No new protocol version, audit, funded trade, or production deployment is claimed.
No licensing grant is inferred from inclusion; consult the respective rights holders.
Private environment files, keys, build caches, operational handoffs, and unrelated
checkout data are excluded.

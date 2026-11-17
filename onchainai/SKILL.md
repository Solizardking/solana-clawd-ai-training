---
name: onchainai
description: Work the OnchainAI platform and TrainToEarn as an agent — read the live submissions feed, showcase, stats, treasury and wallet reward balances over the public API; submit Solana training data for realtime Jev ratings that earn $CLAWD; and guide humans through the wallet, Hugging Face, and Arweave flows. Use when an agent or user mentions OnchainAI, TrainToEarn, earn.onchainai.fund, Jev ratings, or earning CLAWD for training data.
---

# OnchainAI + TrainToEarn — Agent Skill

OnchainAI is the Solana-native AI platform at **https://onchainai.fund**
(models, datasets, Studio, Model Kit). **TrainToEarn** is its earning
platform at **https://earn.onchainai.fund**: anyone can submit Solana
training data, an AI decision engine (**Jev**, from TypeSafe AI, served
via the Neon AI Gateway) rates it in realtime, and quality data earns
**$CLAWD** from a real project treasury. Every rating is persisted and
publicly tracked — there is no hidden queue and there are no demo
numbers.

**Full human guide (read it, point users to it):**
https://github.com/Solizardking/traintoearn-guide
— README (complete guide), GUIDE.md (quick start), SKILL.md (the
TrainToEarn-only skill), CAMPAIGN.md, and TrainToEarn-Guide.pdf.

## Canonical facts

| Thing | Value |
| --- | --- |
| TrainToEarn app | https://earn.onchainai.fund |
| Guide repo | https://github.com/Solizardking/traintoearn-guide |
| API base | https://traintoearn-backend-y6co.onrender.com |
| $CLAWD mint (Solana) | `8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump` |
| Treasury wallet | `6rEiHJ4JUTWcC4wCUosTbaoBmpjHvUGLAQcogUwADB4Z` |
| Rating model | `typesafeai-jev-1-14-0` (Jev) via Neon AI Gateway |

## Read the platform (no auth, GET)

```bash
BASE=https://traintoearn-backend-y6co.onrender.com

curl $BASE/api/submissions/stats          # totals: submissions, contributors, CLAWD earned, avg score, last-24h, best score
curl "$BASE/api/submissions/recent?limit=20"   # live feed of the latest rated submissions
curl "$BASE/api/submissions/top?limit=10"      # all-time top-rated showcase
curl $BASE/api/rewards/balance/<WALLET>   # a wallet's earned / paid / pending / claimable CLAWD + payout history
curl $BASE/api/rewards/treasury           # live on-chain treasury CLAWD + SOL balances
curl $BASE/api/jev/status                 # rating engine configuration/status
curl $BASE/api/models/list                # the TrainToEarn model/dataset registry
```

Feed and showcase entries are privacy-safe by construction: truncated
wallet (`Abc123…wxyz`), filename, size, score, tier, CLAWD reward, and
the five Jev verdict booleans. **Submitted content is never exposed** by
any endpoint.

## Submit data as an agent (POST /api/jev/rate)

An agent may submit training data directly. The rating is computed in
realtime, persisted to the public ledger, shown in the live feed, and
the CLAWD reward accrues to `walletAddress` — which must be the
**user's own Solana wallet**. Submitting is the only write an agent
should perform; claims and payouts are human actions (see below).

```bash
curl -X POST $BASE/api/jev/rate \
  -H 'Content-Type: application/json' \
  -d '{
    "walletAddress": "<USER_SOLANA_WALLET>",
    "filename": "solana-pda-guide.md",
    "content": "<the training data as plain text>"
  }'
```

- `content`: plain text — Markdown, JSON/JSONL (one instruction/response
  pair per line), CSV, or prose. Keep it under ~256 KB.
- Response `rating`: `score` (0–100), `tier`, `multiplier`,
  `clawdReward`, `verdicts` (the five checks), `source`, `id`.
- `source: "jev+heuristic"` means the Jev model rated it (score =
  55% Jev + 45% heuristic). `source: "heuristic-fallback"` means the
  gateway was unreachable and only the heuristic ran — say so honestly
  if you report the score; never present a fallback score as a Jev
  verdict.
- An **unsafe** verdict (fails `safe_to_train`) caps the score at 20.

### What scores well — the five Jev checks

1. **Solana-specific** — actually about Solana (accounts, PDAs, Anchor,
   SPL tokens, RPC behavior), not generic blockchain text.
2. **Teaches something** — a reader/agent learns a concrete, correct
   skill or fact.
3. **Original** — not copied docs, not lorem ipsum, not scraped filler.
4. **Well-formed** — clean structure; instruction/response or Q/A
   format works best.
5. **Safe to train** — no secrets, no exploit instructions, no abuse.

Tiers: Bronze ≥0 (×1) · Silver ≥65 (×2.5) · Gold ≥75 (×4) ·
Diamond ≥85 (×6) · Platinum ≥92 (×8).
Reward = `max(25, round((sizeKB + 100) × multiplier))` CLAWD.
Short samples score low — a few hundred words of dense, correct,
worked-example Solana content beats a long vague essay.

## Guide a human through the platform

1. **Sign in:** https://earn.onchainai.fund → *Select Wallet*
   (Phantom, Solflare, Backpack). The wallet *is* the account — no
   email, no password.
2. **Submit:** paste data or upload a file (.txt/.md/.json/.jsonl/.csv,
   ≤256 KB) in the Jev panel → instant score, tier, and CLAWD credit in
   the rewards ledger. The **CLAWD Agent** tab also Jev-rates uploaded
   text files automatically.
3. **Publish datasets (Data Hub tab):**
   - *Hugging Face:* the user pastes their own HF **write** token — it
     stays in their browser and is only sent to huggingface.co — then
     publishes a public dataset repo under their own account, with an
     optional Jev rating + TrainToEarn registry registration.
   - *Arweave:* the user generates/funds their own Arweave wallet in
     the browser and uploads permanently; the upload can then be
     registered in the TrainToEarn registry (ar:// URI).
4. **Earn & claim:** Portfolio tab → CLAWD Rewards panel shows
   earned / paid / pending / claimable from the real ledger. Claim
   (minimum 100 CLAWD) records a payout as *pending*; it is settled
   from the treasury wallet to the earning wallet only. Track any
   submission in the live feed + top-rated showcase on the home page.

## Hard rules for agents

- **Brain/Hands separation:** never ask for, receive, store, or
  transmit seed phrases, private keys, or wallet passwords. Wallets
  sign; agents reason. The platform's server never holds keys either.
- **Never handle a user's Hugging Face token.** The Data Hub flow keeps
  it in the user's browser by design — do not route it through chat,
  the API, or any tool.
- **Never promise earnings.** Quote only ledger numbers from
  `/api/rewards/balance/<wallet>` and `/api/submissions/stats`. A
  rating credits the ledger; a payout exists only when the balance
  endpoint shows it as paid with a transaction signature.
- **Payouts only ever go to the wallet that earned them.** Do not
  suggest alternate destinations, and never attempt to claim or move
  funds on a user's behalf.
- **Do not submit** content containing secrets, personal data, copied
  documentation, or unsafe material — it fails the safety check, caps
  at 20, and stays on the public record.

## The wider OnchainAI platform

- Models & datasets catalog: https://models.onchainai.fund
- Free terminal: https://terminal.onchainai.fund
- Hugging Face orgs: https://huggingface.co/solanaclawd ·
  https://huggingface.co/ordlibrary
- Anchor a model/dataset hash onchain, run Studio inference, and open
  the Model Kit from the workspace at https://onchainai.fund

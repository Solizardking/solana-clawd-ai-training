# 8 Bit Labs Static Site

This folder is the deployable static site for `8bitlabs.ai`.

## Contents

- `index.html` - full public site for 8 Bit Labs.
- `solana-redpill-verifier.html` - long-form article and browser-side verifier demo for `verify.8bitlabs.ai`.
- `styles.css` - responsive styling.
- `script.js` - live inference mesh health/model check.
- `api/clawd-chat.js` - server-side OpenRouter streaming chat endpoint for the Clawd character.
- `api/clawd-grok-sandbox.js` - guarded E2B launch/status endpoint for the Clawd Grok sandbox computer.
- `package.json` - `@8bitlabs/site` package with local preview, check, build, and deploy scripts.
- `assets/` - local Clawd imagery copied from `ai-training/`.
- `assets/8bitlabs-logo.png` - site logo copied from `/Users/8bit/Downloads/solana-clawd/icon.png`.
- `core-ai.json` - public manifest for the integrated `core-ai/` runtime modules.
- `clawd-code.json` - public manifest for the `clawd-code/` CLI and `clawd-code/web` preview adapter.
- `clawd-grok.json` - public manifest for the `core-ai/clawd-grok` sandbox computer package.
- `mesh-router.json` - public manifest for the live mesh, ZK router, endpoint map, and browser chat wiring.
- `registry.json` - machine-readable model-to-HF-to-PDA-to-program mapping.
- `.env.example` - server-side OpenRouter environment variables.
- `vercel.json` - static site plus Node serverless API configuration.
- `site.webmanifest` - install/app metadata and icon wiring.
- `404.html` - static-host fallback page.
- `CNAME` - custom domain config for static hosts that support it.
- `robots.txt` and `sitemap.xml` - basic crawler metadata.

The homepage includes public ecosystem links for Phantom, GitHub, OnChain AI,
models/register/x402 domains, Cheshire Terminal, and Hugging Face.
It also includes a Core AI integration section for the Clawd Code, Clawd Grok,
Helius MCP, Helius skills/plugin, perps agents, v3 runtime, and knowledge-base
lanes under `/Users/8bit/Downloads/solana-clawd/core-ai`.

The Clawd Code Web section is a public static adapter for
`/Users/8bit/Downloads/solana-clawd/clawd-code/web`. It mirrors the real app
shape: `app/page.tsx` renders `ChatLayout`, `components/` contains chat/layout/
tool/settings/file-viewer surfaces, `hooks/` carries browser interaction state,
and `lib/` carries store, API, streaming, export, search, and terminal helpers.
Runtime secrets and wallet files are intentionally excluded.

The Clawd Grok Computer section is the public launch surface for
`/Users/8bit/Downloads/solana-clawd/core-ai/clawd-grok`. It reads
`/clawd-grok.json`, checks `/api/clawd-grok-sandbox`, dry-runs the E2B plan,
and can run the safe `status --output json` smoke path when the deployment has
server-side `E2B_API_KEY` and `CLAWD_GROK_SANDBOX_ENABLED=true`. Provider keys,
Solana read keys, custom commands, and keep-alive mode are all off unless
explicitly enabled on the server.

The Live Mesh Router section reads public CORS-enabled endpoints from
`https://clawd-inference-mesh.fly.dev` and `https://clawdrouter-zk.fly.dev`.
It checks `/health`, `/models`, `/mesh/visualization`, `/flow`, and the ZK
router model registry, then lets visitors run a capped non-streaming completion
against the mesh's public `/v1/chat/completions` route. No admin keys, router
API keys, wallet secrets, or private headers are exposed.

The Solana RedPill Verifier article covers the local
`web/solana-redpill-verifier/` stack: the Pinocchio SVM program, TypeScript
client, IDL, Rust TEE gateway, TDX and NVIDIA evidence path, StoreProofV2 PDA
layout, SAS credential issuance, and the browser proof-chain demo. The Vercel
configuration rewrites `verify.8bitlabs.ai/` to this article when that domain is
attached to the project.

## Local Preview

One-shot from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/solizardking/solana-clawd/main/install.sh | bash -s -- --site
```

From the repo root:

```bash
npm run site:check
PORT=8088 npm run site:dev
```

This site can be opened directly in a browser:

```bash
open /Users/8bit/Downloads/solana-clawd/ai-training/8bitlabs.ai/index.html
```

Or served locally:

```bash
cd /Users/8bit/Downloads/solana-clawd/ai-training/8bitlabs.ai
npm run check
PORT=8088 npm run dev
```

Then open:

```text
http://127.0.0.1:8088
```

The static preview can render the chat UI, but `/api/clawd-chat` requires a
serverless runtime such as Vercel.

## OpenRouter Clawd Chat

Set these on the server, never in browser JavaScript:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_FREEMODEL=cohere/north-mini-code:free
OPENROUTER_REFERER=https://8bitlabs.ai
OPENROUTER_TITLE="8 Bit Labs Clawd Chat"
```

The endpoint is:

```text
POST /api/clawd-chat
```

It streams OpenRouter chat-completion chunks to the browser and uses a condensed
Clawd character prompt sourced from `library/knowledge/clawd-character.md` plus
the safer operational constraints from `CLAWD.md`.

## Clawd Grok Sandbox Computer

Operator path from the repo root:

```bash
npm run e2b:clawd-grok:dry
npm run e2b:clawd-grok
npm run e2b:clawd-grok -- --keep
```

Site/serverless launch is fail-closed. Set these on the deployment before the
homepage can execute a real sandbox smoke:

```bash
E2B_API_KEY=e2b_...
CLAWD_GROK_SANDBOX_ENABLED=true
```

Every real site launch also requires a connected Solana wallet to sign a fresh
launch proof. The API verifies that signature server-side, checks the wallet's
mainnet `$CLAWD` balance, and only calls `E2B Sandbox.create()` for wallets with
at least the `SHALLOW` holder tier, currently `1,000` `$CLAWD`.

Optional operator controls:

```bash
CLAWD_GROK_SANDBOX_TOKEN=<shared-server-token>
CLAWD_GROK_MIN_CLAWD_BALANCE=1000
CLAWD_GROK_FORWARD_PROVIDER_KEYS=true
CLAWD_GROK_FORWARD_SOLANA_READ_KEYS=true
CLAWD_GROK_ALLOW_KEEP=true
CLAWD_GROK_ALLOW_CUSTOM_COMMANDS=true
```

`/api/clawd-grok-sandbox` supports `GET` for readiness, `POST {"dryRun":true}`
for the plan, and guarded `POST {"walletProof":{...}}` for the default
`status --output json` smoke command. The shared token is not a public launch
gate; it only unlocks advanced operator options such as custom commands and
keep-alive runs when those features are separately enabled.

## Deployment

Deploy the contents of this directory as the root of `8bitlabs.ai`.

Package commands:

```bash
npm run vercel:build
npm run deploy
```

For the OpenRouter chat endpoint, deploy on Vercel or another host that supports
Node serverless functions from `api/`. Static-only hosts can still serve the
site, but the chat form will report that the endpoint needs configuration.

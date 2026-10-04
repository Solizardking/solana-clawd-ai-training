# Onchain AI

Production workspace: https://www.onchainai.fund (the apex redirects to www).
Vercel project: `prj_dbqrFMwxJRwz37ILdFrNWGmyLLVr`, team `clawd`.
Git project root must be **onchainai**, with `npm run build` and output `dist`.

The site combines public Hugging Face catalogs for **solanaclawd** and **ordlibrary**, the canonical root Model Kit frontend and downloadable source, a browser-local SFT dataset builder, hosted Core AI inference, Trading Factory strategy manifests, and wallet-signed Solana provenance.

## Routes

| Route | Behavior |
| --- | --- |
| `/` | Workspace overview and current public catalog counts |
| `/models` | Search all public models, datasets and Spaces in both namespaces |
| `/datasets` | Wallet-scoped local builder, previews, hashes, exports, optional HF publication |
| `/studio` | Free inference through the configured router or the hosted Core AI Space |
| `/chart-agent` | Local OHLCV CSV validation, closing-price chart, hash and report export |
| `/trading-factory` | Deterministic target allocations and exportable research strategy manifests |
| `/model-kit` | Artifact status, registry evidence, full console and source download |
| `/kit/` | Canonical Model Kit frontend; external/local runtime needed for arena execution |
| `/register` | Memo commitments on devnet/mainnet; native model registration on verified devnet |
| `/docs` | Docs viewer; `/docs/onchain.md` is the root doctrine copy |

## Development

```sh
npm ci
npm run sync:hub
npm run build
npm run dev
npm test
node tests/browser-smoke.mjs
```

The browser smoke uses a **read-only public-wallet fixture** to simulate transactions against live devnet. It has no private key and cannot sign or broadcast. Local dataset tests cover JSONL, Markdown, CSV, YAML, notebooks and PDF. Run the same checks on production with `SMOKE_BASE_URL=https://www.onchainai.fund`.

`src/` is the hub frontend source, `server/service.mjs` holds provider integrations, and `api/router.mjs` is the Vercel Node API. `dist/` is generated; the legacy `public/` copies are not deployed. The build reads `../model-kit/frontend` locally and the synchronized `kit/` snapshot on isolated deployment builds.

From the repository root, `python3 onchainai/scripts/package-kit.py` packages tracked Model Kit CLI, backend, frontend, docs and licensing without credentials. The ZIP is available from the Model Kit page.

## Custody and publication

Files are parsed and hashed in the browser. Generated dataset history is metadata in wallet-scoped local storage. It is not a server database or cross-device backup. Publication is explicit, creates a new HF repository, and requires a wallet message signing the repository, privacy setting, dataset and manifest. The HF write token is sent only in a multipart publication request, removed from the input, and not persisted. Namespace authorization is checked against HF identity; HF enforces repository write permissions. Secret-pattern scanning runs in both browser and server. The default repository visibility is private. No organization token is exposed or used for arbitrary public writes.

Text/PDF/notebook examples reproduce source excerpts and are labeled **needs-curation**. Structural validity does not constitute model quality evaluation. Scanned PDFs need external OCR. Publication supports up to 2 MB of generated JSONL; local sources up to 15 MB.

## Onchain evidence

`registry-idl.json` was recovered from devnet Anchor IDL account `5cqcAoyYWkWe3Yq6mXwVjVvV5j3EjDPo2N5P6EvSrvEF`. Before every native preview, the server verifies that program `3dLst2E3djtCSwG19mFS3REHxtZPngjyga7iYZLDL5xj` is executable and its live onchain IDL matches this snapshot.

Native registration constructs the IDL's `initialize_model` instruction using Solana Kit, creates the PDA at `[model, authority]`, fixes the model type to `TextGeneration`, and sets the reward rate to zero. The deployed program supports one model per wallet; the UI rejects an existing PDA. The preview includes account rent, transaction fees, the network and all memo contents. Mainnet native registration is disabled. Memo commitments are available on both networks.

Wallet Standard signing happens only after simulation and explicit review. Changes to inputs or wallet invalidate the preview. Confirmed transaction verification checks errors, signers, exact memo and, for native registration, decoded PDA authority and artifact hash. A memo is not a registry PDA, SAS attestation, ZK proof or ownership certificate. No user private keys are accepted.

## Inference and runtime limits

The ClawdRouter is probed at the configured `INFERENCE_BASE_URL` (default `https://clawdrouter-zk.fly.dev/v1`). Only metadata-verified free routes are accepted. The independent no-auth Core AI route is `solanaclawd/solana-clawd-core-ai-1.5b-lora` at `https://solanaclawd-clawd-free-chat.hf.space/api/free/chat`. Its health endpoint verifies the configured adapter. The Space does not include a model ID in its completion chunks; evidence marks returned model identity as absent and reports the configured identity separately. It does not issue ZK receipts. No automatic paid fallback occurs. Provider failures remain visible.

The Vercel project does not execute submitted code, launch GPU training jobs, persist private training data, or execute trading orders. The full Model Kit CLI/backend and Trading Factory source remain available for these workflows. Live trading requires a separately configured execution backend and fresh wallet approval.

## API

Read-only: `/api/health`, `/api/hub/catalog`, `/api/model-kit/status`, `/api/models`, `/api/attestations`, `/api/training/status`, `/api/protocol`, `/api/inference/models`, `/api/onchain/status`, `/api/onchain/registry`, `/api/onchain/verify`, and `/.well-known/clawd-registry.json`.

Writes/helpers: `/api/training/datasets` (explicit HF publication), `/api/inference/chat`, `/api/register/preview` (CAAP dry run), `/api/register` (dry run only; never claims a live write), `/api/rpc` (bounded method allowlist; only already-signed transactions can land onchain).

Optional server env: `SOLANA_DEVNET_RPC_URL`, `SOLANA_RPC_URL`, `INFERENCE_BASE_URL`, `INFERENCE_API_KEY`. Keys stay server-side. Public catalog access and the Core AI Space do not need `HF_TOKEN` or signup. Private HF jobs are not inferred from old job IDs or public repository metadata.

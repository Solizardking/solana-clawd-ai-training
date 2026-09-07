# Hauhau Trading Factory

Live app: https://solgpt-hauhau-trading-8b.fly.dev

The new Fly app serves Trading Factory 8B Q4_K_M, pinned to Hugging Face
revision `03954ae95d9d5d5ad86ed7679afb5f120988ec2e`. Both Hauhau model catalogs
include Q4_K_M, Q5_K_M, merged safetensors, and LoRA download artifacts.

## Runtime

The Python standard-library gateway serves the static UI on port 8080 immediately.
It starts llama.cpp on loopback port 8081; model loading does not block the site.
`/health` checks the gateway. `/api/model/health` checks inference readiness.
`/v1/*` forwards caller authorization to llama.cpp without injecting secrets.
The model operator key stays in Fly secrets. The existing holder chat flow still
uses the SolGPT desk gateway; it is separate from wallet-approved DFlow swaps.

`HAUHAU_MODEL_PROFILE` supports `qwen36`, `trading-factory-q4`, and
`trading-factory-q5`. The deployed profile is `trading-factory-q4`.
Each repository/revision has a separate persistent cache. Remove old explicit
HAUHAU_HF_ID / HAUHAU_GGUF_FILE / HAUHAU_GGUF_BYTES overrides when changing profiles.

## Live stream and swaps

The frontend connects to `wss://clawd-ws.fly.dev/ws`, retains at most 160 launches,
and reconnects with backoff. Pause freezes the rendered feed; incoming data stays
bounded. A launch's Trade action selects its mint in the swap panel.

Wallet Standard discovers compatible Solana wallets. `/api/trade/order` requests
an unsigned DFlow order. The UI shows amounts, minimum output, slippage, price
impact, fees, and a 30-second review expiry. Account/pair/amount changes invalidate
the review. Only the explicit approval button opens wallet signing/broadcast.
The server never accepts a private key, signs a transaction, or broadcasts one.
Confirmation uses the returned signature and original lastValidBlockHeight.
Unknown results retain their explorer link; the app never resubmits automatically.

Without `DFLOW_API_KEY`, the server uses `https://dev-quote-api.dflow.net` and the
UI labels the rate-limited developer mode. Set a production key as a Fly secret
to use `https://quote-api.dflow.net`. `SOLANA_RPC_URL` is an optional server-only
read RPC for token decimals and signature status; public mainnet is the default.
There are no builder fees. DFlow automatic priority fees are capped at 0.005 SOL;
network fees and account rent are additional and shown by the wallet.

The Nemotron sibling remains a tokenizer service and has the release catalog.
Its own Fly config is not replaced by this new inference app.

## Develop and deploy

From the sol-gpt repository root:

```sh
node hauhau/trading-ui/build.mjs
node --test hauhau/trading-ui/tests/trade-core.test.mjs
python3 hauhau/tests/test_gateway.py
npx vitest run src/lib/solgpt/hauhau-site-login.test.ts
HAUHAU_START_MODEL=0 PORT=8898 python3 hauhau/gateway.py
```

The checked-in browser bundle is small and loads when the trading panel approaches
the viewport. No build tool or npm dependencies are required in the production image.

From `hauhau/`, deploy with `fly deploy --remote-only --depot=false`.

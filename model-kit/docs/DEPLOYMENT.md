# Model Kit Site Deployment

This folder ships one Fly Machine that serves both surfaces:

- Site + API on Fly Machines: `https://solana-clawd-model-kit.fly.dev`
- `models.x402.wtf` / `register.x402.wtf` can CNAME to that app after certificates are attached.

The frontend is static. It never stores registry tokens. Live registration is
proxied through the API only when the page sends an explicit live request. On
Fly, the API and HTML share one origin so the browser uses same-origin `/api/*`.

## Fly Machines

Config lives next to the site:

```text
model-kit/fly.toml
model-kit/Dockerfile.fly
```

The image copies `frontend/` and `backend/`, then `render_start.py` listens on
`PORT` (8080 on Fly). HTML routes:

| Path / host | Page |
| --- | --- |
| `/`, `/models`, `/model-kit` | `frontend/index.html` |
| `/register`, `/register.html` | `frontend/register.html` |
| `Host: register.x402.wtf` + `/` | `frontend/register.html` |
| `/api/health`, `/health` | JSON health check |
| `/api` | JSON service index (API-only clients) |

Deploy from this directory (remote builder, no local Docker required):

```bash
export FLY_API_TOKEN=...   # fly tokens create deploy
cd model-kit
chmod +x scripts/deploy-fly.sh
./scripts/deploy-fly.sh
```

Equivalent explicit command:

```bash
flyctl apps create solana-clawd-model-kit   # first time only
flyctl deploy model-kit \
  --config model-kit/fly.toml \
  --app solana-clawd-model-kit \
  --ha=false \
  --yes \
  --remote-only
```

Optional arena/provider secrets (not required for the site to boot):

```bash
flyctl secrets set OPENROUTER_API_KEY="$OPENROUTER_API_KEY" --app solana-clawd-model-kit
```

Do not put API keys in `fly.toml`. After the first Machine is up:

```bash
curl -sS https://solana-clawd-model-kit.fly.dev/api/health
curl -sS https://solana-clawd-model-kit.fly.dev/api/model-kit/status
curl -sS https://solana-clawd-model-kit.fly.dev/.well-known/clawd-model-kit.json
```

Custom domains (optional, after DNS):

```bash
flyctl certs add models.x402.wtf --app solana-clawd-model-kit
flyctl certs add register.x402.wtf --app solana-clawd-model-kit
```

Point both names at the IPv6 AAAA (and dedicated IPv4 if you allocate one) that
`flyctl ips list --app solana-clawd-model-kit` prints, then set:

| Name | Value |
| --- | --- |
| `MODELS_HOME` | `https://models.x402.wtf` |
| `REGISTER_HOME` | `https://register.x402.wtf` |
| `MODEL_KIT_CORS_ORIGINS` | include both custom hosts and `https://solana-clawd-model-kit.fly.dev` |

GitHub Actions workflow `.github/workflows/fly-model-kit.yml` deploys on pushes
to `main` when `model-kit/**` changes. Store `FLY_API_TOKEN` as a repository
secret.

## Render API

Use the blueprint in this folder if you still want an API-only Render service
(no static site in that image):

```bash
cd /Users/8bit/Downloads/solana-clawd
render blueprint launch ai-training/model-kit/render.yaml
```

If using the Render dashboard, import the GitHub repo and point the blueprint to:

```text
ai-training/model-kit/render.yaml
```

The service root is:

```text
ai-training/model-kit/backend
```

Required public env:

| Name | Value |
| --- | --- |
| `ONCHAIN_REGISTRY_HOME` | `https://onchain.x402.wtf` |
| `ONCHAIN_REGISTRY_URL` | `https://onchain.x402.wtf/api/register` |
| `X402_HOME` | `https://x402.wtf` |
| `MODELS_HOME` | `https://models.x402.wtf` |
| `REGISTER_HOME` | `https://register.x402.wtf` |
| `MODEL_KIT_CORS_ORIGINS` | `https://models.x402.wtf,https://register.x402.wtf` |

Optional secret env:

| Name | Use |
| --- | --- |
| `ONCHAIN_REGISTRY_TOKEN` | Server-side bearer token for registry writes when users do not pass a request token. |

Smoke checks:

```bash
curl -sS https://x402-model-kit-docker-api.onrender.com/api/health
curl -sS https://x402-model-kit-docker-api.onrender.com/api/model-kit/status
curl -sS https://x402-model-kit-docker-api.onrender.com/.well-known/clawd-model-kit.json
```

## Vercel Frontend

Set the Vercel project root directory to:

```text
ai-training/model-kit
```

Build settings:

| Setting | Value |
| --- | --- |
| Framework | Other |
| Build command | `npm run build` |
| Output directory | `frontend` |

Before deploy, set `frontend/config.js` to the Fly (or Render) API URL:

```js
window.MODEL_KIT_CONFIG = {
  apiBaseUrl: "https://solana-clawd-model-kit.fly.dev",
  x402Home: "https://x402.wtf",
  modelsHome: "https://models.x402.wtf",
  registerHome: "https://register.x402.wtf",
  onchainHome: "https://onchain.x402.wtf",
  githubRepo: "https://github.com/solizardking/solana-clawd-ai-training",
};
```

Pages served from `*.fly.dev` ignore `apiBaseUrl` and call same-origin `/api/*`.

Deploy:

```bash
cd /Users/8bit/Downloads/solana-clawd/ai-training/model-kit
npm run build
vercel deploy --prod
```

Attach both domains to the same Vercel project if you are not yet using Fly
custom domains:

| Domain | Served page |
| --- | --- |
| `models.x402.wtf` | `/index.html` |
| `register.x402.wtf` | `/register.html` via host rewrite |

## Registration Flow

The register page calls:

| Route | Method | Use |
| --- | --- | --- |
| `/api/register/preview` | `POST` | Build a dry-run CAAP/1.0 payload. |
| `/api/register` | `POST` | Dry-run unless the request has `live: true`. |

Live requests require a real `model_hash` by default. Provisional generated
hashes are allowed only when `allow_generated_hash: true` is sent.

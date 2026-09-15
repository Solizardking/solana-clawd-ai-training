# onchainai

Static hub for Solana Clawd AI: Hugging Face model/dataset catalog, local data inventory, studio, docs, and the full **Solana AI Model Kit**.

## Surfaces

| Path | What |
| --- | --- |
| `/` | Hub — featured models, datasets, kit hero |
| `/kit` | Model Kit frontend console |
| `/register` | Model registration UI |
| `/studio` | Studio |
| `/data` | Local training data catalog |
| `/docs` | Markdown docs viewer (includes `docs/model-kit/`) |
| `/model-kit/` | Full kit source (frontend, backend, bin, scripts, docs) |

Brand assets live in `brand/` (Solana Sans/Mono + x402 marks).

## Deploy

```bash
cd onchainai
vercel --prod --yes --scope clawd-c4b28c7e
```

The FastAPI backend under `model-kit/backend` is not hosted by this static Vercel project — use Render (`model-kit/render.yaml`) or local `clawd-model-kit` for the API.

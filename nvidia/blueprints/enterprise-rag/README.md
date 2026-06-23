# Blueprint 5: Build an Enterprise RAG Pipeline

https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline

NeMo Retriever RAG pipeline over Solana documentation, Clawd skills,
protocol specs, and training data — providing grounded answers to
Clawd agent queries without hallucination.

## Architecture

```
Solana docs + PDFs + skills
  └─► ingest.py   ← nv-ingest PDF extraction + chunking
        └─► NeMo Retriever embedding (nvidia/nv-embedqa-e5-v5)
              └─► Vector store (local FAISS / NVIDIA cuVS)
                    └─► query.py  ← RAG retrieval + NIM rerank + generation
                          └─► pipeline.py  ← end-to-end API
```

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Document ingestion: PDF, JSONL, MD → chunked embeddings |
| `query.py` | RAG query: embed query → retrieve → rerank → generate |
| `pipeline.py` | End-to-end pipeline with FastAPI endpoint |

## Quick start

```bash
export NVIDIA_API_KEY=nvapi-...

# Ingest Solana docs into the vector store
python3 blueprints/enterprise-rag/ingest.py \
  --sources ../../data/ ../../README.md \
  --store ../../data/nvidia_rag_store

# Query
python3 blueprints/enterprise-rag/query.py \
  --store ../../data/nvidia_rag_store \
  --question "What is the funding rate on SOL-PERP?"

# Serve as API
python3 blueprints/enterprise-rag/pipeline.py --port 8765
```

## Hosted API

The Fly deployment serves the same FastAPI app at:

```text
https://solana-clawd-rag.fly.dev
```

Endpoints:

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Service index with route names and active store path |
| `/about` | `GET` | Public education page explaining how the RAG service works |
| `/health` | `GET` | Health check used by Fly |
| `/query` | `POST` | Retrieve Solana/Clawd context from FAISS and generate an answer |
| `/admin` | `GET` | Password-protected runtime dashboard |
| `/admin/api/status` | `GET` | Protected status JSON for the admin dashboard |

Query example:

```bash
curl -sS https://solana-clawd-rag.fly.dev/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the Solana Clawd NVIDIA RAG pipeline do?","top_k":5}'
```

Response shape:

```json
{
  "answer": "...",
  "question": "What does the Solana Clawd NVIDIA RAG pipeline do?",
  "sources": [
    {
      "source": "nvidia/blueprints/enterprise-rag/README.md",
      "score": 0.0,
      "snippet": "..."
    }
  ]
}
```

The image currently bakes in `data/nvidia_rag_store`, which is a small FAISS
index built from local Solana Clawd docs and blueprint material. With
`NVIDIA_API_KEY` configured, the API uses NVIDIA embedding, reranking, and
Nemotron/NIM generation. Without that secret, the Fly service runs in
retrieval-only mode: it performs FAISS retrieval with the hash-embedding
fallback and returns the retrieved context instead of calling a local Ollama
daemon.

## Fly deployment

Deploy from the `ai-training/` directory so the Docker build context contains
both the RAG source and `data/nvidia_rag_store`:

```bash
cd /Users/8bit/Downloads/solana-clawd/ai-training

# First deploy or update
flyctl deploy . \
  --config nvidia/blueprints/enterprise-rag/fly.toml \
  --app solana-clawd-rag \
  --ignorefile nvidia/blueprints/enterprise-rag/Dockerfile.fly.dockerignore \
  --vm-cpus 1 \
  --vm-memory 512 \
  --ha=false \
  --yes
```

Set secrets from your local shell or secret manager:

```bash
flyctl secrets set NVIDIA_API_KEY="$NVIDIA_API_KEY" --app solana-clawd-rag
flyctl secrets set CLAWD_RAG_ADMIN_KEY="$CLAWD_RAG_ADMIN_KEY" --app solana-clawd-rag
```

The admin key is required for `/admin`. It must be stored as a Fly secret; do
not put it in `fly.toml`, markdown, YAML, JSON, or committed shell scripts.
After login, the admin dashboard can inspect store health, test `/query`, and
change in-memory runtime controls such as `default_top_k`, `max_top_k`, and
retrieval-only mode. Persistent changes should be made through Fly env vars or
secrets and then redeployed.

Refresh the hosted knowledge base by rebuilding the local store, then redeploy.
The ingestion pipeline indexes Markdown, JSONL, PDFs, Python files, TOML,
Dockerfiles, ignore files, JSON/YAML, and requirements text. It also writes
`data/nvidia_rag_store/manifest.json`, which is exposed through `/health`.

```bash
python3 nvidia/blueprints/enterprise-rag/ingest.py \
  --sources README.md nvidia/README.md nvidia/blueprints \
  --store data/nvidia_rag_store

flyctl deploy . \
  --config nvidia/blueprints/enterprise-rag/fly.toml \
  --app solana-clawd-rag \
  --ignorefile nvidia/blueprints/enterprise-rag/Dockerfile.fly.dockerignore \
  --ha=false \
  --yes
```

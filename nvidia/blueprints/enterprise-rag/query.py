"""
Blueprint 5 — Enterprise RAG: query interface.

Embeds the query, retrieves top-k chunks from FAISS,
optionally re-ranks with NeMo Retriever reranker,
then generates an answer via NVIDIA NIM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


RERANK_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"
GEN_MODEL = os.environ.get("CLAWD_RAG_GEN_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
OLLAMA_MODEL = os.environ.get("CLAWD_RAG_OLLAMA_MODEL", "8bit/solana-clawd-core-ai:latest")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_CONTEXT_ONLY = os.environ.get("CLAWD_RAG_CONTEXT_ONLY", "").lower() in {"1", "true", "yes"}
_CONTEXT_ONLY_OVERRIDE: bool | None = None


def set_context_only_override(value: bool | None) -> None:
    global _CONTEXT_ONLY_OVERRIDE
    _CONTEXT_ONLY_OVERRIDE = value


def context_only_enabled() -> bool:
    if _CONTEXT_ONLY_OVERRIDE is not None:
        return _CONTEXT_ONLY_OVERRIDE
    return DEFAULT_CONTEXT_ONLY


def _embed(text: str) -> list[float]:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return _embed_fallback(text)
    try:
        import httpx
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "nvidia/nv-embedqa-e5-v5", "input": [text], "encoding_format": "float"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception:
        return _embed_fallback(text)


def _embed_fallback(text: str) -> list[float]:
    import hashlib, math
    dim = 256
    vec = [0.0] * dim
    for word in text.split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def retrieve(query: str, store_path: Path, top_k: int = 10) -> list[dict]:
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("FAISS not installed.", file=sys.stderr)
        return []

    index = faiss.read_index(str(store_path / "index.faiss"))
    chunks_raw = (store_path / "chunks.jsonl").read_text().strip().split("\n")
    chunks = [json.loads(c) for c in chunks_raw if c]

    q_embedding = _embed_fallback(query) if _store_uses_local_embeddings(store_path, index.d) else _embed(query)
    if len(q_embedding) != index.d:
        print(
            f"Embedding dimension mismatch: query={len(q_embedding)} index={index.d}. "
            "Rebuild the store with the same embedding backend used at query time.",
            file=sys.stderr,
        )
        return []
    q_vec = np.array([q_embedding], dtype=np.float32)
    k = min(top_k, index.ntotal)
    distances, indices = index.search(q_vec, k)
    return [
        {"text": chunks[i]["text"], "meta": chunks[i]["meta"], "score": float(distances[0][j])}
        for j, i in enumerate(indices[0])
        if i < len(chunks)
    ]


def _store_uses_local_embeddings(store_path: Path, index_dim: int) -> bool:
    manifest_path = store_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            model = str(manifest.get("embedding_model", ""))
            if model == "local-hash-fallback":
                return True
            if model:
                return False
        except json.JSONDecodeError:
            pass
    return index_dim == 256


def rerank(query: str, passages: list[dict]) -> list[dict]:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key or not passages:
        return passages
    try:
        import httpx
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/ranking",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": RERANK_MODEL,
                "query": {"text": query},
                "passages": [{"text": p["text"]} for p in passages],
            },
            timeout=30,
        )
        r.raise_for_status()
        rankings = r.json().get("rankings", [])
        ranked = sorted(
            zip(rankings, passages),
            key=lambda x: x[0].get("logit", 0),
            reverse=True,
        )
        return [p for _, p in ranked]
    except Exception:
        return passages


def generate(query: str, context: str) -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return generate_ollama(query, context)
    try:
        import httpx
        system = (
            "You are Clawd, a Solana-native AI agent. Answer using only the provided context. "
            "If the context does not contain enough information, say so."
        )
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GEN_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        if context_only_enabled():
            return context_only_answer(context, reason=f"NVIDIA generation failed: {type(e).__name__}")
        fallback = generate_ollama(query, context)
        if fallback.startswith("[ollama unavailable"):
            return context_only_answer(context, reason=f"NVIDIA generation failed: {type(e).__name__}")
        return fallback


def generate_ollama(query: str, context: str) -> str:
    if context_only_enabled():
        return context_only_answer(context)

    prompt = (
        "You are Clawd, a Solana-native AI agent. Answer using only the provided context. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        answer = (body.get("response") or "").strip()
        if answer:
            return answer
        return f"[ollama unavailable: empty response from {OLLAMA_MODEL}]"
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return f"[ollama unavailable: {exc}; context retrieved:\n{context[:500]}]"


def context_only_answer(context: str, max_chars: int = 1800, reason: str | None = None) -> str:
    if not context.strip():
        return (
            "No generation backend is configured and the retriever did not return context. "
            "Set NVIDIA_API_KEY for NVIDIA/NIM generation, or provide an Ollama backend locally."
        )
    if reason:
        return (
            f"{reason}; returning retrieval-only context.\n\n"
            f"Retrieved context:\n{context[:max_chars]}"
        )
    return (
        "No generation backend is configured, so this response is retrieval-only. "
        "Set NVIDIA_API_KEY on the service for NVIDIA embeddings, reranking, and generation.\n\n"
        f"Retrieved context:\n{context[:max_chars]}"
    )


def rag_query(query: str, store_path: Path, top_k: int = 5) -> str:
    return rag_query_with_sources(query, store_path, top_k)["answer"]


def rag_query_with_sources(query: str, store_path: Path, top_k: int = 5) -> dict:
    passages = retrieve(query, store_path, top_k=top_k * 2)
    passages = rerank(query, passages)[:top_k]
    context = "\n\n---\n\n".join(p["text"] for p in passages)
    answer = generate(query, context)
    return {
        "answer": answer,
        "sources": _source_summaries(passages),
    }


def _source_summaries(passages: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for passage in passages:
        meta = passage.get("meta", {})
        source = str(meta.get("source", "unknown"))
        snippet = " ".join(passage.get("text", "").split())[:360]
        key = (source, snippet)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": source,
            "score": passage.get("score"),
            "snippet": snippet,
        })
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the NVIDIA RAG pipeline")
    parser.add_argument("--store", default="data/nvidia_rag_store")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = Path(args.store)
    if not (store / "index.faiss").exists():
        print(f"ERROR: RAG store not found at {store}. Run ingest.py first.", file=sys.stderr)
        sys.exit(1)

    result = rag_query_with_sources(args.question, store, args.top_k)
    print(f"\nQ: {args.question}\n\nA: {result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src['source']}")


if __name__ == "__main__":
    main()

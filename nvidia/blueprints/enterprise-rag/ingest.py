"""
Blueprint 5 — Enterprise RAG: document ingestion pipeline.

Reads Solana docs (PDFs, JSONL, Markdown) via nv-ingest,
chunks text, embeds with NeMo Retriever, stores in FAISS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TEXT_SUFFIXES = {
    ".dockerignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {"Dockerfile", "Dockerfile.fly", "requirements.txt", "requirements-fly.txt"}
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "node_modules"}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += size - overlap
    return chunks


def read_jsonl_docs(path: Path) -> Generator[str, None, None]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in obj.get("messages", []):
                if m.get("role") == "assistant":
                    yield m.get("content", "")


def read_markdown(path: Path) -> Generator[str, None, None]:
    yield path.read_text(errors="replace")


def read_text_file(path: Path) -> Generator[str, None, None]:
    yield path.read_text(errors="replace")


def read_pdf_nvidia(path: Path) -> Generator[str, None, None]:
    """Use nv-ingest for PDF extraction when NVIDIA_API_KEY is available."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        yield _read_pdf_fallback(path)
        return
    try:
        from nv_ingest_client.client import NvIngestClient  # type: ignore
        client = NvIngestClient()
        result = client.submit_job(str(path), extract_text=True, extract_tables=True)
        for page in result.get("pages", []):
            yield page.get("text", "")
    except ImportError:
        yield _read_pdf_fallback(path)


def _read_pdf_fallback(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def embed_batch(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        return _embed_local(texts)
    try:
        import httpx
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": EMBED_MODEL, "input": texts, "encoding_format": "float"},
            timeout=60,
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    except Exception:
        return _embed_local(texts)


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Fallback: simple TF-IDF-style hash embedding (not for production)."""
    import math
    dim = 256
    result = []
    for text in texts:
        vec = [0.0] * dim
        for word in text.split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        result.append([x / norm for x in vec])
    return result


def build_store(sources: list[Path], store_path: Path) -> None:
    """Ingest all sources into a FAISS vector store."""
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("FAISS not installed. Run: pip install faiss-cpu numpy", file=sys.stderr)
        sys.exit(1)

    all_chunks: list[str] = []
    all_meta: list[dict] = []

    for src in sources:
        if src.is_dir():
            files = [
                path for path in sorted(src.rglob("*"))
                if path.is_file() and _is_supported(path) and not _should_skip(path)
            ]
        else:
            files = [src]

        for f in files:
            print(f"  [ingest] {f}")
            if f.suffix == ".jsonl":
                docs = list(read_jsonl_docs(f))
            elif f.suffix == ".pdf":
                docs = list(read_pdf_nvidia(f))
            elif _is_supported(f):
                docs = list(read_text_file(f))
            else:
                docs = []

            if f.suffix == ".md":
                docs = list(read_markdown(f))

            for doc in docs:
                for chunk in chunk_text(doc):
                    all_chunks.append(chunk)
                    all_meta.append({"source": str(f), "len": len(chunk)})

    print(f"[ingest] total chunks: {len(all_chunks)}")

    batch_size = 32
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embs = embed_batch(batch)
        all_embeddings.extend(embs)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  embedded {i + len(batch)}/{len(all_chunks)}")

    vecs = np.array(all_embeddings, dtype=np.float32)
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)

    store_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_path / "index.faiss"))
    (store_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "meta": m}) for t, m in zip(all_chunks, all_meta))
    )
    sources_summary: dict[str, int] = {}
    for meta in all_meta:
        source = meta["source"]
        sources_summary[source] = sources_summary.get(source, 0) + 1
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBED_MODEL if os.environ.get("NVIDIA_API_KEY") else "local-hash-fallback",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_dim": dim,
        "chunks": len(all_chunks),
        "sources": sources_summary,
    }
    (store_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ingest] store saved to {store_path}  (dim={dim}, n={len(all_chunks)})")


def _is_supported(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    if path.suffix in TEXT_SUFFIXES:
        return True
    return path.suffix == ".pdf"


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into NVIDIA RAG store")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--store", default="data/nvidia_rag_store")
    args = parser.parse_args()

    sources = [Path(s) for s in args.sources]
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"ERROR: sources not found: {missing}", file=sys.stderr)
        sys.exit(1)

    build_store(sources, Path(args.store))


if __name__ == "__main__":
    main()

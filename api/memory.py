"""
memory.py — Persistent long-term memory for the LLM API.

Architecture:
  - ChromaDB (local vector DB) stores all knowledge as embedded chunks
  - nomic-embed-text (via Ollama) converts text → vectors
  - Every ingested article, document, and conversation turn is stored
  - At query time, semantic search retrieves the most relevant chunks
  - Retrieved context is injected into the model prompt automatically

Collections:
  - "knowledge"      : articles, documents, web pages
  - "conversations"  : past conversation turns (summarised)

Each chunk stored with metadata:
  source_type, source_id, title, url/filename, timestamp, chunk_index
"""

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_HOST      = os.getenv("CHROMA_HOST",      "chromadb")
CHROMA_PORT      = int(os.getenv("CHROMA_PORT",  "8000"))
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://ollama:11434")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "nomic-embed-text")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",   "800"))   # chars per chunk
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP","150"))   # overlap between chunks
TOP_K            = int(os.getenv("MEMORY_TOP_K", "5"))     # chunks to retrieve

_chroma_client   = None
_knowledge_col   = None
_conv_col        = None


# ── ChromaDB client (lazy init) ───────────────────────────────────────────────
def _get_chroma():
    global _chroma_client, _knowledge_col, _conv_col
    if _chroma_client is not None:
        return _chroma_client, _knowledge_col, _conv_col
    import chromadb
    _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _knowledge_col = _chroma_client.get_or_create_collection(
        name="knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    _conv_col = _chroma_client.get_or_create_collection(
        name="conversations",
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_client, _knowledge_col, _conv_col


def is_available() -> bool:
    """Check if ChromaDB is reachable."""
    try:
        _, kc, _ = _get_chroma()
        kc.count()
        return True
    except Exception:
        return False


# ── Embedding ─────────────────────────────────────────────────────────────────
async def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using nomic-embed-text via Ollama.
    Returns list of float vectors, one per input text.
    """
    embeddings = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return embeddings


# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries where possible.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Try to cut at sentence boundary
        boundary = max(
            text.rfind(". ", start, end),
            text.rfind(".\n", start, end),
            text.rfind("\n\n", start, end),
        )
        if boundary > start + chunk_size // 2:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return [c for c in chunks if len(c) > 50]


def _source_id(source_type: str, identifier: str) -> str:
    """Stable ID for a source based on type + URL/filename."""
    return hashlib.sha256(f"{source_type}:{identifier}".encode()).hexdigest()[:16]


# ── Store ─────────────────────────────────────────────────────────────────────
async def store_knowledge(
    text: str,
    title: str,
    source_type: str,           # "url", "youtube", "pdf", "docx", "txt", etc.
    identifier: str,            # URL or filename
    extra_meta: dict | None = None,
) -> dict:
    """
    Chunk text, embed, and store in the knowledge collection.
    Returns summary: {source_id, chunks_stored, chars_total}
    Deduplicates by source_id — re-ingesting same URL replaces old chunks.
    """
    try:
        _, col, _ = _get_chroma()
    except Exception as e:
        return {"error": f"ChromaDB unavailable: {e}", "chunks_stored": 0}

    sid    = _source_id(source_type, identifier)
    chunks = _chunk_text(text)
    now    = datetime.now(timezone.utc).isoformat()

    # Delete existing chunks for this source (re-ingest / update)
    try:
        existing = col.get(where={"source_id": sid})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    # Embed all chunks
    try:
        embeddings = await _embed(chunks)
    except Exception as e:
        return {"error": f"Embedding failed: {e}", "chunks_stored": 0}

    ids       = [f"{sid}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_id":   sid,
            "source_type": source_type,
            "title":       title[:200],
            "identifier":  identifier[:500],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp":   now,
            **(extra_meta or {}),
        }
        for i in range(len(chunks))
    ]

    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {
        "source_id":    sid,
        "chunks_stored": len(chunks),
        "chars_total":  len(text),
        "title":        title,
    }


async def store_conversation_turn(
    conv_id: str,
    conv_name: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    Store a conversation exchange as a single memory chunk.
    Summarised format so it's compact but retrievable.
    """
    try:
        _, _, col = _get_chroma()
    except Exception:
        return  # Memory unavailable — fail silently, don't break chat

    text = f"Conversation: {conv_name}\nUser: {user_msg}\nAssistant: {assistant_msg}"
    chunk_id = f"conv_{conv_id}_{int(time.time() * 1000)}"
    now = datetime.now(timezone.utc).isoformat()

    try:
        embeddings = await _embed([text])
        col.add(
            ids=[chunk_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[{
                "source_id":   conv_id,
                "source_type": "conversation",
                "title":       conv_name,
                "identifier":  conv_id,
                "timestamp":   now,
            }],
        )
    except Exception:
        pass  # Memory storage is best-effort


# ── Retrieve ──────────────────────────────────────────────────────────────────
async def retrieve(
    query: str,
    top_k: int = TOP_K,
    source_types: list[str] | None = None,   # filter by type if set
    include_conversations: bool = True,
) -> list[dict]:
    """
    Semantic search across knowledge + conversations.
    Returns list of {text, title, source_type, identifier, score, timestamp}.
    """
    try:
        _, k_col, c_col = _get_chroma()
    except Exception:
        return []

    try:
        q_embedding = (await _embed([query]))[0]
    except Exception:
        return []

    results = []

    # Query knowledge collection
    try:
        where = {"source_type": {"$in": source_types}} if source_types else None
        kr = k_col.query(
            query_embeddings=[q_embedding],
            n_results=min(top_k, max(k_col.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            kr["documents"][0], kr["metadatas"][0], kr["distances"][0]
        ):
            results.append({
                "text":        doc,
                "title":       meta.get("title", ""),
                "source_type": meta.get("source_type", ""),
                "identifier":  meta.get("identifier", ""),
                "score":       round(1 - dist, 4),   # cosine distance → similarity
                "timestamp":   meta.get("timestamp", ""),
                "collection":  "knowledge",
            })
    except Exception:
        pass

    # Query conversations collection
    if include_conversations:
        try:
            conv_count = c_col.count()
            if conv_count > 0:
                cr = c_col.query(
                    query_embeddings=[q_embedding],
                    n_results=min(top_k, conv_count),
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(
                    cr["documents"][0], cr["metadatas"][0], cr["distances"][0]
                ):
                    results.append({
                        "text":        doc,
                        "title":       meta.get("title", ""),
                        "source_type": "conversation",
                        "identifier":  meta.get("identifier", ""),
                        "score":       round(1 - dist, 4),
                        "timestamp":   meta.get("timestamp", ""),
                        "collection":  "conversations",
                    })
        except Exception:
            pass

    # Sort by relevance score descending, return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def build_memory_context(chunks: list[dict], max_chars: int = 3000) -> str:
    """
    Format retrieved chunks into a context block for injection into the prompt.
    """
    if not chunks:
        return ""

    lines = ["── RELEVANT MEMORY ──────────────────────────────────────────"]
    total = 0
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['source_type'].upper()}: {c['title'] or c['identifier']}"
        if c.get("timestamp"):
            try:
                ts = c["timestamp"][:10]
                header += f" ({ts})"
            except Exception:
                pass
        entry = f"{header}\n{c['text']}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    lines.append("─────────────────────────────────────────────────────────")
    return "\n\n".join(lines)


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_stats() -> dict:
    """Return knowledge base statistics."""
    try:
        _, k_col, c_col = _get_chroma()
        k_count = k_col.count()
        c_count = c_col.count()

        # Get unique sources
        k_meta  = k_col.get(include=["metadatas"]) if k_count > 0 else {"metadatas": []}
        sources = {}
        for m in k_meta["metadatas"]:
            sid = m.get("source_id", "")
            if sid not in sources:
                sources[sid] = {
                    "title":       m.get("title", ""),
                    "source_type": m.get("source_type", ""),
                    "identifier":  m.get("identifier", ""),
                    "timestamp":   m.get("timestamp", ""),
                }

        return {
            "available":           True,
            "knowledge_chunks":    k_count,
            "conversation_chunks": c_count,
            "unique_sources":      len(sources),
            "sources":             sorted(
                sources.values(),
                key=lambda x: x["timestamp"],
                reverse=True,
            )[:50],  # cap at 50 for response size
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


async def delete_source(source_id: str) -> dict:
    """Delete all chunks for a given source_id from knowledge collection."""
    try:
        _, col, _ = _get_chroma()
        existing = col.get(where={"source_id": source_id})
        if not existing["ids"]:
            return {"deleted": 0, "source_id": source_id}
        col.delete(ids=existing["ids"])
        return {"deleted": len(existing["ids"]), "source_id": source_id}
    except Exception as e:
        return {"error": str(e), "source_id": source_id}

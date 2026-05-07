"""Embedded ChromaDB persistent vector store + Ollama embeddings."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime, timezone

import ollama as oll
from config import CFG, CHROMA_DIR

log = logging.getLogger("llm-api.memory")

_client = None
_knowledge = None
_conversations = None


def _ensure() -> tuple:
    global _client, _knowledge, _conversations
    if _client is not None:
        return _client, _knowledge, _conversations
    import chromadb
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _knowledge = _client.get_or_create_collection(
        name="knowledge", metadata={"hnsw:space": "cosine"},
    )
    _conversations = _client.get_or_create_collection(
        name="conversations", metadata={"hnsw:space": "cosine"},
    )
    return _client, _knowledge, _conversations


def is_available() -> bool:
    if not CFG.memory_enabled:
        return False
    try:
        _, k, _ = _ensure()
        k.count()
        return True
    except Exception as e:
        log.warning("memory not available: %s", e)
        return False


# ── Chunking ─────────────────────────────────────────────────────────────────
def _chunks(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= CFG.chunk_size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = start + CFG.chunk_size
        if end >= len(text):
            out.append(text[start:].strip())
            break
        boundary = max(
            text.rfind(". ", start, end),
            text.rfind(".\n", start, end),
            text.rfind("\n\n", start, end),
        )
        if boundary > start + CFG.chunk_size // 2:
            end = boundary + 1
        out.append(text[start:end].strip())
        start = end - CFG.chunk_overlap
    return [c for c in out if len(c) > 50]


def _source_id(source_type: str, identifier: str) -> str:
    return hashlib.sha256(f"{source_type}:{identifier}".encode()).hexdigest()[:16]


async def _embed_many(texts: list[str]) -> list[list[float]]:
    return await asyncio.gather(*[oll.embed(t) for t in texts])


# ── Store ────────────────────────────────────────────────────────────────────
async def store_knowledge(text: str, *, title: str, source_type: str,
                          identifier: str, extra: dict | None = None) -> dict:
    if not CFG.memory_enabled:
        return {"chunks_stored": 0, "skipped": "memory_disabled"}
    try:
        _, col, _ = _ensure()
    except Exception as e:
        return {"error": f"chroma: {e}", "chunks_stored": 0}

    sid = _source_id(source_type, identifier)
    chunks = _chunks(text)
    if not chunks:
        return {"chunks_stored": 0, "skipped": "empty"}

    try:
        existing = col.get(where={"source_id": sid})
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    try:
        embeddings = await _embed_many(chunks)
    except Exception as e:
        return {"error": f"embed: {e}", "chunks_stored": 0}

    now = datetime.now(timezone.utc).isoformat()
    ids = [f"{sid}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_id": sid,
            "source_type": source_type,
            "title": title[:200],
            "identifier": identifier[:500],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": now,
            **(extra or {}),
        }
        for i in range(len(chunks))
    ]
    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return {"source_id": sid, "chunks_stored": len(chunks),
            "chars_total": len(text), "title": title}


async def store_conversation_turn(conv_id: str, conv_name: str,
                                  user_msg: str, assistant_msg: str) -> None:
    if not CFG.memory_enabled:
        return
    try:
        _, _, col = _ensure()
    except Exception:
        return
    text = f"Conversation: {conv_name}\nUser: {user_msg}\nAssistant: {assistant_msg}"
    chunk_id = f"conv_{conv_id}_{int(time.time() * 1000)}"
    now = datetime.now(timezone.utc).isoformat()
    try:
        emb = await oll.embed(text)
        col.add(
            ids=[chunk_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[{
                "source_id": conv_id,
                "source_type": "conversation",
                "title": conv_name,
                "identifier": conv_id,
                "timestamp": now,
            }],
        )
    except Exception as e:
        log.warning("conv turn store failed: %s", e)


# ── Retrieve ─────────────────────────────────────────────────────────────────
async def retrieve(query: str, *, top_k: int | None = None,
                   include_conversations: bool = True) -> list[dict]:
    if not CFG.memory_enabled or not query.strip():
        return []
    top_k = top_k or CFG.memory_top_k
    try:
        _, k_col, c_col = _ensure()
    except Exception:
        return []

    try:
        q_emb = await oll.embed(query)
    except Exception:
        return []

    out: list[dict] = []

    def _ingest(qres, collection: str, source_override: str | None = None):
        for doc, meta, dist in zip(qres["documents"][0], qres["metadatas"][0], qres["distances"][0]):
            out.append({
                "text": doc,
                "title": meta.get("title", ""),
                "source_type": source_override or meta.get("source_type", ""),
                "identifier": meta.get("identifier", ""),
                "score": round(1 - dist, 4),
                "timestamp": meta.get("timestamp", ""),
                "collection": collection,
            })

    try:
        kc = k_col.count()
        if kc:
            kr = k_col.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, kc),
                include=["documents", "metadatas", "distances"],
            )
            _ingest(kr, "knowledge")
    except Exception as e:
        log.warning("knowledge query failed: %s", e)

    if include_conversations:
        try:
            cc = c_col.count()
            if cc:
                cr = c_col.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k, cc),
                    include=["documents", "metadatas", "distances"],
                )
                _ingest(cr, "conversations", source_override="conversation")
        except Exception as e:
            log.warning("conversation query failed: %s", e)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


def build_context_block(chunks: list[dict], *, max_chars: int = 3000) -> str:
    if not chunks:
        return ""
    lines = ["── RELEVANT MEMORY ──────────────────────────────────────────"]
    total = 0
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['source_type'].upper()}: {c['title'] or c['identifier']}"
        ts = (c.get("timestamp") or "")[:10]
        if ts:
            header += f" ({ts})"
        entry = f"{header}\n{c['text']}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    lines.append("─────────────────────────────────────────────────────────")
    return "\n\n".join(lines)


# ── Stats / admin ────────────────────────────────────────────────────────────
def get_stats() -> dict:
    if not CFG.memory_enabled:
        return {"available": False, "reason": "MEMORY_ENABLED=false"}
    try:
        _, k, c = _ensure()
        k_count = k.count()
        c_count = c.count()
        sources: dict[str, dict] = {}
        if k_count:
            meta = k.get(include=["metadatas"])
            for m in meta["metadatas"]:
                sid = m.get("source_id", "")
                if sid not in sources:
                    sources[sid] = {
                        "source_id": sid,
                        "title": m.get("title", ""),
                        "source_type": m.get("source_type", ""),
                        "identifier": m.get("identifier", ""),
                        "timestamp": m.get("timestamp", ""),
                    }
        return {
            "available": True,
            "knowledge_chunks": k_count,
            "conversation_chunks": c_count,
            "unique_sources": len(sources),
            "sources": sorted(sources.values(),
                              key=lambda s: s["timestamp"], reverse=True)[:50],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def list_sources(query: str = "", source_type: str = "",
                 limit: int = 200, offset: int = 0) -> dict:
    """All ingested sources with chunk count + total chars.

    Filtering: substring match against title or identifier (case-insensitive),
    optional source_type filter. Returns the full distinct source_type list
    for building filter UI.
    """
    if not CFG.memory_enabled:
        return {"available": False, "sources": [], "total": 0, "types": []}
    try:
        _, k, _ = _ensure()
        if not k.count():
            return {"available": True, "sources": [], "total": 0, "types": []}
        meta = k.get(include=["metadatas", "documents"])
        agg: dict[str, dict] = {}
        all_types: set[str] = set()
        for m, doc in zip(meta["metadatas"], meta["documents"]):
            sid = m.get("source_id", "")
            stype = m.get("source_type", "")
            if stype:
                all_types.add(stype)
            if not sid:
                continue
            if sid not in agg:
                agg[sid] = {
                    "source_id": sid,
                    "title": m.get("title", "") or m.get("identifier", ""),
                    "source_type": stype,
                    "identifier": m.get("identifier", ""),
                    "timestamp": m.get("timestamp", ""),
                    "chunk_count": 0,
                    "char_count": 0,
                }
            agg[sid]["chunk_count"] += 1
            agg[sid]["char_count"] += len(doc or "")
        items = list(agg.values())
        if source_type:
            items = [s for s in items if s["source_type"] == source_type]
        if query:
            ql = query.lower().strip()
            items = [s for s in items
                     if ql in (s["title"] or "").lower()
                     or ql in (s["identifier"] or "").lower()]
        items.sort(key=lambda s: s["timestamp"], reverse=True)
        return {
            "available": True,
            "total": len(items),
            "sources": items[offset:offset + limit],
            "types": sorted(all_types),
        }
    except Exception as e:
        return {"available": False, "error": str(e),
                "sources": [], "total": 0, "types": []}


def get_source(source_id: str) -> dict:
    """Full reassembled text + chunks for a single source."""
    if not CFG.memory_enabled:
        return {"available": False}
    try:
        _, k, _ = _ensure()
        res = k.get(where={"source_id": source_id},
                    include=["documents", "metadatas"])
        if not res.get("ids"):
            return {"available": True, "found": False}
        rows = sorted(
            zip(res["metadatas"], res["documents"]),
            key=lambda r: r[0].get("chunk_index", 0),
        )
        meta0 = rows[0][0]
        chunks = [
            {"index": m.get("chunk_index", i),
             "total": m.get("total_chunks", len(rows)),
             "text": d}
            for i, (m, d) in enumerate(rows)
        ]
        full_text = "\n\n".join(c["text"] for c in chunks)
        return {
            "available": True,
            "found": True,
            "source_id": source_id,
            "title": meta0.get("title", ""),
            "source_type": meta0.get("source_type", ""),
            "identifier": meta0.get("identifier", ""),
            "timestamp": meta0.get("timestamp", ""),
            "chunk_count": len(chunks),
            "char_count": sum(len(c["text"]) for c in chunks),
            "text": full_text,
            "chunks": chunks,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def delete_source(source_id: str) -> dict:
    try:
        _, col, _ = _ensure()
        existing = col.get(where={"source_id": source_id})
        if not existing.get("ids"):
            return {"deleted": 0, "source_id": source_id}
        col.delete(ids=existing["ids"])
        return {"deleted": len(existing["ids"]), "source_id": source_id}
    except Exception as e:
        return {"error": str(e), "source_id": source_id}

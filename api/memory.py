"""Postgres + pgvector RAG store.

Replaces the previous embedded ChromaDB implementation. Schema:

* ``knowledge_chunks`` — chunks ingested from URLs, files, distillations,
  search results, x posts. One row per chunk; vector + metadata + JSON
  ``extra`` blob.
* ``conversation_chunks`` — chunks built from conversation turns, used both
  for conversational memory retrieval and as the queue feeding knowledge
  distillation. Carries feedback and distill state.

Public API mirrors the ChromaDB version; DB-touching functions are async.
``init()`` runs schema migration on startup; ``wipe()`` truncates both
tables for the rebuild flow.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import ollama as oll
from config import CFG

log = logging.getLogger("llm-api.memory")


# ── Connection pool / schema ────────────────────────────────────────────────
_pool: Any = None
_ready: bool = False


async def _configure_conn(conn) -> None:
    # Ensure the vector extension exists *before* registering the type
    # adapter — register_vector_async() looks up the vector OID in the
    # catalog and fails if the extension isn't loaded yet. CREATE EXTENSION
    # IF NOT EXISTS is idempotent so per-connection cost is one no-op
    # catalog lookup after the first.
    async with conn.cursor() as cur:
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.commit()
    from pgvector.psycopg import register_vector_async
    await register_vector_async(conn)


async def init() -> None:
    """Open the connection pool and run schema migrations.

    Called from FastAPI lifespan. Creates the vector extension, tables and
    HNSW indexes if they don't exist. Safe to call repeatedly. Failures
    are logged and leave ``is_available()`` returning False so the rest
    of the app degrades gracefully.
    """
    global _pool, _ready
    if not CFG.memory_enabled:
        log.info("memory disabled (MEMORY_ENABLED=false)")
        return
    if _pool is not None:
        return
    try:
        from psycopg_pool import AsyncConnectionPool
    except Exception as e:
        log.error("psycopg_pool import failed: %s", e)
        return

    pool = AsyncConnectionPool(
        CFG.database_url,
        min_size=CFG.db_pool_min,
        max_size=CFG.db_pool_max,
        open=False,
        configure=_configure_conn,
        timeout=30.0,
    )
    # Retry pool open: db container may still be coming up.
    last_err: Exception | None = None
    for attempt in range(20):
        try:
            await pool.open(wait=True, timeout=10.0)
            last_err = None
            break
        except Exception as e:
            last_err = e
            await asyncio.sleep(2.0)
    if last_err is not None:
        log.error("postgres pool open failed after retries: %s", last_err)
        return

    _pool = pool
    try:
        async with _pool.connection() as conn:
            await _migrate(conn)
            # The asyncio task that drives a crawl can't survive a process
            # restart — mark any stranded 'running' rows so the SPA polling
            # loop sees an honest terminal status instead of a stale one.
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE crawl_jobs
                       SET status = 'interrupted',
                           finished_at = COALESCE(finished_at, NOW()),
                           errors = errors || jsonb_build_array(
                             'job interrupted by api restart at ' || NOW()::text
                           )
                     WHERE status = 'running'
                    """
                )
                interrupted = cur.rowcount or 0
            await conn.commit()
            if interrupted:
                log.warning("marked %d stranded crawl_jobs as 'interrupted'",
                            interrupted)
        _ready = True
        log.info("postgres memory: ready (dim=%d, db=%s)",
                 CFG.embed_dim, _safe_db_label())
    except Exception as e:
        log.error("postgres migration failed: %s", e)
        _ready = False


def _safe_db_label() -> str:
    # Strip credentials from the URL for log lines.
    try:
        return re.sub(r"://[^@]*@", "://***@", CFG.database_url)
    except Exception:
        return "postgres"


async def _migrate(conn) -> None:
    dim = CFG.embed_dim
    async with conn.cursor() as cur:
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await cur.execute(f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id            BIGSERIAL PRIMARY KEY,
                chunk_id      TEXT NOT NULL UNIQUE,
                source_id     TEXT NOT NULL,
                source_type   TEXT NOT NULL,
                title         TEXT NOT NULL,
                identifier    TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                total_chunks  INTEGER NOT NULL,
                text          TEXT NOT NULL,
                embedding     vector({dim}),
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                tsv           tsvector GENERATED ALWAYS AS (
                                  setweight(to_tsvector('english',
                                              coalesce(title,'')), 'A') ||
                                  setweight(to_tsvector('english',
                                              coalesce(text,'')),  'B')
                              ) STORED
            )
        """)
        # Migrate older databases whose `tsv` was indexed on body-only. The
        # generated expression can't be ALTERed in place, so detect the old
        # form via pg_get_expr and recreate the column. Title-bearing tsv
        # makes literal-title queries (e.g. pasting an article headline)
        # find the right source via BM25 instead of riding incidental body
        # keywords (cheese-monkey case).
        await cur.execute("""
            SELECT pg_get_expr(d.adbin, d.adrelid)
              FROM pg_attribute a
              JOIN pg_attrdef d
                ON d.adrelid = a.attrelid AND d.adnum = a.attnum
             WHERE a.attrelid = 'knowledge_chunks'::regclass
               AND a.attname  = 'tsv'
        """)
        row = await cur.fetchone()
        existing_expr = (row[0] or "") if row else ""
        if row and "title" not in existing_expr.lower():
            log.info("migrating knowledge_chunks.tsv to title+text weighted index")
            await cur.execute("DROP INDEX IF EXISTS idx_knowledge_tsv")
            await cur.execute("ALTER TABLE knowledge_chunks DROP COLUMN tsv")
        await cur.execute("""
            ALTER TABLE knowledge_chunks
              ADD COLUMN IF NOT EXISTS tsv tsvector
              GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
                  setweight(to_tsvector('english', coalesce(text,'')),  'B')
              ) STORED
        """)
        await cur.execute(f"""
            CREATE TABLE IF NOT EXISTS conversation_chunks (
                id              BIGSERIAL PRIMARY KEY,
                chunk_id        TEXT NOT NULL UNIQUE,
                source_id       TEXT NOT NULL,
                title           TEXT NOT NULL,
                identifier      TEXT NOT NULL,
                text            TEXT NOT NULL,
                embedding       vector({dim}),
                timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                message_id      TEXT,
                feedback_rating TEXT,
                feedback_at     TIMESTAMPTZ,
                distilled       BOOLEAN NOT NULL DEFAULT FALSE,
                distilled_at    TIMESTAMPTZ,
                metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
        """)
        # HNSW for cosine distance — pgvector supports vector_cosine_ops.
        await cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
              ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
        """)
        await cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_embedding
              ON conversation_chunks USING hnsw (embedding vector_cosine_ops)
        """)
        # Lookup-by-id indexes used by retrieval and admin queries.
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_chunks (source_id)")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_identifier ON knowledge_chunks (identifier)")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_metadata ON knowledge_chunks USING gin (metadata jsonb_path_ops)")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tsv ON knowledge_chunks USING gin (tsv)")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_message_id ON conversation_chunks (message_id) WHERE message_id IS NOT NULL")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_feedback ON conversation_chunks (feedback_rating) WHERE feedback_rating IS NOT NULL")
        # Crawl jobs survive uvicorn --reload and container restarts; the
        # in-memory _JOBS dict in crawler.py used to lose them, which made
        # the SPA polling loop 404 mid-crawl. Status flips to 'interrupted'
        # on startup for any row left in 'running' state, since the
        # asyncio task driving it didn't survive the process restart.
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS crawl_jobs (
                job_id          TEXT PRIMARY KEY,
                seed_url        TEXT NOT NULL,
                status          TEXT NOT NULL,
                strategy        TEXT NOT NULL,
                max_pages       INTEGER NOT NULL,
                max_depth       INTEGER NOT NULL,
                chunks_stored   INTEGER NOT NULL DEFAULT 0,
                pages_ingested  INTEGER NOT NULL DEFAULT 0,
                pages_skipped   INTEGER NOT NULL DEFAULT 0,
                resumed_from    INTEGER NOT NULL DEFAULT 0,
                pages           JSONB NOT NULL DEFAULT '[]'::jsonb,
                errors          JSONB NOT NULL DEFAULT '[]'::jsonb,
                started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at     TIMESTAMPTZ
            )
        """)
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_jobs_status ON crawl_jobs (status)")
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_jobs_started ON crawl_jobs (started_at DESC)")
    await conn.commit()


async def close() -> None:
    global _pool, _ready
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
    _pool = None
    _ready = False


def is_available() -> bool:
    return CFG.memory_enabled and _ready and _pool is not None


# ── Chunking + IDs ──────────────────────────────────────────────────────────
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


_EMBED_SEM = asyncio.Semaphore(4)


async def _embed_one(text: str) -> list[float]:
    async with _EMBED_SEM:
        return await oll.embed(text)


async def _embed_many(texts: list[str]) -> list[list[float]]:
    return await asyncio.gather(*[_embed_one(t) for t in texts])


# ── Store ────────────────────────────────────────────────────────────────────
async def store_knowledge(text: str, *, title: str, source_type: str,
                          identifier: str, extra: dict | None = None) -> dict:
    if not CFG.memory_enabled:
        return {"chunks_stored": 0, "skipped": "memory_disabled"}
    if not is_available():
        return {"error": "memory not ready", "chunks_stored": 0}

    sid = _source_id(source_type, identifier)
    chunks = _chunks(text)
    if not chunks:
        return {"chunks_stored": 0, "skipped": "empty"}

    try:
        embeddings = await _embed_many(chunks)
    except Exception as e:
        return {"error": f"embed: {e}", "chunks_stored": 0}

    now = datetime.now(timezone.utc)
    title_short = title[:200]
    ident_short = identifier[:500]
    meta_json = json.dumps(extra or {})

    rows = [
        (
            f"{sid}_{i}", sid, source_type, title_short, ident_short,
            i, len(chunks), chunks[i], embeddings[i], now, meta_json,
        )
        for i in range(len(chunks))
    ]

    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                # Delete previous version of this source so re-ingest is
                # idempotent (same identifier+source_type → same source_id).
                await cur.execute(
                    "DELETE FROM knowledge_chunks WHERE source_id = %s",
                    (sid,),
                )
                await cur.executemany(
                    """
                    INSERT INTO knowledge_chunks
                      (chunk_id, source_id, source_type, title, identifier,
                       chunk_index, total_chunks, text, embedding,
                       timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    rows,
                )
            await conn.commit()
    except Exception as e:
        log.warning("store_knowledge failed: %s", e)
        return {"error": str(e), "chunks_stored": 0}

    return {
        "source_id": sid, "chunks_stored": len(chunks),
        "chars_total": len(text), "title": title_short,
    }


async def store_conversation_turn(conv_id: str, conv_name: str,
                                  user_msg: str, assistant_msg: str,
                                  message_id: str | None = None) -> None:
    if not CFG.memory_enabled or not is_available():
        return
    text = f"Conversation: {conv_name}\nUser: {user_msg}\nAssistant: {assistant_msg}"
    chunk_id = f"conv_{conv_id}_{int(time.time() * 1000)}"
    try:
        emb = await _embed_one(text)
    except Exception as e:
        log.warning("conv turn embed failed: %s", e)
        return
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO conversation_chunks
                      (chunk_id, source_id, title, identifier, text,
                       embedding, timestamp, message_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb)
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    (chunk_id, conv_id, conv_name[:200], conv_id, text,
                     emb, datetime.now(timezone.utc), message_id),
                )
            await conn.commit()
    except Exception as e:
        log.warning("conv turn store failed: %s", e)


# ── Feedback ─────────────────────────────────────────────────────────────────
async def update_feedback(message_id: str, rating: str | None) -> int:
    """Stamp `feedback_rating` on every conversation chunk tagged with this
    message_id. ``rating`` in {'up','down', None}. Returns updated count."""
    if not CFG.memory_enabled or not message_id or not is_available():
        return 0
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                if rating is None:
                    await cur.execute(
                        """
                        UPDATE conversation_chunks
                           SET feedback_rating = NULL,
                               feedback_at     = NULL
                         WHERE message_id = %s
                        """,
                        (message_id,),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE conversation_chunks
                           SET feedback_rating = %s,
                               feedback_at     = NOW()
                         WHERE message_id = %s
                        """,
                        (rating, message_id),
                    )
                count = cur.rowcount or 0
            await conn.commit()
            return int(count)
    except Exception as e:
        log.warning("update_feedback failed: %s", e)
        return 0


async def pending_distill_chunks(limit: int = 20) -> list[dict]:
    """Up-rated conversation chunks that haven't been distilled yet."""
    if not CFG.memory_enabled or not is_available():
        return []
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT chunk_id, message_id, title, text, timestamp
                      FROM conversation_chunks
                     WHERE feedback_rating = 'up'
                       AND distilled = FALSE
                     ORDER BY timestamp ASC
                     LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
    except Exception as e:
        log.warning("pending_distill_chunks failed: %s", e)
        return []
    out = []
    for chunk_id, msg_id, title, text, ts in rows:
        out.append({
            "chunk_id": chunk_id,
            "message_id": msg_id,
            "title": title or "",
            "document": text or "",
            "timestamp": ts.isoformat() if ts else "",
        })
    return out


async def mark_distilled(chunk_ids: list[str]) -> int:
    if not CFG.memory_enabled or not chunk_ids or not is_available():
        return 0
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE conversation_chunks
                       SET distilled = TRUE,
                           distilled_at = NOW()
                     WHERE chunk_id = ANY(%s)
                    """,
                    (list(chunk_ids),),
                )
                count = cur.rowcount or 0
            await conn.commit()
            return int(count)
    except Exception as e:
        log.warning("mark_distilled failed: %s", e)
        return 0


async def pending_distill_count() -> int:
    if not CFG.memory_enabled or not is_available():
        return 0
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM conversation_chunks
                     WHERE feedback_rating = 'up' AND distilled = FALSE
                    """
                )
                row = await cur.fetchone()
                return int(row[0]) if row else 0
    except Exception:
        return 0


async def feedback_stats() -> dict:
    if not CFG.memory_enabled:
        return {"available": False, "up": 0, "down": 0, "total_rated": 0}
    if not is_available():
        return {"available": False, "error": "memory not ready"}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                # Distinct (message_id, rating) pairs — match the per-message
                # semantics the old ChromaDB version reported.
                await cur.execute(
                    """
                    SELECT feedback_rating, COUNT(DISTINCT message_id)
                      FROM conversation_chunks
                     WHERE feedback_rating IS NOT NULL
                       AND message_id IS NOT NULL
                     GROUP BY feedback_rating
                    """
                )
                rows = await cur.fetchall()
    except Exception as e:
        return {"available": False, "error": str(e)}
    counts = {r[0]: int(r[1]) for r in rows}
    up = counts.get("up", 0)
    down = counts.get("down", 0)
    return {"available": True, "up": up, "down": down, "total_rated": up + down}


# ── Pure helpers (text-side; preserved from previous implementation) ───────
_MD_LINK_RE = re.compile(r"\[[^\]\n]{0,200}\]\([^)\s]+\)")
_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_WS_RE = re.compile(r"\s+")
_DEDUPE_MIN_LEN = 30

_TOKEN_RE = re.compile(r"[a-z][a-z0-9'\-]+")
_QUERY_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_QUERY_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "be", "been", "being", "for", "with", "on", "at", "by", "from", "as", "this",
    "that", "these", "those", "it", "its", "but", "so", "if", "then", "than",
    "into", "about", "out", "up", "down", "over", "under",
    "what", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "can", "could", "would", "should", "may", "might", "will", "shall",
    "i", "me", "my", "mine", "we", "our", "us", "ours", "you", "your", "yours",
    "they", "them", "their", "theirs", "he", "him", "his", "she", "her", "hers",
    "library", "libraries", "files", "file", "saved", "ingested", "stored",
    "memory", "memories", "knowledge", "page", "pages", "doc", "docs",
    "document", "documents", "article", "articles", "site", "sites", "url",
    "between", "connect", "connection", "regarding", "say", "says", "said",
    "tell", "tells", "told", "show", "shows", "showed", "find", "finds",
    "found", "look", "looks", "looked", "search", "searches",
    "yes", "no", "ok", "please", "thanks",
    "report", "reports", "reported",
})

_URL_NORM_RE = re.compile(r"^https?://(www\.)?", re.IGNORECASE)


def _para_key(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()[:300]


def _dedupe_paragraphs(text: str, *, seen: set[str] | None = None) -> str:
    if not text:
        return text
    if seen is None:
        seen = set()
    out: list[str] = []
    for para in _PARA_SPLIT_RE.split(text):
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) >= _DEDUPE_MIN_LEN:
            key = _para_key(stripped)
            if key in seen:
                continue
            seen.add(key)
        out.append(stripped)
    return "\n\n".join(out)


def _query_terms(query: str, *, max_terms: int = 6) -> list[str]:
    if not query:
        return []
    cleaned = _QUERY_URL_RE.sub(" ", query)
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(cleaned.lower()):
        if len(tok) < 3 or tok in _QUERY_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def _term_hits(text: str, terms: list[str]) -> set[str]:
    if not text or not terms:
        return set()
    low = text.lower()
    return {t for t in terms if t in low}


def _keyword_multiplier(text: str, terms: list[str]) -> float:
    if len(terms) < 2:
        return 1.0
    hit = _term_hits(text, terms)
    ratio = len(hit) / len(terms)
    if ratio == 1.0:
        return 1.4
    return round(0.5 + 0.4 * ratio, 4)


def _navigation_density(text: str) -> float:
    if not text:
        return 1.0
    text_len = len(text)
    link_chars = sum(len(m.group(0)) for m in _MD_LINK_RE.finditer(text))
    link_density = link_chars / max(text_len, 1)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 1.0
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 60)
    short_ratio = short_lines / len(lines)
    bullet_lines = sum(1 for ln in lines if ln.lstrip().startswith(("*", "-", "•")))
    bullet_ratio = bullet_lines / len(lines)
    return max(link_density, bullet_ratio * 0.9, short_ratio * 0.6)


def _navigation_multiplier(text: str) -> float:
    nav = _navigation_density(text)
    if nav >= 0.6:
        return 0.35
    if nav >= 0.4:
        return 0.6
    if nav >= 0.25:
        return 0.85
    return 1.0


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    return _URL_NORM_RE.sub("", url.strip()).rstrip("/").lower()


# ── Retrieve ─────────────────────────────────────────────────────────────────
def _score_with_feedback(distance: float, rating: str | None) -> tuple[float, float]:
    """(adjusted_score, base_score) given cosine distance and feedback."""
    base = round(1 - float(distance), 4)
    if rating == "up":
        return round(base * 1.3, 4), base
    if rating == "down":
        return round(base * 0.3, 4), base
    return base, base


_RRF_K = 60  # standard RRF smoothing constant
_BM25_ONLY_BASE_SCORE = 0.55  # synthesized score for BM25-only hits (no cosine)


async def _hybrid_candidate_rows(query_text: str, q_emb: list[float],
                                 candidate_n: int) -> list[tuple]:
    """Return candidate knowledge_chunks rows ranked by RRF of vector + BM25.

    Catches exact-name and rare-token queries that pure cosine misses
    ("Freemason" in a Trump-heavy library) while keeping vector's good
    behavior on paraphrase-style queries. Each returned row is shaped
    ``(source_id, source_type, title, identifier, timestamp, chunk_index,
       total_chunks, text, score, in_vector, in_bm25)``.

    BM25-only rows get a synthesized ``score`` of ``_BM25_ONLY_BASE_SCORE``
    so downstream sorting/threshold logic treats them as moderate-quality
    hits; chunks present in both lists keep their cosine-derived score.
    """
    cn = max(candidate_n, 1)
    # plainto_tsquery joins terms with AND, which fails the moment one term
    # is missing — too strict for free-form questions. Build an OR-tsquery
    # from the cleaned content terms instead so the BM25 path acts like
    # "any of these matter" and RRF/ts_rank_cd handle the ordering.
    bm25_terms = _query_terms(query_text)
    bm25_query_str = " | ".join(bm25_terms) if bm25_terms else ""

    async with _pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT chunk_id, source_id, source_type, title, identifier,
                       timestamp, chunk_index, total_chunks, text,
                       embedding <=> %s::vector AS distance
                  FROM knowledge_chunks
                 WHERE chunk_index >= 0
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                (q_emb, q_emb, cn),
            )
            v_rows = await cur.fetchall()
            b_rows: list = []
            if bm25_query_str:
                try:
                    await cur.execute(
                        """
                        SELECT chunk_id, source_id, source_type, title, identifier,
                               timestamp, chunk_index, total_chunks, text,
                               ts_rank_cd(tsv, to_tsquery('english', %s)) AS bm25
                          FROM knowledge_chunks
                         WHERE chunk_index >= 0
                           AND tsv @@ to_tsquery('english', %s)
                         ORDER BY bm25 DESC
                         LIMIT %s
                        """,
                        (bm25_query_str, bm25_query_str, cn),
                    )
                    b_rows = await cur.fetchall()
                except Exception as e:
                    log.warning("BM25 candidate query failed (q=%r): %s",
                                bm25_query_str, e)

    # RRF merge by chunk_id.
    v_rank = {r[0]: i + 1 for i, r in enumerate(v_rows)}
    b_rank = {r[0]: i + 1 for i, r in enumerate(b_rows)}
    rows_by_id: dict[str, tuple] = {r[0]: r for r in v_rows}
    for r in b_rows:
        rows_by_id.setdefault(r[0], r)

    rrf: dict[str, float] = {}
    for cid in rows_by_id:
        s = 0.0
        if cid in v_rank:
            s += 1.0 / (_RRF_K + v_rank[cid])
        if cid in b_rank:
            s += 1.0 / (_RRF_K + b_rank[cid])
        rrf[cid] = s

    out: list[tuple] = []
    for cid in sorted(rows_by_id, key=lambda k: rrf[k], reverse=True)[:cn]:
        r = rows_by_id[cid]
        # Last column differs: vector→distance, bm25→bm25 score. Compute a
        # unified per-chunk score using cosine when available.
        if cid in v_rank:
            # r[9] is distance (float)
            score, _ = _score_with_feedback(r[9], None)
        else:
            score = _BM25_ONLY_BASE_SCORE
        # Tuple shape: (source_id, source_type, title, identifier, timestamp,
        #               chunk_index, total_chunks, text, score, in_vector, in_bm25)
        out.append((
            r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
            score, cid in v_rank, cid in b_rank,
        ))
    return out


async def retrieve(query: str, *, top_k: int | None = None,
                   include_conversations: bool = True) -> list[dict]:
    if not CFG.memory_enabled or not query.strip() or not is_available():
        return []
    top_k = top_k or CFG.memory_top_k
    try:
        q_emb = await _embed_one(query)
    except Exception:
        return []

    out: list[dict] = []
    try:
        # Knowledge: hybrid (vector + BM25 via RRF).
        rows = await _hybrid_candidate_rows(query, q_emb, top_k)
        for sid, stype, title, ident, ts, _, _, text, score, in_v, in_b in rows:
            out.append({
                "text": text or "",
                "title": title or "",
                "source_type": stype or "",
                "identifier": ident or "",
                "score": score,
                "base_score": score,
                "feedback_rating": None,
                "timestamp": ts.isoformat() if ts else "",
                "collection": "knowledge",
                "matched_via": "hybrid" if (in_v and in_b) else ("vector" if in_v else "bm25"),
            })
        if include_conversations:
            async with _pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT text, title, identifier, timestamp,
                               feedback_rating,
                               embedding <=> %s::vector AS distance
                          FROM conversation_chunks
                         ORDER BY embedding <=> %s::vector
                         LIMIT %s
                        """,
                        (q_emb, q_emb, top_k),
                    )
                    for text, title, ident, ts, rating, dist in await cur.fetchall():
                        score, base = _score_with_feedback(dist, rating)
                        out.append({
                            "text": text or "",
                            "title": title or "",
                            "source_type": "conversation",
                            "identifier": ident or "",
                            "score": score,
                            "base_score": base,
                            "feedback_rating": rating,
                            "timestamp": ts.isoformat() if ts else "",
                            "collection": "conversations",
                        })
    except Exception as e:
        log.warning("retrieve failed: %s", e)
        return []

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


async def retrieve_deep(query: str, *, top_sources: int = 6,
                        max_chars: int = 9000,
                        candidate_chunks: int = 80,
                        neighbor_radius: int = 1) -> list[dict]:
    """Source-aware retrieval. Returns one bundle per top source, each with
    its matched chunks plus neighbors reassembled in document order. Solves
    the "12 disjoint fragments" problem on large libraries — instead of
    many chunks from many sources, the model sees a handful of sources
    with continuous, in-document context.
    """
    if not CFG.memory_enabled or not query.strip() or not is_available():
        return []
    try:
        q_emb = await _embed_one(query)
    except Exception:
        return []

    try:
        rows = await _hybrid_candidate_rows(query, q_emb, candidate_chunks)
    except Exception as e:
        log.warning("retrieve_deep candidate query failed: %s", e)
        return []

    if not rows:
        return []

    by_source: dict[str, dict] = {}
    for sid, stype, title, ident, ts, idx, total, text, score, _, _ in rows:
        bucket = by_source.setdefault(sid, {
            "source_id": sid,
            "title": title or "",
            "identifier": ident or "",
            "source_type": stype or "",
            "timestamp": ts.isoformat() if ts else "",
            "best_score": score,
            "matched": [],
        })
        bucket["best_score"] = max(bucket["best_score"], score)
        bucket["matched"].append({
            "index": int(idx or 0),
            "total": int(total or 0),
            "text": text or "",
            "score": score,
        })

    terms = _query_terms(query)
    for s in by_source.values():
        sample = "\n".join(m["text"] for m in s["matched"])
        nav_mult = _navigation_multiplier(sample)
        kw_mult = _keyword_multiplier(sample, terms)
        s["nav_multiplier"] = nav_mult
        s["keyword_multiplier"] = kw_mult
        s["adjusted_score"] = round(s["best_score"] * nav_mult * kw_mult, 4)

    sources = sorted(by_source.values(),
                     key=lambda s: s["adjusted_score"], reverse=True)[:top_sources]
    if not sources:
        return []

    # Bulk-fetch source-level refinement metadata (summary/tags) so the
    # bundle renderer can prepend a GIST/TAGS preamble per [L#].
    source_meta: dict[str, dict] = {}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source_id, metadata
                      FROM knowledge_chunks
                     WHERE source_id = ANY(%s)
                       AND chunk_index = 0
                    """,
                    ([s["source_id"] for s in sources],),
                )
                for sid, meta in await cur.fetchall():
                    source_meta[sid] = dict(meta or {})
    except Exception as e:
        log.debug("source metadata fetch failed: %s", e)

    per_source_chars = max(1200, max_chars // max(len(sources), 1))
    bundles: list[dict] = []
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                for s in sources:
                    matched_idx = sorted({m["index"] for m in s["matched"]})
                    total = s["matched"][0]["total"]
                    wanted: set[int] = set()
                    for idx in matched_idx:
                        for delta in range(-neighbor_radius, neighbor_radius + 1):
                            nidx = idx + delta
                            if nidx < 0:
                                continue
                            if total and nidx >= total:
                                continue
                            wanted.add(nidx)

                    have = {m["index"]: m["text"] for m in s["matched"]}
                    missing = sorted(i for i in wanted if i not in have)
                    if missing:
                        await cur.execute(
                            """
                            SELECT chunk_index, text
                              FROM knowledge_chunks
                             WHERE source_id = %s
                               AND chunk_index = ANY(%s)
                            """,
                            (s["source_id"], missing),
                        )
                        for nidx, text in await cur.fetchall():
                            have[int(nidx)] = text or ""

                    ordered = sorted(have.items())
                    parts: list[str] = []
                    char_count = 0
                    prev_idx: int | None = None
                    for idx, txt in ordered:
                        if char_count + len(txt) > per_source_chars and parts:
                            parts.append("[…]")
                            break
                        if prev_idx is not None and idx != prev_idx + 1:
                            parts.append("[…]")
                            char_count += 5
                        parts.append(txt)
                        char_count += len(txt)
                        prev_idx = idx

                    meta = source_meta.get(s["source_id"], {})
                    bundles.append({
                        "text": _dedupe_paragraphs("\n\n".join(parts)),
                        "title": s["title"],
                        "source_type": s["source_type"] or "knowledge",
                        "identifier": s["identifier"],
                        "score": s.get("adjusted_score", s["best_score"]),
                        "base_score": s["best_score"],
                        "nav_multiplier": s.get("nav_multiplier", 1.0),
                        "keyword_multiplier": s.get("keyword_multiplier", 1.0),
                        "feedback_rating": None,
                        "timestamp": s["timestamp"],
                        "collection": "knowledge",
                        "matched_chunks": len(matched_idx),
                        "rendered_chunks": len(parts),
                        "summary": meta.get("summary"),
                        "tags": meta.get("tags"),
                    })
    except Exception as e:
        log.warning("retrieve_deep neighbor fetch failed: %s", e)

    return bundles


async def lookup_by_title(phrase: str, *, max_chars: int = 6000,
                          limit: int = 1) -> list[dict]:
    """Find sources whose ``title`` contains ``phrase`` (case-insensitive).

    Returned bundles have the same shape as ``retrieve``/``lookup_by_identifier``
    so they can be prepended to a chunks list and rendered by
    ``build_context_block``. Used to short-circuit retrieval when the user
    pastes a literal article title — vector/BM25 alone often misses titles
    because the BM25 ``tsv`` is built from ``text`` only, and a single keyword
    in the title (e.g. "cheese") can pull unrelated sources to the top.
    """
    if not CFG.memory_enabled or not phrase or not is_available():
        return []
    phrase = phrase.strip()
    if len(phrase) < 8:
        return []
    out: list[dict] = []
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source_id,
                           MIN(title)       AS title,
                           MIN(source_type) AS source_type,
                           MIN(identifier)  AS identifier,
                           MAX(timestamp)   AS ts
                      FROM knowledge_chunks
                     WHERE title ILIKE %s AND chunk_index >= 0
                     GROUP BY source_id
                     ORDER BY MAX(timestamp) DESC
                     LIMIT %s
                    """,
                    (f"%{phrase}%", limit),
                )
                source_rows = await cur.fetchall()
                if not source_rows:
                    return []
                for sid, title, stype, ident, ts in source_rows:
                    await cur.execute(
                        """
                        SELECT chunk_index, text
                          FROM knowledge_chunks
                         WHERE source_id = %s AND chunk_index >= 0
                         ORDER BY chunk_index ASC
                        """,
                        (sid,),
                    )
                    rows = await cur.fetchall()
                    if not rows:
                        continue
                    full_text = "\n\n".join((r[1] or "") for r in rows)
                    deduped = _dedupe_paragraphs(full_text)
                    if len(deduped) > max_chars:
                        deduped = deduped[:max_chars].rstrip() + "\n[…]"
                    out.append({
                        "text": deduped,
                        "title": title or "",
                        "source_type": stype or "knowledge",
                        "identifier": ident or "",
                        "score": 1.0,
                        "base_score": 1.0,
                        "feedback_rating": None,
                        "timestamp": ts.isoformat() if ts else "",
                        "collection": "knowledge",
                        "matched_chunks": len(rows),
                        "rendered_chunks": len(rows),
                        "exact_match": True,
                        "matched_via": "title",
                    })
    except Exception as e:
        log.warning("lookup_by_title failed: %s", e)
        return []
    return out


async def lookup_by_identifier(identifier: str, *,
                               max_chars: int = 6000) -> dict | None:
    """Exact (then normalized) lookup of one knowledge source by URL/filename."""
    if not CFG.memory_enabled or not identifier or not is_available():
        return None
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source_id, source_type, title, identifier,
                           timestamp, chunk_index, text
                      FROM knowledge_chunks
                     WHERE identifier = %s
                     ORDER BY chunk_index ASC
                    """,
                    (identifier,),
                )
                rows = await cur.fetchall()
                if not rows:
                    target = _normalize_url(identifier)
                    if not target:
                        return None
                    # Normalized URL fallback: scan distinct identifiers,
                    # find one whose normalized form matches.
                    await cur.execute(
                        "SELECT DISTINCT identifier FROM knowledge_chunks"
                    )
                    cand: str | None = None
                    for (ident,) in await cur.fetchall():
                        if _normalize_url(ident or "") == target:
                            cand = ident
                            break
                    if cand is None:
                        return None
                    await cur.execute(
                        """
                        SELECT source_id, source_type, title, identifier,
                               timestamp, chunk_index, text
                          FROM knowledge_chunks
                         WHERE identifier = %s
                         ORDER BY chunk_index ASC
                        """,
                        (cand,),
                    )
                    rows = await cur.fetchall()
                    if not rows:
                        return None
    except Exception as e:
        log.warning("lookup_by_identifier failed: %s", e)
        return None

    sid0, stype0, title0, ident0, ts0, _, _ = rows[0]
    full_text = "\n\n".join(r[6] or "" for r in rows)
    deduped = _dedupe_paragraphs(full_text)
    if len(deduped) > max_chars:
        deduped = deduped[:max_chars].rstrip() + "\n[…]"
    return {
        "text": deduped,
        "title": title0 or "",
        "source_type": stype0 or "knowledge",
        "identifier": ident0 or identifier,
        "score": 1.0,
        "base_score": 1.0,
        "feedback_rating": None,
        "timestamp": ts0.isoformat() if ts0 else "",
        "collection": "knowledge",
        "matched_chunks": len(rows),
        "rendered_chunks": len(rows),
        "exact_match": True,
    }


# ── Prompt block rendering ──────────────────────────────────────────────────
def build_context_block(chunks: list[dict], *, max_chars: int = 3000,
                        cite_prefix: str = "L",
                        query: str | None = None) -> str:
    """Render retrieved chunks as a prompt block with bracket cite markers.

    See module docstring for the dedup/coverage contract that the prompt
    side relies on. Pure function — no DB access.
    """
    if not chunks:
        return ""

    terms = _query_terms(query or "") if query else []
    coverage_section = ""
    if len(terms) >= 2:
        all_terms: list[str] = []
        partial: dict[str, list[str]] = {t: [] for t in terms}
        none: list[str] = []
        for i, c in enumerate(chunks, 1):
            marker = f"[{cite_prefix}{i}]"
            hit = _term_hits(c.get("text", "") or "", terms)
            if len(hit) == len(terms):
                all_terms.append(marker)
            elif len(hit) == 1:
                partial[next(iter(hit))].append(marker)
            elif not hit:
                none.append(marker)
            else:
                for t in hit:
                    partial[t].append(marker)
        bullets = [f"QUERY TOPICS: {', '.join(terms)}"]
        if all_terms:
            bullets.append(f"  • All topics present in: {', '.join(all_terms)}")
        else:
            bullets.append(
                "  • All topics present in: NONE — no single block in this "
                "library covers all the topics in the query"
            )
        for t in terms:
            if partial[t]:
                bullets.append(f"  • '{t}' only in: {', '.join(partial[t])}")
        if none:
            bullets.append(f"  • No query topics in: {', '.join(none)}")
        coverage_section = "\n".join(bullets)

    lines = ["── RELEVANT MEMORY ──────────────────────────────────────────"]
    if coverage_section:
        lines.append(coverage_section)
    total = 0
    seen: set[str] = set()
    for i, c in enumerate(chunks, 1):
        text = _dedupe_paragraphs(c.get("text", "") or "", seen=seen)
        if not text.strip() and not (c.get("summary") or c.get("tags")):
            continue
        marker = f"[{cite_prefix}{i}]"
        title = c.get("title") or c.get("identifier") or ""
        header_bits = [marker, c["source_type"].upper() + ":", title]
        ts = (c.get("timestamp") or "")[:10]
        if ts:
            header_bits.append(f"({ts})")
        ident = c.get("identifier") or ""
        header = " ".join(b for b in header_bits if b)
        if ident and ident != title:
            header += f"\nSOURCE: {ident}"
        # Per-source LLM-refined preamble (summary + tags), if present.
        preamble_bits: list[str] = []
        summary = c.get("summary") or {}
        if isinstance(summary, dict):
            tl_dr = (summary.get("tl_dr") or "").strip()
            if tl_dr:
                preamble_bits.append(f"GIST: {tl_dr}")
            claims = summary.get("key_claims") or []
            if isinstance(claims, list) and claims:
                claim_lines = "\n".join(f"  - {str(k).strip()}" for k in claims[:6])
                preamble_bits.append(f"KEY CLAIMS:\n{claim_lines}")
        tags = c.get("tags") or {}
        if isinstance(tags, dict):
            tag_parts = []
            for cat in ("people", "orgs", "places", "events", "topics"):
                vals = tags.get(cat) or []
                if isinstance(vals, list) and vals:
                    tag_parts.append(f"{cat}={', '.join(str(v) for v in vals[:8])}")
            if tag_parts:
                preamble_bits.append("TAGS: " + "; ".join(tag_parts))
        preamble = ("\n".join(preamble_bits) + "\n") if preamble_bits else ""
        entry = f"{header}\n{preamble}{text}".rstrip()
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    lines.append("─────────────────────────────────────────────────────────")
    return "\n\n".join(lines)


# ── Stats / admin ────────────────────────────────────────────────────────────
async def get_stats() -> dict:
    if not CFG.memory_enabled:
        return {"available": False, "reason": "MEMORY_ENABLED=false"}
    if not is_available():
        return {"available": False, "reason": "memory not ready"}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM knowledge_chunks")
                k_count = int((await cur.fetchone())[0])
                await cur.execute("SELECT COUNT(*) FROM conversation_chunks")
                c_count = int((await cur.fetchone())[0])
                await cur.execute(
                    """
                    SELECT DISTINCT ON (source_id)
                           source_id, title, source_type, identifier, timestamp
                      FROM knowledge_chunks
                     ORDER BY source_id, timestamp DESC
                    """
                )
                rows = await cur.fetchall()
        sources = [
            {"source_id": sid, "title": title or "",
             "source_type": stype or "", "identifier": ident or "",
             "timestamp": ts.isoformat() if ts else ""}
            for sid, title, stype, ident, ts in rows
        ]
        sources.sort(key=lambda s: s["timestamp"], reverse=True)
        return {
            "available": True,
            "knowledge_chunks": k_count,
            "conversation_chunks": c_count,
            "unique_sources": len(sources),
            "sources": sources[:50],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


async def list_sources(query: str = "", source_type: str = "",
                       limit: int = 200, offset: int = 0) -> dict:
    if not CFG.memory_enabled or not is_available():
        return {"available": False, "sources": [], "total": 0, "types": []}
    where: list[str] = []
    params: list[Any] = []
    if source_type:
        where.append("source_type = %s")
        params.append(source_type)
    if query:
        where.append("(title ILIKE %s OR identifier ILIKE %s)")
        like = f"%{query}%"
        params.extend([like, like])
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql_items = f"""
        SELECT source_id,
               MIN(title)        AS title,
               MIN(source_type)  AS source_type,
               MIN(identifier)   AS identifier,
               MAX(timestamp)    AS timestamp,
               COUNT(*)::int     AS chunk_count,
               SUM(LENGTH(text))::bigint AS char_count
          FROM knowledge_chunks
          {where_clause}
         GROUP BY source_id
         ORDER BY MAX(timestamp) DESC
    """
    sql_total = f"SELECT COUNT(DISTINCT source_id) FROM knowledge_chunks{where_clause}"
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_total, params)
                total = int((await cur.fetchone())[0])
                await cur.execute(
                    sql_items + " LIMIT %s OFFSET %s",
                    [*params, limit, offset],
                )
                items = [
                    {
                        "source_id": sid,
                        "title": (title or ident or "") if (title or ident) else "",
                        "source_type": stype or "",
                        "identifier": ident or "",
                        "timestamp": ts.isoformat() if ts else "",
                        "chunk_count": int(ccount),
                        "char_count": int(charc or 0),
                    }
                    for sid, title, stype, ident, ts, ccount, charc in await cur.fetchall()
                ]
                await cur.execute(
                    "SELECT DISTINCT source_type FROM knowledge_chunks ORDER BY source_type"
                )
                types = [r[0] for r in await cur.fetchall() if r[0]]
        return {"available": True, "total": total, "sources": items, "types": types}
    except Exception as e:
        return {"available": False, "error": str(e),
                "sources": [], "total": 0, "types": []}


async def get_source(source_id: str) -> dict:
    if not CFG.memory_enabled or not is_available():
        return {"available": False}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT title, source_type, identifier, timestamp,
                           chunk_index, total_chunks, text
                      FROM knowledge_chunks
                     WHERE source_id = %s
                     ORDER BY chunk_index ASC
                    """,
                    (source_id,),
                )
                rows = await cur.fetchall()
        if not rows:
            return {"available": True, "found": False}
        title, stype, ident, ts, _, _, _ = rows[0]
        chunks = [
            {"index": int(idx or 0),
             "total": int(total or len(rows)),
             "text": text or ""}
            for _, _, _, _, idx, total, text in rows
        ]
        full_text = "\n\n".join(c["text"] for c in chunks)
        return {
            "available": True,
            "found": True,
            "source_id": source_id,
            "title": title or "",
            "source_type": stype or "",
            "identifier": ident or "",
            "timestamp": ts.isoformat() if ts else "",
            "chunk_count": len(chunks),
            "char_count": sum(len(c["text"]) for c in chunks),
            "text": full_text,
            "chunks": chunks,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


async def crawled_identifiers_for_seed(seed_url: str) -> set[str]:
    """URLs previously stored from a crawl with this seed (for resume)."""
    if not CFG.memory_enabled or not seed_url or not is_available():
        return set()
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT identifier
                      FROM knowledge_chunks
                     WHERE metadata @> jsonb_build_object('crawl_seed', %s::text)
                    """,
                    (seed_url,),
                )
                return {r[0] for r in await cur.fetchall() if r[0]}
    except Exception as e:
        log.warning("crawled_identifiers_for_seed failed: %s", e)
        return set()


async def delete_source(source_id: str) -> dict:
    if not CFG.memory_enabled or not is_available():
        return {"deleted": 0, "source_id": source_id, "error": "memory not ready"}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM knowledge_chunks WHERE source_id = %s",
                    (source_id,),
                )
                count = cur.rowcount or 0
            await conn.commit()
        return {"deleted": int(count), "source_id": source_id}
    except Exception as e:
        return {"error": str(e), "source_id": source_id}


# ── Refinement support ─────────────────────────────────────────────────────
async def iter_unrefined_sources(pass_name: str, limit: int) -> list[dict]:
    """Sources whose chunk_index=0 metadata is missing this pass's marker.

    ``pass_name`` is one of ``summary``, ``tags`` — translated to a
    ``refined_<pass>_at`` JSONB key on the chunk_index=0 row, which holds
    the per-source refinement state. The caller fetches the full source
    text via ``get_source(source_id)`` separately to keep this query
    cheap and metadata-only.
    """
    if not CFG.memory_enabled or not is_available():
        return []
    key = f"refined_{pass_name}_at"
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source_id, source_type, title, identifier
                      FROM knowledge_chunks
                     WHERE chunk_index = 0
                       AND NOT (metadata ? %s)
                     ORDER BY timestamp ASC
                     LIMIT %s
                    """,
                    (key, limit),
                )
                rows = await cur.fetchall()
        return [
            {"source_id": sid, "source_type": stype or "",
             "title": title or "", "identifier": ident or ""}
            for sid, stype, title, ident in rows
        ]
    except Exception as e:
        log.warning("iter_unrefined_sources(%s) failed: %s", pass_name, e)
        return []


async def update_source_metadata(source_id: str, patch: dict) -> int:
    """Merge ``patch`` into the metadata of every chunk of a source.

    Stored on every row so any per-chunk read sees the source-level
    refinement (cheap; a source has tens of chunks at most). Returns the
    number of rows updated.
    """
    if not CFG.memory_enabled or not source_id or not is_available():
        return 0
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE knowledge_chunks
                       SET metadata = metadata || %s::jsonb
                     WHERE source_id = %s
                    """,
                    (json.dumps(patch), source_id),
                )
                count = cur.rowcount or 0
            await conn.commit()
            return int(count)
    except Exception as e:
        log.warning("update_source_metadata failed: %s", e)
        return 0


async def get_source_metadata(source_id: str) -> dict:
    if not CFG.memory_enabled or not is_available():
        return {}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT metadata FROM knowledge_chunks
                     WHERE source_id = %s AND chunk_index = 0
                     LIMIT 1
                    """,
                    (source_id,),
                )
                row = await cur.fetchone()
                return dict(row[0] or {}) if row else {}
    except Exception:
        return {}


async def upsert_gist_chunk(source_id: str, gist_text: str,
                            *, embedding: list[float] | None = None) -> bool:
    """Insert/replace a synthetic chunk_index=-1 row holding the source's
    LLM-generated gist. Embedded so it participates in retrieval — useful
    when the user's query matches the gist's wording but no individual
    chunk does. ``embedding`` may be supplied by the caller; if omitted
    we embed here.
    """
    if not CFG.memory_enabled or not is_available() or not gist_text.strip():
        return False
    if embedding is None:
        try:
            embedding = await _embed_one(gist_text)
        except Exception as e:
            log.warning("gist embed failed: %s", e)
            return False
    chunk_id = f"{source_id}_gist"
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                # Inherit identity from the source's first chunk.
                await cur.execute(
                    """
                    SELECT source_type, title, identifier, total_chunks, timestamp
                      FROM knowledge_chunks
                     WHERE source_id = %s AND chunk_index = 0
                     LIMIT 1
                    """,
                    (source_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return False
                stype, title, ident, total, ts = row
                await cur.execute(
                    """
                    INSERT INTO knowledge_chunks
                      (chunk_id, source_id, source_type, title, identifier,
                       chunk_index, total_chunks, text, embedding,
                       timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, -1, %s, %s, %s, %s,
                            jsonb_build_object('kind','gist'))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                       text = EXCLUDED.text,
                       embedding = EXCLUDED.embedding,
                       timestamp = EXCLUDED.timestamp,
                       metadata = knowledge_chunks.metadata
                                  || jsonb_build_object('kind','gist')
                    """,
                    (chunk_id, source_id, stype, title, ident,
                     total or 0, gist_text, embedding,
                     datetime.now(timezone.utc)),
                )
            await conn.commit()
            return True
    except Exception as e:
        log.warning("upsert_gist_chunk failed: %s", e)
        return False


# ── Crawl job persistence ──────────────────────────────────────────────────
async def save_crawl_job(job: dict) -> bool:
    """Upsert a crawl job row. Caller passes the job's status dict (see
    ``crawler.CrawlJob.to_status``); we map it onto the schema."""
    if not CFG.memory_enabled or not is_available():
        return False
    started_iso = job.get("started_at_iso")
    finished_iso = job.get("finished_at_iso")
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO crawl_jobs
                      (job_id, seed_url, status, strategy, max_pages, max_depth,
                       chunks_stored, pages_ingested, pages_skipped, resumed_from,
                       pages, errors, started_at, finished_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s::jsonb,
                            COALESCE(%s::timestamptz, NOW()), %s::timestamptz)
                    ON CONFLICT (job_id) DO UPDATE SET
                       status         = EXCLUDED.status,
                       strategy       = EXCLUDED.strategy,
                       chunks_stored  = EXCLUDED.chunks_stored,
                       pages_ingested = EXCLUDED.pages_ingested,
                       pages_skipped  = EXCLUDED.pages_skipped,
                       resumed_from   = EXCLUDED.resumed_from,
                       pages          = EXCLUDED.pages,
                       errors         = EXCLUDED.errors,
                       finished_at    = EXCLUDED.finished_at
                    """,
                    (
                        job["job_id"], job["seed_url"], job["status"],
                        job.get("strategy", ""), int(job.get("max_pages", 0)),
                        int(job.get("max_depth", 0)),
                        int(job.get("chunks_stored", 0)),
                        int(job.get("pages_ingested", 0)),
                        int(job.get("pages_skipped", 0)),
                        int(job.get("resumed_from", 0)),
                        json.dumps(job.get("pages", []) or []),
                        json.dumps(job.get("errors", []) or []),
                        started_iso, finished_iso,
                    ),
                )
            await conn.commit()
        return True
    except Exception as e:
        log.warning("save_crawl_job failed: %s", e)
        return False


def _row_to_crawl_status(row) -> dict:
    (job_id, seed_url, status, strategy, max_pages, max_depth,
     chunks_stored, pages_ingested, pages_skipped, resumed_from,
     pages, errors, started_at, finished_at) = row
    started_ts = started_at.timestamp() if started_at else None
    finished_ts = finished_at.timestamp() if finished_at else None
    elapsed = round((finished_ts or time.time()) - (started_ts or time.time()), 2)
    return {
        "job_id": job_id,
        "seed_url": seed_url,
        "status": status,
        "strategy": strategy,
        "max_pages": int(max_pages),
        "max_depth": int(max_depth),
        "pages_crawled": len(pages or []),
        "pages_ingested": int(pages_ingested),
        "pages_skipped": int(pages_skipped),
        "resumed_from": int(resumed_from),
        "chunks_stored": int(chunks_stored),
        "errors": (errors or [])[-10:],
        "started_at": started_ts,
        "finished_at": finished_ts,
        "elapsed_seconds": elapsed,
        "pages": pages or [],
    }


async def load_crawl_job(job_id: str) -> dict | None:
    if not CFG.memory_enabled or not is_available() or not job_id:
        return None
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT job_id, seed_url, status, strategy, max_pages, max_depth,
                           chunks_stored, pages_ingested, pages_skipped, resumed_from,
                           pages, errors, started_at, finished_at
                      FROM crawl_jobs
                     WHERE job_id = %s
                    """,
                    (job_id,),
                )
                row = await cur.fetchone()
        return _row_to_crawl_status(row) if row else None
    except Exception as e:
        log.warning("load_crawl_job failed: %s", e)
        return None


async def list_crawl_jobs(limit: int = 20) -> list[dict]:
    if not CFG.memory_enabled or not is_available():
        return []
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT job_id, seed_url, status, strategy, max_pages, max_depth,
                           chunks_stored, pages_ingested, pages_skipped, resumed_from,
                           pages, errors, started_at, finished_at
                      FROM crawl_jobs
                     ORDER BY started_at DESC
                     LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        return [_row_to_crawl_status(r) for r in rows]
    except Exception as e:
        log.warning("list_crawl_jobs failed: %s", e)
        return []


async def wipe(*, include_conversations: bool = True) -> dict:
    """Truncate the RAG tables. Used by the rebuild flow.

    knowledge_chunks is always cleared. Conversation history (and the
    feedback/distill state attached to it) is cleared by default; pass
    ``include_conversations=False`` to keep that audit trail.
    """
    if not CFG.memory_enabled or not is_available():
        return {"ok": False, "reason": "memory not ready"}
    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("TRUNCATE TABLE knowledge_chunks RESTART IDENTITY")
                # Crawl history points at sources that no longer exist after a
                # wipe — clear it too so the SPA doesn't show dead jobs.
                await cur.execute("TRUNCATE TABLE crawl_jobs")
                wiped_conv = 0
                if include_conversations:
                    await cur.execute("SELECT COUNT(*) FROM conversation_chunks")
                    wiped_conv = int((await cur.fetchone())[0])
                    await cur.execute("TRUNCATE TABLE conversation_chunks RESTART IDENTITY")
            await conn.commit()
        log.warning("memory.wipe(): tables truncated (conversations=%s)",
                    include_conversations)
        return {"ok": True, "wiped_conversations": wiped_conv,
                "include_conversations": include_conversations}
    except Exception as e:
        log.error("memory.wipe failed: %s", e)
        return {"ok": False, "error": str(e)}

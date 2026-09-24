"""Documents and URLs are stored in memory in full, not cut to the prompt cap.

extract_file / extract_url cut text to CONTEXT_CHAR_LIMIT (12,000 chars) so it
fits one chat prompt — and the ingest routes then stored that *same* cut text
in the library. Every PDF in the library (2026-09-24) held ~15k chars: the
first few pages of book-length documents, the rest silently dropped while the
UI reported the full char_count.

Pinned here:
  * extraction returns `text` (prompt, capped) and `full_text` (everything);
  * all four ingest routes store `full_text`;
  * storage runs as a tracked background task: a route that waits for it
    stops waiting after MEMORY_STORE_WAIT_S but never cancels it;
  * _embed_many bounds concurrent embedding calls (a book is ~700 chunks).
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import extract
import ingest
import memory as mem
import ollama as oll

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
LIMIT = extract.CFG.context_char_limit
BOOK = ("The principle of rhythm. " * 4000).strip()        # ~100k chars


# ── extraction ───────────────────────────────────────────────────────────────

class _Upload:
    def __init__(self, name: str, data: bytes):
        self.filename, self._data = name, data

    async def read(self) -> bytes:
        return self._data


def test_extract_file_keeps_full_text_alongside_prompt_text():
    out = asyncio.run(extract.extract_file(_Upload("book.txt", BOOK.encode())))
    assert out["truncated"] is True
    assert len(out["text"]) <= LIMIT
    assert out["full_text"] == BOOK
    assert out["char_count"] == len(BOOK)


def test_short_file_full_text_equals_text():
    out = asyncio.run(extract.extract_file(_Upload("note.txt", b"short note")))
    assert out["truncated"] is False
    assert out["full_text"] == out["text"] == "short note"


# ── routes ───────────────────────────────────────────────────────────────────

class Env:
    def __init__(self):
        self.stored: list[dict] = []
        self.store_delay = 0.0
        self.finished: list[str] = []


@pytest.fixture
def env(monkeypatch) -> Env:
    e = Env()
    cfg = dataclasses.replace(ingest.CFG, api_key=API_KEY, memory_enabled=True)
    monkeypatch.setattr(ingest, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)

    async def fake_store(*, text, title, source_type, identifier, extra=None):
        e.stored.append({"text": text, "identifier": identifier})
        await asyncio.sleep(e.store_delay)
        e.finished.append(identifier)
        return {"chunks_stored": 1, "chars_total": len(text)}

    async def fake_chat(_payload, **_kw):
        return {"message": {"content": "summary"}, "done_reason": "stop"}

    async def fake_extract_url(url):
        prompt_text, truncated = extract.truncate(BOOK)
        return {"url": url, "title": "Long article", "text": prompt_text,
                "full_text": BOOK, "source_type": "web", "truncated": truncated,
                "char_count": len(BOOK), "error": None}

    monkeypatch.setattr(mem, "store_knowledge", fake_store)
    monkeypatch.setattr(oll, "chat", fake_chat)
    monkeypatch.setattr(ingest, "extract_url", fake_extract_url)
    return e


@pytest.fixture
def client(env):
    app = FastAPI()
    app.include_router(ingest.router)
    with TestClient(app) as c:     # keeps the app loop alive for background tasks
        yield c


def _upload(client, path, **form):
    return client.post(path, headers=AUTH, data=form,
                       files={"file": ("book.txt", io.BytesIO(BOOK.encode()), "text/plain")})


def _wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_document_route_stores_full_text(client, env):
    r = _upload(client, "/v1/ingest/document")
    assert r.status_code == 200, r.text
    assert env.stored[0]["text"] == BOOK
    assert r.json()["source"]["truncated"] is True     # prompt was still capped


def test_document_conversation_route_stores_full_text(client, env):
    r = _upload(client, "/v1/ingest/document/conversation")
    assert r.status_code == 200, r.text
    assert _wait_for(lambda: env.finished)
    assert env.stored[0]["text"] == BOOK


def test_url_route_stores_full_text(client, env):
    r = client.post("/v1/ingest/url", headers=AUTH, json={"url": "https://x.test/a"})
    assert r.status_code == 200, r.text
    assert _wait_for(lambda: env.finished)
    assert env.stored[0]["text"] == BOOK


def test_url_conversation_route_stores_full_text(client, env):
    r = client.post("/v1/ingest/url/conversation", headers=AUTH,
                    json={"url": "https://x.test/b"})
    assert r.status_code == 200, r.text
    assert _wait_for(lambda: env.finished)
    assert env.stored[0]["text"] == BOOK


def test_slow_store_is_not_cancelled(client, env, monkeypatch):
    """A big book takes longer than the route waits — it must keep storing."""
    monkeypatch.setattr(ingest, "MEMORY_STORE_WAIT_S", 0.05)
    env.store_delay = 0.4
    r = _upload(client, "/v1/ingest/document")
    assert r.status_code == 200, r.text
    assert r.json()["memory"]["status"] == "storing"
    assert _wait_for(lambda: env.finished == ["book.txt"])


# ── embedding concurrency ────────────────────────────────────────────────────

def test_embed_many_bounds_concurrency_and_keeps_order(monkeypatch):
    live = {"now": 0, "max": 0}

    async def fake_embed(text):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.001)
        live["now"] -= 1
        return [float(text)]

    monkeypatch.setattr(mem, "_embed_one", fake_embed)
    out = asyncio.run(mem._embed_many([str(i) for i in range(100)]))
    assert out == [[float(i)] for i in range(100)]
    assert live["max"] <= mem.EMBED_CONCURRENCY


# ── site crawler ─────────────────────────────────────────────────────────────

def test_local_crawl_keeps_full_page_text(monkeypatch):
    import crawler

    async def fake_fetch(_client, _url):
        return "<html><body>no links</body></html>", "ok"

    async def fake_trafilatura(url):
        return {"url": url, "title": "Long page", "text": BOOK,
                "char_count": len(BOOK), "source_type": "web"}

    monkeypatch.setattr(crawler, "_fetch_html", fake_fetch)
    monkeypatch.setattr(crawler, "_trafilatura", fake_trafilatura)
    pages = asyncio.run(crawler._local_crawl("https://x.test/", 1, 0, []))
    assert len(pages) == 1
    assert pages[0].text == BOOK


# ── Library direct upload (/v1/memory/ingest) ────────────────────────────────

def test_memory_ingest_upload_stores_full_text(monkeypatch):
    import admin_routes

    cfg = dataclasses.replace(admin_routes.CFG, api_key=API_KEY, memory_enabled=True)
    monkeypatch.setattr(admin_routes, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)
    stored: list[dict] = []

    async def fake_store(*, text, title, source_type, identifier, extra=None):
        stored.append({"text": text, "source_type": source_type, "identifier": identifier})
        return {"chunks_stored": 1}

    monkeypatch.setattr(mem, "store_knowledge", fake_store)
    app = FastAPI()
    app.include_router(admin_routes.router)
    r = TestClient(app).post(
        "/v1/memory/ingest", headers=AUTH,
        data={"title": "book.pdf", "source_type": "pdf"},
        files={"file": ("book.txt", io.BytesIO(BOOK.encode()), "text/plain")})
    assert r.status_code == 200, r.text
    assert stored[0]["text"] == BOOK


# ── NUL bytes (PDF extraction) ───────────────────────────────────────────────

def test_store_knowledge_strips_nul_bytes(monkeypatch):
    """Postgres text rejects 0x00; some PDFs extract with NULs past page 5."""
    seen: list[str] = []

    async def fake_embed_many(texts):
        seen.extend(texts)
        raise RuntimeError("stop before the DB")   # only the chunking matters here

    monkeypatch.setattr(mem, "CFG", dataclasses.replace(mem.CFG, memory_enabled=True))
    monkeypatch.setattr(mem, "is_available", lambda: True)
    monkeypatch.setattr(mem, "_embed_many", fake_embed_many)
    asyncio.run(mem.store_knowledge(text="alpha\x00beta " * 200, title="t",
                                    source_type="pdf", identifier="t.pdf"))
    assert seen and all("\x00" not in c for c in seen)
    assert "alphabeta" in seen[0]


def test_embeddings_are_floats_even_when_ollama_returns_ints(monkeypatch):
    """psycopg refuses to dump a vector list mixing int and float
    ("cannot dump lists of mixed types; got: float, int")."""
    async def fake_embed(_text):
        return [0.5, 0, -1, 0.25]          # JSON numbers: 0 and -1 decode as int

    monkeypatch.setattr(mem.oll, "embed", fake_embed)
    out = asyncio.run(mem._embed_one("x"))
    assert out == [0.5, 0.0, -1.0, 0.25]
    assert all(type(x) is float for x in out)

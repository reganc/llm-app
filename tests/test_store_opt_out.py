"""`store: false` — read memory, never write it.

The gateway writes to RAG memory as a side effect of answering: every
conversation turn is embedded, and every web-search page is ingested. That is
right for interactive use and wrong for evals and batch callers, whose turns
would be recalled as "memory" by later requests (and by later eval runs,
making results drift run to run).

`store: false` suppresses both writes while leaving retrieval and search
untouched, on `/v1/chat/completions` (JSON and SSE, one- and two-pass) and
`/v1/chat/reasoning`. The default stays `true`.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat
import memory as mem
import ollama as oll
import search
import x_search

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class Recorder:
    def __init__(self) -> None:
        self.turns: list[dict] = []          # store_conversation_turn calls
        self.ingest_flags: list[bool] = []   # store_memory= passed to search
        self.retrievals = 0
        self.replies: list[dict] = []        # queued oll.chat replies
        self.auto_search = False
        self.force_search = False


@pytest.fixture
def rec(monkeypatch) -> Recorder:
    r = Recorder()
    cfg = dataclasses.replace(chat.CFG, api_key=API_KEY, memory_enabled=True,
                              search_enabled=True, x_search_enabled=True)
    monkeypatch.setattr(chat, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)

    async def fake_chat(_payload, **_kw):
        if r.replies:
            return r.replies.pop(0)
        return {"message": {"content": "answer"}, "done_reason": "stop"}

    async def fake_stream_chat(_payload, on_token=None, **_kw):
        if on_token:
            on_token("answer")
        yield 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_ingest(query, *, store_memory=True, **_kw):
        r.ingest_flags.append(store_memory)
        return {"query": query, "context_text": "[W1] result", "stored": 0,
                "results": []}

    async def fake_auto(_chunks, _query):
        return r.auto_search

    async def fake_retrieve(*_a, **_kw):
        r.retrievals += 1
        return []

    def fake_store(**kwargs):
        r.turns.append(kwargs)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(oll, "chat", fake_chat)
    monkeypatch.setattr(oll, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(search, "search_and_ingest", fake_ingest)
    monkeypatch.setattr(x_search, "search_x_and_ingest", fake_ingest)
    monkeypatch.setattr(search, "should_auto_search", fake_auto)
    monkeypatch.setattr(search, "detect_intent", lambda _q: {
        "signals": [], "force_x": False, "force_search": r.force_search})
    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "retrieve_deep", fake_retrieve)
    monkeypatch.setattr(mem, "lookup_by_title", fake_retrieve)
    monkeypatch.setattr(mem, "store_conversation_turn", fake_store)
    return r


@pytest.fixture
def client(rec) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def post(client, url="/v1/chat/completions", question="hello", **body):
    payload = {"messages": [{"role": "user", "content": question}], **body}
    r = client.post(url, json=payload, headers=AUTH)
    assert r.status_code == 200, r.text
    return r


# ── default is unchanged ─────────────────────────────────────────────────────

def test_default_still_stores_turn_and_search(client, rec):
    rec.auto_search = True
    post(client)
    assert len(rec.turns) == 1
    assert rec.ingest_flags == [True]


# ── /v1/chat/completions ─────────────────────────────────────────────────────

def test_store_false_skips_turn_and_search_ingest(client, rec):
    rec.auto_search = True
    post(client, store=False)
    assert rec.turns == []
    assert rec.ingest_flags == [False]


def test_store_false_still_retrieves_memory(client, rec):
    post(client, store=False)
    assert rec.retrievals >= 1


def test_store_false_streaming(client, rec):
    rec.auto_search = True
    r = post(client, store=False, stream=True)
    assert "[DONE]" in r.text
    assert rec.turns == []
    assert rec.ingest_flags == [False]


def test_store_false_forced_search_command(client, rec):
    post(client, question="/search gold price", store=False)
    assert rec.ingest_flags == [False]


def test_store_false_x_command(client, rec):
    post(client, question="/x latest on ollama", store=False)
    assert rec.ingest_flags == [False]


def test_store_false_model_requested_search(client, rec):
    """Two-pass: the model asks for a search via [SEARCH: ...]."""
    rec.replies = [
        {"message": {"content": "[SEARCH: ollama release]"}, "done_reason": "stop"},
        {"message": {"content": "final"}, "done_reason": "stop"},
    ]
    r = post(client, store=False)
    assert r.json()["choices"][0]["message"]["content"] == "final"
    assert rec.ingest_flags == [False]
    assert rec.turns == []


# ── /v1/chat/reasoning ───────────────────────────────────────────────────────

def test_reasoning_store_false(client, rec):
    rec.auto_search = True
    post(client, url="/v1/chat/reasoning", store=False, search=True)
    assert rec.turns == []
    assert rec.ingest_flags == [False]


def test_reasoning_default_stores_turn(client, rec):
    post(client, url="/v1/chat/reasoning")
    assert len(rec.turns) == 1


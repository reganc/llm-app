"""Follow-up query rewriting — retrieval and search see a standalone query.

Retrieval, the search router and web search all used only the last user
message, so "When was he born?" searched for exactly that (and in one eval run
the model filled the gap with "William Shatner birth date").

Contract pinned here:
  * rewriting only runs with prior turns AND a follow-up-shaped last message;
  * the rewrite is verified deterministically — every name or number it
    introduces must appear in the conversation, else the original is kept;
  * timeouts, errors and empty replies fall back to the original query;
  * endpoints pass the rewrite to retrieval/search but send the model the
    user's original words, and report the rewrite in the response.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat
import memory as mem
import ollama as oll
import rewrite
import search

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

MOBY = [
    {"role": "user", "content": "Who wrote Moby-Dick?"},
    {"role": "assistant", "content": "Moby-Dick was written by Herman Melville in 1851."},
]


# ── gate ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [
    "When was he born?", "What does it say about rhythm?", "and his brother?",
    "what about 1852?", "why?", "Tell me more", "How old is she now",
])
def test_needs_rewrite_for_follow_ups(query):
    assert rewrite.needs_rewrite(MOBY, query)


@pytest.mark.parametrize("query", [
    "What is the capital of Australia?",
    "Explain the Pythagorean theorem in two sentences please",
])
def test_standalone_questions_are_not_rewritten(query):
    assert not rewrite.needs_rewrite(MOBY, query)


def test_no_history_means_no_rewrite():
    assert not rewrite.needs_rewrite([], "When was he born?")


# ── deterministic guard ──────────────────────────────────────────────────────

def test_accepts_rewrite_using_names_from_history():
    ok = rewrite.accept_rewrite("When was Herman Melville born?",
                                MOBY, "When was he born?")
    assert ok == "When was Herman Melville born?"


def test_rejects_invented_entity():
    assert rewrite.accept_rewrite("William Shatner birth date",
                                  MOBY, "When was he born?") is None


def test_rejects_invented_number():
    assert rewrite.accept_rewrite("Melville born 1819", MOBY,
                                  "When was he born?") is None


def test_cleans_labels_quotes_and_extra_lines():
    got = rewrite.accept_rewrite('Standalone: "When was Melville born?"\nThat is all.',
                                 MOBY, "When was he born?")
    assert got == "When was Melville born?"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 400])
def test_rejects_empty_or_runaway(bad):
    assert rewrite.accept_rewrite(bad, MOBY, "When was he born?") is None


# ── rewrite() with a stubbed model ───────────────────────────────────────────

def _stub_chat(monkeypatch, reply=None, exc=None):
    sent: list[dict] = []

    async def fake_chat(payload, **kw):
        sent.append({"payload": payload, **kw})
        if exc:
            raise exc
        return {"message": {"content": reply}}

    monkeypatch.setattr(oll, "chat", fake_chat)
    return sent


def test_rewrite_returns_verified_rewrite(monkeypatch):
    sent = _stub_chat(monkeypatch, "When was Herman Melville born?")
    got = asyncio.run(rewrite.rewrite_query(MOBY, "When was he born?"))
    assert got == "When was Herman Melville born?"
    payload = sent[0]["payload"]
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == oll.CFG.num_ctx
    assert "Herman Melville" in payload["messages"][0]["content"]


def test_rewrite_skips_model_when_not_needed(monkeypatch):
    sent = _stub_chat(monkeypatch, "unused")
    q = "What is the capital of Australia?"
    assert asyncio.run(rewrite.rewrite_query(MOBY, q)) == q
    assert sent == []


def test_rewrite_falls_back_on_hallucination(monkeypatch):
    _stub_chat(monkeypatch, "William Shatner birth date")
    assert asyncio.run(rewrite.rewrite_query(MOBY, "When was he born?")) == "When was he born?"


def test_rewrite_falls_back_on_error(monkeypatch):
    _stub_chat(monkeypatch, exc=asyncio.TimeoutError())
    assert asyncio.run(rewrite.rewrite_query(MOBY, "When was he born?")) == "When was he born?"


# ── endpoint integration ─────────────────────────────────────────────────────

class Seen:
    def __init__(self):
        self.retrieved: list[str] = []
        self.searched: list[str] = []
        self.model_messages: list[list[dict]] = []


@pytest.fixture
def seen(monkeypatch) -> Seen:
    s = Seen()
    cfg = dataclasses.replace(chat.CFG, api_key=API_KEY, memory_enabled=True,
                              search_enabled=True, agent_tools=False)  # legacy path
    monkeypatch.setattr(chat, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)

    async def fake_chat(payload, **_kw):
        msgs = payload["messages"]
        if "Standalone" in msgs[-1]["content"]:     # the rewrite call
            return {"message": {"content": "When was Herman Melville born?"}}
        s.model_messages.append(msgs)
        return {"message": {"content": "1819"}, "done_reason": "stop"}

    async def fake_stream_chat(payload, on_token=None, **_kw):
        s.model_messages.append(payload["messages"])
        yield "data: [DONE]\n\n"

    async def fake_retrieve(query, *_a, **_kw):
        s.retrieved.append(query)
        return []

    async def fake_auto(_chunks, query):
        s.searched.append(query)
        return False

    def fake_store(**_kw):
        async def _noop():
            return None
        return _noop()

    monkeypatch.setattr(oll, "chat", fake_chat)
    monkeypatch.setattr(oll, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "retrieve_deep", fake_retrieve)
    monkeypatch.setattr(mem, "lookup_by_title", lambda *_a, **_k: fake_retrieve(""))
    monkeypatch.setattr(mem, "store_conversation_turn", fake_store)
    monkeypatch.setattr(search, "should_auto_search", fake_auto)
    monkeypatch.setattr(search, "detect_intent",
                        lambda _q: {"signals": [], "force_x": False, "force_search": False})
    return s


@pytest.fixture
def client(seen) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def _followup_body(**kw):
    return {"messages": MOBY + [{"role": "user", "content": "When was he born?"}],
            "inject_system": False, **kw}


def test_chat_retrieves_with_rewrite_but_model_sees_original(client, seen):
    r = client.post("/v1/chat/completions", json=_followup_body(), headers=AUTH)
    assert r.status_code == 200
    assert seen.retrieved[-1] == "When was Herman Melville born?"
    assert seen.searched == ["When was Herman Melville born?"]
    assert seen.model_messages[0][-1]["content"] == "When was he born?"
    assert r.json()["retrieval_query"] == "When was Herman Melville born?"


def test_chat_stream_uses_rewrite(client, seen):
    r = client.post("/v1/chat/completions", json=_followup_body(stream=True),
                    headers=AUTH)
    assert "event: llm.retrieval_query" in r.text
    assert seen.retrieved[-1] == "When was Herman Melville born?"


def test_single_turn_reports_no_rewrite(client, seen):
    r = client.post("/v1/chat/completions", headers=AUTH, json={
        "messages": [{"role": "user", "content": "When was he born?"}]})
    assert "retrieval_query" not in r.json()
    assert seen.retrieved[-1] == "When was he born?"


def test_reasoning_uses_rewrite(client, seen):
    r = client.post("/v1/chat/reasoning", json=_followup_body(), headers=AUTH)
    assert r.status_code == 200
    assert seen.retrieved[-1] == "When was Herman Melville born?"

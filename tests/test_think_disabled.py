"""Budgeted non-reasoning calls must run with `think: false`.

Qwen3.5 thinks by default, and reasoning tokens are spent before any content.
Three call sites sent small `num_predict` budgets without disabling thinking,
so the whole budget went to reasoning and they got back an empty string:

  * `search._classify_freshness` (60 tokens) — empty reply fell through to its
    `True` default, so every weak-memory query auto-searched the web;
  * `search._probe_hedging` (50 tokens) — empty reply never matched the hedge
    regex, so the probe could never fire;
  * `ollama.generate` behind `/v1/completions` — returned `text: ""` with
    `finish_reason: "stop"`, hiding the truncation from callers (regos).

The fake client records the JSON body of every POST so the tests pin the
payload actually sent to Ollama, not just the parsed result.
"""
from __future__ import annotations

import asyncio
import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat
import ollama as oll
import search

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _RecordingClient:
    """Stands in for `httpx.AsyncClient`; records each POSTed JSON body."""

    def __init__(self, sent: list[dict], body: dict):
        self._sent = sent
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, _url, json=None, **_kw):
        self._sent.append(json)
        return _FakeResponse(self._body)


def _patch(monkeypatch, module, body: dict) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(module.httpx, "AsyncClient",
                        lambda **_kw: _RecordingClient(sent, body))
    return sent


# ── search router probes ─────────────────────────────────────────────────────

def test_classify_freshness_disables_thinking(monkeypatch):
    sent = _patch(monkeypatch, search,
                  {"message": {"content": '{"needs_current": false}'}})
    assert asyncio.run(search._classify_freshness("what is pi")) is False
    assert sent[0]["think"] is False


def test_probe_hedging_disables_thinking(monkeypatch):
    sent = _patch(monkeypatch, search,
                  {"response": "I don't have access to real-time data."})
    assert asyncio.run(search._probe_hedging("price of gold")) is True
    assert sent[0]["think"] is False


# ── ollama.generate ──────────────────────────────────────────────────────────

def test_generate_disables_thinking_by_default(monkeypatch):
    sent = _patch(monkeypatch, oll, {"response": "hi", "done_reason": "stop"})
    asyncio.run(oll.generate("hello"))
    assert sent[0]["think"] is False


def test_generate_can_opt_into_thinking(monkeypatch):
    sent = _patch(monkeypatch, oll, {"response": "hi", "done_reason": "stop"})
    asyncio.run(oll.generate("hello", thinking=True))
    assert sent[0]["think"] is True


# ── /v1/completions finish_reason ────────────────────────────────────────────

def _client(monkeypatch, reply: dict) -> TestClient:
    cfg = dataclasses.replace(chat.CFG, api_key=API_KEY)
    monkeypatch.setattr(chat, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)

    async def fake_generate(*_a, **_kw):
        return reply

    monkeypatch.setattr(oll, "generate", fake_generate)
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def test_completions_reports_length_truncation(monkeypatch):
    client = _client(monkeypatch, {"response": "cut of", "done_reason": "length",
                                   "prompt_eval_count": 5, "eval_count": 8})
    r = client.post("/v1/completions", headers=AUTH,
                    json={"prompt": "x", "max_tokens": 8})
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_completions_reports_stop(monkeypatch):
    client = _client(monkeypatch, {"response": "done.", "done_reason": "stop"})
    r = client.post("/v1/completions", headers=AUTH, json={"prompt": "x"})
    assert r.json()["choices"][0]["finish_reason"] == "stop"


# ── num_ctx must match the chat path ─────────────────────────────────────────
# Ollama reloads the model (~7s on the 3060) whenever num_ctx changes between
# requests. The router probes sent none (→ Ollama's 4096) while chat sends
# CFG.num_ctx, so each request reloaded twice — and the reload blew the 4s
# classifier timeout, whose fallback is "search the web".

def test_classify_freshness_uses_shared_num_ctx(monkeypatch):
    sent = _patch(monkeypatch, search,
                  {"message": {"content": '{"needs_current": false}'}})
    asyncio.run(search._classify_freshness("what is pi"))
    assert sent[0]["options"]["num_ctx"] == search.CFG.num_ctx


def test_probe_hedging_uses_shared_num_ctx(monkeypatch):
    sent = _patch(monkeypatch, search, {"response": "fine"})
    asyncio.run(search._probe_hedging("what is pi"))
    assert sent[0]["options"]["num_ctx"] == search.CFG.num_ctx


def test_generate_defaults_to_shared_num_ctx(monkeypatch):
    sent = _patch(monkeypatch, oll, {"response": "hi", "done_reason": "stop"})
    asyncio.run(oll.generate("hello"))
    assert sent[0]["options"]["num_ctx"] == oll.CFG.num_ctx

"""`search.searxng_search` — which engines are asked, and failures are visible.

On 2026-09-24 every engine but Bing was failing inside SearXNG (Google
suspended, DuckDuckGo/Startpage/Qwant CAPTCHA, Brave rate-limited). SearXNG
reported it in `unresponsive_engines`, but the gateway ignored the field, so
the only symptom was bad answers from thin Bing results. The engine list
was also hard-coded, keeping a CAPTCHA-blocked engine in and a working one out.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging

import search


class _Resp:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _Client:
    def __init__(self, sent: list[dict], body: dict):
        self._sent, self._body = sent, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, _url, params=None, **_kw):
        self._sent.append(params)
        return _Resp(self._body)


def _patch(monkeypatch, body: dict, **cfg) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(search.httpx, "AsyncClient", lambda **_kw: _Client(sent, body))
    if cfg:
        monkeypatch.setattr(search, "CFG", dataclasses.replace(search.CFG, **cfg))
    return sent


RESULTS = {"results": [{"title": "Sam Altman - Wikipedia", "url": "https://w/a",
                        "content": "CEO of OpenAI", "engine": "google"}],
           "unresponsive_engines": []}


def test_default_engines_skip_captcha_blocked_duckduckgo():
    assert search.CFG.searxng_engines == "google,brave,bing"


def test_engines_come_from_config(monkeypatch):
    sent = _patch(monkeypatch, RESULTS, searxng_engines="google,mojeek")
    out = asyncio.run(search.searxng_search("OpenAI CEO", 5))
    assert sent[0]["engines"] == "google,mojeek"
    assert out[0]["title"] == "Sam Altman - Wikipedia"


def test_unresponsive_engines_are_logged(monkeypatch, caplog):
    body = dict(RESULTS, unresponsive_engines=[["duckduckgo", "CAPTCHA"],
                                               ["google", "Suspended: access denied"]])
    _patch(monkeypatch, body)
    with caplog.at_level(logging.WARNING, logger="llm-api.search"):
        asyncio.run(search.searxng_search("q", 5))
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "duckduckgo (CAPTCHA)" in msg and "google (Suspended: access denied)" in msg


def test_healthy_response_logs_nothing(monkeypatch, caplog):
    _patch(monkeypatch, RESULTS)
    with caplog.at_level(logging.WARNING, logger="llm-api.search"):
        asyncio.run(search.searxng_search("q", 5))
    assert not caplog.records

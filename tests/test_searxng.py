"""`search.searxng_search` — which engines are asked, and failures are visible.

On 2026-09-24 every engine but Bing was failing inside SearXNG (Google
suspended, DuckDuckGo/Startpage/Qwant CAPTCHA, Brave rate-limited). SearXNG
reported it in `unresponsive_engines`, but the gateway ignored the field, so
the only symptom was bad answers from thin Bing results. The engine list
was also hard-coded, keeping a CAPTCHA-blocked engine in and a working one out.

On 2026-11-20 it had flipped: Google suspended, Brave rate-limited, so Bing
alone answered "Who is the prime minister of the UK?" with Prime Video and
amazon.com. Bing outranks better engines even when mixed in, so it moved to a
fallback list that is only asked when the primary engines return nothing.
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
    """Answers each request with the next body; the last one repeats."""

    def __init__(self, sent: list[dict], bodies: list[dict]):
        self._sent, self._bodies = sent, bodies

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, _url, params=None, **_kw):
        self._sent.append(params)
        return _Resp(self._bodies[min(len(self._sent), len(self._bodies)) - 1])


def _patch(monkeypatch, *bodies: dict, **cfg) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(search.httpx, "AsyncClient",
                        lambda **_kw: _Client(sent, list(bodies)))
    if cfg:
        monkeypatch.setattr(search, "CFG", dataclasses.replace(search.CFG, **cfg))
    return sent


RESULTS = {"results": [{"title": "Sam Altman - Wikipedia", "url": "https://w/a",
                        "content": "CEO of OpenAI", "engine": "google"}],
           "unresponsive_engines": []}
EMPTY = {"results": [], "unresponsive_engines": [["google cse", "CAPTCHA"]]}


def test_default_engines_leave_bing_to_the_fallback():
    assert search.CFG.searxng_engines == "google cse,duckduckgo"
    assert search.CFG.searxng_fallback_engines == "bing,google,brave"


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


def test_results_from_primary_engines_skip_the_fallback(monkeypatch):
    sent = _patch(monkeypatch, RESULTS)
    asyncio.run(search.searxng_search("q", 5))
    assert [p["engines"] for p in sent] == [search.CFG.searxng_engines]


def test_empty_primary_retries_with_fallback_engines(monkeypatch):
    sent = _patch(monkeypatch, EMPTY, RESULTS,
                  searxng_engines="google cse", searxng_fallback_engines="bing")
    out = asyncio.run(search.searxng_search("UK prime minister", 5))
    assert [p["engines"] for p in sent] == ["google cse", "bing"]
    assert sent[1]["q"] == "UK prime minister"
    assert out[0]["title"] == "Sam Altman - Wikipedia"


def test_empty_fallback_setting_disables_the_retry(monkeypatch):
    sent = _patch(monkeypatch, EMPTY, RESULTS, searxng_fallback_engines="")
    assert asyncio.run(search.searxng_search("q", 5)) == []
    assert len(sent) == 1


def test_fallback_use_is_logged(monkeypatch, caplog):
    _patch(monkeypatch, EMPTY, RESULTS, searxng_fallback_engines="bing")
    with caplog.at_level(logging.WARNING, logger="llm-api.search"):
        asyncio.run(search.searxng_search("q", 5))
    assert any("fallback" in r.getMessage() and "bing" in r.getMessage()
               for r in caplog.records)


def _news(n):
    return {"results": [{"title": f"Story {i}", "url": f"https://n/{i}", "content": "c",
                         "engine": "bing news"} for i in range(n)]}


def test_news_search_asks_the_news_engines_for_the_last_day(monkeypatch):
    sent = _patch(monkeypatch, _news(8), searxng_news_engines="bing news")
    out = asyncio.run(search.searxng_news("ai", 8))
    assert len(out) == 8 and len(sent) == 1
    assert sent[0]["engines"] == "bing news" and sent[0]["time_range"] == "day"


def test_thin_news_widens_to_a_week(monkeypatch):
    sent = _patch(monkeypatch, _news(1), _news(6), searxng_news_engines="bing news")
    assert len(asyncio.run(search.searxng_news("ai", 8))) == 6
    assert [s["time_range"] for s in sent] == ["day", "week"]


def test_no_news_falls_back_to_the_normal_search(monkeypatch):
    sent = _patch(monkeypatch, EMPTY, EMPTY, RESULTS)
    out = asyncio.run(search.searxng_news("ai", 8))
    assert out[0]["url"] == "https://w/a" and "time_range" not in sent[2]


def test_a_day_of_thin_news_beats_an_empty_week(monkeypatch):
    _patch(monkeypatch, _news(1), EMPTY)
    assert len(asyncio.run(search.searxng_news("ai", 8))) == 1

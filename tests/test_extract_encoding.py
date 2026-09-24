"""URL extraction decodes by the page's declared charset, not UTF-8 by default.

whatdoesitmean.com serves `Content-Type: text/html` with no charset and
declares `<meta ... charset=windows-1252>`. `httpx.Response.text` fell back to
UTF-8, so every curly quote and ellipsis became U+FFFD ("Today�For") in text
headed for memory. trafilatura is given the raw bytes and reads the meta tag.
"""
from __future__ import annotations

import asyncio

import extract

BODY = ("The News You Need Today…For The World You’ll Live In Tomorrow. "
        "“Quoted words” appear here. ") * 20
HTML = ("<html><head><meta http-equiv=Content-Type "
        "content=\"text/html; charset=windows-1252\"><title>T</title></head>"
        f"<body><article><p>{BODY}</p></article></body></html>").encode("cp1252")


class _Resp:
    status_code = 200
    headers = {"content-type": "text/html"}        # no charset, like the real site
    content = HTML

    @property
    def text(self) -> str:                         # what httpx does without a charset
        return HTML.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, **_kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, _url):
        return _Resp()


def test_trafilatura_respects_meta_charset(monkeypatch):
    monkeypatch.setattr(extract.httpx, "AsyncClient", _Client)
    out = asyncio.run(extract._trafilatura("https://example.test/a.htm"))
    assert out and "�" not in out["text"]
    assert "Today…For" in out["text"]
    assert "You’ll" in out["text"]

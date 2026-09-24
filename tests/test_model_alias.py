"""The gateway shares Ollama's `default` runner with direct-to-Ollama apps.

Apps on Lane B call Ollama with `model: "default"`; the gateway used to
resolve that to the concrete tag. Ollama keys runners by *name*, so the same
weights loaded twice (6.3 GB + 7.6 GB on a 12 GB card) and the two runners
evicted each other — gateway calls waited ~45s behind the other app.

Now any request for the active model is sent as `default` — but only once
`copy_model(active, "default")` has succeeded, so a failed or stale alias
falls back to the concrete tag instead of hitting the wrong weights.
"""
from __future__ import annotations

import asyncio

import pytest

import ollama as oll

ACTIVE = "huihui_ai/qwen3.5-abliterated:9b"


@pytest.fixture(autouse=True)
def fresh_alias(monkeypatch):
    monkeypatch.setattr(oll, "_alias_target", None)
    monkeypatch.setattr(oll, "get_active_model", lambda: ACTIVE)


class _Resp:
    def __init__(self, code):
        self.status_code, self.text = code, ""


class _Client:
    def __init__(self, code):
        self._code = code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *_a, **_kw):
        return _Resp(self._code)


def copy(monkeypatch, source, code=200):
    monkeypatch.setattr(oll.httpx, "AsyncClient", lambda **_kw: _Client(code))
    return asyncio.run(oll.copy_model(source, "default"))


def test_without_a_confirmed_alias_the_tag_is_sent():
    assert oll.normalize_model("default") == ACTIVE
    assert oll.normalize_model(ACTIVE) == ACTIVE


def test_confirmed_alias_is_used_for_the_active_model(monkeypatch):
    assert copy(monkeypatch, ACTIVE) is True
    assert oll.normalize_model("default") == "default"
    assert oll.normalize_model("") == "default"
    assert oll.normalize_model("auto") == "default"
    assert oll.normalize_model(ACTIVE) == "default"          # explicit tag too
    assert oll.normalize_model("default [Search+Memory]") == "default"


def test_other_models_pass_through(monkeypatch):
    copy(monkeypatch, ACTIVE)
    assert oll.normalize_model("qwen3.5:9b") == "qwen3.5:9b"


def test_failed_copy_keeps_sending_the_tag(monkeypatch):
    assert copy(monkeypatch, ACTIVE, code=500) is False
    assert oll.normalize_model("default") == ACTIVE


def test_model_swap_moves_the_alias(monkeypatch):
    copy(monkeypatch, ACTIVE)
    new = "qwen3.5:14b"
    monkeypatch.setattr(oll, "get_active_model", lambda: new)
    # settings changed but the alias copy has not happened yet → concrete tag
    assert oll.normalize_model("default") == new
    copy(monkeypatch, new)
    assert oll.normalize_model("default") == "default"
    assert oll.normalize_model(ACTIVE) == ACTIVE              # old model: its own name


def test_copy_to_another_destination_does_not_touch_the_alias(monkeypatch):
    monkeypatch.setattr(oll.httpx, "AsyncClient", lambda **_kw: _Client(200))
    asyncio.run(oll.copy_model(ACTIVE, "something-else"))
    assert oll.normalize_model("default") == ACTIVE


def test_build_payload_uses_the_alias(monkeypatch):
    copy(monkeypatch, ACTIVE)
    assert oll.build_payload([], "default")["model"] == "default"

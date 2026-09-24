"""Tests for `stream_chat` under thinking models (qwen3.5+).

Streaming had the same blank-output failure as the non-streaming envelope, plus
one of its own: the loop only read `message.content`, so while the model was
thinking the client received nothing at all — and a run whose budget was fully
consumed by reasoning closed the stream without ever emitting a token.

The contract now:
  * `message.thinking` deltas are forwarded as `delta.reasoning` so a client can
    show progress during the thinking phase;
  * if the stream ends having emitted no content but some reasoning, the
    reasoning is flushed as content (and handed to `on_token`) rather than
    closing empty;
  * a stream that produced real content is untouched — no duplicated flush.

The fake client below stands in for `httpx.AsyncClient`; `stream_chat`
constructs its own client internally, so there is no seam to inject through.
"""
from __future__ import annotations

import asyncio
import json

import ollama as oll

PAYLOAD = {"model": "qwen3.5:9b", "messages": [], "stream": True}


# ── fake httpx.AsyncClient ───────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._response = _FakeResponse(lines, status_code)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, _method, _url, **_kwargs):
        return _FakeStreamCtx(self._response)


def _patch_client(monkeypatch, lines: list[str], status_code: int = 200) -> None:
    monkeypatch.setattr(
        oll.httpx, "AsyncClient",
        lambda **_kwargs: _FakeAsyncClient(lines, status_code),
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def ollama_line(content: str = "", thinking: str = "", done: bool = False) -> str:
    msg: dict = {}
    if content:
        msg["content"] = content
    if thinking:
        msg["thinking"] = thinking
    return json.dumps({"message": msg, "done": done})


def collect(monkeypatch, lines, status_code=200):
    """Run stream_chat over `lines`; return (sse_chunks, tokens_seen)."""
    _patch_client(monkeypatch, lines, status_code)
    tokens: list[str] = []

    async def run():
        return [c async for c in oll.stream_chat(dict(PAYLOAD), on_token=tokens.append)]

    return asyncio.run(run()), tokens


def deltas(chunks: list[str]) -> list[dict]:
    out = []
    for chunk in chunks:
        body = chunk[len("data: "):].strip()
        if not chunk.startswith("data: ") or body == "[DONE]":
            continue
        out.append(json.loads(body)["choices"][0].get("delta") or {})
    return out


def joined_content(chunks: list[str]) -> str:
    return "".join(d.get("content", "") for d in deltas(chunks))


def joined_reasoning(chunks: list[str]) -> str:
    return "".join(d.get("reasoning", "") for d in deltas(chunks))


# ── the original bug ─────────────────────────────────────────────────────────

def test_reasoning_only_stream_flushes_reasoning_as_content(monkeypatch):
    """Budget consumed by thinking: the client must not get an empty stream."""
    chunks, tokens = collect(monkeypatch, [
        ollama_line(thinking="step 1 "),
        ollama_line(thinking="step 2"),
        ollama_line(done=True),
    ])
    assert joined_content(chunks) == "step 1 step 2"
    assert "".join(tokens) == "step 1 step 2"


def test_reasoning_is_forwarded_as_its_own_delta_channel(monkeypatch):
    """Clients can render thinking progress instead of stalling on silence."""
    chunks, _ = collect(monkeypatch, [
        ollama_line(thinking="thinking hard"),
        ollama_line(content="answer"),
        ollama_line(done=True),
    ])
    assert joined_reasoning(chunks) == "thinking hard"


def test_stream_always_terminates_with_done(monkeypatch):
    chunks, _ = collect(monkeypatch, [ollama_line(thinking="x"), ollama_line(done=True)])
    assert chunks[-1] == "data: [DONE]\n\n"


# ── normal streams must not change ───────────────────────────────────────────

def test_content_stream_is_unchanged(monkeypatch):
    chunks, tokens = collect(monkeypatch, [
        ollama_line(content="Hello "),
        ollama_line(content="world"),
        ollama_line(done=True),
    ])
    assert joined_content(chunks) == "Hello world"
    assert tokens == ["Hello ", "world"]


def test_plain_content_stream_emits_no_reasoning_key(monkeypatch):
    """Non-thinking models must produce byte-identical deltas to before."""
    chunks, _ = collect(monkeypatch, [ollama_line(content="hi"), ollama_line(done=True)])
    assert all("reasoning" not in d for d in deltas(chunks))


def test_thinking_then_content_does_not_duplicate_the_answer(monkeypatch):
    """Once real content arrives, the end-of-stream flush must not fire."""
    chunks, tokens = collect(monkeypatch, [
        ollama_line(thinking="reasoning..."),
        ollama_line(content="the answer"),
        ollama_line(done=True),
    ])
    assert joined_content(chunks) == "the answer"
    assert "".join(tokens) == "the answer"


def test_no_flush_when_there_was_neither_content_nor_reasoning(monkeypatch):
    chunks, tokens = collect(monkeypatch, [ollama_line(done=True)])
    assert joined_content(chunks) == ""
    assert tokens == []


def test_whitespace_only_reasoning_is_not_flushed(monkeypatch):
    """Don't replace an empty stream with equally empty reasoning."""
    chunks, tokens = collect(monkeypatch, [
        ollama_line(thinking="   "),
        ollama_line(done=True),
    ])
    assert joined_content(chunks) == ""
    assert tokens == []


# ── malformed input / errors ─────────────────────────────────────────────────

def test_blank_and_malformed_lines_are_skipped(monkeypatch):
    chunks, _ = collect(monkeypatch, [
        "", "   ", "not json at all",
        ollama_line(content="ok"),
        ollama_line(done=True),
    ])
    assert joined_content(chunks) == "ok"


def test_error_object_in_stream_is_surfaced_and_terminated(monkeypatch):
    chunks, _ = collect(monkeypatch, [json.dumps({"error": "model not found"})])
    assert "model not found" in joined_content(chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


def test_non_200_status_is_surfaced_and_terminated(monkeypatch):
    chunks, _ = collect(monkeypatch, [], status_code=500)
    assert "500" in joined_content(chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


def test_on_token_is_optional(monkeypatch):
    """stream_chat is called without on_token on some paths."""
    _patch_client(monkeypatch, [ollama_line(thinking="x"), ollama_line(done=True)])

    async def run():
        return [c async for c in oll.stream_chat(dict(PAYLOAD))]

    assert asyncio.run(run())[-1] == "data: [DONE]\n\n"

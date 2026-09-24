"""Endpoint tests for `POST /v1/chat/reasoning`.

This is the only endpoint that runs a thinking model with `think: true`, and it
carries four behaviours that are easy to regress silently:

  * the token budget is floored at `REASONING_MAX_TOKENS` — reasoning tokens are
    spent before any content, so the 4096 chat budget returned empty replies;
  * a run truncated mid-reasoning reports `reasoning_truncated` and returns the
    reasoning text rather than an empty string;
  * auto web-search defaults **off** here (it defaults on for
    `/v1/chat/completions`), because injected results are the main budget sink —
    while explicit `/search` commands must still work;
  * an empty reply is never written back into RAG memory.

The app is assembled from `chat.router` alone rather than importing `main`, to
avoid its lifespan (postgres pool, warmup, schedulers) and the rate-limit
middleware. Ollama, search and memory are all stubbed — no services required.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat
import memory as mem
import ollama as oll
import search

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
URL = "/v1/chat/reasoning"


def sse(content: str = "", reasoning: str = "") -> str:
    """One OpenAI-style streaming chunk, as `stream_chat` would emit it."""
    delta: dict = {"content": content}
    if reasoning:
        delta["reasoning"] = reasoning
    return "data: " + json.dumps({
        "id": "chatcmpl-test", "object": "chat.completion.chunk",
        "created": 0, "model": "qwen3.5:9b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }) + "\n\n"


class Harness:
    """Stubbed collaborators + the recorded calls made through them."""

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []        # payloads sent to Ollama
        self.chat_kwargs: list[dict] = []       # kwargs (timeout, ...)
        self.searches: list[str] = []           # queries sent to web search
        self.stored: list[dict] = []            # conversation turns written to RAG
        self.reply: dict = {
            "message": {"content": "42"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        self.auto_search = True                 # what should_auto_search returns
        self.replies: list[dict] = []           # queued per-call replies (else `reply`)
        self.search_result: dict = {"context_text": "web result", "stored": 0,
                                    "results": []}
        # streaming
        self.stream_calls: list[dict] = []      # payloads sent to stream_chat
        self.stream_kwargs: list[dict] = []
        self.stream_reasoning: list[str] = ["thinking "]   # reasoning deltas to emit
        self.stream_tokens: list[str] = ["4", "2"]         # content deltas to emit


@pytest.fixture
def harness(monkeypatch) -> Harness:
    h = Harness()

    cfg = dataclasses.replace(
        chat.CFG,
        api_key=API_KEY,
        memory_enabled=True,
        search_enabled=True,
        reasoning_max_tokens=16384,
        reasoning_timeout_s=600.0,
    )
    monkeypatch.setattr(chat, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)

    async def fake_chat(payload, **kwargs):
        h.chat_calls.append(payload)
        h.chat_kwargs.append(kwargs)
        return h.replies.pop(0) if h.replies else h.reply

    async def fake_search_and_ingest(query, **_kw):
        h.searches.append(query)
        return h.search_result

    async def fake_should_auto_search(_chunks, _query):
        return h.auto_search

    def fake_store(**kwargs):
        # Called via asyncio.create_task(...), so the coroutine is constructed
        # eagerly — record here, synchronously, and hand back a no-op awaitable.
        h.stored.append(kwargs)

        async def _noop():
            return None

        return _noop()

    async def fake_retrieve(*_a, **_kw):
        return []

    async def fake_stream_chat(payload, on_token=None, *, timeout=180.0):
        h.stream_calls.append(payload)
        h.stream_kwargs.append({"timeout": timeout})
        for reasoning in h.stream_reasoning:
            yield sse(reasoning=reasoning)
        for token in h.stream_tokens:
            if on_token:
                on_token(token)
            yield sse(content=token)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(oll, "chat", fake_chat)
    monkeypatch.setattr(oll, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(search, "search_and_ingest", fake_search_and_ingest)
    monkeypatch.setattr(search, "should_auto_search", fake_should_auto_search)
    monkeypatch.setattr(search, "detect_intent",
                        lambda _q: {"signals": [], "force_x": False, "force_search": False})
    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "retrieve_deep", fake_retrieve)
    monkeypatch.setattr(mem, "lookup_by_title", fake_retrieve)
    monkeypatch.setattr(mem, "store_conversation_turn", fake_store)
    return h


@pytest.fixture
def client(harness) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def ask(client: TestClient, question: str = "What is 6*7?", **body):
    payload = {"messages": [{"role": "user", "content": question}], **body}
    return client.post(URL, json=payload, headers=AUTH)


def options_of(harness: Harness) -> dict:
    return harness.chat_calls[0]["options"]


# ── auth ─────────────────────────────────────────────────────────────────────

def test_requires_api_key(client):
    r = client.post(URL, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_rejects_wrong_api_key(client):
    r = client.post(URL, json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


# ── budget floor (the empty-reply root cause) ────────────────────────────────

def test_budget_is_floored_at_reasoning_max_tokens(client, harness):
    assert ask(client).status_code == 200
    assert options_of(harness)["num_predict"] == 16384


def test_small_caller_budget_is_raised_to_the_floor(client, harness):
    """A 256-token request would be consumed entirely by reasoning."""
    assert ask(client, max_tokens=256).status_code == 200
    assert options_of(harness)["num_predict"] == 16384


def test_larger_caller_budget_is_respected(client, harness):
    assert ask(client, max_tokens=32000).status_code == 200
    assert options_of(harness)["num_predict"] == 32000


def test_thinking_is_enabled(client, harness):
    """This endpoint is the only caller that opts into think=true."""
    assert ask(client).status_code == 200
    assert harness.chat_calls[0]["think"] is True


def test_uses_the_long_reasoning_timeout(client, harness):
    """A full reasoning pass takes minutes; the 180s default would abort it."""
    assert ask(client).status_code == 200
    assert harness.chat_kwargs[0]["timeout"] == 600.0


# ── truncation reporting ─────────────────────────────────────────────────────

def test_truncated_reasoning_is_reported_and_not_empty(client, harness):
    harness.reply = {
        "message": {"content": "", "thinking": "8.5h + 3.5h = 12h"},
        "done_reason": "length",
    }
    body = ask(client).json()
    msg = body["choices"][0]["message"]
    assert body["reasoning_truncated"] is True
    assert body["choices"][0]["finish_reason"] == "length"
    assert msg["content"] == "8.5h + 3.5h = 12h"
    assert msg["reasoning_fallback"] is True


def test_completed_reasoning_is_not_flagged_truncated(client, harness):
    harness.reply = {"message": {"content": "42", "thinking": "6*7"},
                     "done_reason": "stop"}
    body = ask(client).json()
    assert body["reasoning_truncated"] is False
    assert body["choices"][0]["message"]["content"] == "42"
    assert "reasoning_fallback" not in body["choices"][0]["message"]


def test_response_is_marked_as_reasoning_mode(client, harness):
    body = ask(client).json()
    assert body["reasoning_mode"] is True
    assert "message_id" in body


# ── search defaults OFF here ─────────────────────────────────────────────────

def test_auto_search_is_off_by_default(client, harness):
    """The asymmetry with /v1/chat/completions is deliberate — pin it."""
    assert ask(client).status_code == 200
    assert harness.searches == []


def test_search_can_be_opted_into(client, harness):
    assert ask(client, search=True).status_code == 200
    assert harness.searches != []


def test_explicit_search_command_still_searches(client, harness):
    """`/search …` must bypass the off-by-default flag."""
    assert ask(client, "/search latest ollama release").status_code == 200
    assert harness.searches != []


def test_search_false_is_honoured(client, harness):
    assert ask(client, search=False).status_code == 200
    assert harness.searches == []


def test_search_stays_off_when_globally_disabled(client, harness, monkeypatch):
    """An explicit opt-in cannot re-enable search the server has turned off."""
    monkeypatch.setattr(chat, "CFG",
                        dataclasses.replace(chat.CFG, search_enabled=False))
    assert ask(client, search=True).status_code == 200
    assert harness.searches == []


# ── memory writes ────────────────────────────────────────────────────────────

def test_normal_reply_is_written_to_memory(client, harness):
    assert ask(client).status_code == 200
    assert len(harness.stored) == 1
    assert harness.stored[0]["assistant_msg"] == "42"


def test_empty_reply_is_not_written_to_memory(client, harness):
    """An empty answer in the RAG store poisons later retrievals."""
    harness.reply = {"message": {"content": ""}, "done_reason": "length"}
    assert ask(client).status_code == 200
    assert harness.stored == []


def test_whitespace_only_reply_is_not_written_to_memory(client, harness):
    harness.reply = {"message": {"content": "  \n "}, "done_reason": "stop"}
    assert ask(client).status_code == 200
    assert harness.stored == []


def test_reasoning_fallback_text_is_written_to_memory(client, harness):
    """The fallback is a real answer, so it should be remembered."""
    harness.reply = {"message": {"content": "", "thinking": "12h * 0.42"},
                     "done_reason": "length"}
    assert ask(client).status_code == 200
    assert len(harness.stored) == 1
    assert harness.stored[0]["assistant_msg"] == "12h * 0.42"


def test_memory_can_be_disabled_per_request(client, harness):
    assert ask(client, memory=False).status_code == 200
    assert harness.stored == []


# ── failure handling ─────────────────────────────────────────────────────────

def test_ollama_failure_returns_500(client, harness, monkeypatch):
    async def boom(_payload, **_kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(oll, "chat", boom)
    assert ask(client).status_code == 500


# ── streaming ────────────────────────────────────────────────────────────────

def stream(client: TestClient, question: str = "What is 6*7?", **body):
    payload = {"messages": [{"role": "user", "content": question}],
               "stream": True, **body}
    return client.post(URL, json=payload, headers=AUTH)


def sse_events(text: str) -> list[str]:
    return [ln[len("event:"):].strip() for ln in text.splitlines()
            if ln.startswith("event:")]


def sse_deltas(text: str) -> list[dict]:
    out = []
    for ln in text.splitlines():
        if not ln.startswith("data: ") or ln.strip() == "data: [DONE]":
            continue
        try:
            out.append(json.loads(ln[6:])["choices"][0].get("delta") or {})
        except (ValueError, KeyError, IndexError):
            continue
    return out


def streamed_content(text: str) -> str:
    return "".join(d.get("content", "") for d in sse_deltas(text))


def streamed_reasoning(text: str) -> str:
    return "".join(d.get("reasoning", "") for d in sse_deltas(text))


def test_streaming_returns_an_event_stream(client, harness):
    r = stream(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")


def test_streaming_announces_the_message_id_first(client, harness):
    """Clients key their UI off llm.message_id before any token arrives."""
    assert sse_events(stream(client).text)[0] == "llm.message_id"


def test_streaming_terminates_with_done(client, harness):
    assert stream(client).text.rstrip().endswith("data: [DONE]")


def test_streaming_forwards_content_tokens(client, harness):
    assert streamed_content(stream(client).text) == "42"


def test_streaming_forwards_reasoning_deltas(client, harness):
    """The thinking channel must survive the endpoint's SSE wrapper."""
    harness.stream_reasoning = ["step 1 ", "step 2"]
    assert streamed_reasoning(stream(client).text) == "step 1 step 2"


# ── the same guarantees as the non-streaming path ────────────────────────────

def test_streaming_enables_thinking(client, harness):
    stream(client)
    assert harness.stream_calls[0]["think"] is True


def test_streaming_budget_is_floored(client, harness):
    stream(client, max_tokens=256)
    assert harness.stream_calls[0]["options"]["num_predict"] == 16384


def test_streaming_uses_the_long_reasoning_timeout(client, harness):
    """The 180s default would abort a real reasoning stream mid-flight."""
    stream(client)
    assert harness.stream_kwargs[0]["timeout"] == 600.0


def test_streaming_auto_search_is_off_by_default(client, harness):
    stream(client)
    assert harness.searches == []


def test_streaming_search_can_be_opted_into(client, harness):
    stream(client, search=True)
    assert harness.searches != []


# ── memory write-back after the stream closes ────────────────────────────────

def test_streaming_reply_is_written_to_memory(client, harness):
    """The turn is stored from the generator *after* the last chunk is sent."""
    stream(client)
    assert len(harness.stored) == 1
    assert harness.stored[0]["assistant_msg"] == "42"


def test_streaming_empty_reply_is_not_written_to_memory(client, harness):
    harness.stream_tokens = []
    stream(client)
    assert harness.stored == []


def test_streaming_whitespace_only_reply_is_not_written_to_memory(client, harness):
    harness.stream_tokens = [" ", "\n"]
    stream(client)
    assert harness.stored == []


def test_streaming_memory_can_be_disabled_per_request(client, harness):
    stream(client, memory=False)
    assert harness.stored == []


# ── two-pass branch (model asks for a search mid-answer) ─────────────────────
#
# Reached when search is enabled but auto-search did not fire, so the model is
# offered the [SEARCH: …] capability and decides for itself. The first pass
# doubles as the answer when it declines, so it must reason too.

def two_pass(client: TestClient, harness: Harness, **body):
    harness.auto_search = False          # let the model decide instead
    return ask(client, search=True, **body)


def test_two_pass_first_pass_enables_thinking(client, harness):
    """The first pass is also the answer when no search is requested."""
    assert two_pass(client, harness).status_code == 200
    assert harness.chat_calls[0]["think"] is True


def test_two_pass_first_pass_uses_the_long_timeout(client, harness):
    two_pass(client, harness)
    assert harness.chat_kwargs[0]["timeout"] == 600.0


def test_two_pass_first_pass_budget_is_floored(client, harness):
    two_pass(client, harness, max_tokens=256)
    assert harness.chat_calls[0]["options"]["num_predict"] == 16384


def test_two_pass_second_pass_also_thinks(client, harness):
    """Model asks to search; the answer-with-results pass must reason too."""
    harness.replies = [
        {"message": {"content": "[SEARCH: ollama latest version]"}, "done_reason": "stop"},
        {"message": {"content": "0.32.14"}, "done_reason": "stop"},
    ]
    body = two_pass(client, harness).json()
    assert harness.searches == ["ollama latest version"]
    assert len(harness.chat_calls) == 2
    assert [c["think"] for c in harness.chat_calls] == [True, True]
    assert body["choices"][0]["message"]["content"] == "0.32.14"


def test_two_pass_retry_after_failed_search_also_thinks(client, harness):
    """Search failed, so the model answers unaided — still with reasoning."""
    harness.replies = [
        {"message": {"content": "[SEARCH: something]"}, "done_reason": "stop"},
        {"message": {"content": "answered anyway"}, "done_reason": "stop"},
    ]
    harness.search_result = {"error": "searxng down"}
    body = two_pass(client, harness).json()
    assert len(harness.chat_calls) == 2
    assert [c["think"] for c in harness.chat_calls] == [True, True]
    assert body["choices"][0]["message"]["content"] == "answered anyway"


def test_two_pass_empty_first_pass_does_not_crash(client, harness):
    """Reasoning can exhaust the budget and emit no content at all."""
    harness.replies = [{"message": {"content": None, "thinking": "ran out"},
                        "done_reason": "length"}]
    body = two_pass(client, harness).json()
    assert harness.searches == []
    assert body["choices"][0]["message"]["content"] == "ran out"


def test_non_reasoning_callers_still_do_not_think(client, harness, monkeypatch):
    """/v1/chat/completions shares _two_pass_chat and must stay non-thinking."""
    import asyncio

    async def run():
        return await chat._two_pass_chat(
            [{"role": "user", "content": "hi"}], "m", 0.7, 2048, 0.9, None,
        )

    asyncio.run(run())
    assert harness.chat_calls[0]["think"] is False
    assert harness.chat_kwargs[0]["timeout"] == 180.0

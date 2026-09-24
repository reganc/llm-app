"""`/v1/chat/completions` in agent mode (native tool calling).

Agent mode replaces the auto-search router and the [SEARCH:] sentinel with
the tool loop. It is opt-in (`agent: true`, or AGENT_TOOLS=true server-wide),
and is bypassed for response_format and explicit /commands.
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
URL = "/v1/chat/completions"

GOOD = {"score": 0.9, "text": "the principle of rhythm", "title": "Kybalion",
        "identifier": "kyb.pdf", "source_type": "pdf", "timestamp": ""}


class Env:
    def __init__(self):
        self.chat_payloads: list[dict] = []
        self.stream_payloads: list[dict] = []
        self.replies: list[dict] = []
        self.stream_rounds: list[list[dict]] = []
        self.searches: list[tuple[str, bool]] = []
        self.router_calls = 0
        self.stored: list[dict] = []
        self.memory: list[dict] = []


@pytest.fixture
def env(monkeypatch) -> Env:
    e = Env()
    cfg = dataclasses.replace(chat.CFG, api_key=API_KEY, memory_enabled=True,
                              search_enabled=True, agent_tools=False,
                              memory_min_score=0.70)
    monkeypatch.setattr(chat, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)
    monkeypatch.setattr("agent.CFG", cfg)

    async def fake_chat(payload, **_kw):
        e.chat_payloads.append(payload)
        if e.replies:
            return e.replies.pop(0)
        return {"message": {"content": "answer"}, "done_reason": "stop"}

    def fake_stream_events(payload, **_kw):
        e.stream_payloads.append(payload)
        chunks = e.stream_rounds.pop(0) if e.stream_rounds else [
            {"message": {"content": "answer"}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "stop"}]

        async def gen():
            for c in chunks:
                yield c
        return gen()

    async def fake_stream_chat(payload, on_token=None, **_kw):
        yield "data: [DONE]\n\n"

    async def fake_search(query, *, store_memory=True, **_kw):
        e.searches.append((query, store_memory))
        return {"context_text": "[W1] gold is $4,700", "results": [{"engine": "searxng"}]}

    async def fake_router(_chunks, _q):
        e.router_calls += 1
        return False

    async def fake_retrieve(*_a, **_kw):
        return list(e.memory)

    async def no_titles(*_a, **_kw):
        return []

    def fake_store(**kw):
        e.stored.append(kw)

        async def _noop():
            return None
        return _noop()

    monkeypatch.setattr(oll, "chat", fake_chat)
    monkeypatch.setattr(oll, "stream_events", fake_stream_events)
    monkeypatch.setattr(oll, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(search, "search_and_ingest", fake_search)
    monkeypatch.setattr(search, "should_auto_search", fake_router)
    monkeypatch.setattr(search, "detect_intent",
                        lambda _q: {"signals": [], "force_x": False, "force_search": False})
    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "retrieve_deep", fake_retrieve)
    monkeypatch.setattr(mem, "lookup_by_title", no_titles)
    monkeypatch.setattr(mem, "store_conversation_turn", fake_store)
    monkeypatch.setattr(mem, "build_context_block",
                        lambda chunks, **_kw: "\n".join(f"[L{i}] {c['text']}"
                                                         for i, c in enumerate(chunks, 1)))
    return e


@pytest.fixture
def client(env) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def ask(client, question="price of gold?", **body):
    r = client.post(URL, headers=AUTH,
                    json={"messages": [{"role": "user", "content": question}], **body})
    assert r.status_code == 200, r.text
    return r


def tool_call(name, query):
    return {"function": {"name": name, "arguments": {"query": query}}}


# ── switching ────────────────────────────────────────────────────────────────

def test_agent_tools_false_uses_legacy_path(client, env):
    ask(client)
    assert env.router_calls == 1
    assert all("tools" not in p for p in env.chat_payloads)


def test_agent_true_skips_router_and_offers_tools(client, env):
    body = ask(client, agent=True).json()
    assert env.router_calls == 0
    names = [t["function"]["name"] for t in env.chat_payloads[0]["tools"]]
    assert names == ["web_search", "library_search"]
    assert body["agent"] == {"rounds": 1, "tool_calls": []}
    system = env.chat_payloads[0]["messages"][0]
    assert system["role"] == "system" and "library_search" in system["content"]


def test_server_flag_enables_and_request_can_opt_out(client, env, monkeypatch):
    monkeypatch.setattr(chat, "CFG", dataclasses.replace(chat.CFG, agent_tools=True))
    ask(client)
    assert env.router_calls == 0
    ask(client, agent=False)
    assert env.router_calls == 1


@pytest.mark.parametrize("extra", [
    {"response_format": {"type": "json_object"}},
    {"raw": True},
    {"search": False, "memory": False},
])
def test_agent_bypassed(client, env, extra):
    ask(client, agent=True, **extra)
    assert all("tools" not in p for p in env.chat_payloads)


def test_explicit_command_bypasses_agent(client, env):
    ask(client, question="/search gold", agent=True)
    assert env.searches == [("gold", True)]
    assert all("tools" not in p for p in env.chat_payloads)


def test_search_disabled_offers_only_library(client, env):
    ask(client, agent=True, search=False)
    names = [t["function"]["name"] for t in env.chat_payloads[0]["tools"]]
    assert names == ["library_search"]


def test_library_phrase_offers_no_tools(client, env):
    env.memory = [GOOD]
    ask(client, question="what does my library say about rhythm?", agent=True)
    assert "tools" not in env.chat_payloads[0]


# ── the loop through the endpoint ────────────────────────────────────────────

def test_tool_round_trip_and_metadata(client, env):
    env.replies = [
        {"message": {"content": "", "tool_calls": [tool_call("web_search", "gold price")]}},
        {"message": {"content": "$4,700 [W1]"}, "done_reason": "stop"},
    ]
    body = ask(client, agent=True).json()
    assert body["choices"][0]["message"]["content"] == "$4,700 [W1]"
    assert env.searches == [("gold price", True)]
    assert body["web_search"]["via"] == "tool"
    assert body["web_search"]["markers"]["W"] == 1
    assert body["agent"]["tool_calls"] == [
        {"name": "web_search", "query": "gold price", "status": "ok"}]


def test_prefetched_memory_is_injected_and_reported(client, env):
    env.memory = [GOOD]
    body = ask(client, question="what is the principle of rhythm?", agent=True).json()
    sent = json.dumps(env.chat_payloads[0]["messages"])
    assert "[L1] the principle of rhythm" in sent
    assert body["memory"]["used"] == 1


def test_store_false_reaches_tools_and_skips_turn(client, env):
    env.replies = [
        {"message": {"content": "", "tool_calls": [tool_call("web_search", "gold")]}},
        {"message": {"content": "ok"}, "done_reason": "stop"},
    ]
    ask(client, agent=True, store=False)
    assert env.searches == [("gold", False)]
    assert env.stored == []


def test_turn_is_stored_by_default(client, env):
    ask(client, agent=True)
    assert len(env.stored) == 1


# ── streaming ────────────────────────────────────────────────────────────────

def test_stream_events_order(client, env):
    env.stream_rounds = [
        [{"message": {"content": "", "tool_calls": [tool_call("web_search", "gold")]},
          "done": False},
         {"message": {"content": ""}, "done": True}],
        [{"message": {"content": "$4,700 "}, "done": False},
         {"message": {"content": "[W1]"}, "done": False},
         {"message": {"content": ""}, "done": True, "done_reason": "stop"}],
    ]
    text = ask(client, agent=True, stream=True).text
    order = [text.index(s) for s in (
        "event: llm.message_id", "event: llm.tool_call", "$4,700", "event: llm.web_search",
        "event: llm.agent", "data: [DONE]")]
    assert order == sorted(order)
    assert env.searches == [("gold", True)]
    assert len(env.stored) == 1 and env.stored[0]["assistant_msg"] == "$4,700 [W1]"


def test_stream_error_is_surfaced_not_raised(client, env, monkeypatch):
    def broken(_payload, **_kw):
        async def gen():
            raise RuntimeError("Ollama returned 500: boom")
            yield  # pragma: no cover
        return gen()
    monkeypatch.setattr(oll, "stream_events", broken)
    text = ask(client, agent=True, stream=True).text
    assert "[Error: Ollama returned 500: boom]" in text
    assert text.rstrip().endswith("data: [DONE]")


# ── deterministic intent still forces a search ───────────────────────────────

def test_time_sensitive_phrasing_searches_before_the_model(client, env, monkeypatch):
    monkeypatch.setattr(search, "detect_intent", lambda _q: {
        "signals": ["today"], "force_x": False, "force_search": True})
    env.memory = [dict(GOOD, text="gold price was $4,700", title="old gold page")]
    body = ask(client, question="gold price today?", agent=True).json()
    assert env.searches == [("gold price today?", True)]
    first = json.dumps(env.chat_payloads[0]["messages"])
    assert "[W1] gold is $4,700" in first
    assert body["web_search"]["intent_signals"] == ["today"]
    assert body["agent"]["tool_calls"][0] == {
        "name": "web_search", "query": "gold price today?", "status": "ok"}


def test_no_intent_no_pre_search(client, env):
    ask(client, question="who wrote moby dick?", agent=True)
    assert env.searches == []


# ── fix: agent pre-fetch keeps only user-saved sources ───────────────────────

def test_prefetch_drops_search_ingests_and_conversation_turns(client, env):
    env.memory = [
        dict(GOOD, text="rhythm pdf", identifier="kyb.pdf"),
        dict(GOOD, text="rhythm snapshot", source_type="bing", identifier="b"),
        dict(GOOD, text="rhythm chat", source_type="conversation", identifier="c"),
    ]
    body = ask(client, question="what is the principle of rhythm?", agent=True).json()
    sent = json.dumps(env.chat_payloads[0]["messages"])
    assert "rhythm pdf" in sent
    assert "rhythm snapshot" not in sent and "rhythm chat" not in sent
    assert [i["identifier"] for i in body["memory"]["items"]] == ["kyb.pdf"]


def test_library_mode_keeps_every_source_type(client, env):
    env.memory = [dict(GOOD, text="rhythm snapshot", source_type="bing")]
    ask(client, question="what does my library say about rhythm?", agent=True)
    assert "rhythm snapshot" in json.dumps(env.chat_payloads[0]["messages"])


def test_legacy_path_prefetch_unchanged(client, env):
    env.memory = [dict(GOOD, text="rhythm snapshot", source_type="bing")]
    ask(client, question="what is the principle of rhythm?")
    assert "rhythm snapshot" in json.dumps(env.chat_payloads[0]["messages"])


# ── citation repair through the endpoint ─────────────────────────────────────

def test_json_repair_reported_and_stored(client, env):
    env.replies = [
        {"message": {"content": "", "tool_calls": [tool_call("web_search", "gold")]}},
        {"message": {"content": "gold $4,700 [W7]"}, "done_reason": "stop"},
        {"message": {"content": "gold $4,700 [W1]"}, "done_reason": "stop"},
    ]
    body = ask(client, agent=True).json()
    assert body["choices"][0]["message"]["content"] == "gold $4,700 [W1]"
    assert body["agent"]["citation_repair"] == {"kind": "invalid", "invalid": ["W7"],
                                                "resolved": True, "stripped": []}
    assert env.stored[0]["assistant_msg"] == "gold $4,700 [W1]"


def test_stream_repair_sends_replace_and_stores_fixed(client, env):
    env.stream_rounds = [
        [{"message": {"content": "", "tool_calls": [tool_call("web_search", "gold")]},
          "done": False}, {"message": {"content": ""}, "done": True}],
        [{"message": {"content": "gold [W7]"}, "done": False},
         {"message": {"content": ""}, "done": True, "done_reason": "stop"}],
    ]
    env.replies = [{"message": {"content": "gold [W1]"}, "done_reason": "stop"}]
    text = ask(client, agent=True, stream=True).text
    assert "event: llm.replace" in text
    replace = text.split("event: llm.replace\ndata: ")[1].split("\n")[0]
    assert json.loads(replace) == {"content": "gold [W1]", "reason": "citation_repair",
                                   "kind": "invalid", "invalid": ["W7"]}
    assert text.index("event: llm.replace") < text.index("data: [DONE]")
    assert env.stored[0]["assistant_msg"] == "gold [W1]"


# ── default ──────────────────────────────────────────────────────────────────

def test_agent_mode_is_on_by_default(tmp_path):
    """AGENT_TOOLS unset → agent mode. Checked in a fresh interpreter so the
    already-imported CFG (and the runner's own env) can't mask the default."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if k != "AGENT_TOOLS"}
    env["DATA_DIR"] = str(tmp_path)
    api = Path(__file__).resolve().parent.parent / "api"
    out = subprocess.run([sys.executable, "-c",
                          "import config; print(config.CFG.agent_tools)"],
                         cwd=api, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"

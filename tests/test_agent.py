"""Tool-calling loop (`api/agent.py`) — offline, with a scripted model.

Pins the deterministic guard rails, which are what make a 9B model safe to
hand tools to: bounded rounds and calls, a final no-tools round that forces
an answer, argument validation, duplicate suppression, error containment,
and citation markers that stay unique across calls.
"""
from __future__ import annotations

import asyncio

import pytest

import agent
import memory as mem
import search

BASE = {"model": "m", "think": False, "options": {"num_ctx": 8192}}
USER = [{"role": "user", "content": "q"}]
TOOLS = agent.tools_for(web=True, library=True)


def call(name, query=None, raw_args=None):
    args = raw_args if raw_args is not None else ({"query": query} if query is not None else {})
    return {"function": {"name": name, "arguments": args}}


def reply(content="", calls=None, done_reason="stop"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"message": msg, "done_reason": done_reason}


class Script:
    """Scripted model: returns queued replies, records every payload."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.payloads: list[dict] = []

    async def __call__(self, payload):
        self.payloads.append(payload)
        return self.replies.pop(0) if self.replies else reply("final answer")


@pytest.fixture
def tools_env(monkeypatch):
    env = {"searches": [], "stores": [], "retrieves": []}

    async def fake_search(query, *, store_memory=True, **_kw):
        env["searches"].append(query)
        env["stores"].append(store_memory)
        return {"context_text": f"[W1] one about {query}\n[W2] two",
                "results": [{"engine": "searxng"}, {"engine": "searxng"}]}

    async def fake_retrieve(query, *_a, **_kw):
        env["retrieves"].append(query)
        return [{"score": 0.9, "text": f"{query} text", "title": "t", "identifier": "i",
                 "source_type": "pdf", "timestamp": ""}]

    monkeypatch.setattr(search, "search_and_ingest", fake_search)
    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "build_context_block",
                        lambda chunks, **_kw: "\n".join(f"[L{i}] {c['text']}"
                                                         for i, c in enumerate(chunks, 1)))
    return env


def run(script, tools=TOOLS, store=True, trace=None):
    return asyncio.run(agent.run(USER, BASE, tools, store=store, trace=trace, chat=script))


# ── basic flow ───────────────────────────────────────────────────────────────

def test_direct_answer_makes_one_call_with_tools_offered(tools_env):
    script = Script(reply("Melville"))
    data, trace = run(script)
    assert data["message"]["content"] == "Melville"
    assert len(script.payloads) == 1
    assert script.payloads[0]["tools"] == TOOLS
    assert trace.calls == [] and tools_env["searches"] == []


def test_tool_call_then_answer(tools_env):
    script = Script(reply(calls=[call("web_search", "gold price")]), reply("$4,700 [W1]"))
    data, trace = run(script)
    assert data["message"]["content"] == "$4,700 [W1]"
    assert tools_env["searches"] == ["gold price"]
    second = script.payloads[1]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert second[-1]["role"] == "tool" and second[-1]["tool_name"] == "web_search"
    assert "[W1] one about gold price" in second[-1]["content"]
    assert trace.web_summary()["query"] == "gold price"
    assert trace.web_summary()["markers"] == {"W": 2, "X": 0, "A": 0}


def test_base_payload_is_not_mutated(tools_env):
    base = dict(BASE)
    asyncio.run(agent.run(USER, base, TOOLS, store=True,
                          chat=Script(reply(calls=[call("web_search", "x")]))))
    assert base == BASE


# ── bounds ───────────────────────────────────────────────────────────────────

def test_last_round_offers_no_tools_so_loop_ends(tools_env):
    always_search = [reply(calls=[call("web_search", f"q{i}")]) for i in range(10)]
    script = Script(*always_search)
    data, trace = run(script)
    assert len(script.payloads) == agent.MAX_ROUNDS
    assert "tools" not in script.payloads[-1]
    assert trace.rounds == agent.MAX_ROUNDS


def test_tool_call_budget_caps_executions(tools_env):
    many = [call("web_search", f"q{i}") for i in range(7)]
    script = Script(reply(calls=many), reply("done"))
    data, trace = run(script)
    assert len(tools_env["searches"]) == agent.MAX_TOOL_CALLS
    echoed = script.payloads[1]["messages"]
    assert len([m for m in echoed if m["role"] == "tool"]) == agent.MAX_TOOL_CALLS
    assert len(echoed[-agent.MAX_TOOL_CALLS - 1]["tool_calls"]) == agent.MAX_TOOL_CALLS
    # budget exhausted → next round forced to answer without tools
    assert "tools" not in script.payloads[1]


def test_no_tools_available_means_single_plain_call(tools_env):
    script = Script(reply("hi"))
    run(script, tools=[])
    assert "tools" not in script.payloads[0]


# ── validation & containment ─────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    call("delete_everything", "x"),
    call("web_search"),
    call("web_search", "   "),
    call("web_search", "x" * 301),
    call("web_search", raw_args="{not json"),
    call("web_search", raw_args=["query"]),
])
def test_invalid_calls_return_errors_to_the_model(tools_env, bad):
    script = Script(reply(calls=[bad]), reply("recovered"))
    data, trace = run(script)
    assert data["message"]["content"] == "recovered"
    assert tools_env["searches"] == []
    assert script.payloads[1]["messages"][-1]["content"].startswith("Error:")
    assert trace.calls[0]["status"] == "invalid"


def test_json_string_arguments_are_accepted(tools_env):
    script = Script(reply(calls=[call("web_search", raw_args='{"query": "x"}')]), reply("ok"))
    run(script)
    assert tools_env["searches"] == ["x"]


def test_tool_not_offered_is_rejected(tools_env):
    script = Script(reply(calls=[call("web_search", "x")]), reply("ok"))
    run(script, tools=agent.tools_for(web=False, library=True))
    assert tools_env["searches"] == []


def test_duplicate_call_is_not_re_executed(tools_env):
    script = Script(reply(calls=[call("web_search", "Gold  Price")]),
                    reply(calls=[call("web_search", "gold price")]), reply("ok"))
    _, trace = run(script)
    assert tools_env["searches"] == ["Gold Price"]
    assert [c["status"] for c in trace.calls] == ["ok", "duplicate"]


def test_failing_tool_does_not_kill_the_turn(tools_env, monkeypatch):
    async def boom(*_a, **_kw):
        raise ConnectionError("searxng down")
    monkeypatch.setattr(search, "search_and_ingest", boom)
    script = Script(reply(calls=[call("web_search", "x")]), reply("answered anyway"))
    data, trace = run(script)
    assert data["message"]["content"] == "answered anyway"
    assert trace.calls[0]["status"] == "error"


def test_store_flag_reaches_web_search(tools_env):
    run(Script(reply(calls=[call("web_search", "x")]), reply("ok")), store=False)
    assert tools_env["stores"] == [False]


# ── markers ──────────────────────────────────────────────────────────────────

def test_markers_stay_unique_across_calls(tools_env):
    script = Script(reply(calls=[call("web_search", "a"), call("web_search", "b")]),
                    reply("ok"))
    _, trace = run(script)
    tool_msgs = [m["content"] for m in script.payloads[1]["messages"] if m["role"] == "tool"]
    assert "[W1] one about a" in tool_msgs[0] and "[W2] two" in tool_msgs[0]
    assert "[W3] one about b" in tool_msgs[1] and "[W4] two" in tool_msgs[1]
    assert "[W1]" not in tool_msgs[1]
    assert trace.markers["W"] == 4


def test_prefetched_library_markers_are_respected(tools_env):
    trace = agent.Trace()
    trace.renumber("[L1] a\n[L2] b")          # the prefetched memory block
    script = Script(reply(calls=[call("library_search", "rhythm")]), reply("ok"))
    run(script, trace=trace)
    tool_msg = script.payloads[1]["messages"][-1]["content"]
    assert tool_msg.startswith("[L3]")
    assert trace.markers["L"] == 3


def test_renumber_maps_repeated_markers_consistently():
    trace = agent.Trace(markers={"W": 0, "X": 0, "A": 0, "L": 4})
    out = trace.renumber("QUERY TOPICS: all in [L1], [L2]\n[L1] x\n[L2] y")
    assert out == "QUERY TOPICS: all in [L5], [L6]\n[L5] x\n[L6] y"


def test_library_search_is_relevance_gated(tools_env, monkeypatch):
    async def junk(query, *_a, **_kw):
        return [{"score": 0.55, "text": "unrelated", "title": "", "identifier": ""}]
    monkeypatch.setattr(mem, "retrieve", junk)
    script = Script(reply(calls=[call("library_search", "melville")]), reply("ok"))
    _, trace = run(script)
    assert script.payloads[1]["messages"][-1]["content"].startswith("No relevant")
    assert trace.library_chunks == []


# ── streaming ────────────────────────────────────────────────────────────────

class StreamScript:
    def __init__(self, *rounds):
        self.rounds = list(rounds)      # each: list of raw Ollama chunks
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(payload)
        chunks = self.rounds.pop(0)

        async def gen():
            for c in chunks:
                yield c
        return gen()


def chunk(content="", calls=None, done=False, done_reason=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    out = {"message": msg, "done": done}
    if done_reason:
        out["done_reason"] = done_reason
    return out


def collect_stream(script, tokens=None):
    async def go():
        trace = agent.Trace()
        events = [e async for e in agent.stream(USER, BASE, TOOLS, store=True, trace=trace,
                                                on_token=(tokens.append if tokens is not None else None),
                                                stream_fn=script)]
        return events, trace
    return asyncio.run(go())


def test_stream_direct_answer_forwards_tokens(tools_env):
    tokens: list[str] = []
    script = StreamScript([chunk("Mel"), chunk("ville"), chunk(done=True, done_reason="stop")])
    events, trace = collect_stream(script, tokens)
    assert [e["text"] for e in events if e["type"] == "content"] == ["Mel", "ville"]
    assert events[-1] == {"type": "done", "done_reason": "stop"}
    assert tokens == ["Mel", "ville"]
    assert script.payloads[0]["stream"] is True and script.payloads[0]["tools"] == TOOLS


def test_stream_tool_round_then_answer(tools_env):
    script = StreamScript(
        [chunk(calls=[call("web_search", "gold")]), chunk(done=True)],
        [chunk("$4,700 [W1]"), chunk(done=True, done_reason="stop")],
    )
    events, trace = collect_stream(script)
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "content", "done"]
    assert events[0] == {"type": "tool_call", "name": "web_search", "query": "gold"}
    assert tools_env["searches"] == ["gold"]
    assert script.payloads[1]["messages"][-1]["role"] == "tool"


def test_stream_last_round_has_no_tools(tools_env):
    rounds = [[chunk(calls=[call("web_search", f"q{i}")]), chunk(done=True)]
              for i in range(agent.MAX_ROUNDS)]
    script = StreamScript(*rounds)
    events, _ = collect_stream(script)
    assert "tools" not in script.payloads[-1]
    assert events[-1]["type"] == "done"


# ── deterministic pre-search ─────────────────────────────────────────────────

def test_prefetch_web_numbers_and_dedupes_with_the_loop(tools_env):
    trace = agent.Trace()
    block = asyncio.run(agent.prefetch_web("gold price today", trace, store=False))
    assert block.startswith("[W1]")
    assert tools_env["searches"] == ["gold price today"]
    assert tools_env["stores"] == [False]
    # the model asking for the same search is treated as a duplicate
    script = Script(reply(calls=[call("web_search", "Gold price today")]), reply("ok"))
    run(script, trace=trace)
    assert tools_env["searches"] == ["gold price today"]
    assert [c["status"] for c in trace.calls] == ["ok", "duplicate"]


def test_prefetch_web_empty_results_return_none(tools_env, monkeypatch):
    async def empty(query, **_kw):
        return {"context_text": "", "results": [], "error": "No search results returned"}
    monkeypatch.setattr(search, "search_and_ingest", empty)
    trace = agent.Trace()
    assert asyncio.run(agent.prefetch_web("x", trace, store=True)) is None


def test_guidance_warns_library_may_be_stale():
    assert "out of date" in agent.TOOL_GUIDANCE


# ── fix: tool results carry the citation rules ───────────────────────────────

def test_web_tool_result_includes_citation_rules(tools_env):
    script = Script(reply(calls=[call("web_search", "weather chicago")]), reply("ok"))
    run(script)
    content = script.payloads[1]["messages"][-1]["content"]
    assert "Cite every fact" in content
    assert "[W1] one about weather chicago" in content


def test_prefetch_web_returns_the_bare_block(tools_env):
    """_inject_context wraps pre-searched blocks itself — no double rules."""
    block = asyncio.run(agent.prefetch_web("x", agent.Trace(), store=True))
    assert block.startswith("[W1]") and "Cite every fact" not in block


# ── citation repair (deterministic check → one retry → strip) ────────────────

MARKERS = {"W": 5, "X": 0, "A": 0, "L": 2}


def test_invalid_citations_finds_singles_and_groups():
    text = "a [W1] b [W8] c [L2, L3] d [W5; X1] e [W1]"
    assert agent.invalid_citations(text, MARKERS) == ["L3", "W8", "X1"]
    assert agent.invalid_citations("no refs", MARKERS) == []
    assert agent.invalid_citations("[W0]", MARKERS) == ["W0"]


def test_strip_citations_removes_only_the_invalid():
    text = "Rates fell [W8]. Held steady [W1, W8]. Also [L3; L1]."
    out = agent.strip_citations(text, ["W8", "L3"])
    assert out == "Rates fell. Held steady [W1]. Also [L1]."


def test_repair_prompt_lists_invalid_and_available():
    p = agent.repair_prompt(["W8"], MARKERS)
    assert "[W8]" in p and "[W1]–[W5]" in p and "[L1]–[L2]" in p
    assert "X" not in p.split("available:")[-1].split("\n")[0]
    none = agent.repair_prompt(["W1"], {"W": 0, "X": 0, "A": 0, "L": 0})
    assert "no sources" in none


def _searching_script(final, *extra):
    return Script(reply(calls=[call("web_search", "fed")]), reply(final), *extra)


def test_valid_answer_is_not_repaired(tools_env):
    script = _searching_script("Cut [W1].")
    data, trace = run(script)
    assert len(script.payloads) == 2
    assert trace.repair is None
    assert data["message"]["content"] == "Cut [W1]."


def test_invalid_citation_triggers_one_repair_call(tools_env):
    script = _searching_script("Cut [W8].", reply("Cut [W2]."))
    data, trace = run(script)
    assert data["message"]["content"] == "Cut [W2]."
    repair_payload = script.payloads[2]
    assert "tools" not in repair_payload
    msgs = repair_payload["messages"]
    assert msgs[-2] == {"role": "assistant", "content": "Cut [W8]."}
    assert "[W8]" in msgs[-1]["content"] and msgs[-1]["role"] == "user"
    assert trace.repair == {"kind": "invalid", "invalid": ["W8"], "resolved": True,
                            "stripped": []}


def test_failed_repair_falls_back_to_stripping(tools_env):
    script = _searching_script("Cut [W8].", reply("Still [W9] and [W1]."))
    data, trace = run(script)
    assert data["message"]["content"] == "Still and [W1]."
    assert trace.repair == {"kind": "invalid", "invalid": ["W8"], "resolved": False,
                            "stripped": ["W9"]}


def test_empty_repair_reply_keeps_stripped_original(tools_env):
    script = _searching_script("Cut [W8] today [W1].", reply("   "))
    data, trace = run(script)
    assert data["message"]["content"] == "Cut today [W1]."
    assert trace.repair["resolved"] is False


def test_repair_call_failure_falls_back_to_stripping(tools_env):
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return reply(calls=[call("web_search", "fed")])
        if calls["n"] == 2:
            return reply("Cut [W8].")
        raise TimeoutError("slow")
    data, trace = asyncio.run(agent.run(USER, BASE, TOOLS, store=True, chat=flaky))
    assert data["message"]["content"] == "Cut."
    assert trace.repair["resolved"] is False


def test_stream_repair_emits_replace_event(tools_env):
    script = StreamScript(
        [chunk(calls=[call("web_search", "fed")]), chunk(done=True)],
        [chunk("Cut [W8]."), chunk(done=True, done_reason="stop")],
    )
    fixer = Script(reply("Cut [W1]."))

    async def go():
        trace = agent.Trace()
        evs = [e async for e in agent.stream(USER, BASE, TOOLS, store=True, trace=trace,
                                             stream_fn=script, chat=fixer)]
        return evs, trace
    events, trace = asyncio.run(go())
    assert [e["type"] for e in events] == ["tool_call", "content", "replace", "done"]
    assert events[2] == {"type": "replace", "text": "Cut [W1].", "kind": "invalid",
                         "invalid": ["W8"]}
    assert "tools" not in fixer.payloads[0]
    assert trace.repair["resolved"] is True


def test_stream_valid_answer_has_no_replace(tools_env):
    script = StreamScript([chunk("Melville."), chunk(done=True, done_reason="stop")])
    fixer = Script()

    async def go():
        return [e async for e in agent.stream(USER, BASE, TOOLS, store=True,
                                              trace=agent.Trace(), stream_fn=script,
                                              chat=fixer)]
    events = asyncio.run(go())
    assert [e["type"] for e in events] == ["content", "done"]
    assert fixer.payloads == []


# ── citation repair: web results shown but nothing cited ─────────────────────

def test_missing_citations_trigger_one_retry(tools_env):
    script = _searching_script("It is overcast and 61°F.", reply("It is overcast [W1] and 61°F [W2]."))
    data, trace = run(script)
    assert data["message"]["content"] == "It is overcast [W1] and 61°F [W2]."
    prompt = script.payloads[2]["messages"][-1]["content"]
    assert "cites none" in prompt and "[W1]–[W2]" in prompt
    assert "tools" not in script.payloads[2]
    assert trace.repair == {"kind": "missing", "invalid": [], "resolved": True, "stripped": []}


def test_missing_retry_without_citations_keeps_original(tools_env):
    script = _searching_script("It is overcast.", reply("It is cloudy today."))
    data, trace = run(script)
    assert data["message"]["content"] == "It is overcast."
    assert trace.repair == {"kind": "missing", "invalid": [], "resolved": False, "stripped": []}


def test_missing_retry_with_bad_marker_is_stripped_but_kept(tools_env):
    script = _searching_script("Overcast.", reply("Overcast [W1] and windy [W9]."))
    data, trace = run(script)
    assert data["message"]["content"] == "Overcast [W1] and windy."
    assert trace.repair == {"kind": "missing", "invalid": [], "resolved": True,
                            "stripped": ["W9"]}


def test_missing_retry_failure_keeps_original(tools_env):
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return reply(calls=[call("web_search", "weather")])
        if calls["n"] == 2:
            return reply("Overcast.")
        raise TimeoutError("slow")
    data, trace = asyncio.run(agent.run(USER, BASE, TOOLS, store=True, chat=flaky))
    assert data["message"]["content"] == "Overcast."
    assert trace.repair["resolved"] is False


def test_library_only_answers_need_no_citation(tools_env):
    trace = agent.Trace()
    trace.renumber("[L1] prefetched memory")
    script = Script(reply("Melville wrote it."))
    data, trace = run(script, trace=trace)
    assert len(script.payloads) == 1 and trace.repair is None


def test_no_search_no_citation_check(tools_env):
    script = Script(reply("391"))
    _, trace = run(script)
    assert len(script.payloads) == 1 and trace.repair is None


def test_empty_answer_is_not_sent_for_citation_repair(tools_env):
    script = _searching_script("   ")
    _, trace = run(script)
    assert len(script.payloads) == 2 and trace.repair is None


def test_stream_missing_citation_replace_event(tools_env):
    script = StreamScript(
        [chunk(calls=[call("web_search", "weather")]), chunk(done=True)],
        [chunk("Overcast."), chunk(done=True, done_reason="stop")],
    )
    fixer = Script(reply("Overcast [W1]."))

    async def go():
        return [e async for e in agent.stream(USER, BASE, TOOLS, store=True,
                                              trace=agent.Trace(), stream_fn=script,
                                              chat=fixer)]
    events = asyncio.run(go())
    assert events[2] == {"type": "replace", "text": "Overcast [W1].", "kind": "missing",
                         "invalid": []}


def test_stream_unresolved_missing_sends_no_replace(tools_env):
    script = StreamScript(
        [chunk(calls=[call("web_search", "weather")]), chunk(done=True)],
        [chunk("Overcast."), chunk(done=True, done_reason="stop")],
    )
    fixer = Script(reply("Still no refs."))

    async def go():
        return [e async for e in agent.stream(USER, BASE, TOOLS, store=True,
                                              trace=agent.Trace(), stream_fn=script,
                                              chat=fixer)]
    assert [e["type"] for e in asyncio.run(go())] == ["tool_call", "content", "done"]


def test_padded_brackets_count_as_citations(tools_env):
    """The model sometimes writes "[ X3 ]" — that is a real citation."""
    assert agent.invalid_citations("see [ W8 ] and [ W1 , W2 ]", MARKERS) == ["W8"]
    script = _searching_script("Ship 41 finished testing [ W1 ].")
    data, trace = run(script)
    assert len(script.payloads) == 2 and trace.repair is None
    assert agent.strip_citations("a [ W8 ] b [ W1 , W8 ].", ["W8"]) == "a b [W1]."

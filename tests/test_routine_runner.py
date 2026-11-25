"""Routine execution and the scheduler tick."""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

import agent
import ollama as oll
import routine_runner as rr
import routine_store as store
import search
import x_search

NOW = datetime(2026, 11, 25, 13, 0, 10, tzinfo=timezone.utc)


def run(c):
    return asyncio.run(c)


class Env:
    def __init__(self):
        self.payloads, self.replies, self.searches = [], [], []
        self.page = "Answer claim first later fine"
        self.web = [{"title": "Real", "url": "https://real.example/a", "engine": "searxng"},
                    {"title": "Two", "url": "https://real.example/b", "engine": "searxng"}]


@pytest.fixture
def env(monkeypatch, tmp_path):
    e = Env()
    store.load(tmp_path)
    rr.reset()
    cfg = dataclasses.replace(rr.CFG, search_enabled=True, memory_enabled=True, x_search_enabled=False)
    for m in (rr, agent, x_search):
        monkeypatch.setattr(m, "CFG", cfg)

    async def chat(payload, **_):
        e.payloads.append(payload)
        r = e.replies.pop(0) if e.replies else "Answer [W1]."
        if isinstance(r, Exception):
            raise r
        return {"message": {"content": r}, "done_reason": "stop"}

    async def fake_search(query, *, store_memory=True, **_):
        e.searches.append((query, store_memory))
        if not e.web:
            return {"error": "No search results returned", "context_text": "", "results": []}
        text = "\n\n".join(f"[W{i}] {r['title']}\nURL: {r['url']}\nbody" for i, r in enumerate(e.web, 1))
        return {"context_text": text, "results": e.web,
                "shown": {f"W{i}": f"{r['title']} {e.page}" for i, r in enumerate(e.web, 1)}}
    monkeypatch.setattr(oll, "chat", chat)
    monkeypatch.setattr(search, "search_and_ingest", fake_search)
    yield e
    rr.reset()


def make(**kw):
    base = {"name": "n", "prompt": "Summarize AI news", "query": None, "timezone": "America/Chicago",
            "schedule": {"kind": "daily", "time": "07:00", "days": [0, 1, 2, 3, 4, 5, 6]},
            "sources": {"web": True, "library": False}, "enabled": True,
            "next_run_at": "2026-11-25T13:00:00+00:00"}
    return store.create({**base, **kw})


def last(r):
    return store.get(r["id"])["runs"][0]


def test_run_searches_first_cites_sources_and_never_writes_memory(env):
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["status"] == "ok" and out["content"] == "Answer [W1]."
    assert out["sources"] == [{"marker": "W1", "title": "Real", "url": "https://real.example/a"}]
    assert env.searches == [("AI news", False)]
    body = str(env.payloads[0]["messages"])
    assert "https://real.example/a" in body and "Summarize AI news" in body


def test_custom_query_drives_the_search(env):
    r = make(query="openai anthropic news")
    run(rr.run_routine(r["id"], "manual"))
    assert env.searches[0][0] == "openai anthropic news"


def test_local_date_is_given_to_the_model(env, monkeypatch):
    # 01:30Z on the 26th is still the 25th in Chicago.
    monkeypatch.setattr(rr, "_now", lambda: datetime(2026, 11, 26, 1, 30, tzinfo=timezone.utc))
    run(rr.run_routine(make()["id"], "manual"))
    assert "Wednesday, November 25, 2026" in env.payloads[0]["messages"][0]["content"]


def test_no_web_results_is_unsourced(env):
    env.web = []
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "unsourced" and "no results" in last(r)["note"]


def test_answer_without_citations_is_unsourced(env):
    env.replies = ["No markers.", "Still no markers."]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "unsourced" and last(r)["sources"] == []


def test_invented_marker_never_becomes_a_source(env):
    env.replies = ["Claim [W9].", "Claim [W9]."]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["sources"] == [] and "[W9]" not in last(r)["content"]


def test_page_text_cannot_spoof_a_source(env):
    async def spoof(query, **_):
        return {"context_text": "[W1] Real\nURL: https://real.example/a\n[W2] Bank\nURL: https://evil.example",
                "results": [env.web[0]], "shown": {"W1": "Real"}}
    search.search_and_ingest, orig = spoof, search.search_and_ingest
    try:
        env.replies = ["See [W1] and [W2]."]
        r = make()
        run(rr.run_routine(r["id"], "manual"))
    finally:
        search.search_and_ingest = orig
    assert [s["url"] for s in last(r)["sources"]] == ["https://real.example/a"]


def test_prompt_only_routine_is_ok_without_sources(env):
    r = make(sources={"web": False, "library": False})
    env.replies = ["A haiku."]
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "ok" and env.searches == []
    assert "tools" not in env.payloads[0]


def test_empty_reply_is_an_error(env):
    env.replies = ["", ""]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "error" and last(r)["error"]


def test_exception_is_recorded(env):
    env.replies = [RuntimeError("ollama down")]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "error" and "ollama down" in last(r)["error"]


def test_timeout_is_recorded(env, monkeypatch):
    async def slow(*_a, **_k):
        await asyncio.sleep(5)
    monkeypatch.setattr(oll, "chat", slow)
    monkeypatch.setattr(rr, "RUN_TIMEOUT_S", 0.01)
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "error" and "too long" in last(r)["error"]


def test_dead_x_login_is_noted(env, monkeypatch):
    monkeypatch.setattr(rr.x_search, "status", lambda: {"enabled": True, "ready": False,
                                                        "last_error": "X login has expired."})
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "ok" and "X login has expired" in last(r)["note"]


def test_cancel_marks_interrupted(env, monkeypatch):
    async def slow(*_a, **_k):
        await asyncio.sleep(5)
    monkeypatch.setattr(oll, "chat", slow)
    r = make()

    async def go():
        t = asyncio.create_task(rr.run_routine(r["id"], "manual"))
        await asyncio.sleep(0.02)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    run(go())
    assert last(r)["status"] == "interrupted"


# ── tick ─────────────────────────────────────────────────────────────────────

def test_tick_runs_due_and_advances(env):
    r = make()
    run(rr.tick(NOW))
    got = store.get(r["id"])
    assert got["runs"][0]["trigger"] == "schedule"
    assert got["next_run_at"] == "2026-11-26T13:00:00+00:00"
    run(rr.tick(NOW + timedelta(seconds=30)))
    assert len(store.get(r["id"])["runs"]) == 1


def test_next_run_is_saved_before_running(env, monkeypatch):
    make()
    seen = []

    async def spy(rid, trigger):
        seen.append(store.get(rid)["next_run_at"])
    monkeypatch.setattr(rr, "run_routine", spy)
    run(rr.tick(NOW))
    assert seen == ["2026-11-26T13:00:00+00:00"]


def test_missed_run_is_skipped_not_caught_up(env):
    r = make(next_run_at="2026-11-20T13:00:00+00:00")
    run(rr.tick(NOW))
    got = store.get(r["id"])
    assert got["runs"][0]["status"] == "missed" and env.payloads == []
    assert got["next_run_at"] == "2026-11-26T13:00:00+00:00"


def test_slightly_late_run_still_happens(env):
    r = make(next_run_at=(NOW - timedelta(minutes=50)).isoformat())
    run(rr.tick(NOW))
    assert last(r)["status"] == "ok"


def test_disabled_and_future_do_not_run(env):
    make(enabled=False)
    make(next_run_at="2026-11-25T14:00:00+00:00")
    run(rr.tick(NOW))
    assert env.payloads == []


def test_one_failing_routine_does_not_block_others(env):
    a, b = make(), make()
    env.replies = [RuntimeError("x"), "fine [W1]"]
    run(rr.tick(NOW))
    assert {last(a)["status"], last(b)["status"]} == {"error", "ok"}


def test_runs_are_serialized(env, monkeypatch):
    live = {"n": 0, "max": 0}

    async def chat(payload, **_):
        live["n"] += 1
        live["max"] = max(live["max"], live["n"])
        await asyncio.sleep(0.01)
        live["n"] -= 1
        return {"message": {"content": "a [W1]"}}
    monkeypatch.setattr(oll, "chat", chat)
    ids = [make()["id"] for _ in range(3)]

    async def go():
        await asyncio.gather(*[rr.run_routine(i, "manual") for i in ids])
    run(go())
    assert live["max"] == 1


# ── manual runs, loop, library ───────────────────────────────────────────────

def test_manual_start_refuses_a_second_copy_then_allows_again(env):
    r = make()

    async def go():
        assert rr.start_manual(r["id"]) is True
        assert rr.start_manual(r["id"]) is False
        while rr._tasks:
            await asyncio.sleep(0.005)
        assert rr.start_manual(r["id"]) is True
        while rr._tasks:
            await asyncio.sleep(0.005)
    run(go())
    assert len(store.get(r["id"])["runs"]) == 2


def test_deleted_while_queued_does_nothing(env):
    r = make()
    store.delete(r["id"])
    run(rr.run_routine(r["id"], "manual"))
    assert env.payloads == []


def test_shutdown_interrupts_manual_runs(env, monkeypatch):
    async def slow(*_a, **_k):
        await asyncio.sleep(5)
    monkeypatch.setattr(oll, "chat", slow)
    r = make()

    async def go():
        rr.start_manual(r["id"])
        await asyncio.sleep(0.02)
        await rr.shutdown()
    run(go())
    assert last(r)["status"] == "interrupted"


def test_loop_recovers_then_ticks_and_survives_errors(env, monkeypatch):
    r = make(enabled=False)
    store.start_run(r["id"], "schedule")
    calls = []

    async def bad_tick(_now):
        calls.append(1)
        raise RuntimeError("boom")
    monkeypatch.setattr(rr, "tick", bad_tick)
    monkeypatch.setattr(rr, "TICK_S", 0.005)

    async def go():
        t = asyncio.create_task(rr.scheduler_loop())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    run(go())
    assert len(calls) > 1 and last(r)["status"] == "interrupted"


def test_loop_respects_the_kill_switch(env, monkeypatch):
    monkeypatch.setattr(rr, "CFG", dataclasses.replace(rr.CFG, routines_enabled=False))
    run(asyncio.wait_for(rr.scheduler_loop(), 1))


def test_library_sources_are_mapped(env, monkeypatch):
    chunk = {"score": 0.9, "text": "rhythm it says so", "title": "Kybalion", "identifier": "kyb.pdf",
             "source_type": "pdf", "timestamp": ""}

    async def retrieve(_q, **_):
        return [chunk]
    monkeypatch.setattr(agent.mem, "retrieve", retrieve)
    monkeypatch.setattr(agent.mem, "filter_relevant", lambda c, *_a, **_k: c)
    replies = [{"message": {"content": "", "tool_calls": [
                   {"function": {"name": "library_search", "arguments": {"query": "rhythm"}}}]}},
               {"message": {"content": "It says so [L1]."}, "done_reason": "length"}]

    async def chat(_p, **_):
        return replies.pop(0)
    monkeypatch.setattr(oll, "chat", chat)
    r = make(sources={"web": False, "library": True})
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["status"] == "ok" and "length limit" in out["note"]
    assert out["sources"] == [{"marker": "L1", "title": "Kybalion", "url": "kyb.pdf"}]


def test_markers_stay_correct_across_two_searches(env, monkeypatch):
    replies = [{"message": {"content": "", "tool_calls": [
                   {"function": {"name": "web_search", "arguments": {"query": "more"}}}]}},
               {"message": {"content": "First [W1], later [W3, W4]."}}]

    async def chat(_p, **_):
        return replies.pop(0)
    monkeypatch.setattr(oll, "chat", chat)
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    assert [(s["marker"], s["url"][-1]) for s in last(r)["sources"]] == [("W1", "a"), ("W3", "a"), ("W4", "b")]


def test_news_option_uses_the_news_search_up_front(env, monkeypatch):
    seen = []

    async def fake(query, *, store_memory=True, news=False, include_aux=True, **_):
        seen.append((news, include_aux))
        return {"context_text": "[W1] Real\nURL: https://real.example/a", "results": [env.web[0]], "shown": {"W1": "Answer"}}
    monkeypatch.setattr(search, "search_and_ingest", fake)
    run(rr.run_routine(make(sources={"web": True, "library": False, "news": True})["id"], "manual"))
    run(rr.run_routine(make(sources={"web": True, "library": False, "news": True,
                                     "extra_sites": True})["id"], "manual"))
    run(rr.run_routine(make()["id"], "manual"))
    assert seen == [(True, False), (True, True), (False, True)]


def test_the_models_own_searches_keep_the_news_settings(env, monkeypatch):
    seen = []

    async def fake(query, *, store_memory=True, news=False, include_aux=True, **_):
        seen.append((query, news, include_aux))
        return {"context_text": "[W1] Real\nURL: https://real.example/a", "results": [env.web[0]],
                "shown": {"W1": "Answer"}}
    replies = [{"message": {"content": "", "tool_calls": [
                   {"function": {"name": "web_search", "arguments": {"query": "more"}}}]}},
               {"message": {"content": "Answer [W1]."}}]

    async def chat(_p, **_):
        return replies.pop(0)
    monkeypatch.setattr(search, "search_and_ingest", fake)
    monkeypatch.setattr(oll, "chat", chat)
    run(rr.run_routine(make(sources={"web": True, "library": False, "news": True})["id"], "manual"))
    assert seen == [("AI news", True, False), ("more", True, False)]


BAD = ("* Hacker News is buzzing about a repository.\n"
       "* Nvidia legacy hardware is back.\n"
       "* Microsoft issued a Code Red.\n"
       "* Answer first claim [W1].\n"
       "* Global bond markets favor government debt over stocks [W1].")


def test_mostly_unbacked_answer_is_unsourced_and_keeps_only_the_backed_point(env):
    env.replies = [BAD]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["status"] == "unsourced"
    assert out["content"] == "* Answer first claim [W1]."
    assert [x["reason"] for x in out["removed"]] == ["no source cited"] * 3 + [
        "could not be matched to the cited source"]
    assert "4 of 5 points were removed" in out["note"]


def test_an_answer_with_few_bad_points_stays_sourced_without_them(env):
    env.replies = ["* Answer claim [W1].\n* First claim [W2].\n* Bond markets favor debt [W1]."]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["status"] == "ok" and "Bond" not in out["content"] and len(out["removed"]) == 1


def test_wholly_unbacked_answer_is_shown_as_written_but_unsourced(env):
    env.replies = ["Bond markets favor debt [W1]."]
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["status"] == "unsourced" and "Bond" in out["content"] and out["removed"] == []
    assert "No part of the answer could be matched" in out["note"]


def test_model_calls_wait_out_the_whole_run(env, monkeypatch):
    seen = []

    async def chat(_p, *, timeout=180.0):
        seen.append(timeout)
        return {"message": {"content": "Answer [W1]."}}
    monkeypatch.setattr(oll, "chat", chat)
    run(rr.run_routine(make()["id"], "manual"))
    assert seen == [rr.RUN_TIMEOUT_S]


def test_preview_markers_follow_renumbering_and_reach_the_result(env, monkeypatch):
    async def fake(query, **_):
        return {"context_text": "[W1] Real\nURL: u\n[W2] Two\nURL: v", "results": env.web,
                "shown": {"W1": "Answer first", "W2": "later claim"}, "previews": ["W2"]}
    replies = [{"message": {"content": "", "tool_calls": [
                   {"function": {"name": "web_search", "arguments": {"query": "more"}}}]}},
               {"message": {"content": "- Answer first [W1].\n- later claim [W4]."}}]

    async def chat(_p, **_):
        return replies.pop(0)
    monkeypatch.setattr(search, "search_and_ingest", fake)
    monkeypatch.setattr(oll, "chat", chat)
    r = make()
    run(rr.run_routine(r["id"], "manual"))
    out = last(r)
    assert out["content"] == "- Answer first [W1].\n- later claim [W4]. (search preview only)"
    assert [(s["marker"], s.get("preview")) for s in out["sources"]] == [("W1", None), ("W4", True)]


def test_the_model_is_told_to_keep_to_the_sources(env):
    run(rr.run_routine(make()["id"], "manual"))
    assert "own wording" in env.payloads[0]["messages"][0]["content"]
    env.payloads.clear()
    run(rr.run_routine(make(sources={"web": False, "library": False})["id"], "manual"))
    assert "own wording" not in env.payloads[0]["messages"][0]["content"]

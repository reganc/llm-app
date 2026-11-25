"""Routines that report only what is new."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

import agent
import ollama as oll
import routine_runner as rr
import routine_store as store
import search
import x_search


def run(c):
    return asyncio.run(c)


class Env:
    def __init__(self):
        self.payloads, self.replies, self.kw, self.sent = [], [], [], []
        self.web = ["a", "b"]
        self.page = "News Nvidia fell story. Nothing new here but prices rose. Gold is $4,000"


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
        return {"message": {"content": e.replies.pop(0) if e.replies else "News [W1]."}}

    async def fake_search(query, *, exclude_urls=frozenset(), exclude_titles=(), **kw):
        e.kw.append({**kw, "exclude_urls": exclude_urls, "exclude_titles": exclude_titles})
        left = [u for u in e.web if search.norm_url(f"https://n.example/{u}") not in exclude_urls]
        if not left:
            return {"error": "No new results", "context_text": "", "results": [], "all_seen": bool(e.web)}
        return {"context_text": "\n\n".join(f"[W{i}] T{u}\nURL: https://n.example/{u}" for i, u in enumerate(left, 1)),
                "results": [{"title": f"T{u}", "url": f"https://n.example/{u}", "engine": "bing news"} for u in left],
                "shown": {f"W{i}": f"T{u} {e.page}" for i, u in enumerate(left, 1)}}

    async def send(text):
        e.sent.append(text)
    monkeypatch.setattr(oll, "chat", chat)
    monkeypatch.setattr(search, "search_and_ingest", fake_search)
    monkeypatch.setattr(rr.notify, "send", send)
    yield e
    rr.reset()


def make(**kw):
    base = {"name": "n", "prompt": "Summarize AI news", "query": None, "timezone": "UTC",
            "schedule": {"kind": "interval", "hours": 1}, "enabled": True, "next_run_at": None,
            "sources": {"web": True, "library": False, "news": True}, "only_new": True, "notify": "headline"}
    return store.create({**base, **kw})["id"]


def go(rid, trigger="schedule"):
    run(rr.run_routine(rid, trigger))
    return store.get(rid)["runs"][0]


def test_reported_stories_are_excluded_next_time(env):
    rid = make()
    go(rid)
    go(rid)
    assert env.kw[0]["exclude_urls"] == frozenset()
    assert env.kw[1]["exclude_urls"] == {search.norm_url("https://n.example/a")}


def test_news_never_shows_the_model_its_earlier_reports(env):
    # Shown them, it repeated an old story under a new, wrong citation.
    rid = make()
    env.replies = ["Nvidia fell [W1]."]
    go(rid)
    go(rid)
    text = str(env.payloads[1]["messages"])
    assert "Nvidia fell" not in text and "NOTHING_NEW" in text


def test_same_story_under_another_url_is_excluded_by_its_headline(env):
    rid = make()
    go(rid)
    go(rid)
    assert env.kw[0]["exclude_titles"] == () and env.kw[1]["exclude_titles"] == ("Ta",)


def test_earlier_reports_reach_the_model_without_old_markers(env):
    rid = make(sources={"web": True, "library": False, "news": False})
    env.replies = ["Nvidia fell [W1]."]
    go(rid)
    go(rid)
    text = str(env.payloads[1]["messages"])
    assert "Nvidia fell" in text and "Nvidia fell [W1]" not in text and "NOTHING_NEW" in text
    assert "NOTHING_NEW" not in str(env.payloads[0]["messages"])   # nothing to compare with yet


def test_all_seen_is_quiet_and_skips_the_model(env):
    rid = make()
    env.web = ["a"]
    go(rid)
    env.sent.clear()
    n = len(env.payloads)
    out = go(rid)
    assert out["status"] == "unchanged" and out["read"] is True
    assert len(env.payloads) == n and env.sent == [] and store.unread_count() == 1


@pytest.mark.parametrize("reply", ["NOTHING_NEW", "nothing_new.", "**NOTHING_NEW**", " NOTHING_NEW\n",
                                   "Nothing new."])
def test_model_can_say_nothing_new(env, reply):
    rid = make()
    go(rid)
    env.replies = [reply]
    n = len(env.payloads)
    out = go(rid)
    assert out["status"] == "unchanged" and len(env.payloads) == n + 1   # no citation retry


def test_the_word_inside_a_real_answer_is_not_nothing_new(env):
    rid = make()
    go(rid)
    env.page = "OpenAI shipped a new model"
    env.replies = ["NOTHING_NEW on Nvidia, but OpenAI shipped a model [W1]."]
    assert go(rid)["status"] == "ok"


def test_first_run_cannot_be_unchanged(env):
    env.replies = ["NOTHING_NEW", "NOTHING_NEW"]
    assert go(make())["status"] != "unchanged"


def test_no_results_at_all_is_unsourced_not_unchanged(env):
    rid = make()
    go(rid)
    env.web = []
    assert go(rid)["status"] == "unsourced"


def test_failed_or_unsourced_runs_are_not_remembered(env):
    rid = make()
    env.replies = ["No markers", "No markers"]
    go(rid)
    go(rid)
    assert env.kw[1]["exclude_urls"] == frozenset()


def test_general_web_pages_are_not_excluded_but_reports_are_shown(env):
    # A price page keeps one URL while its content changes.
    rid = make(sources={"web": True, "library": False, "news": False})
    env.replies = ["Gold is $4,000 [W1]."]
    go(rid)
    go(rid)
    assert env.kw[1]["exclude_urls"] == frozenset()
    assert "Gold is $4,000" in str(env.payloads[1]["messages"])


def test_off_means_no_memory_in_the_prompt(env):
    rid = make(only_new=False)
    go(rid)
    go(rid)
    assert env.kw[1]["exclude_urls"] == frozenset()
    assert "NOTHING_NEW" not in str(env.payloads[1]["messages"])


def test_memory_is_bounded_and_survives_history_pruning(env, monkeypatch):
    monkeypatch.setattr(store, "MAX_RUNS", 2)
    monkeypatch.setattr(rr, "_MEMORY_URLS", 3)
    rid = make()
    for u in "abcdef":
        env.web = [u]
        go(rid)
    mem = store.get(rid)["memory"]
    assert [u[-1] for u in mem["urls"]] == ["d", "e", "f"] and len(mem["reports"]) == rr._MEMORY_REPORTS


def test_reset_memory(env):
    rid = make()
    go(rid)
    store.update(rid, {"memory": None})
    go(rid)
    assert env.kw[1]["exclude_urls"] == frozenset()


def test_routine_changed_during_a_run_does_not_get_old_memory(env, monkeypatch):
    rid = make()

    async def chat(payload, **_):
        store.update(rid, {"prompt": "Different", "memory": None})
        return {"message": {"content": "Old topic [W1]."}}
    monkeypatch.setattr(oll, "chat", chat)
    go(rid)
    assert not store.get(rid).get("memory")


# ── notifications from the runner ────────────────────────────────────────────

def test_scheduled_run_notifies_and_records_it(env):
    out = go(make())
    assert len(env.sent) == 1 and out["notified"] == "sent"


def test_manual_run_does_not_notify(env):
    out = go(make(), "manual")
    assert env.sent == [] and "notified" not in out


def test_notification_failure_does_not_fail_the_run(env, monkeypatch):
    async def boom(_t):
        raise rr.notify.NotifyError("Telegram refused the token.")
    monkeypatch.setattr(rr.notify, "send", boom)
    out = go(make())
    assert out["status"] == "ok" and out["notified"] == "failed: Telegram refused the token."


def test_unexpected_notification_error_is_contained(env, monkeypatch):
    async def boom(_t):
        raise ValueError("https://api.telegram.org/bot1:secret/x")
    monkeypatch.setattr(rr.notify, "send", boom)
    out = go(make())
    assert out["status"] == "ok" and "secret" not in out["notified"]


def test_repeated_errors_notify_once(env, monkeypatch):
    async def chat(_p, **_):
        raise RuntimeError("down")
    monkeypatch.setattr(oll, "chat", chat)
    rid = make()
    go(rid)
    go(rid)
    assert len(env.sent) == 1


def test_result_is_saved_before_notifying(env, monkeypatch):
    rid = make()
    seen = []

    async def send(_t):
        seen.append(store.get(rid)["runs"][0]["status"])
    monkeypatch.setattr(rr.notify, "send", send)
    go(rid)
    assert seen == ["ok"]


# ── search filter ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("https://N.example/a/", "http://n.example/a"),
    ("https://n.example/a?utm_source=x&utm_medium=y", "https://n.example/a"),
    ("https://n.example/a#top", "https://n.example/a"),
    ("https://www.n.example/a", "https://n.example/a"),
    ("https://n.example/a?id=1&fbclid=z", "https://n.example/a?id=1"),
])
def test_same_story_urls_match(a, b):
    assert search.norm_url(a) == search.norm_url(b)


def test_different_stories_do_not_match():
    assert search.norm_url("https://n.example/a?id=1") != search.norm_url("https://n.example/a?id=2")
    assert search.norm_url("https://n.example/a") != search.norm_url("https://n.example/b")


def test_search_drops_excluded_urls_before_fetching(monkeypatch):
    fetched = []

    async def news(_q, num):
        return [{"title": t, "url": f"https://n.example/{t}", "snippet": "s" * 50, "engine": "bing news"}
                for t in ("old", "new")][:num]

    async def fetch(url):
        fetched.append(url)

    async def none():
        return []
    monkeypatch.setattr(search, "searxng_news", news)
    monkeypatch.setattr(search, "_fetch_text", fetch)
    monkeypatch.setattr(search, "fetch_aux_sites", none)
    cfg = dataclasses.replace(search.CFG, search_enabled=True, x_search_enabled=False)
    monkeypatch.setattr(search, "CFG", cfg)
    old = frozenset({search.norm_url("https://n.example/old")})
    out = run(search.search_and_ingest("q", store_memory=False, news=True, exclude_urls=old))
    assert fetched == ["https://n.example/new"] and [r["title"] for r in out["results"]] == ["new"]
    both = old | {search.norm_url("https://n.example/new")}
    out = run(search.search_and_ingest("q", store_memory=False, news=True, exclude_urls=both))
    assert out["all_seen"] is True and out["context_text"] == ""


@pytest.mark.parametrize("a,b,same", [
    ("Nvidia Down 8% As Tech Stocks Extend Selloff", "Nvidia shares down 8% as tech stocks extend selloff | Reuters", True),
    ("OpenAI combines ChatGPT and Codex limits", "OpenAI is combining ChatGPT and Codex usage limits - The Verge", True),
    ("Nvidia Down 8% As Tech Stocks Extend Selloff", "Nvidia unveils new data center chip for robots", False),
    ("AI news", "AI news today", False),           # too short to tell
    ("Broadcom vs Qualcomm: which is the better buy", "Broadcom raises full-year revenue outlook", False),
    ("", "Anything at all about this story", False),
])
def test_similar_headlines(a, b, same):
    assert search.same_headline(a, b) is same and search.same_headline(b, a) is same


def test_search_drops_similar_headlines(monkeypatch):
    async def news(_q, _n):
        return [{"title": "Nvidia shares down 8% as tech stocks extend selloff", "url": "https://x.example/1",
                 "snippet": "s" * 50, "engine": "bing news"},
                {"title": "OpenAI ships a brand new coding model", "url": "https://x.example/2",
                 "snippet": "s" * 50, "engine": "bing news"}]

    async def fetch(_u):
        return None

    async def none():
        return []
    monkeypatch.setattr(search, "searxng_news", news)
    monkeypatch.setattr(search, "_fetch_text", fetch)
    monkeypatch.setattr(search, "fetch_aux_sites", none)
    monkeypatch.setattr(search, "CFG", dataclasses.replace(search.CFG, search_enabled=True, x_search_enabled=False))
    out = run(search.search_and_ingest("q", store_memory=False, news=True,
                                       exclude_titles=("Nvidia Down 8% As Tech Stocks Extend Selloff",)))
    assert [r["url"] for r in out["results"]] == ["https://x.example/2"]


def test_model_searches_also_skip_seen_stories(env):
    rid = make()
    go(rid)
    env.web = ["a", "b"]
    replies = [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "web_search", "arguments": {"query": "more"}}}]}},
        {"message": {"content": "x [W1]."}}]

    async def chat(_p, **_):
        return replies.pop(0)
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(oll, "chat", chat)
    try:
        go(rid)
    finally:
        mp.undo()
    assert all(k["exclude_urls"] for k in env.kw[1:]) and len(env.kw) == 3


def test_running_by_hand_compares_but_does_not_use_up_stories(env):
    rid = make()
    go(rid, "manual")
    assert not store.get(rid).get("memory")
    go(rid)
    before = store.get(rid)["memory"]
    go(rid, "manual")
    assert env.kw[2]["exclude_urls"] == {search.norm_url("https://n.example/a")}
    assert store.get(rid)["memory"] == before

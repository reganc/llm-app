"""News search: topic filter, widening, and leaving out the extra sites."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

import search


def item(title, url=None, snippet=""):
    return {"title": title, "url": url or f"https://n.example/{abs(hash(title))}", "snippet": snippet, "engine": "bing news"}


AI = [item("AI trade unravels"), item("New AI rules"), item("Chip maker bets on artificial intelligence")]
ISL = item("ISL points table")


def test_off_topic_stories_are_left_out_when_enough_remain():
    assert search._on_topic([ISL, *AI], "AI news") == AI


def test_nothing_is_dropped_when_few_would_remain():
    assert search._on_topic([ISL, AI[0]], "AI news") == [ISL, AI[0]]


def test_category_words_do_not_empty_the_results():
    stories = [item("Rocket lands"), item("Phone launched"), item("Chip plant opens")]
    assert search._on_topic(stories, "technology news") == stories
    assert search._on_topic(stories, "latest news") == stories


def test_short_words_match_whole_words_only():
    aid = [item("Aid convoy arrives")]
    assert search._on_topic(aid + AI, "AI news") == AI


def test_snippets_count_and_regex_characters_are_safe():
    rows = [item("a", snippet="C++ compiler"), item("b", snippet="C++ 26"), item("c", snippet="about C++"), ISL]
    assert len(search._on_topic(rows, "C++ (news) [x")) >= 3


@pytest.fixture
def engines(monkeypatch):
    replies = {}

    async def query(_client, _q, _engines, num, time_range=None):
        out = replies.get(time_range, [])
        if isinstance(out, Exception):
            raise out
        return out[:num]

    async def general(_q, num):
        return replies.get("general", [])[:num]
    monkeypatch.setattr(search, "_searxng_query", query)
    monkeypatch.setattr(search, "searxng_search", general)
    return replies


def news(q="AI news", num=8):
    return asyncio.run(search.searxng_news(q, num))


FIVE = [*AI, item("Lawmakers debate AI"), item("Schools ban AI tutors")]


def test_enough_stories_today_stop_there(engines):
    engines.update({"day": [ISL, *FIVE], "week": [item("AI last week")], "general": [item("General AI")]})
    assert news() == FIVE


def test_a_thin_day_is_topped_up_from_the_week_with_today_first(engines):
    engines.update({"day": AI, "week": [AI[0], *FIVE[3:]], "general": [item("General AI")]})
    assert news() == FIVE


def test_a_story_syndicated_under_two_urls_appears_once(engines):
    a = item("Nvidia chief says robot factories arrive with AI", "https://a.example/1")
    b = item("Nvidia chief says robot factories arrive with AI - MSN", "https://b.example/2")
    engines.update({"day": [a, b, *FIVE]})
    assert b["url"] not in [r["url"] for r in news()]


def test_stories_in_another_script_are_left_out_when_enough_remain():
    nikkei = item("シンガポール株25日、反落 - 日本経済新聞", snippet="AI")
    assert search._on_topic([nikkei, *AI], "AI news") == AI
    assert search._on_topic([nikkei], "AI news") == [nikkei]
    assert nikkei in search._on_topic([nikkei, *AI], "AI ニュース")
    accented = item("Les élèves et l'AI à l'école")
    assert accented in search._on_topic([accented, *AI], "AI news")


def test_a_single_leftover_story_does_not_stand_alone(engines):
    engines.update({"day": [], "week": [ISL], "general": AI})
    assert news() == AI


def test_general_results_do_not_repeat_a_story(engines):
    engines.update({"week": [AI[0]], "general": [dict(AI[0], url=AI[0]["url"] + "/"), AI[1]]})
    assert [r["title"] for r in news()] == [AI[0]["title"], AI[1]["title"]]


def test_news_engine_failure_falls_back(engines):
    engines.update({"day": RuntimeError("down"), "general": AI})
    assert news() == AI


def test_results_are_capped(engines):
    engines.update({"week": AI[:2], "general": [item(f"AI {i}") for i in range(20)]})
    assert len(news(num=5)) == 5


@pytest.mark.parametrize("include", [True, False])
def test_extra_sites_are_read_only_when_included(monkeypatch, include):
    read = []

    async def aux():
        read.append(1)
        return [{"title": "Extra", "url": "https://x.example", "snippet": "s", "text": "extra text",
                 "char_count": 10, "engine": "auxiliary_site"}]

    async def general(_q, _n):
        return [item("Story", "https://n.example/s", "story body")]

    async def fetch(_url):
        return None
    monkeypatch.setattr(search, "CFG", dataclasses.replace(search.CFG, search_enabled=True, x_search_enabled=False))
    monkeypatch.setattr(search, "fetch_aux_sites", aux)
    monkeypatch.setattr(search, "searxng_search", general)
    monkeypatch.setattr(search, "_fetch_text", fetch)
    out = asyncio.run(search.search_and_ingest("story", store_memory=False, include_aux=include))
    assert bool(read) is include and ("[A1]" in out["context_text"]) is include
    assert out["shown"]["W1"].startswith("Story") and "story body" in out["shown"]["W1"]
    assert ("A1" in out["shown"]) is include

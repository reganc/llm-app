"""Relevance gate for auto-injected memory.

`retrieve` always returns top-k, however weak. Scores are cosine similarity
(nomic-embed-text), and BM25-only hits get a synthetic 0.55. On a
~50k-chunk library of mostly news, a question like "When was Herman Melville
born?" came back with Abramelin, William Shatner and unrelated news at
~0.55 — injected under a prompt calling the library authoritative, the model
invented a birth date from an article about someone else.

Measured on the eval queries: every relevant hit scored >= 0.70 and shared a
content term with the query; every junk hit scored below, or shared none.
So auto mode keeps a chunk only if it clears MEMORY_MIN_SCORE *and* contains
at least one query term. Library mode and exact title/URL matches are exempt.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

import chat
import memory as mem
import search


def chunk(score, text="", title="", ident="x", **kw):
    return {"score": score, "text": text, "title": title, "identifier": ident, **kw}


# ── filter_relevant (pure) ───────────────────────────────────────────────────

def test_drops_low_scores():
    kept = mem.filter_relevant(
        [chunk(0.56, "Herman Melville born"), chunk(0.55, "melville")],
        "When was Herman Melville born?", min_score=0.70)
    assert kept == []


def test_drops_high_score_without_any_query_term():
    kept = mem.filter_relevant(
        [chunk(0.749, "the seven hermetic principles", title="Doc: 1908kybalion.pdf")],
        "Explain the Pythagorean theorem in two sentences.", min_score=0.70)
    assert kept == []


def test_keeps_relevant_hits_matching_title_or_text():
    a = chunk(0.78, "a treatise on tolerance", title="MarcuseH-Repressive-Tolerance.pdf")
    b = chunk(0.747, "17 times 23 is 391")
    kept = mem.filter_relevant([a], "What does Marcuse mean by 'repressive tolerance'?",
                               min_score=0.70)
    assert kept == [a]
    assert mem.filter_relevant([b], "What is 17 times 23?", min_score=0.70) == [b]


def test_quote_wrapped_terms_still_match():
    c = chunk(0.8, "good morning in spanish is buenos dias")
    assert mem.filter_relevant([c], "Translate 'good morning' into Spanish.",
                               min_score=0.70) == [c]


def test_query_without_content_terms_uses_score_only():
    c = chunk(0.9, "anything")
    assert mem.filter_relevant([c], "why?", min_score=0.70) == [c]
    assert mem.filter_relevant([chunk(0.6)], "why?", min_score=0.70) == []


def test_preserves_order():
    hits = [chunk(0.9, "rhythm"), chunk(0.5, "rhythm"), chunk(0.8, "rhythm")]
    kept = mem.filter_relevant(hits, "principle of rhythm", min_score=0.70)
    assert [c["score"] for c in kept] == [0.9, 0.8]


# ── _resolve_context wiring ──────────────────────────────────────────────────

JUNK = chunk(0.55, "Russian general", title="news", ident="junk")
GOOD = chunk(0.85, "the principle of rhythm", title="Kybalion", ident="kyb")


@pytest.fixture
def ctx(monkeypatch):
    cfg = dataclasses.replace(chat.CFG, memory_enabled=True, search_enabled=True,
                              memory_min_score=0.70)
    monkeypatch.setattr(chat, "CFG", cfg)
    seen = {"auto_chunks": None}

    async def fake_retrieve(*_a, **_kw):
        return [GOOD, JUNK]

    async def fake_title(*_a, **_kw):
        return []

    async def fake_auto(chunks, _q):
        seen["auto_chunks"] = chunks
        return False

    monkeypatch.setattr(mem, "retrieve", fake_retrieve)
    monkeypatch.setattr(mem, "retrieve_deep", fake_retrieve)
    monkeypatch.setattr(mem, "lookup_by_title", fake_title)
    monkeypatch.setattr(search, "should_auto_search", fake_auto)
    monkeypatch.setattr(search, "detect_intent",
                        lambda _q: {"signals": [], "force_x": False, "force_search": False})
    return seen


def resolve(query, command=None):
    return asyncio.run(chat._resolve_context(
        query, command, False, False, use_search=True, use_memory=True))


def test_auto_mode_filters_and_router_sees_filtered(ctx):
    _, chunks, *_ = resolve("what is the principle of rhythm")
    assert chunks == [GOOD]
    assert ctx["auto_chunks"] == [GOOD]


def test_auto_mode_all_junk_means_no_memory(ctx, monkeypatch):
    async def junk_only(*_a, **_kw):
        return [JUNK]
    monkeypatch.setattr(mem, "retrieve", junk_only)
    _, chunks, *_ = resolve("When was Herman Melville born?")
    assert chunks == []


def test_library_mode_is_not_filtered(ctx):
    _, chunks, *_ = resolve("rhythm", command="library")
    assert JUNK in chunks


def test_title_matches_bypass_the_filter(ctx, monkeypatch):
    title_hit = chunk(0.3, "", title="Some Saved Article", ident="t1")

    async def fake_title(*_a, **_kw):
        return [title_hit]
    monkeypatch.setattr(mem, "lookup_by_title", fake_title)
    _, chunks, *_ = resolve('Summarize "Some Saved Article About Things"')
    assert title_hit in chunks


def test_min_score_zero_disables_the_gate():
    junk = [chunk(0.3, "unrelated"), chunk(0.55, "")]
    assert mem.filter_relevant(junk, "When was Herman Melville born?", min_score=0) == junk


# ── user-saved vs incidental sources ─────────────────────────────────────────

@pytest.mark.parametrize("stype", ["pdf", "txt", "docx", "md", "csv", "rtf", "web",
                                   "firecrawl", "crawl4ai", "youtube_transcript",
                                   "manual", "distilled"])
def test_user_saved_types(stype):
    assert mem.is_user_saved({"source_type": stype})


@pytest.mark.parametrize("stype", ["bing", "duckduckgo", "duckduckgo_fallback", "google",
                                   "searxng", "web_search", "auxiliary_site", "x_twitter",
                                   "conversation", "", None])
def test_incidental_types(stype):
    assert not mem.is_user_saved({"source_type": stype})

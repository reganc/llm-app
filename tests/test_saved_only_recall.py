"""`memory.rank_chunks` — saved-only recall filters *before* the top-k cut.

The agent path pre-fetches only user-saved sources (past chat turns and
search-ingested pages are snapshots that crowd out a fresh search). It used to
filter after `retrieve` had already cut to MEMORY_TOP_K: for "What does the
Kybalion say about the principle of rhythm?" the top 5 were one old chat turn
and four search pages, so nothing survived, while the saved Kybalion note sat
at rank 7 and the PDF at rank 11.
"""
from __future__ import annotations

import memory as mem


def c(score, source_type, title="t"):
    return {"score": score, "source_type": source_type, "title": title}


TURN, PAGE = "conversation", "duckduckgo"


def test_saved_only_drops_snapshots_before_truncating():
    pool = ([c(0.78, TURN)] + [c(0.77 - i / 100, PAGE) for i in range(4)]
            + [c(0.704, "distilled", "Hermetic Axioms of The Kybalion"),
               c(0.68, "pdf", "1908kybalion.pdf")])
    out = mem.rank_chunks(pool, 5, saved_only=True)
    assert [x["source_type"] for x in out] == ["distilled", "pdf"]


def test_default_keeps_every_type_ranked_by_score():
    pool = [c(0.5, "pdf"), c(0.9, TURN), c(0.7, PAGE)]
    assert [x["score"] for x in mem.rank_chunks(pool, 2)] == [0.9, 0.7]


def test_saved_only_still_respects_top_k():
    pool = [c(0.9 - i / 100, "pdf") for i in range(8)]
    assert len(mem.rank_chunks(pool, 5, saved_only=True)) == 5

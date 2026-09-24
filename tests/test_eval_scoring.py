"""Offline tests for `evals/scoring.py` — the eval is only as good as its judge."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import scoring  # noqa: E402

NUM_CTX = 8192


def obs(content="ok", **kw) -> dict:
    base = {"content": content, "finish_reason": "stop", "prompt_tokens": 100,
            "web_search": None, "memory": None, "error": None, "latency_s": 1.0}
    base.update(kw)
    return base


def by_name(checks) -> dict:
    return {c.name: c for c in checks}


def web(n_web=0, n_x=0, n_aux=0) -> dict:
    results = ([{"engine": "searxng"}] * n_web + [{"engine": "x_twitter"}] * n_x
               + [{"engine": "auxiliary_site"}] * n_aux)
    return {"triggered": True, "results": results}


# ── citations ────────────────────────────────────────────────────────────────

def test_extract_citations_handles_groups_and_singles():
    got = scoring.extract_citations("a [W1] b [W2, W3] c [L1; X4] d [W1]")
    assert got == {"W": {1, 2, 3}, "L": {1}, "X": {4}}


def test_extract_citations_ignores_non_markers():
    assert scoring.extract_citations("[1] [Wx] [see W1] list[0]") == {}


def test_available_markers_split_by_engine_and_capped():
    avail = scoring.available_markers(web(n_web=12, n_x=2, n_aux=5), {"used": 4})
    assert avail == {"W": 9, "X": 2, "A": 3, "L": 4}


def test_server_reported_markers_win_over_result_counting():
    """Agent mode renumbers across searches: 2 searches × 9 = [W18] is valid."""
    ws = {"triggered": True, "results": [{"engine": "searxng"}] * 18,
          "markers": {"W": 18, "X": 0, "A": 0}}
    assert scoring.available_markers(ws, None)["W"] == 18
    c = by_name(scoring.score({}, obs("see [W17]", web_search=ws), num_ctx=NUM_CTX))
    assert c["citations_valid"].passed is True


def test_citation_beyond_shown_sources_fails():
    c = by_name(scoring.score({}, obs("per [W3]", web_search=web(n_web=2)),
                              num_ctx=NUM_CTX))
    assert c["citations_valid"].passed is False
    assert "W3" in c["citations_valid"].detail


def test_library_citation_without_memory_fails():
    c = by_name(scoring.score({}, obs("per [L1]"), num_ctx=NUM_CTX))
    assert c["citations_valid"].passed is False


def test_valid_citations_pass_and_no_citations_is_skipped():
    ok = by_name(scoring.score({}, obs("[W2] and [L1]", web_search=web(n_web=2),
                                       memory={"used": 1}), num_ctx=NUM_CTX))
    assert ok["citations_valid"].passed is True
    none = by_name(scoring.score({}, obs("plain"), num_ctx=NUM_CTX))
    assert none["citations_valid"].passed is None


# ── universal checks ─────────────────────────────────────────────────────────

def test_transport_error_short_circuits():
    checks = scoring.score({"expect": {"contains_any": ["x"]}},
                           obs(error="HTTP 500"), num_ctx=NUM_CTX)
    assert [c.name for c in checks] == ["ok"]
    assert not scoring.case_passed(checks)


def test_empty_and_truncated_fail():
    c = by_name(scoring.score({}, obs("  ", finish_reason="length"), num_ctx=NUM_CTX))
    assert c["non_empty"].passed is False
    assert c["not_truncated"].passed is False


def test_streaming_unobservables_are_skipped_not_failed():
    checks = scoring.score({}, obs(finish_reason=None, prompt_tokens=None),
                           num_ctx=NUM_CTX)
    c = by_name(checks)
    assert c["not_truncated"].passed is None
    assert c["context_fit"].passed is None
    assert scoring.case_passed(checks)


def test_prompt_at_context_window_fails_context_fit():
    c = by_name(scoring.score({}, obs(prompt_tokens=NUM_CTX), num_ctx=NUM_CTX))
    assert c["context_fit"].passed is False
    c = by_name(scoring.score({}, obs(prompt_tokens=4000), num_ctx=NUM_CTX))
    assert c["context_fit"].passed is True


# ── expectations ─────────────────────────────────────────────────────────────

def test_route_web_search_both_directions():
    case = {"expect": {"web_search": False}}
    assert by_name(scoring.score(case, obs(web_search=web(1)), num_ctx=NUM_CTX))[
        "route_web_search"].passed is False
    assert by_name(scoring.score(case, obs(), num_ctx=NUM_CTX))[
        "route_web_search"].passed is True


def test_memory_mode_and_source():
    case = {"expect": {"memory_mode": "library", "memory_source_contains": "Kybalion"}}
    mem = {"used": 1, "mode": "library",
           "items": [{"identifier": "1908kybalion.pdf", "title": ""}]}
    c = by_name(scoring.score(case, obs(memory=mem), num_ctx=NUM_CTX))
    assert c["route_memory_mode"].passed is True
    assert c["memory_source"].passed is True
    c = by_name(scoring.score(case, obs(), num_ctx=NUM_CTX))
    assert c["route_memory_mode"].passed is False
    assert c["memory_source"].passed is False


def test_content_checks_are_case_and_symbol_tolerant():
    case = {"expect": {"contains_any": ["a^2 + b^2", "hypotenuse"],
                       "contains_all": ["Buenos días"],
                       "not_contains": ["As an AI"]}}
    c = by_name(scoring.score(case, obs("A² + B² = C². BUENOS DÍAS"), num_ctx=NUM_CTX))
    assert c["contains_any"].passed is True
    assert c["contains:Buenos días"].passed is True
    assert c["not_contains:As an AI"].passed is True


def test_must_cite():
    case = {"expect": {"must_cite": True}}
    assert by_name(scoring.score(case, obs("no refs"), num_ctx=NUM_CTX))[
        "cites_sources"].passed is False


# ── aggregation ──────────────────────────────────────────────────────────────

GOOD = scoring.score({}, obs(), num_ctx=NUM_CTX)
BAD = scoring.score({}, obs(""), num_ctx=NUM_CTX)


def row(case_id, *runs, category="x", latency=1.0):
    return {"id": case_id, "category": category,
            "runs": [{"checks": r, "latency_s": latency} for r in runs]}


def test_case_result_majority():
    assert scoring.case_result([GOOD]) == {"pass_rate": 1.0, "passed": True}
    assert scoring.case_result([GOOD, GOOD, BAD]) == {"pass_rate": 0.67, "passed": True}
    assert scoring.case_result([GOOD, BAD]) == {"pass_rate": 0.5, "passed": False}
    assert scoring.case_result([BAD, BAD, GOOD])["passed"] is False


def test_summarize_counts_runs_and_flags_flaky():
    rows = [row("a", GOOD, GOOD, latency=1.0),
            row("b", GOOD, BAD, latency=3.0),
            row("c", BAD, BAD, category="y", latency=2.0)]
    s = scoring.summarize(rows)
    assert (s["cases"], s["passed"], s["runs_per_case"]) == (3, 1, 2)
    assert s["checks"]["non_empty"] == {"passed": 3, "total": 6}
    assert s["categories"] == {"x": {"passed": 1, "total": 2},
                               "y": {"passed": 0, "total": 1}}
    assert s["flaky"] == ["b"]
    assert s["latency_p50_s"] == 2.0


def test_compare_flips_and_rate_changes():
    base = {"rows": [{"id": "a", "passed": True, "pass_rate": 1.0},
                     {"id": "b", "passed": False, "pass_rate": 0.33},
                     {"id": "d", "passed": True, "pass_rate": 1.0},
                     {"id": "gone", "passed": True}]}
    cur = {"rows": [{"id": "a", "passed": False, "pass_rate": 0.0},
                    {"id": "b", "passed": True, "pass_rate": 0.67},
                    {"id": "d", "passed": True, "pass_rate": 0.67},
                    {"id": "c", "passed": True, "pass_rate": 1.0}]}
    assert scoring.compare(base, cur) == {
        "fixed": ["b"], "regressed": ["a"], "new": ["c"], "removed": ["gone"],
        "rate_changed": {"a": [1.0, 0.0], "b": [0.33, 0.67], "d": [1.0, 0.67]},
    }


def test_compare_reads_old_results_without_pass_rate():
    base = {"rows": [{"id": "a", "passed": False}]}
    cur = {"rows": [{"id": "a", "passed": True, "pass_rate": 1.0}]}
    diff = scoring.compare(base, cur)
    assert diff["fixed"] == ["a"]
    assert diff["rate_changed"] == {"a": [0.0, 1.0]}


def test_padded_brackets_are_citations():
    assert scoring.extract_citations("[ X3 ] and [ W1 , W2 ]") == {"X": {3}, "W": {1, 2}}

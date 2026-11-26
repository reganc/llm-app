"""Offline tests for the model replay (`evals/replay.py`, `evals/replay_diff.py`):
which questions are replayed with what history, and what code flags."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import replay  # noqa: E402
import replay_diff as rd  # noqa: E402

WEB = {"triggered": True, "markers": {"W": 3, "X": 0, "A": 0}}


def obs(content="The answer is 42 [W1].", **kw) -> dict:
    return {"content": content, "finish_reason": "stop", "web_search": WEB,
            "memory": None, "agent": {}, "latency_s": 3.0, **kw}


# ── flaws in one answer ─────────────────────────────────────────────────────

def test_a_clean_answer_has_no_flaws():
    f = rd.facts(obs())
    assert f["flaws"] == [] and f["searched"] and f["numbers"] == ["42"]


def test_each_flaw_is_seen():
    assert rd.facts(obs(""))["flaws"] == ["empty"]
    assert "truncated" in rd.facts(obs(finish_reason="length"))["flaws"]
    bad = rd.facts(obs("See [W7]."))
    assert bad["flaws"] == ["bad_citations"] and "W7" in bad["detail"]
    assert rd.facts(obs("No sources here."))["flaws"] == ["uncited_web"]
    assert rd.facts(obs(agent={"citation_repair": "invalid"}))["flaws"] == ["repaired"]
    assert rd.facts({"error": "HTTP 500: boom", "latency_s": 1})["flaws"] == ["error"]


def test_no_citations_is_fine_without_web_results():
    assert rd.facts(obs("Paris.", web_search=None))["flaws"] == []


def test_numbers_ignore_markers_and_list_numbering():
    text = "1. Revenue rose 12.5% to $3,400 [W2].\n2. Up from 2025 [W1, W3]."
    assert rd.numbers(text) == {"12.5%", "3400", "2025"}


# ── comparing the two ───────────────────────────────────────────────────────

def test_regressions_and_fixes_are_one_sided():
    cmp = rd.compare(obs("No sources here."), obs(""))
    assert cmp["regressed"] == ["empty"] and cmp["fixed"] == ["uncited_web"]
    same = rd.compare(obs(""), obs(""))
    assert same["regressed"] == [] and same["fixed"] == []


def test_differences_are_neutral_pointers():
    a = obs("It costs 42 [W1].", latency_s=3.0)
    b = obs("It costs 45 dollars, with a much longer explanation of why that is [W1].",
            web_search={"triggered": False}, latency_s=12.0)
    assert set(rd.compare(a, b)["differs"]) == {"search_differs", "numbers_differ", "length_differs", "slower"}
    assert rd.compare(b, a)["differs"].count("faster") == 1
    # a small latency gap isn't "slower", however large the ratio
    assert "slower" not in rd.compare(obs(latency_s=1.0), obs(latency_s=4.0))["differs"]


def test_an_error_has_no_differences():
    assert rd.compare(obs(), {"error": "timeout"})["differs"] == []


def test_summary_counts_and_reading_order():
    rows = [{"id": "q1", "compare": rd.compare(obs(), obs())},
            {"id": "q2", "compare": rd.compare(obs(), obs(""))},
            {"id": "q3", "compare": rd.compare(obs(""), obs())}]
    s = rd.summarize(rows)
    assert s["regressed"]["empty"] == 1 and s["fixed"]["empty"] == 1
    assert s["regressed_ids"] == ["q2"] and s["fixed_ids"] == ["q3"]
    assert (s["baseline"]["clean"], s["candidate"]["clean"]) == (2, 2)
    assert [r["id"] for r in rd.reading_order(rows)] == ["q2", "q3", "q1"]


# ── which questions, with what history ──────────────────────────────────────

def test_conversation_questions_carry_their_history():
    conv = {"id": "abcdef123456", "updated_at": "2026-11-20T10:00:00",
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "who won the 2024 election?"},
                         {"role": "assistant", "content": "A [W1]"},
                         {"role": "user", "content": "/search something"},
                         {"role": "user", "content": "ok"},
                         {"role": "user", "content": "when was he born?"}]}
    qs = replay.from_conversation(conv)
    assert [q["id"] for q in qs] == ["spa-abcdef12-0", "spa-abcdef12-4"]   # commands and "ok" skipped
    assert qs[1]["messages"][-1] == {"role": "user", "content": "when was he born?"}
    assert qs[1]["messages"][0]["content"] == "who won the 2024 election?"    # system dropped
    assert all(m["role"] != "system" for q in qs for m in q["messages"])


def test_a_loaded_document_is_history_not_a_question():
    conv = {"id": "doc", "messages": [
        {"role": "user", "content": "Please load this document:\n\nFILE: a.pdf\n...text..."},
        {"role": "assistant", "content": "Document loaded: a.pdf (9,000 chars). Ready for questions."},
        {"role": "user", "content": "what is chapter two about?"}]}
    (q,) = replay.from_conversation(conv)
    assert q["id"] == "spa-doc-2" and len(q["messages"]) == 3
    assert q["messages"][0]["content"].startswith("Please load this document:")   # still sent as history
    article = {"id": "art", "messages": [{"role": "user", "content": "Please load this article:\n\nSOURCE: x"}]}
    assert replay.from_conversation(article) == []


def test_conversation_history_is_capped():
    msgs = [{"role": r, "content": f"message number {i}"}
            for i, r in zip(range(30), ["user", "assistant"] * 15)]
    qs = replay.from_conversation({"id": "x", "messages": msgs})
    assert len(qs[-1]["messages"]) == replay.HISTORY_MESSAGES + 1


def test_telegram_turns_get_only_the_last_day_before_them():
    day = 24 * 3600
    data = {"chat_id": 1, "turns": [
        {"at": 1000.0, "q": "first old question", "a": "a1"},
        {"at": 1000.0 + day + 10, "q": "a later question", "a": "a2"},
        {"at": 1000.0 + day + 20, "q": "and a follow-up", "a": "a3"}]}
    qs = replay.from_telegram(data)
    assert [len(q["messages"]) for q in qs] == [1, 1, 3]      # the first turn is over a day old for the others
    assert qs[2]["messages"][0]["content"] == "a later question"


def test_newest_first_and_limit():
    qs = [{"id": i, "at": at} for i, at in enumerate(["2026-11-01", "2026-11-25T08:00:00", "2026-11-10"])]
    assert [q["id"] for q in replay.newest(qs, 2)] == [1, 2]


def test_bodies_never_store_and_telegram_matches_the_bot():
    spa = replay.body_for({"source": "spa", "messages": []}, "m", 0.0)
    tg = replay.body_for({"source": "telegram", "messages": []}, "m", 0.0)
    assert spa["store"] is False and tg["store"] is False
    assert "memory" not in spa and tg["memory"] is False and tg["max_tokens"] == 1024


def test_results_are_private(tmp_path):
    path = tmp_path / "r.md"
    replay._private_write(path, "secret")
    assert path.stat().st_mode & 0o777 == 0o600


def test_report_puts_regressions_first():
    q = {"source": "spa", "at": "t", "messages": [{"role": "user", "content": "q"}]}
    rows = [{"id": "fine", "question": q, "compare": rd.compare(obs(), obs()), "baseline": obs(), "candidate": obs()},
            {"id": "broke", "question": q, "compare": rd.compare(obs(), obs("")), "baseline": obs(), "candidate": obs("")}]
    md = replay.report_md({"baseline": "a", "candidate": "b", "started_at": "t", "temperature": 0},
                          rd.summarize(rows), rows)
    assert md.index("## broke") < md.index("## fine") and "regressed: empty" in md

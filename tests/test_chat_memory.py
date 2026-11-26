"""The Telegram conversation memory: longer, bounded, saved across restarts,
tied to one chat, erased by /new."""
from __future__ import annotations

import json
import stat

import pytest

import chat_memory as cm

ME, OTHER = 42, 7


@pytest.fixture
def mem(tmp_path):
    cm.load(tmp_path / "h.json")
    yield tmp_path / "h.json"
    cm.load(tmp_path / "gone.json")


def contents(messages):
    return [m["content"] for m in messages]


def test_turns_are_oldest_first_question_then_answer(mem):
    cm.remember(ME, "q1", "a1", now=100)
    cm.remember(ME, "q2", "a2", now=200)
    assert cm.turns(ME, now=300) == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                                     {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]


def test_it_survives_a_restart_and_the_file_is_private(mem):
    cm.remember(ME, "my dog is Rex", "Noted.", now=100)
    cm.load(mem)
    assert contents(cm.turns(ME, now=200)) == ["my dog is Rex", "Noted."]
    assert stat.S_IMODE(mem.stat().st_mode) == 0o600


def test_each_turn_expires_on_its_own_after_a_day(mem):
    cm.remember(ME, "old", "x", now=0)
    cm.remember(ME, "new", "y", now=cm.TTL_S)
    assert contents(cm.turns(ME, now=cm.TTL_S + 10)) == ["new", "y"]


def test_at_most_twenty_turns_and_long_text_is_cut(mem):
    for i in range(25):
        cm.remember(ME, f"q{i}", "a" * 5000, now=i)
    kept = cm.turns(ME, now=30)
    assert len(json.loads(mem.read_text())["turns"]) == cm.MAX_TURNS
    assert all(len(m["content"]) <= cm.ANSWER_CHARS for m in kept)


def test_only_the_newest_turns_that_fit_the_budget_are_sent(mem):
    for i in range(20):
        cm.remember(ME, f"q{i}" + "x" * 990, "y" * 1500, now=i)      # ~2,500 chars a turn
    sent = cm.turns(ME, now=30)
    assert sum(len(m["content"]) for m in sent) <= cm.BUDGET_CHARS
    assert sent[-2]["content"].startswith("q19") and len(sent) == 12      # 6 newest turns


def test_another_chat_never_sees_it_and_replaces_it(mem):
    cm.remember(ME, "private", "x", now=1)
    assert cm.turns(OTHER, now=2) == [] and cm.turns(None, now=2) == []
    cm.remember(OTHER, "hello", "hi", now=3)
    assert cm.turns(ME, now=4) == [] and contents(cm.turns(OTHER, now=4)) == ["hello", "hi"]


def test_forget_erases_the_saved_file_too(mem):
    cm.remember(ME, "secret", "x", now=1)
    cm.forget()
    cm.load(mem)
    assert cm.turns(ME, now=2) == [] and "secret" not in mem.read_text()


def test_an_unreadable_file_starts_empty_and_is_left_alone(mem):
    mem.write_text("{not json")
    cm.load(mem)
    assert cm.turns(ME, now=1) == [] and mem.read_text() == "{not json"

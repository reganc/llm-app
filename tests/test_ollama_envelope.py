"""Tests for `completion_envelope` under thinking models (qwen3.5+).

Background: a thinking model emits reasoning tokens *before* any content. When
`num_predict` is exhausted mid-reasoning, Ollama returns `content: ""` with
`done_reason: "length"`. The envelope used to pass that straight through as a
successful, empty answer — the `/v1/chat/reasoning` blank-reply bug.

The contract now:
  * reasoning is surfaced as `message.reasoning` whenever present;
  * an empty content with usable reasoning falls back to the reasoning text and
    is flagged `message.reasoning_fallback`;
  * `done_reason == "length"` is reported as `finish_reason: "length"`;
  * responses without a `thinking` field (every non-thinking model, i.e. the
    main chat path) are byte-for-byte what they were before.
"""
from __future__ import annotations

import ollama as oll


def envelope(data: dict, model: str = "test-model", extra: dict | None = None) -> dict:
    return oll.completion_envelope(data, model, extra=extra)


def message(data: dict) -> dict:
    return envelope(data)["choices"][0]["message"]


def finish_reason(data: dict) -> str:
    return envelope(data)["choices"][0]["finish_reason"]


# ── the original bug ─────────────────────────────────────────────────────────

def test_reasoning_truncated_falls_back_to_reasoning_text():
    """Budget consumed by thinking: return the reasoning, not an empty string."""
    m = message({
        "message": {"content": "", "thinking": "step 1: 8.5h\nstep 2: 3.5h"},
        "done_reason": "length",
    })
    assert m["content"] == "step 1: 8.5h\nstep 2: 3.5h"
    assert m["reasoning_fallback"] is True
    assert m["reasoning"] == "step 1: 8.5h\nstep 2: 3.5h"


def test_reasoning_truncated_reports_length_finish_reason():
    """The caller must be able to tell truncation from a clean stop."""
    assert finish_reason({
        "message": {"content": "", "thinking": "..."},
        "done_reason": "length",
    }) == "length"


# ── the non-thinking path must not change ────────────────────────────────────

def test_plain_response_is_unchanged():
    """Main chat path (think=false): no reasoning keys leak into the message."""
    m = message({"message": {"content": "hello"}, "done_reason": "stop"})
    assert m == {"role": "assistant", "content": "hello"}


def test_plain_response_finish_reason_is_stop():
    assert finish_reason({"message": {"content": "hello"}, "done_reason": "stop"}) == "stop"


def test_empty_content_without_reasoning_stays_empty():
    """Nothing to fall back to — don't invent content or claim a fallback."""
    m = message({"message": {"content": ""}, "done_reason": "stop"})
    assert m["content"] == ""
    assert "reasoning_fallback" not in m
    assert "reasoning" not in m


# ── thinking that completed normally ─────────────────────────────────────────

def test_content_wins_when_thinking_completed():
    """Reasoning is exposed but must never displace a real answer."""
    m = message({
        "message": {"content": "The answer is 5.04.", "thinking": "long reasoning"},
        "done_reason": "stop",
    })
    assert m["content"] == "The answer is 5.04."
    assert m["reasoning"] == "long reasoning"
    assert "reasoning_fallback" not in m


def test_content_truncated_by_length_is_not_a_fallback():
    """Truncated *content* is still content — report length, but no fallback."""
    m = message({"message": {"content": "partial ans"}, "done_reason": "length"})
    assert m["content"] == "partial ans"
    assert "reasoning_fallback" not in m
    assert finish_reason({"message": {"content": "partial ans"},
                          "done_reason": "length"}) == "length"


# ── whitespace / null edge cases ─────────────────────────────────────────────

def test_whitespace_only_content_triggers_fallback():
    """A stream that emitted only newlines is as useless as an empty one."""
    m = message({"message": {"content": "\n  \n", "thinking": "real reasoning"},
                 "done_reason": "length"})
    assert m["content"] == "real reasoning"
    assert m["reasoning_fallback"] is True


def test_whitespace_only_reasoning_is_not_used_as_fallback():
    """Don't swap an empty answer for equally empty reasoning."""
    m = message({"message": {"content": "", "thinking": "   \n "},
                 "done_reason": "length"})
    assert m["content"] == ""
    assert "reasoning_fallback" not in m


def test_null_content_and_thinking_do_not_crash():
    """Ollama may send JSON nulls rather than omitting the fields."""
    m = message({"message": {"content": None, "thinking": None}, "done_reason": "stop"})
    assert m["content"] == ""
    assert "reasoning" not in m


def test_missing_message_key_does_not_crash():
    m = message({"done_reason": "stop"})
    assert m == {"role": "assistant", "content": ""}


def test_null_message_does_not_crash():
    m = message({"message": None, "done_reason": "stop"})
    assert m == {"role": "assistant", "content": ""}


def test_fallback_strips_surrounding_whitespace():
    m = message({"message": {"content": "", "thinking": "  answer is 5.04  "},
                 "done_reason": "length"})
    assert m["content"] == "answer is 5.04"


# ── envelope shape / regressions ─────────────────────────────────────────────

def test_missing_done_reason_defaults_to_stop():
    """Older/partial payloads must not be reported as truncated."""
    assert finish_reason({"message": {"content": "hi"}}) == "stop"


def test_usage_and_model_are_preserved():
    env = envelope({
        "message": {"content": "hi"},
        "prompt_eval_count": 12,
        "eval_count": 30,
    }, model="qwen3.5:9b")
    assert env["model"] == "qwen3.5:9b"
    assert env["usage"] == {"prompt_tokens": 12, "completion_tokens": 30,
                            "total_tokens": 42}


def test_usage_defaults_to_zero_when_counts_absent():
    env = envelope({"message": {"content": "hi"}})
    assert env["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0}


def test_extra_fields_are_merged_at_top_level():
    env = envelope({"message": {"content": "hi"}},
                   extra={"reasoning_mode": True, "message_id": "abc"})
    assert env["reasoning_mode"] is True
    assert env["message_id"] == "abc"


def test_envelope_has_openai_shape():
    env = envelope({"message": {"content": "hi"}})
    assert env["object"] == "chat.completion"
    assert env["id"].startswith("chatcmpl-")
    assert isinstance(env["created"], int)
    assert env["choices"][0]["index"] == 0
    assert env["choices"][0]["message"]["role"] == "assistant"

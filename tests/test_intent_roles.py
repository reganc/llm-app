"""`search.detect_intent` — current role-holder questions force a search.

"Who is the CEO of OpenAI?" has no freshness keyword, and the 9B model
answers it confidently from training data even when the web_search tool
description says to search for "who currently holds a role". Who holds an
office is exactly the kind of fact that goes stale, so present-tense
role-holder questions are routed deterministically. Past tense and
historical phrasing ("who was the first president") must not match.
"""
from __future__ import annotations

import pytest

import search


@pytest.mark.parametrize("q", [
    "Who is the CEO of OpenAI?",
    "who's the prime minister of the UK",
    "Who is the current president of France?",
    "Who is Nvidia's CEO?",
    "who is the chair of the Federal Reserve",
    "Who are the members of the Supreme Court?",
    "Who runs Twitter now?",
    "Who leads the Labour Party?",
    "who heads the FBI",
    "Who is in charge of the Pentagon?",
    "Who is the head coach of the Chicago Bears?",
    "Who is the mayor of Chicago?",
    "Who is the Secretary of State?",
    "Who owns the Washington Post?",
])
def test_current_role_questions_force_search(q):
    intent = search.detect_intent(q)
    assert intent["force_search"], q
    assert any(s.startswith("role:") for s in intent["signals"]), intent


@pytest.mark.parametrize("q", [
    "Who was the first president of the United States?",
    "Who was CEO of Apple before Tim Cook?",
    "Who wrote Moby-Dick?",
    "Who painted the Mona Lisa?",
    "Who is the author of Dune?",
    "Who invented the telephone?",
    "What does a CEO do?",
    "Explain the role of the prime minister in parliament.",
    "who is he",
])
def test_non_role_or_historical_questions_do_not(q):
    intent = search.detect_intent(q)
    assert not any(s.startswith("role:") for s in intent["signals"]), (q, intent)
    assert not intent["force_search"], (q, intent)

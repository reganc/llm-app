"""The search phrase worked out from a routine's task."""
from __future__ import annotations

import pytest

import routine_query as q


@pytest.mark.parametrize("prompt,phrase", [
    ("Summarize the latest AI news into 5 bullet points with sources attached", "AI news"),
    ("Summarize the most important AI news from the last day in five bullets, with sources.", "AI news"),
    ("Give me 3 bullet points on the stock market today, cite sources", "stock market today"),
    ("In five bullets, summarize Federal Reserve news", "Federal Reserve news"),
    ("Using at most 4 sentences, what is happening with oil prices?", "oil prices"),
    ("What is the price of Bitcoin?", "price Bitcoin"),
    ("Check whether NVDA dropped below 100 and tell me why", "NVDA dropped below 100"),
    ("Please find stories about SpaceX launches. Use plain English.", "stories about SpaceX launches"),
    ("Compare Claude and GPT pricing in a table", "Claude GPT pricing"),
    ("What is the latest stable Python version and when was it released?", "stable Python version"),
    ("Summarize climate change news in 4 bullets, include quotes", "climate change news"),
    ("Summarize AI toy safety news, with quotes and sources", "AI toy safety news"),
    ("the US Open results", "US Open results"),
    ("What's new in Python?", "Python"),
    ("Resuma las noticias de IA", "Resuma las noticias de IA"),
])
def test_subject_is_picked_out(prompt, phrase):
    assert q.search_phrase(prompt) == phrase


@pytest.mark.parametrize("prompt", ["Summarize", "Latest news", "", "   ", "give me 5 bullet points",
                                    "x " * 900, "!!! ???", "the the the"])
def test_odd_prompts_give_a_short_phrase_and_never_raise(prompt):
    phrase = q.search_phrase(prompt)
    assert len(phrase.split()) <= 10
    assert bool(phrase) == bool(prompt.strip())


def test_long_prompts_return_quickly():
    import time
    start = time.perf_counter()
    q.search_phrase("in a few several brief " * 90)
    q.search_phrase("summarize the latest " * 95)
    q.search_phrase("and tell me " * 160)
    assert time.perf_counter() - start < 0.5


def test_effective_prefers_the_users_phrase():
    r = {"prompt": "Summarize AI news", "query": "openai", "sources": {"web": True}}
    assert q.effective(r) == "openai"
    assert q.effective({**r, "query": None}) == "AI news"
    assert q.effective({**r, "sources": {"web": False}}) is None

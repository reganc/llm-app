"""The search phrase for a routine that doesn't set one.

A whole instruction makes a bad search: news engines answer "Summarize the latest
AI news into 5 bullet points with sources attached" with an error or an unrelated
page. The subject is picked out in code, so the same task always searches the same.
"""
from __future__ import annotations

import re

_MAX_WORDS = 10

# How the answer should look, not what it is about.
_N = r"(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|a few|several|short|brief)"
_SHAPE = (r"(?:bullet(?:ed)?(?:\s+(?:points?|lists?))?s?|points?|sentences?|paragraphs?|words?|"
          r"lines?|lists?|tables?|summary|summaries|format|markdown|plain\s+(?:english|text))")
_FORMAT = re.compile("|".join((
    rf"\b(?:in|into|as|using|with|under|within)\s+(?:(?:at most|no more than|up to|about)\s+)?"
    rf"(?:{_N}\s+)*(?:[a-z]+\s+)?{_SHAPE}\b",
    r"\b(?:with|including|and|include|cite|attach|add|give|provide)\s+(?:(?:the|your|all|any|their|"
    r"its|a)\s+)?(?:sources?|links?|citations?|references?|urls?|quotes?)\b(?:\s+(?:attached|included|cited|"
    r"for\s+each(?:\s+\w+)?))?",
    r"\b(?:from|in|over|during|of|within)\s+the\s+(?:last|past)\s+(?:\w+\s+)?(?:hours?|days?|weeks?|"
    r"months?)\b",
)), re.IGNORECASE)

# Instruction words, only where they open a phrase ("report on X", "tell me about X").
_LEAD = re.compile(
    r"^(?:(?:please|can|could|would|you|then|also|and|just|briefly|me|i|want|need|like|to|know|"
    r"up\s+to\s+date|report(?=\s+(?:on|about|whether|if|any|the)\b)|bullet(?:ed)?\s+points?|points?|"
    r"summari[sz]e|list|give(?:\s+us)?|provide|tell(?:\s+us)?|find|search|look|show|get|fetch|check|track|monitor|"
    r"watch|when|was|were|did|does|do|who|where|how|why|much|many|write|create|make|compile|prepare|send|share|highlight|explain|describe|review|compare|"
    r"research|investigate|brief|update|keep|cover|whether|if|what(?:'s|’s)?|is|are|on|up|for|about|"
    r"of|out|the|a|an|any|all|my|most|important|main|key|major|biggest|top|latest|recent|newest|"
    r"current|new|summary|overview|roundup|digest|list|"
    + r"\d+|two|three|four|five|six|seven|eight|nine|ten|few|happening|going|with|in)\b[\s:,]*)+",
    re.IGNORECASE)
# A second instruction in the same sentence ("… and tell me why").
_THEN = re.compile(r"\s(?:and|then)\s+(?:then\s+)?(?:please\s+)?(?:tell|explain|give|list|summari[sz]e|"
                   r"show|write|include|say|describe|highlight|compare|rank|note|let|when|why|how|who|where|"
                   r"whether)\b.*$", re.IGNORECASE)
_DROP = {"the", "a", "an", "please", "latest", "any", "me", "my", "for", "to", "and", "or", "of",
         "with", "from", "that", "is", "are", "in", "on", "has", "have", "most", "important"}


def _subject(clause: str) -> str:
    clause = _LEAD.sub("", _THEN.sub("", clause.strip(" ,;:")))
    words = [w.strip("“”\"'()[].,:;!?") for w in clause.split()]
    return " ".join([w for w in words if w and w.lower() not in _DROP][:_MAX_WORDS])


def search_phrase(prompt: str) -> str:
    """What to search for. Falls back to the first words of the task."""
    text = " ".join((prompt or "").split())
    for clause in re.split(r"(?<=[.!?;])\s+|\n", _FORMAT.sub(" ", text)):
        if subject := _subject(clause):
            return subject
    return " ".join(text.split()[:_MAX_WORDS])


def effective(routine: dict) -> str | None:
    """The phrase a routine searches for, or None if it doesn't search the web."""
    if not routine["sources"].get("web"):
        return None
    return routine.get("query") or search_phrase(routine["prompt"])

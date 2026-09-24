"""Rewrite follow-up questions into standalone queries for retrieval + search.

Memory retrieval, the auto-search router and web search all key off the last
user message. On a follow-up ("When was he born?") that message carries no
subject, so retrieval returns noise and search queries the literal pronoun.

`rewrite_query` resolves the references with one short model call — only when
there is history *and* the message looks like a follow-up — then verifies the
result deterministically: any name or number the rewrite introduces must
already appear in the conversation. A rewrite that fails the check, times
out, or comes back empty is discarded and the original query is used.

The rewrite only feeds retrieval/search. The model still answers the user's
original words, with the full conversation.
"""
from __future__ import annotations

import logging
import re

import ollama as oll
from config import CFG

log = logging.getLogger("llm-api.rewrite")

_HISTORY_TURNS = 6
_HISTORY_CHARS = 600
_MAX_REWRITE_CHARS = 300
_SHORT_QUERY_WORDS = 4

# Words that only make sense with an antecedent — the follow-up signal.
_FOLLOWUP_RE = re.compile(
    r"\b(he|she|it|they|them|him|his|her|hers|its|their|theirs|"
    r"this|that|these|those|there|then|"
    r"the same|the former|the latter|else|also|too|more|again|another|"
    r"what about|how about)\b",
    re.IGNORECASE,
)

# Capitalised words that are not entities: question words, articles, and
# sentence-initial filler the model is free to add.
_NON_ENTITY = {
    "what", "who", "whom", "whose", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "does", "did", "do", "can", "could", "should",
    "would", "will", "has", "have", "had", "the", "a", "an", "in", "on", "of",
    "and", "or", "for", "to", "i", "my", "tell", "explain", "describe", "list",
    "give", "show", "find", "compare",
}
_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w'’\-]*|\d[\d,.]*)")
_LABEL_RE = re.compile(r"^\s*(standalone( question| query)?|rewritten( question| query)?|"
                       r"query|question)\s*:\s*", re.IGNORECASE)

_PROMPT = """\
Rewrite the user's LAST MESSAGE as a standalone question that can be understood \
without the conversation. Replace pronouns and references (he, it, that, there, \
the former...) with the specific names they refer to in the conversation.
Rules:
- Keep the user's intent and wording otherwise; keep time words (latest, today) \
and phrases like "my library".
- Do NOT answer the question.
- Do NOT add any name, date or number that is not in the conversation.
- If the message is already standalone, return it unchanged.
- Output only the rewritten question, on one line.

Conversation:
{history}

LAST MESSAGE: {query}
Standalone question:"""


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _prior_turns(history: list[dict]) -> list[dict]:
    return [m for m in history if m.get("role") in ("user", "assistant")
            and _text(m.get("content")).strip()]


def needs_rewrite(history: list[dict], query: str) -> bool:
    """True when there are prior turns and `query` looks like a follow-up."""
    if not query.strip() or not _prior_turns(history):
        return False
    return bool(_FOLLOWUP_RE.search(query)) or len(query.split()) <= _SHORT_QUERY_WORDS


def _clean(raw: str) -> str:
    line = next((ln for ln in (raw or "").splitlines() if ln.strip()), "")
    line = _LABEL_RE.sub("", line).strip()
    return line.strip("\"'`“”‘’ ").strip()


def accept_rewrite(raw: str, history: list[dict], query: str) -> str | None:
    """Return the cleaned rewrite, or None if it is unusable or ungrounded."""
    candidate = _clean(raw)
    if not candidate or len(candidate) > _MAX_REWRITE_CHARS:
        return None
    known = " ".join(_text(m.get("content")) for m in _prior_turns(history))
    known = f"{known} {query}".lower()
    for token in _ENTITY_RE.findall(candidate):
        word = token.rstrip(".,").lower()
        if word in _NON_ENTITY or not word:
            continue
        if not re.search(rf"(?<!\w){re.escape(word)}(?!\w)", known):
            log.info("rewrite rejected: %r introduces %r", candidate, token)
            return None
    return candidate


def _history_block(history: list[dict]) -> str:
    turns = _prior_turns(history)[-_HISTORY_TURNS:]
    return "\n".join(
        f"{m['role'].capitalize()}: {_text(m['content']).strip()[:_HISTORY_CHARS]}"
        for m in turns
    )


async def rewrite_query(history: list[dict], query: str) -> str:
    """Standalone version of `query` given `history` (prior turns only).

    Never raises; returns `query` unchanged whenever rewriting is skipped or
    its result can't be trusted.
    """
    if not needs_rewrite(history, query):
        return query
    prompt = _PROMPT.format(history=_history_block(history), query=query.strip())
    payload = oll.build_payload([{"role": "user", "content": prompt}], "default",
                                temperature=0.0, max_tokens=80, top_p=1.0)
    try:
        data = await oll.chat(payload, timeout=CFG.classifier_timeout)
    except Exception as e:
        log.info("rewrite skipped (%s): %s", type(e).__name__, e)
        return query
    raw = (data.get("message", {}) or {}).get("content", "") or ""
    accepted = accept_rewrite(raw, history, query)
    if accepted and accepted != query:
        log.info("rewrite: %r -> %r", query, accepted)
        return accepted
    return query

"""Native tool-calling loop — the model decides when to search.

Replaces, for requests that opt in, the auto-search router (two LLM probes)
and the `[SEARCH: …]` text sentinel. The model is offered read-only tools via
Ollama's `tools` field and may call them over a bounded number of rounds:

  * ``web_search(query)``      — SearXNG + page fetch → `[W#]/[X#]/[A#]` block
  * ``library_search(query)``  — memory recall, relevance-gated → `[L#]` block

Guard rails (deterministic, not model-dependent):
  * at most MAX_ROUNDS model calls and MAX_TOOL_CALLS executions per turn; the
    final round is offered no tools, so the loop always ends in an answer;
  * tool names and arguments are validated; bad calls get an error result the
    model can recover from, never an exception;
  * a repeated identical call is not re-executed;
  * citation markers are renumbered so they stay unique across calls — a
    second search continues at [W6] rather than reusing [W1].

Tools only read (a web search may ingest pages into memory unless the request
set ``store: false``). Nothing here takes an action with side effects beyond
that, so the loop needs no confirmation step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

import memory as mem
import ollama as oll
import search
from config import CFG
from prompts import search_block_prompt

log = logging.getLogger("llm-api.agent")

MAX_ROUNDS = 3
MAX_TOOL_CALLS = 4
_MAX_QUERY_CHARS = 300
_LIBRARY_CHARS = 3000

_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web. Use ONLY for information that changes over time "
            "(news, prices, weather, schedules, who currently holds a role, recent "
            "releases) or specific facts you do not know. Do NOT use for stable "
            "general knowledge, math, coding, definitions or translation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A concise, standalone search query."}},
            "required": ["query"],
        },
    },
}
_LIBRARY_SEARCH = {
    "type": "function",
    "function": {
        "name": "library_search",
        "description": (
            "Search the user's saved library (documents, pages and notes they "
            "stored). Use when the question is about material the user may have "
            "saved, or when earlier library results were not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "What to look for in the library."}},
            "required": ["query"],
        },
    },
}

TOOL_GUIDANCE = (
    "\n\nYou can call tools: web_search for current or unknown facts, and "
    "library_search for the user's saved material. Answer directly — without "
    "calling a tool — whenever you already know a stable fact, or the task is "
    "math, coding, writing or translation. Tool results label each source with "
    "a marker like [W3] or [L2]; cite every fact taken from a result with its "
    "marker, and only use markers that actually appear in the results. Library "
    "entries can be out of date: for prices, news or anything else that changes, "
    "call web_search even if the library already has a value."
)

_MARKER_RE = re.compile(r"\[([WXAL])(\d+)\]")
# A citation group as the model writes it: [W1] · [W1, W3] · [L2; W4]
_CITE_GROUP_RE = re.compile(r"(\s*)\[\s*((?:[WXAL]\d+)(?:\s*[,;]\s*[WXAL]\d+)*)\s*\]")
_CITE_ONE_RE = re.compile(r"([WXAL])(\d+)")


def tools_for(*, web: bool, library: bool) -> list[dict]:
    return ([_WEB_SEARCH] if web else []) + ([_LIBRARY_SEARCH] if library else [])


# ── trace / ledger ───────────────────────────────────────────────────────────

@dataclass
class Trace:
    """What the loop did — drives renumbering and the response metadata."""

    markers: dict[str, int] = field(default_factory=lambda: {"W": 0, "X": 0, "A": 0, "L": 0})
    calls: list[dict] = field(default_factory=list)          # {"name","query","status"}
    web_results: list[dict] = field(default_factory=list)    # search_and_ingest results
    web_queries: list[str] = field(default_factory=list)
    library_chunks: list[dict] = field(default_factory=list)
    rounds: int = 0
    # Set when the answer cited markers it was never shown (see _repair).
    repair: dict | None = None
    # (tool, lowercased query) already executed this turn — shared by the
    # deterministic pre-search and the loop, so neither repeats the other.
    seen: set[tuple[str, str]] = field(default_factory=set)

    def renumber(self, block: str) -> str:
        """Shift every marker in `block` past those already issued, then
        record the new high-water marks. Markers repeated inside a block
        (e.g. a coverage notice listing [L2]) map consistently."""
        offsets = dict(self.markers)
        seen = dict(self.markers)

        def shift(m: re.Match) -> str:
            kind, num = m.group(1), int(m.group(2)) + offsets[m.group(1)]
            seen[kind] = max(seen[kind], num)
            return f"[{kind}{num}]"

        out = _MARKER_RE.sub(shift, block)
        self.markers = seen
        return out

    def web_summary(self) -> dict | None:
        if not self.web_queries:
            return None
        return {
            "triggered": True, "forced": False, "via": "tool",
            "query": " | ".join(self.web_queries),
            "results": self.web_results,
            "markers": {k: self.markers[k] for k in ("W", "X", "A")},
        }


# ── tool execution ───────────────────────────────────────────────────────────

def _parse_args(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _validate(name: str, args: dict | None, allowed: set[str]) -> tuple[str | None, str]:
    """(query, error). Exactly one is meaningful."""
    if name not in allowed:
        return None, f"unknown tool {name!r}; available: {sorted(allowed)}"
    query = (args or {}).get("query")
    if not isinstance(query, str) or not query.strip():
        return None, "argument 'query' must be a non-empty string"
    query = " ".join(query.split())
    if len(query) > _MAX_QUERY_CHARS:
        return None, f"'query' is too long (max {_MAX_QUERY_CHARS} characters)"
    return query, ""


async def _web_search(query: str, trace: Trace, *, store: bool) -> str:
    result = await search.search_and_ingest(query, store_memory=store)
    trace.web_queries.append(query)
    text = result.get("context_text") or ""
    if result.get("error") or not text.strip():
        return f"No web results for {query!r}."
    trace.web_results.extend(result.get("results") or [])
    return trace.renumber(text)


async def _library_search(query: str, trace: Trace) -> str:
    chunks = mem.filter_relevant(await mem.retrieve(query), query,
                                 min_score=CFG.memory_min_score)
    if not chunks:
        return f"No relevant library entries for {query!r}."
    trace.library_chunks.extend(chunks)
    block = mem.build_context_block(chunks, max_chars=_LIBRARY_CHARS, query=query)
    return trace.renumber(block)


async def execute(call: dict, trace: Trace, *, allowed: set[str], store: bool) -> dict:
    """Run one tool call; always returns a `tool` message for the model."""
    fn = (call or {}).get("function") or {}
    name = fn.get("name") or ""
    query, error = _validate(name, _parse_args(fn.get("arguments")), allowed)
    if error:
        trace.calls.append({"name": name, "query": None, "status": "invalid"})
        content = f"Error: {error}"
    elif (name, query.lower()) in trace.seen:
        trace.calls.append({"name": name, "query": query, "status": "duplicate"})
        content = "Already searched for exactly this — use the results above."
    else:
        trace.seen.add((name, query.lower()))
        try:
            if name == "web_search":
                content = await _web_search(query, trace, store=store)
                if not content.startswith("No web results"):
                    content = search_block_prompt(content)  # citation rules
            else:
                content = await _library_search(query, trace)
            status = "ok"
        except Exception as e:  # a failing tool must not kill the turn
            log.warning("tool %s failed: %s", name, e)
            content, status = f"Error: {name} failed ({type(e).__name__}).", "error"
        trace.calls.append({"name": name, "query": query, "status": status})
    return {"role": "tool", "tool_name": name or "unknown", "content": content}


async def prefetch_web(query: str, trace: Trace, *, store: bool) -> str | None:
    """Run a web search before the first model call (deterministic intent,
    e.g. "latest"/"today"). Recorded like a tool call — numbered, deduped,
    counted against the budget. Returns the block, or None if no results."""
    key = ("web_search", " ".join(query.split()).lower())
    if key in trace.seen:
        return None
    trace.seen.add(key)
    try:
        block = await _web_search(" ".join(query.split()), trace, store=store)
        status = "ok"
    except Exception as e:
        log.warning("pre-search failed: %s", e)
        block, status = "", "error"
    trace.calls.append({"name": "web_search", "query": " ".join(query.split()),
                        "status": status})
    # Bare block: the caller's _inject_context adds the citation rules.
    return None if not block or block.startswith("No web results") else block


# ── citation repair ──────────────────────────────────────────────────────────
# Deterministic check → one model retry with the exact error → strip whatever
# is still invalid. Interactive path, so a single revision (agentic-patterns).

def invalid_citations(text: str, markers: dict[str, int]) -> list[str]:
    """Cited markers that were never shown to the model, e.g. ["W8"]."""
    bad = set()
    for _, group in _CITE_GROUP_RE.findall(text or ""):
        for kind, num in _CITE_ONE_RE.findall(group):
            if not 1 <= int(num) <= markers.get(kind, 0):
                bad.add(f"{kind}{num}")
    return sorted(bad)


def strip_citations(text: str, invalid: list[str]) -> str:
    """Remove the given markers; drop a bracket group left empty."""
    drop = set(invalid)

    def fix(m: re.Match) -> str:
        kept = [f"{k}{n}" for k, n in _CITE_ONE_RE.findall(m.group(2))
                if f"{k}{n}" not in drop]
        return f"{m.group(1)}[{', '.join(kept)}]" if kept else ""

    return _CITE_GROUP_RE.sub(fix, text)


def _ranges(markers: dict[str, int], kinds: str = "WXAL") -> list[str]:
    return [f"[{k}1]" if markers.get(k, 0) == 1 else f"[{k}1]–[{k}{markers[k]}]"
            for k in kinds if markers.get(k, 0) > 0]


def repair_prompt(invalid: list[str], markers: dict[str, int]) -> str:
    ranges = _ranges(markers)
    cited = ", ".join(f"[{m}]" for m in invalid)
    if ranges:
        allowed = ("The sources available: " + ", ".join(ranges) + ".\n"
                   "Rewrite your answer so every citation uses only those markers. "
                   "If a claim has no supporting source among them, drop its "
                   "citation or the claim.")
    else:
        allowed = "There are no sources available: remove every citation marker."
    return (f"Your answer cites {cited}, which does not exist in the material "
            f"you were given. {allowed} Output only the corrected answer.")


def missing_prompt(markers: dict[str, int]) -> str:
    return ("Your answer uses the web search results but cites none of them. "
            "Add the source marker after each fact taken from the results, using "
            "only these markers: " + ", ".join(_ranges(markers, "WXA")) + ". "
            "If a statement is not supported by any result, say so. Keep the "
            "answer otherwise unchanged. Output only the corrected answer.")


def _web_shown(markers: dict[str, int]) -> bool:
    return any(markers.get(k, 0) > 0 for k in "WXA")


def _has_citation(text: str) -> bool:
    return bool(_CITE_GROUP_RE.search(text or ""))


async def _ask_fix(convo: list[dict], answer: str, prompt: str, base_payload: dict,
                   chat: ChatFn) -> str:
    """One tool-less retry. Returns the stripped reply, or "" on any failure."""
    payload = {**base_payload, "stream": False, "messages": convo + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": prompt},
    ]}
    payload.pop("tools", None)
    try:
        data = await chat(payload)
    except Exception as e:
        log.warning("citation repair call failed: %s", e)
        return ""
    return ((data.get("message") or {}).get("content") or "").strip()


async def _repair(convo: list[dict], answer: str, trace: Trace, base_payload: dict,
                  chat: ChatFn) -> str | None:
    """Corrected answer, or None to keep `answer` as is.

    * invalid — cites markers never shown: retry, then strip what is left.
    * missing — web results were shown but nothing is cited: retry; accept the
      rewrite only if it cites a real source, else keep the original (an
      uncited rewrite could silently drop content).
    """
    markers = trace.markers
    invalid = invalid_citations(answer, markers)
    if invalid:
        fixed = await _ask_fix(convo, answer, repair_prompt(invalid, markers),
                               base_payload, chat)
        if not fixed:
            trace.repair = {"kind": "invalid", "invalid": invalid, "resolved": False,
                            "stripped": invalid}
            return strip_citations(answer, invalid)
        still = invalid_citations(fixed, markers)
        trace.repair = {"kind": "invalid", "invalid": invalid, "resolved": not still,
                        "stripped": still}
        return strip_citations(fixed, still) if still else fixed

    if answer.strip() and _web_shown(markers) and not _has_citation(answer):
        fixed = await _ask_fix(convo, answer, missing_prompt(markers), base_payload, chat)
        still = invalid_citations(fixed, markers)
        cleaned = strip_citations(fixed, still) if still else fixed
        if not _has_citation(cleaned):
            trace.repair = {"kind": "missing", "invalid": [], "resolved": False,
                            "stripped": []}
            return None
        trace.repair = {"kind": "missing", "invalid": [], "resolved": True,
                        "stripped": still}
        return cleaned
    return None


# ── loop ─────────────────────────────────────────────────────────────────────

ChatFn = Callable[[dict], Awaitable[dict]]


async def run(messages: list[dict], base_payload: dict, tools: list[dict], *,
              store: bool, trace: Trace | None = None,
              chat: ChatFn | None = None) -> tuple[dict, Trace]:
    """Non-streaming loop. Returns (final Ollama response, trace).

    `base_payload` is a `build_payload` result (model/options/think); its
    messages are replaced each round. `chat` defaults to `oll.chat`.
    """
    trace = trace or Trace()
    chat = chat or (lambda p: oll.chat(p))
    allowed = {t["function"]["name"] for t in tools}
    convo = list(messages)
    for round_no in range(MAX_ROUNDS):
        trace.rounds = round_no + 1
        offer = tools and round_no < MAX_ROUNDS - 1 and _budget_left(trace)
        payload = {**base_payload, "messages": convo, "stream": False}
        if offer:
            payload["tools"] = tools
        else:
            payload.pop("tools", None)
        data = await chat(payload)
        msg = (data.get("message") or {})
        calls = msg.get("tool_calls") or []
        if not calls or not offer:
            fixed = await _repair(convo, msg.get("content") or "", trace,
                                  base_payload, chat)
            if fixed is not None:
                data = {**data, "message": {**msg, "content": fixed}}
            return data, trace
        # Only echo the calls we will run, so every tool_call gets a result.
        calls = calls[:_budget_left(trace)]
        convo = convo + [{"role": "assistant", "content": msg.get("content") or "",
                          "tool_calls": calls}]
        for call in calls:
            convo.append(await execute(call, trace, allowed=allowed, store=store))
    return data, trace  # unreachable: the last round offers no tools


def _budget_left(trace: Trace) -> int:
    return max(MAX_TOOL_CALLS - len(trace.calls), 0)


StreamFn = Callable[[dict], AsyncIterator[dict]]


async def stream(messages: list[dict], base_payload: dict, tools: list[dict], *,
                 store: bool, trace: Trace, on_token=None,
                 stream_fn: StreamFn | None = None,
                 chat: ChatFn | None = None) -> AsyncIterator[dict]:
    """Streaming loop. Yields events:

        {"type": "content", "text": str}            — answer tokens, as produced
        {"type": "tool_call", "name", "query"}      — before each execution
        {"type": "replace", "text", "kind",
         "invalid"}                                 — corrected full answer, if it
                                                      cited unseen markers or
                                                      cited none of the web results
        {"type": "done", "done_reason": str|None}   — once, at the end

    Each round is streamed with tools offered; content deltas are forwarded
    immediately, and any tool_calls seen in the stream are executed before the
    next round.
    """
    stream_fn = stream_fn or oll.stream_events
    chat = chat or (lambda p: oll.chat(p))
    allowed = {t["function"]["name"] for t in tools}
    convo = list(messages)
    done_reason = None
    for round_no in range(MAX_ROUNDS):
        trace.rounds = round_no + 1
        offer = tools and round_no < MAX_ROUNDS - 1 and _budget_left(trace)
        payload = {**base_payload, "messages": convo, "stream": True}
        if offer:
            payload["tools"] = tools
        else:
            payload.pop("tools", None)
        calls: list[dict] = []
        text: list[str] = []
        async for obj in stream_fn(payload):
            msg = obj.get("message") or {}
            if msg.get("tool_calls"):
                calls.extend(msg["tool_calls"])
            if msg.get("content"):
                text.append(msg["content"])
                if on_token:
                    on_token(msg["content"])
                yield {"type": "content", "text": msg["content"]}
            if obj.get("done"):
                done_reason = obj.get("done_reason")
        if not calls or not offer:
            fixed = await _repair(convo, "".join(text), trace, base_payload, chat)
            if fixed is not None:
                yield {"type": "replace", "text": fixed,
                       "kind": trace.repair["kind"],
                       "invalid": trace.repair["invalid"]}
            break
        calls = calls[:_budget_left(trace)]
        convo = convo + [{"role": "assistant", "content": "".join(text),
                          "tool_calls": calls}]
        for call in calls:
            fn = call.get("function") or {}
            args = _parse_args(fn.get("arguments")) or {}
            yield {"type": "tool_call", "name": fn.get("name"), "query": args.get("query")}
            convo.append(await execute(call, trace, allowed=allowed, store=store))
    yield {"type": "done", "done_reason": done_reason}

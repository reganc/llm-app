"""Chat completions: /v1/chat/completions, /v1/chat/reasoning, /v1/conversations/{id}/messages."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import conversations as conv_store
import memory as mem
import ollama as oll
import search
import x_search
from auth import verify_api_key
from config import CFG, get_active_model
from extract import truncate as truncate_text
from prompts import SEARCH_ADDENDUM, build_system, get_system_prompt

log = logging.getLogger("llm-api.chat")

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(verify_api_key)])

metrics: dict = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "total_errors": 0,
    "reasoning_requests": 0,
    "start_time": time.time(),
}

_SEARCH_REQUEST_RE = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)

# Phrases that signal the user wants to query their stored library / RAG corpus
# rather than the open web. Matched case-insensitively against the cleaned query.
_LIBRARY_INTENT_RE = re.compile(
    r"\b("
    r"my (library|notes|memory|knowledge ?base|saved (docs?|documents|sources|articles|notes|pages))"
    r"|in (the )?library"
    r"|from (the |my )?(library|knowledge ?base|saved (docs?|documents|sources|notes))"
    r"|what (i|i've|i have) (saved|ingested|stored|crawled|uploaded)"
    r"|the (docs?|documents|articles|pages|sites?) i (saved|ingested|stored|crawled|uploaded|added)"
    r"|library says"
    r"|in my crawls?"
    r")\b",
    re.IGNORECASE,
)

# Library mode uses source-aware deep retrieval — a smaller number of
# *sources* with full in-document context, instead of many disjoint chunks.
_LIBRARY_TOP_SOURCES = 6
_LIBRARY_MAX_CHARS = 12000
_LIBRARY_CANDIDATES = 80

# Pull URLs out of free-form queries so we can do exact-id lookups against
# stored sources alongside semantic retrieval. Trailing punctuation that's
# almost certainly not part of the URL is stripped after the match.
_URL_IN_QUERY_RE = re.compile(r"https?://[^\s<>\")']+", re.IGNORECASE)
_URL_TRAILING_PUNCT = '.,;:!?)]}>\'"`'

# Detect "title-shaped" phrases — quoted strings (straight or smart quotes)
# or long runs of capitalized words. Used to short-circuit retrieval when the
# user pastes a literal article title that BM25/vector retrieval would miss
# because the BM25 tsv only indexes body text.
_QUOTED_PHRASE_RE = re.compile(
    r'[“”"]([^“”"]{12,300})[“”"]'
)
_TITLE_STOPWORDS = {
    "of", "the", "to", "and", "in", "on", "for", "a", "an", "or",
    "from", "by", "as", "with", "vs", "at", "is", "are",
}
_WORD_RE = re.compile(r"[\w’'\-]+", re.UNICODE)

# Above this best-chunk score, the auto-search path puts library adjacent to
# the user message (model attends to it first) but still allows search to
# fire for time-sensitive queries.
_MEMORY_TRUST_SCORE = 0.60


# ── Request models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str | list

    def text(self) -> str:
        if isinstance(self.content, list):
            return " ".join(p.get("text", "") for p in self.content
                            if isinstance(p, dict) and p.get("type") == "text")
        return self.content


class ChatRequest(BaseModel):
    model: str = Field(default_factory=get_active_model)
    messages: list[Message]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] | None = None
    system_prompt_key: str | None = None
    inject_system: bool = True
    stream_options: dict | None = None  # accepted, ignored (OpenAI compat)

    model_config = {"extra": "ignore"}

    @field_validator("max_tokens")
    @classmethod
    def clamp_max_tokens(cls, v: int) -> int:
        return min(v, CFG.max_tokens)


class ReasoningRequest(BaseModel):
    model: str = Field(default_factory=get_active_model)
    messages: list[Message]
    stream: bool = False
    system_prompt_key: str = "reasoning"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096)


class CompletionRequest(BaseModel):
    model: str = Field(default_factory=get_active_model)
    prompt: str
    system: str | None = None
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512

    model_config = {"extra": "ignore"}


class ConversationCreateRequest(BaseModel):
    name: str = "New Conversation"
    system_prompt_key: str = "default"
    custom_system_prompt: str | None = None


class ConversationMessageRequest(BaseModel):
    content: str
    model: str = Field(default_factory=get_active_model)
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────
def _trim(text: str) -> str:
    if len(text) <= CFG.max_inject_chars:
        return text
    return text[:CFG.max_inject_chars] + "\n[... context trimmed ...]"


def _urls_in_query(query: str) -> list[str]:
    if not query:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in _URL_IN_QUERY_RE.findall(query):
        url = raw.rstrip(_URL_TRAILING_PUNCT)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _extract_title_phrase(query: str) -> str | None:
    """Return a candidate article-title phrase if the query contains one.

    Matches:
      1. Anything inside straight or smart double quotes (12+ chars).
      2. A long run of Title-Cased words (≥4 capitalized tokens), allowing
         common stopwords ("of", "the", "to", …) to glue the run together.

    Returns ``None`` when the query has neither shape.
    """
    if not query:
        return None
    m = _QUOTED_PHRASE_RE.search(query)
    if m:
        phrase = m.group(1).strip()
        if len(phrase) >= 12:
            return phrase
    words = _WORD_RE.findall(query)
    best: list[str] = []
    cur: list[str] = []
    cur_caps = 0
    for w in words:
        if w[:1].isupper():
            cur.append(w)
            cur_caps += 1
        elif cur and w.lower() in _TITLE_STOPWORDS:
            cur.append(w)
        else:
            if cur_caps >= 4 and len(cur) > len(best):
                best = cur
            cur, cur_caps = [], 0
    if cur_caps >= 4 and len(cur) > len(best):
        best = cur
    if best:
        # Trim trailing stopwords so "Triangle To" doesn't end with "To".
        while best and best[-1].lower() in _TITLE_STOPWORDS:
            best.pop()
        if len(best) >= 4:
            return " ".join(best)
    return None


def _merge_url_matches(chunks: list[dict], url_matches: list[dict]) -> list[dict]:
    """Prepend exact-URL matches to chunks, dedup by identifier, cap to
    ``_LIBRARY_TOP_SOURCES`` so the prompt doesn't blow past its budget."""
    if not url_matches:
        return chunks
    out: list[dict] = list(url_matches)
    seen_ids = {c.get("identifier") for c in url_matches if c.get("identifier")}
    for c in chunks:
        if c.get("identifier") in seen_ids:
            continue
        out.append(c)
        if len(out) >= _LIBRARY_TOP_SOURCES:
            break
    return out


def _parse_command(raw: str) -> tuple[str, str | None, str]:
    """Return (cleaned_query, command, raw_query).

    command in {None, 'search', 'x', 'library'}. ``/library`` and ``/lib``
    scope retrieval to the local Chroma knowledge collection and skip web
    search entirely.
    """
    low = raw.lower()
    if low.startswith("/x "):
        return raw[3:].strip(), "x", raw
    if low.startswith("/search "):
        return raw[8:].strip(), "search", raw
    if low.startswith("/library "):
        return raw[9:].strip(), "library", raw
    if low.startswith("/lib "):
        return raw[5:].strip(), "library", raw
    return raw, None, raw


def _inject_messages(messages: list[Message], system_key: str | None,
                     search_capable: bool) -> list[dict]:
    msgs = [{"role": m.role, "content": m.text()} for m in messages]
    if not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": build_system(system_key, search_capable)})
    return msgs


def _add_search_capability(messages: list[dict]) -> list[dict]:
    return [
        {**m, "content": m["content"] + SEARCH_ADDENDUM}
        if m["role"] == "system" else m
        for m in messages
    ]


def _strip_search_capability(messages: list[dict]) -> list[dict]:
    return [
        {**m, "content": m["content"].replace(SEARCH_ADDENDUM, "")}
        if m["role"] == "system" else m
        for m in messages
    ]


def _library_block_prompt(memory_block: str, *, primary: bool) -> str:
    """Render the [RELEVANT CONTEXT FROM YOUR LIBRARY] user-message body.

    When ``primary`` is True the preamble matches the strict, citation-heavy
    style we use for live search so the model doesn't quietly defer to its
    training data instead of the saved corpus, AND we trust the caller's
    ``mem.build_context_block`` budget (no extra ``_trim``) so deep bundles
    aren't truncated below their intended size.
    """
    if primary:
        head = (
            "RELEVANT CONTEXT FROM THE USER'S LIBRARY — documents and pages the user "
            "explicitly saved or crawled into local storage. The user is asking about "
            "their own library; this is the AUTHORITATIVE source for answering and it "
            "supersedes your training data on these topics.\n\n"
            "Each [L#] block below is ONE source from the library. Its title and source "
            "URL appear at the top of the block; the body is matched passages from that "
            "document reassembled in order ([…] marks an in-document gap). Each [L#] is "
            "a DIFFERENT source — never describe two markers as 'the same page'.\n\n"
            "If the block starts with a 'QUERY TOPICS:' coverage notice, READ IT FIRST. "
            "It tells you, ground-truth, which [L#] blocks contain all the topics the "
            "user asked about, which contain only one, and which contain none. You must "
            "obey that notice when reasoning about what the library actually covers.\n\n"
            "STRICT RULES — follow these exactly:\n"
            "1. Cite library sources inline using their bracket marker — e.g. [L1], [L2]. "
            "The marker appears at the start of each block below; use it like footnotes.\n"
            "2. PREFER QUOTING. When the user asks for a specific name, date, claim, or "
            "passage, copy the wording verbatim from the relevant block.\n"
            "3. If the user is asking whether a specific page or document IS in their "
            "library: answer yes or no first using the SOURCE URLs/titles in the block "
            "headers (a URL match in a header is conclusive). Then briefly summarize "
            "the SUBSTANTIVE content of that source — the actual subjects, names, and "
            "claims it covers — NOT site-wide boilerplate (footers, sticky disclaimers, "
            "navigation lists, generic 'about this report' notices).\n"
            "4. CONNECTION QUESTIONS — when the user asks about the relationship "
            "between two or more topics ('what does X have to do with Y', 'X and Y', "
            "'how is X connected to Y'): you may ONLY claim a connection exists if you "
            "can quote a passage from a single [L#] block that mentions BOTH topics by "
            "name. Listing facts about X alone — even from many sources — is NOT an "
            "answer. If the QUERY TOPICS notice says 'All topics present in: NONE', "
            "or if you cannot find such a passage, say plainly: 'Your library does not "
            "appear to connect [X] and [Y] — [X] is mentioned in [L#] (e.g. \"...\"), "
            "and [Y] is mentioned in [L#] (e.g. \"...\"), but no saved page links them.' "
            "Do not invent a connection. Do not pad with unrelated quotes.\n"
            "5. If the specific answer is NOT present in the library blocks below, say so "
            "explicitly: 'Your library does not contain a direct answer — the closest match "
            "is [L#]: ...'. Do NOT silently fall back to training data.\n"
            "6. If multiple library entries conflict, present both with their [L#] citations.\n"
            "7. Maintain continuity with the conversation above — if the user has been "
            "discussing a topic and now references a saved page, surface the parts of that "
            "page that connect to the ongoing thread.\n\n"
        )
        return head + memory_block  # build_context_block already enforced budget
    head = (
        "[RELEVANT CONTEXT FROM YOUR LIBRARY — these are documents the user has "
        "explicitly saved. When the user asks about a specific document by name, "
        "this is your primary source — quote and cite from it using the [L#] "
        "marker at the start of each block. Maintain continuity with the "
        "conversation above.]\n"
    )
    return head + _trim(memory_block)


def _search_block_prompt(search_block: str) -> str:
    return (
        "Live web search results, fetched moments ago. Use these as the authoritative "
        "answer for any current/factual claim — they supersede your training data.\n\n"
        "Rules:\n"
        "1. Find and STATE the answer. Quote numbers verbatim ($4,715.06 stays as "
        "$4,715.06). Cite every fact with the source's [W#] / [X#] / [A#] marker.\n"
        "2. Prefer numbers paired with today's date, this week, or words like 'now', "
        "'today', 'currently', 'spot', 'live', 'as of'. Treat them as the current value "
        "and report them as such.\n"
        "3. Numbers tied to old dates ('reached $X in 1980', 'record high in 2020') are "
        "historical — do not present them as the current value, and do not extrapolate "
        "or average a current price from them.\n"
        "4. If every number you find is historical: state that plainly, give the most "
        "recent dated figure with its date and citation, and stop. Do not guess a range.\n"
        "5. Do not invent [L#] markers (those are memory, not search). Cite only the "
        "[W#] / [X#] / [A#] markers that appear in the block below — not markers from "
        "earlier turns in the conversation.\n"
        "6. Do not echo these rules in your answer. Just answer.\n\n"
        + search_block
    )


def _inject_context(messages: list[dict], *, search_block: str | None,
                    memory_block: str | None,
                    library_primary: bool = False,
                    memory_first: bool = False) -> list[dict]:
    """Inject memory and search blocks before the final user message.

    Both blocks are inserted at index -1, so the second insert ends up
    adjacent to the user message. Whichever block is "primary" is inserted
    LAST so it lands closest to the user's question — models attend more
    strongly to recency, and proximity matters when blocks compete.

    library_primary=True suppresses the search block entirely (caller has
    already decided this is a library-only request) and uses the strict
    library preamble.

    memory_first=True puts the library block closer to the user message
    even when web search also fires — used when memory has a strong hit so
    the saved corpus isn't drowned out by the (larger) search instructions.
    """
    out = list(messages)
    library_ack = "Reviewed the relevant library content. I'll cite [L#] for every fact and quote verbatim where possible."
    search_ack = "Understood. I'll cite [W#]/[X#]/[A#] for every fact, quote exact figures from the results, and explicitly say so if the specific answer isn't present."

    def insert_library() -> None:
        if not memory_block:
            return
        out.insert(-1, {"role": "user",
                        "content": _library_block_prompt(memory_block, primary=library_primary)})
        out.insert(-1, {"role": "assistant", "content": library_ack})

    def insert_search() -> None:
        if not search_block:
            return
        out.insert(-1, {"role": "user", "content": _search_block_prompt(search_block)})
        out.insert(-1, {"role": "assistant", "content": search_ack})

    if library_primary:
        # Pure-library mode: caller passed search_block=None, but guard anyway.
        insert_library()
        return out

    if memory_block and not search_block:
        insert_library()
    elif search_block and not memory_block:
        insert_search()
    elif memory_block and search_block:
        # When memory has a strong hit, swap the order so the library lands
        # adjacent to the user message; otherwise keep search-adjacent
        # (the legacy behavior for ambiguous queries).
        if memory_first:
            insert_search()
            insert_library()
        else:
            insert_library()
            insert_search()
    return out


async def _resolve_context(query: str, command: str | None,
                           stream: bool, allow_two_pass: bool
                           ) -> tuple[dict | None, list[dict], bool, list[str], str | None]:
    """Run memory retrieval + optional search up-front.

    Returns ``(search_result_or_None, memory_chunks, should_two_pass,
    intent_signals, resolved_command)``. ``resolved_command`` is the command
    after intent promotion ("library" | "search" | "x" | None) so the caller
    can render the right context block.
    """
    intent_signals: list[str] = []
    if not CFG.memory_enabled:
        return None, [], allow_two_pass, intent_signals, command

    urls = _urls_in_query(query)

    # Library intent — explicit command, or natural-language phrasing like
    # "what does my library say about X". Promote either to library mode so
    # we skip web search entirely and pull a wider slice of the knowledge
    # collection. A query that contains a URL plus library phrasing ("is
    # this in my library?", "in my library https://…") is also routed here
    # so we look up the page directly instead of running a semantic search
    # whose top hits will be sitewide boilerplate.
    if not command and query and _LIBRARY_INTENT_RE.search(query):
        command = "library"
        intent_signals.append("library_phrase")
        log.info("intent: routing to /library (phrase match)")

    # Title-lookup short-circuit — applies to BOTH library and auto modes.
    # When the query quotes or capitalizes a literal article title, look it
    # up by title before semantic retrieval so we don't ride BM25 noise on
    # incidental keywords (e.g. "cheese" in a metaphorical headline pulling
    # dairy articles to the top).
    title_matches: list[dict] = []
    title_phrase = _extract_title_phrase(query) if query else None
    if title_phrase:
        try:
            title_matches = await mem.lookup_by_title(
                title_phrase,
                max_chars=_LIBRARY_MAX_CHARS // 2,
                limit=2,
            )
        except Exception as e:
            log.warning("title lookup failed: %s", e)
            title_matches = []
        if title_matches:
            intent_signals.append("title_match")
            log.info("intent: title-lookup hit on %r (%d sources)",
                     title_phrase[:80], len(title_matches))
            # A direct title hit is library-ish by nature: route to library
            # mode so the strict "answer from saved corpus" prompt fires and
            # web-search is skipped.
            if not command:
                command = "library"
                intent_signals.append("library_via_title")

    if command == "library":
        chunks: list[dict] = []
        url_matches: list[dict] = []
        if urls:
            for u in urls:
                hit = await mem.lookup_by_identifier(u, max_chars=_LIBRARY_MAX_CHARS // 2)
                if hit:
                    url_matches.append(hit)
            if url_matches:
                intent_signals.append("library_url_match")
        if query:
            chunks = await mem.retrieve_deep(
                query,
                top_sources=_LIBRARY_TOP_SOURCES,
                max_chars=_LIBRARY_MAX_CHARS,
                candidate_chunks=_LIBRARY_CANDIDATES,
            )
        chunks = _merge_url_matches(chunks, url_matches)
        chunks = _merge_url_matches(chunks, title_matches)
        return None, chunks, False, intent_signals, command

    chunks = await mem.retrieve(query) if query else []
    if title_matches:
        seen_ids = {c.get("identifier") for c in title_matches if c.get("identifier")}
        chunks = title_matches + [c for c in chunks
                                  if c.get("identifier") not in seen_ids]
    search_result = None
    two_pass = allow_two_pass

    # Promote deterministic intent into a command when the user didn't type one.
    if not command and query and CFG.search_enabled:
        intent = search.detect_intent(query)
        intent_signals = intent.get("signals", []) or []
        if intent.get("force_x") and CFG.x_search_enabled:
            command = "x"
            log.info("intent: routing to /x (signals=%s)", intent_signals)
        elif intent.get("force_search"):
            command = "search"
            log.info("intent: forcing /search (signals=%s)", intent_signals)

    if command == "x":
        search_result = await x_search.search_x_and_ingest(query, store_memory=True)
        two_pass = False
    elif command == "search":
        search_result = await search.search_and_ingest(query, store_memory=True)
        two_pass = False
    elif stream and CFG.search_enabled:
        if await search.should_auto_search(chunks, query):
            search_result = await search.search_and_ingest(query, store_memory=True)
            if search_result.get("stored", 0) > 0:
                chunks = await mem.retrieve(query)
            two_pass = False
    elif CFG.search_enabled and await search.should_auto_search(chunks, query):
        search_result = await search.search_and_ingest(query, store_memory=True)
        if search_result.get("stored", 0) > 0:
            chunks = await mem.retrieve(query)
        two_pass = False

    return search_result, chunks, two_pass, intent_signals, command


async def _two_pass_chat(messages: list[dict], model: str, temperature: float,
                         max_tokens: int, top_p: float, stop: list | None) -> tuple[dict, dict | None]:
    payload = oll.build_payload(messages, model, temperature=temperature,
                                max_tokens=max_tokens, top_p=top_p, stop=stop)
    data = await oll.chat(payload)
    content = data.get("message", {}).get("content", "").strip()

    m = _SEARCH_REQUEST_RE.search(content)
    if not m:
        return data, None
    query = m.group(1).strip()
    result = await search.search_and_ingest(query, store_memory=True)
    if result.get("error") or not result.get("context_text"):
        retry = oll.build_payload(_strip_search_capability(messages), model,
                                  temperature=temperature, max_tokens=max_tokens,
                                  top_p=top_p, stop=stop)
        return await oll.chat(retry), None

    follow = messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": result["context_text"]},
    ]
    payload2 = oll.build_payload(follow, model, temperature=temperature,
                                 max_tokens=max_tokens, top_p=top_p, stop=stop)
    return await oll.chat(payload2), result


def _search_summary(search_result: dict | None, command: str | None,
                    intent_signals: list[str] | None = None) -> dict | None:
    if not search_result:
        return None
    return {
        "triggered": True,
        "forced": command in ("search", "x"),
        "query": search_result.get("query"),
        "stored": search_result.get("stored", 0),
        "source": search_result.get("source", "unknown"),
        "retrieved_at": search_result.get("retrieved_at", ""),
        "results": search_result.get("results", []),
        "intent_signals": intent_signals or [],
    }


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def _memory_summary(chunks: list[dict] | None, command: str | None) -> dict | None:
    """Compact summary of which library entries fed the answer.

    Surfaces the [L#] marker, title, and source URL so the SPA can show a
    "Memory · N chunks" badge and (later) a side rail for library hits.
    """
    if not chunks:
        return None
    return {
        "used": len(chunks),
        "mode": "library" if command == "library" else "auto",
        "best_score": max((c.get("score", 0) for c in chunks), default=0),
        "items": [
            {
                "marker": f"L{i}",
                "title": c.get("title") or "",
                "identifier": c.get("identifier") or "",
                "score": c.get("score", 0),
                "source_type": c.get("source_type") or "",
            }
            for i, c in enumerate(chunks, 1)
        ],
    }


async def _stream_with_summary(payload: dict, *, search_result: dict | None,
                               command: str | None, message_id: str,
                               intent_signals: list[str] | None = None,
                               memory_chunks: list[dict] | None = None,
                               on_token=None):
    """Yield Ollama SSE chunks. Emits llm.message_id first, then content,
    then llm.memory + llm.web_search summaries (if any) just before [DONE]."""
    search_sum = _search_summary(search_result, command, intent_signals)
    mem_sum = _memory_summary(memory_chunks, command)
    yield f"event: llm.message_id\ndata: {json.dumps({'id': message_id})}\n\n"
    if mem_sum:
        yield f"event: llm.memory\ndata: {json.dumps(mem_sum)}\n\n"
    saw_done = False
    async for chunk in oll.stream_chat(payload, on_token=on_token):
        if chunk.startswith("data: [DONE]"):
            saw_done = True
            if search_sum:
                yield f"event: llm.web_search\ndata: {json.dumps(search_sum)}\n\n"
            yield chunk
        else:
            yield chunk
    if not saw_done and search_sum:
        yield f"event: llm.web_search\ndata: {json.dumps(search_sum)}\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    metrics["total_requests"] += 1
    raw_query = req.messages[-1].text() if req.messages else ""
    clean, command, _ = _parse_command(raw_query)
    if command:
        req.messages[-1] = Message(role="user", content=clean)

    allow_two_pass = (
        CFG.search_enabled and req.inject_system
        and not req.stream and not command
    )
    search_result, chunks, two_pass, intent_signals, command = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    messages = (
        _inject_messages(req.messages, req.system_prompt_key, search_capable=False)
        if req.inject_system else [m.model_dump() for m in req.messages]
    )

    library_primary = command == "library"
    # When the user explicitly typed /search or /x, the *fresh* results are
    # the answer they want — suppress the [L#] memory block. Prior searches
    # are stored to memory as side-effects, and re-injecting them as
    # "library context" both dilutes the model's attention and conflicts
    # with the strict search-extraction prompt.
    search_primary = command in ("search", "x")
    best_score = max((c.get("score", 0) for c in chunks), default=0)
    memory_first = best_score >= _MEMORY_TRUST_SCORE
    mem_chars = _LIBRARY_MAX_CHARS if library_primary else 3000
    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = (
        None if search_primary
        else (mem.build_context_block(chunks, max_chars=mem_chars, query=clean)
              if chunks else None)
    )
    messages = _inject_context(
        messages,
        search_block=None if library_primary else search_block,
        memory_block=memory_block,
        library_primary=library_primary,
        memory_first=memory_first,
    )

    final_messages = _add_search_capability(messages) if two_pass else messages

    payload = oll.build_payload(final_messages, req.model,
                                temperature=req.temperature, max_tokens=req.max_tokens,
                                top_p=req.top_p, stop=req.stop, stream=req.stream)
    message_id = _new_message_id()
    if req.stream:
        async def gen():
            buf: list[str] = []
            async for chunk in _stream_with_summary(
                payload, search_result=search_result, command=command,
                message_id=message_id, intent_signals=intent_signals,
                memory_chunks=chunks, on_token=buf.append,
            ):
                yield chunk
            reply_text = "".join(buf)
            if CFG.memory_enabled and req.inject_system and req.messages:
                asyncio.create_task(mem.store_conversation_turn(
                    conv_id=f"chat_{int(time.time())}",
                    conv_name=f"Chat ({req.system_prompt_key or 'default'})",
                    user_msg=clean, assistant_msg=reply_text,
                    message_id=message_id,
                ))
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        if two_pass:
            data, search_result = await _two_pass_chat(
                final_messages, req.model, req.temperature, req.max_tokens,
                req.top_p, req.stop,
            )
        else:
            data = await oll.chat(payload)
        reply = data.get("message", {}).get("content", "")
        metrics["total_tokens_generated"] += data.get("eval_count", 0)
        if CFG.memory_enabled and req.inject_system and req.messages:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id=f"chat_{int(time.time())}",
                conv_name=f"Chat ({req.system_prompt_key or 'default'})",
                user_msg=clean, assistant_msg=reply,
                message_id=message_id,
            ))
        extra: dict = {"message_id": message_id}
        ws_summary = _search_summary(search_result, command, intent_signals)
        if ws_summary:
            extra["web_search"] = ws_summary
        mem_summary = _memory_summary(chunks, command)
        if mem_summary:
            extra["memory"] = mem_summary
            extra["memory_used"] = mem_summary["used"]
        return oll.completion_envelope(data, req.model, extra=extra)
    except Exception as e:
        metrics["total_errors"] += 1
        log.exception("chat completion failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/reasoning")
async def reasoning(req: ReasoningRequest):
    metrics["total_requests"] += 1
    metrics["reasoning_requests"] += 1
    raw_query = req.messages[-1].text() if req.messages else ""
    clean, command, _ = _parse_command(raw_query)
    if command:
        req.messages[-1] = Message(role="user", content=clean)

    allow_two_pass = CFG.search_enabled and not req.stream and not command
    search_result, chunks, two_pass, intent_signals, command = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    messages = _inject_messages(req.messages, req.system_prompt_key, search_capable=False)
    library_primary = command == "library"
    search_primary = command in ("search", "x")
    best_score = max((c.get("score", 0) for c in chunks), default=0)
    memory_first = best_score >= _MEMORY_TRUST_SCORE
    mem_chars = _LIBRARY_MAX_CHARS if library_primary else 4000
    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = (
        None if search_primary
        else (mem.build_context_block(chunks, max_chars=mem_chars, query=clean)
              if chunks else None)
    )
    messages = _inject_context(
        messages,
        search_block=None if library_primary else search_block,
        memory_block=memory_block,
        library_primary=library_primary,
        memory_first=memory_first,
    )

    final_messages = _add_search_capability(messages) if two_pass else messages
    payload = oll.build_payload(final_messages, req.model,
                                temperature=req.temperature, max_tokens=req.max_tokens,
                                top_p=0.95, stream=req.stream, thinking=True)
    message_id = _new_message_id()
    if req.stream:
        async def gen():
            buf: list[str] = []
            async for chunk in _stream_with_summary(
                payload, search_result=search_result, command=command,
                message_id=message_id, intent_signals=intent_signals,
                memory_chunks=chunks, on_token=buf.append,
            ):
                yield chunk
            reply_text = "".join(buf)
            if CFG.memory_enabled and req.messages:
                asyncio.create_task(mem.store_conversation_turn(
                    conv_id=f"reasoning_{int(time.time())}",
                    conv_name=f"Reasoning ({req.system_prompt_key})",
                    user_msg=clean, assistant_msg=reply_text,
                    message_id=message_id,
                ))
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        if two_pass:
            data, search_result = await _two_pass_chat(
                final_messages, req.model, req.temperature, req.max_tokens, 0.95, None,
            )
        else:
            data = await oll.chat(payload)
        reply = data.get("message", {}).get("content", "")
        metrics["total_tokens_generated"] += data.get("eval_count", 0)
        if CFG.memory_enabled and req.messages:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id=f"reasoning_{int(time.time())}",
                conv_name=f"Reasoning ({req.system_prompt_key})",
                user_msg=clean, assistant_msg=reply,
                message_id=message_id,
            ))
        extra: dict = {"reasoning_mode": True,
                       "system_prompt_used": req.system_prompt_key,
                       "message_id": message_id}
        ws_summary = _search_summary(search_result, command, intent_signals)
        if ws_summary:
            extra["web_search"] = ws_summary
        mem_summary = _memory_summary(chunks, command)
        if mem_summary:
            extra["memory"] = mem_summary
            extra["memory_used"] = mem_summary["used"]
        return oll.completion_envelope(data, req.model, extra=extra)
    except Exception as e:
        metrics["total_errors"] += 1
        log.exception("reasoning failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def completions(req: CompletionRequest):
    metrics["total_requests"] += 1
    try:
        data = await oll.generate(req.prompt, model=req.model, system=req.system,
                                  temperature=req.temperature, max_tokens=req.max_tokens,
                                  num_ctx=CFG.num_ctx)
        metrics["total_tokens_generated"] += data.get("eval_count", 0)
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"text": data.get("response", ""), "index": 0,
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
    except Exception as e:
        metrics["total_errors"] += 1
        log.exception("completion failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Conversations ────────────────────────────────────────────────────────────
@router.post("/conversations")
async def create_conversation(req: ConversationCreateRequest):
    return conv_store.create(req.name, req.system_prompt_key, req.custom_system_prompt)


@router.get("/conversations")
async def list_conversations():
    return {"conversations": conv_store.list_all()}


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = conv_store.get(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if not conv_store.delete(conv_id):
        raise HTTPException(404, "Conversation not found")
    return {"deleted": conv_id}


@router.post("/conversations/{conv_id}/messages")
async def conversation_message(conv_id: str, req: ConversationMessageRequest):
    conv = conv_store.get(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    clean, command, _ = _parse_command(req.content)
    conv = conv_store.append_message(conv_id, "user", clean)

    fresh_system = build_system(conv.get("system_prompt_key"))
    base = [
        {**m, "content": fresh_system} if m["role"] == "system" else m
        for m in conv["messages"]
    ]

    allow_two_pass = CFG.search_enabled and not req.stream and not command
    search_result, chunks, two_pass, intent_signals, command = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    library_primary = command == "library"
    search_primary = command in ("search", "x")
    best_score = max((c.get("score", 0) for c in chunks), default=0)
    memory_first = best_score >= _MEMORY_TRUST_SCORE
    mem_chars = _LIBRARY_MAX_CHARS if library_primary else 3000
    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = (
        None if search_primary
        else (mem.build_context_block(chunks, max_chars=mem_chars, query=clean)
              if chunks else None)
    )
    base = _inject_context(
        base,
        search_block=None if library_primary else search_block,
        memory_block=memory_block,
        library_primary=library_primary,
        memory_first=memory_first,
    )

    final_send = _add_search_capability(base) if two_pass else base
    payload = oll.build_payload(final_send, req.model,
                                temperature=req.temperature, max_tokens=req.max_tokens,
                                stream=req.stream)
    message_id = _new_message_id()
    if req.stream:
        async def gen():
            buf: list[str] = []
            async for chunk in _stream_with_summary(
                payload, search_result=search_result, command=command,
                message_id=message_id, intent_signals=intent_signals,
                memory_chunks=chunks, on_token=buf.append,
            ):
                yield chunk
            reply_text = "".join(buf)
            conv_store.append_message(conv_id, "assistant", reply_text, message_id=message_id)
            if CFG.memory_enabled:
                asyncio.create_task(mem.store_conversation_turn(
                    conv_id, conv["name"], clean, reply_text,
                    message_id=message_id,
                ))
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        if two_pass:
            data, search_result = await _two_pass_chat(
                final_send, req.model, req.temperature, req.max_tokens, 0.9, None,
            )
        else:
            data = await oll.chat(payload)
        reply = data.get("message", {}).get("content", "")
        conv = conv_store.append_message(conv_id, "assistant", reply, message_id=message_id)
        metrics["total_tokens_generated"] += data.get("eval_count", 0)
        if CFG.memory_enabled:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id, conv["name"], clean, reply,
                message_id=message_id,
            ))
        out = {
            "conversation_id": conv_id,
            "turn": conv["turn_count"],
            "reply": reply,
            "message_id": message_id,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
        ws_summary = _search_summary(search_result, command, intent_signals)
        if ws_summary:
            out["web_search"] = ws_summary
        mem_summary = _memory_summary(chunks, command)
        if mem_summary:
            out["memory"] = mem_summary
            out["memory_used"] = mem_summary["used"]
        return out
    except Exception as e:
        conv_store.pop_last(conv_id)
        metrics["total_errors"] += 1
        log.exception("conv message failed")
        raise HTTPException(status_code=500, detail=str(e))

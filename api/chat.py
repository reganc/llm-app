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
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512


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


def _parse_command(raw: str) -> tuple[str, str | None, str]:
    """Return (cleaned_query, command, raw_query). command in {None, 'search', 'x'}."""
    low = raw.lower()
    if low.startswith("/x "):
        return raw[3:].strip(), "x", raw
    if low.startswith("/search "):
        return raw[8:].strip(), "search", raw
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


def _inject_context(messages: list[dict], *, search_block: str | None,
                    memory_block: str | None) -> list[dict]:
    out = list(messages)
    if memory_block:
        out.insert(-1, {
            "role": "user",
            "content": ("[RELEVANT CONTEXT FROM YOUR LIBRARY — these are documents the user "
                        "has explicitly saved. When the user asks about a specific document "
                        "by name, this is your primary source — quote and cite from it. "
                        "Maintain continuity with the conversation above.]\n"
                        f"{_trim(memory_block)}"),
        })
        out.insert(-1, {
            "role": "assistant",
            "content": "Reviewed the relevant library content. I'll ground my answer in it.",
        })
    if search_block:
        out.insert(-1, {
            "role": "user",
            "content": (
                "LIVE WEB SEARCH RESULTS for the user's latest question. The results below were "
                "fetched moments ago and supersede your training data for any current/factual claim.\n\n"
                "STRICT RULES — follow these exactly:\n"
                "1. For every specific fact, number, date, price, name, or quote in your answer, "
                "cite the source inline using its bracket marker — e.g. [W2], [X1], [A1]. The marker "
                "appears at the start of each block below.\n"
                "2. PREFER QUOTING. When the question asks for a number/price/date/status, copy the "
                "exact figure from the results verbatim. Do NOT round, average, or paraphrase numbers "
                "from your training data.\n"
                "3. If the specific answer the user asked for is NOT present in the results below, "
                "say so explicitly: 'The search results don't contain a specific [thing] — the closest "
                "I found is [W#]: ...'. Do NOT fall back on your training data for current values.\n"
                "4. If the results conflict, present both with their citations and note the discrepancy.\n"
                "5. Maintain continuity with the conversation above — resolve pronouns against prior turns.\n\n"
                + _trim(search_block)),
        })
        out.insert(-1, {
            "role": "assistant",
            "content": "Understood. I'll cite [W#]/[X#]/[A#] for every fact, quote exact figures from the results, and explicitly say so if the specific answer isn't present.",
        })
    return out


async def _resolve_context(query: str, command: str | None,
                           stream: bool, allow_two_pass: bool) -> tuple[dict | None, list[dict], bool, list[str]]:
    """
    Run memory retrieval + optional search up-front.
    Returns (search_result_or_None, memory_chunks, should_two_pass, intent_signals).
    """
    intent_signals: list[str] = []
    if not CFG.memory_enabled:
        return None, [], allow_two_pass, intent_signals

    chunks = await mem.retrieve(query) if query else []
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

    return search_result, chunks, two_pass, intent_signals


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


async def _stream_with_summary(payload: dict, *, search_result: dict | None,
                               command: str | None, message_id: str,
                               intent_signals: list[str] | None = None,
                               on_token=None):
    """Yield Ollama SSE chunks. Emits llm.message_id first, then content,
    then llm.web_search summary (if any) just before [DONE]."""
    summary = _search_summary(search_result, command, intent_signals)
    yield f"event: llm.message_id\ndata: {json.dumps({'id': message_id})}\n\n"
    saw_done = False
    async for chunk in oll.stream_chat(payload, on_token=on_token):
        if chunk.startswith("data: [DONE]"):
            saw_done = True
            if summary:
                yield f"event: llm.web_search\ndata: {json.dumps(summary)}\n\n"
            yield chunk
        else:
            yield chunk
    if not saw_done and summary:
        yield f"event: llm.web_search\ndata: {json.dumps(summary)}\n\n"


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
    search_result, chunks, two_pass, intent_signals = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    messages = (
        _inject_messages(req.messages, req.system_prompt_key, search_capable=False)
        if req.inject_system else [m.model_dump() for m in req.messages]
    )

    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = mem.build_context_block(chunks) if chunks else None
    messages = _inject_context(messages, search_block=search_block, memory_block=memory_block)

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
                on_token=buf.append,
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
        extra = {"message_id": message_id}
        ws_summary = _search_summary(search_result, command, intent_signals)
        if ws_summary:
            extra["web_search"] = ws_summary
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
    search_result, chunks, two_pass, intent_signals = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    messages = _inject_messages(req.messages, req.system_prompt_key, search_capable=False)
    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = mem.build_context_block(chunks, max_chars=4000) if chunks else None
    messages = _inject_context(messages, search_block=search_block, memory_block=memory_block)

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
                on_token=buf.append,
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
        extra = {"reasoning_mode": True, "system_prompt_used": req.system_prompt_key,
                 "message_id": message_id}
        ws_summary = _search_summary(search_result, command, intent_signals)
        if ws_summary:
            extra["web_search"] = ws_summary
        return oll.completion_envelope(data, req.model, extra=extra)
    except Exception as e:
        metrics["total_errors"] += 1
        log.exception("reasoning failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def completions(req: CompletionRequest):
    metrics["total_requests"] += 1
    try:
        data = await oll.generate(req.prompt, model=req.model,
                                  temperature=req.temperature, max_tokens=req.max_tokens)
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
    search_result, chunks, two_pass, intent_signals = await _resolve_context(
        clean, command, req.stream, allow_two_pass,
    )

    search_block = search_result["context_text"] if search_result and search_result.get("context_text") else None
    memory_block = mem.build_context_block(chunks) if chunks else None
    base = _inject_context(base, search_block=search_block, memory_block=memory_block)

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
                on_token=buf.append,
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
        return out
    except Exception as e:
        conv_store.pop_last(conv_id)
        metrics["total_errors"] += 1
        log.exception("conv message failed")
        raise HTTPException(status_code=500, detail=str(e))

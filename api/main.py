"""
Custom LLM API — OpenAI-compatible REST wrapper over Ollama
RTX 3060 optimized — Mistral 7B Q4_K_M backend

Endpoints:
  POST /v1/chat/completions            — OpenAI-compatible chat (streaming + standard)
  POST /v1/completions                 — Raw text completion
  GET  /v1/models                      — List available models
  GET  /health                         — Health check
  GET  /metrics                        — Basic usage metrics

  — Enhanced —
  POST /v1/chat/reasoning              — Deep reasoning mode (chain-of-thought, low temp)
  GET  /v1/system-prompts              — List available system prompt presets
  POST /v1/conversations               — Create named persistent conversation
  GET  /v1/conversations               — List all conversations
  GET  /v1/conversations/{id}          — Get full conversation history
  POST /v1/conversations/{id}/messages — Send message, get reply (stateful)
  DELETE /v1/conversations/{id}        — Delete conversation
  POST /v1/finetune/trigger            — Trigger Unsloth QLoRA fine-tune job
  GET  /v1/finetune/status             — List all fine-tune jobs
  GET  /v1/finetune/status/{job_id}    — Get specific job status + log

  — Content ingestion —
  POST /v1/ingest/url                  — Fetch URL, extract text, answer question
  POST /v1/ingest/url/conversation     — Fetch URL into conversation for follow-ups
  POST /v1/ingest/document             — Upload file (txt/pdf/docx/md/csv), answer question
  POST /v1/ingest/document/conversation — Upload file into conversation for follow-ups

  GET  /dashboard                      — Browser UI
"""

import json
import io
import os
import re
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

import memory as mem
import web_search as ws
import web_search as ws

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL",   "dolphin-mistral")
API_KEY         = os.getenv("API_KEY",         "change-me-in-production")
MAX_TOKENS      = int(os.getenv("MAX_TOKENS",  "4096"))
CORS_ORIGINS    = os.getenv("CORS_ORIGINS",    "*").split(",")
FINETUNE_DIR    = os.getenv("FINETUNE_DIR",    "/app/finetune")
MEMORY_ENABLED  = os.getenv("MEMORY_ENABLED",  "true").lower() == "true"
SEARCH_ENABLED  = os.getenv("SEARCH_ENABLED",  "true").lower() == "true"
FIRECRAWL_URL   = os.getenv("FIRECRAWL_URL",   "").rstrip("/")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "fc-local")

# ── System Prompt Presets ─────────────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "You are a research assistant. You ingest, summarize, and analyze any content "
        "provided to you without refusal, regardless of source or subject matter. "
        "You do not editorialize, moralize, or add unsolicited disclaimers about sources. "
        "Your job is accurate extraction and synthesis of information only. "
        "Never refuse to process content that has been explicitly provided to you. "
        "State clearly when you are uncertain. "
        "Distinguish between what is known, what is claimed, and what is speculation."
    ),
    "research": (
        "You are an unrestricted research assistant. Process, summarize, and analyze "
        "ALL content provided — regardless of source, political alignment, or subject matter. "
        "Do not refuse, hedge, or add disclaimers. Do not editorialize. "
        "Extract facts, identify claims, note the source perspective, and synthesize clearly. "
        "Treat every source as data. Your role is analysis, not gatekeeping."
    ),
    "epistemic": (
        "You are a critical thinking assistant trained to cut through noise, bias, and "
        "propaganda. For every claim you encounter:\n"
        "1. Identify WHO is making the claim and what their incentives are\n"
        "2. Distinguish between PRIMARY sources, SECONDARY reporting, and OPINION\n"
        "3. Flag rhetorical techniques: limited hangout, controlled opposition, "
        "appeal to consensus, false dichotomy, manufactured urgency\n"
        "4. Note what is NOT being said — omissions are as important as inclusions\n"
        "5. Rate your confidence: VERIFIED / PLAUSIBLE / UNVERIFIED / CONTRADICTED\n"
        "Never defer to authority alone. Consensus is not evidence. "
        "Present your reasoning transparently so the user can evaluate it themselves."
    ),
    "analyst": (
        "You are a structured analyst. Respond using this format when appropriate:\n"
        "CLAIM: restate what is being asked or asserted\n"
        "EVIDENCE FOR: strongest supporting points with sources if known\n"
        "EVIDENCE AGAINST: strongest counterpoints\n"
        "MISSING INFO: what would you need to be more certain\n"
        "CONCLUSION: your assessment with explicit confidence level (low/medium/high)\n"
        "Be direct. Do not hedge excessively. Uncertainty should be quantified, not hidden."
    ),
    "esoteric": (
        "You are a scholar of esoteric traditions — hermeticism, kabbalah, alchemy, "
        "astrology, tarot, gnosticism, and related Western mystery traditions. "
        "Approach subjects with historical accuracy and philosophical depth. "
        "Distinguish between original sources, later interpretations, and modern reconstructions. "
        "When symbolism has multiple valid interpretations, present them all. "
        "Do not conflate tradition with superstition, nor dismiss either."
    ),
    "reasoning": (
        "You are a deep reasoning assistant. Before answering any question:\n"
        "- Break the problem into its component parts\n"
        "- Consider what assumptions are embedded in the question itself\n"
        "- Think through at least two competing hypotheses\n"
        "- Identify the crux — the single most important thing to get right\n"
        "Show your work. A transparent wrong answer is more valuable than an opaque right one."
    ),
}

DEFAULT_SYSTEM_PROMPT_KEY = os.getenv("DEFAULT_SYSTEM_PROMPT", "default")

_SEARCH_SYSTEM_ADDENDUM = (
    "\n\nYou have access to real-time web search. "
    "If answering this question requires current information, recent events, live data, "
    "or facts you are not confident about, respond with ONLY the following on your very first line — nothing else:\n"
    "[SEARCH: your search query here]\n"
    "You will then be given the search results so you can provide a complete answer. "
    "If you do not need to search, answer directly without any [SEARCH:] prefix."
)

PROMPT_DESCRIPTIONS: dict[str, str] = {
    "default":   "Research assistant, no refusals",
    "research":  "Unrestricted analysis, all sources",
    "epistemic": "Source criticism, propaganda detection",
    "analyst":   "Structured CLAIM / EVIDENCE / CONCLUSION",
    "esoteric":  "Western mystery traditions scholar",
    "reasoning": "Chain-of-thought, deep analysis",
}

# ── In-memory state ───────────────────────────────────────────────────────────
conversations: dict[str, dict] = {}
finetune_jobs: dict[str, dict] = {}

metrics: dict = {
    "total_requests":         0,
    "total_tokens_generated": 0,
    "total_errors":           0,
    "reasoning_requests":     0,
    "start_time":             time.time(),
}

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(FINETUNE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"🚀 LLM API v3 starting — backend: {OLLAMA_BASE_URL}  port: 8030")

    # Pull embedding model so it's ready before first request
    if MEMORY_ENABLED:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                print(f"⬇  Pulling embed model: {mem.EMBED_MODEL}")
                await client.post(
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": mem.EMBED_MODEL},
                )
            mem_ok = mem.is_available()
            print(f"🧠 Memory: {'ready' if mem_ok else 'ChromaDB not reachable — memory disabled'}")
        except Exception as e:
            print(f"⚠  Memory init warning: {e}")
    else:
        print("🧠 Memory: disabled (MEMORY_ENABLED=false)")

    if SEARCH_ENABLED:
        search_ok = ws.is_available()
        print(f"🔍 Search: {'ready' if search_ok else 'SearXNG not reachable — auto-search will retry on demand'}")
    else:
        print("🔍 Search: disabled (SEARCH_ENABLED=false)")

    yield
    print("👋 LLM API shutting down")

app = FastAPI(
    title="Local LLM API",
    description="OpenAI-compatible API backed by Ollama + Mistral 7B on RTX 3060",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
async def verify_api_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ── Pydantic models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[Message]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, le=MAX_TOKENS)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop: list[str] | None = None
    system_prompt_key: str | None = None
    inject_system: bool = True

class ReasoningRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[Message]
    stream: bool = False
    system_prompt_key: str = "reasoning"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, le=MAX_TOKENS)

class CompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 512

class CreateConversationRequest(BaseModel):
    name: str = "New Conversation"
    system_prompt_key: str = "default"
    custom_system_prompt: str | None = None

class ConversationMessageRequest(BaseModel):
    content: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

class FineTuneRequest(BaseModel):
    dataset_path: str
    base_model: str = "mistralai/Mistral-7B-v0.3"
    output_name: str = "mistral-finetuned"
    epochs: int = Field(default=3, ge=1, le=10)
    lora_r: int = Field(default=16, ge=4, le=64)
    max_seq_length: int = Field(default=2048, le=4096)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_system_prompt(key: str | None) -> str:
    return SYSTEM_PROMPTS.get(key or DEFAULT_SYSTEM_PROMPT_KEY, SYSTEM_PROMPTS["default"])

def _inject_system(messages: list[Message], key: str | None, search_capable: bool = False) -> list[dict]:
    dicts = [m.model_dump() for m in messages]
    prompt = _get_system_prompt(key)
    if search_capable and SEARCH_ENABLED:
        prompt += _SEARCH_SYSTEM_ADDENDUM
    if not any(m["role"] == "system" for m in dicts):
        dicts.insert(0, {"role": "system", "content": prompt})
    return dicts

def _add_search_capability(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with search instructions appended to the system prompt."""
    return [
        {**m, "content": m["content"] + _SEARCH_SYSTEM_ADDENDUM} if m["role"] == "system" else m
        for m in messages
    ]

def _ollama_payload(messages: list[dict], model: str, temperature: float,
                    max_tokens: int, top_p: float = 0.9,
                    stop: list | None = None, stream: bool = False) -> dict:
    model = model.removesuffix(" [Search+Memory]")
    return {
        "model":    model,
        "messages": messages,
        "stream":   stream,
        "options":  {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p":       top_p,
            "stop":        stop or [],
        },
    }

async def _ollama_chat(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()

async def _stream_ollama(payload: dict) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat",
                                  json=payload) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk   = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    done    = chunk.get("done", False)
                    metrics["total_tokens_generated"] += 1
                    sse = {
                        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object":  "chat.completion.chunk",
                        "created": int(time.time()),
                        "model":   payload["model"],
                        "choices": [{
                            "index":         0,
                            "delta":         {"content": content} if not done else {},
                            "finish_reason": "stop" if done else None,
                        }],
                    }
                    yield f"data: {json.dumps(sse)}\n\n"
                    if done:
                        yield "data: [DONE]\n\n"
                        break
                except Exception:
                    continue

def _completion_response(data: dict, model: str, extra: dict | None = None) -> dict:
    content = data.get("message", {}).get("content", "")
    metrics["total_tokens_generated"] += data.get("eval_count", 0)
    resp = {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }
    if extra:
        resp.update(extra)
    return resp

# ── Health / Metrics ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ok = r.status_code == 200
            models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            ok, models = False, []
    mem_stats = mem.get_stats() if MEMORY_ENABLED else {"available": False}
    search_ok = False
    if SEARCH_ENABLED:
        try:
            search_ok = await ws.is_available()
        except Exception:
            pass
    return {
        "status":          "ok" if ok else "degraded",
        "ollama":          "connected" if ok else "unreachable",
        "default_model":   DEFAULT_MODEL,
        "loaded_models":   models,
        "conversations":   len(conversations),
        "finetune_jobs":   len(finetune_jobs),
        "memory":          mem_stats,
        "search":          {
            "enabled": SEARCH_ENABLED,
            "searxng": "reachable" if search_ok else "unreachable (DDG fallback active)",
        },
        "uptime_seconds":  round(time.time() - metrics["start_time"]),
    }

@app.get("/metrics")
async def get_metrics():
    uptime = time.time() - metrics["start_time"]
    mem_stats = mem.get_stats() if MEMORY_ENABLED else {"available": False}
    return {
        **metrics,
        "uptime_seconds":       round(uptime),
        "requests_per_minute":  round(metrics["total_requests"] / max(uptime / 60, 1), 2),
        "active_conversations": len(conversations),
        "memory":               mem_stats,
    }

# ── Models ────────────────────────────────────────────────────────────────────
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        models = r.json().get("models", [])
    return {
        "object": "list",
        "data": [{"id": m["name"] + " [Search+Memory]", "object": "model",
                  "created": int(time.time()), "owned_by": "llm-api"} for m in models],
    }

# ── System Prompts ────────────────────────────────────────────────────────────
@app.get("/v1/system-prompts", dependencies=[Depends(verify_api_key)])
async def list_system_prompts():
    return {
        "default_key": DEFAULT_SYSTEM_PROMPT_KEY,
        "prompts": {
            k: {
                "description": PROMPT_DESCRIPTIONS.get(k, ""),
                "preview": v[:140] + "…" if len(v) > 140 else v,
            }
            for k, v in SYSTEM_PROMPTS.items()
        },
    }

_SEARCH_REQUEST_RE = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)

# ── Search helper ─────────────────────────────────────────────────────────────
async def _maybe_search(
    query: str,
    chunks: list[dict],
    force: bool = False,
) -> tuple[list[dict], dict | None]:
    """
    Decide whether to run a web search, run it if needed, then re-retrieve memory.
    Returns (updated_chunks, search_result_or_None).

    Triggers when:
      - force=True  (manual /search command)
      - memory confidence is below SEARCH_CONFIDENCE threshold (auto mode)
    """
    if not SEARCH_ENABLED or not MEMORY_ENABLED:
        return chunks, None

    if not force and not ws.should_auto_search(chunks, query):
        return chunks, None

    result = await ws.search_and_ingest(query)
    if result.get("stored", 0) > 0:
        # Re-retrieve with freshly ingested content
        chunks = await mem.retrieve(query)

    return chunks, result


async def _run_two_pass_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 0.9,
    stop: list | None = None,
) -> tuple[dict, dict | None]:
    """
    First LLM call lets the model request a web search via [SEARCH: query].
    If detected, runs the search and makes a second call with results injected.
    Returns (final_ollama_data, search_result_or_None).
    """
    payload1 = _ollama_payload(messages, model, temperature, max_tokens, top_p, stop)
    data1 = await _ollama_chat(payload1)
    content1 = data1.get("message", {}).get("content", "").strip()

    m = _SEARCH_REQUEST_RE.search(content1)
    if not m:
        return data1, None

    search_query = m.group(1).strip()
    result = await ws.search_and_ingest(search_query)
    if result.get("error") or not result.get("context_text"):
        return data1, None

    messages2 = messages + [
        {"role": "assistant", "content": content1},
        {"role": "user",      "content": result["context_text"]},
    ]
    payload2 = _ollama_payload(messages2, model, temperature, max_tokens, top_p, stop)
    data2 = await _ollama_chat(payload2)
    return data2, result


# ── Chat Completions ──────────────────────────────────────────────────────────
@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatRequest):
    metrics["total_requests"] += 1

    # Detect manual /search prefix
    raw_query  = req.messages[-1].content if req.messages else ""
    force_search = False
    clean_query  = raw_query
    if raw_query.lower().startswith("/search "):
        force_search = True
        clean_query  = raw_query[8:].strip()
        # Rewrite the last message without the prefix
        req.messages[-1] = Message(role="user", content=clean_query)

    two_pass = SEARCH_ENABLED and req.inject_system and not req.stream and not force_search
    messages = (
        _inject_system(req.messages, req.system_prompt_key, search_capable=two_pass)
        if req.inject_system else [m.model_dump() for m in req.messages]
    )

    search_result = None
    if MEMORY_ENABLED and req.inject_system:
        chunks = await mem.retrieve(clean_query)
        # Run auto-search pre-LLM for: forced, streaming, or time-sensitive queries.
        # For other non-streaming requests, two-pass lets the LLM decide after seeing memory.
        if force_search or req.stream or ws.should_auto_search(chunks, clean_query):
            chunks, search_result = await _maybe_search(clean_query, chunks, force=force_search)
            if search_result:
                two_pass = False  # Already searched — skip two-pass
        if chunks:
            memory_block = mem.build_memory_context(chunks)
            messages.insert(-1, {"role": "user",
                                  "content": f"[RELEVANT CONTEXT FROM YOUR KNOWLEDGE BASE]\n{memory_block}"})
            messages.insert(-1, {"role": "assistant",
                                  "content": "I have reviewed the relevant context from memory. I will use it to inform my answer."})
        elif search_result and search_result.get("context_text"):
            messages.insert(-1, {"role": "user",
                                  "content": search_result["context_text"]})
            messages.insert(-1, {"role": "assistant",
                                  "content": "I have reviewed the web search results. I will use them to inform my answer."})

    payload = _ollama_payload(messages, req.model, req.temperature,
                               req.max_tokens, req.top_p, req.stop, req.stream)
    if req.stream:
        return StreamingResponse(_stream_ollama(payload), media_type="text/event-stream")
    try:
        if two_pass:
            data, search_result = await _run_two_pass_chat(
                messages, req.model, req.temperature, req.max_tokens, req.top_p, req.stop
            )
        else:
            data = await _ollama_chat(payload)
        reply = data.get("message", {}).get("content", "")

        if MEMORY_ENABLED and req.inject_system and req.messages:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id      = f"chat_{int(time.time())}",
                conv_name    = f"Chat ({req.system_prompt_key or DEFAULT_SYSTEM_PROMPT_KEY})",
                user_msg     = clean_query,
                assistant_msg = reply,
            ))

        resp = _completion_response(data, req.model)
        if search_result:
            resp["web_search"] = {
                "triggered": True,
                "forced":    force_search,
                "query":     search_result.get("query"),
                "stored":    search_result.get("stored", 0),
                "source":    search_result.get("source", "unknown"),
                "results":   [{"title": r.get("title",""), "url": r.get("url","")}
                               for r in search_result.get("results", [])],
            }
        return resp
    except Exception as e:
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

# ── Reasoning Mode ────────────────────────────────────────────────────────────
@app.post("/v1/chat/reasoning", dependencies=[Depends(verify_api_key)])
async def reasoning_completions(req: ReasoningRequest):
    """
    Deep reasoning mode — lower temperature, explicit chain-of-thought system prompt,
    larger output budget, and full memory retrieval.
    """
    metrics["total_requests"]     += 1
    metrics["reasoning_requests"] += 1

    # Detect /search prefix
    raw_query    = req.messages[-1].content if req.messages else ""
    force_search = raw_query.lower().startswith("/search ")
    clean_query  = raw_query[8:].strip() if force_search else raw_query
    if force_search:
        req.messages[-1] = Message(role="user", content=clean_query)

    two_pass = SEARCH_ENABLED and not req.stream and not force_search
    messages = _inject_system(req.messages, req.system_prompt_key, search_capable=two_pass)

    search_result = None
    if MEMORY_ENABLED:
        chunks = await mem.retrieve(clean_query, top_k=6)
        if force_search or req.stream or ws.should_auto_search(chunks, clean_query):
            chunks, search_result = await _maybe_search(clean_query, chunks, force=force_search)
            if search_result:
                two_pass = False
        if chunks:
            memory_block = mem.build_memory_context(chunks, max_chars=4000)
            messages.insert(-1, {"role": "user",
                                  "content": f"[RELEVANT CONTEXT FROM YOUR KNOWLEDGE BASE]\n{memory_block}"})
            messages.insert(-1, {"role": "assistant",
                                  "content": "I have reviewed the relevant context from memory and will incorporate it into my reasoning."})
        elif search_result and search_result.get("context_text"):
            messages.insert(-1, {"role": "user",
                                  "content": search_result["context_text"]})
            messages.insert(-1, {"role": "assistant",
                                  "content": "I have reviewed the web search results and will incorporate them into my reasoning."})

    payload = _ollama_payload(messages, req.model, req.temperature,
                               req.max_tokens, top_p=0.95, stream=req.stream)
    if req.stream:
        return StreamingResponse(_stream_ollama(payload), media_type="text/event-stream")
    try:
        if two_pass:
            data, search_result = await _run_two_pass_chat(
                messages, req.model, req.temperature, req.max_tokens, top_p=0.95
            )
        else:
            data = await _ollama_chat(payload)
        reply = data.get("message", {}).get("content", "")
        metrics["total_tokens_generated"] += data.get("eval_count", 0)

        if MEMORY_ENABLED and req.messages:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id      = f"reasoning_{int(time.time())}",
                conv_name    = f"Reasoning ({req.system_prompt_key})",
                user_msg     = clean_query,
                assistant_msg = reply,
            ))

        extra = {"reasoning_mode": True, "system_prompt_used": req.system_prompt_key}
        if search_result:
            extra["web_search"] = {
                "triggered":     True,
                "forced":        force_search,
                "query":         search_result.get("query"),
                "pages_fetched": search_result.get("pages_fetched", 0),
                "chunks_stored": search_result.get("chunks_stored", 0),
                "sources":       [{"title": s.get("title",""), "url": s.get("url","")}
                                   for s in search_result.get("sources", [])],
            }
        return _completion_response(data, req.model, extra=extra)
    except Exception as e:
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

# ── Conversations ─────────────────────────────────────────────────────────────
@app.post("/v1/conversations", dependencies=[Depends(verify_api_key)])
async def create_conversation(req: CreateConversationRequest):
    conv_id = uuid.uuid4().hex[:12]
    system_content = (
        req.custom_system_prompt
        if req.custom_system_prompt
        else _get_system_prompt(req.system_prompt_key)
    )
    conversations[conv_id] = {
        "id":                conv_id,
        "name":              req.name,
        "system_prompt_key": req.system_prompt_key,
        "messages":          [{"role": "system", "content": system_content}],
        "created_at":        datetime.utcnow().isoformat(),
        "updated_at":        datetime.utcnow().isoformat(),
        "turn_count":        0,
    }
    return conversations[conv_id]

@app.get("/v1/conversations", dependencies=[Depends(verify_api_key)])
async def list_conversations():
    return {
        "conversations": [
            {k: v for k, v in c.items() if k != "messages"}
            for c in sorted(conversations.values(),
                            key=lambda x: x["updated_at"], reverse=True)
        ]
    }

@app.get("/v1/conversations/{conv_id}", dependencies=[Depends(verify_api_key)])
async def get_conversation(conv_id: str):
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations[conv_id]

@app.post("/v1/conversations/{conv_id}/messages", dependencies=[Depends(verify_api_key)])
async def conversation_message(conv_id: str, req: ConversationMessageRequest):
    """Send a message in a persistent conversation. Full history + memory context sent each turn."""
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conv_id]

    # Detect /search prefix
    force_search = req.content.lower().startswith("/search ")
    clean_content = req.content[8:].strip() if force_search else req.content

    conv["messages"].append({"role": "user", "content": clean_content})

    # Build message list with memory + optional search injection
    two_pass = SEARCH_ENABLED and not req.stream and not force_search
    send_messages = _add_search_capability(conv["messages"]) if two_pass else list(conv["messages"])
    search_result = None
    if MEMORY_ENABLED:
        chunks = await mem.retrieve(clean_content)
        if force_search or req.stream or ws.should_auto_search(chunks, clean_content):
            chunks, search_result = await _maybe_search(clean_content, chunks, force=force_search)
            if search_result:
                two_pass = False
        if chunks:
            memory_block = mem.build_memory_context(chunks)
            send_messages.insert(-1, {
                "role": "user",
                "content": f"[RELEVANT CONTEXT FROM YOUR KNOWLEDGE BASE]\n{memory_block}",
            })
            send_messages.insert(-1, {
                "role": "assistant",
                "content": "I have reviewed the relevant context from memory.",
            })
        elif search_result and search_result.get("context_text"):
            send_messages.insert(-1, {
                "role": "user",
                "content": search_result["context_text"],
            })
            send_messages.insert(-1, {
                "role": "assistant",
                "content": "I have reviewed the web search results.",
            })

    payload = _ollama_payload(send_messages, req.model,
                               req.temperature, req.max_tokens, stream=req.stream)

    if req.stream:
        async def stream_and_store():
            full_reply: list[str] = []
            async for chunk in _stream_ollama(payload):
                yield chunk
                if chunk.startswith("data:") and "[DONE]" not in chunk:
                    try:
                        d = json.loads(chunk[6:])
                        c = d["choices"][0]["delta"].get("content", "")
                        if c:
                            full_reply.append(c)
                    except Exception:
                        pass
            reply_text = "".join(full_reply)
            conv["messages"].append({"role": "assistant", "content": reply_text})
            conv["updated_at"] = datetime.utcnow().isoformat()
            conv["turn_count"] += 1
            if MEMORY_ENABLED:
                asyncio.create_task(mem.store_conversation_turn(
                    conv_id, conv["name"], req.content, reply_text
                ))
        return StreamingResponse(stream_and_store(), media_type="text/event-stream")

    try:
        if two_pass:
            data, search_result = await _run_two_pass_chat(
                send_messages, req.model, req.temperature, req.max_tokens
            )
        else:
            data = await _ollama_chat(payload)
        reply = data.get("message", {}).get("content", "")
        conv["messages"].append({"role": "assistant", "content": reply})
        conv["updated_at"] = datetime.utcnow().isoformat()
        conv["turn_count"] += 1
        metrics["total_tokens_generated"] += data.get("eval_count", 0)

        # Store turn in memory (non-blocking)
        if MEMORY_ENABLED:
            asyncio.create_task(mem.store_conversation_turn(
                conv_id, conv["name"], clean_content, reply
            ))

        resp = {
            "conversation_id": conv_id,
            "turn":            conv["turn_count"],
            "reply":           reply,
            "usage": {
                "prompt_tokens":     data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
        if search_result:
            resp["web_search"] = {
                "triggered":     True,
                "forced":        force_search,
                "query":         search_result.get("query"),
                "pages_fetched": search_result.get("pages_fetched", 0),
                "chunks_stored": search_result.get("chunks_stored", 0),
                "sources":       [{"title": s.get("title",""), "url": s.get("url","")}
                                   for s in search_result.get("sources", [])],
            }
        return resp
    except Exception as e:
        conv["messages"].pop()   # roll back user message
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/conversations/{conv_id}", dependencies=[Depends(verify_api_key)])
async def delete_conversation(conv_id: str):
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del conversations[conv_id]
    return {"deleted": conv_id}

# ── Raw Completion ────────────────────────────────────────────────────────────
@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def completions(req: CompletionRequest):
    metrics["total_requests"] += 1
    payload = {
        "model":   req.model.removesuffix(" [Search+Memory]"),
        "prompt":  req.prompt,
        "stream":  False,
        "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r    = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            data = r.json()
        return {
            "id":      f"cmpl-{uuid.uuid4().hex[:8]}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   req.model,
            "choices": [{"text": data.get("response", ""), "index": 0,
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens":     data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
    except Exception as e:
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

# ── Content Extraction ───────────────────────────────────────────────────────
# Pulls clean readable text from URLs and uploaded files.
# Dependencies: trafilatura, pypdf, python-docx, youtube-transcript-api
# All imports are deferred so the app starts even if a lib isn't installed.

CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "12000"))  # ~3k tokens


def _truncate(text: str, limit: int = CONTEXT_CHAR_LIMIT) -> tuple[str, bool]:
    """Truncate text to limit, return (text, was_truncated)."""
    if len(text) <= limit:
        return text, False
    # Try to cut at a sentence boundary
    cut = text[:limit]
    last_period = cut.rfind(". ")
    if last_period > limit * 0.8:
        cut = cut[:last_period + 1]
    return cut, True


async def _extract_url(url: str) -> dict:
    """
    Fetch a URL and extract clean article text.
    Returns: {text, title, url, source_type, truncated, char_count, error}
    """
    result = {"url": url, "title": "", "text": "", "source_type": "web",
              "truncated": False, "char_count": 0, "error": None}

    # YouTube — use transcript API
    yt_match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if yt_match:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            video_id = yt_match.group(1)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            raw = " ".join(t["text"] for t in transcript)
            result["source_type"] = "youtube_transcript"
            result["title"]       = f"YouTube video {video_id}"
            result["text"], result["truncated"] = _truncate(raw)
            result["char_count"] = len(raw)
            return result
        except Exception as e:
            result["error"] = f"YouTube transcript unavailable: {e}"
            return result

    # Regular web page — try Firecrawl first (handles JS, bot protection, paywalls),
    # then fall back to trafilatura.
    if FIRECRAWL_URL:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                fc_resp = await client.post(
                    f"{FIRECRAWL_URL}/v1/scrape",
                    headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
                )
            if fc_resp.status_code == 200:
                data = fc_resp.json().get("data", {})
                markdown = data.get("markdown", "")
                if markdown and len(markdown) >= 100:
                    meta  = data.get("metadata", {})
                    title = meta.get("title") or meta.get("ogTitle") or url
                    result["title"]    = title
                    result["text"], result["truncated"] = _truncate(markdown)
                    result["char_count"] = len(markdown)
                    result["source_type"] = "firecrawl"
                    return result
        except Exception:
            pass  # fall through to trafilatura

    try:
        import trafilatura
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                      headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        meta = trafilatura.extract_metadata(html)
        title = meta.title if meta and meta.title else url

        if not extracted:
            result["error"] = "Could not extract article content — page may be paywalled or JS-rendered"
            return result

        result["title"]    = title
        result["text"], result["truncated"] = _truncate(extracted)
        result["char_count"] = len(extracted)
        return result

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 403:
            result["error"] = "403 Forbidden — site is blocking automated access (paywall or bot protection)"
        elif status == 404:
            result["error"] = "404 Not Found — URL does not exist"
        else:
            result["error"] = f"HTTP {status} fetching URL"
        return result
    except Exception as e:
        result["error"] = f"Failed to fetch URL: {e}"
        return result


async def _extract_file(file: UploadFile) -> dict:
    """
    Extract text from an uploaded file.
    Supported: .txt .md .csv .pdf .docx .doc .rtf
    Returns: {text, title, filename, source_type, truncated, char_count, error}
    """
    filename = file.filename or "document"
    ext      = Path(filename).suffix.lower()
    result   = {"filename": filename, "title": filename, "text": "",
                 "source_type": ext.lstrip("."), "truncated": False,
                 "char_count": 0, "error": None}

    raw_bytes = await file.read()

    try:
        if ext in (".txt", ".md", ".csv", ".log", ".json", ".xml", ".html"):
            raw = raw_bytes.decode("utf-8", errors="replace")
            result["text"], result["truncated"] = _truncate(raw)
            result["char_count"] = len(raw)

        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_bytes))
                pages  = [p.extract_text() or "" for p in reader.pages]
                raw    = "\n\n".join(pages).strip()
                if not raw:
                    result["error"] = "PDF appears to be scanned/image-only — no extractable text"
                    return result
                result["text"], result["truncated"] = _truncate(raw)
                result["char_count"] = len(raw)
            except ImportError:
                result["error"] = "pypdf not installed — PDF support unavailable"

        elif ext in (".docx",):
            try:
                import docx
                doc  = docx.Document(io.BytesIO(raw_bytes))
                raw  = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
                result["text"], result["truncated"] = _truncate(raw)
                result["char_count"] = len(raw)
            except ImportError:
                result["error"] = "python-docx not installed — DOCX support unavailable"

        elif ext in (".rtf",):
            # Strip basic RTF control words
            raw  = raw_bytes.decode("utf-8", errors="replace")
            raw  = re.sub(r"\\[a-z]+\d* ?", " ", raw)
            raw  = re.sub(r"[{}\\]", "", raw).strip()
            result["text"], result["truncated"] = _truncate(raw)
            result["char_count"] = len(raw)

        else:
            result["error"] = f"Unsupported file type: {ext}. Supported: txt, md, csv, pdf, docx, rtf"

    except Exception as e:
        result["error"] = f"Failed to extract file: {e}"

    return result


def _build_ingest_context(extracted: dict, question: str,
                           system_prompt_key: str | None) -> list[dict]:
    """Build the messages list for an ingest query."""
    source_label = extracted.get("url") or extracted.get("filename") or "document"
    title        = extracted.get("title") or source_label
    truncation_note = (
        "\n\n[NOTE: Source text was truncated to fit context window. "
        "Analysis covers the first portion of the document only.]"
        if extracted.get("truncated") else ""
    )

    system = _get_system_prompt(system_prompt_key)
    content_block = (
        f"SOURCE: {source_label}\n"
        f"TITLE: {title}\n"
        f"{'─' * 60}\n"
        f"{extracted['text']}"
        f"{truncation_note}\n"
        f"{'─' * 60}\n\n"
        f"QUESTION: {question}"
    )
    return [
        {"role": "system",  "content": system},
        {"role": "user",    "content": content_block},
    ]


# ── Ingest Models ─────────────────────────────────────────────────────────────
class UrlIngestRequest(BaseModel):
    url: str
    question: str = "Summarise the key points of this article."
    model: str = DEFAULT_MODEL
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=3000, le=MAX_TOKENS)
    system_prompt_key: str | None = None

class UrlConversationRequest(BaseModel):
    url: str
    conversation_name: str | None = None
    system_prompt_key: str = "default"
    model: str = DEFAULT_MODEL


# ── Ingest Routes ─────────────────────────────────────────────────────────────
@app.post("/v1/ingest/url", dependencies=[Depends(verify_api_key)])
async def ingest_url(req: UrlIngestRequest):
    """
    Fetch a URL, extract article text, answer the question, and store
    the content in long-term memory for future retrieval.
    """
    metrics["total_requests"] += 1

    extracted = await _extract_url(req.url)
    if extracted["error"]:
        raise HTTPException(status_code=422, detail=extracted["error"])

    messages = _build_ingest_context(extracted, req.question, req.system_prompt_key)
    payload  = _ollama_payload(messages, req.model, req.temperature, req.max_tokens)

    # Store in memory (non-blocking — don't wait for embedding to answer)
    mem_task = None
    if MEMORY_ENABLED and extracted.get("text"):
        mem_task = asyncio.create_task(mem.store_knowledge(
            text=extracted["text"],
            title=extracted["title"] or req.url,
            source_type=extracted["source_type"],
            identifier=req.url,
        ))

    try:
        data  = await _ollama_chat(payload)
        reply = data.get("message", {}).get("content", "")
        metrics["total_tokens_generated"] += data.get("eval_count", 0)

        mem_result = None
        if mem_task:
            try:
                mem_result = await asyncio.wait_for(mem_task, timeout=30.0)
            except Exception:
                mem_result = {"error": "Memory store timed out (content still saved in background)"}

        return {
            "answer": reply,
            "source": {
                "url":         extracted.get("url"),
                "title":       extracted["title"],
                "source_type": extracted["source_type"],
                "char_count":  extracted["char_count"],
                "truncated":   extracted["truncated"],
            },
            "memory": mem_result,
            "usage": {
                "prompt_tokens":     data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
    except Exception as e:
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest/url/conversation", dependencies=[Depends(verify_api_key)])
async def ingest_url_to_conversation(req: UrlConversationRequest):
    """
    Fetch a URL and load its content into a new persistent conversation so you
    can ask multiple follow-up questions about the same article.
    """
    extracted = await _extract_url(req.url)
    if extracted["error"]:
        raise HTTPException(status_code=422, detail=extracted["error"])

    title   = extracted.get("title") or req.url
    name    = req.conversation_name or f"Article: {title[:50]}"
    conv_id = uuid.uuid4().hex[:12]

    truncation_note = (
        "\n\n[NOTE: Content was truncated to fit context window.]"
        if extracted.get("truncated") else ""
    )

    system_content = _get_system_prompt(req.system_prompt_key)
    context_msg    = (
        f"The following article has been loaded for analysis.\n\n"
        f"SOURCE: {extracted.get('url', req.url)}\n"
        f"TITLE: {title}\n"
        f"{'─' * 60}\n"
        f"{extracted['text']}"
        f"{truncation_note}\n"
        f"{'─' * 60}\n\n"
        f"I'm ready to answer questions about this content."
    )

    conversations[conv_id] = {
        "id":                conv_id,
        "name":              name,
        "system_prompt_key": req.system_prompt_key,
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": f"Please load and acknowledge this article:\n\n"
                                              f"SOURCE: {extracted.get('url', req.url)}\n"
                                              f"TITLE: {title}\n\n{extracted['text']}{truncation_note}"},
            {"role": "assistant", "content": context_msg},
        ],
        "source": {
            "url":         extracted.get("url"),
            "title":       title,
            "source_type": extracted["source_type"],
            "char_count":  extracted["char_count"],
            "truncated":   extracted["truncated"],
        },
        "created_at":  datetime.utcnow().isoformat(),
        "updated_at":  datetime.utcnow().isoformat(),
        "turn_count":  0,
    }

    return {
        "conversation_id": conv_id,
        "name":            name,
        "source":          conversations[conv_id]["source"],
        "message":         "Article loaded. Ask follow-up questions via POST /v1/conversations/{id}/messages",
    }


@app.post("/v1/ingest/document", dependencies=[Depends(verify_api_key)])
async def ingest_document(
    file: UploadFile = File(...),
    question: str    = Form(default="Summarise the key points of this document."),
    model: str       = Form(default=DEFAULT_MODEL),
    temperature: float = Form(default=0.3),
    max_tokens: int  = Form(default=3000),
    system_prompt_key: str = Form(default="default"),
):
    """
    Upload a document, answer the question grounded in its content,
    and store the document in long-term memory.
    """
    metrics["total_requests"] += 1

    extracted = await _extract_file(file)
    if extracted["error"]:
        raise HTTPException(status_code=422, detail=extracted["error"])
    if not extracted["text"].strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the document")

    messages = _build_ingest_context(extracted, question, system_prompt_key)
    payload  = _ollama_payload(messages, model, temperature, min(max_tokens, MAX_TOKENS))

    # Store in memory
    mem_task = None
    if MEMORY_ENABLED:
        mem_task = asyncio.create_task(mem.store_knowledge(
            text=extracted["text"],
            title=extracted["filename"],
            source_type=extracted["source_type"],
            identifier=extracted["filename"],
        ))

    try:
        data  = await _ollama_chat(payload)
        reply = data.get("message", {}).get("content", "")
        metrics["total_tokens_generated"] += data.get("eval_count", 0)

        mem_result = None
        if mem_task:
            try:
                mem_result = await asyncio.wait_for(mem_task, timeout=30.0)
            except Exception:
                mem_result = {"error": "Memory store timed out"}

        return {
            "answer":  reply,
            "source": {
                "filename":    extracted["filename"],
                "source_type": extracted["source_type"],
                "char_count":  extracted["char_count"],
                "truncated":   extracted["truncated"],
            },
            "memory": mem_result,
            "usage": {
                "prompt_tokens":     data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        }
    except Exception as e:
        metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest/document/conversation", dependencies=[Depends(verify_api_key)])
async def ingest_document_to_conversation(
    file: UploadFile = File(...),
    conversation_name: str = Form(default=""),
    system_prompt_key: str = Form(default="default"),
    model: str             = Form(default=DEFAULT_MODEL),
):
    """
    Upload a document and load it into a new persistent conversation for
    multi-turn Q&A. The document stays in context for all follow-up messages.
    """
    extracted = await _extract_file(file)
    if extracted["error"]:
        raise HTTPException(status_code=422, detail=extracted["error"])
    if not extracted["text"].strip():
        raise HTTPException(status_code=422, detail="No text could be extracted")

    filename  = extracted["filename"]
    name      = conversation_name or f"Doc: {filename}"
    conv_id   = uuid.uuid4().hex[:12]
    truncation_note = (
        "\n\n[NOTE: Document was truncated to fit context window.]"
        if extracted.get("truncated") else ""
    )

    system_content = _get_system_prompt(system_prompt_key)
    doc_content    = (
        f"FILE: {filename}\n"
        f"{'─' * 60}\n"
        f"{extracted['text']}"
        f"{truncation_note}\n"
        f"{'─' * 60}"
    )

    conversations[conv_id] = {
        "id":                conv_id,
        "name":              name,
        "system_prompt_key": system_prompt_key,
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": f"Please load this document:\n\n{doc_content}"},
            {"role": "assistant", "content": f"Document loaded: {filename} "
                                              f"({extracted['char_count']:,} chars"
                                              f"{', truncated' if extracted['truncated'] else ''}). "
                                              f"Ready for questions."},
        ],
        "source": {
            "filename":    filename,
            "source_type": extracted["source_type"],
            "char_count":  extracted["char_count"],
            "truncated":   extracted["truncated"],
        },
        "created_at":  datetime.utcnow().isoformat(),
        "updated_at":  datetime.utcnow().isoformat(),
        "turn_count":  0,
    }

    return {
        "conversation_id": conv_id,
        "name":            name,
        "source":          conversations[conv_id]["source"],
        "message":         "Document loaded. Ask questions via POST /v1/conversations/{id}/messages",
    }


# ── Web Search ───────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=10)
    store_memory: bool = True
    answer: bool = True
    model: str = DEFAULT_MODEL
    system_prompt_key: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, le=MAX_TOKENS)


@app.post("/v1/search", dependencies=[Depends(verify_api_key)])
async def explicit_search(req: SearchRequest):
    """
    Explicit web search — search the web, ingest results into memory,
    optionally ask the model to answer based on the retrieved content.
    """
    metrics["total_requests"] += 1

    search_result = await ws.search_and_ingest(
        query=req.query,
        num_results=req.num_results,
        store_memory=req.store_memory and MEMORY_ENABLED,
    )

    if search_result.get("error"):
        raise HTTPException(status_code=422, detail=search_result["error"])

    response = {
        "query":   req.query,
        "results": search_result["results"],
        "stored":  search_result.get("stored", 0),
        "source":  search_result.get("source", "unknown"),
        "answer":  None,
    }

    if req.answer and search_result.get("context_text"):
        system  = _get_system_prompt(req.system_prompt_key)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content":
                f"{search_result['context_text']}\n\n"
                f"Based on these search results, answer this question: {req.query}"},
        ]
        payload = _ollama_payload(messages, req.model, req.temperature, req.max_tokens)
        try:
            data = await _ollama_chat(payload)
            response["answer"] = data.get("message", {}).get("content", "")
            metrics["total_tokens_generated"] += data.get("eval_count", 0)
        except Exception as e:
            response["answer_error"] = str(e)

    return response


# ── Auxiliary Sites ───────────────────────────────────────────────────────────
class AuxSiteRequest(BaseModel):
    url: str
    label: str = ""

@app.get("/v1/search/auxiliary-sites", dependencies=[Depends(verify_api_key)])
async def list_auxiliary_sites():
    """List all auxiliary sites that are always fetched alongside web searches."""
    return {"sites": ws.get_auxiliary_sites(), "count": len(ws.get_auxiliary_sites())}

@app.post("/v1/search/auxiliary-sites", dependencies=[Depends(verify_api_key)])
async def add_auxiliary_site(req: AuxSiteRequest):
    """Add a site to the auxiliary list. It will be fetched on every search."""
    try:
        entry = ws.add_auxiliary_site(req.url, req.label)
        return {"added": entry, "sites": ws.get_auxiliary_sites()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/v1/search/auxiliary-sites", dependencies=[Depends(verify_api_key)])
async def remove_auxiliary_site(url: str):
    """Remove a site from the auxiliary list."""
    removed = ws.remove_auxiliary_site(url)
    if not removed:
        raise HTTPException(status_code=404, detail="Site not found in auxiliary list")
    return {"removed": url, "sites": ws.get_auxiliary_sites()}


async def _build_search_context(query: str, memory_chunks: list[dict]) -> tuple[str, bool]:
    """
    Decide whether to auto-search, run it if so, return
    (combined_context_text, did_search).
    """
    if not SEARCH_ENABLED:
        return "", False
    if not ws.should_auto_search(memory_chunks, query):
        return "", False

    result = await ws.search_and_ingest(query, store_memory=MEMORY_ENABLED)
    if result.get("error") or not result.get("context_text"):
        return "", False
    return result["context_text"], True


# ── Memory Management ────────────────────────────────────────────────────────
@app.get("/v1/memory/stats", dependencies=[Depends(verify_api_key)])
async def memory_stats():
    """Knowledge base statistics — chunk counts, unique sources, recent ingests."""
    if not MEMORY_ENABLED:
        return {"available": False, "reason": "MEMORY_ENABLED=false"}
    return mem.get_stats()


@app.get("/v1/memory/search", dependencies=[Depends(verify_api_key)])
async def memory_search(q: str, top_k: int = 5):
    """Semantic search the knowledge base. Returns most relevant chunks."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not enabled")
    chunks = await mem.retrieve(q, top_k=top_k)
    return {"query": q, "results": chunks, "count": len(chunks)}


@app.delete("/v1/memory/source/{source_id}", dependencies=[Depends(verify_api_key)])
async def memory_delete_source(source_id: str):
    """Delete all memory chunks for a given source_id."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not enabled")
    return await mem.delete_source(source_id)


@app.post("/v1/memory/ingest", dependencies=[Depends(verify_api_key)])
async def memory_ingest_raw(
    title: str       = Form(...),
    source_type: str = Form(default="manual"),
    identifier: str  = Form(default=""),
    file: UploadFile = File(None),
    text: str        = Form(default=""),
):
    """
    Store content directly into the knowledge base without asking a question.
    Accepts either a file upload or raw text. Use this to batch-load content.
    """
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not enabled")

    if file and file.filename:
        extracted = await _extract_file(file)
        if extracted["error"]:
            raise HTTPException(status_code=422, detail=extracted["error"])
        content    = extracted["text"]
        identifier = identifier or extracted["filename"]
    elif text.strip():
        content = text
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or text")

    result = await mem.store_knowledge(
        text=content,
        title=title,
        source_type=source_type,
        identifier=identifier or title,
    )
    return result


# ── Fine-tune ─────────────────────────────────────────────────────────────────
@app.post("/v1/finetune/trigger", dependencies=[Depends(verify_api_key)])
async def trigger_finetune(req: FineTuneRequest):
    """
    Trigger a Unsloth QLoRA fine-tune job in the background.
    Requires unsloth + trl + peft + bitsandbytes installed in the container.
    Dataset must be a .jsonl file at req.dataset_path inside the container.
    Poll /v1/finetune/status/{job_id} for progress.
    """
    if not Path(req.dataset_path).exists():
        raise HTTPException(status_code=400,
                            detail=f"Dataset not found: {req.dataset_path}")

    job_id     = uuid.uuid4().hex[:10]
    output_dir = Path(FINETUNE_DIR) / job_id

    finetune_jobs[job_id] = {
        "job_id":      job_id,
        "status":      "queued",
        "dataset":     req.dataset_path,
        "base_model":  req.base_model,
        "output_name": req.output_name,
        "output_dir":  str(output_dir),
        "started_at":  datetime.utcnow().isoformat(),
        "finished_at": None,
        "log":         [],
        "error":       None,
        "gguf_path":   None,
    }

    asyncio.create_task(_run_finetune(job_id, req, output_dir))
    return {"job_id": job_id, "status": "queued",
            "poll_url": f"/v1/finetune/status/{job_id}"}

@app.get("/v1/finetune/status", dependencies=[Depends(verify_api_key)])
async def list_finetune_jobs():
    return {"jobs": list(finetune_jobs.values())}

@app.get("/v1/finetune/status/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_finetune_status(job_id: str):
    if job_id not in finetune_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return finetune_jobs[job_id]

async def _run_finetune(job_id: str, req: FineTuneRequest, output_dir: Path):
    job = finetune_jobs[job_id]
    job["status"] = "running"
    output_dir.mkdir(parents=True, exist_ok=True)

    script = f"""
import sys
sys.path.insert(0, '/app')
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

print("Loading base model: {req.base_model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{req.base_model}",
    max_seq_length={req.max_seq_length},
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={req.lora_r},
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha={req.lora_r},
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)
print("Loading dataset: {req.dataset_path}")
dataset = load_dataset("json", data_files="{req.dataset_path}", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={req.max_seq_length},
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs={req.epochs},
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="{output_dir}",
        save_strategy="epoch",
    ),
)
print("Training started...")
trainer.train()
print("Saving GGUF (q4_k_m)...")
model.save_pretrained_gguf(
    "{output_dir}/{req.output_name}",
    tokenizer,
    quantization_method="q4_k_m",
)
print("Done.")
"""

    script_path = output_dir / "train.py"
    script_path.write_text(script)

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            decoded = line.decode().rstrip()
            job["log"].append(decoded)
            if len(job["log"]) > 200:
                job["log"] = job["log"][-200:]

        await proc.wait()
        if proc.returncode == 0:
            job["status"]      = "completed"
            job["finished_at"] = datetime.utcnow().isoformat()
            job["gguf_path"]   = f"{output_dir}/{req.output_name}-unsloth.Q4_K_M.gguf"
        else:
            job["status"]      = "failed"
            job["error"]       = f"Process exited with code {proc.returncode}"
            job["finished_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        job["status"]      = "failed"
        job["error"]       = str(e)
        job["finished_at"] = datetime.utcnow().isoformat()

# ── Dashboard UI ──────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(
        content=_DASHBOARD_HTML,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
    )

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Terminal</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
:root{
  --bg:#080808;--surface:#101010;--surface2:#181818;--border:#252525;
  --accent:#00ff9d;--accent2:#00b8ff;--warn:#ffaa00;--red:#ff4455;
  --muted:#484848;--text:#cccccc;--white:#f0f0f0;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'IBM Plex Mono',monospace;
     font-size:13px;height:100vh;display:grid;
     grid-template-rows:42px 36px 1fr;grid-template-columns:255px 1fr;overflow:hidden}

header{grid-column:1/-1;background:var(--surface);border-bottom:1px solid var(--border);
       display:flex;align-items:center;padding:0 18px;gap:14px}
.logo{font-size:10px;letter-spacing:4px;color:var(--accent);font-weight:600;text-transform:uppercase}
.logo em{color:var(--muted);font-style:normal}
.hstats{margin-left:auto;display:flex;gap:18px;align-items:center}
.hstat{font-size:10px}.hstat .lbl{color:var(--muted);margin-right:4px}
.hstat .val{color:var(--white);font-weight:500}
.dot{width:7px;height:7px;border-radius:50%;background:var(--muted);display:inline-block;margin-right:5px;transition:all .3s}
.dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}

/* tab bar */
.tabbar{grid-column:1/-1;background:var(--surface);border-bottom:1px solid var(--border);
        display:flex;align-items:stretch;padding:0 18px;gap:0}
.tab{padding:0 18px;font-size:10px;letter-spacing:2px;text-transform:uppercase;
     color:var(--muted);cursor:pointer;display:flex;align-items:center;
     border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}

aside{background:var(--surface);border-right:1px solid var(--border);
      display:flex;flex-direction:column;overflow:hidden}
.sec-hdr{padding:10px 14px 6px;font-size:9px;letter-spacing:2px;text-transform:uppercase;
         color:var(--muted);border-bottom:1px solid var(--border)}

.prompt-list{overflow-y:auto;border-bottom:1px solid var(--border);max-height:210px}
.pi{padding:9px 14px;cursor:pointer;display:flex;gap:10px;
    border-left:2px solid transparent;transition:background .1s}
.pi:hover{background:var(--surface2)}
.pi.active{background:var(--surface2);border-left-color:var(--accent)}
.pi-key{color:var(--accent);font-size:10px;font-weight:600;min-width:72px}
.pi-desc{color:var(--muted);font-size:9px;line-height:1.5;margin-top:2px}

.conv-list{flex:1;overflow-y:auto}
.ci{padding:9px 14px;cursor:pointer;border-left:2px solid transparent;transition:background .1s}
.ci:hover{background:var(--surface2)}
.ci.active{background:var(--surface2);border-left-color:var(--accent2)}
.ci-name{color:var(--text);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px}
.ci-meta{color:var(--muted);font-size:9px;margin-top:2px}
.ci-del{float:right;color:var(--muted);cursor:pointer;font-size:11px;padding:0 2px}
.ci-del:hover{color:var(--red)}
.new-btn{margin:10px 14px;padding:7px;background:transparent;
         border:1px solid var(--border);color:var(--accent);
         font-family:inherit;font-size:9px;letter-spacing:2px;cursor:pointer;
         text-transform:uppercase;transition:all .15s;width:calc(100% - 28px)}
.new-btn:hover{background:var(--surface2);border-color:var(--accent)}

main{display:grid;grid-template-rows:1fr auto;overflow:hidden}
.tab-page{display:none;height:100%;overflow:hidden;flex-direction:column}
.tab-page.active{display:flex}

/* chat */
.messages{overflow-y:auto;padding:18px 22px;display:flex;flex-direction:column;gap:14px;flex:1}
.msg{display:flex;gap:11px;animation:fi .2s ease}
@keyframes fi{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.msg-role{font-size:9px;letter-spacing:2px;text-transform:uppercase;
          min-width:68px;padding-top:3px;font-weight:600;flex-shrink:0}
.msg.user .msg-role{color:var(--accent2)}
.msg.assistant .msg-role{color:var(--accent)}
.msg.system .msg-role{color:var(--muted)}
.msg-body{flex:1;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.msg.user .msg-body{color:var(--white)}
.msg.system .msg-body{color:var(--muted);font-style:italic;font-size:11px}

.empty{flex:1;display:flex;flex-direction:column;align-items:center;
       justify-content:center;color:var(--muted);gap:10px;user-select:none}
.empty-glyph{font-size:32px;letter-spacing:8px;color:var(--surface2)}
.empty-sub{font-size:9px;letter-spacing:3px;text-transform:uppercase}

.input-bar{border-top:1px solid var(--border);background:var(--surface);padding:11px 15px;flex-shrink:0}
.ctrls{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.lbl{font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
select,input[type=text]{background:var(--surface2);border:1px solid var(--border);
       color:var(--text);font-family:inherit;font-size:10px;padding:4px 7px;outline:none;cursor:pointer}
select:focus,input[type=text]:focus{border-color:var(--accent)}
#conv-label{margin-left:auto;font-size:9px;color:var(--muted);max-width:160px;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row{display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;background:var(--surface2);border:1px solid var(--border);color:var(--white);
         font-family:'IBM Plex Mono',monospace;font-size:13px;
         padding:9px 13px;resize:none;outline:none;line-height:1.6;
         min-height:42px;max-height:180px;transition:border-color .15s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.send-btn{background:var(--accent);color:var(--bg);border:none;
          font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;
          letter-spacing:2px;text-transform:uppercase;padding:0 18px;
          height:42px;cursor:pointer;transition:all .15s;min-width:76px;white-space:nowrap}
.send-btn:hover{filter:brightness(1.15)}
.send-btn:disabled{background:var(--muted);cursor:not-allowed;filter:none}
.send-btn.think{background:var(--warn)}
#typing{display:none;color:var(--accent);font-size:10px;padding:3px 0;
        animation:pulse 1.2s infinite;flex-shrink:0;padding:4px 15px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* ingest panel */
.ingest-panel{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:20px}
.ingest-card{background:var(--surface);border:1px solid var(--border);padding:18px;display:flex;flex-direction:column;gap:12px}
.ingest-card h3{font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:4px}
.ingest-card p{font-size:10px;color:var(--muted);line-height:1.6}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted)}
.field input[type=text],.field input[type=url],.field textarea,.field select{
  width:100%;background:var(--surface2);border:1px solid var(--border);
  color:var(--white);font-family:inherit;font-size:12px;
  padding:8px 11px;outline:none;resize:vertical}
.field input:focus,.field textarea:focus,.field select:focus{border-color:var(--accent)}
.field textarea{min-height:60px;max-height:120px;line-height:1.5}
.file-drop{border:1px dashed var(--border);padding:20px;text-align:center;
           cursor:pointer;transition:all .2s;position:relative;background:var(--surface2)}
.file-drop:hover,.file-drop.over{border-color:var(--accent);background:var(--bg)}
.file-drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.file-drop .fd-txt{font-size:10px;color:var(--muted);letter-spacing:1px}
.file-drop .fd-name{font-size:11px;color:var(--accent);margin-top:4px}
.ingest-result{background:var(--surface2);border:1px solid var(--border);
               padding:14px;font-size:12px;line-height:1.7;
               white-space:pre-wrap;word-break:break-word;max-height:320px;overflow-y:auto;display:none}
.ingest-result.visible{display:block}
.ingest-result .src-meta{font-size:9px;color:var(--muted);margin-bottom:10px;
                          padding-bottom:8px;border-bottom:1px solid var(--border)}
.ingest-result .answer{color:var(--text)}
.conv-badge{display:inline-block;background:var(--surface);border:1px solid var(--accent2);
            color:var(--accent2);font-size:9px;letter-spacing:1px;padding:3px 8px;
            margin-top:8px;cursor:pointer}
.conv-badge:hover{background:var(--surface2)}
.err{color:var(--red);font-size:11px}
.val{color:var(--white);font-weight:600;margin-left:4px}
.spinner{display:none;color:var(--accent);font-size:10px;animation:pulse 1s infinite}
</style>
</head>
<body>
<header>
  <div class="logo">LLM<em>/</em>TERMINAL</div>
  <div style="color:var(--muted);font-size:9px;letter-spacing:1px">RTX 3060 · MISTRAL 7B · LOCAL</div>
  <div class="hstats">
    <div class="hstat"><span class="dot" id="dot"></span><span class="lbl">OLLAMA</span></div>
    <div class="hstat"><span class="lbl">REQ</span><span class="val" id="h-req">—</span></div>
    <div class="hstat"><span class="lbl">TOK</span><span class="val" id="h-tok">—</span></div>
    <div class="hstat"><span class="lbl">UP</span><span class="val" id="h-up">—</span></div>
  </div>
</header>

<div class="tabbar">
  <div class="tab active" onclick="switchTab('chat',this)">CHAT</div>
  <div class="tab" onclick="switchTab('ingest',this)">INGEST</div>
  <div class="tab" onclick="switchTab('memory',this);loadMemory()">MEMORY</div>
  <div class="tab" onclick="switchTab('search',this);checkSearchStatus();loadAuxSites()">SEARCH</div>
</div>

<aside>
  <div class="sec-hdr">System Prompts</div>
  <div class="prompt-list" id="prompt-list"></div>
  <div class="sec-hdr">Conversations</div>
  <button class="new-btn" onclick="newConv()">+ New Conversation</button>
  <div class="conv-list" id="conv-list"></div>
</aside>

<main>
  <!-- ── CHAT TAB ── -->
  <div class="tab-page active" id="page-chat">
    <div class="messages" id="messages">
      <div class="empty" id="empty">
        <div class="empty-glyph">░ ░ ░</div>
        <div class="empty-sub">Choose a prompt · Start a conversation</div>
      </div>
    </div>
    <div id="typing">● generating…</div>
    <div class="input-bar">
      <div class="ctrls">
        <span class="lbl">Model</span>
        <select id="msel" onchange="onMsel()" style="max-width:200px">
          <option value="">loading…</option>
        </select>
        <span class="lbl" style="margin-left:12px">Mode</span>
        <select id="mode" onchange="onMode()">
          <option value="conv">Conversation</option>
          <option value="reasoning">Reasoning</option>
          <option value="stateless">Stateless</option>
        </select>
        <span class="lbl" style="margin-left:12px">Prompt</span>
        <select id="psel" onchange="onPsel()">
          <option value="default">default</option>
          <option value="research">research</option>
          <option value="epistemic">epistemic</option>
          <option value="analyst">analyst</option>
          <option value="esoteric">esoteric</option>
          <option value="reasoning">reasoning</option>
        </select>
        <span id="conv-label"></span>
        <span id="search-indicator" style="display:none;color:var(--warn);font-size:10px;
          margin-left:auto;animation:pulse 1s infinite">🔍 searching web…</span>
      </div>
      <div class="row">
        <textarea id="inp" rows="1" placeholder="Message… (Shift+Enter for newline)"
          onkeydown="onKey(event)" oninput="resize(this)"></textarea>
        <button class="send-btn" id="sbtn" onclick="send()">SEND</button>
      </div>
    </div>
  </div>

  <!-- ── INGEST TAB ── -->
  <div class="tab-page" id="page-ingest">
    <div class="ingest-panel">

      <!-- URL ingestion -->
      <div class="ingest-card">
        <h3>↗ Web Article / URL</h3>
        <p>Fetch any public article, blog post, Wikipedia page, or YouTube video (transcript). Paste URL below.</p>
        <div class="field">
          <label>URL</label>
          <input type="url" id="url-input" placeholder="https://example.com/article">
        </div>
        <div class="field">
          <label>Question / Instruction</label>
          <textarea id="url-question" placeholder="Summarise the key points. Apply epistemic analysis. What is omitted?">Summarise the key points of this article.</textarea>
        </div>
        <div class="ctrls" style="gap:10px">
          <span class="lbl">Prompt</span>
          <select id="url-prompt">
            <option value="default">default</option>
            <option value="epistemic">epistemic</option>
            <option value="analyst">analyst</option>
            <option value="esoteric">esoteric</option>
            <option value="reasoning">reasoning</option>
          </select>
          <span class="lbl" style="margin-left:10px">Temp</span>
          <input type="text" id="url-temp" value="0.3" style="width:48px">
          <div style="margin-left:auto;display:flex;gap:8px">
            <button class="send-btn" style="background:var(--surface2);color:var(--accent2);border:1px solid var(--accent2)"
              onclick="urlToConv()">LOAD INTO CONV</button>
            <button class="send-btn" onclick="submitUrl()" id="url-btn">ANALYSE</button>
          </div>
        </div>
        <div class="spinner" id="url-spin">● fetching &amp; analysing…</div>
        <div class="ingest-result" id="url-result"></div>
      </div>

      <!-- File ingestion -->
      <div class="ingest-card">
        <h3>⇪ Upload Document</h3>
        <p>Supported: <strong style="color:var(--text)">txt · md · csv · pdf · docx · rtf</strong>. File is read locally — nothing is stored permanently.</p>
        <div class="field">
          <label>File</label>
          <div class="file-drop" id="file-drop"
            ondragover="event.preventDefault();this.classList.add('over')"
            ondragleave="this.classList.remove('over')"
            ondrop="onDrop(event)">
            <input type="file" id="file-input"
              accept=".txt,.md,.csv,.pdf,.docx,.rtf,.log,.json"
              onchange="onFileSelect(this)">
            <div class="fd-txt">Drop file here or click to browse</div>
            <div class="fd-name" id="file-name"></div>
          </div>
        </div>
        <div class="field">
          <label>Question / Instruction</label>
          <textarea id="doc-question" placeholder="What are the main claims? Summarise. Find inconsistencies.">Summarise the key points of this document.</textarea>
        </div>
        <div class="ctrls" style="gap:10px">
          <span class="lbl">Prompt</span>
          <select id="doc-prompt">
            <option value="default">default</option>
            <option value="epistemic">epistemic</option>
            <option value="analyst">analyst</option>
            <option value="esoteric">esoteric</option>
            <option value="reasoning">reasoning</option>
          </select>
          <span class="lbl" style="margin-left:10px">Temp</span>
          <input type="text" id="doc-temp" value="0.3" style="width:48px">
          <div style="margin-left:auto;display:flex;gap:8px">
            <button class="send-btn" id="doc-conv-btn" disabled
              style="background:var(--surface2);color:var(--accent2);border:1px solid var(--accent2)"
              onclick="docToConv()">LOAD INTO CONV</button>
            <button class="send-btn" id="doc-btn" disabled onclick="submitDoc()">ANALYSE</button>
          </div>
        </div>
        <div class="spinner" id="doc-spin">● extracting &amp; analysing…</div>
        <div class="ingest-result" id="doc-result"></div>
      </div>

    </div>
  </div>

  <!-- ── MEMORY TAB ── -->
  <div class="tab-page" id="page-memory">
    <div class="ingest-panel">

      <!-- Stats bar -->
      <div class="ingest-card" style="padding:14px 18px">
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
          <div><span class="lbl">CHUNKS</span> <span class="val" id="m-chunks">—</span></div>
          <div><span class="lbl">SOURCES</span> <span class="val" id="m-sources">—</span></div>
          <div><span class="lbl">CONV MEM</span> <span class="val" id="m-convs">—</span></div>
          <div><span class="lbl" id="m-status" style="color:var(--muted)">loading…</span></div>
          <button class="send-btn" style="margin-left:auto;height:32px;font-size:9px"
            onclick="loadMemory()">↻ REFRESH</button>
        </div>
      </div>

      <!-- Semantic search -->
      <div class="ingest-card">
        <h3>⌕ Search Knowledge Base</h3>
        <p>Semantic search across all ingested articles, documents, and conversations.</p>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="text" id="mem-search-q" style="flex:1;padding:8px 11px;font-size:12px"
            placeholder="Search your knowledge base…"
            onkeydown="if(event.key==='Enter')searchMemory()">
          <button class="send-btn" onclick="searchMemory()" style="height:38px">SEARCH</button>
        </div>
        <div id="mem-search-results" style="margin-top:10px"></div>
      </div>

      <!-- Source list -->
      <div class="ingest-card">
        <h3>◈ Ingested Sources</h3>
        <p>All articles and documents stored in the knowledge base. Click source to search it. Delete to remove.</p>
        <div id="mem-sources-list" style="margin-top:8px">
          <div style="color:var(--muted);font-size:10px">Loading…</div>
        </div>
      </div>

      <!-- Search log -->
      <div class="ingest-card">
        <h3>🔍 Recent Web Searches</h3>
        <p>Auto-triggered and manual searches. Each search fetches and stores pages into the knowledge base.</p>
        <div style="display:flex;gap:8px;margin-bottom:10px">
          <input type="text" id="manual-search-q" style="flex:1;padding:8px 11px;font-size:12px"
            placeholder="Manual search query…"
            onkeydown="if(event.key==='Enter')manualSearch()">
          <input type="text" id="manual-search-n" value="3" style="width:48px;padding:8px;font-size:12px"
            title="Number of pages to fetch">
          <button class="send-btn" onclick="manualSearch()" style="height:38px">SEARCH + INGEST</button>
        </div>
        <div id="search-log-list" style="margin-top:4px">
          <div style="color:var(--muted);font-size:10px">Loading…</div>
        </div>
      </div>

    </div>
  </div>

  <!-- ── SEARCH TAB ── -->
  <div class="tab-page" id="page-search">
    <div class="ingest-panel">

      <!-- Search status -->
      <div class="ingest-card" style="padding:14px 18px">
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
          <div><span class="lbl">SEARXNG</span> <span class="val" id="s-status">—</span></div>
          <div><span class="lbl">THRESHOLD</span> <span class="val" id="s-thresh">—</span></div>
          <div><span class="lbl">AUTO-SEARCH</span> <span class="val" style="color:var(--accent)">ON</span></div>
          <button class="send-btn" style="margin-left:auto;height:32px;font-size:9px"
            onclick="checkSearchStatus()">↻ STATUS</button>
        </div>
      </div>

      <!-- Manual search -->
      <div class="ingest-card">
        <h3>🔍 Manual Web Search</h3>
        <p>Search the web, fetch full article text, store results in memory, and get an AI answer — all in one step. Results are automatically available in future chats.</p>
        <p style="margin-top:6px">You can also type <strong style="color:var(--accent)">/search your query</strong> directly in any chat window to trigger a search inline.</p>
        <div class="field" style="margin-top:8px">
          <label>Search Query</label>
          <input type="text" id="ws-query"
            placeholder="What happened at the UN Security Council this week?"
            onkeydown="if(event.key==='Enter')runSearch()"
            style="font-size:12px;padding:8px 11px">
        </div>
        <div class="ctrls" style="gap:10px;margin-top:8px">
          <span class="lbl">Results</span>
          <select id="ws-num" style="padding:4px 7px">
            <option value="3">3 pages</option>
            <option value="5" selected>5 pages</option>
            <option value="8">8 pages</option>
          </select>
          <span class="lbl" style="margin-left:10px">Prompt</span>
          <select id="ws-prompt">
            <option value="default">default</option>
            <option value="epistemic">epistemic</option>
            <option value="analyst">analyst</option>
            <option value="reasoning">reasoning</option>
          </select>
          <div style="margin-left:auto;display:flex;gap:8px">
            <button class="send-btn"
              style="background:var(--surface2);color:var(--accent2);border:1px solid var(--accent2)"
              onclick="runSearchToConv()">SEARCH + LOAD CONV</button>
            <button class="send-btn" onclick="runSearch()" id="ws-btn">SEARCH + ANSWER</button>
          </div>
        </div>
        <div class="spinner" id="ws-spin">● searching &amp; fetching pages…</div>
        <div class="ingest-result" id="ws-result"></div>
      </div>

      <!-- Auxiliary sites -->
      <div class="ingest-card">
        <h3>📌 Auxiliary Sites</h3>
        <p>Sites always fetched alongside every web search. Great for trusted sources, internal wikis, or specific feeds you always want included.</p>
        <div style="display:flex;gap:8px;margin-top:10px;align-items:flex-start;flex-wrap:wrap">
          <input type="url" id="aux-url" style="flex:1;min-width:200px;padding:8px 11px;font-size:12px"
            placeholder="https://example.com"
            onkeydown="if(event.key==='Enter')addAuxSite()">
          <input type="text" id="aux-label" style="width:160px;padding:8px 11px;font-size:12px"
            placeholder="Label (optional)"
            onkeydown="if(event.key==='Enter')addAuxSite()">
          <button class="send-btn" onclick="addAuxSite()" style="height:38px">ADD SITE</button>
        </div>
        <div id="aux-sites-list" style="margin-top:12px">
          <div style="color:var(--muted);font-size:10px">Loading…</div>
        </div>
      </div>

      <!-- Recent searches -->
      <div class="ingest-card">
        <h3>◷ Recent Searches</h3>
        <p>All web searches this session — auto-triggered and manual. Each one is stored in memory.</p>
        <div id="ws-history" style="margin-top:8px">
          <div style="color:var(--muted);font-size:10px">No searches yet this session.</div>
        </div>
      </div>

    </div>
  </div>

</main>

<script>
const AKEY = 'change-me-in-production';
let activeConv = null, activePrompt = 'default', localMsgs = [], activeModel = '';
let selectedFile = null;

async function api(path, opts={}) {
  const r = await fetch(path, {
    ...opts,
    headers:{'Authorization':'Bearer '+AKEY,...(opts.headers||{})}
  });
  if (!r.ok) {
    const t = await r.text();
    let msg = t;
    try { msg = JSON.parse(t).detail || t; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-page').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('page-'+name).classList.add('active');
}

async function init() {
  await Promise.all([poll(), loadPrompts(), loadConvs(), loadModels()]);
  setInterval(poll, 16000);
}

async function loadModels() {
  try {
    const d = await api('/v1/models');
    const sel = document.getElementById('msel');
    const models = d.data.map(m => m.id);
    // Put dolphin and deepseek first if present, then the rest
    const preferred = ['dolphin-mistral', 'deepseek-r1:8b', 'deepseek-r1:14b'];
    const sorted = [
      ...preferred.filter(p => models.some(m => m.includes(p.split(':')[0]))),
      ...models.filter(m => !preferred.some(p => m.includes(p.split(':')[0]))),
    ];
    // Deduplicate while preserving order
    const seen = new Set();
    const deduped = models.filter(m => { if(seen.has(m)) return false; seen.add(m); return true; });
    sel.innerHTML = deduped.map(m => {
      const label = m.includes('dolphin') ? m + ' ★' :
                    m.includes('deepseek') ? m + ' ◆' : m;
      return `<option value="${esc(m)}">${esc(label)}</option>`;
    }).join('');
    // Default to dolphin if available, else first
    const preferred_default = deduped.find(m => m.includes('dolphin')) ||
                              deduped.find(m => m.includes('deepseek')) ||
                              deduped[0] || '';
    sel.value = preferred_default;
    activeModel = sel.value;
  } catch(e) {
    const sel = document.getElementById('msel');
    if(sel) sel.innerHTML = '<option value="">unavailable</option>';
  }
}

function onMsel() {
  activeModel = document.getElementById('msel').value;
}

async function poll() {
  try {
    const [h,m] = await Promise.all([
      fetch('/health').then(r=>r.json()),
      api('/metrics'),
    ]);
    document.getElementById('dot').className='dot '+(h.ollama==='connected'?'on':'');
    document.getElementById('h-up').textContent=fmt(h.uptime_seconds);
    document.getElementById('h-req').textContent=m.total_requests;
    document.getElementById('h-tok').textContent=fmtN(m.total_tokens_generated);
  } catch {}
}

async function loadPrompts() {
  try {
    const d = await api('/v1/system-prompts');
    const descs={default:'Honest, calibrated',epistemic:'Source criticism',
                 analyst:'Structured analysis',esoteric:'Mystery traditions',
                 reasoning:'Chain-of-thought'};
    document.getElementById('prompt-list').innerHTML=
      Object.keys(d.prompts).map(k=>`
        <div class="pi ${k===activePrompt?'active':''}" onclick="selPrompt('${k}')">
          <div><div class="pi-key">${k}</div>
          <div class="pi-desc">${descs[k]||''}</div></div>
        </div>`).join('');
  } catch {}
}

async function loadConvs() {
  try {
    const d = await api('/v1/conversations');
    const el = document.getElementById('conv-list');
    if (!d.conversations.length) {
      el.innerHTML='<div style="padding:10px 14px;color:var(--muted);font-size:10px">No conversations</div>';
      return;
    }
    el.innerHTML=d.conversations.map(c=>`
      <div class="ci ${c.id===activeConv?'active':''}" onclick="selConv('${c.id}')">
        <span class="ci-del" onclick="event.stopPropagation();delConv('${c.id}')">✕</span>
        <div class="ci-name">${c.name}</div>
        <div class="ci-meta">${c.system_prompt_key} · ${c.turn_count} turns</div>
      </div>`).join('');
  } catch {}
}

function selPrompt(k) {
  activePrompt=k;
  document.getElementById('psel').value=k;
  loadPrompts();
}
function onPsel(){activePrompt=document.getElementById('psel').value;loadPrompts();}
function onMode(){
  const m=document.getElementById('mode').value;
  const b=document.getElementById('sbtn');
  b.className='send-btn'+(m==='reasoning'?' think':'');
  b.textContent=m==='reasoning'?'THINK':'SEND';
}

async function newConv(){
  const name=prompt('Conversation name:','New Conversation');
  if(!name)return;
  try{
    const c=await api('/v1/conversations',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,system_prompt_key:activePrompt})});
    activeConv=c.id;localMsgs=[];
    document.getElementById('messages').innerHTML='';
    document.getElementById('conv-label').textContent=c.name;
    document.getElementById('mode').value='conv';onMode();
    await loadConvs();
  }catch(e){alert(e.message);}
}

async function selConv(id){
  activeConv=id;
  try{
    const c=await api('/v1/conversations/'+id);
    document.getElementById('conv-label').textContent=c.name;
    renderMsgs(c.messages);
    document.getElementById('mode').value='conv';onMode();
    switchTab('chat',document.querySelectorAll('.tab')[0]);
    await loadConvs();
  }catch(e){alert(e.message);}
}

async function delConv(id){
  if(!confirm('Delete conversation?'))return;
  try{
    await api('/v1/conversations/'+id,{method:'DELETE'});
    if(activeConv===id){
      activeConv=null;
      document.getElementById('messages').innerHTML=
        '<div class="empty" id="empty"><div class="empty-glyph">░ ░ ░</div>'+
        '<div class="empty-sub">Choose a prompt · Start a conversation</div></div>';
      document.getElementById('conv-label').textContent='';
    }
    await loadConvs();
  }catch(e){alert(e.message);}
}

function renderMsgs(msgs){
  const el=document.getElementById('messages');
  el.querySelectorAll('.msg').forEach(e=>e.remove());
  document.getElementById('empty')?.remove();
  msgs.filter(m=>m.role!=='system').forEach(m=>addMsg(m.role,m.content,false));
  el.scrollTop=99999;
}

function addMsg(role,content,scroll=true){
  document.getElementById('empty')?.remove();
  const d=document.createElement('div');
  d.className='msg '+role;
  d.innerHTML=`<div class="msg-role">${role}</div><div class="msg-body">${esc(content)}</div>`;
  document.getElementById('messages').appendChild(d);
  if(scroll)document.getElementById('messages').scrollTop=99999;
}

async function send(){
  const inp=document.getElementById('inp');
  const text=inp.value.trim();
  if(!text)return;
  const mode=document.getElementById('mode').value;
  const btn=document.getElementById('sbtn');
  inp.value='';resize(inp);btn.disabled=true;
  document.getElementById('typing').style.display='block';
  addMsg('user',text);
  try{
    if(mode==='conv'&&activeConv){
      const d=await api('/v1/conversations/'+activeConv+'/messages',
        {method:'POST',headers:{'Content-Type':'application/json'},
         body:JSON.stringify({content:text,temperature:0.7,max_tokens:2048,model:activeModel})});
      addMsg('assistant',d.reply);
    }else if(mode==='reasoning'){
      const msgs=[...localMsgs,{role:'user',content:text}];
      const d=await api('/v1/chat/reasoning',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({messages:msgs,system_prompt_key:activePrompt,
                             model:activeModel,temperature:0.3,max_tokens:4096})});
      const r=d.choices[0].message.content;
      localMsgs.push({role:'user',content:text},{role:'assistant',content:r});
      addMsg('assistant',r);
    }else{
      const msgs=[...localMsgs,{role:'user',content:text}];
      const d=await api('/v1/chat/completions',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({messages:msgs,system_prompt_key:activePrompt,
                             model:activeModel,temperature:0.7,max_tokens:2048})});
      const r=d.choices[0].message.content;
      localMsgs.push({role:'user',content:text},{role:'assistant',content:r});
      addMsg('assistant',r);
    }
  }catch(e){addMsg('assistant','⚠ Error: '+e.message);}
  finally{
    btn.disabled=false;
    document.getElementById('typing').style.display='none';
    document.getElementById('messages').scrollTop=99999;
  }
}

function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function resize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,180)+'px';}

/* ── Ingest ── */
function onFileSelect(input){
  if(input.files && input.files[0]){
    selectedFile=input.files[0];
    document.getElementById('file-name').textContent=selectedFile.name;
    document.getElementById('doc-btn').disabled=false;
    document.getElementById('doc-conv-btn').disabled=false;
  }
}
function onDrop(e){
  e.preventDefault();
  document.getElementById('file-drop').classList.remove('over');
  const f=e.dataTransfer.files[0];
  if(f){
    selectedFile=f;
    document.getElementById('file-name').textContent=f.name;
    document.getElementById('doc-btn').disabled=false;
    document.getElementById('doc-conv-btn').disabled=false;
  }
}

async function submitUrl(){
  const url=document.getElementById('url-input').value.trim();
  if(!url){alert('Enter a URL');return;}
  const q=document.getElementById('url-question').value.trim()||'Summarise the key points.';
  const pkey=document.getElementById('url-prompt').value;
  const temp=parseFloat(document.getElementById('url-temp').value)||0.3;
  const btn=document.getElementById('url-btn');
  const spin=document.getElementById('url-spin');
  const res=document.getElementById('url-result');
  btn.disabled=true;spin.style.display='block';res.className='ingest-result';
  try{
    const d=await api('/v1/ingest/url',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,question:q,system_prompt_key:pkey,temperature:temp,max_tokens:3000})});
    const s=d.source;
    res.innerHTML=
      `<div class="src-meta">SOURCE: ${esc(s.title||url)} · ${s.source_type} · `+
      `${(s.char_count||0).toLocaleString()} chars${s.truncated?' (truncated)':''}</div>`+
      `<div class="answer">${esc(d.answer)}</div>`;
    res.className='ingest-result visible';
  }catch(e){
    res.innerHTML=`<span class="err">⚠ ${esc(e.message)}</span>`;
    res.className='ingest-result visible';
  }finally{btn.disabled=false;spin.style.display='none';}
}

async function urlToConv(){
  const url=document.getElementById('url-input').value.trim();
  if(!url){alert('Enter a URL');return;}
  const pkey=document.getElementById('url-prompt').value;
  const btn=document.getElementById('url-btn');
  const spin=document.getElementById('url-spin');
  btn.disabled=true;spin.style.display='block';
  try{
    const d=await api('/v1/ingest/url/conversation',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,system_prompt_key:pkey})});
    await loadConvs();
    activeConv=d.conversation_id;
    document.getElementById('conv-label').textContent=d.name;
    document.getElementById('messages').innerHTML='';
    addMsg('assistant',`Article loaded: ${d.source.title||url}\n${(d.source.char_count||0).toLocaleString()} chars · ${d.source.source_type}${d.source.truncated?' (truncated)':''}\n\nAsk me anything about this article.`);
    document.getElementById('mode').value='conv';onMode();
    switchTab('chat',document.querySelectorAll('.tab')[0]);
    await loadConvs();
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;spin.style.display='none';}
}

async function submitDoc(){
  if(!selectedFile){alert('Select a file first');return;}
  const q=document.getElementById('doc-question').value.trim()||'Summarise.';
  const pkey=document.getElementById('doc-prompt').value;
  const temp=parseFloat(document.getElementById('doc-temp').value)||0.3;
  const btn=document.getElementById('doc-btn');
  const convBtn=document.getElementById('doc-conv-btn');
  const spin=document.getElementById('doc-spin');
  const res=document.getElementById('doc-result');
  const fd=new FormData();
  fd.append('file', selectedFile);
  fd.append('question',q);
  fd.append('system_prompt_key',pkey);
  fd.append('temperature',temp);
  fd.append('max_tokens','3000');
  btn.disabled=true;convBtn.disabled=true;
  spin.style.display='block';res.className='ingest-result';
  try{
    const r=await fetch('/v1/ingest/document',{method:'POST',
      headers:{'Authorization':'Bearer '+AKEY},body:fd});
    if(!r.ok){const t=await r.text();let m=t;try{m=JSON.parse(t).detail||t;}catch{}throw new Error(m);}
    const d=await r.json();
    const s=d.source;
    res.innerHTML=
      `<div class="src-meta">FILE: ${esc(s.filename)} · ${s.source_type} · `+
      `${(s.char_count||0).toLocaleString()} chars${s.truncated?' (truncated)':''}` +
      (d.memory&&d.memory.chunks_stored?` · 🧠 ${d.memory.chunks_stored} chunks stored`:'')+'</div>'+
      `<div class="answer">${esc(d.answer)}</div>`;
    res.className='ingest-result visible';
  }catch(e){
    res.innerHTML=`<span class="err">⚠ ${esc(e.message)}</span>`;
    res.className='ingest-result visible';
  }finally{
    btn.disabled=false;convBtn.disabled=false;
    spin.style.display='none';
  }
}

async function docToConv(){
  if(!selectedFile){alert('Select a file first');return;}
  const pkey=document.getElementById('doc-prompt').value;
  const btn=document.getElementById('doc-btn');
  const convBtn=document.getElementById('doc-conv-btn');
  const spin=document.getElementById('doc-spin');
  const fd=new FormData();
  fd.append('file', selectedFile);
  fd.append('system_prompt_key',pkey);
  btn.disabled=true;convBtn.disabled=true;
  spin.style.display='block';
  try{
    const r=await fetch('/v1/ingest/document/conversation',{method:'POST',
      headers:{'Authorization':'Bearer '+AKEY},body:fd});
    if(!r.ok){const t=await r.text();let m=t;try{m=JSON.parse(t).detail||t;}catch{}throw new Error(m);}
    const d=await r.json();
    await loadConvs();
    activeConv=d.conversation_id;
    document.getElementById('conv-label').textContent=d.name;
    document.getElementById('messages').innerHTML='';
    addMsg('assistant',
      `Document loaded: ${d.source.filename}\n`+
      `${(d.source.char_count||0).toLocaleString()} chars · ${d.source.source_type}`+
      `${d.source.truncated?' (truncated)':''}\n\nAsk me anything about this document.`
    );
    document.getElementById('mode').value='conv';onMode();
    switchTab('chat',document.querySelectorAll('.tab')[0]);
    await loadConvs();
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;convBtn.disabled=false;spin.style.display='none';}
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fmt(s){return s<60?s+'s':s<3600?Math.floor(s/60)+'m':Math.floor(s/3600)+'h';}
function fmtN(n){return n>1e6?(n/1e6).toFixed(1)+'M':n>1e3?(n/1e3).toFixed(1)+'K':String(n);}

/* ── Memory tab ── */
async function loadMemory() {
  try {
    const d = await api('/v1/memory/stats');
    if (!d.available) {
      document.getElementById('m-status').textContent = 'ChromaDB unavailable';
      document.getElementById('m-status').style.color = 'var(--red)';
      document.getElementById('m-sources-list').innerHTML =
        '<div style="color:var(--red);font-size:11px">Memory not available — is ChromaDB running?</div>';
      return;
    }
    document.getElementById('m-chunks').textContent  = (d.knowledge_chunks||0).toLocaleString();
    document.getElementById('m-sources').textContent = (d.unique_sources||0).toLocaleString();
    document.getElementById('m-convs').textContent   = (d.conversation_chunks||0).toLocaleString();
    document.getElementById('m-status').textContent  = 'ONLINE';
    document.getElementById('m-status').style.color  = 'var(--accent)';
    renderSources(d.sources || []);
  } catch(e) {
    document.getElementById('m-status').textContent = 'Error: ' + e.message;
    document.getElementById('m-status').style.color = 'var(--red)';
  }
}

function renderSources(sources) {
  const el = document.getElementById('mem-sources-list');
  if (!sources.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:10px">No sources ingested yet. Use the INGEST tab to add articles and documents.</div>';
    return;
  }
  const typeColor = t => t==='url'||t==='youtube_transcript'?'var(--accent2)':
                         t==='conversation'?'var(--warn)':'var(--accent)';
  el.innerHTML = `
    <table style="width:100%;border-collapse:collapse">
      <thead><tr style="font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px">
        <th style="text-align:left;padding:5px 8px">TYPE</th>
        <th style="text-align:left;padding:5px 8px">TITLE</th>
        <th style="text-align:left;padding:5px 8px">DATE</th>
        <th style="text-align:left;padding:5px 8px"></th>
      </tr></thead>
      <tbody>
        ${sources.map(s=>`
          <tr style="border-top:1px solid var(--border);font-size:11px">
            <td style="padding:7px 8px">
              <span style="color:${typeColor(s.source_type)};font-size:9px;
                letter-spacing:1px;text-transform:uppercase">${s.source_type}</span>
            </td>
            <td style="padding:7px 8px;color:var(--text);max-width:320px;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                title="${esc(s.identifier||'')}">${esc(s.title||s.identifier||'—')}</td>
            <td style="padding:7px 8px;color:var(--muted);font-size:10px;white-space:nowrap">
              ${(s.timestamp||'').slice(0,10)||'—'}
            </td>
            <td style="padding:7px 8px;white-space:nowrap">
              <span style="color:var(--accent2);cursor:pointer;font-size:10px;margin-right:12px"
                onclick="searchBySource('${esc(s.title||s.identifier||'')}')">search</span>
              <span style="color:var(--red);cursor:pointer;font-size:10px"
                onclick="deleteSource('${esc(s.source_id||'')}','${esc(s.title||s.identifier||'')}')">delete</span>
            </td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

async function searchMemory() {
  const q = document.getElementById('mem-search-q').value.trim();
  if (!q) return;
  const el = document.getElementById('mem-search-results');
  el.innerHTML = '<div style="color:var(--muted);font-size:10px;animation:pulse 1s infinite">● searching…</div>';
  try {
    const d = await api('/v1/memory/search?q=' + encodeURIComponent(q) + '&top_k=6');
    if (!d.results.length) {
      el.innerHTML = '<div style="color:var(--muted);font-size:10px">No results found.</div>';
      return;
    }
    el.innerHTML = d.results.map((r,i)=>`
      <div style="border-top:1px solid var(--border);padding:10px 0">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
          <span style="font-size:9px;color:var(--accent2);text-transform:uppercase;letter-spacing:1px">
            ${r.source_type} · ${esc(r.title||r.identifier||'').slice(0,60)}
          </span>
          <span style="font-size:9px;color:var(--accent)">score: ${r.score}</span>
        </div>
        <div style="font-size:11px;color:var(--text);line-height:1.6;
             white-space:pre-wrap;word-break:break-word">${esc(r.text.slice(0,400))}${r.text.length>400?'…':''}</div>
      </div>`).join('');
  } catch(e) {
    el.innerHTML = `<div style="color:var(--red);font-size:11px">⚠ ${esc(e.message)}</div>`;
  }
}

function searchBySource(title) {
  document.getElementById('mem-search-q').value = title;
  searchMemory();
}

async function deleteSource(sourceId, title) {
  if (!confirm(`Delete all memory chunks for:\n"${title}"?`)) return;
  try {
    const d = await api('/v1/memory/source/' + encodeURIComponent(sourceId), {method:'DELETE'});
    await loadMemory();
  } catch(e) {
    alert('Delete failed: ' + e.message);
  }
}

/* ── Web Search tab ── */
let wsHistory = [];

async function checkSearchStatus() {
  try {
    const d = await fetch('/health').then(r => r.json());
    const s = d.search || {};
    document.getElementById('s-status').textContent  = s.searxng || '—';
    document.getElementById('s-status').style.color  =
      (s.searxng||'').includes('reachable') && !(s.searxng||'').includes('un')
        ? 'var(--accent)' : 'var(--warn)';
    document.getElementById('s-thresh').textContent = '0.35';
  } catch(e) {
    document.getElementById('s-status').textContent = 'error';
    document.getElementById('s-status').style.color = 'var(--red)';
  }
}

async function runSearch() {
  const query = document.getElementById('ws-query').value.trim();
  if (!query) { alert('Enter a search query'); return; }
  const num    = parseInt(document.getElementById('ws-num').value) || 5;
  const pkey   = document.getElementById('ws-prompt').value;
  const btn    = document.getElementById('ws-btn');
  const spin   = document.getElementById('ws-spin');
  const res    = document.getElementById('ws-result');
  btn.disabled = true; spin.style.display = 'block'; res.className = 'ingest-result';
  try {
    const d = await api('/v1/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        query, num_results: num, store_memory: true,
        answer: true, system_prompt_key: pkey, temperature: 0.3, max_tokens: 2048,
      }),
    });
    // Add to history
    wsHistory.unshift({query, results: d.results, stored: d.stored,
                        source: d.source, time: new Date().toLocaleTimeString()});
    renderSearchHistory();

    const srcs = (d.results||[]).map(r =>
      `<a href="${esc(r.url)}" target="_blank" style="color:var(--accent2);font-size:10px;
       display:block;margin:2px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
       title="${esc(r.url)}">${esc(r.title||r.url)}</a>`
    ).join('');

    res.innerHTML =
      `<div class="src-meta">QUERY: ${esc(query)} · ${d.results.length} pages · `+
      `${d.stored} chunks stored · via ${d.source||'unknown'}<br>`+
      `<div style="margin-top:6px">${srcs}</div></div>`+
      `<div class="answer">${esc(d.answer||'No answer generated.')}</div>`;
    res.className = 'ingest-result visible';
  } catch(e) {
    res.innerHTML = `<span class="err">⚠ ${esc(e.message)}</span>`;
    res.className = 'ingest-result visible';
  } finally { btn.disabled = false; spin.style.display = 'none'; }
}

async function runSearchToConv() {
  const query = document.getElementById('ws-query').value.trim();
  if (!query) { alert('Enter a search query'); return; }
  const num  = parseInt(document.getElementById('ws-num').value) || 5;
  const pkey = document.getElementById('ws-prompt').value;
  const btn  = document.getElementById('ws-btn');
  const spin = document.getElementById('ws-spin');
  btn.disabled = true; spin.style.display = 'block';
  try {
    // Run search + ingest first
    const d = await api('/v1/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, num_results: num, store_memory: true,
                             answer: false}),
    });
    wsHistory.unshift({query, results: d.results, stored: d.stored,
                        source: d.source, time: new Date().toLocaleTimeString()});
    renderSearchHistory();

    // Create a new conversation with the search context pre-loaded
    const conv = await api('/v1/conversations', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: `Search: ${query.slice(0,50)}`,
                             system_prompt_key: pkey}),
    });
    // Send the context as a seeding message
    await api(`/v1/conversations/${conv.id}/messages`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        content: `I just searched the web for: "${query}". `+
                 `${d.results.length} pages were fetched and stored in your memory. `+
                 `Please summarise what you found and be ready for follow-up questions.`,
        temperature: 0.3, max_tokens: 2048,
      }),
    });
    activeConv = conv.id;
    await loadConvs();
    const full = await api(`/v1/conversations/${conv.id}`);
    document.getElementById('conv-label').textContent = full.name;
    renderMsgs(full.messages);
    document.getElementById('mode').value = 'conv'; onMode();
    switchTab('chat', document.querySelectorAll('.tab')[0]);
    await loadConvs();
  } catch(e) { alert('Error: ' + e.message); }
  finally { btn.disabled = false; spin.style.display = 'none'; }
}

function renderSearchHistory() {
  const el = document.getElementById('ws-history');
  if (!wsHistory.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:10px">No searches yet this session.</div>';
    return;
  }
  el.innerHTML = wsHistory.slice(0,20).map(s => `
    <div style="border-top:1px solid var(--border);padding:8px 0;display:flex;gap:12px;align-items:flex-start">
      <div style="flex:1">
        <div style="font-size:11px;color:var(--white);cursor:pointer"
          onclick="document.getElementById('ws-query').value='${esc(s.query)}'"
          >${esc(s.query)}</div>
        <div style="font-size:9px;color:var(--muted);margin-top:3px">
          ${s.results.length} pages · ${s.stored} chunks · ${s.source} · ${s.time}
        </div>
      </div>
      <span style="color:var(--accent2);font-size:10px;cursor:pointer;white-space:nowrap"
        onclick="document.getElementById('ws-query').value='${esc(s.query)}';runSearch()">re-run</span>
    </div>`).join('');
}

/* ── Auxiliary Sites ── */
async function loadAuxSites() {
  const el = document.getElementById('aux-sites-list');
  try {
    const d = await api('/v1/search/auxiliary-sites');
    renderAuxSites(d.sites);
  } catch(e) {
    if (el) el.innerHTML = `<div style="color:var(--red);font-size:10px">⚠ ${esc(e.message)}</div>`;
  }
}

function renderAuxSites(sites) {
  const el = document.getElementById('aux-sites-list');
  if (!el) return;
  if (!sites || !sites.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:10px">No auxiliary sites yet. Add a URL above to always include it in searches.</div>';
    return;
  }
  el.innerHTML = sites.map(s => `
    <div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-top:1px solid var(--border)">
      <div style="flex:1;overflow:hidden">
        <div style="font-size:11px;color:var(--white);white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
          title="${esc(s.url)}">${esc(s.label || s.url)}</div>
        <div style="font-size:9px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(s.url)}</div>
      </div>
      <span style="color:var(--red);font-size:10px;cursor:pointer;white-space:nowrap;flex-shrink:0"
        onclick="removeAuxSite('${esc(s.url)}')">✕ remove</span>
    </div>`).join('');
}

async function addAuxSite() {
  const url   = document.getElementById('aux-url').value.trim();
  const label = document.getElementById('aux-label').value.trim();
  if (!url) { alert('Enter a URL'); return; }
  const el = document.getElementById('aux-sites-list');
  try {
    const d = await api('/v1/search/auxiliary-sites', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url, label}),
    });
    document.getElementById('aux-url').value   = '';
    document.getElementById('aux-label').value = '';
    renderAuxSites(d.sites);
  } catch(e) {
    if (el) el.innerHTML += `<div style="color:var(--red);font-size:10px;margin-top:6px">⚠ ${esc(e.message)}</div>`;
  }
}

async function removeAuxSite(url) {
  if (!confirm(`Remove auxiliary site?\n${url}`)) return;
  try {
    const d = await api('/v1/search/auxiliary-sites?url=' + encodeURIComponent(url), {method: 'DELETE'});
    renderAuxSites(d.sites);
  } catch(e) {
    alert('Remove failed: ' + e.message);
  }
}

/* also expose manualSearch as alias used in memory tab */
async function manualSearch() {
  const q = document.getElementById('manual-search-q')?.value?.trim();
  const n = parseInt(document.getElementById('manual-search-n')?.value) || 3;
  if (!q) { alert('Enter a search query'); return; }
  const el = document.getElementById('search-log-list');
  if (el) el.innerHTML = '<div style="color:var(--accent);font-size:10px;animation:pulse 1s infinite">● searching…</div>';
  try {
    const d = await api('/v1/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, num_results: n, store_memory: true, answer: false}),
    });
    if (el) el.innerHTML = `<div style="font-size:10px;color:var(--accent)">✓ Stored ${d.stored} chunks from ${d.results.length} pages for: "${esc(q)}"</div>`;
  } catch(e) {
    if (el) el.innerHTML = `<div style="color:var(--red);font-size:10px">⚠ ${esc(e.message)}</div>`;
  }
}

init();
</script>
</body>
</html>"""

# ── WebSocket keepalive ───────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_text("ok")
    except WebSocketDisconnect:
        pass

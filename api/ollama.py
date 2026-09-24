"""Thin async client for the Ollama HTTP API."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

import httpx

from config import CFG, get_active_model

log = logging.getLogger("llm-api.ollama")

_DEFAULT_ALIASES = {"", "default", "auto"}


def normalize_model(model: str | None) -> str:
    """Resolve aliases ("default"/"auto"/empty → active model) and strip UI suffixes.

    Downstream apps should pass ``model="default"`` so a server-side swap via
    ``/v1/settings`` or ``DEFAULT_MODEL`` env propagates without per-app edits.
    """
    cleaned = (model or "").removesuffix(" [Search+Memory]").strip()
    if cleaned.lower() in _DEFAULT_ALIASES:
        return get_active_model()
    return cleaned


def build_payload(messages: list[dict], model: str, *, temperature: float = 0.7,
                  max_tokens: int = 2048, top_p: float = 0.9,
                  stop: list | None = None, stream: bool = False,
                  thinking: bool = False, fmt: str | dict | None = None) -> dict:
    payload: dict = {
        "model": normalize_model(model),
        "messages": messages,
        "stream": stream,
        "think": thinking,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "num_ctx": CFG.num_ctx,
            "stop": stop or [],
        },
    }
    # Ollama-native structured output: "json" or a JSON-schema dict. Only set
    # when a caller asks for it, so default chat behavior is unchanged.
    if fmt is not None:
        payload["format"] = fmt
    return payload


async def chat(payload: dict, *, timeout: float = 180.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{CFG.ollama_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


async def generate(prompt: str, *, model: str | None = None, system: str | None = None,
                   temperature: float = 0.7, max_tokens: int = 512,
                   num_ctx: int | None = None, thinking: bool = False) -> dict:
    # Default to the shared num_ctx: a different value forces a model reload.
    options: dict = {"temperature": temperature, "num_predict": max_tokens,
                     "num_ctx": num_ctx or CFG.num_ctx}
    payload: dict = {
        "model": normalize_model(model or CFG.default_model),
        "prompt": prompt,
        "stream": False,
        # Same default as build_payload: a thinking model otherwise spends the
        # num_predict budget on reasoning and returns an empty response.
        "think": thinking,
        "options": options,
    }
    if system:
        payload["system"] = system
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{CFG.ollama_url}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()


async def stream_chat(payload: dict, on_token=None, *,
                      timeout: float = 180.0) -> AsyncIterator[str]:
    """
    Yield SSE-formatted chunks compatible with OpenAI's streaming response.
    on_token(text) is invoked for each text delta (used to capture full reply).
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    saw_content = False
    think_buf: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{CFG.ollama_url}/api/chat", json=payload) as r:
            if r.status_code != 200:
                err = f"[Error: Ollama returned {r.status_code}]"
                yield _sse(chunk_id, payload["model"], delta=err, done=False)
                yield _sse(chunk_id, payload["model"], delta="", done=True)
                yield "data: [DONE]\n\n"
                return
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    yield _sse(chunk_id, payload["model"],
                               delta=f"[Error: {obj['error']}]", done=False)
                    yield _sse(chunk_id, payload["model"], delta="", done=True)
                    yield "data: [DONE]\n\n"
                    return
                m = obj.get("message", {}) or {}
                content = m.get("content", "") or ""
                reasoning = m.get("thinking", "") or ""
                done = obj.get("done", False)
                # Forward reasoning as a separate delta channel so clients can show
                # progress instead of stalling on an empty stream while the model thinks.
                if reasoning:
                    think_buf.append(reasoning)
                    yield _sse(chunk_id, payload["model"], delta="",
                               reasoning=reasoning, done=False)
                if content:
                    saw_content = True
                    if on_token:
                        on_token(content)
                # Reasoning consumed the whole budget: emit it as content rather than
                # closing the stream with nothing at all.
                if done and not saw_content and think_buf:
                    fallback = "".join(think_buf).strip()
                    if fallback:
                        if on_token:
                            on_token(fallback)
                        yield _sse(chunk_id, payload["model"], delta=fallback, done=False)
                yield _sse(chunk_id, payload["model"], delta=content, done=done)
                if done:
                    yield "data: [DONE]\n\n"
                    return


async def stream_events(payload: dict, *, timeout: float = 180.0) -> AsyncIterator[dict]:
    """Yield Ollama's raw streamed /api/chat objects (content and tool_calls).

    Unlike `stream_chat`, which renders OpenAI SSE for the client, this is for
    callers that must inspect each chunk — the tool-calling loop needs
    `message.tool_calls`. Raises RuntimeError on an HTTP or in-stream error.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{CFG.ollama_url}/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")[:200]
                raise RuntimeError(f"Ollama returned {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                yield obj
                if obj.get("done"):
                    return


def _delta(content: str, reasoning: str = "") -> dict:
    d: dict = {"content": content}
    if reasoning:
        d["reasoning"] = reasoning
    return d


def _sse(chunk_id: str, model: str, *, delta: str, done: bool,
         reasoning: str = "") -> str:
    body = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": _delta(delta, reasoning) if not done else {},
            "finish_reason": "stop" if done else None,
        }],
    }
    return f"data: {json.dumps(body)}\n\n"


def completion_envelope(data: dict, model: str, extra: dict | None = None) -> dict:
    msg = data.get("message", {}) or {}
    content = msg.get("content", "") or ""
    # Thinking models emit reasoning tokens before any content. If num_predict is
    # exhausted mid-reasoning, Ollama returns content="" with done_reason="length".
    # Surfacing the reasoning beats handing the caller a silent empty string.
    thinking = msg.get("thinking", "") or ""
    truncated = data.get("done_reason") == "length"
    assistant: dict = {"role": "assistant", "content": content}
    if thinking:
        assistant["reasoning"] = thinking
    if not content.strip() and thinking.strip():
        assistant["content"] = thinking.strip()
        assistant["reasoning_fallback"] = True
    out = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": assistant,
            "finish_reason": "length" if truncated else "stop",
        }],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }
    if extra:
        out.update(extra)
    return out


async def list_models() -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{CFG.ollama_url}/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{CFG.ollama_url}/api/embeddings",
            json={"model": CFG.embed_model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def pull(model: str, *, timeout: float = 1800.0) -> bool:
    """Pull a model — used by lifespan startup. Returns True on success."""
    log.info("pulling model: %s", model)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{CFG.ollama_url}/api/pull", json={"name": model, "stream": False})
            return r.status_code == 200
    except Exception as e:
        log.warning("model pull failed for %s: %s", model, e)
        return False


async def copy_model(source: str, destination: str) -> bool:
    """Create/refresh an Ollama-side alias tag pointing at ``source``.

    Used to publish a stable ``default`` tag in Ollama so direct Ollama clients
    (apps that bypass llm-app) can also request ``model="default"``. If the
    destination already exists, Ollama replaces its manifest with ``source``'s.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CFG.ollama_url}/api/copy",
                json={"source": source, "destination": destination},
            )
            ok = r.status_code == 200
            if not ok:
                log.warning("ollama copy %s→%s failed: %s %s",
                            source, destination, r.status_code, r.text[:200])
            return ok
    except Exception as e:
        log.warning("ollama copy %s→%s failed: %s", source, destination, e)
        return False


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{CFG.ollama_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

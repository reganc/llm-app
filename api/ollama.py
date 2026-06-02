"""Thin async client for the Ollama HTTP API."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import AsyncIterator

import httpx

from config import CFG, get_active_model

log = logging.getLogger("llm-api.ollama")

_THINKING_MODEL_RE = re.compile(r"qwen3|deepseek-r1|phi4.*reason|qwopus", re.IGNORECASE)

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


def is_thinking_model(model: str) -> bool:
    return bool(_THINKING_MODEL_RE.search(normalize_model(model)))


def maybe_no_think(messages: list[dict], model: str) -> list[dict]:
    """qwen3/deepseek-r1/qwopus: prepend /no_think to skip the thinking phase."""
    if not is_thinking_model(normalize_model(model)):
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i]["role"] == "user":
            content = out[i]["content"]
            if not content.startswith("/no_think"):
                out[i] = {**out[i], "content": "/no_think " + content}
            break
    return out


def build_payload(messages: list[dict], model: str, *, temperature: float = 0.7,
                  max_tokens: int = 2048, top_p: float = 0.9,
                  stop: list | None = None, stream: bool = False,
                  thinking: bool = False) -> dict:
    return {
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


async def chat(payload: dict, *, timeout: float = 180.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{CFG.ollama_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


async def generate(prompt: str, *, model: str | None = None, temperature: float = 0.7,
                   max_tokens: int = 512) -> dict:
    payload = {
        "model": normalize_model(model or CFG.default_model),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{CFG.ollama_url}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()


async def stream_chat(payload: dict, on_token=None) -> AsyncIterator[str]:
    """
    Yield SSE-formatted chunks compatible with OpenAI's streaming response.
    on_token(text) is invoked for each text delta (used to capture full reply).
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", f"{CFG.ollama_url}/api/chat", json=payload) as r:
            if r.status_code != 200:
                err = f"Ollama returned {r.status_code}"
                yield _sse(chunk_id, payload["model"], delta=err, done=True)
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
                    yield _sse(chunk_id, payload["model"], delta=f"[Error: {obj['error']}]", done=True)
                    yield "data: [DONE]\n\n"
                    return
                content = obj.get("message", {}).get("content", "")
                done = obj.get("done", False)
                if content and on_token:
                    on_token(content)
                yield _sse(chunk_id, payload["model"], delta=content, done=done)
                if done:
                    yield "data: [DONE]\n\n"
                    return


def _sse(chunk_id: str, model: str, *, delta: str, done: bool) -> str:
    body = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta} if not done else {},
            "finish_reason": "stop" if done else None,
        }],
    }
    return f"data: {json.dumps(body)}\n\n"


def completion_envelope(data: dict, model: str, extra: dict | None = None) -> dict:
    content = data.get("message", {}).get("content", "")
    out = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
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

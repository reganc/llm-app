"""
Local LLM API — OpenAI-compatible REST + RAG + web search + new SPA UI.

Backend: Ollama (GPU). Vector store: embedded ChromaDB. Search: SearXNG.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import conversations as conv_store
import finetune
import memory as mem
import ollama as oll
import search
from admin_routes import router as admin_router
from chat import metrics
from chat import router as chat_router
from config import CFG, STATIC_DIR, get_active_model
from finetune import router as finetune_router
from ingest import router as ingest_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("llm-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Local LLM API starting — backend=%s default=%s",
             CFG.ollama_url, get_active_model())

    # Wait for Ollama to be reachable, then pull required models
    asyncio.create_task(_warmup())
    yield
    log.info("shutting down")


async def _warmup() -> None:
    # Wait for Ollama
    for attempt in range(60):
        if await oll.is_available():
            break
        await asyncio.sleep(2.0)
    else:
        log.error("Ollama unreachable after 120s — model pull skipped")
        return

    # Pull default + embed model in parallel
    pulls = [oll.pull(get_active_model())]
    if CFG.memory_enabled:
        pulls.append(oll.pull(CFG.embed_model))
    await asyncio.gather(*pulls, return_exceptions=True)

    log.info("memory: %s", "ready" if mem.is_available() else "disabled")
    if CFG.search_enabled:
        log.info("search: %s",
                 "ready" if await search.is_available() else "fallback only")
    log.info("warmup complete")


app = FastAPI(
    title="Local LLM",
    description="OpenAI-compatible Ollama wrapper + RAG + web search + UI",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    body = await request.body()
    response = await call_next(request)
    if response.status_code == 422:
        log.warning("422 %s %s body=%s", request.method, request.url.path,
                    body.decode(errors="replace")[:500])
    return response


# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(ingest_router)
app.include_router(finetune_router)


# ── Health / metrics (no auth) ───────────────────────────────────────────────
@app.get("/health")
async def health():
    ok = await oll.is_available()
    models = await oll.list_models() if ok else []
    mem_stats = mem.get_stats() if CFG.memory_enabled else {"available": False}
    search_ok = False
    if CFG.search_enabled:
        try:
            search_ok = await search.is_available()
        except Exception:
            pass
    return {
        "status": "ok" if ok else "degraded",
        "ollama": "connected" if ok else "unreachable",
        "default_model": get_active_model(),
        "loaded_models": [m["name"] for m in models],
        "conversations": conv_store.count(),
        "finetune_jobs": finetune.count(),
        "memory": mem_stats,
        "search": {
            "enabled": CFG.search_enabled,
            "searxng": "reachable" if search_ok else "unreachable (DDG fallback)",
        },
        "uptime_seconds": round(time.time() - metrics["start_time"]),
    }


@app.get("/metrics")
async def get_metrics():
    uptime = time.time() - metrics["start_time"]
    return {
        **metrics,
        "uptime_seconds": round(uptime),
        "requests_per_minute": round(metrics["total_requests"] / max(uptime / 60, 1), 2),
        "active_conversations": conv_store.count(),
        "memory": mem.get_stats() if CFG.memory_enabled else {"available": False},
    }


# ── Static UI ────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "UI not built"}, status_code=500)
    return FileResponse(str(index_file), headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    f = STATIC_DIR / "favicon.svg"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({}, status_code=204)

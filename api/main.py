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
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

import conversations as conv_store
import distill
import finetune
import memory as mem
import ollama as oll
import refine
import search
from admin_routes import router as admin_router
from chat import metrics
from chat import router as chat_router
from config import CFG, STATIC_DIR, get_active_model
from distill import router as distill_router
from feedback import router as feedback_router
from finetune import router as finetune_router
from ingest import router as ingest_router
from refine import router as refine_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("llm-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Local LLM API starting — backend=%s default=%s",
             CFG.ollama_url, get_active_model())

    # Open the postgres pool + run schema migrations BEFORE warmup so the
    # first health check sees a ready memory layer. Failures are logged
    # inside init() and degrade gracefully (is_available() stays False).
    await mem.init()

    # Wait for Ollama to be reachable, then pull required models
    asyncio.create_task(_warmup())
    distill_task = asyncio.create_task(distill.scheduler_loop())
    refine_task = asyncio.create_task(refine.scheduler_loop())
    try:
        yield
    finally:
        for t in (distill_task, refine_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await mem.close()
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
    active = get_active_model()
    pulls = [oll.pull(active)]
    if CFG.memory_enabled:
        pulls.append(oll.pull(CFG.embed_model))
    await asyncio.gather(*pulls, return_exceptions=True)

    # Publish/refresh the Ollama-side "default" alias so direct Ollama clients
    # (apps that bypass llm-app) can also resolve model="default".
    await oll.copy_model(active, "default")

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


# ── Rate limiting ─────────────────────────────────────────────────────────────
# Defense-in-depth for internet exposure (Tailscale Funnel). Every /v1/* route
# already requires the bearer token; the limiter additionally caps a leaked key
# and throttles anonymous floods of the public routes (/, /health, /static).
def _client_key(request: Request) -> str:
    """Rate-limit bucket per client IP.

    Behind Tailscale Funnel the socket peer is the local tailscaled proxy, so
    prefer the X-Forwarded-For client address when present; fall back to the
    socket peer for direct tailnet connections.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_client_key,
    default_limits=[CFG.rate_limit_default] if CFG.rate_limit_enabled else [],
    enabled=CFG.rate_limit_enabled,
    headers_enabled=True,
)
app.state.limiter = limiter


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    retry_after = getattr(exc, "detail", CFG.rate_limit_default)
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded",
                 "detail": f"Rate limit exceeded: {retry_after}"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


def _public_mem_stats(stats: dict) -> dict:
    """Strip the per-source listing from memory stats for unauthenticated
    endpoints — titles + URLs of every ingested source must not leak publicly.
    The full list stays available via the authenticated /v1/memory/stats."""
    return {k: v for k, v in stats.items() if k != "sources"}


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
app.include_router(feedback_router)
app.include_router(distill_router)
app.include_router(refine_router)


# ── Health / metrics (no auth) ───────────────────────────────────────────────
@app.get("/health")
@limiter.exempt
async def health():
    ok = await oll.is_available()
    models = await oll.list_models() if ok else []
    mem_stats = await mem.get_stats() if CFG.memory_enabled else {"available": False}
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
        "memory": _public_mem_stats(mem_stats),
        "search": {
            "enabled": CFG.search_enabled,
            "searxng": "reachable" if search_ok else "unreachable (DDG fallback)",
        },
        "uptime_seconds": round(time.time() - metrics["start_time"]),
    }


@app.get("/metrics")
@limiter.exempt
async def get_metrics():
    uptime = time.time() - metrics["start_time"]
    mem_stats = await mem.get_stats() if CFG.memory_enabled else {"available": False}
    return {
        **metrics,
        "uptime_seconds": round(uptime),
        "requests_per_minute": round(metrics["total_requests"] / max(uptime / 60, 1), 2),
        "active_conversations": conv_store.count(),
        "memory": _public_mem_stats(mem_stats),
    }


# ── Static UI ────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@limiter.exempt
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "UI not built"}, status_code=500)
    return FileResponse(str(index_file), headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico", include_in_schema=False)
@limiter.exempt
async def favicon():
    f = STATIC_DIR / "favicon.svg"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({}, status_code=204)

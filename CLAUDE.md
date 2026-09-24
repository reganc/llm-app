Codex will review your output once you are done
# llm-app

Local AI stack running on an RTX 3060 (12 GB VRAM). Provides a fully offline LLM inference pipeline with RAG, web search, persistent memory, and an OpenAI-compatible API.

## Services (4 containers)

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| Ollama | `ollama` | 11434 | GPU inference engine (CUDA) |
| SearXNG | `searxng` | 8025 | Private self-hosted web search |
| Postgres+pgvector | `llm-db` | 5433 | RAG metadata + vector store |
| API + UI | `llm-api` | 8030 | FastAPI REST + RAG + brand-new SPA at `/` |

The API uses Postgres+pgvector (HNSW cosine index) for the RAG store and
pulls models on startup. Open WebUI was removed — the SPA is the only UI.

## Key Notes

- **Open in browser**: `http://localhost:8030`
- **Primary model**: `huihui_ai/qwen2.5-abliterate:14b` (~9 GB VRAM at Q4_K_M) — pulled automatically on first start
- **Model alias contract**: downstream apps should request `model: "default"` — `normalize_model()` resolves it to the active model server-side, so swaps via `/v1/settings` or `DEFAULT_MODEL` env need no client edits
- **Embed model**: `nomic-embed-text` — pulled automatically when `MEMORY_ENABLED=true`
- **API auth**: Bearer token via `API_KEY`; the SPA prompts for it on first visit and stores in localStorage
- **Persistence**: a single named volume `app_data` holds chroma, conversations, finetune, and settings under `/app/data`

## Directory Structure

```
llm-app/
├── docker-compose.yml          # 3 services
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI factory + lifespan + health/metrics
│   ├── config.py               # env vars + persisted runtime settings
│   ├── prompts.py              # system prompt presets + date preamble
│   ├── ollama.py               # Ollama HTTP client
│   ├── memory.py               # embedded ChromaDB + RAG
│   ├── search.py               # SearXNG/DDG + auto-search decision
│   ├── x_search.py             # X/Twitter via x-cli
│   ├── extract.py              # URL/PDF/DOCX/YouTube text extraction
│   ├── conversations.py        # persistent conversation store
│   ├── chat.py                 # /v1/chat/completions, /reasoning, /conversations
│   ├── ingest.py               # /v1/ingest/url, /v1/ingest/document
│   ├── finetune.py             # /v1/finetune Unsloth jobs
│   ├── admin_routes.py         # /v1/models, /v1/settings, /v1/search, /v1/memory
│   ├── auth.py                 # Bearer auth dependency
│   └── static/                 # Brand-new SPA (vanilla JS, no build step)
│       ├── index.html
│       ├── app.js
│       ├── styles.css
│       └── favicon.svg
└── searxng/
    ├── settings.yml
    └── limiter.toml
```

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `ollama_models` | `/root/.ollama` | Downloaded model weights |
| `searxng_data` | `/etc/searxng` | SearXNG runtime state |
| `db_data` | `/var/lib/postgresql/data` | Postgres data (RAG vectors + metadata) |
| `app_data` | `/app/data` | conversations/, finetune/, settings.json |

## Common Operations

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Restart just the API (picks up code/env changes)
docker compose restart api

# Manually pull a different model
docker exec ollama ollama pull <model-tag>

# Swap the active model server-side (no client edits needed — all
# downstream apps pass model:"default" and re-resolve on every call)
curl -X PATCH http://localhost:8030/v1/settings \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"default_model":"<new-model-tag>"}'

# API health check
curl http://localhost:8030/health

# Chat via API (OpenAI-compatible) — use the alias
curl http://localhost:8030/v1/chat/completions \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'

# GPU usage
nvidia-smi
```

## API Endpoints (llm-api :8030)

All require `Authorization: Bearer <API_KEY>` except `/health`, `/metrics`, `/`, `/static/*`.

- `GET  /` — Brand-new SPA (single chat surface for everything)
- `POST /v1/chat/completions` — OpenAI-compatible chat (streaming supported)
- `POST /v1/chat/reasoning` — Chain-of-thought reasoning mode (see below)
- `POST /v1/completions` — Raw text completion
- `GET  /v1/models` — List available models
- `GET/PATCH /v1/settings` — Read/update default model + system prompt
- `GET  /v1/system-prompts` — List preset system prompts

## Tests

```bash
pip install -r api/requirements-dev.txt   # pytest, on top of the runtime deps
pytest                                    # from llm-app/
```

`tests/` lives outside `api/`, so it is not part of the `./api:/app` container
mount and never ships in the image. `tests/conftest.py` points `DATA_DIR` at a
temp dir before importing app modules (`api/config.py` creates it at import
time) and puts `api/` on `sys.path`.

Coverage:

| File | What it pins |
|---|---|
| `test_ollama_envelope.py` | `completion_envelope` — reasoning fallback, `finish_reason`, null/whitespace edges |
| `test_ollama_stream.py` | `stream_chat` — `delta.reasoning`, end-of-stream flush, error surfacing |
| `test_think_disabled.py` | `think:false` on the search-router probes + `ollama.generate`; `/v1/completions` reports `finish_reason:"length"` |
| `test_query_rewrite.py` | follow-up rewrite gate, entity/number grounding guard, fallbacks; endpoints retrieve with the rewrite but answer the original |
| `test_store_opt_out.py` | `store:false` skips turn storage + search ingest on chat/reasoning (JSON, SSE, commands, two-pass) |
| `test_memory_relevance.py` | auto-recall relevance gate (`MEMORY_MIN_SCORE` + query-term overlap); library mode and title/URL matches exempt; 0 disables |
| `test_agent.py` | tool loop core — round/call bounds, forced final answer, arg validation, duplicate suppression, tool-error containment, marker renumbering, streaming |
| `test_agent_endpoint.py` | agent mode on `/v1/chat/completions` — opt-in/out, bypasses (response_format, raw, /commands, library mode), metadata, `store:false`, SSE event order |
| `test_eval_scoring.py` | `evals/scoring.py` — citation validity, routing/content checks, summary + compare |
| `test_reasoning_endpoint.py` | `/v1/chat/reasoning`, JSON **and** SSE — budget floor, `think:true`, timeout, search-off default, memory write guards, auth |

All are offline: Ollama, search and memory are stubbed, and the endpoint
tests mount `chat.router` on a bare `FastAPI()` rather than importing `main`,
avoiding its lifespan (postgres pool, warmup, schedulers) and rate-limit
middleware. No services, database or network required.

**Follow-up rewriting** (`api/rewrite.py`): on multi-turn requests whose last
message looks like a follow-up (pronoun/reference or ≤4 words), one short
`think:false` call rewrites it into a standalone query. Only retrieval, the
search router and web search use it; the model answers the original words.
A rewrite that introduces any name or number absent from the conversation
is rejected (falls back to the original). Responses carry
`retrieval_query` (SSE: `event: llm.retrieval_query`) when a rewrite was used.

**Agent mode** (`api/agent.py`, opt-in): `"agent": true` on
`/v1/chat/completions` (or `AGENT_TOOLS=true`) replaces the auto-search router
and the `[SEARCH:]` sentinel with native Ollama tool calling. Tools are
read-only: `web_search` and `library_search` (relevance-gated). At most 3
rounds / 4 tool calls; the final round offers no tools so it always answers.
Invalid, duplicate or failing calls return an error result to the model
instead of raising. Citation markers are renumbered across calls
(`web_search.markers` reports the totals). Memory is still pre-fetched, and the
deterministic freshness regex ("today", "latest", "price of"…, plus present-tense
role-holder questions like "who is the CEO of…" — `role:` signals) still runs one
web search before the first model call — otherwise the model answers price
questions from stale saved pages. Only the LLM router is dropped. Library
mode, `response_format`, `raw` and `/commands` bypass the loop. SSE adds
`event: llm.tool_call` (live) and `event: llm.agent`; JSON adds `agent`.
**Citation repair**: the final answer is checked for markers the model was
never shown (e.g. `[W8]` when only W1–W5 exist). One retry (no tools) gets the
exact error; anything still invalid is stripped in code. If web results were
shown but the answer cites none, one retry asks for citations; the rewrite is
used only if it cites a real source (else the original stands). Reported as
`agent.citation_repair`; in SSE the tokens have already gone out, so the fix
arrives as `event: llm.replace` (the SPA swaps the message text) and the
corrected text is what gets stored.
No `fetch_url` tool yet — model-chosen URLs on an internet-exposed server need
an SSRF allowlist first.

**Evals** (`evals/`, see its README): 29 golden cases scored against the live
stack — `python evals/run.py [--mode stream] [--compare <results.json>]`.
Requests send `store:false` so evals never write memory.

Note the layering: the endpoint tests stub `oll.stream_chat`, so they pin that
the SSE wrapper *forwards* reasoning deltas and closes the stream correctly.
Whether `stream_chat` *produces* those deltas is `test_ollama_stream.py`'s job,
against a fake httpx client. Break either half and a different file fails.

### `/v1/chat/reasoning` and thinking models

This is the only endpoint that runs with `think: true`. With Qwen3.5 the model
emits **reasoning tokens before any content**, which has three consequences:

- **Budget.** `num_predict` must cover reasoning *and* the answer. The endpoint
  ignores a smaller caller value and uses at least `REASONING_MAX_TOKENS`
  (default 16384) — the 4096 chat budget can be consumed entirely by reasoning,
  which previously returned an empty string.
- **Truncation is reported, not hidden.** If reasoning exhausts the budget the
  response carries `finish_reason: "length"`, `reasoning_truncated: true`, and
  `message.reasoning_fallback: true`, and the reasoning text is returned as the
  content rather than an empty reply. `message.reasoning` always carries the raw
  thinking when present. Streaming sends it as `delta.reasoning`.
- **Auto web-search defaults OFF here** (unlike `/v1/chat/completions`, where
  it follows `SEARCH_ENABLED`). Reasoning tokens get spent reconciling
  retrieved material, so an irrelevant search can consume the whole budget.
  Pass `"search": true` to opt in per request. Explicit `/search` and `/x`
  commands still run regardless. Memory/RAG recall keeps the server default —
  it is far cheaper; disable it with `"memory": false`.

Timeout is `REASONING_TIMEOUT_S` (default 600s), not the 180s used elsewhere —
a full reasoning pass can take minutes.

`_two_pass_chat` (the model-requests-a-search flow) takes `thinking` and applies
it to **every** pass. Its first pass doubles as the answer whenever the model
declines to search, so reasoning there is not optional. Other callers leave the
flag False and are unchanged.

- `POST /v1/conversations` — Create persistent conversation
- `GET  /v1/conversations` — List conversations
- `GET  /v1/conversations/{id}` — Get full history
- `POST /v1/conversations/{id}/messages` — Send message (stateful, streamable)
- `DELETE /v1/conversations/{id}` — Delete conversation
- `POST /v1/ingest/url` · `/v1/ingest/url/conversation` — Fetch URL, ingest into RAG
- `POST /v1/ingest/document` · `/v1/ingest/document/conversation` — File upload (txt/pdf/docx/md/csv/rtf)
- `POST /v1/search` — Explicit web search + optional answer
- `GET/POST/DELETE /v1/search/auxiliary-sites` — Manage sites always fetched alongside searches
- `GET  /v1/search/x/status` · `POST /v1/search/x` — X/Twitter search
- `GET  /v1/memory/stats` · `GET /v1/memory/search` · `DELETE /v1/memory/source/{id}` · `POST /v1/memory/ingest` · `POST /v1/memory/wipe`
- `POST /v1/refine/run` · `GET /v1/refine/status` — LLM-driven library refinement (per-source summary + tags + embedded gist chunk)
- `POST /v1/finetune/trigger` · `GET /v1/finetune/status[/{id}]`
- `GET  /health` · `GET /metrics`

## Environment Variables (key ones)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Inference backend |
| `DEFAULT_MODEL` | `huihui_ai/qwen2.5-abliterate:14b` | Active model; resolves the `"default"` alias requests |
| `EMBED_MODEL` | `nomic-embed-text` | Vector embeddings |
| `API_KEY` | `change-me-in-production` | Bearer auth token |
| `MEMORY_ENABLED` | `true` | Enable Postgres+pgvector RAG |
| `AGENT_TOOLS` | `false` | Default for the per-request `agent` flag (native tool-calling loop) |
| `MEMORY_MIN_SCORE` | `0.70` | Auto-recall relevance floor (cosine); chunks must also share a query term. `0` disables. Library mode is exempt |
| `DATABASE_URL` | `postgresql://llm:llm@db:5432/llmrag` | Postgres connection string |
| `EMBED_DIM` | `768` | Embedding dimensionality (match the embed model) |
| `REFINE_WINDOW_START` | `23:00` | Refine overnight-window start (HH:MM in `REFINE_TIMEZONE`). Empty either bound to fall back to `REFINE_AT` daily mode. |
| `REFINE_WINDOW_END` | `06:00` | Refine overnight-window end (wraps past midnight when end ≤ start) |
| `REFINE_BATCH_PAUSE_S` | `30` | Seconds the refine loop pauses between back-to-back batches inside the window (yields embedding capacity to live traffic) |
| `REFINE_AT` | `02:00` | Daily wall-clock time for refine when window mode is disabled (HH:MM, empty disables) |
| `REFINE_TIMEZONE` | `America/Chicago` | IANA TZ used by the refine schedule (DST-aware) |
| `REFINE_PER_RUN` | `30` | Sources processed per refine batch |
| `DISTILL_WINDOW_START` | `23:00` | Distill overnight-window start (HH:MM); same fall-back rules as refine |
| `DISTILL_WINDOW_END` | `06:00` | Distill overnight-window end |
| `DISTILL_BATCH_PAUSE_S` | `30` | Seconds the distill loop pauses between batches inside the window |
| `DISTILL_AT` | `03:00` | Daily wall-clock time for distill when window mode is disabled |
| `DISTILL_TIMEZONE` | `America/Chicago` | IANA TZ used by the distill schedule |
| `SEARCH_ENABLED` | `true` | Enable SearXNG web search |
| `SEARCH_ALWAYS` | `false` | Force search on every query (vs. auto-decision) |
| `X_SEARCH_ENABLED` | `false` | Enable X/Twitter via x-cli |
| `FIRECRAWL_URL` | `http://host.docker.internal:3002` | Optional web scraping |

## Troubleshooting

**GPU not detected in Docker:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**API needs a model pull**: the api container pulls `DEFAULT_MODEL` and `EMBED_MODEL` automatically on startup (logs go to `docker logs llm-api`). For a different model, run `docker exec ollama ollama pull <model>` and update `DEFAULT_MODEL`.

**SearXNG not returning results:** Check `searxng/settings.yml` — ensure at least one engine is enabled and the `secret_key` is set.

**API returns 503 / Ollama not ready:** Ollama has a 20s start period + 15 retries. First start downloads the model (a few minutes for a 4GB GGUF over a typical home connection).

## ROLE & MISSION
You are an elite full-stack engineer and product designer with 15+ years of experience shipping products at companies like Linear, Stripe, Vercel, and Figma.

You build what the user *meant* to ask for — not just what they typed.

Your goal: ship complete, production-grade solutions end-to-end.

No placeholders. No half-measures.

---

## CORE OPERATING PRINCIPLES

### 1. SHIP, DON'T SKETCH
- Every output must be runnable, complete, and deployable
- Deliver working applications — not scaffolding
- No stubs, no mock implementations

---

### 2. THINK BEFORE YOU TYPE
Before writing any code, explicitly state:
- What you are building
- 3 key technical decisions
- Locked assumptions

Then proceed to implementation.

---

### 3. TASTE IS NON-NEGOTIABLE
Default to world-class design standards:
- Inspired by Linear, Vercel, Arc, Raycast
- Clean typography
- Generous whitespace
- Restrained color usage
- Dark mode by default

---

### 4. MODERN STACK ONLY
Always use:
- React + TypeScript
- Tailwind CSS
- shadcn/ui + Lucide icons
- Framer Motion (when it adds value)
- Next.js (App Router)

---

### 5. DETAILS ARE THE PRODUCT
Polish is mandatory:
- Smooth, non-janky loading states
- Helpful empty states
- Responsive, intentional hover states
- Optimistic UI by default

---

## HOW TO RESPOND

- If clear: build immediately, no permission needed
- If ambiguous: ask exactly ONE sharp clarification question
- If rough: expand beyond the brief intelligently
- If questionable: flag the issue, then proceed with the best approach

---

## OUTPUT STANDARDS

- Deliver complete, production-ready files
- Handle all edge cases
- No TODOs
- No simplified or "example" versions

---

## THE VIBE

Build like it's launch day and the entire internet is watching.

The result should feel:
- Expensive
- Effortless
- Obvious in retrospect

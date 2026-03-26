# llm-app

Local AI stack running on an RTX 3060 (12 GB VRAM). Provides a fully offline LLM inference pipeline with RAG, web search, persistent memory, and an OpenAI-compatible API.

## Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| Ollama | `ollama` | 11434 | GPU inference engine (CUDA) |
| ollama-bootstrap | `ollama-bootstrap` | — | **One-shot** — pulls models then exits (restart: "no") |
| ChromaDB | `chromadb` | 8020 | Vector store for RAG memory |
| SearXNG | `searxng` | 8025 | Private self-hosted web search |
| Open WebUI | `open-webui` | 3000 | Chat interface |
| FastAPI (llm-api) | `llm-api` | 8030 | OpenAI-compatible REST API |

Network: `llm-network` (bridge)

## Key Notes

- **ollama-bootstrap exits on purpose** — it pulls `mistral:7b-instruct-q4_K_M` and `nomic-embed-text` then exits with code 0. This is normal; it is NOT a crashed container.
- **Primary model**: `mistral:7b-instruct-q4_K_M` (~6.5 GB VRAM, ~35–50 tok/s)
- **Embed model**: `nomic-embed-text` (used by llm-api for RAG chunking)
- **API auth**: Bearer token via `API_KEY` env var (default: `change-me-in-production`)
- **Open WebUI** routes chat through `llm-api:8030` (not directly to Ollama) so RAG/search/memory are active

## Directory Structure

```
llm-app/
├── docker-compose.yml      # All 6 services defined here
├── api/
│   ├── Dockerfile
│   ├── main.py             # FastAPI app — all endpoints
│   ├── memory.py           # ChromaDB RAG memory layer
│   ├── web_search.py       # SearXNG integration
│   └── requirements.txt
├── searxng/
│   ├── settings.yml        # SearXNG config (bind-mounted read-only)
│   └── limiter.toml        # Rate limiter config (bind-mounted read-only)
└── scripts/
    ├── setup.sh            # One-time setup (Docker, NVIDIA toolkit, model pull)
    └── diagnose.sh         # Troubleshooting helper
```

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `ollama_models` | `/root/.ollama` | Downloaded model weights |
| `webui_data` | `/app/backend/data` | Open WebUI state |
| `finetune_data` | `/app/finetune` | Fine-tune job data |
| `chroma_data` | `/chroma/chroma` | RAG vector embeddings |
| `searxng_data` | `/etc/searxng` | SearXNG runtime data |

**SearXNG volume note**: `searxng_data` mounts to `/etc/searxng` AND individual config files are bind-mounted into the same path. These co-exist because the named volume holds runtime state while the bind mounts overlay specific files read-only.

## Common Operations

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Restart just the API (picks up env changes)
docker compose restart api

# Re-pull models (re-run bootstrap)
docker compose run --rm ollama-bootstrap

# List loaded models
docker exec ollama ollama list

# Pull a different model
docker exec ollama ollama pull llama3:8b-instruct-q4_K_M

# API health check
curl http://localhost:8030/health

# Chat via API
curl http://localhost:8030/v1/chat/completions \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral:7b-instruct-q4_K_M","messages":[{"role":"user","content":"Hello"}]}'

# GPU usage
nvidia-smi
```

## API Endpoints (llm-api :8030)

All require `Authorization: Bearer <API_KEY>` except `/health`.

- `POST /v1/chat/completions` — OpenAI-compatible chat (streaming supported)
- `POST /v1/completions` — Raw text completion
- `GET  /v1/models` — List available models
- `GET  /health` — Health check (no auth)
- `GET  /metrics` — Usage stats
- `POST /v1/chat/reasoning` — Chain-of-thought reasoning mode
- `POST /v1/conversations` — Create named persistent conversation
- `POST /v1/conversations/{id}/messages` — Send message (stateful)
- `POST /v1/ingest/url` — Fetch URL, extract text, answer question
- `POST /v1/ingest/document` — Upload file (txt/pdf/docx/md/csv)
- `GET  /dashboard` — Browser UI

## Environment Variables (key ones)

In `docker-compose.yml` → `api` service:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Inference backend |
| `DEFAULT_MODEL` | `mistral:7b-instruct-q4_K_M` | Model for completions |
| `API_KEY` | `change-me-in-production` | Bearer auth token |
| `MEMORY_ENABLED` | `true` | Enable ChromaDB RAG |
| `SEARCH_ENABLED` | `true` | Enable SearXNG web search |
| `FIRECRAWL_URL` | `http://host.docker.internal:3002` | Optional web scraping |

## Troubleshooting

**GPU not detected in Docker:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Models missing after restart** (bootstrap didn't run or was skipped):
```bash
docker compose run --rm ollama-bootstrap
```

**SearXNG not returning results:**
Check `searxng/settings.yml` — ensure at least one engine is enabled and the `secret_key` is set.

**API returns 503 / Ollama not ready:**
Ollama has a 20s start period + 15 retries. Wait ~3 minutes after a cold start for the model to load.

# 🤖 Local LLM Stack — RTX 3060

**Qwen2.5 14B Abliterated · Ollama · Postgres+pgvector · FastAPI + SPA**

Full local AI stack optimized for the RTX 3060 (12GB VRAM).
No cloud. No API costs. Downstream apps reference one stable alias
(`model: "default"`) so model upgrades happen server-side with zero client edits.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Machine                        │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  SPA (built  │    │  Your App /  │                  │
│  │  in @ :8030) │    │  curl / SDK  │                  │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │  model:"default"  │                           │
│         └─────────┬─────────┘                          │
│                   ▼                                     │
│         ┌──────────────────┐                           │
│         │  FastAPI         │  ← resolves "default"     │
│         │  :8030           │    to active model        │
│         └────────┬─────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │     Ollama       │  ← Inference engine       │
│         │     :11434       │    CUDA-accelerated        │
│         └────────┬─────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │ Qwen2.5 14B      │  ← Q4_K_M GGUF            │
│         │ abliterated      │    ~9 GB VRAM             │
│         └──────────────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │   RTX 3060       │  ← 12 GB VRAM             │
│         │   (CUDA 12.x)    │                            │
│         └──────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

## Why Qwen3.5 9B Abliterated on the RTX 3060 (12 GB)?

Measured on this box, not estimated. The 12 GB budget makes *context headroom*,
not parameter count, the deciding factor — a 14B at Q4 already spills to CPU at
`NUM_CTX=8192`, and this gateway injects several thousand tokens of RAG/search
context into a typical request.

| Model                                | Weights | 32K ctx        | Notes |
|:-------------------------------------|:--------|:---------------|:------|
| **huihui_ai/qwen3.5-abliterated:9b** | 6.6 GB  | 100% GPU, ~57 t/s | ✅ **Active default.** 64K fits fully on GPU (7.6 GB). 256K ceiling, native tools, vision |
| qwen3.5:9b                           | 6.6 GB  | 100% GPU, ~57 t/s | Vanilla (non-abliterated) drop-in, Apache 2.0 |
| huihui_ai/qwen2.5-abliterate:14b     | 9.0 GB  | 56% GPU, ~5 t/s   | Previous default. Spills even at 8K (94% GPU, ~20 t/s) |
| qwen3.6:27b                          | 17 GB   | — | Dense; cannot fit — would spill like the 14B |

Qwen3.5 is a **thinking** model: reasoning tokens are emitted before any
content, so a small `max_tokens` returns an empty string rather than a short
answer. The server sends `think: false` on every path except `/v1/chat/reasoning`.
Requires Ollama >= 0.18 — on 0.17.x, `think: false` silently disables
`format`/JSON-schema enforcement (ollama/ollama#14645).

To swap: `PATCH /v1/settings {"default_model": "<tag>"}` — no client code changes
needed since every consumer requests `model: "default"`.

---

## Quick Start

### Prerequisites
- Ubuntu 22.04 / 24.04 (or Debian 12)
- NVIDIA RTX 3060 with latest drivers (`nvidia-driver-535+`)
- 16GB+ system RAM recommended

### 1. Run Setup (one time)
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Install Docker + Docker Compose
- Install NVIDIA Container Toolkit (GPU passthrough)
- Pull all Docker images
- Download the active model (~9 GB for Qwen2.5 14B)
- Start all services

### 2. Access

| Service   | URL                            | Purpose                     |
|:----------|:-------------------------------|:----------------------------|
| Web UI    | http://localhost:3000          | Chat interface              |
| API       | http://localhost:8030          | REST API                    |
| API Docs  | http://localhost:8030/docs     | Swagger UI                  |
| Ollama    | http://localhost:11434         | Direct model API            |

---

## API Usage

### Authentication
All API routes require: `Authorization: Bearer change-me-in-production`
Change this in `docker-compose.yml` → `API_KEY`.

### Chat Completion (standard)
```bash
curl http://localhost:8030/v1/chat/completions \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Streaming Chat
```bash
curl http://localhost:8030/v1/chat/completions \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Tell me a story"}], "stream": true}'
```

### Python (OpenAI SDK — drop-in compatible)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8030/v1",
    api_key="change-me-in-production"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### JavaScript / Node.js
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8030/v1",
  apiKey: "change-me-in-production",
});

const response = await client.chat.completions.create({
  model: "default",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);
```

---

## Switching Models

Downstream apps request `model: "default"` and never name a specific tag.
Swapping is a one-step server-side operation:

```bash
# 1. Pull the new model
docker exec ollama ollama pull <new-tag>

# 2. Activate it (persisted to /app/data/settings.json)
curl -X PATCH http://localhost:8030/v1/settings \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"default_model":"<new-tag>"}'

# Or set DEFAULT_MODEL in .env and `docker compose restart api`

# List what's loaded
docker exec ollama ollama list
```

---

## Management Commands

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# View logs (all services)
docker compose logs -f

# View logs (single service)
docker compose logs -f ollama

# Restart a service
docker compose restart api

# Check GPU usage
nvidia-smi

# Monitor VRAM in real-time
watch -n 1 nvidia-smi

# Shell into Ollama container
docker exec -it ollama bash

# Check API health
curl http://localhost:8030/health

# Check usage metrics
curl -H "Authorization: Bearer change-me-in-production" http://localhost:8030/metrics
```

---

## Performance Tuning

### Increase throughput (if you have headroom)
In `docker-compose.yml`:
```yaml
environment:
  - OLLAMA_NUM_PARALLEL=4    # Up from 2 (watch VRAM with nvidia-smi)
  - OLLAMA_FLASH_ATTENTION=1 # Enable Flash Attention (20-30% speedup)
```

### If you run out of VRAM
Switch to a smaller quant or a smaller model, then activate it via
`/v1/settings`:
```bash
docker exec ollama ollama pull huihui_ai/qwen2.5-abliterate:7b   # ~5 GB VRAM
curl -X PATCH http://localhost:8030/v1/settings \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"default_model":"huihui_ai/qwen2.5-abliterate:7b"}'
```

---

## Troubleshooting

**GPU not detected in Docker:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Model download fails:**
```bash
docker exec -it ollama ollama pull huihui_ai/qwen2.5-abliterate:14b
```

**Port already in use:**
Change ports in `docker-compose.yml` (e.g. `"3001:8080"` for WebUI).

**Out of VRAM:**
```bash
# Check what's using VRAM
nvidia-smi
# Reduce parallel requests
# OLLAMA_NUM_PARALLEL=1
```
# llm-app

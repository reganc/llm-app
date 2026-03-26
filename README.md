# 🤖 Local LLM Stack — RTX 3060

**Mistral 7B Q4_K_M · Ollama · Open WebUI · Custom FastAPI**

Full local AI stack optimized for the RTX 3060 (12GB VRAM).
No cloud. No API costs. ~35–50 tokens/sec.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Machine                        │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  Open WebUI  │    │  Your App /  │                  │
│  │  :3000       │    │  curl / SDK  │                  │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                           │
│         └─────────┬─────────┘                          │
│                   ▼                                     │
│         ┌──────────────────┐                           │
│         │  FastAPI Wrapper │  ← OpenAI-compatible API  │
│         │  :8030           │    /v1/chat/completions   │
│         └────────┬─────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │     Ollama       │  ← Inference engine       │
│         │     :11434       │    CUDA-accelerated        │
│         └────────┬─────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │  Mistral 7B      │  ← Q4_K_M quantized       │
│         │  Q4_K_M GGUF     │    6.5 GB VRAM            │
│         └──────────────────┘                           │
│                  │                                      │
│         ┌────────▼─────────┐                           │
│         │   RTX 3060       │  ← 12 GB VRAM             │
│         │   (CUDA 12.x)    │    ~35–50 tok/s           │
│         └──────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

## Why Mistral 7B Q4_K_M on the RTX 3060?

| Model             | VRAM   | Quality | Speed      | Verdict      |
|:------------------|:-------|:--------|:-----------|:-------------|
| Mistral 7B Q4_K_M | ~6.5GB | ★★★★☆   | ~40 tok/s  | ✅ **Best**  |
| Llama 3 8B Q4_K_M | ~6.6GB | ★★★★☆   | ~38 tok/s  | ✅ Great alt |
| Gemma 2 9B Q4_K_M | ~7.2GB | ★★★★☆   | ~32 tok/s  | ⚠️ Tight    |
| Mistral 7B Q8     | ~8.5GB | ★★★★★   | ~25 tok/s  | ⚠️ Slower   |
| Llama 3 70B Q2    | ~26GB  | ★★★☆☆   | ✗ OOM      | ❌ Too big  |

Q4_K_M is the sweet spot: near-full quality at 4-bit with K-quant middle weighting.

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
- Download Mistral 7B Q4_K_M (~4.1 GB)
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
    "model": "mistral:7b-instruct-q4_K_M",
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
  -d '{"model": "mistral:7b-instruct-q4_K_M", "messages": [{"role": "user", "content": "Tell me a story"}], "stream": true}'
```

### Python (OpenAI SDK — drop-in compatible)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8030/v1",
    api_key="change-me-in-production"
)

response = client.chat.completions.create(
    model="mistral:7b-instruct-q4_K_M",
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
  model: "mistral:7b-instruct-q4_K_M",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);
```

---

## Switching Models

```bash
# Pull an alternative model
docker exec ollama ollama pull llama3:8b-instruct-q4_K_M

# List available models
docker exec ollama ollama list

# Update default in docker-compose.yml:
# DEFAULT_MODEL=llama3:8b-instruct-q4_K_M
# Then: docker compose restart api open-webui
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
Switch to a smaller quantization:
```bash
docker exec ollama ollama pull mistral:7b-instruct-q3_K_M  # ~5.1 GB VRAM
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
docker exec -it ollama ollama pull mistral:7b-instruct-q4_K_M
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

# ── Stage 1: build x-cli (X/Twitter search binary) ───────────────────────────
# The upstream repo (github.com/Gladium-AI/x-cli) is gone, so the source comes
# from a local checkout passed as the `xcli` build context (see docker-compose).
FROM golang:1.26-alpine AS xcli-builder
WORKDIR /build
COPY --from=xcli go.mod go.sum main.go ./
COPY --from=xcli cmd ./cmd
COPY --from=xcli internal ./internal
RUN CGO_ENABLED=0 go build -o /out/x-cli .

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc g++ build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Chromium + all system libs Playwright needs at runtime. Crawl4AI's
# setup CLI is the supported one-liner: it runs `playwright install --with-deps
# chromium` and primes the local model/db files crawl4ai uses on first run.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright \
    CRAWL4AI_DB_PATH=/app/data/crawl4ai
RUN python -m playwright install --with-deps chromium \
 && (crawl4ai-setup || true)

COPY --from=xcli-builder /out/x-cli /usr/local/bin/x-cli

# Application code (one module per concern)
COPY *.py ./
COPY static ./static

RUN mkdir -p /app/data/conversations /app/data/finetune

EXPOSE 8030

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8030/health > /dev/null || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8030", "--workers", "1"]

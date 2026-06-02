#!/usr/bin/env bash
# Wipe the RAG database (knowledge_chunks + conversation_chunks) so the user
# can rebuild from scratch. Two modes:
#   --soft (default): TRUNCATE the tables via the API and keep the volume.
#   --hard:           Drop the db volume entirely. Forces a full Postgres
#                     re-init and pgvector re-creation. Use after schema
#                     changes or if soft wipe didn't take.
#
# Both modes leave knowledge re-ingest paths (URL/site/document/distill)
# fully functional — only the stored data goes away.
set -euo pipefail

mode="soft"
keep_conversations=0
api_key="${API_KEY:-change-me-in-production}"
api_url="${API_URL:-http://localhost:8030}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soft) mode="soft"; shift ;;
    --hard) mode="hard"; shift ;;
    --keep-conversations) keep_conversations=1; shift ;;
    --api-key) api_key="$2"; shift 2 ;;
    --url) api_url="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--soft|--hard] [--keep-conversations] [--api-key KEY] [--url URL]

  --soft                 TRUNCATE both RAG tables via the API (default)
  --hard                 docker compose down -v db, then up -d db api
  --keep-conversations   Soft mode only: keep conversation_chunks (and the
                         feedback/distill audit trail) intact
  --api-key KEY          Bearer token (default: \$API_KEY env)
  --url URL              API base URL (default: http://localhost:8030)
EOF
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

if [[ "$mode" == "soft" ]]; then
  echo "==> Soft wipe via $api_url/v1/memory/wipe"
  body="{\"confirm\":\"WIPE\""
  if [[ $keep_conversations -eq 1 ]]; then
    body+=",\"include_conversations\":false"
  fi
  body+="}"
  curl -fsS -X POST \
    -H "Authorization: Bearer $api_key" \
    -H "Content-Type: application/json" \
    -d "$body" \
    "$api_url/v1/memory/wipe"
  echo
  echo "==> Memory stats:"
  curl -fsS -H "Authorization: Bearer $api_key" \
    "$api_url/v1/memory/stats"
  echo
  exit 0
fi

# --hard
echo "==> Hard wipe: dropping db volume + recreating Postgres"
docker compose stop api
docker compose rm -fsv db
docker volume rm llm-app_db_data 2>/dev/null || true
docker compose up -d db
echo "==> Waiting for db healthcheck..."
for i in {1..30}; do
  status=$(docker inspect -f '{{.State.Health.Status}}' llm-db 2>/dev/null || echo "unknown")
  if [[ "$status" == "healthy" ]]; then break; fi
  sleep 2
done
docker compose up -d api
echo "==> Done. The api container will run schema migrations on startup."
echo "==> Tail logs with: docker logs -f llm-api"

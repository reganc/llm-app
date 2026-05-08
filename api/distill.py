"""Knowledge distillation: turn up-rated conversation turns into permanent
knowledge entries that participate in future RAG retrieval."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

import memory as mem
import ollama as oll
from auth import verify_api_key
from config import CFG, DATA_DIR, get_active_model

log = logging.getLogger("llm-api.distill")

router = APIRouter(prefix="/v1", tags=["distill"], dependencies=[Depends(verify_api_key)])

STATE_FILE: Path = DATA_DIR / "distill_state.json"

# Default to mistral 7b for speed; falls back to active model if mistral isn't pulled
DISTILL_MODEL = os.getenv("DISTILL_MODEL", "mistral:7b-instruct-q4_K_M")
DISTILL_INTERVAL_HOURS = float(os.getenv("DISTILL_INTERVAL_HOURS", "24"))
DISTILL_PER_RUN = int(os.getenv("DISTILL_PER_RUN", "20"))
DISTILL_MIN_ANSWER_CHARS = 120  # skip trivially short answers

_QA_RE = re.compile(r"User:\s*(?P<u>.*?)\nAssistant:\s*(?P<a>.*)", re.DOTALL)
_TITLE_RE = re.compile(r"^TITLE:\s*(.+?)$", re.IGNORECASE | re.MULTILINE)

_DISTILL_PROMPT = (
    "You are a knowledge distiller. The user marked the following Q&A as helpful. "
    "Extract the durable, reusable knowledge into a compact entry that future questions can retrieve. "
    "Drop conversational fluff, hedges, and meta-commentary. Keep concrete facts, definitions, and procedures.\n\n"
    "Output EXACTLY in this format. No preamble, no commentary outside the format:\n\n"
    "TITLE: <a short topical title, max 80 chars>\n"
    "KEY POINTS:\n"
    "- <fact or insight in one sentence>\n"
    "- <fact or insight in one sentence>\n"
    "(2 to 6 bullets total)\n\n"
    "USER QUESTION:\n{user_msg}\n\n"
    "ASSISTANT ANSWER:\n{assistant_msg}\n"
)


# ── State ────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_run_at": None, "last_run_distilled": 0,
                "total_distilled": 0, "last_error": None}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Helpers ──────────────────────────────────────────────────────────────────
def _split_qa(document: str) -> tuple[str, str] | None:
    m = _QA_RE.search(document)
    if not m:
        return None
    return m.group("u").strip(), m.group("a").strip()


def _parse_title(text: str) -> str:
    m = _TITLE_RE.search(text)
    if not m:
        return "Distilled knowledge"
    return m.group(1).strip()[:120] or "Distilled knowledge"


async def _distill_one(user_msg: str, assistant_msg: str, model: str) -> str | None:
    prompt = _DISTILL_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
    payload = oll.build_payload(
        [{"role": "user", "content": prompt}],
        model,
        temperature=0.2, max_tokens=600, top_p=0.9, stream=False,
    )
    try:
        data = await oll.chat(payload, timeout=120.0)
    except Exception as e:
        log.warning("distill chat failed: %s", e)
        return None
    text = (data.get("message", {}) or {}).get("content", "").strip()
    return text or None


# ── Public API ───────────────────────────────────────────────────────────────
async def run(*, model: str | None = None, limit: int | None = None) -> dict:
    if not CFG.memory_enabled:
        return {"ran": False, "reason": "memory_disabled", "distilled": 0}

    use_model = model or DISTILL_MODEL or get_active_model()
    use_limit = limit if limit is not None else DISTILL_PER_RUN

    pending = mem.pending_distill_chunks(limit=use_limit)
    if not pending:
        state = _load_state()
        state["last_run_at"] = datetime.now(timezone.utc).isoformat()
        state["last_run_distilled"] = 0
        state["last_error"] = None
        _save_state(state)
        return {"ran": True, "model": use_model, "distilled": 0,
                "skipped": 0, "pending_remaining": 0}

    distilled_count = 0
    skipped = 0
    chunks_to_mark: list[str] = []
    last_error: str | None = None

    for item in pending:
        qa = _split_qa(item["document"])
        if not qa:
            skipped += 1
            continue
        user_msg, assistant_msg = qa
        if len(assistant_msg) < DISTILL_MIN_ANSWER_CHARS:
            skipped += 1
            chunks_to_mark.append(item["chunk_id"])  # mark to avoid re-processing
            continue

        body = await _distill_one(user_msg, assistant_msg, use_model)
        if not body:
            last_error = "llm returned empty"
            continue

        title = _parse_title(body)
        identifier = f"distilled:{item['message_id'] or item['chunk_id']}"
        try:
            await mem.store_knowledge(
                text=body,
                title=title,
                source_type="distilled",
                identifier=identifier,
                extra={
                    "origin_message_id": item.get("message_id") or "",
                    "origin_chunk_id": item["chunk_id"],
                    "origin_timestamp": item.get("timestamp") or "",
                    "distilled_at": datetime.now(timezone.utc).isoformat(),
                    "model": use_model,
                },
            )
            distilled_count += 1
            chunks_to_mark.append(item["chunk_id"])
        except Exception as e:
            log.warning("store_knowledge failed for %s: %s", item["chunk_id"], e)
            last_error = str(e)

    if chunks_to_mark:
        mem.mark_distilled(chunks_to_mark)

    state = _load_state()
    state["last_run_at"] = datetime.now(timezone.utc).isoformat()
    state["last_run_distilled"] = distilled_count
    state["total_distilled"] = state.get("total_distilled", 0) + distilled_count
    state["last_error"] = last_error
    _save_state(state)

    return {
        "ran": True,
        "model": use_model,
        "distilled": distilled_count,
        "skipped": skipped,
        "pending_remaining": mem.pending_distill_count(),
        "last_error": last_error,
    }


def status() -> dict:
    state = _load_state()
    return {
        **state,
        "pending": mem.pending_distill_count(),
        "interval_hours": DISTILL_INTERVAL_HOURS,
        "model": DISTILL_MODEL,
        "per_run": DISTILL_PER_RUN,
    }


# ── Background scheduler ─────────────────────────────────────────────────────
async def scheduler_loop() -> None:
    if DISTILL_INTERVAL_HOURS <= 0:
        log.info("distill scheduler disabled (DISTILL_INTERVAL_HOURS=0)")
        return
    interval = DISTILL_INTERVAL_HOURS * 3600
    log.info("distill scheduler running every %.1fh", DISTILL_INTERVAL_HOURS)
    # Wait one interval before the first auto-run so startup isn't blocked
    while True:
        try:
            await asyncio.sleep(interval)
            log.info("distill scheduler: starting auto-run")
            result = await run()
            log.info("distill scheduler: %s", result)
        except asyncio.CancelledError:
            log.info("distill scheduler stopping")
            raise
        except Exception as e:
            log.exception("distill scheduler iteration failed: %s", e)


# ── HTTP routes ──────────────────────────────────────────────────────────────
class DistillRunRequest(BaseModel):
    model: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


@router.post("/distill/run")
async def distill_run(req: DistillRunRequest = DistillRunRequest()):
    return await run(model=req.model, limit=req.limit)


@router.get("/distill/status")
async def distill_status():
    return status()

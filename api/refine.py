"""Library refinement passes — make stored sources more searchable.

Two passes today, both LLM-driven and idempotent:

* ``summary`` — produce ``{title, tl_dr, key_claims, date_mentioned}`` for
  each source, persist it as ``metadata.summary`` on every chunk of that
  source, and add a synthetic ``chunk_index = -1`` "gist" row whose embedded
  text is the TL;DR + key claims joined together. The gist participates in
  retrieval so the model can find a source by its summary even when no
  individual chunk vectorizes well against the user's phrasing.

* ``tags`` — extract ``{people, orgs, places, events, topics}`` per source
  and persist as ``metadata.tags``. Surfaces in the retrieval prompt block
  so the model can reason about whether a source actually covers a topic
  rather than guessing from incidental keywords.

State is stored entirely in ``knowledge_chunks.metadata``:

* ``refined_summary_at`` / ``refined_tags_at`` — ISO timestamps marking the
  per-source pass as complete; ``iter_unrefined_sources`` looks for missing
  keys to find work.
* ``summary`` / ``tags`` — the JSON output blobs.

A trigger route (``POST /v1/refine/run``) and a once-a-day scheduler keep
the corpus current. Both run in the same async loop so they share the
embedding semaphore in ``memory.py`` — no risk of starving live traffic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

import memory as mem
import ollama as oll
import scheduler as sched
from auth import verify_api_key
from config import CFG, get_active_model

log = logging.getLogger("llm-api.refine")

router = APIRouter(prefix="/v1", tags=["refine"], dependencies=[Depends(verify_api_key)])


REFINE_AT = os.getenv("REFINE_AT", "02:00")  # HH:MM, empty disables daily mode
REFINE_TIMEZONE = os.getenv("REFINE_TIMEZONE", "America/Chicago")
REFINE_PER_RUN = int(os.getenv("REFINE_PER_RUN", "30"))
REFINE_MAX_CHARS = int(os.getenv("REFINE_MAX_CHARS", "9000"))
# After this many consecutive failures, a source is permanently marked
# refined_<pass>_at with refined_<pass>_failed=true so iter_unrefined_sources
# stops returning it. Prevents perpetually retrying sources the LLM can't parse.
REFINE_MAX_ATTEMPTS = int(os.getenv("REFINE_MAX_ATTEMPTS", "3"))

# Overnight window mode: when both bounds are set, scheduler runs back-to-back
# batches inside [start, end] each night until the backlog drains, then idles
# until the next window. Defaults to 23:00→06:00 Central, the user's "run all
# night until done" preference. Empty either bound to fall back to daily mode.
REFINE_WINDOW_START = os.getenv("REFINE_WINDOW_START", "23:00")
REFINE_WINDOW_END = os.getenv("REFINE_WINDOW_END", "06:00")
REFINE_BATCH_PAUSE_S = float(os.getenv("REFINE_BATCH_PAUSE_S", "30"))


# ── Prompts ─────────────────────────────────────────────────────────────────
_SUMMARY_PROMPT = (
    "You are a librarian. Produce a compact, factual summary of the document below "
    "for retrieval indexing. Output STRICT JSON with this exact schema, no preamble, "
    "no markdown fence:\n"
    '{{"title": "string up to 80 chars",\n'
    '  "tl_dr": "1-2 sentence neutral summary, prefer concrete subjects/names over generic phrasing",\n'
    '  "key_claims": ["concrete claim or fact stated by the document", "..."],\n'
    '  "date_mentioned": "YYYY-MM-DD or null"}}\n\n'
    "Rules:\n"
    "- 3 to 7 key_claims, each under 200 chars, stated as the document states them.\n"
    "- Skip site-wide boilerplate (sticky disclaimers, footers, navigation).\n"
    "- date_mentioned is the most prominent date IN the document (not today's date), or null if none.\n"
    "- Output ONLY the JSON object.\n\n"
    "DOCUMENT (title: {title}):\n"
    "{text}\n"
)

_TAGS_PROMPT = (
    "Extract entities and topics from the document below for retrieval indexing. "
    "Output STRICT JSON with this exact schema, no preamble, no markdown fence:\n"
    '{{"people": ["string"],\n'
    '  "orgs": ["string"],\n'
    '  "places": ["string"],\n'
    '  "events": ["string"],\n'
    '  "topics": ["string"]}}\n\n'
    "Rules:\n"
    "- Use proper nouns where possible. Lowercase only the topics list.\n"
    "- 0 to 12 items per category. Drop categories with no real entries (use []).\n"
    "- 'topics' = general subject areas, e.g. 'geopolitics', 'freemasonry', 'crypto regulation'.\n"
    "- Skip generic site-wide boilerplate.\n"
    "- Output ONLY the JSON object.\n\n"
    "DOCUMENT (title: {title}):\n"
    "{text}\n"
)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(reply: str) -> dict | None:
    """Best-effort JSON parse for LLM output. Strips markdown fences and
    matches the first {...} span if the model added prose before/after."""
    if not reply:
        return None
    cleaned = reply.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = _JSON_OBJECT_RE.search(cleaned)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# ── State (filesystem) ──────────────────────────────────────────────────────
from pathlib import Path  # noqa: E402

from config import DATA_DIR  # noqa: E402

STATE_FILE: Path = DATA_DIR / "refine_state.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Helpers ─────────────────────────────────────────────────────────────────
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit * 0.7:
        cut = cut[:period + 1]
    return cut + "\n[…]"


async def _llm_json(prompt: str, *, model: str, max_tokens: int = 700) -> dict | None:
    payload = oll.build_payload(
        [{"role": "user", "content": prompt}],
        model,
        temperature=0.1, max_tokens=max_tokens, top_p=0.9, stream=False,
    )
    try:
        data = await oll.chat(payload, timeout=120.0)
    except Exception as e:
        log.warning("refine LLM call failed: %s", e)
        return None
    text = (data.get("message", {}) or {}).get("content", "").strip()
    return _parse_json(text)


def _norm_str_list(xs, *, lower: bool = False, cap: int = 12) -> list[str]:
    if not isinstance(xs, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        if x is None:
            continue
        s = str(x).strip()
        if lower:
            s = s.lower()
        if not s or len(s) > 200:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _normalize_summary(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    title = (raw.get("title") or "").strip()[:80]
    tl_dr = (raw.get("tl_dr") or "").strip()
    if not tl_dr:
        return None
    claims = _norm_str_list(raw.get("key_claims"), cap=7)
    date = raw.get("date_mentioned")
    if isinstance(date, str):
        date = date.strip() or None
        if date and not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            date = None
    else:
        date = None
    return {
        "title": title or None,
        "tl_dr": tl_dr[:600],
        "key_claims": claims,
        "date_mentioned": date,
    }


def _normalize_tags(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = {
        "people": _norm_str_list(raw.get("people")),
        "orgs":   _norm_str_list(raw.get("orgs")),
        "places": _norm_str_list(raw.get("places")),
        "events": _norm_str_list(raw.get("events")),
        "topics": _norm_str_list(raw.get("topics"), lower=True),
    }
    if not any(out.values()):
        return None
    return out


def _gist_text(summary: dict) -> str:
    lines = []
    title = summary.get("title")
    if title:
        lines.append(f"TITLE: {title}")
    tl_dr = summary.get("tl_dr")
    if tl_dr:
        lines.append(f"SUMMARY: {tl_dr}")
    claims = summary.get("key_claims") or []
    if claims:
        lines.append("KEY CLAIMS:")
        lines.extend(f"- {c}" for c in claims)
    return "\n".join(lines)


# ── Passes ──────────────────────────────────────────────────────────────────
async def _refine_one_summary(source: dict, model: str) -> tuple[bool, str | None]:
    """Run the summary pass for one source. Returns (success, error)."""
    full = await mem.get_source(source["source_id"])
    if not full.get("found"):
        return False, "source vanished"
    text = _truncate(full.get("text", ""), REFINE_MAX_CHARS)
    if not text.strip():
        return False, "empty source"
    prompt = _SUMMARY_PROMPT.format(title=source.get("title", ""), text=text)
    raw = await _llm_json(prompt, model=model)
    summary = _normalize_summary(raw)
    if not summary:
        return False, "summary parse failed"
    now = datetime.now(timezone.utc).isoformat()
    updated = await mem.update_source_metadata(source["source_id"], {
        "summary": summary,
        "refined_summary_at": now,
        "refined_summary_model": model,
    })
    if not updated:
        return False, "metadata update failed"
    # Gist chunk — embed the summary so it joins retrieval directly.
    gist = _gist_text(summary)
    if gist.strip():
        await mem.upsert_gist_chunk(source["source_id"], gist)
    return True, None


async def _refine_one_tags(source: dict, model: str) -> tuple[bool, str | None]:
    full = await mem.get_source(source["source_id"])
    if not full.get("found"):
        return False, "source vanished"
    text = _truncate(full.get("text", ""), REFINE_MAX_CHARS)
    if not text.strip():
        return False, "empty source"
    prompt = _TAGS_PROMPT.format(title=source.get("title", ""), text=text)
    raw = await _llm_json(prompt, model=model, max_tokens=600)
    tags = _normalize_tags(raw)
    now = datetime.now(timezone.utc).isoformat()
    patch: dict = {"refined_tags_at": now, "refined_tags_model": model}
    if tags:
        patch["tags"] = tags
    updated = await mem.update_source_metadata(source["source_id"], patch)
    return (updated > 0), (None if tags else "no tags extracted")


async def _record_failure(source_id: str, pass_name: str, err: str) -> None:
    """Bump per-source attempts counter; at REFINE_MAX_ATTEMPTS, stamp the
    refined_<pass>_at marker so iter_unrefined_sources permanently excludes
    this source. Best-effort — never raises."""
    try:
        meta = await mem.get_source_metadata(source_id)
        attempts_key = f"refined_{pass_name}_attempts"
        attempts = int(meta.get(attempts_key) or 0) + 1
        patch: dict = {
            attempts_key: attempts,
            f"refined_{pass_name}_last_error": err[:200],
        }
        if attempts >= REFINE_MAX_ATTEMPTS:
            patch[f"refined_{pass_name}_at"] = datetime.now(timezone.utc).isoformat()
            patch[f"refined_{pass_name}_failed"] = True
        await mem.update_source_metadata(source_id, patch)
    except Exception as e:
        log.warning("_record_failure(%s,%s) failed: %s", source_id, pass_name, e)


async def _run_pass(pass_name: str, fn, *, limit: int, model: str) -> dict:
    sources = await mem.iter_unrefined_sources(pass_name, limit)
    refined = 0
    skipped = 0
    errors: list[str] = []
    for src in sources:
        ok, err = await fn(src, model)
        if ok:
            refined += 1
        else:
            skipped += 1
            if err:
                errors.append(f"{src.get('identifier','?')}: {err}")
            await _record_failure(src["source_id"], pass_name, err or "unknown")
    return {
        "pass": pass_name,
        "candidates": len(sources),
        "refined": refined,
        "skipped": skipped,
        "errors": errors[:10],
    }


# ── Public API ──────────────────────────────────────────────────────────────
async def run(*, model: str | None = None, limit: int | None = None,
              passes: list[str] | None = None) -> dict:
    if not CFG.memory_enabled:
        return {"ran": False, "reason": "memory_disabled"}
    use_model = model or get_active_model()
    use_limit = limit if limit is not None else REFINE_PER_RUN
    passes = passes or ["summary", "tags"]

    started = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    if "summary" in passes:
        results.append(await _run_pass("summary", _refine_one_summary,
                                       limit=use_limit, model=use_model))
    if "tags" in passes:
        results.append(await _run_pass("tags", _refine_one_tags,
                                       limit=use_limit, model=use_model))

    state = _load_state()
    state["last_run_at"] = started
    state["last_results"] = results
    state["last_model"] = use_model
    _save_state(state)
    return {"ran": True, "model": use_model, "started": started,
            "results": results}


async def status() -> dict:
    state = _load_state()
    if not CFG.memory_enabled or not mem.is_available():
        return {**state, "available": False}
    counts: dict[str, int] = {}
    try:
        async with mem._pool.connection() as conn:  # type: ignore[attr-defined]
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                      COUNT(*) FILTER (WHERE chunk_index = 0)                            AS sources,
                      COUNT(*) FILTER (WHERE chunk_index = 0
                                        AND metadata ? 'refined_summary_at')             AS summary_done,
                      COUNT(*) FILTER (WHERE chunk_index = 0
                                        AND metadata ? 'refined_tags_at')                AS tags_done,
                      COUNT(*) FILTER (WHERE chunk_index = -1)                           AS gist_chunks
                    FROM knowledge_chunks
                    """
                )
                row = await cur.fetchone()
        if row:
            counts = {"sources": int(row[0]), "summary_done": int(row[1]),
                      "tags_done": int(row[2]), "gist_chunks": int(row[3])}
    except Exception as e:
        return {**state, "available": False, "error": str(e)}
    now_utc = datetime.now(timezone.utc)
    window = sched.window_bounds(now_utc, REFINE_WINDOW_START,
                                 REFINE_WINDOW_END, REFINE_TIMEZONE)
    if window is not None:
        ws, we = window
        schedule = {
            "mode": "window",
            "window_start": REFINE_WINDOW_START,
            "window_end": REFINE_WINDOW_END,
            "timezone": REFINE_TIMEZONE,
            "batch_pause_s": REFINE_BATCH_PAUSE_S,
            "next_window_start_utc": ws.isoformat(timespec="minutes"),
            "next_window_end_utc": we.isoformat(timespec="minutes"),
            "in_window": ws <= now_utc < we,
        }
    else:
        next_run = sched.next_at(now_utc, REFINE_AT, REFINE_TIMEZONE)
        schedule = {
            "mode": "daily",
            "at": REFINE_AT or None,
            "timezone": REFINE_TIMEZONE,
            "next_run_utc": next_run.isoformat(timespec="minutes") if next_run else None,
        }
    return {
        **state,
        "available": True,
        "schedule": schedule,
        "per_run": REFINE_PER_RUN,
        "counts": counts,
        "pending": {
            "summary": max(counts.get("sources", 0) - counts.get("summary_done", 0), 0),
            "tags":    max(counts.get("sources", 0) - counts.get("tags_done", 0), 0),
        },
    }


# ── Background scheduler ────────────────────────────────────────────────────
def _is_done(result: dict) -> bool:
    """Window-mode "backlog drained" predicate.

    True when the run did nothing useful — either memory was disabled, or
    every pass found zero candidates. Tags can intentionally produce no
    extracted tags but still count as work, so we look at ``candidates``
    (i.e. sources that were actually fetched and processed) rather than
    ``refined``.
    """
    if not result.get("ran"):
        return True
    results = result.get("results") or []
    if not results:
        return True
    return all((r or {}).get("candidates", 0) == 0 for r in results)


async def scheduler_loop() -> None:
    if sched.parse_hhmm(REFINE_WINDOW_START) and sched.parse_hhmm(REFINE_WINDOW_END):
        await sched.window_loop(
            name="refine",
            start_hhmm=REFINE_WINDOW_START,
            end_hhmm=REFINE_WINDOW_END,
            tz_name=REFINE_TIMEZONE,
            run=run,
            is_done=_is_done,
            batch_pause_s=REFINE_BATCH_PAUSE_S,
        )
        return
    await sched.daily_loop(
        name="refine",
        hhmm=REFINE_AT,
        tz_name=REFINE_TIMEZONE,
        run=run,
    )


# ── HTTP routes ─────────────────────────────────────────────────────────────
class RefineRunRequest(BaseModel):
    model: str | None = None
    limit: int | None = Field(default=None, ge=1, le=200)
    passes: list[str] | None = Field(default=None,
                                     description="Subset of ['summary','tags']")


@router.post("/refine/run")
async def refine_run(req: RefineRunRequest = RefineRunRequest()):
    return await run(model=req.model, limit=req.limit, passes=req.passes)


@router.get("/refine/status")
async def refine_status():
    return await status()

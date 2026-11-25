"""The morning briefing: today's calendar, new page-watch alerts, routine results and the
homelab digest, in one Telegram message.

No model is involved: every line is quoted from its source. Routine results follow
the same rule as their notifications, so a result that may quote the library is
only counted, never quoted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StringConstraints, field_validator

import calendar_feed
import notify
import routine_runner
import routine_store
import watch_store
from auth import verify_api_key
from config import CFG, DATA_DIR, get_setting, set_setting
from routine_schedule import Schedule, next_run, valid_timezone

log = logging.getLogger("llm-api.briefing")

TICK_S = 30.0
GRACE = timedelta(hours=1)        # found later than this (server off): skipped, not sent late
_WAIT_FOR_ROUTINES_S = 600        # routines due at the same time go in first
_FIRST_LOOK_BACK = timedelta(hours=24)
_DIGEST_MAX_AGE = timedelta(hours=26)
_MAX_ALERTS, _MAX_RESULTS, _MAX_PROJECTS = 5, 5, 6
_ALERT_LINES = 2
_SETTING = "briefing"
_STATE: Path = DATA_DIR / "briefing.json"
DIGEST: Path = Path(os.getenv("HOMELAB_DIGEST", "/triage/latest.md"))
DEFAULTS = {"enabled": False, "time": "07:00", "days": [0, 1, 2, 3, 4, 5, 6], "timezone": None}


def config() -> dict:
    stored = get_setting(_SETTING) or {}
    merged = {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}
    merged["timezone"] = merged["timezone"] or CFG.routine_timezone
    return merged


def _load() -> dict:
    try:
        return json.loads(_STATE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save(state: dict) -> None:
    tmp = _STATE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(state, f)
    os.replace(tmp, _STATE)


def upcoming(cfg: dict, now: datetime) -> str | None:
    if not cfg["enabled"]:
        return None
    when = next_run(Schedule(kind="daily", days=cfg["days"], time=cfg["time"]), now, cfg["timezone"])
    return when.isoformat(timespec="seconds")


# ── sections ─────────────────────────────────────────────────────────────────

def _clock(d: datetime) -> str:
    return f"{d.hour % 12 or 12}:{d.minute:02d}{'am' if d.hour < 12 else 'pm'}"


async def calendar_section(day: datetime) -> str | None:
    url = CFG.google_calendar_ics_url
    if not url:
        return None
    try:
        events = calendar_feed.events_on(await calendar_feed.fetch(url), day.date(), day.tzinfo)  # type: ignore[arg-type]
    except calendar_feed.CalendarError as e:
        return f"Calendar: couldn't be read. {e}"
    except Exception as e:  # its text could hold the address
        log.warning("calendar failed: %s", type(e).__name__)
        return "Calendar: couldn't be read."
    if not events:
        return "Calendar: nothing today."
    rows = [f"• {'All day' if e.all_day else _clock(e.start)} {e.title}" for e in events]
    return "Calendar:\n" + "\n".join(rows)


def watches_section(since: datetime) -> str | None:
    rows = []
    for w in watch_store.list_all():
        for a in w["alerts"]:
            if datetime.fromisoformat(a["at"]) > since:
                lines = a["text"].splitlines()
                text = "\n  ".join(lines[:_ALERT_LINES]) + ("\n  …" if len(lines) > _ALERT_LINES else "")
                rows.append((a["at"], f"• {w['name']}: {text}"))
    if not rows:
        return None
    rows.sort(reverse=True)
    more = len(rows) - _MAX_ALERTS
    return "Watches:\n" + "\n".join(r for _, r in rows[:_MAX_ALERTS]) + (f"\n• and {more} more" if more > 0 else "")


def _result(routine: dict, run: dict) -> str:
    if run["status"] == "error":
        return f"• {routine['name']}: the run failed"
    if run["status"] == "unsourced":
        return f"• {routine['name']}: a result that needs checking (too little matched its sources)"
    headline = notify.message({**routine, "notify": "headline"}, run).split("\n")[2:3]
    if not headline:  # it may quote the library, or nothing to quote
        return f"• {routine['name']}: a new result"
    return f"• {routine['name']}: {headline[0]}"


def routines_section(since: datetime) -> str | None:
    rows = []
    for r in routine_store.list_all():
        latest = next((x for x in r["runs"] if x["status"] in ("ok", "unsourced", "error")
                       and datetime.fromisoformat(x["started_at"]) > since), None)
        if latest:
            rows.append((latest["started_at"], _result(r, latest)))
    if not rows:
        return None
    rows.sort(reverse=True)
    return "Routines:\n" + "\n".join(r for _, r in rows[:_MAX_RESULTS]) + "\nSend /more for the latest in full."


_HEADING = re.compile(r"^## (?:🔴|🟠) (.+?) — (critical|high)\s*$", re.MULTILINE)
_COUNTS = re.compile(r"\*\*((?:\d+ (?:critical|high|medium|low)(?:, )?)+)\.?\*\*")
_FAILED = re.compile(r"^.*\b(?:failed|could not be read)\b.*$", re.IGNORECASE | re.MULTILINE)


def homelab_section(now: datetime) -> str | None:
    try:
        stat = DIGEST.stat()
        text = DIGEST.read_text(errors="replace")[:200_000]
    except OSError:
        return None
    written = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    if now - written > _DIGEST_MAX_AGE:
        return f"Homelab: the digest is out of date (last written {written.astimezone(now.tzinfo):%b %d}). Check the 6:15 job."
    counts = _COUNTS.search(text[:2000])
    names = [n for n, _ in _HEADING.findall(text)]
    if not counts and not names:
        return "Homelab: nothing to report." if "nothing" in text[:2000].lower() else "Homelab: see the digest."
    line = counts.group(1) if counts else f"{len(names)} serious"
    shown = ", ".join(names[:_MAX_PROJECTS]) + (f" and {len(names) - _MAX_PROJECTS} more" if len(names) > _MAX_PROJECTS else "")
    return f"Homelab: {line}." + (f"\nNeeds a look: {shown}" if shown else "")


async def compose(now: datetime, since: datetime) -> str:
    cfg = config()
    local = now.astimezone(ZoneInfo(cfg["timezone"]))
    parts = [f"Good morning. {local:%A, %B} {local.day}"]
    parts += [s for s in (await calendar_section(local), watches_section(since),
                          routines_section(since), homelab_section(now)) if s]
    if len(parts) == 1:
        parts.append("Nothing new since the last briefing.")
    return "\n\n".join(parts)


# ── sending and schedule ─────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _routines_settled(due: datetime) -> None:
    """Let routines scheduled for the same minute run first, so their results are in."""
    async def busy() -> bool:
        if routine_runner._active:
            return True
        return any(r["enabled"] and r["next_run_at"]
                   and datetime.fromisoformat(r["next_run_at"]) <= due + timedelta(minutes=1)
                   for r in routine_store.list_all())
    waited = 0.0
    await asyncio.sleep(1)
    while await busy() and waited < _WAIT_FOR_ROUTINES_S:
        await asyncio.sleep(5)
        waited += 5


async def send_now(trigger: str) -> dict:
    """Compose, store and send. Returns what was stored."""
    state = _load()
    now = _now()
    since = datetime.fromisoformat(state["last_at"]) if state.get("last_at") else now - _FIRST_LOOK_BACK
    text = await compose(now, since)
    try:
        await notify.send(text)
        notified = "sent"
    except notify.NotifyError as e:
        notified = f"not sent: {e}"
    except Exception as e:  # its text could hold the token
        notified = f"not sent: {type(e).__name__}"
    last = {"at": now.isoformat(timespec="seconds"), "trigger": trigger, "text": text, "notified": notified}
    _save({**_load(), "last_at": last["at"], "last": last})
    log.info("briefing (%s): %s", trigger, "sent" if notified == "sent" else "not sent")
    return last


async def tick(now: datetime) -> None:
    cfg, state = config(), _load()
    due = state.get("next_at")
    expected = upcoming(cfg, now)
    if not cfg["enabled"] or not due:
        if state.get("next_at") != expected:
            _save({**state, "next_at": expected})
        return
    due_at = datetime.fromisoformat(due)
    if due_at > now:
        return
    _save({**state, "next_at": upcoming(cfg, now + timedelta(minutes=1))})  # first: a crash must not repeat it
    if now - due_at > GRACE:
        log.warning("briefing: missed the one due at %s", due)
        return
    await _routines_settled(due_at)
    await send_now("schedule")


async def scheduler_loop() -> None:
    log.info("briefing: scheduler on (%s)", "enabled" if config()["enabled"] else "off until enabled")
    while True:
        try:
            await tick(_now())
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("briefing: tick failed")
        await asyncio.sleep(TICK_S)


# ── API ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/briefing", tags=["briefing"], dependencies=[Depends(verify_api_key)])


class BriefingUpdate(BaseModel):
    enabled: bool | None = None
    time: Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")] | None = None
    days: list[Annotated[int, Field(ge=0, le=6)]] | None = Field(default=None, min_length=1, max_length=7)
    timezone: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("timezone")
    @classmethod
    def _tz(cls, value: str | None) -> str | None:
        if value is not None and not valid_timezone(value):
            raise ValueError("unknown timezone")
        return value

    @field_validator("days")
    @classmethod
    def _days(cls, value: list[int] | None) -> list[int] | None:
        return sorted(set(value)) if value is not None else None


def _status() -> dict:
    state = _load()
    return {**config(), "next_at": state.get("next_at"), "last": state.get("last"),
            "calendar_set": bool(CFG.google_calendar_ics_url),
            "homelab_digest": DIGEST.exists(),
            "telegram_ready": bool(CFG.telegram_bot_token and get_setting(notify.CHAT_SETTING) is not None)}


@router.get("")
async def get_briefing():
    return _status()


@router.patch("")
async def update_briefing(req: BriefingUpdate):
    fields = {k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None}
    stored = {**(get_setting(_SETTING) or {}), **fields}
    set_setting(_SETTING, stored)
    _save({**_load(), "next_at": upcoming(config(), _now())})
    return _status()


@router.post("/preview")
async def preview():
    """What would be sent now. Nothing is sent or recorded."""
    state = _load()
    since = datetime.fromisoformat(state["last_at"]) if state.get("last_at") else _now() - _FIRST_LOOK_BACK
    return {"text": await compose(_now(), since)}


@router.post("/send")
async def send():
    return await send_now("manual")

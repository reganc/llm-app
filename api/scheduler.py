"""Shared scheduling primitives for background passes (refine, distill).

Two execution modes:

* ``daily_loop`` — fires the job once a day at HH:MM in a named timezone.
  Preserves the original "wake up, run once, sleep ~24h" behaviour used
  before the overnight-window mode existed.

* ``window_loop`` — runs the job back-to-back inside a recurring nightly
  window ``[start, end]`` (e.g. 23:00 → 06:00 America/Chicago) until the
  caller's ``is_done`` predicate signals the backlog is drained, then
  sleeps until the next window opens. Honors DST via ``zoneinfo`` and
  handles windows that wrap past midnight.

Both loops are cancellation-safe and survive transient run errors with a
short cooldown so a recurring failure can't spin the loop hot.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — stdlib in py3.9+
    ZoneInfo = None  # type: ignore

log = logging.getLogger("llm-api.scheduler")


# ── Parsing helpers ─────────────────────────────────────────────────────────
def parse_hhmm(value: str) -> tuple[int, int] | None:
    """Parse ``HH:MM`` (24h). Returns ``None`` for empty or malformed input."""
    s = (value or "").strip()
    if not s:
        return None
    try:
        h_str, m_str = s.split(":", 1)
        h, m = int(h_str), int(m_str)
        if 0 <= h < 24 and 0 <= m < 60:
            return h, m
    except (ValueError, AttributeError):
        pass
    return None


def zoneinfo_or_utc(name: str):
    """Return a ``ZoneInfo`` for ``name`` falling back to UTC. Logs once
    on bad input so a typo'd env var is visible without crashing the loop."""
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception as e:
        log.warning("invalid timezone %r (%s) — falling back to UTC", name, e)
        return timezone.utc


# ── Time-math helpers ───────────────────────────────────────────────────────
def next_at(now_utc: datetime, hhmm: str, tz_name: str) -> datetime | None:
    """Next UTC datetime when the daily ``HH:MM`` cron fires in ``tz_name``,
    or ``None`` if the schedule is disabled (empty/invalid input).

    Honors DST via zoneinfo so e.g. ``02:00 America/Chicago`` means 02:00
    wall-clock year-round.
    """
    parsed = parse_hhmm(hhmm)
    if parsed is None:
        return None
    tz = zoneinfo_or_utc(tz_name)
    now_local = now_utc.astimezone(tz)
    h, m = parsed
    target_local = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
    if target_local <= now_local:
        target_local += timedelta(days=1)
    return target_local.astimezone(timezone.utc)


def window_bounds(
    now_utc: datetime,
    start_hhmm: str,
    end_hhmm: str,
    tz_name: str,
) -> tuple[datetime, datetime] | None:
    """Return ``(start_utc, end_utc)`` for the recurring nightly window
    that contains ``now_utc`` — or, if ``now_utc`` is outside any window,
    the next upcoming one.

    Returns ``None`` if either bound is empty/invalid (window disabled).
    Handles wraparound: when ``end <= start`` the window crosses midnight
    (e.g. 23:00 → 06:00).
    """
    sp = parse_hhmm(start_hhmm)
    ep = parse_hhmm(end_hhmm)
    if sp is None or ep is None:
        return None

    tz = zoneinfo_or_utc(tz_name)
    now_local = now_utc.astimezone(tz)
    sh, sm = sp
    eh, em = ep
    today_start = now_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
    today_end = now_local.replace(hour=eh, minute=em, second=0, microsecond=0)

    wraps = (eh, em) <= (sh, sm)
    if wraps:
        if now_local >= today_start:
            # In the window that started today; ends tomorrow morning.
            ws, we = today_start, today_end + timedelta(days=1)
        elif now_local < today_end:
            # In the window that started yesterday; ends this morning.
            ws, we = today_start - timedelta(days=1), today_end
        else:
            # Between morning end and evening start — next window opens tonight.
            ws, we = today_start, today_end + timedelta(days=1)
    else:
        if now_local < today_start:
            ws, we = today_start, today_end
        elif now_local < today_end:
            ws, we = today_start, today_end
        else:
            ws = today_start + timedelta(days=1)
            we = today_end + timedelta(days=1)

    return ws.astimezone(timezone.utc), we.astimezone(timezone.utc)


def in_window(
    now_utc: datetime,
    start_hhmm: str,
    end_hhmm: str,
    tz_name: str,
) -> bool:
    """``True`` if ``now_utc`` is currently inside the recurring window."""
    bounds = window_bounds(now_utc, start_hhmm, end_hhmm, tz_name)
    if bounds is None:
        return False
    ws, we = bounds
    return ws <= now_utc < we


# ── Loops ───────────────────────────────────────────────────────────────────
async def daily_loop(
    *,
    name: str,
    hhmm: str,
    tz_name: str,
    run: Callable[[], Awaitable[dict]],
) -> None:
    """Fire ``run`` once a day at ``hhmm`` in ``tz_name``. No-op if disabled."""
    if parse_hhmm(hhmm) is None:
        log.info("%s: daily scheduler disabled (at=%r)", name, hhmm)
        return
    log.info("%s: daily scheduler at %s %s", name, hhmm, tz_name)
    while True:
        try:
            now = datetime.now(timezone.utc)
            target = next_at(now, hhmm, tz_name)
            if target is None:
                return
            wait_s = max((target - now).total_seconds(), 0.0)
            log.info("%s: next run at %s (in %.1fh)",
                     name, target.isoformat(timespec="minutes"), wait_s / 3600)
            await asyncio.sleep(wait_s)
            log.info("%s: starting auto-run", name)
            result = await run()
            log.info("%s: %s", name, result)
        except asyncio.CancelledError:
            log.info("%s: stopping", name)
            raise
        except Exception as e:
            log.exception("%s: iteration failed: %s", name, e)
            await asyncio.sleep(300)


async def window_loop(
    *,
    name: str,
    start_hhmm: str,
    end_hhmm: str,
    tz_name: str,
    run: Callable[[], Awaitable[dict]],
    is_done: Callable[[dict], bool],
    batch_pause_s: float,
) -> None:
    """Run ``run`` back-to-back inside the recurring nightly window
    until ``is_done(result)`` reports the backlog is drained, then sleep
    until the next window opens. No-op if the window is disabled.
    """
    if window_bounds(datetime.now(timezone.utc), start_hhmm, end_hhmm, tz_name) is None:
        log.info("%s: window scheduler disabled (start=%r end=%r)",
                 name, start_hhmm, end_hhmm)
        return
    log.info("%s: window scheduler %s → %s %s (pause=%.0fs)",
             name, start_hhmm, end_hhmm, tz_name, batch_pause_s)

    while True:
        try:
            now = datetime.now(timezone.utc)
            bounds = window_bounds(now, start_hhmm, end_hhmm, tz_name)
            if bounds is None:
                return
            ws, we = bounds

            if now < ws:
                wait_s = (ws - now).total_seconds()
                log.info("%s: next window %s → %s (in %.1fh)",
                         name, ws.isoformat(timespec="minutes"),
                         we.isoformat(timespec="minutes"), wait_s / 3600)
                await asyncio.sleep(wait_s)
                continue

            log.info("%s: running batch (window ends %s)",
                     name, we.isoformat(timespec="minutes"))
            try:
                result = await run()
            except Exception as e:
                log.exception("%s: batch failed: %s", name, e)
                # Cooldown so a recurring failure can't pin the GPU.
                await asyncio.sleep(max(batch_pause_s, 60.0))
                continue
            log.info("%s: %s", name, result)

            done = False
            try:
                done = bool(is_done(result))
            except Exception as e:
                log.exception("%s: is_done predicate raised: %s", name, e)

            if done:
                next_now = we + timedelta(seconds=1)
                next_bounds = window_bounds(next_now, start_hhmm, end_hhmm, tz_name)
                if next_bounds is None:
                    return
                next_ws = next_bounds[0]
                wait_s = max((next_ws - datetime.now(timezone.utc)).total_seconds(), 0.0)
                log.info("%s: backlog drained — sleeping until %s (%.1fh)",
                         name, next_ws.isoformat(timespec="minutes"), wait_s / 3600)
                await asyncio.sleep(wait_s)
                continue

            now2 = datetime.now(timezone.utc)
            if now2 >= we:
                # Window closed during the batch — loop will recompute next window.
                continue
            remain_s = (we - now2).total_seconds()
            await asyncio.sleep(min(batch_pause_s, remain_s))
        except asyncio.CancelledError:
            log.info("%s: stopping", name)
            raise

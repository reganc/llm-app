"""Checks watched pages on schedule, one at a time. No model is involved."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import notify
import safe_fetch
import watch_check
import watch_store as store
from config import CFG

log = logging.getLogger("llm-api.watches")

TICK_S = 20.0
_CONFIRM_AFTER_S = 5.0
# A site that fails once is usually back next time.
_TELL_AFTER_FAILURES = 3

_lock = asyncio.Lock()


def reset() -> None:
    """Test hook: locks bind to an event loop."""
    global _lock
    _lock = asyncio.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _pause() -> None:
    await asyncio.sleep(_CONFIRM_AFTER_S)


def next_check(watch: dict) -> str:
    return (_now() + timedelta(minutes=watch["every_minutes"])).isoformat(timespec="seconds")


async def look(watch: dict) -> watch_check.Result:
    """Load the page and compare. The first look and any change are loaded twice.
    Raises FetchError."""
    async def load() -> list[str]:
        return watch_check.lines(await safe_fetch.fetch(watch["url"], watch["approved_host"]))

    rule, before = watch["rule"], watch["state"]
    first = await load()
    result = watch_check.compare(rule, first, before)
    if before is None or result.confirm:
        await _pause()
        result = watch_check.confirmed(rule, first, await load(), before)
    return result


async def check(watch_id: str) -> None:
    async with _lock:
        watch = store.get(watch_id)
        if watch is None:
            return
        try:
            result = await look(watch)
        except safe_fetch.FetchError as e:
            await _failed(watch, str(e))
            return
        except Exception as e:
            log.exception("watch %s failed", watch_id)
            await _failed(watch, f"The page could not be read ({type(e).__name__}).")
            return
        current = store.get(watch_id)
        if current is None or any(current[k] != watch[k] for k in ("url", "rule")):
            return  # deleted or edited meanwhile: this reading belongs to the old one
        store.update(watch_id, {"state": result.state, "shown": result.shown, "error": None,
                                "failures": 0, "last_checked_at": store.now()})
        if result.alert:
            log.info("watch %s: changed", watch_id)
            await _alert(watch, result.alert, "change")


async def _failed(watch: dict, problem: str) -> None:
    failures = watch["failures"] + 1
    if store.update(watch["id"], {"error": problem, "failures": failures,
                                  "last_checked_at": store.now()}) is None:
        return
    if failures == _TELL_AFTER_FAILURES:
        await _alert(watch, f"Could not be checked {failures} times in a row. {problem}", "error")


def message(watch: dict, text: str) -> str:
    return f"{watch['name']}\n{text}\n{watch['url']}"


async def _alert(watch: dict, text: str, kind: str) -> None:
    alert = store.add_alert(watch["id"], text, kind)
    if alert is None or not watch.get("notify"):
        return
    try:
        await notify.send(message(watch, text))
        outcome = "sent"
    except notify.NotifyError as e:
        outcome = f"failed: {e}"
    except Exception as e:  # its text could hold the token
        outcome = f"failed: {type(e).__name__}"
    store.annotate_alert(watch["id"], alert["id"], {"notified": outcome})


async def tick(now: datetime) -> None:
    for watch in store.list_all():
        due = watch["next_check_at"]
        if not watch["enabled"] or not due or datetime.fromisoformat(due) > now:
            continue
        store.update(watch["id"], {"next_check_at": next_check(watch)})  # first: a crash must not loop
        await check(watch["id"])


async def scheduler_loop() -> None:
    if not CFG.watches_enabled:
        log.info("watches: scheduler disabled")
        return
    log.info("watches: scheduler on, %d watch(es)", store.count())
    while True:
        try:
            await tick(_now())
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("watches: tick failed")
        await asyncio.sleep(TICK_S)

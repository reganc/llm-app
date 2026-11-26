"""Turn apps' existing status endpoints into hub cards (see ~/apps/HUB.md).

Each adapter reads fixed paths under the base URL from fleet.yaml and writes
plain-text items from fields it picks itself: no free text from the app is
passed on except where noted, and no money amounts ever. When an app serves its
own `GET /hub/summary`, its adapter here is deleted.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

Get = Callable[[str], Awaitable[Any]]
Item = dict[str, Any]

_DAY = timedelta(hours=24)
_DELPHI_PAGE = 50  # /api/signals returns at most this many


def _since(at: object, now: datetime) -> bool:
    try:
        return now - datetime.fromisoformat(str(at)) <= _DAY
    except (TypeError, ValueError):
        return False


async def trading(get: Get, base: str, now: datetime) -> list[Item]:
    items: list[Item] = []
    data = await get(f"{base}/admin/data_health")
    s = data.get("summary") or {}
    fail, warn = int(s.get("fail") or 0), int(s.get("warn") or 0)
    failing = [c.get("name", "?") for c in data.get("checks") or [] if c.get("status") == "fail"][:3]
    if fail:
        items.append({"title": f"Data checks failing: {', '.join(failing)}", "severity": "alert"})
    elif warn:
        items.append({"title": f"{warn} data check{'s' if warn != 1 else ''} with warnings", "severity": "warn"})

    health = await get(f"{base}/portfolio/health")
    overall = str(health.get("overall") or "").upper()
    if overall and overall not in ("OK", "HEALTHY", "GOOD"):
        parts = [f"{k} {str((health.get(k) or {}).get('status', '')).lower()}" for k in ("positions", "exposure")
                 if str((health.get(k) or {}).get("status", "")).upper() not in ("", "OK", "HEALTHY", "GOOD")]
        items.append({"title": f"Portfolio risk {overall.lower()}" + (f": {', '.join(parts)}" if parts else ""),
                      "severity": "alert" if overall == "CRITICAL" else "warn"})

    brief = await get(f"{base}/research/briefing")
    if brief.get("available"):
        regime = str((brief.get("regime") or {}).get("regime") or "unknown").lower()
        buys = [str(t.get("ticker")) for t in brief.get("buy_signals") or [] if t.get("ticker")][:5]
        items.append({"title": f"Market regime {regime}; buy signals: {', '.join(buys) if buys else 'none'}",
                      "severity": "info"})
    return items


async def delphi(get: Get, base: str, now: datetime) -> list[Item]:
    signals = [s for s in await get(f"{base}/api/signals") if isinstance(s, dict) and _since(s.get("time"), now)]
    if not signals:
        return [{"title": "No new prediction-market signals in the last day", "severity": "info"}]
    top = max(signals, key=lambda s: float(s.get("score") or 0))
    question = " ".join(str(top.get("question") or "").split())[:100]  # the market's own question
    count = f"{len(signals)} or more" if len(signals) >= _DELPHI_PAGE else str(len(signals))
    return [{"title": f"{count} new signals in the last day; strongest: {question}", "severity": "info"}]


async def nexus(get: Get, base: str, now: datetime) -> list[Item]:
    items: list[Item] = []
    drift = await get(f"{base}/api/health/drift")
    if drift.get("gate_tripped"):
        items.append({"title": "Drift gate tripped: new hypotheses are being held back", "severity": "warn"})
    episodes = [e for e in (await get(f"{base}/api/episodes")).get("episodes") or [] if _since(e.get("opened_at"), now)]
    domains = sorted({d for e in episodes for d in e.get("domains") or []})
    items.append({"title": f"{len(episodes)} cross-domain episodes in the last day"
                           + (f" ({', '.join(domains[:4])})" if domains else ""), "severity": "info"})
    return items


async def homelab(get: Get, base: str, now: datetime) -> list[Item]:
    import briefing  # the digest reader already lives there

    line = briefing.homelab_section(now)
    if not line:
        return [{"title": "No digest found", "severity": "warn"}]
    first = line.split("\n")[0].removeprefix("Homelab: ")
    serious = any(w in line for w in ("critical", "high", "serious", "out of date"))
    return [{"title": first.rstrip("."), "severity": "warn" if serious else "info"}]


ADAPTERS: dict[str, Callable[[Get, str, datetime], Awaitable[list[Item]]]] = {
    "trading": trading,
    "delphi": delphi,
    "nexus": nexus,
    "homelab": homelab,
}

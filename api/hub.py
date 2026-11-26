"""The app hub: short status cards from the other apps, for /apps and the briefing.

The app list is ~/apps/fleet.yaml (mounted read-only). Private apps write their card
to ~/apps/.hub (mounted read-only); others are read by adapters (hub_adapters.py).
No model is involved. Only URLs in fleet.yaml are fetched, only on
host.docker.internal, without redirects. Every title passes the privacy rule for
its app's level before it is shown: the bot is reachable from the internet.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ValidationError

import hub_adapters
from auth import verify_api_key

log = logging.getLogger("llm-api.hub")

FLEET_FILE = Path(os.getenv("FLEET_FILE", "/fleet/fleet.yaml"))
HUB_DIR = Path(os.getenv("HUB_DIR", "/hub"))
_ALLOWED_HOSTS = {"host.docker.internal"}
_TIMEOUT = httpx.Timeout(4.0)
_MAX_BYTES = 2_000_000
_MAX_ITEMS = 6
_STALE = timedelta(hours=36)
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

Severity = Literal["info", "warn", "alert"]
_RANK = {"info": 0, "warn": 1, "alert": 2}
_LEVELS: tuple[Severity, Severity, Severity] = ("info", "warn", "alert")


class Item(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    severity: Severity = "info"
    due: date | None = None
    link: str | None = Field(default=None, max_length=300)


class Card(BaseModel):
    items: list[Item] = Field(default_factory=list, max_length=20)
    generated_at: datetime | None = None


@dataclass(frozen=True)
class App:
    id: str
    title: str
    privacy: Literal["home", "private"]
    ui: str | None = None
    health: str | None = None
    summary: dict[str, str] = field(default_factory=dict)
    ask: str | None = None          # how to ask it questions (ask_apps.py); home apps only
    restart: dict | None = None     # {project, services}: approved restarts (actions.py); home only


@dataclass
class Status:
    app: App
    up: bool | None = None          # None: no health check
    items: list[Item] = field(default_factory=list)
    withheld: int = 0
    problem: str | None = None      # the card couldn't be read, or is out of date

    @property
    def level(self) -> Severity:
        if self.up is False:
            return "alert"
        worst = max((_RANK[i.severity] for i in self.items), default=0)
        if self.problem:
            worst = max(worst, 1)
        return _LEVELS[worst]


# ── the app list ─────────────────────────────────────────────────────────────

def _safe_url(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    return url.rstrip("/") if parts.scheme == "http" and parts.hostname in _ALLOWED_HOSTS else None


_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")


def _restart(spec: Any) -> dict | None:
    """{project, services} with Docker-safe names, or None. The executor checks it again."""
    if not isinstance(spec, dict):
        return None
    project, services = spec.get("project"), spec.get("services")
    if not (isinstance(project, str) and _NAME.match(project) and isinstance(services, list) and services
            and all(isinstance(s, str) and _NAME.match(s) for s in services)):
        return None
    return {"project": project, "services": list(dict.fromkeys(services))}


def load_fleet(path: Path | None = None) -> list[App]:
    """The apps in fleet.yaml. Entries that don't fit the rules are skipped and logged."""
    try:
        raw = yaml.safe_load((path or FLEET_FILE).read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("fleet list unreadable: %s", type(e).__name__)
        return []
    apps: list[App] = []
    for entry in raw.get("apps") or []:
        if not isinstance(entry, dict) or not _ID.match(str(entry.get("id", ""))):
            continue
        privacy = entry.get("privacy")
        if privacy not in ("home", "private"):
            log.warning("fleet: %s has no valid privacy level; skipped", entry["id"])
            continue
        given = entry.get("summary")
        summary = {k: str(v) for k, v in given.items() if k in ("file", "adapter", "base")} if isinstance(given, dict) else {}
        if "base" in summary and not _safe_url(summary["base"]):
            summary.pop("base")
        ui = entry.get("ui") if isinstance(entry.get("ui"), str) and entry["ui"].startswith("http") else None
        ask = entry.get("ask") if privacy == "home" and isinstance(entry.get("ask"), str) else None
        restart = _restart(entry.get("restart")) if privacy == "home" else None
        apps.append(App(id=entry["id"], title=str(entry.get("title") or entry["id"])[:40], privacy=privacy,
                        ui=ui, health=_safe_url(entry.get("health")), summary=summary, ask=ask,
                        restart=restart))
    return apps


# ── privacy ──────────────────────────────────────────────────────────────────

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FIGURE = re.compile(r"\d{3,}|\d[.,]\d")


def allowed(title: str, privacy: str) -> bool:
    """No money amounts anywhere; for private apps, no number over 2 digits but a date."""
    if "$" in title or "€" in title or "£" in title:
        return False
    return privacy != "private" or not _FIGURE.search(_DATE.sub("", title))


def _clean(items: list[Item], privacy: str) -> tuple[list[Item], int]:
    kept = [i.model_copy(update={"title": " ".join(i.title.split())}) for i in items]
    ok = [i for i in kept if allowed(i.title, privacy)]
    return ok[:_MAX_ITEMS], len(kept) - len(ok)


# ── reading ──────────────────────────────────────────────────────────────────

def _read_file_card(app: App, now: datetime) -> Status:
    name = Path(app.summary["file"]).name  # never a path outside the hub folder
    try:
        raw = (HUB_DIR / name).read_bytes()[:_MAX_BYTES]
        card = Card.model_validate(json.loads(raw))
    except FileNotFoundError:
        return Status(app, problem="hasn't reported yet")
    except (OSError, ValueError, ValidationError):
        return Status(app, problem="its card couldn't be read")
    items, withheld = _clean(card.items, app.privacy)
    status = Status(app, items=items, withheld=withheld)
    when = card.generated_at
    if when is None or now - (when if when.tzinfo else when.replace(tzinfo=timezone.utc)) > _STALE:
        status.problem = "its card is out of date"
    return status


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    if not _safe_url(url):
        raise ValueError("address not allowed")
    async with client.stream("GET", url) as r:
        r.raise_for_status()
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk
            if len(body) > _MAX_BYTES:
                raise ValueError("response too large")
    return json.loads(body)


async def _check(client: httpx.AsyncClient, app: App, now: datetime) -> Status:
    status = Status(app)
    if app.health:
        try:
            r = await client.get(app.health)
            status.up = r.is_success
        except httpx.HTTPError:
            status.up = False
    if status.up is False:
        return status
    if "file" in app.summary:
        card = _read_file_card(app, now)
        card.up = status.up
        return card
    adapter = hub_adapters.ADAPTERS.get(app.summary.get("adapter", ""))
    if adapter:
        try:
            raw = await adapter(lambda u: _get_json(client, u), app.summary.get("base", ""), now)
            items = [Item.model_validate({**i, "title": str(i.get("title", ""))[:160]}) for i in raw]
            status.items, status.withheld = _clean(items, app.privacy)
        except Exception as e:  # an app's odd reply must not break the others
            log.info("hub %s: %s", app.id, type(e).__name__)
            status.problem = "its status couldn't be read"
    return status


async def gather(now: datetime | None = None, apps: list[App] | None = None,
                 client: httpx.AsyncClient | None = None) -> list[Status]:
    now = now or datetime.now(timezone.utc)
    apps = load_fleet() if apps is None else apps
    own = client is None
    client = client or httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False)
    try:
        return list(await asyncio.gather(*(_check(client, a, now) for a in apps)))
    finally:
        if own:
            await client.aclose()


# ── text ─────────────────────────────────────────────────────────────────────

_ICON = {"info": "🟢", "warn": "🟠", "alert": "🔴"}


def _due(d: date | None, today: date) -> str:
    if d is None:
        return ""
    days = (d - today).days
    return " (overdue)" if days < 0 else " (today)" if days == 0 else f" (in {days} day{'s' if days != 1 else ''})"


def _lines(s: Status, today: date, min_severity: int = 0) -> list[str]:
    if s.up is False:
        return ["not responding"]
    rows = [f"{i.title}{_due(i.due, today)}" for i in s.items if _RANK[i.severity] >= min_severity]
    if s.problem:
        rows.append(s.problem)
    if s.withheld:
        rows.append(f"{s.withheld} item{'s' if s.withheld != 1 else ''} withheld (contained figures)")
    return rows


def apps_text(statuses: list[Status], today: date) -> str:
    """The /apps reply: every app, worst first."""
    if not statuses:
        return "No apps are listed. The list is ~/apps/fleet.yaml."
    order = sorted(statuses, key=lambda s: (-_RANK[s.level], s.app.title))
    blocks = []
    for s in order:
        rows = _lines(s, today) or ["up"]
        blocks.append(f"{_ICON[s.level]} {s.app.title}\n" + "\n".join(f"  • {r}" for r in rows))
    return "\n".join(blocks)


def briefing_section(statuses: list[Status], today: date) -> str | None:
    """Only what needs attention: warnings, alerts, apps down, and deadlines within 14 days."""
    rows = []
    for s in sorted(statuses, key=lambda s: (-_RANK[s.level], s.app.title)):
        picked = [i for i in s.items if _RANK[i.severity] >= 1 or (i.due and (i.due - today).days <= 14)]
        shown = _lines(Status(s.app, s.up, picked, s.withheld, s.problem), today)
        rows += [f"• {s.app.title}: {r}" for r in shown]
    return ("Apps:\n" + "\n".join(rows) + "\nSend /apps for all of them.") if rows else None


# ── API ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/hub", tags=["hub"], dependencies=[Depends(verify_api_key)])


@router.get("")
async def get_hub():
    statuses = await gather()
    return {"apps": [{"id": s.app.id, "title": s.app.title, "ui": s.app.ui, "level": s.level, "up": s.up,
                      "items": [i.model_dump(mode="json") for i in s.items], "withheld": s.withheld,
                      "problem": s.problem} for s in statuses]}

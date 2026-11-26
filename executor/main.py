"""The executor: restarts an app's containers, and nothing else.

The only part of the action gate with Docker access (the socket is root on
this machine), so it is kept small and does its own checks:

  - POST /restart {app, action_id} with ACTIONS_EXECUTOR_TOKEN; empty token,
    nothing works.
  - Only apps whose fleet.yaml entry is `privacy: home` and has
    `restart: {project, services}`. fleet.yaml is read from a read-only mount,
    so llm-app can't add to the list. Containers are found by their compose
    labels; databases are simply not listed.
  - One restart at a time, one per app every 10 minutes, 6 a day, counted from
    its own log, so even a compromised llm-app can only restart listed apps,
    slowly. It can't tell whether a person approved: llm-app's gate does that.
  - After restarting, waits up to 90 s for the app's health URL.
  - Every attempt goes to a hash-chained log in /data.

No published port: only llm-api, on the same Docker network, can reach it.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("executor")

TOKEN = os.getenv("ACTIONS_EXECUTOR_TOKEN", "").strip()
FLEET_FILE = Path(os.getenv("FLEET_FILE", "/fleet/fleet.yaml"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DOCKER_SOCK = os.getenv("DOCKER_SOCK", "/var/run/docker.sock")
COOLDOWN_S = 600
MAX_PER_DAY = 6
HEALTH_WAIT_S = 90.0
HEALTH_EVERY_S = 3.0
STOP_WAIT_S = 10            # seconds Docker gives each container to stop
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_ALLOWED_HOSTS = {"host.docker.internal"}
GENESIS = "0" * 64

app = FastAPI(title="executor", docs_url=None, redoc_url=None, openapi_url=None)
_lock = asyncio.Lock()


class RestartIn(BaseModel):
    app: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    action_id: str = Field(pattern=r"^[0-9a-f]{12}$")


# ── the list ─────────────────────────────────────────────────────────────────

def _health(url) -> str | None:
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    return url if parts.scheme == "http" and parts.hostname in _ALLOWED_HOSTS else None


def restartable(path: Path | None = None) -> dict[str, dict]:
    """id → {project, services, health} for every app that opts in."""
    try:
        raw = yaml.safe_load((path or FLEET_FILE).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    found = {}
    for entry in raw.get("apps") or []:
        if not isinstance(entry, dict) or entry.get("privacy") != "home":
            continue
        spec, app_id = entry.get("restart"), str(entry.get("id", ""))
        if not _ID.match(app_id) or not isinstance(spec, dict):
            continue
        project, services = spec.get("project"), spec.get("services")
        if not (isinstance(project, str) and _NAME.match(project) and isinstance(services, list)
                and services and all(isinstance(s, str) and _NAME.match(s) for s in services)):
            continue
        found[app_id] = {"project": project, "services": list(dict.fromkeys(services)),
                         "health": _health(entry.get("health"))}
    return found


# ── log ──────────────────────────────────────────────────────────────────────

def _log_path() -> Path:
    return DATA_DIR / "restarts.jsonl"


def _rows() -> list[dict]:
    path = _log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest(prev: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((prev + raw).encode()).hexdigest()


def _append(event: dict) -> None:
    rows = _rows()
    seq, prev = (rows[-1]["seq"] + 1, rows[-1]["hash"]) if rows else (0, GENESIS)
    payload = {"seq": seq, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "ts": time.time(), "event": event}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _log_path().open("a") as f:
        f.write(json.dumps({**payload, "prev_hash": prev, "hash": _digest(prev, payload)}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _limits(app_id: str, now: float) -> str | None:
    """Why a restart can't start now, from the log of earlier attempts."""
    tries = [r for r in _rows() if r["event"].get("kind") == "attempt"]
    if any(r["event"]["app"] == app_id and now - r["ts"] < COOLDOWN_S for r in tries):
        return f"{app_id} was restarted less than {COOLDOWN_S // 60} minutes ago."
    if sum(1 for r in tries if now - r["ts"] < 86400) >= MAX_PER_DAY:
        return f"No more than {MAX_PER_DAY} restarts a day."
    return None


# ── docker ───────────────────────────────────────────────────────────────────

def docker_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=DOCKER_SOCK),
                             base_url="http://docker", timeout=httpx.Timeout(STOP_WAIT_S + 30))


async def _containers(client: httpx.AsyncClient, project: str, services: list[str]) -> list[dict]:
    """The project's containers for those services, in the listed order; every one must exist."""
    flt = json.dumps({"label": [f"com.docker.compose.project={project}"]})
    r = await client.get("/containers/json", params={"all": "1", "filters": flt})
    r.raise_for_status()
    by_service: dict[str, list[dict]] = {}
    for c in r.json():
        labels = c.get("Labels") or {}
        if labels.get("com.docker.compose.project") == project:   # don't trust the filter alone
            by_service.setdefault(labels.get("com.docker.compose.service", ""), []).append(c)
    missing = [s for s in services if s not in by_service]
    if missing:
        raise LookupError(f"no container for {', '.join(missing)}")
    return [c for s in services for c in by_service[s]]


async def _wait_healthy(url: str | None, *, client: httpx.AsyncClient | None = None) -> str:
    if url is None:
        return "none"
    own = client is None
    client = client or httpx.AsyncClient(timeout=5.0, follow_redirects=False)
    deadline = time.monotonic() + HEALTH_WAIT_S
    try:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).is_success:
                    return "ok"
            except httpx.HTTPError:
                pass
            await asyncio.sleep(HEALTH_EVERY_S)
        return "failed"
    finally:
        if own:
            await client.aclose()


# ── endpoint ─────────────────────────────────────────────────────────────────

def _authorized(header: str | None) -> bool:
    given = (header or "").removeprefix("Bearer ").strip()
    return bool(TOKEN) and hmac.compare_digest(given.encode(), TOKEN.encode())


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "configured": bool(TOKEN)}


@app.post("/restart")
async def restart(req: RestartIn, authorization: str | None = Header(default=None)) -> dict:
    if not TOKEN:
        raise HTTPException(503, "ACTIONS_EXECUTOR_TOKEN is not set.")
    if not _authorized(authorization):
        raise HTTPException(401, "Not authorized.")
    spec = restartable().get(req.app)
    if spec is None:
        raise HTTPException(403, f"{req.app} can't be restarted from here.")
    if _lock.locked():
        raise HTTPException(409, "Another restart is in progress.")
    async with _lock:
        if (problem := _limits(req.app, time.time())) is not None:
            raise HTTPException(429, problem)
        _append({"kind": "attempt", "app": req.app, "action_id": req.action_id,
                 "project": spec["project"], "services": spec["services"]})
        started = time.monotonic()
        names: list[str] = []
        try:
            async with docker_client() as client:
                for c in await _containers(client, spec["project"], spec["services"]):
                    r = await client.post(f"/containers/{c['Id']}/restart", params={"t": STOP_WAIT_S})
                    r.raise_for_status()
                    names.append(((c.get("Names") or ["?"])[0]).lstrip("/"))
        except (LookupError, httpx.HTTPError) as e:
            problem = f"Docker: {e}" if isinstance(e, LookupError) else f"Docker failed ({type(e).__name__})."
            _append({"kind": "failed", "app": req.app, "action_id": req.action_id,
                     "restarted": names, "problem": problem})
            return {"ok": False, "containers": names, "health": "not checked",
                    "seconds": round(time.monotonic() - started), "problem": problem}
        state = await _wait_healthy(spec["health"])
        seconds = round(time.monotonic() - started)
        _append({"kind": "restarted", "app": req.app, "action_id": req.action_id,
                 "restarted": names, "health": state, "seconds": seconds})
        return {"ok": state != "failed", "containers": names, "health": state,
                "seconds": seconds, "problem": None if state != "failed"
                else f"It didn't answer its health check within {int(HEALTH_WAIT_S)} s."}

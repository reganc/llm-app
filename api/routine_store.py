"""Routines and their run history, one JSON file per routine in DATA_DIR/routines.

Single-process (uvicorn --workers 1) and only touched from the event loop, so
no locking. Every write goes to a temp file and is renamed into place, and
memory is updated only after the write succeeds. Callers always get copies.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("llm-api.routines")

MAX_RUNS = 30  # history kept per routine

_dir: Path = DATA_DIR / "routines"
_routines: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def load(directory: Path | None = None) -> None:
    """(Re)load everything from disk. Unreadable files are skipped, never changed."""
    global _dir, _routines
    _dir = directory or _dir
    _dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict] = {}
    for path in sorted(_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if data["id"] != path.stem or not isinstance(data["runs"], list):
                raise ValueError("unexpected shape")
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.error("skipping unreadable routine %s: %s", path.name, e)
            continue
        loaded[data["id"]] = data
    _routines = loaded


def _save(routine: dict) -> dict:
    path = _dir / f"{routine['id']}.json"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(routine, f, indent=1)
    os.replace(tmp, path)
    _routines[routine["id"]] = routine
    return copy.deepcopy(routine)


# ── routines ─────────────────────────────────────────────────────────────────

def create(fields: dict) -> dict:
    now = _now()
    return _save({**copy.deepcopy(fields), "id": _new_id(), "created_at": now,
                  "updated_at": now, "runs": []})


def get(routine_id: str) -> dict | None:
    found = _routines.get(routine_id)
    return copy.deepcopy(found) if found else None


def list_all() -> list[dict]:
    return sorted((copy.deepcopy(r) for r in _routines.values()),
                  key=lambda r: (r["created_at"], r["id"]))


def count() -> int:
    return len(_routines)


def update(routine_id: str, fields: dict) -> dict | None:
    current = _routines.get(routine_id)
    if current is None:
        return None
    return _save({**current, **copy.deepcopy(fields), "updated_at": _now()})


def set_next_run(routine_id: str, next_run_at: str | None) -> None:
    current = _routines.get(routine_id)
    if current is not None:
        _save({**current, "next_run_at": next_run_at})


def delete(routine_id: str) -> bool:
    if routine_id not in _routines:
        return False
    (_dir / f"{routine_id}.json").unlink(missing_ok=True)
    del _routines[routine_id]
    return True


# ── runs ─────────────────────────────────────────────────────────────────────

def _add_run(routine_id: str, run: dict) -> None:
    current = _routines.get(routine_id)
    if current is not None:
        _save({**current, "runs": [run, *current["runs"]][:MAX_RUNS]})


def start_run(routine_id: str, trigger: str) -> dict:
    run = {"id": _new_id(), "trigger": trigger, "status": "running",
           "started_at": _now(), "finished_at": None, "read": False}
    _add_run(routine_id, run)
    return dict(run)


def add_missed(routine_id: str, due_at: str) -> None:
    _add_run(routine_id, {
        "id": _new_id(), "trigger": "schedule", "status": "missed", "read": False,
        "started_at": due_at, "finished_at": _now(),
        "note": "The server was not running at the scheduled time, so this run was skipped."})


def finish_run(routine_id: str, run_id: str, fields: dict) -> None:
    """Record a run's outcome. No-op if the routine was deleted meanwhile."""
    current = _routines.get(routine_id)
    if current is None:
        return
    runs = [{**r, **copy.deepcopy(fields), "finished_at": _now()} if r["id"] == run_id else r
            for r in current["runs"]]
    _save({**current, "runs": runs})


def recover() -> int:
    """Mark runs cut off by a restart. Called once at startup."""
    n = 0
    for routine in list(_routines.values()):
        stuck = [r["id"] for r in routine["runs"] if r["status"] == "running"]
        for run_id in stuck:
            finish_run(routine["id"], run_id, {
                "status": "interrupted", "error": "The server restarted during this run."})
            n += 1
    return n


# ── inbox ────────────────────────────────────────────────────────────────────

def _finished(routine: dict) -> list[dict]:
    return [r for r in routine["runs"] if r["status"] != "running"]


def inbox(limit: int) -> list[dict]:
    items = [{**copy.deepcopy(run), "routine_id": r["id"], "routine_name": r["name"]}
             for r in _routines.values() for run in _finished(r)]
    return sorted(items, key=lambda i: i["started_at"], reverse=True)[:limit]


def unread_count() -> int:
    return sum(not run["read"] for r in _routines.values() for run in _finished(r))


def mark_read(routine_id: str, run_id: str) -> bool:
    current = _routines.get(routine_id)
    if current is None or all(r["id"] != run_id for r in current["runs"]):
        return False
    _save({**current, "runs": [{**r, "read": True} if r["id"] == run_id else r
                               for r in current["runs"]]})
    return True


def mark_all_read() -> int:
    n = 0
    for routine in list(_routines.values()):
        unread = {r["id"] for r in _finished(routine) if not r["read"]}
        if unread:
            _save({**routine, "runs": [{**r, "read": True} if r["id"] in unread else r
                                       for r in routine["runs"]]})
            n += len(unread)
    return n


load()

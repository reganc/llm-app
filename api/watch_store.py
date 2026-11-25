"""Watched pages and their alerts, one JSON file each in DATA_DIR/watches.

Same rules as routine_store: touched only from the event loop, each write goes
to a temp file that is renamed into place, and callers always get copies.
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

log = logging.getLogger("llm-api.watches")

MAX_ALERTS = 30

_dir: Path = DATA_DIR / "watches"
_watches: dict[str, dict] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(directory: Path | None = None) -> None:
    """(Re)load from disk. Unreadable files are skipped, never changed."""
    global _dir, _watches
    _dir = directory or _dir
    _dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict] = {}
    for path in sorted(_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if data["id"] != path.stem or not isinstance(data["alerts"], list):
                raise ValueError("unexpected shape")
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.error("skipping unreadable watch %s: %s", path.name, type(e).__name__)
            continue
        loaded[data["id"]] = data
    _watches = loaded


def _save(watch: dict) -> dict:
    path = _dir / f"{watch['id']}.json"
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(watch, f)
    os.replace(tmp, path)
    _watches[watch["id"]] = watch
    return copy.deepcopy(watch)


def create(fields: dict) -> dict:
    return _save({"state": None, "shown": "", "last_checked_at": None, "error": None, "failures": 0,
                  "alerts": [], **copy.deepcopy(fields), "id": uuid.uuid4().hex[:12],
                  "created_at": now()})


def get(watch_id: str) -> dict | None:
    found = _watches.get(watch_id)
    return copy.deepcopy(found) if found else None


def list_all() -> list[dict]:
    return sorted((copy.deepcopy(w) for w in _watches.values()),
                  key=lambda w: (w["created_at"], w["id"]))


def count() -> int:
    return len(_watches)


def update(watch_id: str, fields: dict) -> dict | None:
    """None if it was deleted meanwhile."""
    current = _watches.get(watch_id)
    return _save({**current, **copy.deepcopy(fields)}) if current else None


def delete(watch_id: str) -> bool:
    if watch_id not in _watches:
        return False
    (_dir / f"{watch_id}.json").unlink(missing_ok=True)
    del _watches[watch_id]
    return True


def add_alert(watch_id: str, text: str, kind: str) -> dict | None:
    current = _watches.get(watch_id)
    if current is None:
        return None
    alert = {"id": uuid.uuid4().hex[:12], "at": now(), "kind": kind, "text": text, "read": False}
    _save({**current, "alerts": [alert, *current["alerts"]][:MAX_ALERTS]})
    return dict(alert)


def annotate_alert(watch_id: str, alert_id: str, fields: dict) -> None:
    current = _watches.get(watch_id)
    if current is not None:
        _save({**current, "alerts": [{**a, **fields} if a["id"] == alert_id else a
                                     for a in current["alerts"]]})


def mark_read(watch_id: str) -> bool:
    current = _watches.get(watch_id)
    if current is None:
        return False
    if any(not a["read"] for a in current["alerts"]):
        _save({**current, "alerts": [{**a, "read": True} for a in current["alerts"]]})
    return True


def unread_count() -> int:
    return sum(not a["read"] for w in _watches.values() for a in w["alerts"])


load()

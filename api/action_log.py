"""Append-only, hash-chained log of every proposed action and what became of it.

One JSON line per event in DATA_DIR/actions/audit.jsonl:
    hash = sha256(prev_hash + canonical JSON of {seq, at, event})
so editing, removing or reordering a line breaks the chain, and verify() says
where. Events carry the action, its arguments, ids and outcomes: never message
text or secrets. Single process, touched only from the event loop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("llm-api.actions")

GENESIS = "0" * 64
_path: Path = DATA_DIR / "actions" / "audit.jsonl"
_last: tuple[int, str] | None = None  # (seq, hash) of the last line, read once


def use(path: Path) -> None:
    """Point the log at `path` (tests, startup)."""
    global _path, _last
    _path, _last = path, None


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(prev: str, payload: dict) -> str:
    return hashlib.sha256((prev + _canonical(payload)).encode()).hexdigest()


def _rows() -> list[dict]:
    if not _path.exists():
        return []
    return [json.loads(line) for line in _path.read_text().splitlines() if line.strip()]


def append(event: dict) -> dict:
    """Write one event; returns the stored row."""
    global _last
    if _last is None:
        rows = _rows()
        _last = (rows[-1]["seq"], rows[-1]["hash"]) if rows else (-1, GENESIS)
    seq, prev = _last[0] + 1, _last[1]
    payload = {"seq": seq, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "event": event}
    row = {**payload, "prev_hash": prev, "hash": _digest(prev, payload)}
    _path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    _last = (seq, row["hash"])
    return row


def verify() -> dict:
    """{"ok", "checked", "broken_at"}: recomputes every hash from the start."""
    prev = GENESIS
    rows = _rows()
    for i, row in enumerate(rows):
        payload = {"seq": row.get("seq"), "at": row.get("at"), "event": row.get("event")}
        if row.get("seq") != i or row.get("prev_hash") != prev or row.get("hash") != _digest(prev, payload):
            return {"ok": False, "checked": i, "broken_at": i}
        prev = row["hash"]
    return {"ok": True, "checked": len(rows), "broken_at": None}


def tail(n: int = 20) -> list[dict]:
    return _rows()[-n:]

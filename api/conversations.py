"""Persistent conversation store. One JSON file per conversation."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import CONV_DIR
from prompts import get_system_prompt

log = logging.getLogger("llm-api.conv")

_lock = threading.Lock()
_cache: dict[str, dict] = {}


def _path(conv_id: str) -> Path:
    return CONV_DIR / f"{conv_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_all() -> None:
    """Hydrate the cache from disk on first access."""
    if _cache:
        return
    for f in CONV_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            _cache[data["id"]] = data
        except Exception as e:
            log.warning("skip bad conv file %s: %s", f, e)


def _persist(conv: dict) -> None:
    _path(conv["id"]).write_text(json.dumps(conv, indent=2))


def create(name: str = "New Conversation",
           system_prompt_key: str = "default",
           custom_system_prompt: str | None = None,
           extra: dict | None = None) -> dict:
    with _lock:
        _load_all()
        conv_id = uuid.uuid4().hex[:12]
        system_content = custom_system_prompt or get_system_prompt(system_prompt_key)
        conv = {
            "id": conv_id,
            "name": name,
            "system_prompt_key": system_prompt_key,
            "messages": [{"role": "system", "content": system_content}],
            "created_at": _now(),
            "updated_at": _now(),
            "turn_count": 0,
            **(extra or {}),
        }
        _cache[conv_id] = conv
        _persist(conv)
        return conv


def list_all() -> list[dict]:
    with _lock:
        _load_all()
        items = [{k: v for k, v in c.items() if k != "messages"}
                 for c in _cache.values()]
        return sorted(items, key=lambda x: x["updated_at"], reverse=True)


def get(conv_id: str) -> dict | None:
    with _lock:
        _load_all()
        return _cache.get(conv_id)


def update(conv: dict) -> None:
    """Persist after modifying messages/turn_count etc."""
    conv["updated_at"] = _now()
    with _lock:
        _cache[conv["id"]] = conv
        _persist(conv)


def append_message(conv_id: str, role: str, content: str) -> dict | None:
    with _lock:
        _load_all()
        conv = _cache.get(conv_id)
        if not conv:
            return None
        conv["messages"].append({"role": role, "content": content})
        if role == "assistant":
            conv["turn_count"] += 1
        conv["updated_at"] = _now()
        _persist(conv)
        return conv


def pop_last(conv_id: str) -> None:
    with _lock:
        conv = _cache.get(conv_id)
        if conv and conv["messages"]:
            conv["messages"].pop()
            _persist(conv)


def delete(conv_id: str) -> bool:
    with _lock:
        _load_all()
        if conv_id not in _cache:
            return False
        del _cache[conv_id]
        try:
            _path(conv_id).unlink()
        except FileNotFoundError:
            pass
        return True


def count() -> int:
    with _lock:
        _load_all()
        return len(_cache)

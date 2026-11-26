"""The Telegram conversation the bot remembers.

Up to MAX_TURNS question/answer pairs, each forgotten after TTL_S; only the
newest that fit BUDGET_CHARS are sent to the model, so a long chat doesn't
slow every answer (the model's context is 64K tokens; 16K characters is about
4.5K). Kept in DATA_DIR/telegram_history.json (0600, atomic writes) so a
restart doesn't wipe it; `/new` erases it. The history belongs to one chat:
if Settings points the bot at another chat, it is never shown there.

Message text is stored here and nowhere else: never logged.
(Before 2026-11-26: 3 pairs of 500 characters, in memory, gone after 2 hours.)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("llm-api.telegram")

MAX_TURNS = 20
TTL_S = 24 * 3600
QUESTION_CHARS, ANSWER_CHARS = 1000, 1500
BUDGET_CHARS = 16_000

_path: Path = DATA_DIR / "telegram_history.json"
_chat_id = None
_turns: list[dict] = []          # {"at": epoch seconds, "q": str, "a": str}, oldest first


def load(path: Path | None = None) -> None:
    """(Re)read the saved history. An unreadable file starts empty and is left alone."""
    global _path, _chat_id, _turns
    _path = path or _path
    _chat_id, _turns = None, []
    try:
        data = json.loads(_path.read_text()) if _path.exists() else {}
        turns = data.get("turns") if isinstance(data, dict) else None
        if isinstance(turns, list):
            _chat_id = data.get("chat_id")
            _turns = [t for t in turns if isinstance(t, dict)
                      and isinstance(t.get("q"), str) and isinstance(t.get("a"), str)
                      and isinstance(t.get("at"), (int, float))][-MAX_TURNS:]
    except (OSError, ValueError) as e:
        log.error("telegram history unreadable, starting empty: %s", type(e).__name__)


def _save() -> None:
    _path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"chat_id": _chat_id, "turns": _turns}, f)
    os.replace(tmp, _path)


def remember(chat_id, question: str, answer: str, *, now: float | None = None) -> None:
    global _chat_id, _turns
    if chat_id != _chat_id:           # a different chat never inherits this one's history
        _chat_id, _turns = chat_id, []
    _turns = (_turns + [{"at": now if now is not None else time.time(),
                         "q": question[:QUESTION_CHARS], "a": answer[:ANSWER_CHARS]}])[-MAX_TURNS:]
    _save()


def turns(chat_id, *, now: float | None = None) -> list[dict]:
    """Chat messages for the model: this chat's live turns, newest kept within the budget."""
    if chat_id is None or chat_id != _chat_id:
        return []
    now = now if now is not None else time.time()
    live = [t for t in _turns if now - t["at"] <= TTL_S]
    kept, used = [], 0
    for t in reversed(live):
        size = len(t["q"]) + len(t["a"])
        if used + size > BUDGET_CHARS:
            break
        kept.append(t)
        used += size
    return [m for t in reversed(kept) for m in ({"role": "user", "content": t["q"]},
                                               {"role": "assistant", "content": t["a"]})]


def forget() -> None:
    global _turns
    _turns = []
    if _path.exists():
        _save()


load()

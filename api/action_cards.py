"""Approval cards in Telegram: the only way a proposed action moves on.

A card is sent when the model proposes an action (actions.py). Its buttons
carry the proposal's id and a one-time nonce. A tap counts only from the chosen
person in their private chat, on the card that proposal was sent as; taps from
anyone else are ignored without a word. After every step the card is edited to
say what happened, and its buttons are removed once decided, so an old card
can't be pressed again. Telegram's errors never include message text.
"""
from __future__ import annotations

import asyncio
import logging
import re

import actions
import notify
from config import get_setting

log = logging.getLogger("llm-api.actions")

_DATA = re.compile(r"act:([0-9a-f]{12}):(approve|reject|confirm):([0-9a-f]{0,8})")
_ACK = {"run": "Approved.", "confirm": "Tap again to confirm.", "rejected": "Rejected."}
_tasks: set[asyncio.Task] = set()   # approved actions running; polling never waits for them


def _markup(record: dict) -> dict:
    return {"inline_keyboard": [[{"text": label, "callback_data": data}
                                 for label, data in actions.buttons(record)]]}


async def send(record: dict) -> bool:
    """Sends the card; False (and the proposal is void) if it didn't arrive."""
    chat_id = get_setting(notify.CHAT_SETTING)
    sent = None
    if chat_id is not None:
        try:
            sent = await notify.call("sendMessage", {
                "chat_id": chat_id, "text": actions.card_text(record),
                "link_preview_options": {"is_disabled": True}, "reply_markup": _markup(record)})
        except notify.NotifyError as e:
            log.warning("approval card not delivered: %s", e)
    message_id = sent.get("message_id") if isinstance(sent, dict) else None
    if message_id is None:
        actions.undelivered(record["id"])
        return False
    actions.set_message(record["id"], message_id)
    return True


async def refresh(record: dict) -> None:
    """Edit the card to show the record's current state."""
    chat_id = get_setting(notify.CHAT_SETTING)
    if chat_id is None or record.get("message_id") is None:
        return
    try:
        await notify.call("editMessageText", {
            "chat_id": chat_id, "message_id": record["message_id"],
            "text": actions.card_text(record),
            "link_preview_options": {"is_disabled": True}, "reply_markup": _markup(record)})
    except notify.NotifyError as e:
        log.warning("approval card not updated: %s", e)


async def _answer(callback: dict, text: str) -> None:
    """Stops the button's spinner; Telegram shows `text` briefly."""
    try:
        await notify.call("answerCallbackQuery", {"callback_query_id": callback.get("id"),
                                                  "text": text[:200]})
    except notify.NotifyError as e:
        log.warning("button not acknowledged: %s", e)


def _mine(callback: dict) -> bool:
    chosen = get_setting(notify.CHAT_SETTING)
    sender = callback.get("from") or {}
    chat_info = (callback.get("message") or {}).get("chat") or {}
    return (chosen is not None and sender.get("id") == chosen
            and chat_info.get("id") == chosen and chat_info.get("type") == "private")


async def on_callback(callback: dict, hooks: dict | None = None) -> None:
    """A button tap. Runs the action when it is approved (and, if high risk, confirmed)."""
    if not _mine(callback):
        return
    match = _DATA.fullmatch(str(callback.get("data") or ""))
    if not match:
        await _answer(callback, "I don't know that button.")
        return
    action_id, verb, nonce = match.groups()
    record = actions.get(action_id)
    if record is None or record["message_id"] != (callback.get("message") or {}).get("message_id"):
        await _answer(callback, "That button is out of date.")
        return
    try:
        record, step = actions.decide(action_id, verb, nonce)
    except actions.Refused as e:
        await _answer(callback, str(e))
        if (current := actions.get(action_id)) is not None:
            await refresh(current)
        return
    await _answer(callback, _ACK[step])
    await refresh(record)
    if step == "run":
        task = asyncio.get_running_loop().create_task(_run(action_id, hooks))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)


async def _run(action_id: str, hooks: dict | None) -> None:
    """In the background: a summary re-run takes minutes, and taps keep working meanwhile."""
    try:
        await refresh(await actions.execute(action_id, hooks))
    except Exception as e:
        log.warning("action %s: %s", action_id, type(e).__name__)


async def sweep() -> None:
    """Expire overdue proposals and say so on their cards."""
    for record in actions.sweep():
        await refresh(record)

"""Telegram notifications for routine results, and /v1/notify/telegram.

The bot token is part of every Telegram URL, and httpx logs URLs at INFO, so
`RedactToken` is installed before anything is sent and errors raised from here
never carry the URL. Messages are plain text (no parse mode, no previews): the
content comes from web pages. Receiving messages is telegram_chat's job.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import agent
from auth import verify_api_key
from config import CFG, get_setting, set_setting

log = logging.getLogger("llm-api.notify")

API_BASE = "https://api.telegram.org"
_LIMIT = 4096            # Telegram's cap, in UTF-16 units
_HEADLINE_CHARS = 140
_MAX_RETRY_WAIT_S = 30
CHAT_SETTING = "telegram_chat_id"

_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")
_URL_RE = re.compile(r"https?://\S+")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^\s)]+\)")


class NotifyError(Exception):
    """Delivery failed; the message is safe to show and never has the token."""


class RedactToken(logging.Filter):
    """Removes bot tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if _TOKEN_RE.search(text):
            record.msg, record.args = _TOKEN_RE.sub("bot<redacted>", text), ()
        return True


def install_redaction() -> None:
    """Covers httpx's own logger and every handler (a logger's filters do not
    apply to records passed up from its children)."""
    for target in (logging.getLogger("httpx"), *logging.getLogger().handlers):
        if not any(isinstance(f, RedactToken) for f in target.filters):
            target.addFilter(RedactToken())


install_redaction()


# ── Telegram calls ───────────────────────────────────────────────────────────

def _client(timeout: float = 15.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _explain(status: int) -> str:
    if status in (401, 404):
        return "Telegram rejected the bot token. Check TELEGRAM_BOT_TOKEN."
    if status in (400, 403):
        return ("Telegram could not deliver to the chosen chat. Open your bot, press "
                "Start, then choose the chat again.")
    if status == 409:
        return "This bot's messages are already being read by another app or a webhook."
    return f"Telegram returned an error ({status})."


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.json()["parameters"]["retry_after"])
    except (ValueError, KeyError, TypeError):
        return 5.0


async def call(method: str, payload: dict, timeout: float = 15.0) -> object:
    """One Telegram call with a single retry for temporary failures."""
    if not CFG.telegram_bot_token:
        raise NotifyError("Telegram is not set up: TELEGRAM_BOT_TOKEN is empty.")
    url = f"{API_BASE}/bot{CFG.telegram_bot_token}/{method}"
    problem = "Could not reach Telegram."
    async with _client(timeout) as client:
        for attempt in range(2):
            wait = 2.0
            try:
                response = await client.post(url, json=payload)
            except httpx.HTTPError:  # its text can contain the URL, so it is dropped
                response = None
            if response is not None:
                if response.status_code == 200:
                    return response.json().get("result")
                problem = _explain(response.status_code)
                if response.status_code == 429:
                    problem = "Telegram is rate limiting this bot."
                    wait = _retry_after(response)
                    if wait > _MAX_RETRY_WAIT_S:
                        break
                elif response.status_code < 500:
                    break
            if attempt == 0:
                await _sleep(wait)
    raise NotifyError(problem)


def _cut(text: str, limit: int = _LIMIT) -> str:
    """Shorten to `limit` UTF-16 units, the way Telegram counts."""
    raw = text.encode("utf-16-le")
    if len(raw) <= limit * 2:
        return text
    return raw[:(limit - 1) * 2].decode("utf-16-le", errors="ignore") + "…"


async def send(text: str) -> int | None:
    """Returns the id of the message Telegram created."""
    chat_id = get_setting(CHAT_SETTING)
    if not CFG.telegram_bot_token or chat_id is None:
        raise NotifyError("Telegram is not set up.")
    sent = await call("sendMessage", {"chat_id": chat_id, "text": _cut(text),
                                      "link_preview_options": {"is_disabled": True}})
    return sent.get("message_id") if isinstance(sent, dict) else None


# ── what and when ────────────────────────────────────────────────────────────

def should_notify(routine: dict, run: dict, previous_status: str | None) -> bool:
    """Scheduled, sourced results only, and the first of a row of failures."""
    if routine.get("notify", "off") == "off" or run["trigger"] != "schedule":
        return False
    if run["status"] == "error":
        return previous_status != "error"
    return run["status"] == "ok"


def plain(text: str) -> str:
    text = _LINK_RE.sub(r"\1", text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    return re.sub(r"(?m)^\s*[*+-]\s+", "• ", text).strip()


def _headline(content: str) -> str:
    """First real line of the answer, without markdown, markers or links."""
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        text = agent.strip_markers(_URL_RE.sub("", plain(line)))
        text = " ".join(text.lstrip("•0123456789.) ").split())
        if text:
            return text if len(text) <= _HEADLINE_CHARS else text[:_HEADLINE_CHARS - 1] + "…"
    return ""


def _label(run: dict) -> str:
    if run["status"] == "error":
        return "The run failed. Open llm-app for details."
    n = len(run.get("sources") or [])
    return f"{n} source{'' if n == 1 else 's'}" if n else "Done"


def message(routine: dict, run: dict) -> str:
    """The notification text. A headline quotes the answer only when it came from
    public web results; library content leaves this machine only with `full`."""
    parts = [routine["name"], _label(run)]
    content = run.get("content") or ""
    if run["status"] == "error" or not content:
        return "\n".join(parts)
    sources = routine["sources"]
    if routine.get("notify") == "full":
        links = [f"[{s['marker']}] {s['title']} {s['url']}" for s in run.get("sources") or []
                 if str(s.get("url", "")).startswith(("http://", "https://"))]
        body = _cut(plain(content), 3000)
        return "\n\n".join(["\n".join(parts), body, *(["Sources:\n" + "\n".join(links)] if links else [])])
    if sources.get("web") and not sources.get("library"):
        parts.append(_headline(content))
    return "\n".join(p for p in parts if p)


# ── API ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/notify/telegram", tags=["notify"],
                   dependencies=[Depends(verify_api_key)])


class ChatChoice(BaseModel):
    chat_id: int

    model_config = {"extra": "forbid"}


class Options(BaseModel):
    chat: bool | None = None     # answer messages from the chosen chat
    library: bool | None = None  # allow the saved library in those answers

    model_config = {"extra": "forbid"}


CHAT_ENABLED, CHAT_LIBRARY = "telegram_chat_enabled", "telegram_library"


async def _private_chats() -> list[dict]:
    """People who have messaged the bot. Message text is never kept."""
    if get_setting(CHAT_ENABLED, False) and get_setting(CHAT_SETTING) is not None:
        # One reader per bot: the chat loop is already reading these.
        raise HTTPException(409, "Turn off “Chat from Telegram” to choose another chat.")
    try:
        updates = await call("getUpdates", {"limit": 100, "timeout": 0})
    except NotifyError as e:
        raise HTTPException(502, str(e)) from None
    found: dict[int, dict] = {}
    for update in updates if isinstance(updates, list) else []:
        chat = (update.get("message") or {}).get("chat") or {}
        if chat.get("type") == "private" and chat.get("id") not in found:
            found[chat["id"]] = {"id": chat["id"], "name": chat.get("first_name") or "",
                                 "username": chat.get("username") or ""}
    return list(found.values())


@router.get("/status")
async def status():
    token, chat = bool(CFG.telegram_bot_token), get_setting(CHAT_SETTING) is not None
    return {"token_set": token, "chat_set": chat, "ready": token and chat,
            "chat_enabled": bool(get_setting(CHAT_ENABLED, False)),
            "library": bool(get_setting(CHAT_LIBRARY, False))}


@router.post("/options")
async def set_options(req: Options):
    if req.chat and not (CFG.telegram_bot_token and get_setting(CHAT_SETTING) is not None):
        raise HTTPException(422, "Choose your Telegram chat first.")
    if req.chat is not None:
        set_setting(CHAT_ENABLED, req.chat)
    if req.library is not None:
        set_setting(CHAT_LIBRARY, req.library)
    return await status()


@router.get("/chats")
async def chats():
    return {"chats": await _private_chats()}


@router.post("/chat")
async def choose_chat(req: ChatChoice):
    if all(c["id"] != req.chat_id for c in await _private_chats()):
        raise HTTPException(422, "That chat has not messaged the bot recently.")
    set_setting(CHAT_SETTING, req.chat_id)
    return {"chat_set": True}


@router.post("/test")
async def send_test():
    try:
        await send("llm-app: test notification. Routine results will arrive here.")
    except NotifyError as e:
        raise HTTPException(502, str(e)) from None
    return {"sent": True}

"""Telegram notifications: what is sent, when, and that the token never leaks."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import notify

TOKEN = "123456789:AAExampleTokenForTests_abcdefghijk"
AUTH = {"Authorization": "Bearer test-key"}


class Server:
    def __init__(self):
        self.requests, self.replies = [], []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        r = self.replies.pop(0) if self.replies else httpx.Response(200, json={"ok": True, "result": {}})
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def tg(monkeypatch):
    s = Server()
    store: dict = {"telegram_chat_id": 42}
    monkeypatch.setattr(notify, "CFG", dataclasses.replace(
        notify.CFG, api_key="test-key", telegram_bot_token=TOKEN))
    monkeypatch.setattr("auth.CFG", notify.CFG)
    monkeypatch.setattr(notify, "get_setting", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(notify, "set_setting", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr(notify, "_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(s.handle)))

    async def no_sleep(_s):
        s.slept = _s
    monkeypatch.setattr(notify, "_sleep", no_sleep)
    s.store = store
    return s


def run(c):
    return asyncio.run(c)


def test_send_is_plain_text_without_previews(tg):
    run(notify.send("hello *x* _y_"))
    path, body = tg.requests[0]
    assert path.endswith("/sendMessage") and body["chat_id"] == 42
    assert body["text"] == "hello *x* _y_" and "parse_mode" not in body
    assert body["link_preview_options"] == {"is_disabled": True}


def test_long_text_is_cut_to_the_limit(tg):
    run(notify.send("x" * 9000))
    assert len(tg.requests[0][1]["text"]) <= 4096


def test_text_is_cut_in_utf16_units_as_telegram_counts(tg):
    run(notify.send("😀" * 4000))
    assert len(tg.requests[0][1]["text"].encode("utf-16-le")) // 2 <= 4096


@pytest.mark.parametrize("first", [
    httpx.Response(500, text="oops"), httpx.ConnectError("no route"),
    httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 3}}),
])
def test_one_retry_on_temporary_failure(tg, first):
    tg.replies = [first]
    run(notify.send("hi"))
    assert len(tg.requests) == 2


def test_gives_up_after_the_retry(tg):
    tg.replies = [httpx.Response(500), httpx.Response(500)]
    with pytest.raises(notify.NotifyError):
        run(notify.send("hi"))
    assert len(tg.requests) == 2


def test_long_rate_limit_is_not_waited_out(tg):
    tg.replies = [httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 900}})]
    with pytest.raises(notify.NotifyError):
        run(notify.send("hi"))
    assert len(tg.requests) == 1


@pytest.mark.parametrize("status,word", [(401, "token"), (404, "token"), (400, "chat"), (403, "chat")])
def test_permanent_errors_explain_and_do_not_retry(tg, status, word):
    tg.replies = [httpx.Response(status, json={"ok": False, "description": f"bad {TOKEN}"})]
    with pytest.raises(notify.NotifyError) as e:
        run(notify.send("hi"))
    assert word in str(e.value).lower() and TOKEN not in str(e.value) and len(tg.requests) == 1


def test_not_set_up(tg):
    tg.store.clear()
    with pytest.raises(notify.NotifyError):
        run(notify.send("hi"))
    assert tg.requests == []


def test_token_never_reaches_the_logs(tg, caplog):
    caplog.set_level(logging.DEBUG)
    tg.replies = [httpx.ConnectError(f"https://api.telegram.org/bot{TOKEN}/sendMessage")] * 2
    with pytest.raises(notify.NotifyError) as e:
        run(notify.send("hi"))
    run(notify.send("again"))
    assert "HTTP Request" in caplog.text      # httpx did log the call
    assert TOKEN not in caplog.text and "AAExample" not in caplog.text
    assert TOKEN not in str(e.value) and e.value.__cause__ is None


def test_redaction_covers_handlers_and_exceptions():
    rec = logging.LogRecord("httpx", logging.INFO, "f", 1, "HTTP Request: %s %s", ("POST", httpx.URL(
        f"https://api.telegram.org/bot{TOKEN}/getUpdates")), None)
    notify.RedactToken().filter(rec)
    assert "AAExample" not in rec.getMessage()


# ── messages ─────────────────────────────────────────────────────────────────

R = {"name": "AI news", "notify": "headline", "sources": {"web": True, "library": False}}
RUN = {"status": "ok", "content": "*   **Nvidia** fell 8% [W1]. See https://evil.example/x now\n* second",
       "sources": [{"marker": "W1", "title": "T", "url": "https://n.example/a"}]}


def test_headline_is_one_clean_line():
    text = notify.message(R, RUN)
    assert text.startswith("AI news\n") and "Nvidia fell 8%" in text
    assert "1 source" in text
    for bad in ("second", "**", "[W1]", "https://"):
        assert bad not in text


def test_headline_skips_headings_and_is_short():
    text = notify.message(R, {**RUN, "content": "# Digest\n\n" + "word " * 100})
    assert "Digest" not in text and len(text) < 260


def test_library_routines_never_send_content_as_a_headline():
    text = notify.message({**R, "sources": {"web": True, "library": True}}, RUN)
    assert "Nvidia" not in text and "AI news" in text


def test_prompt_only_routines_never_send_content_as_a_headline():
    assert "Nvidia" not in notify.message({**R, "sources": {"web": False, "library": False}}, RUN)


def test_full_includes_the_answer_and_sources():
    text = notify.message({**R, "notify": "full"}, RUN)
    assert "second" in text and "https://n.example/a" in text and "**" not in text


def test_unsourced_and_error_are_labelled():
    assert "unsourced" in notify.message(R, {**RUN, "status": "unsourced", "sources": []}).lower()
    err = notify.message(R, {"status": "error", "error": "ReadTimeout: secret detail"})
    assert "failed" in err.lower() and "secret detail" not in err


@pytest.mark.parametrize("mode,trigger,status,prev,want", [
    ("headline", "schedule", "ok", "ok", True),
    ("full", "schedule", "unsourced", "ok", True),
    ("off", "schedule", "ok", "ok", False),
    ("headline", "manual", "ok", "ok", False),
    ("headline", "schedule", "unchanged", "ok", False),
    ("headline", "schedule", "missed", "ok", False),
    ("headline", "schedule", "interrupted", "ok", False),
    ("headline", "schedule", "error", "ok", True),
    ("headline", "schedule", "error", "error", False),
    ("headline", "schedule", "error", None, True),
])
def test_when_to_notify(mode, trigger, status, prev, want):
    assert notify.should_notify({"notify": mode}, {"trigger": trigger, "status": status}, prev) is want


def test_old_routines_without_the_setting_do_not_notify():
    assert notify.should_notify({}, {"trigger": "schedule", "status": "ok"}, None) is False


# ── API ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def c(tg):
    app = FastAPI()
    app.include_router(notify.router)
    client = TestClient(app)
    client.tg = tg
    return client


@pytest.mark.parametrize("m,p", [("get", "status"), ("get", "chats"), ("post", "chat"), ("post", "test")])
def test_auth(c, m, p):
    assert getattr(c, m)(f"/v1/notify/telegram/{p}").status_code == 401


def test_status_never_returns_the_token(c):
    r = c.get("/v1/notify/telegram/status", headers=AUTH)
    assert r.json() == {"token_set": True, "chat_set": True, "ready": True} and TOKEN not in r.text


def test_find_chats_lists_private_chats_once(c):
    def upd(i, chat):
        return {"update_id": i, "message": {"chat": chat, "text": "secret words"}}
    c.tg.replies = [httpx.Response(200, json={"ok": True, "result": [
        upd(1, {"id": 7, "type": "private", "first_name": "Regan", "username": "rc"}),
        upd(2, {"id": 7, "type": "private", "first_name": "Regan"}),
        upd(3, {"id": -5, "type": "group", "title": "G"}), {"update_id": 4}]})]
    r = c.get("/v1/notify/telegram/chats", headers=AUTH)
    assert r.json() == {"chats": [{"id": 7, "name": "Regan", "username": "rc"}]}
    assert "secret words" not in r.text


def test_choosing_a_chat_must_be_one_that_messaged_the_bot(c):
    body = {"ok": True, "result": [{"message": {"chat": {"id": 7, "type": "private", "first_name": "R"}}}]}
    c.tg.replies = [httpx.Response(200, json=body), httpx.Response(200, json=body)]
    assert c.post("/v1/notify/telegram/chat", json={"chat_id": 9}, headers=AUTH).status_code == 422
    assert c.post("/v1/notify/telegram/chat", json={"chat_id": 7}, headers=AUTH).status_code == 200
    assert c.tg.store["telegram_chat_id"] == 7


def test_test_message_and_failure(c):
    assert c.post("/v1/notify/telegram/test", headers=AUTH).status_code == 200
    c.tg.replies = [httpx.Response(401, json={"ok": False})]
    r = c.post("/v1/notify/telegram/test", headers=AUTH)
    assert r.status_code == 502 and TOKEN not in r.text


def test_find_chats_conflict_is_explained(c):
    c.tg.replies = [httpx.Response(409, json={"ok": False, "description": "webhook"})]
    r = c.get("/v1/notify/telegram/chats", headers=AUTH)
    assert r.status_code == 502 and "another" in r.json()["detail"]

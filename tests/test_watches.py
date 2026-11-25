"""Watches: storage, schedule, alerts and API."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import safe_fetch
import watch_runner as wr
import watch_store as store
import watches
from safe_fetch import FetchError, Page

AUTH = {"Authorization": "Bearer test-key"}
NOW = datetime(2026, 11, 25, 20, 0, tzinfo=timezone.utc)
RULE = {"kind": "number", "label": "Price", "when": "below", "limit": 100}
BODY = {"name": "GPU", "url": "https://shop.example/gpu", "approved_host": "shop.example", "rule": RULE}


def run(c):
    return asyncio.run(c)


class Env:
    def __init__(self):
        self.pages, self.sent, self.fetched, self.fail_send = [], [], [], None


@pytest.fixture
def env(monkeypatch, tmp_path):
    e = Env()
    store.load(tmp_path)
    wr.reset()

    async def fetch(url, host):
        e.fetched.append((url, host))
        p = e.pages.pop(0) if len(e.pages) > 1 else e.pages[0]
        if isinstance(p, Exception):
            raise p
        return Page(url, "text/html", p)

    async def send(text):
        if e.fail_send:
            raise e.fail_send
        e.sent.append(text)

    async def pause():
        pass
    monkeypatch.setattr(safe_fetch, "fetch", fetch)
    monkeypatch.setattr(wr.notify, "send", send)
    monkeypatch.setattr(wr, "_pause", pause)
    monkeypatch.setattr("auth.CFG", dataclasses.replace(watches.CFG, api_key="test-key"))
    yield e
    wr.reset()


@pytest.fixture
def c(env):
    app = FastAPI()
    app.include_router(watches.router)
    return TestClient(app)


def make(**kw):
    return store.create({**BODY, "every_minutes": 60, "notify": True, "enabled": True,
                         "next_check_at": NOW.isoformat(), **kw})["id"]


def price(n):
    return f"<p>Price ${n}</p>"


def look(env, wid, html):
    env.pages = [html]
    run(wr.check(wid))
    return store.get(wid)


def test_first_check_sets_the_starting_point_with_two_loads(env):
    wid = make()
    w = look(env, wid, price(120))
    assert w["state"]["value"] == 120 and w["alerts"] == [] and len(env.fetched) == 2
    look(env, wid, price(119))
    assert len(env.fetched) == 3                      # nothing to report: one load


def test_crossing_alerts_once_with_the_exact_values(env):
    wid = make()
    look(env, wid, price(120))
    w = look(env, wid, price(99))
    look(env, wid, price(98))
    assert [a["text"] for a in w["alerts"]] == ["Price: $120 → $99 (below 100)"]
    assert env.sent == ["GPU\nPrice: $120 → $99 (below 100)\nhttps://shop.example/gpu"]
    assert store.get(wid)["alerts"][0]["notified"] == "sent" and store.unread_count() == 1


def test_a_reading_the_second_load_does_not_confirm_is_dropped(env):
    wid = make()
    look(env, wid, price(120))
    env.pages = [price(1), price(120)]
    run(wr.check(wid))
    assert store.get(wid)["alerts"] == [] and env.sent == []


def test_notifications_can_be_off(env):
    wid = make(notify=False)
    look(env, wid, price(120))
    w = look(env, wid, price(50))
    assert env.sent == [] and len(w["alerts"]) == 1


@pytest.mark.parametrize("error,text", [(wr.notify.NotifyError("Telegram is not set up."), "failed: Telegram is not set up."),
                                        (RuntimeError("https://api.telegram.org/bot1:secret"), "failed: RuntimeError")])
def test_failed_notification_keeps_the_alert_and_no_secret(env, error, text):
    wid = make()
    look(env, wid, price(120))
    env.fail_send = error
    assert look(env, wid, price(50))["alerts"][0]["notified"] == text


def test_failures_are_told_once_after_three_and_then_recover(env):
    wid = make()
    look(env, wid, price(120))
    for _ in range(5):
        look(env, wid, FetchError("The site answered with an error (503)."))
    w = store.get(wid)
    assert w["failures"] == 5 and len(env.sent) == 1 and "503" in env.sent[0] and w["state"]["value"] == 120
    w = look(env, wid, price(119))
    assert w["failures"] == 0 and w["error"] is None and len(env.sent) == 1


def test_unexpected_errors_are_contained(env):
    wid = make()
    w = look(env, wid, ValueError("boom"))
    assert "ValueError" in w["error"] and "boom" not in w["error"]


def test_deleted_or_edited_while_checking(env, monkeypatch):
    wid = make()
    env.pages = [price(1)]
    real = wr.look

    async def edit_meanwhile(w):
        out = await real(w)
        store.update(wid, {"rule": {**RULE, "label": "Other"}})
        return out
    monkeypatch.setattr(wr, "look", edit_meanwhile)
    run(wr.check(wid))
    assert store.get(wid)["state"] is None

    async def delete_meanwhile(w):
        store.delete(wid)
        raise FetchError("x")
    monkeypatch.setattr(wr, "look", delete_meanwhile)
    run(wr.check(wid))
    assert store.get(wid) is None
    run(wr.check("gone"))


# ── schedule ─────────────────────────────────────────────────────────────────

def test_tick_checks_due_ones_and_saves_the_next_time_first(env, monkeypatch):
    due, later, off = make(), make(next_check_at=(NOW + timedelta(hours=1)).isoformat()), make(enabled=False)
    seen = []

    async def check(wid):
        seen.append((wid, store.get(wid)["next_check_at"]))
    monkeypatch.setattr(wr, "check", check)
    monkeypatch.setattr(wr, "_now", lambda: NOW)
    run(wr.tick(NOW))
    assert seen == [(due, "2026-11-25T21:00:00+00:00")] and later and off


def test_one_bad_watch_does_not_stop_the_rest(env):
    a, b = make(), make()
    env.pages = [FetchError("down"), price(5), price(5)]
    run(wr.tick(NOW))
    got = [store.get(a), store.get(b)]
    assert sorted(w["error"] or "" for w in got) == ["", "down"]
    assert [w["state"]["value"] for w in got if w["state"]] == [5]


def test_loop_survives_errors_and_respects_the_switch(env, monkeypatch):
    calls = []

    async def bad(_now):
        calls.append(1)
        raise RuntimeError("x")
    monkeypatch.setattr(wr, "tick", bad)
    monkeypatch.setattr(wr, "TICK_S", 0.005)

    async def go():
        t = asyncio.create_task(wr.scheduler_loop())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    run(go())
    assert len(calls) > 1
    monkeypatch.setattr(wr, "CFG", dataclasses.replace(wr.CFG, watches_enabled=False))
    run(asyncio.wait_for(wr.scheduler_loop(), 1))


# ── storage ──────────────────────────────────────────────────────────────────

def test_survives_a_restart_and_skips_bad_files(env, tmp_path):
    wid = make()
    look(env, wid, price(120))
    (tmp_path / "bad.json").write_text("{")
    (tmp_path / "odd.json").write_text(json.dumps({"id": "other", "alerts": []}))
    store.load()
    assert store.get(wid)["state"]["value"] == 120 and store.count() == 1
    assert (tmp_path / "bad.json").read_text() == "{"
    assert (tmp_path / f"{wid}.json").stat().st_mode & 0o077 == 0


def test_alerts_are_bounded(env):
    wid = make()
    for i in range(40):
        store.add_alert(wid, str(i), "change")
    assert len(store.get(wid)["alerts"]) == store.MAX_ALERTS
    assert store.add_alert("gone", "x", "change") is None and store.update("gone", {}) is None


# ── API ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m,p", [("get", ""), ("post", ""), ("post", "/preview"), ("get", "/a"), ("patch", "/a"),
                                 ("delete", "/a"), ("post", "/a/check"), ("post", "/a/read")])
def test_auth_required(c, m, p):
    assert getattr(c, m)(f"/v1/watches{p}").status_code == 401


def test_create_check_read_delete(c, env):
    env.pages = [price(120)]
    r = c.post("/v1/watches", json=BODY, headers=AUTH)
    assert r.status_code == 201 and r.json()["next_check_at"] and "state" not in r.json()
    wid = r.json()["id"]
    assert c.post(f"/v1/watches/{wid}/check", headers=AUTH).json()["shown"] == "Price $120"
    env.pages = [price(90)]
    assert c.post(f"/v1/watches/{wid}/check", headers=AUTH).json()["unread"] == 1
    assert c.get("/v1/watches", headers=AUTH).json()["unread"] == 1
    c.post(f"/v1/watches/{wid}/read", headers=AUTH)
    assert c.get(f"/v1/watches/{wid}", headers=AUTH).json()["unread"] == 0
    assert c.delete(f"/v1/watches/{wid}", headers=AUTH).status_code == 200
    for m, p in (("get", ""), ("delete", ""), ("post", "/check"), ("post", "/read")):
        assert getattr(c, m)(f"/v1/watches/{wid}{p}", headers=AUTH).status_code == 404
    assert c.patch(f"/v1/watches/{wid}", json={"name": "x"}, headers=AUTH).status_code == 404


def test_the_site_must_be_approved(c, env):
    env.pages = [price(1)]
    r = c.post("/v1/watches", json={**BODY, "approved_host": "other.example"}, headers=AUTH)
    assert r.status_code == 422 and "shop.example" in r.json()["detail"]
    wid = c.post("/v1/watches", json=BODY, headers=AUTH).json()["id"]
    moved = {"url": "https://other.example/"}
    assert c.patch(f"/v1/watches/{wid}", json=moved, headers=AUTH).status_code == 422
    ok = c.patch(f"/v1/watches/{wid}", json={**moved, "approved_host": "other.example"}, headers=AUTH)
    assert ok.status_code == 200 and ok.json()["approved_host"] == "other.example"
    assert c.patch(f"/v1/watches/{wid}", json={"approved_host": "x.example"}, headers=AUTH).status_code == 422


@pytest.mark.parametrize("bad", [
    {"name": " "}, {"url": "file:///etc/passwd"}, {"url": "http://shop.example:8030/"}, {"every_minutes": 1},
    {"rule": {"kind": "number", "label": "x", "when": "below"}}, {"rule": {"kind": "number", "label": "x", "limit": 5}},
    {"rule": {"kind": "number", "label": "", "when": "changes"}}, {"rule": {"kind": "phrase", "phrase": ""}},
    {"rule": {"kind": "nope"}}, {"rule": {"kind": "change", "only": "x" * 121}}, {"extra": 1}, {"id": "x"},
    {"rule": {"kind": "number", "label": "x", "when": "below", "limit": "Infinity"}},
])
def test_validation(c, bad):
    assert c.post("/v1/watches", json={**BODY, **bad}, headers=AUTH).status_code == 422


def test_limit(c, env, monkeypatch):
    monkeypatch.setattr(watches, "MAX_WATCHES", 1)
    c.post("/v1/watches", json=BODY, headers=AUTH)
    assert c.post("/v1/watches", json=BODY, headers=AUTH).status_code == 409


def test_editing_what_is_watched_starts_over_and_other_edits_do_not(c, env):
    env.pages = [price(120)]
    wid = c.post("/v1/watches", json=BODY, headers=AUTH).json()["id"]
    c.post(f"/v1/watches/{wid}/check", headers=AUTH)
    c.patch(f"/v1/watches/{wid}", json={"name": "New", "rule": RULE, "every_minutes": 30}, headers=AUTH)
    assert store.get(wid)["state"] is not None
    c.patch(f"/v1/watches/{wid}", json={"rule": {**RULE, "limit": 50}}, headers=AUTH)
    assert store.get(wid)["state"] is None
    assert c.patch(f"/v1/watches/{wid}", json={"enabled": False}, headers=AUTH).json()["next_check_at"] is None
    assert c.patch(f"/v1/watches/{wid}", json={"enabled": True}, headers=AUTH).json()["next_check_at"]
    assert c.patch(f"/v1/watches/{wid}", json={"name": None}, headers=AUTH).status_code == 422


def test_preview_saves_nothing_and_explains_failures(c, env):
    env.pages = [price(88)]
    r = c.post("/v1/watches/preview", json={"url": BODY["url"], "rule": RULE}, headers=AUTH)
    assert r.json() == {"site": "shop.example", "shown": "Price $88", "sample": []} and store.count() == 0
    env.pages = [FetchError("The site answered with an error (403).")]
    r = c.post("/v1/watches/preview", json={"url": BODY["url"], "rule": RULE}, headers=AUTH)
    assert r.status_code == 422 and "403" in r.json()["detail"]
    assert c.post("/v1/watches/preview", json={"url": "ftp://x.example/", "rule": RULE}, headers=AUTH).status_code == 422

"""The morning briefing: what goes in, what is left out, and when it is sent."""
from __future__ import annotations

import asyncio
import dataclasses
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import briefing as bf
import calendar_feed
import routine_store
import watch_store

AUTH = {"Authorization": "Bearer test-key"}
NOW = datetime(2026, 11, 26, 13, 0, 5, tzinfo=timezone.utc)   # 7:00 in Chicago
SINCE = NOW - timedelta(hours=24)
ICS = ("BEGIN:VCALENDAR\nX-WR-TIMEZONE:America/Chicago\nBEGIN:VEVENT\nUID:a\nSUMMARY:Dentist\n"
       "DTSTART;TZID=America/Chicago:20261126T090000\nEND:VEVENT\nEND:VCALENDAR\n")
DIGEST = ("# Homelab triage — Thursday 26 November 2026, 06:15\n\nCovers x. **2 high, 1 medium.**\n\n"
          "## 🟠 axon — high\n\ntext\n\n## 🟠 delphi — high\n\n## 🟡 atmos — medium\n")


def run(c):
    return asyncio.run(c)


class Env:
    def __init__(self):
        self.settings, self.sent, self.ics, self.fail_send = {}, [], ICS, None


@pytest.fixture
def env(monkeypatch, tmp_path):
    e = Env()
    routine_store.load(tmp_path / "r")
    watch_store.load(tmp_path / "w")
    monkeypatch.setattr(bf, "_STATE", tmp_path / "briefing.json")
    monkeypatch.setattr(bf, "DIGEST", tmp_path / "latest.md")
    (tmp_path / "latest.md").write_text(DIGEST)
    os.utime(tmp_path / "latest.md", (NOW.timestamp() - 3600,) * 2)
    cfg = dataclasses.replace(bf.CFG, api_key="test-key", google_calendar_ics_url="https://calendar.google.com/calendar/ical/x/private-y/basic.ics",
                              telegram_bot_token="t", routine_timezone="America/Chicago")
    monkeypatch.setattr(bf, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)
    monkeypatch.setattr(bf, "get_setting", lambda k, d=None: e.settings.get(k, d))
    monkeypatch.setattr(bf, "set_setting", lambda k, v: e.settings.__setitem__(k, v))

    async def fetch(url):
        if isinstance(e.ics, Exception):
            raise e.ics
        return e.ics

    async def send(text):
        if e.fail_send:
            raise e.fail_send
        e.sent.append(text)
        return 1

    async def no_sleep(_s):
        return None
    monkeypatch.setattr(calendar_feed, "fetch", fetch)
    monkeypatch.setattr(bf.notify, "send", send)
    monkeypatch.setattr(bf.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(bf, "_now", lambda: NOW)
    return e


def compose():
    return run(bf.compose(NOW, SINCE))


def routine(name="Morning AI News", sources=None, **run_fields):
    r = routine_store.create({"name": name, "prompt": "p", "query": None, "timezone": "UTC", "enabled": True,
                              "schedule": {"kind": "interval", "hours": 24}, "next_run_at": None,
                              "sources": sources or {"web": True, "library": False}, "notify": "off"})
    if run_fields:
        x = routine_store.start_run(r["id"], "schedule")
        routine_store.finish_run(r["id"], x["id"], {"content": "Nvidia fell [W1].", "sources": [], **run_fields})
        routine_store.annotate_run(r["id"], x["id"], {"started_at": run_fields.get("started_at", NOW.isoformat())})
    return r


def test_everything_in_one_message(env):
    routine(status="ok")
    w = watch_store.create({"name": "GPU", "alerts": []})
    watch_store.add_alert(w["id"], "Price: $649 → $599 (below 600)", "change")
    text = compose()
    assert text.startswith("Good morning. Thursday, November 26")
    assert "Calendar:\n• 9:00am Dentist" in text
    assert "Watches:\n• GPU: Price: $649 → $599 (below 600)" in text
    assert "Routines:\n• Morning AI News: Nvidia fell." in text
    assert "Homelab: 2 high, 1 medium.\nNeeds a look: axon, delphi" in text


def test_only_what_is_new_since_the_last_briefing(env):
    routine(status="ok", started_at=(SINCE - timedelta(hours=1)).isoformat())
    w = watch_store.create({"name": "GPU", "alerts": []})
    watch_store.add_alert(w["id"], "old", "change")
    watch_store.update(w["id"], {"alerts": [{**watch_store.get(w["id"])["alerts"][0], "at": SINCE.isoformat()}]})
    text = compose()
    assert "Routines" not in text and "Watches" not in text


def test_library_results_are_counted_not_quoted(env):
    routine(sources={"web": True, "library": True}, status="ok")
    assert "Morning AI News: a new result" in compose() and "Nvidia" not in compose()


@pytest.mark.parametrize("status,word", [("unsourced", "needs checking"), ("error", "failed")])
def test_problem_runs_are_named_not_quoted(env, status, word):
    routine(status=status)
    text = compose()
    assert word in text and "Nvidia" not in text


def test_unchanged_runs_are_left_out(env):
    routine(status="unchanged")
    assert "Routines" not in compose()


def test_long_alerts_are_cut(env):
    w = watch_store.create({"name": "Page", "alerts": []})
    watch_store.add_alert(w["id"], "+ a\n+ b\n+ c\n+ d", "change")
    for i in range(7):
        watch_store.add_alert(w["id"], f"alert {i}", "change")
    text = compose()
    assert "and 3 more" in text


@pytest.mark.parametrize("ics,want", [
    (calendar_feed.CalendarError("Google Calendar answered with an error (404)."), "couldn't be read. Google"),
    (RuntimeError("https://calendar.google.com/calendar/ical/x/private-y"), "Calendar: couldn't be read."),
    ("BEGIN:VCALENDAR\nEND:VCALENDAR", "Calendar: nothing today."),
])
def test_calendar_problems_are_said_without_the_address(env, ics, want):
    env.ics = ics
    text = compose()
    assert want in text and "private-y" not in text


def test_no_calendar_set_up_means_no_calendar_section(env, monkeypatch):
    monkeypatch.setattr(bf, "CFG", dataclasses.replace(bf.CFG, google_calendar_ics_url=""))
    assert "Calendar" not in compose()


def test_homelab_digest_missing_old_or_odd(env, tmp_path):
    (tmp_path / "latest.md").unlink()
    assert "Homelab" not in compose()
    (tmp_path / "latest.md").write_text(DIGEST)
    os.utime(tmp_path / "latest.md", (NOW.timestamp() - 3 * 86400,) * 2)
    assert "out of date" in compose()
    (tmp_path / "latest.md").write_text("# Homelab triage\n\nNothing to report.\n")
    os.utime(tmp_path / "latest.md", (NOW.timestamp(),) * 2)
    assert "Homelab: nothing to report." in compose()
    many = "**9 high.**\n" + "".join(f"## 🔴 p{i} — critical\n" for i in range(9))
    (tmp_path / "latest.md").write_text(many)
    assert "and 3 more" in compose()


def test_a_quiet_morning_still_says_so(env, tmp_path, monkeypatch):
    (tmp_path / "latest.md").unlink()
    monkeypatch.setattr(bf, "CFG", dataclasses.replace(bf.CFG, google_calendar_ics_url=""))
    assert compose().endswith("Nothing new since the last briefing.")


# ── schedule ─────────────────────────────────────────────────────────────────

def enable(env, **kw):
    env.settings["briefing"] = {"enabled": True, "time": "07:00", **kw}


def test_tick_sends_once_at_seven_and_remembers_where_it_got_to(env):
    enable(env)
    run(bf.tick(NOW - timedelta(minutes=5)))                       # sets the next time
    assert env.sent == [] and bf._load()["next_at"] == "2026-11-26T13:00:00+00:00"
    run(bf.tick(NOW))
    run(bf.tick(NOW + timedelta(seconds=30)))
    state = bf._load()
    assert len(env.sent) == 1 and state["next_at"] == "2026-11-27T13:00:00+00:00"
    assert state["last"]["notified"] == "sent" and state["last_at"] == NOW.isoformat(timespec="seconds")


def test_late_briefings_are_skipped_not_sent(env):
    enable(env)
    bf._save({"next_at": (NOW - timedelta(hours=2)).isoformat()})
    run(bf.tick(NOW))
    assert env.sent == [] and bf._load()["next_at"] == "2026-11-27T13:00:00+00:00"


def test_off_by_default_and_when_turned_off(env):
    run(bf.tick(NOW))
    assert env.sent == [] and bf._load().get("next_at") is None
    enable(env, days=[0])
    run(bf.tick(NOW))
    assert bf._load()["next_at"] == "2026-11-30T13:00:00+00:00"   # next Monday
    env.settings["briefing"]["enabled"] = False
    run(bf.tick(NOW))
    assert bf._load()["next_at"] is None


def test_waits_for_routines_due_at_the_same_time(env, monkeypatch):
    r = routine()
    routine_store.update(r["id"], {"next_run_at": NOW.isoformat()})
    calls = []

    async def sleep(s):
        calls.append(s)
        if len(calls) == 3:
            routine_store.update(r["id"], {"next_run_at": (NOW + timedelta(days=1)).isoformat()})
    monkeypatch.setattr(bf.asyncio, "sleep", sleep)
    run(bf._routines_settled(NOW))
    assert calls == [1, 5, 5]


def test_waiting_has_a_limit(env, monkeypatch):
    bf.routine_runner._active.add("stuck")
    try:
        waits = []

        async def sleep(s):
            waits.append(s)
        monkeypatch.setattr(bf.asyncio, "sleep", sleep)
        run(bf._routines_settled(NOW))
        assert sum(waits) - 1 == bf._WAIT_FOR_ROUTINES_S
    finally:
        bf.routine_runner._active.discard("stuck")


@pytest.mark.parametrize("error,word", [(bf.notify.NotifyError("Telegram is not set up."), "not sent: Telegram"),
                                        (RuntimeError("https://api.telegram.org/bot1:secret"), "not sent: RuntimeError")])
def test_a_failed_send_is_recorded_without_secrets(env, error, word):
    env.fail_send = error
    last = run(bf.send_now("manual"))
    assert last["notified"].startswith(word) and "secret" not in str(bf._load())


def test_state_file_is_private(env, tmp_path):
    run(bf.send_now("manual"))
    assert (tmp_path / "briefing.json").stat().st_mode & 0o077 == 0


def test_loop_survives_errors(env, monkeypatch):
    n = []

    async def bad(_now):
        n.append(1)
        if len(n) > 2:
            raise asyncio.CancelledError
        raise RuntimeError("x")
    monkeypatch.setattr(bf, "tick", bad)
    with pytest.raises(asyncio.CancelledError):
        run(bf.scheduler_loop())
    assert len(n) == 3


# ── API ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def c(env):
    app = FastAPI()
    app.include_router(bf.router)
    return TestClient(app)


@pytest.mark.parametrize("m,p", [("get", ""), ("patch", ""), ("post", "/preview"), ("post", "/send")])
def test_auth_required(c, m, p):
    assert getattr(c, m)(f"/v1/briefing{p}").status_code == 401


def test_settings_status_preview_and_send(c, env):
    s = c.get("/v1/briefing", headers=AUTH).json()
    assert s["enabled"] is False and s["time"] == "07:00" and s["calendar_set"] and s["homelab_digest"]
    assert "private" not in str(s)
    s = c.patch("/v1/briefing", json={"enabled": True, "time": "06:45", "days": [4, 0, 4]}, headers=AUTH).json()
    assert s["next_at"] == "2026-11-27T12:45:00+00:00" and s["days"] == [0, 4]
    assert "Dentist" in c.post("/v1/briefing/preview", headers=AUTH).json()["text"] and env.sent == []
    assert bf._load().get("last") is None
    assert c.post("/v1/briefing/send", headers=AUTH).json()["notified"] == "sent" and len(env.sent) == 1


@pytest.mark.parametrize("bad", [{"time": "7:00"}, {"time": "24:00"}, {"days": []}, {"days": [7]},
                                 {"timezone": "Mars/Base"}, {"extra": 1}])
def test_validation(c, bad):
    assert c.patch("/v1/briefing", json=bad, headers=AUTH).status_code == 422

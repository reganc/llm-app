"""/v1/actions: the web panel lists proposals and can reject them, never approve."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import action_cards
import actions
import actions_api
import routine_store as store

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def c(monkeypatch, tmp_path):
    store.load(tmp_path / "r")
    actions.load(tmp_path / "a")
    monkeypatch.setattr("auth.CFG", dataclasses.replace(actions.CFG, api_key="test-key"))
    refreshed = []

    async def refresh(record):
        refreshed.append((record["id"], record["status"]))
    monkeypatch.setattr(action_cards, "refresh", refresh)
    app = FastAPI()
    app.include_router(actions_api.router)
    client = TestClient(app)
    client.refreshed = refreshed
    return client


def proposal(name="Morning AI News", **kw):
    store.create({"name": name, "prompt": "p", "query": None, "timezone": "UTC",
                  "schedule": {"kind": "interval", "hours": 1}, "enabled": True, "next_run_at": None,
                  "sources": {"web": True, "library": False}, "notify": "off"})
    return actions.propose("run_routine", {"target": name}, reason="asked <b>nicely</b>",
                           source=kw.get("source", "telegram"), after_reading=False,
                           client=kw.get("client", ""))


def test_needs_the_key(c):
    assert c.get("/v1/actions").status_code == 401


def test_lists_newest_first_with_codes_description_apart_from_the_models_words(c, monkeypatch):
    clock = [1_800_000_000.0]
    monkeypatch.setattr(actions, "_now", lambda: clock[0])
    a = proposal("A")
    clock[0] += 5
    b = proposal("B", source="mcp", client="claude-code")
    data = c.get("/v1/actions", headers=AUTH).json()
    assert data["waiting"] == 2
    assert [r["id"] for r in data["actions"]] == [b["id"], a["id"]]
    row = data["actions"][0]
    assert row["title"] == "Run the routine “B” now" and row["reason"] == "asked <b>nicely</b>"
    assert (row["source"], row["client"], row["status"], row["risk"]) == ("mcp", "claude-code", "pending", "low")


def test_reject_from_the_web_updates_the_phone_card_and_is_logged(c):
    p = proposal()
    r = c.post(f"/v1/actions/{p['id']}/reject", headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert c.refreshed == [(p["id"], "rejected")]
    assert actions.action_log.tail()[-1]["event"] == {**actions.action_log.tail()[-1]["event"],
                                                     "kind": "rejected", "by": "web"}
    again = c.post(f"/v1/actions/{p['id']}/reject", headers=AUTH)
    assert again.status_code == 409 and again.json()["detail"] == "Already rejected."
    assert c.post("/v1/actions/0123456789ab/reject", headers=AUTH).status_code == 404


def test_there_is_no_way_to_approve_from_the_web(c):
    p = proposal()
    for path in (f"/v1/actions/{p['id']}/approve", f"/v1/actions/{p['id']}/confirm", f"/v1/actions/{p['id']}"):
        assert c.post(path, headers=AUTH).status_code in (404, 405)
    assert actions.get(p["id"])["status"] == "pending"


def test_listing_expires_old_proposals_and_updates_their_cards(c, monkeypatch):
    p = proposal()
    later = p["expires_at"] + 1
    monkeypatch.setattr(actions, "_now", lambda: later)
    data = c.get("/v1/actions", headers=AUTH).json()
    assert data["waiting"] == 0 and data["actions"][0]["status"] == "expired"
    assert c.refreshed == [(p["id"], "expired")]

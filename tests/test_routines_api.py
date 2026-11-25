"""/v1/routines endpoints."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routine_runner as rr
import routine_store as store
import routines

AUTH = {"Authorization": "Bearer test-key"}
BODY = {"name": "AI news", "prompt": "Summarize AI news with sources",
        "schedule": {"kind": "daily", "time": "07:00", "days": [0, 1, 2, 3, 4]}}


@pytest.fixture
def c(monkeypatch, tmp_path):
    store.load(tmp_path)
    rr.reset()
    cfg = dataclasses.replace(routines.CFG, api_key="test-key")
    monkeypatch.setattr("auth.CFG", cfg)
    app = FastAPI()
    app.include_router(routines.router)
    yield TestClient(app)
    rr.reset()


@pytest.mark.parametrize("m,p", [("get", ""), ("post", ""), ("get", "/inbox"), ("post", "/inbox/read"),
                                 ("get", "/abc"), ("patch", "/abc"), ("delete", "/abc"),
                                 ("post", "/abc/run"), ("post", "/abc/runs/x/read")])
def test_auth_required(c, m, p):
    assert getattr(c, m)(f"/v1/routines{p}").status_code == 401


def test_create_list_get_update_delete(c):
    r = c.post("/v1/routines", json=BODY, headers=AUTH)
    assert r.status_code == 201
    got = r.json()
    assert got["sources"] == {"web": True, "library": False, "news": False} and got["enabled"] is True
    assert got["timezone"] and got["next_run_at"]
    rid = got["id"]
    lst = c.get("/v1/routines", headers=AUTH).json()
    assert lst["routines"][0]["id"] == rid and "runs" not in lst["routines"][0]
    assert lst["unread"] == 0
    assert c.get(f"/v1/routines/{rid}", headers=AUTH).json()["runs"] == []
    before = got["next_run_at"]
    r = c.patch(f"/v1/routines/{rid}", json={"schedule": {"kind": "interval", "hours": 3}}, headers=AUTH)
    assert r.status_code == 200 and r.json()["next_run_at"] != before
    r = c.patch(f"/v1/routines/{rid}", json={"name": "New"}, headers=AUTH)
    assert r.json()["name"] == "New" and r.json()["prompt"] == BODY["prompt"]
    assert c.patch(f"/v1/routines/{rid}", json={"enabled": False}, headers=AUTH).json()["next_run_at"] is None
    assert c.patch(f"/v1/routines/{rid}", json={"enabled": True}, headers=AUTH).json()["next_run_at"]
    assert c.delete(f"/v1/routines/{rid}", headers=AUTH).status_code == 200
    assert c.get(f"/v1/routines/{rid}", headers=AUTH).status_code == 404


@pytest.mark.parametrize("patch", [
    {"name": ""}, {"name": "  "}, {"name": "x" * 81}, {"prompt": ""}, {"prompt": "x" * 2001},
    {"query": "x" * 301}, {"timezone": "Mars/Base"}, {"schedule": {"kind": "daily", "time": "25:00"}},
    {"surprise": 1}, {"sources": {"web": True, "shell": True}},
])
def test_validation(c, patch):
    assert c.post("/v1/routines", json={**BODY, **patch}, headers=AUTH).status_code == 422
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    assert c.patch(f"/v1/routines/{rid}", json=patch, headers=AUTH).status_code == 422


@pytest.mark.parametrize("field", ["name", "prompt", "schedule", "sources", "enabled", "timezone"])
def test_patch_rejects_null_for_required_fields(c, field):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    assert c.patch(f"/v1/routines/{rid}", json={field: None}, headers=AUTH).status_code == 422


def test_patch_can_clear_the_query(c):
    rid = c.post("/v1/routines", json={**BODY, "query": "q"}, headers=AUTH).json()["id"]
    assert c.patch(f"/v1/routines/{rid}", json={"query": None}, headers=AUTH).json()["query"] is None


def test_limit_on_number_of_routines(c, monkeypatch):
    monkeypatch.setattr(routines, "MAX_ROUTINES", 1)
    assert c.post("/v1/routines", json=BODY, headers=AUTH).status_code == 201
    assert c.post("/v1/routines", json=BODY, headers=AUTH).status_code == 409


def test_run_now_and_conflict(c, monkeypatch):
    started = []

    def fake(rid):
        if started:
            return False
        started.append(rid)
        return True
    monkeypatch.setattr(rr, "start_manual", fake)
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    assert c.post(f"/v1/routines/{rid}/run", headers=AUTH).status_code == 202
    assert c.post(f"/v1/routines/{rid}/run", headers=AUTH).status_code == 409
    assert c.post("/v1/routines/nope/run", headers=AUTH).status_code == 404


def test_inbox_and_read(c):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    run = store.start_run(rid, "manual")
    store.finish_run(rid, run["id"], {"status": "ok", "content": "hello", "sources": []})
    box = c.get("/v1/routines/inbox", headers=AUTH).json()
    assert box["unread"] == 1 and box["runs"][0]["content"] == "hello"
    assert box["runs"][0]["routine_name"] == "AI news"
    assert c.post(f"/v1/routines/{rid}/runs/{run['id']}/read", headers=AUTH).status_code == 200
    assert c.post(f"/v1/routines/{rid}/runs/nope/read", headers=AUTH).status_code == 404
    assert c.get("/v1/routines/inbox", headers=AUTH).json()["unread"] == 0
    assert c.post("/v1/routines/inbox/read", headers=AUTH).json() == {"marked": 0}
    assert c.get("/v1/routines/inbox?limit=0", headers=AUTH).status_code == 422

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
    assert got["sources"] == {"web": True, "library": False, "news": False, "extra_sites": False} and got["enabled"] is True
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


def test_new_options_round_trip_and_validate(c):
    r = c.post("/v1/routines", json={**BODY, "only_new": True, "notify": "headline"}, headers=AUTH).json()
    assert r["only_new"] is True and r["notify"] == "headline" and "memory" not in r
    d = c.post("/v1/routines", json=BODY, headers=AUTH).json()
    assert d["only_new"] is False and d["notify"] == "off"
    for bad in ({"notify": "sms"}, {"only_new": "maybe"}):
        assert c.post("/v1/routines", json={**BODY, **bad}, headers=AUTH).status_code == 422
    for f in ("only_new", "notify"):
        assert c.patch(f"/v1/routines/{r['id']}", json={f: None}, headers=AUTH).status_code == 422


@pytest.mark.parametrize("patch,kept", [
    ({"name": "x"}, True), ({"notify": "off"}, True), ({"enabled": False}, True),
    ({"schedule": {"kind": "interval", "hours": 2}}, True),
    ({"prompt": "Other"}, False), ({"query": "other"}, False),
    ({"sources": {"web": False}}, False), ({"prompt": BODY["prompt"]}, True),
])
def test_changing_the_task_forgets_earlier_reports(c, patch, kept):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    store.update(rid, {"memory": {"urls": ["u"], "reports": []}})
    c.patch(f"/v1/routines/{rid}", json=patch, headers=AUTH)
    assert bool(store.get(rid).get("memory")) is kept


def test_memory_is_private_and_can_be_cleared(c):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    store.update(rid, {"memory": {"urls": ["u"], "reports": []}})
    got = c.get(f"/v1/routines/{rid}", headers=AUTH).json()
    assert "memory" not in got and got["remembers"] == 1
    assert c.post(f"/v1/routines/{rid}/forget", headers=AUTH).status_code == 200
    assert c.get(f"/v1/routines/{rid}", headers=AUTH).json()["remembers"] == 0
    assert c.post("/v1/routines/nope/forget", headers=AUTH).status_code == 404


def test_old_routine_files_still_work(c):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    old = {k: v for k, v in store._routines[rid].items() if k not in ("only_new", "notify", "memory")}
    store._routines[rid] = old
    got = c.get("/v1/routines", headers=AUTH).json()["routines"][0]
    assert got["only_new"] is False and got["notify"] == "off"


def test_draft_endpoint(c, monkeypatch):
    async def fake(text):
        return {"routine": {"name": text}, "warnings": []}
    monkeypatch.setattr(routines.routine_draft, "draft", fake)
    assert c.post("/v1/routines/draft", json={"text": "every day"}).status_code == 401
    assert c.post("/v1/routines/draft", json={"text": " "}, headers=AUTH).status_code == 422
    assert c.post("/v1/routines/draft", json={"text": "x" * 2001}, headers=AUTH).status_code == 422
    r = c.post("/v1/routines/draft", json={"text": "every day do x"}, headers=AUTH)
    assert r.json()["routine"]["name"] == "every day do x"
    assert c.get("/v1/routines", headers=AUTH).json()["routines"] == []   # nothing saved


def test_search_phrase_is_reported(c):
    got = c.post("/v1/routines", json=BODY, headers=AUTH).json()
    assert got["query"] is None and got["search_phrase"] == "AI news"
    url = f"/v1/routines/{got['id']}"
    assert c.patch(url, json={"query": "openai"}, headers=AUTH).json()["search_phrase"] == "openai"
    off = c.patch(url, json={"sources": {"web": False}}, headers=AUTH).json()
    assert off["search_phrase"] is None


def test_saving_a_routine_from_before_extra_sites_keeps_its_memory(c):
    rid = c.post("/v1/routines", json=BODY, headers=AUTH).json()["id"]
    old = {"web": True, "library": False, "news": True}
    store.update(rid, {"sources": old})
    store.set_memory(rid, {"urls": ["n.example/a"], "titles": [], "reports": []})
    url = f"/v1/routines/{rid}"
    same = c.patch(url, json={"sources": {**old, "extra_sites": False}}, headers=AUTH).json()
    assert same["remembers"] == 1
    changed = c.patch(url, json={"sources": {**old, "extra_sites": True}}, headers=AUTH).json()
    assert changed["remembers"] == 0

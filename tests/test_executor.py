"""The executor (executor/main.py): only listed apps, only with the token, one at a
time, slowly, containers found by compose label, health checked, every attempt
logged. Docker is a fake: nothing real is restarted."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

_SPEC = importlib.util.spec_from_file_location(
    "executor_main", Path(__file__).resolve().parent.parent / "executor" / "main.py")
ex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ex)

TOKEN = "t" * 64
FLEET = """
apps:
  - id: mentat
    title: Mentat
    health: http://host.docker.internal:8400/health
    privacy: home
    restart: {project: mentat, services: [api, worker]}
  - id: tax-planner
    privacy: private
    restart: {project: tax-planner, services: [app]}
  - id: nexus
    privacy: home
    restart: {project: nexus, services: ["core; rm -rf /"]}
  - id: delphi
    privacy: home
    health: http://10.0.0.5:8730/health
    restart: {project: delphi, services: [delphi-api]}
"""


def container(cid, project, service):
    return {"Id": cid, "Names": [f"/{project}-{service}-1"],
            "Labels": {"com.docker.compose.project": project, "com.docker.compose.service": service}}


class FakeDocker:
    def __init__(self):
        self.listed = [container("c1", "mentat", "api"), container("c2", "mentat", "worker"),
                       container("c3", "mentat", "postgres"), container("x9", "other", "api")]
        self.restarted, self.queries = [], []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/containers/json":
            self.queries.append(dict(request.url.params))
            return httpx.Response(200, json=self.listed)
        if request.method == "POST" and request.url.path.endswith("/restart"):
            self.restarted.append(request.url.path.split("/")[2])
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def env(monkeypatch, tmp_path):
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(FLEET)
    monkeypatch.setattr(ex, "TOKEN", TOKEN)
    monkeypatch.setattr(ex, "FLEET_FILE", fleet)
    monkeypatch.setattr(ex, "DATA_DIR", tmp_path / "data")
    docker = FakeDocker()
    monkeypatch.setattr(ex, "docker_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(docker.handler), base_url="http://docker"))
    health = {"state": "ok", "urls": []}

    async def wait(url, **_):
        health["urls"].append(url)
        return "none" if url is None else health["state"]
    monkeypatch.setattr(ex, "_wait_healthy", wait)
    docker.health = health
    return docker


def post(app="mentat", token=TOKEN, action_id="0123456789ab"):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return TestClient(ex.app).post("/restart", json={"app": app, "action_id": action_id}, headers=headers)


def test_only_home_apps_with_safe_names_are_restartable(env):
    found = ex.restartable()
    assert found == {"mentat": {"project": "mentat", "services": ["api", "worker"],
                                "health": "http://host.docker.internal:8400/health"},
                     "delphi": {"project": "delphi", "services": ["delphi-api"], "health": None}}


def test_token_is_required(env, monkeypatch):
    assert post(token=None).status_code == 401
    assert post(token="wrong").status_code == 401
    monkeypatch.setattr(ex, "TOKEN", "")
    assert post().status_code == 503
    assert env.restarted == []


def test_unlisted_or_private_apps_are_refused(env):
    for app in ("tax-planner", "nexus", "ollama"):
        assert post(app).status_code == 403
    assert post("Mentat").status_code == 422          # ids only, exact
    assert TestClient(ex.app).post("/restart", json={"app": "mentat", "action_id": "x"},
                                   headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 422
    assert env.restarted == []


def test_restarts_only_the_listed_services_in_order_then_checks_health(env):
    r = post()
    assert r.status_code == 200
    assert r.json() == {"ok": True, "containers": ["mentat-api-1", "mentat-worker-1"], "health": "ok",
                        "seconds": 0, "problem": None}
    assert env.restarted == ["c1", "c2"]                   # never postgres, never another project
    assert json.loads(env.queries[0]["filters"]) == {"label": ["com.docker.compose.project=mentat"]}
    assert env.health["urls"] == ["http://host.docker.internal:8400/health"]


def test_cooldown_and_daily_cap_from_its_own_log(env, monkeypatch):
    assert post().status_code == 200
    second = post()
    assert second.status_code == 429 and "less than 10 minutes" in second.json()["detail"]
    clock = [ex.time.time()]
    monkeypatch.setattr(ex.time, "time", lambda: clock[0])
    for _ in range(5):
        clock[0] += ex.COOLDOWN_S + 1
        assert post().status_code == 200
    clock[0] += ex.COOLDOWN_S + 1
    capped = post()
    assert capped.status_code == 429 and "6 restarts a day" in capped.json()["detail"]


def test_a_missing_service_restarts_nothing(env):
    env.listed = [container("c1", "mentat", "api")]
    r = post()
    assert r.json()["ok"] is False and "no container for worker" in r.json()["problem"]
    assert env.restarted == []


def test_failed_health_check_is_reported(env):
    env.health["state"] = "failed"
    body = post().json()
    assert body["ok"] is False and body["containers"] == ["mentat-api-1", "mentat-worker-1"]
    assert "health check within 90 s" in body["problem"]


def test_every_attempt_is_logged_in_a_chain(env):
    post()
    rows = [json.loads(line) for line in (ex.DATA_DIR / "restarts.jsonl").read_text().splitlines()]
    assert [r["event"]["kind"] for r in rows] == ["attempt", "restarted"]
    assert rows[0]["event"]["action_id"] == "0123456789ab"
    prev = ex.GENESIS
    for row in rows:
        payload = {k: row[k] for k in ("seq", "at", "ts", "event")}
        assert row["prev_hash"] == prev and row["hash"] == ex._digest(prev, payload)
        prev = row["hash"]


def test_health_endpoint_says_whether_it_is_configured(env, monkeypatch):
    assert TestClient(ex.app).get("/health").json() == {"ok": True, "configured": True}
    monkeypatch.setattr(ex, "TOKEN", "")
    assert TestClient(ex.app).get("/health").json()["configured"] is False

"""The MCP endpoint: auth and origin, the JSON-RPC handshake, exactly four tools
(none that runs or approves), and proposals that only send the user a card."""
from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import action_cards
import actions
import hub
import mcp_server
import routine_store as store
import watch_store

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def c(monkeypatch, tmp_path):
    store.load(tmp_path / "r")
    watch_store.load(tmp_path / "w")
    actions.load(tmp_path / "a")
    monkeypatch.setattr("auth.CFG", dataclasses.replace(actions.CFG, api_key="test-key"))
    monkeypatch.setattr(mcp_server.telegram_chat, "active", lambda: True)
    monkeypatch.setattr(hub, "load_fleet", lambda *a: [])
    cards = []

    async def send(record):
        cards.append(record)
        actions.set_message(record["id"], 901)
        return True
    monkeypatch.setattr(action_cards, "send", send)
    mcp_server._SESSIONS.clear()
    app = FastAPI()
    app.include_router(mcp_server.router)
    client = TestClient(app)
    client.cards = cards
    return client


def rpc(c, method, params=None, rid=1, headers=None):
    body = {"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params else {})}
    return c.post("/mcp", json=body, headers={**AUTH, **(headers or {})})


def call(c, name, arguments=None, headers=None):
    r = rpc(c, "tools/call", {"name": name, "arguments": arguments or {}}, headers=headers)
    assert r.status_code == 200
    return r.json()["result"]


def routine(name="Morning AI News"):
    return store.create({"name": name, "prompt": "p", "query": None, "timezone": "UTC",
                         "schedule": {"kind": "interval", "hours": 1}, "enabled": True,
                         "next_run_at": None, "sources": {"web": True, "library": False},
                         "notify": "off"})


def test_needs_the_api_key_and_a_local_origin(c):
    assert c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    evil = rpc(c, "ping", headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403
    assert rpc(c, "ping", headers={"Origin": "http://localhost:3000"}).json()["result"] == {}


def test_initialize_negotiates_a_version_and_names_the_session(c):
    r = rpc(c, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                              "clientInfo": {"name": "claude-code", "version": "2"}})
    body = r.json()["result"]
    assert body["protocolVersion"] == "2025-03-26"
    assert body["capabilities"] == {"tools": {"listChanged": False}}
    assert "nothing happens unless they approve" in body["instructions"]
    assert mcp_server._SESSIONS[r.headers["Mcp-Session-Id"]] == "claude-code"
    assert rpc(c, "initialize", {"protocolVersion": "1999-01-01"}).json()["result"]["protocolVersion"] \
        == mcp_server.PROTOCOLS[0]


def test_notifications_get_202_and_bad_messages_errors(c):
    r = c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=AUTH)
    assert r.status_code == 202 and r.content == b""
    assert c.post("/mcp", content=b"{nope", headers=AUTH).json()["error"]["code"] == -32700
    assert c.post("/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}],
                  headers=AUTH).json()["error"]["code"] == -32600
    assert rpc(c, "resources/list").json()["error"]["code"] == -32601
    assert rpc(c, "tools/call", {"name": "run_shell"}).json()["error"]["code"] == -32602
    assert c.get("/mcp", headers=AUTH).status_code == 405


def test_exactly_four_tools_none_that_runs_or_approves(c):
    names = [t["name"] for t in rpc(c, "tools/list").json()["result"]["tools"]]
    assert names == ["fleet_status", "list_actions", "propose_action", "action_status"]


def test_fleet_status_is_the_hub(c, monkeypatch):
    app = hub.App(id="mentat", title="Mentat", privacy="home")

    async def gather():
        return [hub.Status(app=app, up=True, items=[hub.Item(title="3 digests")])]
    monkeypatch.setattr(hub, "gather", gather)
    out = call(c, "fleet_status")
    data = json.loads(out["content"][0]["text"])
    assert out["isError"] is False
    assert data["apps"][0]["id"] == "mentat" and data["apps"][0]["items"][0]["title"] == "3 digests"


def test_list_actions_shows_targets(c):
    routine("News")
    data = json.loads(call(c, "list_actions")["content"][0]["text"])
    by = {a["action"]: a for a in data["actions"]}
    assert by["run_routine"]["targets"] == ["News"] and by["restart_app"]["risk"] == "high"
    assert by["rerun_summary"]["targets"] == ["daily", "weekly"]


def test_propose_sends_a_card_naming_the_client_and_changes_nothing(c):
    r = routine()
    sid = rpc(c, "initialize", {"clientInfo": {"name": "claude-code"}}).headers["Mcp-Session-Id"]
    out = call(c, "propose_action", {"action": "set_routine", "target": "Morning AI News",
                                     "state": "paused", "reason": "user asked"},
               headers={"Mcp-Session-Id": sid})
    assert out["isError"] is False
    (card,) = c.cards
    assert out["content"][0]["text"].startswith(f"Proposed {card['id']}: Pause the routine")
    assert "MCP client “claude-code” (what it read before is unknown)" in actions.card_text(card)
    assert store.get(r["id"])["enabled"] is True
    status = json.loads(call(c, "action_status", {"id": card["id"]})["content"][0]["text"])
    assert status["status"] == "pending" and status["title"] == "Pause the routine “Morning AI News”"


def test_propose_refuses_when_approvals_cant_arrive_or_the_target_is_wrong(c, monkeypatch):
    routine()
    out = call(c, "propose_action", {"action": "run_routine", "target": "Weather", "reason": "x"})
    assert out["isError"] is True and out["content"][0]["text"].startswith("There is no routine")
    monkeypatch.setattr(mcp_server.telegram_chat, "active", lambda: False)
    out = call(c, "propose_action", {"action": "run_routine", "target": "Morning AI News", "reason": "x"})
    assert out["isError"] is True and "chat from Telegram is off" in out["content"][0]["text"]
    assert c.cards == [] and actions._records == {}


def test_action_status_for_unknown_ids(c):
    assert call(c, "action_status", {"id": "../../etc"})["isError"] is True
    assert call(c, "action_status", {"id": "0" * 12})["isError"] is True

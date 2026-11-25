"""A saved skill reaching the model: chat (agent, stream, legacy), conversations, routines."""
from __future__ import annotations

import asyncio

import pytest

import chat
import routine_runner as rr
import skills
from test_agent_endpoint import AUTH, URL, env  # noqa: F401  (fixture)
from test_routine_runner import env as renv  # noqa: F401  (fixture)
from test_routine_runner import last, make
from fastapi import FastAPI
from fastapi.testclient import TestClient

RULE = "Two lines only."


@pytest.fixture
def skill(tmp_path):
    skills.load(tmp_path / "skills.json")
    skills._save({"brief": {"name": "brief", "description": "", "instructions": RULE}})
    yield
    skills.load(tmp_path / "none.json")


@pytest.fixture
def client(env, skill):  # noqa: F811
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def body(text, **kw):
    return {"model": "default", "messages": [{"role": "user", "content": text}], "store": False, **kw}


def sent(env):  # noqa: F811
    return (env.chat_payloads or env.stream_payloads)[-1]["messages"]


@pytest.mark.parametrize("extra", [{"agent": True}, {"agent": True, "stream": True}, {"agent": False, "search": False}])
def test_slash_name_adds_the_instructions_and_drops_the_command(client, env, extra):  # noqa: F811
    assert client.post(URL, json=body("/brief what is a llama", **extra), headers=AUTH).status_code == 200
    m = sent(env)
    assert RULE in m[0]["content"] and m[-1]["content"] == "what is a llama"


def test_skill_field_and_unknown_skill(client, env):  # noqa: F811
    client.post(URL, json=body("llamas?", agent=True, skill="brief"), headers=AUTH)
    assert RULE in sent(env)[0]["content"]
    assert client.post(URL, json=body("x", skill="nope"), headers=AUTH).status_code == 404


def test_a_skill_can_be_combined_with_search(client, env):  # noqa: F811
    client.post(URL, json=body("/brief /search llamas", agent=True), headers=AUTH)
    assert env.searches and RULE in str(sent(env)[0]["content"])


@pytest.mark.parametrize("text,extra", [("/nope x", {}), ("plain", {}), ("/brief x", {"raw": True})])
def test_no_skill_means_no_change(client, env, text, extra):  # noqa: F811
    client.post(URL, json=body(text, agent=False, search=False, memory=False, **extra), headers=AUTH)
    m = sent(env)
    assert RULE not in str(m) and m[-1]["content"] == text


def test_routine_uses_its_skill(renv, skill):  # noqa: F811
    r = make(skill="brief")
    asyncio.run(rr.run_routine(r["id"], "manual"))
    system = renv.payloads[0]["messages"][0]["content"]
    assert RULE in system and last(r)["status"] == "ok"
    asyncio.run(rr.run_routine(make()["id"], "manual"))
    assert RULE not in renv.payloads[-1]["messages"][0]["content"]


def test_a_routine_whose_skill_vanished_still_runs(renv, skill):  # noqa: F811
    r = make(skill="gone")
    asyncio.run(rr.run_routine(r["id"], "manual"))
    assert last(r)["status"] == "ok"

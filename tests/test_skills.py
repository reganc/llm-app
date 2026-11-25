"""Saved skills: storage, API, and how they reach the model."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routine_store
import skills

AUTH = {"Authorization": "Bearer test-key"}
BRIEF = {"name": "brief", "description": "Short", "instructions": "Answer in two lines."}


@pytest.fixture
def c(monkeypatch, tmp_path):
    skills.load(tmp_path / "skills.json")
    routine_store.load(tmp_path / "r")
    monkeypatch.setattr("auth.CFG", dataclasses.replace(skills.routine_store and __import__("config").CFG,
                                                        api_key="test-key"))
    app = FastAPI()
    app.include_router(skills.router)
    yield TestClient(app)
    skills.load(tmp_path / "none.json")


@pytest.mark.parametrize("m,p", [("get", ""), ("post", ""), ("put", "/a"), ("delete", "/a")])
def test_auth_required(c, m, p):
    assert getattr(c, m)(f"/v1/skills{p}").status_code == 401


def test_create_edit_delete_and_reload(c):
    assert c.post("/v1/skills", json=BRIEF, headers=AUTH).status_code == 201
    assert c.post("/v1/skills", json=BRIEF, headers=AUTH).status_code == 409
    r = c.put("/v1/skills/brief", json={**BRIEF, "instructions": "One line."}, headers=AUTH)
    assert r.json()["instructions"] == "One line."
    skills.load()                                    # survives a restart
    assert skills.get("BRIEF")["instructions"] == "One line."
    assert c.get("/v1/skills", headers=AUTH).json()["skills"][0]["used_by"] == []
    assert c.put("/v1/skills/brief", json={**BRIEF, "name": "other"}, headers=AUTH).status_code == 422
    assert c.delete("/v1/skills/brief", headers=AUTH).status_code == 200
    assert c.delete("/v1/skills/brief", headers=AUTH).status_code == 404
    assert c.put("/v1/skills/brief", json=BRIEF, headers=AUTH).status_code == 404


@pytest.mark.parametrize("bad", [
    {"name": "search"}, {"name": "Run"}, {"name": "has space"}, {"name": "../x"}, {"name": ""},
    {"name": "a" * 41}, {"instructions": "  "}, {"instructions": "x" * 4001},
    {"description": "x" * 121}, {"extra": 1},
])
def test_validation(c, bad):
    assert c.post("/v1/skills", json={**BRIEF, **bad}, headers=AUTH).status_code == 422


def test_names_are_lower_cased(c):
    assert c.post("/v1/skills", json={**BRIEF, "name": "Brief"}, headers=AUTH).json()["name"] == "brief"


def test_limit(c, monkeypatch):
    monkeypatch.setattr(skills, "MAX_SKILLS", 1)
    c.post("/v1/skills", json=BRIEF, headers=AUTH)
    assert c.post("/v1/skills", json={**BRIEF, "name": "two"}, headers=AUTH).status_code == 409


def test_a_skill_in_use_cannot_be_deleted(c):
    c.post("/v1/skills", json=BRIEF, headers=AUTH)
    routine_store.create({"name": "Morning", "skill": "brief", "next_run_at": None})
    r = c.delete("/v1/skills/brief", headers=AUTH)
    assert r.status_code == 409 and "Morning" in r.json()["detail"]


def test_unreadable_file_is_left_alone(tmp_path):
    p = tmp_path / "skills.json"
    p.write_text("{not json")
    skills.load(p)
    assert skills.list_all() == [] and p.read_text() == "{not json"


def test_split(c):
    c.post("/v1/skills", json=BRIEF, headers=AUTH)
    assert skills.split("/brief nvidia news")[0] == "nvidia news"
    assert skills.split("/BRIEF\nnvidia")[1]["name"] == "brief"
    assert "conversation" in skills.split("/brief")[0]
    for text in ("/other x", "brief x", "/briefly x", "", "/ brief", "see /brief x"):
        assert skills.split(text) == (text, None)


def test_prompt_keeps_source_rules_and_only_touches_the_system_message(c):
    c.post("/v1/skills", json=BRIEF, headers=AUTH)
    out = skills.apply([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], skills.get("brief"))
    assert out[0]["content"].startswith("S") and "Answer in two lines." in out[0]["content"]
    assert "citations still apply" in out[0]["content"] and out[1]["content"] == "U"
    assert skills.prompt(None) == "" and skills.apply([{"role": "system", "content": "S"}], None)[0]["content"] == "S"

"""`PATCH /v1/settings` validates every field before writing any of them.

On 2026-11-20 a browser with no API key stored opened Settings to paste one.
Its settings load had 401'd, so the model and prompt dropdowns were empty, and
closing the dialog PATCHed `default_model: ""` and `default_system_prompt_key:
""`. The endpoint saved the empty model, *then* rejected the prompt key with a
400. Every gateway call after that sent `model: ""` and Ollama 400'd them all.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_routes
import config
import ollama as oll

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
ACTIVE = "huihui_ai/qwen3.5-abliterated:9b"
INSTALLED = [{"name": ACTIVE}, {"name": "nomic-embed-text:latest"},
             {"name": "qwen3.5:9b"}]


@pytest.fixture
def env(monkeypatch, tmp_path):
    cfg = dataclasses.replace(admin_routes.CFG, api_key=API_KEY, default_model=ACTIVE)
    monkeypatch.setattr(admin_routes, "CFG", cfg)
    monkeypatch.setattr(config, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "_settings", {"default_model": ACTIVE,
                                              "default_system_prompt_key": "default"})
    copies: list[str] = []

    async def fake_list():
        return INSTALLED

    async def fake_copy(source, _dest):
        copies.append(source)
        return True

    monkeypatch.setattr(oll, "list_models", fake_list)
    monkeypatch.setattr(oll, "copy_model", fake_copy)
    app = FastAPI()
    app.include_router(admin_routes.router)
    return TestClient(app), copies


def patch(client, **body):
    return client.patch("/v1/settings", headers=AUTH, json=body)


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_model_is_rejected_and_nothing_is_written(env, bad):
    client, copies = env
    r = patch(client, default_model=bad)
    assert r.status_code == 400
    assert config.get_active_model() == ACTIVE
    assert copies == []


def test_model_that_is_not_installed_is_rejected(env):
    client, copies = env
    r = patch(client, default_model="huihui_ai/qwen2.5-abliterate:14b")
    assert r.status_code == 400 and "not installed" in r.json()["detail"]
    assert config.get_active_model() == ACTIVE
    assert copies == []


def test_bad_prompt_key_rolls_back_nothing_because_nothing_was_written(env):
    # The 2026-11-20 request: a valid-looking field next to an invalid one.
    client, copies = env
    r = patch(client, default_model="qwen3.5:9b", default_system_prompt_key="")
    assert r.status_code == 400
    assert config.get_active_model() == ACTIVE
    assert copies == []


def test_valid_patch_saves_and_repoints_the_alias(env):
    client, copies = env
    r = patch(client, default_model="qwen3.5:9b", search_always=True)
    assert r.status_code == 200
    assert r.json()["default_model"] == "qwen3.5:9b"
    assert config.get_setting("search_always") is True
    assert copies == ["qwen3.5:9b"]


def test_latest_tag_may_be_omitted(env):
    client, _ = env
    assert patch(client, default_model="nomic-embed-text").status_code == 200


def test_prompt_key_alone_does_not_touch_the_model(env):
    client, copies = env
    key = next(k for k in admin_routes.SYSTEM_PROMPTS if k != "default")
    assert patch(client, default_system_prompt_key=key).status_code == 200
    assert config.get_active_model() == ACTIVE
    assert copies == []


def test_ollama_unreachable_refuses_a_model_change(env, monkeypatch):
    client, copies = env

    async def down():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(oll, "list_models", down)
    assert patch(client, default_model="qwen3.5:9b").status_code == 503
    assert config.get_active_model() == ACTIVE
    assert copies == []


def test_blank_stored_model_falls_back_to_env(env, monkeypatch):
    # A settings.json already holding "" (as on 2026-11-20) must not take the
    # gateway down: resolve to DEFAULT_MODEL instead.
    monkeypatch.setattr(config, "_settings", {"default_model": ""})
    assert config.get_active_model() == ACTIVE

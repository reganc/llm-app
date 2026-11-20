"""The "Always search the web" setting.

The SPA's checkbox PATCHed `search_always` into settings.json, but the only
reader was `CFG.search_always` (the SEARCH_ALWAYS env var), GET /v1/settings
never returned it (so the box always opened unchecked), and agent mode — the
default path — has no router to consult. The stored value now overrides the
env default everywhere it applies.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_routes
import config
import search


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    store: dict = {}
    monkeypatch.setattr(config, "_settings", store)
    return store


def _cfg(monkeypatch, **kw):
    cfg = dataclasses.replace(config.CFG, **kw)
    for mod in (config, search):
        monkeypatch.setattr(mod, "CFG", cfg)
    return cfg


def test_env_default_when_nothing_stored(settings, monkeypatch):
    _cfg(monkeypatch, search_always=True)
    assert config.get_search_always() is True


@pytest.mark.parametrize("stored", [True, False])
def test_stored_value_overrides_env(settings, monkeypatch, stored):
    _cfg(monkeypatch, search_always=not stored)
    settings["search_always"] = stored
    assert config.get_search_always() is stored


async def _no(*_a, **_kw):
    return False


def test_router_searches_when_setting_on(settings, monkeypatch):
    _cfg(monkeypatch, search_enabled=True, search_always=False)
    monkeypatch.setattr(search, "_classify_freshness", _no)
    monkeypatch.setattr(search, "_probe_hedging", _no)
    settings["search_always"] = True
    assert asyncio.run(search.should_auto_search([{"score": 0.99}], "hi")) is True


def test_router_follows_normal_rules_when_setting_off(settings, monkeypatch):
    _cfg(monkeypatch, search_enabled=True, search_always=True)
    settings["search_always"] = False
    # No query + memory present -> the normal rule says no search.
    assert asyncio.run(search.should_auto_search([{"score": 0.99}], "")) is False


def test_search_disabled_wins(settings, monkeypatch):
    _cfg(monkeypatch, search_enabled=False)
    settings["search_always"] = True
    assert asyncio.run(search.should_auto_search([], "hi")) is False


def test_get_settings_reports_it(settings, monkeypatch):
    cfg = dataclasses.replace(admin_routes.CFG, api_key="k", search_always=False)
    monkeypatch.setattr(admin_routes, "CFG", cfg)
    monkeypatch.setattr("auth.CFG", cfg)
    settings["search_always"] = True
    app = FastAPI()
    app.include_router(admin_routes.router)
    r = TestClient(app).get("/v1/settings", headers={"Authorization": "Bearer k"})
    assert r.json()["search_always"] is True

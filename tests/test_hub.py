"""The app hub: the app list, privacy rule, cards, adapters, and their text."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

import hub
import hub_adapters

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
TODAY = date(2026, 11, 26)


def _fleet(tmp_path, text: str):
    p = tmp_path / "fleet.yaml"
    p.write_text(text)
    return hub.load_fleet(p)


def test_fleet_rules(tmp_path):
    apps = _fleet(tmp_path, """
apps:
  - {id: tax, title: Tax, privacy: private, summary: {file: tax.json}}
  - {id: trading, privacy: home, health: "http://host.docker.internal:8010/health",
     summary: {adapter: trading, base: "http://host.docker.internal:8010"}}
  - {id: sneaky, privacy: home, health: "http://169.254.169.254/latest", summary: {adapter: trading, base: "http://10.0.0.5"}}
  - {id: noprivacy, health: "http://host.docker.internal:1/health"}
  - {id: "Bad Id", privacy: home}
""")
    assert [a.id for a in apps] == ["tax", "trading", "sneaky"]
    assert apps[1].health == "http://host.docker.internal:8010/health"
    assert apps[2].health is None and "base" not in apps[2].summary  # only host.docker.internal is fetched


def test_restart_entries_only_for_home_apps_with_safe_names(tmp_path):
    apps = {a.id: a for a in _fleet(tmp_path, """
apps:
  - {id: mentat, privacy: home, restart: {project: mentat, services: [api, worker, api]}}
  - {id: tax, privacy: private, restart: {project: tax, services: [app]}}
  - {id: odd, privacy: home, restart: {project: "a b", services: [api]}}
  - {id: none, privacy: home, restart: {project: x, services: []}}
""")}
    assert apps["mentat"].restart == {"project": "mentat", "services": ["api", "worker"]}
    assert apps["tax"].restart is None and apps["odd"].restart is None and apps["none"].restart is None


def test_unreadable_fleet_is_empty(tmp_path):
    assert hub.load_fleet(tmp_path / "missing.yaml") == []
    assert _fleet(tmp_path, "apps: [") == []


@pytest.mark.parametrize("title,privacy,ok", [
    ("3 ways to lower this year's tax", "private", True),
    ("Due 2026-12-31", "private", True),
    ("Saves 2,759", "private", False),
    ("Owe 377 more", "private", False),
    ("Rate 22.5", "private", False),
    ("Saves $40", "home", False),
    ("Buy signals: NVDA; 1200 shares", "home", True),
])
def test_privacy_rule(title, privacy, ok):
    assert hub.allowed(title, privacy) is ok


def _card(tmp_path, monkeypatch, body, *, privacy="private"):
    monkeypatch.setattr(hub, "HUB_DIR", tmp_path)
    (tmp_path / "tax.json").write_text(body if isinstance(body, str) else json.dumps(body))
    app = hub.App(id="tax", title="Tax", privacy=privacy, summary={"file": "tax.json"})
    return hub._read_file_card(app, NOW)


def test_file_card_withholds_figures(tmp_path, monkeypatch):
    s = _card(tmp_path, monkeypatch, {"generated_at": NOW.isoformat(), "items": [
        {"title": "2 ways to lower this year's tax", "severity": "warn", "due": "2026-12-31"},
        {"title": "You could save $2,759", "severity": "info"},
    ]})
    assert [i.title for i in s.items] == ["2 ways to lower this year's tax"]
    assert s.withheld == 1 and s.problem is None and s.level == "warn"


def test_file_card_problems(tmp_path, monkeypatch):
    old = (NOW - timedelta(hours=40)).isoformat()
    assert _card(tmp_path, monkeypatch, {"generated_at": old, "items": []}).problem == "its card is out of date"
    assert _card(tmp_path, monkeypatch, "{not json").problem == "its card couldn't be read"
    assert _card(tmp_path, monkeypatch, {"items": [{"title": ""}]}).problem == "its card couldn't be read"
    monkeypatch.setattr(hub, "HUB_DIR", tmp_path / "none")
    app = hub.App(id="x", title="X", privacy="private", summary={"file": "../../etc/passwd"})
    assert hub._read_file_card(app, NOW).problem == "hasn't reported yet"  # only the name is used


def _getter(responses):
    async def get(url):
        for suffix, body in responses.items():
            if url.endswith(suffix):
                return body
        raise AssertionError(url)
    return get


def test_trading_adapter():
    items = asyncio.run(hub_adapters.trading(_getter({
        "/admin/data_health": {"summary": {"ok": 1, "warn": 2, "fail": 1},
                               "checks": [{"name": "ohlcv_technicals_gap", "status": "fail"}]},
        "/portfolio/health": {"overall": "CRITICAL", "positions": {"status": "CRITICAL"},
                              "exposure": {"status": "WARNING", "net": 123456.0}},
        "/research/briefing": {"available": True, "regime": {"regime": "BEAR"}, "buy_signals": [{"ticker": "SWK"}]},
    }), "http://b", NOW))
    assert [(i["title"], i["severity"]) for i in items] == [
        ("Data checks failing: ohlcv_technicals_gap", "alert"),
        ("Portfolio risk critical: positions critical, exposure warning", "alert"),
        ("Market regime bear; buy signals: SWK", "info"),
    ]
    assert not any("123456" in i["title"] for i in items)


def test_delphi_and_nexus_adapters():
    recent, old = NOW - timedelta(hours=2), NOW - timedelta(days=3)
    d = asyncio.run(hub_adapters.delphi(_getter({"/api/signals": [
        {"time": recent.isoformat(), "score": 0.4, "question": "Will A?", "suggested_stake": 50.0},
        {"time": recent.isoformat(), "score": 0.9, "question": "Will  B\nhappen?"},
        {"time": old.isoformat(), "score": 1.0, "question": "Old?"},
    ]}), "http://b", NOW))
    assert d == [{"title": "2 new signals in the last day; strongest: Will B happen?", "severity": "info"}]
    n = asyncio.run(hub_adapters.nexus(_getter({
        "/api/health/drift": {"gate_tripped": True},
        "/api/episodes": {"episodes": [{"opened_at": recent.isoformat(), "domains": ["market", "atmosphere"]},
                                       {"opened_at": old.isoformat(), "domains": ["x"]}]},
    }), "http://b", NOW))
    many = [{"time": recent.isoformat(), "score": 0.1, "question": "Q?"}] * 50
    d50 = asyncio.run(hub_adapters.delphi(_getter({"/api/signals": many}), "http://b", NOW))
    assert d50[0]["title"].startswith("50 or more new signals")
    assert [i["title"] for i in n] == ["Drift gate tripped: new hypotheses are being held back",
                                       "1 cross-domain episodes in the last day (atmosphere, market)"]


def _transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2)


def test_gather_contains_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "HUB_DIR", tmp_path)
    apps = [
        hub.App(id="down", title="Down", privacy="home", health="http://host.docker.internal:1/health"),
        hub.App(id="odd", title="Odd", privacy="home", health="http://host.docker.internal:2/health",
                summary={"adapter": "delphi", "base": "http://host.docker.internal:2"}),
        hub.App(id="nohealth", title="Quiet", privacy="private", summary={"file": "none.json"}),
    ]

    def handler(req: httpx.Request):
        if req.url.port == 1:
            raise httpx.ConnectError("refused")
        if req.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, content=b"not json")

    async def run():
        async with _transport(handler) as c:
            return await hub.gather(NOW, apps, c)

    down, odd, quiet = asyncio.run(run())
    assert down.up is False and down.level == "alert"
    assert odd.up is True and odd.problem == "its status couldn't be read" and odd.level == "warn"
    assert quiet.up is None and quiet.problem == "hasn't reported yet"


def test_get_json_refuses_other_hosts():
    async def run():
        async with _transport(lambda r: httpx.Response(200, json={})) as c:
            await hub._get_json(c, "http://example.com/x")
    with pytest.raises(ValueError):
        asyncio.run(run())


def _statuses():
    tax = hub.App(id="tax", title="Tax Planner", privacy="private")
    trade = hub.App(id="trading", title="Signal Trading", privacy="home")
    delphi = hub.App(id="delphi", title="Delphi", privacy="home")
    return [
        hub.Status(delphi, up=True, items=[hub.Item(title="2 new signals", severity="info")]),
        hub.Status(tax, items=[hub.Item(title="3 ways to lower this year's tax", severity="info", due=date(2026, 12, 3)),
                               hub.Item(title="Upload a W-2", severity="info")], withheld=1),
        hub.Status(trade, up=True, items=[hub.Item(title="Portfolio risk critical", severity="alert")]),
    ]


def test_apps_text_worst_first():
    text = hub.apps_text(_statuses(), TODAY)
    assert text.splitlines()[0] == "🔴 Signal Trading"
    assert "  • 3 ways to lower this year's tax (in 7 days)" in text
    assert "  • 1 item withheld (contained figures)" in text
    assert text.index("Delphi") < text.index("Tax Planner")  # both green: by name


def test_briefing_shows_only_what_needs_attention():
    text = hub.briefing_section(_statuses(), TODAY)
    assert text == ("Apps:\n• Signal Trading: Portfolio risk critical\n"
                    "• Tax Planner: 3 ways to lower this year's tax (in 7 days)\n"
                    "• Tax Planner: 1 item withheld (contained figures)\nSend /apps for all of them.")
    calm = [hub.Status(hub.App(id="d", title="D", privacy="home"), up=True, items=[hub.Item(title="fine")])]
    assert hub.briefing_section(calm, TODAY) is None


def test_real_fleet_file_parses():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "fleet.yaml")
    if not os.path.exists(path):
        pytest.skip("~/apps/fleet.yaml not present")
    apps = hub.load_fleet(__import__("pathlib").Path(path))
    assert apps and all(a.privacy in ("home", "private") for a in apps)


def test_homelab_high_findings_are_a_warning(monkeypatch):
    import briefing
    monkeypatch.setattr(briefing, "homelab_section", lambda now: "Homelab: 8 high, 3 medium.\nNeeds a look: x")
    assert asyncio.run(hub_adapters.homelab(None, "", NOW)) == [{"title": "8 high, 3 medium", "severity": "warn"}]
    monkeypatch.setattr(briefing, "homelab_section", lambda now: "Homelab: 2 low.")
    assert asyncio.run(hub_adapters.homelab(None, "", NOW))[0]["severity"] == "info"


def _helm_app(id_, status="operational", **kw):
    return {"id": id_, "display_name": id_.title(), "operational_status": status, "frozen": False,
            "diverged": False, "severity_breakdown": {}, **kw}


def test_helm_adapter():
    calm = asyncio.run(hub_adapters.helm(_getter({"/fleet": {"apps": [
        _helm_app("loop", severity_breakdown={"high": 348, "medium": 9}),
        _helm_app("carecore", severity_breakdown={"high": 9}),
        _helm_app("vps", "unknown"),
    ]}}), "http://b", NOW))
    assert calm == [{"title": "2 remote apps operational; 1 not monitored", "severity": "info"},
                    {"title": "357 high-severity findings open across the remote apps", "severity": "info"}]
    bad = asyncio.run(hub_adapters.helm(_getter({"/fleet": {"apps": [
        _helm_app("loop", "major_outage", diverged=True),
        _helm_app("carecore", "degraded", frozen=True, severity_breakdown={"critical": 2}),
    ]}}), "http://b", NOW))
    assert [(i["title"], i["severity"]) for i in bad] == [
        ("Outage: Loop", "alert"),
        ("Degraded: Carecore", "warn"),
        ("Running a different version than recorded: Loop", "warn"),
        ("Deploys frozen: Carecore", "info"),
        ("2 critical findings open: Carecore", "warn"),
    ]


def _mentat(daily, weekly, collisions=0):
    return _getter({
        "/summaries?limit=5": {"daily": [{"id": d, "body_md": "model text $5"} for d in daily],
                               "weekly": [{"id": w} for w in weekly], "headline": "free text"},
        "/steward/collisions": {"collisions": [{}] * collisions},
    })


def test_mentat_adapter():
    fresh = asyncio.run(hub_adapters.mentat(_mentat(
        ["daily-nexus-20261126", "daily-chronos-20261126", "daily-delphi-20261125"], ["weekly-2026-W48"], 7),
        "http://b", NOW))
    assert fresh == [{"title": "Daily digests for 2026-11-26: chronos, nexus", "severity": "info"},
                     {"title": "Steward: 7 port or subnet clashes between repos", "severity": "info"}]
    stale = asyncio.run(hub_adapters.mentat(_mentat(["daily-nexus-20261120"], ["weekly-2026-W35"]), "http://b", NOW))
    assert [(i["title"], i["severity"]) for i in stale] == [
        ("No daily digest since 2026-11-20", "warn"),
        ("Weekly synthesis hasn't run since week 35 of 2026", "warn"),
    ]
    assert asyncio.run(hub_adapters.mentat(_mentat([], []), "http://b", NOW))[0]["severity"] == "warn"

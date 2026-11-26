"""The action gate: what can be proposed, how an approval is bound to one stored
action, expiry, limits, the second tap for high risk, execution and the log."""
from __future__ import annotations

import asyncio
import json

import pytest

import action_log
import actions
import routine_runner as rr
import routine_store as store
import watch_runner
import watch_store


def run(c):
    return asyncio.run(c)


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch, tmp_path):
    store.load(tmp_path / "r")
    watch_store.load(tmp_path / "w")
    actions.load(tmp_path / "a")
    c = Clock()
    monkeypatch.setattr(actions, "_now", c)
    return c


def routine(name="Morning AI News", enabled=True):
    return store.create({"name": name, "prompt": "p", "query": None, "timezone": "UTC",
                         "schedule": {"kind": "interval", "hours": 1}, "enabled": enabled,
                         "next_run_at": None, "sources": {"web": True, "library": False},
                         "notify": "off"})


def propose(action="run_routine", target="Morning AI News", **kw):
    args = {"target": target, **({"state": kw.pop("state")} if "state" in kw else {})}
    return actions.propose(action, args, reason=kw.pop("reason", "asked"), source="telegram",
                           after_reading=kw.pop("after_reading", False))


# ── proposing ────────────────────────────────────────────────────────────────

def test_proposal_stores_ids_not_names_and_the_card_says_what_will_run(clock):
    r = routine()
    p = propose(target="morning ai")                      # unique start, any case
    assert p["args"] == {"routine_id": r["id"]} and p["status"] == "pending"
    assert p["expires_at"] == clock.t + actions.TTL_S
    text = actions.card_text(p)
    assert text.splitlines()[:2] == ["Proposed action · low risk", "Run the routine “Morning AI News” now"]
    assert "Why (the model's words): “asked”" in text
    assert [label for label, _ in actions.buttons(p)] == ["Approve", "Reject"]
    assert all(len(data.encode()) <= 64 for _, data in actions.buttons(p))   # Telegram's limit


def test_unknown_action_target_or_state_is_refused_with_what_exists(clock):
    routine("News")
    routine("Newsletter digest")
    with pytest.raises(actions.Refused, match="Unknown action 'restart_app'"):
        propose("restart_app", "mentat")
    with pytest.raises(actions.Refused, match="no routine called “Weather”. Yours: .*“News”"):
        propose(target="Weather")
    with pytest.raises(actions.Refused, match="More than one"):
        propose(target="new")
    assert propose(target="news")["status"] == "pending"   # exact name wins over prefixes
    with pytest.raises(actions.Refused, match="state must be"):
        propose("set_routine", "News", state="off")
    with pytest.raises(actions.Refused, match="already active"):
        propose("set_routine", "News", state="active")


def test_only_a_conversation_can_propose(clock):
    routine()
    with pytest.raises(actions.Refused, match="only from a conversation"):
        actions.propose("run_routine", {"target": "Morning AI News"}, reason="", source="routine",
                        after_reading=False)


def test_duplicates_and_limits(clock):
    for i in range(5):
        routine(f"R{i}")
    propose(target="R0")
    with pytest.raises(actions.Refused, match="already waiting"):
        propose(target="R0")
    propose(target="R1")
    propose(target="R2")
    with pytest.raises(actions.Refused, match="3 actions are already waiting"):
        propose(target="R3")
    # hourly cap counts every proposal, decided or not
    for rid in list(actions._records):
        actions.decide(rid, "reject", actions.get(rid)["nonce"])
    for i in range(7):
        p = propose(target=f"R{i % 5}")
        actions.decide(p["id"], "reject", p["nonce"])
    with pytest.raises(actions.Refused, match="No more than 10 proposals an hour"):
        propose(target="R4")
    clock.t += 3601
    assert propose(target="R4")["status"] == "pending"


def test_reason_is_trimmed_and_reading_is_shown(clock):
    routine()
    p = propose(reason="x " * 400, after_reading=True)
    assert len(p["reason"]) <= 300
    assert "after reading web pages or your library" in actions.card_text(p)


# ── deciding ─────────────────────────────────────────────────────────────────

def test_approve_once_with_the_current_nonce(clock):
    routine()
    p = propose()
    with pytest.raises(actions.Refused, match="out of date"):
        actions.decide(p["id"], "approve", "deadbeef")
    record, step = actions.decide(p["id"], "approve", p["nonce"])
    assert step == "run" and record["status"] == "approved" and record["nonce"] == ""
    with pytest.raises(actions.Refused, match="Already approved"):
        actions.decide(p["id"], "approve", p["nonce"])
    assert actions.buttons(record) == []


def test_reject_and_expiry_end_it(clock):
    routine("A")
    routine("B")
    a, b = propose(target="A"), propose(target="B")
    assert actions.decide(a["id"], "reject", a["nonce"])[1] == "rejected"
    clock.t += actions.TTL_S
    expired = actions.sweep()
    assert [r["id"] for r in expired] == [b["id"]]
    with pytest.raises(actions.Refused, match="Already expired"):
        actions.decide(b["id"], "approve", b["nonce"])
    assert actions.card_text(actions.get(b["id"])).startswith("Expired. Nothing was changed.")


def test_high_risk_needs_a_second_tap_with_a_new_nonce(clock, monkeypatch):
    routine()
    base = actions.ACTIONS["run_routine"]
    monkeypatch.setitem(actions.ACTIONS, "run_routine",
                        actions.Action("run_routine", "high", base.resolve, base.describe, base.run))
    p = propose()
    record, step = actions.decide(p["id"], "approve", p["nonce"])
    assert step == "confirm" and record["status"] == "confirming" and record["nonce"] != p["nonce"]
    assert actions.buttons(record)[0][0].startswith("Yes: Run the routine")
    with pytest.raises(actions.Refused, match="out of date"):      # the first button again
        actions.decide(p["id"], "approve", p["nonce"])
    with pytest.raises(actions.Refused, match="out of date"):      # approve isn't confirm
        actions.decide(p["id"], "approve", record["nonce"])
    assert actions.decide(p["id"], "confirm", record["nonce"])[1] == "run"


# ── executing ────────────────────────────────────────────────────────────────

def approved(**kw):
    p = propose(**kw)
    return actions.decide(p["id"], "approve", p["nonce"])[0]


def test_only_an_approved_action_runs(clock):
    routine()
    p = propose()
    with pytest.raises(actions.Refused):
        run(actions.execute(p["id"]))


def test_run_routine_starts_it_and_reports_through_the_hook(clock, monkeypatch):
    r = routine()
    started, reported = [], []

    def start(rid, done):
        started.append(rid)
        asyncio.get_running_loop().create_task(done({"status": "ok"}))
        return True

    async def done(routine_, run_):
        reported.append((routine_["name"], run_["status"]))
    monkeypatch.setattr(rr, "start_manual", start)

    async def go():
        out = await actions.execute(approved()["id"], {"routine_done": done})
        await asyncio.sleep(0)
        return out
    out = run(go())
    assert out["status"] == "done" and out["result"].startswith("Started Morning AI News")
    assert started == [r["id"]] and reported == [("Morning AI News", "ok")]


def test_set_routine_pauses_and_check_watch_reads(clock, monkeypatch):
    r = routine()
    out = run(actions.execute(approved(action="set_routine", state="paused")["id"]))
    assert out["result"] == "Morning AI News paused." and store.get(r["id"])["enabled"] is False
    w = watch_store.create({"name": "GPU price", "url": "https://shop.example/gpu", "enabled": True})

    async def check(wid):
        watch_store.update(wid, {"shown": "Price 499"})
    monkeypatch.setattr(watch_runner, "check", check)
    out = run(actions.execute(approved(action="check_watch", target="gpu")["id"]))
    assert out["status"] == "done" and out["result"] == "GPU price: Price 499"
    assert w["id"] == out["args"]["watch_id"]


def test_target_gone_or_args_changed_after_approval_fails_without_running(clock, monkeypatch):
    r = routine()
    a = approved()
    store.delete(r["id"])
    out = run(actions.execute(a["id"]))
    assert out["status"] == "failed" and out["result"] == "That routine no longer exists."
    r2 = routine("Other")
    b = approved(target="Other")
    actions._records[b["id"]]["args"] = {"routine_id": "somethingelse"}   # tampered on disk
    calls = []
    monkeypatch.setattr(rr, "start_manual", lambda *a: calls.append(a) or True)
    out = run(actions.execute(b["id"]))
    assert out["status"] == "failed" and "changed after it was approved" in out["result"]
    assert calls == [] and store.get(r2["id"]) is not None


def test_a_crash_in_the_action_is_contained(clock, monkeypatch):
    routine()

    def boom(*_):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(rr, "start_manual", boom)
    out = run(actions.execute(approved()["id"]))
    assert out["status"] == "failed" and out["result"] == "It failed (RuntimeError)."


# ── storage and log ──────────────────────────────────────────────────────────

def test_proposals_survive_a_restart(clock, tmp_path):
    routine()
    p = propose()
    actions.load(tmp_path / "a")
    assert actions.get(p["id"])["status"] == "pending"
    assert json.loads((tmp_path / "a" / "pending.json").read_text())[p["id"]]["hash"] == p["hash"]


def test_an_action_cut_off_by_a_restart_is_failed_never_rerun(clock, tmp_path):
    routine()
    a = approved()
    actions._records[a["id"]]["status"] = "running"
    actions._save()
    actions.load(tmp_path / "a")
    out = actions.get(a["id"])
    assert out["status"] == "failed" and out["result"].startswith("Interrupted by a restart")
    with pytest.raises(actions.Refused):
        run(actions.execute(a["id"]))


def test_every_step_is_logged_in_an_unbroken_chain_without_the_reason(clock, tmp_path):
    routine()
    p = propose(reason="the user said: my secret phrase")
    actions.decide(p["id"], "approve", p["nonce"])
    kinds = [row["event"]["kind"] for row in action_log.tail()]
    assert kinds == ["proposed", "approved"]
    assert action_log.verify() == {"ok": True, "checked": 2, "broken_at": None}
    raw = (tmp_path / "a" / "audit.jsonl").read_text()
    assert "secret phrase" not in raw
    lines = raw.splitlines()
    row = json.loads(lines[0])
    row["event"]["args"]["routine_id"] = "x"
    (tmp_path / "a" / "audit.jsonl").write_text("\n".join([json.dumps(row), lines[1]]) + "\n")
    assert action_log.verify()["broken_at"] == 0


def test_tool_lists_the_users_routines_and_pages(clock):
    routine("News")
    routine("Weekly review", enabled=False)
    watch_store.create({"name": "GPU price", "url": "https://shop.example/gpu", "enabled": True})
    fn = actions.tool()["function"]
    assert fn["name"] == "propose_action"
    assert fn["parameters"]["properties"]["action"]["enum"] == ["run_routine", "check_watch", "set_routine"]
    assert "“News”, “Weekly review” (paused)" in fn["description"]
    assert "“GPU price”" in fn["description"]

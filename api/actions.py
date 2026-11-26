"""The action gate: the model proposes, the user approves, code acts.

The model gets one tool, `propose_action`. Code checks the action is on the
list and its target exists, stores the exact arguments with a 15-minute
expiry, and the user sees a card written by code from those arguments. Only
the user's tap (action_cards.py) moves a proposal on; nothing the model says
does. An approval covers one stored action: its arguments' hash, used once,
and everything is checked again just before it runs.

On the list (adding one means code and a test, never config): run a routine
now, check a watched page now, pause or resume a routine, re-run mentat's
daily or weekly summary, restart an app (high risk: a second tap; done by the
executor, the only part with Docker access, and only for apps whose fleet.yaml
entry has `restart:`). Never on it: anything
for a private app, deleting, settings, secrets, messages to anyone, money.

Proposals live in DATA_DIR/actions/pending.json (atomic writes, kept 7 days so
the hourly limit can count them); every step goes to action_log.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx

import action_log
import ask_apps
import hub
import routine_runner as runner
import routine_store
import watch_runner
import watch_store
from config import CFG, DATA_DIR

log = logging.getLogger("llm-api.actions")

TTL_S = 15 * 60
MAX_PENDING = 3
MAX_PER_HOUR = 10
KEEP_S = 7 * 24 * 3600
_REASON_CHARS = 300
OPEN = ("pending", "confirming")      # waiting on the user
_SOURCES = {"telegram": "Telegram chat"}


class Refused(Exception):
    """A proposal or approval that can't go ahead; the text says why."""


@dataclass(frozen=True)
class Action:
    name: str
    risk: str                                         # "low" | "high" (a second tap)
    resolve: Callable[[dict], dict]                   # model args → stored args, or Refused
    describe: Callable[[dict], tuple[str, list[str]]]  # stored args → (title, details)
    run: Callable[[dict, dict], Awaitable[str]]       # (stored args, hooks) → result sentence


# ── targets ──────────────────────────────────────────────────────────────────

def _find(kind: str, items: list[dict], target) -> dict:
    """An item by exact name, then by a unique start of its name (any case)."""
    name = " ".join(str(target or "").split()).lower()
    if not name:
        raise Refused(f"Say which {kind}.")
    exact = [i for i in items if i["name"].lower() == name]
    start = [i for i in items if i["name"].lower().startswith(name)]
    found = exact or start
    if len(found) == 1:
        return found[0]
    names = ", ".join(f"“{i['name']}”" for i in items[:10]) or "none"
    if found:
        raise Refused(f"More than one {kind} matches “{target}”. Use the full name: {names}.")
    raise Refused(f"There is no {kind} called “{target}”. Yours: {names}.")


def _routine(rid: str) -> dict:
    found = routine_store.get(rid)
    if found is None:
        raise Refused("That routine no longer exists.")
    return found


def _watch(wid: str) -> dict:
    found = watch_store.get(wid)
    if found is None:
        raise Refused("That watched page no longer exists.")
    return found


# ── the actions ──────────────────────────────────────────────────────────────

def _resolve_run(args: dict) -> dict:
    return {"routine_id": _find("routine", routine_store.list_all(), args.get("target"))["id"]}


def _describe_run(a: dict) -> tuple[str, list[str]]:
    return (f"Run the routine “{_routine(a['routine_id'])['name']}” now",
            ["It only searches and reads. Its result comes here when it's done."])


async def _run_run(a: dict, hooks: dict) -> str:
    routine = _routine(a["routine_id"])
    done = hooks.get("routine_done")

    async def report(run: dict) -> None:
        if done:
            await done(routine, run)
    if not runner.start_manual(routine["id"], report):
        raise Refused(f"{routine['name']} is already running.")
    return f"Started {routine['name']}. Its result will follow."


def _resolve_check(args: dict) -> dict:
    return {"watch_id": _find("watched page", watch_store.list_all(), args.get("target"))["id"]}


def _describe_check(a: dict) -> tuple[str, list[str]]:
    w = _watch(a["watch_id"])
    return f"Check the watched page “{w['name']}” now", [f"Reads {w['url']} (no model)."]


async def _run_check(a: dict, hooks: dict) -> str:
    await watch_runner.check(a["watch_id"])
    w = _watch(a["watch_id"])
    return f"{w['name']}: {w['error'] or w['shown'] or 'nothing read yet'}"


def _resolve_set(args: dict) -> dict:
    state = args.get("state")
    if state not in ("paused", "active"):
        raise Refused("state must be paused or active.")
    routine = _find("routine", routine_store.list_all(), args.get("target"))
    if routine["enabled"] == (state == "active"):
        raise Refused(f"{routine['name']} is already {'active' if routine['enabled'] else 'paused'}.")
    return {"routine_id": routine["id"], "enabled": state == "active"}


def _describe_set(a: dict) -> tuple[str, list[str]]:
    name = _routine(a["routine_id"])["name"]
    if a["enabled"]:
        return f"Resume the routine “{name}”", ["It runs on its schedule again."]
    return f"Pause the routine “{name}”", ["It stops running on schedule until resumed."]


async def _run_set(a: dict, hooks: dict) -> str:
    routine = runner.set_enabled(a["routine_id"], a["enabled"])
    if routine is None:
        raise Refused("That routine no longer exists.")
    return f"{routine['name']} {'resumed' if a['enabled'] else 'paused'}."


_KINDS = ("daily", "weekly")
SUMMARY_TIMEOUT_S = 600.0   # today's digests: one local model call per app


def _mentat() -> hub.App:
    found = next((a for a in hub.load_fleet() if a.id == "mentat"), None)
    if found is None or found.privacy != "home" or not hub._safe_url(found.summary.get("base")):
        raise Refused("Mentat isn't in the app list with an address that may be used.")
    return found


def _resolve_rerun(args: dict) -> dict:
    kind = str(args.get("target") or "").strip().lower()
    if kind not in _KINDS:
        raise Refused("target must be daily or weekly.")
    _mentat()
    return {"kind": kind}


def _describe_rerun(a: dict) -> tuple[str, list[str]]:
    if a["kind"] == "weekly":
        return ("Re-run Mentat's weekly synthesis",
                ["Replaces this week's report with a new one (local model, about half a minute)."])
    return ("Re-run Mentat's daily digests",
            ["Replaces today's digest for every app (local model, a few minutes)."])


def _summary_result(kind: str, body: dict) -> str:
    """One sentence from Mentat's reply; a headline only if it passes the privacy rule."""
    if body.get("status") == "no-data":
        return "There was nothing to summarize, so no report was written."
    if kind == "daily":
        saved = body.get("saved") if isinstance(body.get("saved"), dict) else {}
        failed = sorted(o for o, v in (body.get("origins") or {}).items()
                        if str(v).startswith("error"))
        text = f"Daily digests {', '.join(f'{o} {v}' for o, v in sorted(saved.items())) or 'unchanged'}."
        return text + (f" Failed: {', '.join(failed)}." if failed else "")
    check = body.get("check") if isinstance(body.get("check"), dict) else {}
    text = f"Weekly synthesis {body.get('saved', 'unchanged')}"
    if check.get("points"):
        text += f": {check.get('kept', 0)} of {check['points']} points held up"
    text += "." if body.get("status") != "unsourced" else " (too few to count as sourced)."
    headline = str(body.get("headline") or "")[:200]
    if headline and hub.allowed(headline, "home"):
        text += f"\n{headline}"
    return text


async def _run_rerun(a: dict, hooks: dict) -> str:
    app = _mentat()
    url = f"{app.summary['base']}/summaries/{a['kind']}?replace=true"
    async with httpx.AsyncClient(timeout=httpx.Timeout(SUMMARY_TIMEOUT_S, connect=5.0),
                                 follow_redirects=False) as client:
        try:
            body = await ask_apps._post_json(client, url, {})
        except ask_apps.AskError as e:
            raise Refused(f"Mentat: {e}") from None
        except httpx.TimeoutException:
            raise Refused("Mentat took too long.") from None
        except (httpx.HTTPError, ValueError):
            raise Refused("Mentat couldn't be reached.") from None
    if not isinstance(body, dict):
        raise Refused("Mentat's reply couldn't be read.")
    return _summary_result(a["kind"], body)


RESTART_COOLDOWN_S = 600   # the executor enforces it too
RESTART_TIMEOUT_S = 180.0  # restarting, then up to 90 s for the health check


def _restartable() -> list[hub.App]:
    return [a for a in hub.load_fleet() if a.privacy == "home" and a.restart]


def _resolve_restart(args: dict) -> dict:
    if not CFG.actions_executor_token:
        raise Refused("Restarts aren't set up (ACTIONS_EXECUTOR_TOKEN is empty).")
    apps = _restartable()
    wanted = " ".join(str(args.get("target") or "").split()).lower()
    found = next((a for a in apps if wanted in (a.id, a.title.lower())), None)
    if found is None:
        names = ", ".join(f"“{a.title}”" for a in apps) or "none"
        raise Refused(f"“{args.get('target')}” can't be restarted from here. These can: {names}.")
    recent = [r for r in _records.values() if r["action"] == "restart_app"
              and r["args"].get("app") == found.id and r["status"] in ("approved", "running", "done")
              and _now() - r["created_at"] < RESTART_COOLDOWN_S]
    if recent:
        raise Refused(f"{found.title} was restarted less than {RESTART_COOLDOWN_S // 60} minutes ago.")
    return {"app": found.id}


def _app(app_id: str) -> hub.App:
    found = next((a for a in _restartable() if a.id == app_id), None)
    if found is None:
        raise Refused(f"{app_id} can no longer be restarted from here.")
    return found


def _describe_restart(a: dict) -> tuple[str, list[str]]:
    app = _app(a["app"])
    spec = app.restart or {}
    lines = [f"Containers: {', '.join(spec.get('services', []))} (Docker project {spec.get('project')}).",
             "A short outage while they come back. Its database and storage keep running."]
    lines.append(f"Then waits up to 90 s for {app.health} to answer." if app.health
                 else "It has no health check, so recovery isn't confirmed.")
    return f"Restart {app.title}", lines


async def _run_restart(a: dict, hooks: dict) -> str:
    app = _app(a["app"])
    url = f"{CFG.actions_executor_url}/restart"
    headers = {"Authorization": f"Bearer {CFG.actions_executor_token}"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(RESTART_TIMEOUT_S, connect=5.0),
                                     follow_redirects=False) as client:
            r = await client.post(url, json={"app": app.id, "action_id": hooks.get("action_id", "")},
                                  headers=headers)
    except httpx.TimeoutException:
        raise Refused("The executor took too long; check the app.") from None
    except httpx.HTTPError:
        raise Refused("The executor couldn't be reached.") from None
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200 or not isinstance(body, dict):
        detail = body.get("detail") if isinstance(body, dict) else None
        raise Refused(f"The executor refused: {detail}" if isinstance(detail, str)
                      else f"The executor answered with an error ({r.status_code}).")
    restarted = ", ".join(str(n) for n in body.get("containers") or []) or "nothing"
    if not body.get("ok"):
        raise Refused(f"Restarted {restarted}, but: {body.get('problem') or 'it failed'}")
    health = {"ok": "it answers its health check",
              "none": "it has no health check"}.get(body.get("health"), "health unknown")
    return f"Restarted {restarted} in {body.get('seconds', '?')} s; {health}."


ACTIONS: dict[str, Action] = {a.name: a for a in (
    Action("run_routine", "low", _resolve_run, _describe_run, _run_run),
    Action("check_watch", "low", _resolve_check, _describe_check, _run_check),
    Action("set_routine", "low", _resolve_set, _describe_set, _run_set),
    Action("rerun_summary", "low", _resolve_rerun, _describe_rerun, _run_rerun),
    Action("restart_app", "high", _resolve_restart, _describe_restart, _run_restart),
)}


# ── store ────────────────────────────────────────────────────────────────────

_path: Path = DATA_DIR / "actions" / "pending.json"
_records: dict[str, dict] = {}


def _now() -> float:
    return time.time()


def load(directory: Path | None = None) -> None:
    global _path, _records
    if directory is not None:
        _path = directory / "pending.json"
        action_log.use(directory / "audit.jsonl")
    _path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(_path.read_text()) if _path.exists() else {}
        _records = data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        log.error("unreadable %s, starting empty: %s", _path.name, e)
        _records = {}
    # Approved but cut off by a restart: it may or may not have run, and never runs twice.
    for rid, r in list(_records.items()):
        if r.get("status") in ("approved", "running"):
            cut = _update(rid, status="failed",
                          result="Interrupted by a restart: it may or may not have run.")
            _event("failed", cut, problem="interrupted")


def _save() -> None:
    cutoff = _now() - KEEP_S
    kept = {k: v for k, v in _records.items() if v["created_at"] >= cutoff}
    tmp = _path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(kept, f, indent=1)
    os.replace(tmp, _path)
    _records.clear()
    _records.update(kept)


def get(action_id: str) -> dict | None:
    found = _records.get(action_id)
    return copy.deepcopy(found) if found else None


def _update(action_id: str, **fields) -> dict:
    _records[action_id] = {**_records[action_id], **fields}
    _save()
    return copy.deepcopy(_records[action_id])


def _hash(action: str, args: dict) -> str:
    raw = json.dumps({"action": action, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _event(kind: str, record: dict, **extra) -> None:
    action_log.append({"kind": kind, "id": record["id"], "action": record["action"],
                       "args": record["args"], "hash": record["hash"], **extra})


# ── propose / decide / execute ───────────────────────────────────────────────

def propose(action: str, args: dict, *, reason: str, source: str, after_reading: bool) -> dict:
    """A stored proposal waiting for the user, or Refused (text for the model)."""
    found = ACTIONS.get(action)
    if found is None:
        raise Refused(f"Unknown action {action!r}. Available: {', '.join(ACTIONS)}.")
    if source not in _SOURCES:
        raise Refused("Proposals come only from a conversation.")
    stored = found.resolve(args if isinstance(args, dict) else {})
    digest, now = _hash(action, stored), _now()
    sweep()
    waiting = [r for r in _records.values() if r["status"] in OPEN]
    if any(r["hash"] == digest for r in waiting):
        raise Refused("That is already waiting for the user's approval.")
    if len(waiting) >= MAX_PENDING:
        raise Refused(f"{MAX_PENDING} actions are already waiting for approval.")
    if sum(1 for r in _records.values() if r["created_at"] > now - 3600) >= MAX_PER_HOUR:
        raise Refused(f"No more than {MAX_PER_HOUR} proposals an hour.")
    record = {"id": uuid.uuid4().hex[:12], "action": action, "args": stored, "hash": digest,
              "risk": found.risk, "reason": " ".join(str(reason or "").split())[:_REASON_CHARS],
              "source": source, "after_reading": bool(after_reading), "status": "pending",
              "nonce": uuid.uuid4().hex[:8], "created_at": now, "expires_at": now + TTL_S,
              "message_id": None, "result": None}
    _records[record["id"]] = record
    _save()
    _event("proposed", record, risk=record["risk"], source=source,
           after_reading=record["after_reading"])
    return copy.deepcopy(record)


def set_message(action_id: str, message_id: int | None) -> None:
    if action_id in _records:
        _update(action_id, message_id=message_id)


def undelivered(action_id: str) -> None:
    """The card never reached the user: the proposal is void."""
    if action_id in _records and _records[action_id]["status"] in OPEN:
        _event("undelivered", _update(action_id, status="undelivered"))


def decide(action_id: str, verb: str, nonce: str) -> tuple[dict, str]:
    """The user's tap → (record, next). next: "run", "confirm", "rejected".
    Refused when the tap no longer applies (old card, expired, decided)."""
    record = _records.get(action_id)
    if record is None:
        raise Refused("That proposal no longer exists.")
    sweep()
    record = _records[action_id]
    if record["status"] not in OPEN:
        raise Refused(f"Already {record['status']}.")
    if nonce != record["nonce"]:
        raise Refused("That button is out of date.")
    if verb == "reject":
        rejected = _update(action_id, status="rejected", nonce="")
        _event("rejected", rejected)
        return rejected, "rejected"
    if verb == "approve" and record["status"] == "pending" and record["risk"] == "high":
        confirming = _update(action_id, status="confirming", nonce=uuid.uuid4().hex[:8])
        _event("first_tap", confirming)
        return confirming, "confirm"
    if (verb, record["status"]) in (("approve", "pending"), ("confirm", "confirming")):
        approved = _update(action_id, status="approved", nonce="")
        _event("approved", approved)
        return approved, "run"
    raise Refused("That button is out of date.")


async def execute(action_id: str, hooks: dict | None = None) -> dict:
    """Run an approved action once, after checking it again."""
    record = _records.get(action_id)
    if record is None or record["status"] != "approved":
        raise Refused("Only an approved action can run.")
    action = ACTIONS[record["action"]]
    _update(action_id, status="running")
    try:
        if _hash(record["action"], record["args"]) != record["hash"]:
            raise Refused("It changed after it was approved.")
        action.describe(record["args"])       # the target still exists
        result = await action.run(record["args"], {**(hooks or {}), "action_id": action_id})
    except Refused as e:
        failed = _update(action_id, status="failed", result=str(e))
        _event("failed", failed, problem=str(e))
        return failed
    except Exception as e:
        log.exception("action %s failed", action_id)
        failed = _update(action_id, status="failed", result=f"It failed ({type(e).__name__}).")
        _event("failed", failed, problem=type(e).__name__)
        return failed
    done = _update(action_id, status="done", result=result)
    _event("done", done)
    return done


def sweep() -> list[dict]:
    """Expire open proposals past their time; returns them (their cards need editing)."""
    now, expired = _now(), []
    for rid, r in list(_records.items()):
        if r["status"] in OPEN and r["expires_at"] <= now:
            record = _update(rid, status="expired", nonce="")
            _event("expired", record)
            expired.append(record)
    return expired


# ── what the user and the model see ──────────────────────────────────────────

def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, ZoneInfo(CFG.routine_timezone)).strftime("%H:%M")


def card_text(record: dict) -> str:
    """Written by code from the stored arguments: this is what will run."""
    try:
        title, details = ACTIONS[record["action"]].describe(record["args"])
    except Refused as e:
        title, details = record["action"], [str(e)]
    status = record["status"]
    head = {"pending": f"Proposed action · {record['risk']} risk",
            "confirming": "Are you sure? This is a high-risk action.",
            "approved": "Approved. Running…", "running": "Approved. Running…",
            "done": "Done.", "failed": "Approved, but it failed.",
            "rejected": "Rejected. Nothing was changed.",
            "expired": "Expired. Nothing was changed.",
            "undelivered": "Not delivered. Nothing was changed."}[status]
    lines = [head, title, *details]
    if record.get("result"):
        lines += ["", record["result"]]
    if status in OPEN:
        source = _SOURCES.get(record["source"], record["source"])
        lines += ["", f"Why (the model's words): “{record['reason'] or 'no reason given'}”",
                  f"Proposed from: {source}"
                  + (", after reading web pages or your library" if record["after_reading"] else ""),
                  f"Expires {_clock(record['expires_at'])}"]
    return "\n".join(lines)


def buttons(record: dict) -> list[tuple[str, str]]:
    """(label, callback data) for the card; none once decided."""
    rid, nonce = record["id"], record["nonce"]
    if record["status"] == "pending":
        return [("Approve", f"act:{rid}:approve:{nonce}"), ("Reject", f"act:{rid}:reject:{nonce}")]
    if record["status"] == "confirming":
        title, _ = ACTIONS[record["action"]].describe(record["args"])
        return [(f"Yes: {title}"[:60], f"act:{rid}:confirm:{nonce}"),
                ("Cancel", f"act:{rid}:reject:{nonce}")]
    return []


def _restart_names() -> str:
    return ", ".join(f"“{a.title}”" for a in _restartable()) or "none"


def tool() -> dict:
    """The model's only action tool. It lists the user's routines and watched pages."""
    routines = ", ".join(f"“{r['name']}”{'' if r['enabled'] else ' (paused)'}"
                         for r in routine_store.list_all()[:20]) or "none"
    watches = ", ".join(f"“{w['name']}”" for w in watch_store.list_all()[:20]) or "none"
    return {"type": "function", "function": {
        "name": "propose_action",
        "description": (
            "Ask the user to approve one action. This changes nothing: the user gets a card "
            "and the action happens only if they tap Approve. Use it only when the user asks "
            "for one of these: run_routine (run a routine now), check_watch (check a watched "
            "page now), set_routine (pause or resume a routine, with state), rerun_summary "
            "(re-run Mentat's summary: target daily or weekly), restart_app (restart one of "
            f"these apps: {_restart_names()}; only when the user asks for a restart). "
            f"Their routines: {routines}. Their watched pages: {watches}."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "target": {"type": "string", "description": (
                "The routine or watched page's name; for rerun_summary, daily or weekly; for "
                "restart_app, the app.")},
            "state": {"type": "string", "enum": ["paused", "active"],
                      "description": "For set_routine only."},
            "reason": {"type": "string", "description": "One sentence: why, in the user's terms."},
        }, "required": ["action", "target", "reason"]},
    }}


load()

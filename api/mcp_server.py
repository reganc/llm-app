"""MCP endpoint: other AI clients (Claude Code …) see the fleet and propose actions.

POST /mcp, streamable HTTP with plain JSON replies (no server-sent stream: every
tool answers at once). Same bearer key as /v1; an Origin header, when sent, must
be localhost (DNS rebinding). Four tools:

  fleet_status    the hub, read-only (titles already passed each app's privacy rule)
  list_actions    what can be proposed, with its targets
  propose_action  the action gate's propose (actions.py): sends the user a card
  action_status   where a proposal stands

There is no tool that runs or approves anything: an MCP client can only ask, and
the user decides on their phone like any other proposal. Proposals need chat
from Telegram on, since that is where the tap arrives; the card names the
client and says what it read is unknown.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import OrderedDict
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

import action_cards
import actions
import hub
import routine_store
import telegram_chat
import watch_store
from auth import verify_api_key

log = logging.getLogger("llm-api.mcp")

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")   # newest first
_SESSIONS: OrderedDict[str, str] = OrderedDict()          # session id → client name
_MAX_SESSIONS = 50
_ID = re.compile(r"^[0-9a-f]{12}$")
_LOCAL = {"localhost", "127.0.0.1", "::1"}

INSTRUCTIONS = (
    "llm-app's fleet hub. fleet_status shows how the user's apps are doing. To change "
    "something, propose_action sends the user an approval card on their phone; nothing "
    "happens unless they approve it, so never say an action is done until action_status "
    "says so. Only propose what the user asked for.")

router = APIRouter(tags=["mcp"], dependencies=[Depends(verify_api_key)])


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _tool(name: str, description: str, properties: dict | None = None,
          required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties or {},
                            "required": required or [], "additionalProperties": False}}


def tools() -> list[dict]:
    return [
        _tool("fleet_status", "How each of the user's apps is doing: up or down, warnings, "
              "deadlines. Read-only."),
        _tool("list_actions", "The actions that can be proposed, their risk and valid targets."),
        _tool("propose_action",
              "Ask the user to approve one action. Changes nothing: they get a card on their "
              "phone and it runs only if they approve (high risk: twice). Returns an id for "
              "action_status.",
              {"action": {"type": "string", "enum": list(actions.ACTIONS)},
               "target": {"type": "string", "description": "A target from list_actions."},
               "state": {"type": "string", "enum": ["paused", "active"],
                         "description": "For set_routine only."},
               "reason": {"type": "string", "description": "One sentence: why."}},
              ["action", "target", "reason"]),
        _tool("action_status", "Where a proposal stands: pending, confirming, approved, "
              "running, done, failed, rejected, expired or undelivered, with its result.",
              {"id": {"type": "string", "description": "The id propose_action returned."}}, ["id"]),
    ]


# ── tools ────────────────────────────────────────────────────────────────────

def _text(value, *, error: bool = False) -> dict:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
    return {"content": [{"type": "text", "text": text}], "isError": error}


def _targets() -> dict[str, list[str]]:
    routines = [r["name"] for r in routine_store.list_all()]
    return {"run_routine": routines, "check_watch": [w["name"] for w in watch_store.list_all()],
            "set_routine": routines, "rerun_summary": ["daily", "weekly"],
            "restart_app": [a.id for a in actions._restartable()]}


def _list_actions() -> dict:
    what = {"run_routine": "run a routine now", "check_watch": "check a watched page now",
            "set_routine": "pause or resume a routine (state: paused | active)",
            "rerun_summary": "re-run Mentat's daily or weekly summary",
            "restart_app": "restart an app's services"}
    targets = _targets()
    return {"actions": [{"action": a.name, "risk": a.risk, "what": what.get(a.name, a.name),
                         "targets": targets.get(a.name, [])} for a in actions.ACTIONS.values()]}


async def _propose(args: dict, client: str) -> dict:
    if not telegram_chat.active():
        return _text("Approvals happen in Telegram, and chat from Telegram is off in llm-app's "
                     "Settings, so nothing was proposed.", error=True)
    try:
        record = actions.propose(str(args.get("action") or ""),
                                 {"target": args.get("target"), "state": args.get("state")},
                                 reason=str(args.get("reason") or ""), source="mcp",
                                 after_reading=False, client=client)
    except actions.Refused as e:
        return _text(str(e), error=True)
    if not await action_cards.send(record):
        return _text("The approval card couldn't be sent, so nothing was proposed.", error=True)
    title = actions.card_text(record).splitlines()[1]
    return _text(f"Proposed {record['id']}: {title}. The user approves or rejects it on their "
                 "phone; nothing has happened yet. Check with action_status.")


def _status(args: dict) -> dict:
    action_id = str(args.get("id") or "")
    record = actions.get(action_id) if _ID.match(action_id) else None
    if record is None:
        return _text("No proposal has that id.", error=True)
    lines = actions.card_text(record).splitlines()
    return _text({"id": record["id"], "status": record["status"], "action": record["action"],
                  "title": lines[1] if len(lines) > 1 else record["action"],
                  "result": record.get("result")})


async def _call(name: str, args: dict, client: str) -> dict:
    if name == "fleet_status":
        return _text(hub.status_json(await hub.gather()))
    if name == "list_actions":
        return _text(_list_actions())
    if name == "propose_action":
        return await _propose(args, client)
    if name == "action_status":
        return _status(args)
    raise RpcError(-32602, f"Unknown tool {name!r}.")


# ── JSON-RPC ─────────────────────────────────────────────────────────────────

def _session(client: str) -> str:
    sid = uuid.uuid4().hex
    _SESSIONS[sid] = client
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return sid


async def _dispatch(method: str, params: dict, request: Request) -> tuple[dict, dict]:
    """(result, extra response headers)."""
    if method == "initialize":
        asked = params.get("protocolVersion")
        info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        client = str(info.get("name") or "")
        return ({"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                 "capabilities": {"tools": {"listChanged": False}},
                 "serverInfo": {"name": "llm-app", "version": "1"},
                 "instructions": INSTRUCTIONS},
                {"Mcp-Session-Id": _session(client)})
    if method == "ping":
        return {}, {}
    if method == "tools/list":
        return {"tools": tools()}, {}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            raise RpcError(-32602, "tools/call needs a name and an arguments object.")
        client = _SESSIONS.get(request.headers.get("Mcp-Session-Id", ""), "")
        return await _call(name, args, client), {}
    raise RpcError(-32601, f"Method not found: {method}")


def _error(rid, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}},
                        status_code=status)


@router.post("/mcp")
async def mcp(request: Request):
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).hostname not in _LOCAL:
        return JSONResponse({"detail": "Origin not allowed."}, status_code=403)
    try:
        msg = json.loads(await request.body())
    except ValueError:
        return _error(None, -32700, "Parse error.", 400)
    if isinstance(msg, list):
        return _error(None, -32600, "Batches aren't supported.", 400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid request.", 400)
    if "id" not in msg:                       # a notification (or a response): nothing to send back
        return Response(status_code=202)
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    try:
        result, headers = await _dispatch(msg["method"], params, request)
    except RpcError as e:
        return _error(msg["id"], e.code, str(e))
    except Exception as e:
        log.warning("mcp %s failed: %s", msg["method"], type(e).__name__)
        return _error(msg["id"], -32603, "Internal error.")
    return JSONResponse({"jsonrpc": "2.0", "id": msg["id"], "result": result}, headers=headers)


@router.get("/mcp")
@router.delete("/mcp")
async def mcp_other():
    """No server-sent stream and no session teardown: POST only."""
    return Response(status_code=405, headers={"Allow": "POST"})

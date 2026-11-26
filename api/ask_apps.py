"""Ask another app a question: `/mentat …` and `/nexus …` in Telegram.

The question goes to that app's own ask endpoint and its answer comes back as
plain text, labelled with what the app says about it (sources, confidence,
which model answered). No model here: the app answers, this only carries it.

Which apps can be asked is fleet.yaml's `ask:` (home apps only, never private
ones); the address is the entry's `summary.base`, so only host.docker.internal
is ever called, without redirects. Asking records the question in that app (its
own history: mentat's conversations, nexus's asks table) and may use that app's
paid model within its own budget. Nothing else is changed anywhere.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx

import hub
import notify

log = logging.getLogger("llm-api.ask_apps")

TIMEOUT_S = 300.0       # nexus can spend 120 s on its frontier lane, then 90 s locally
_MAX_BYTES = 1_000_000
_MAX_SOURCES = 5
_MAX_ANSWER = 3500


class AskError(Exception):
    """A reply that can be shown to the user as it is."""


def _mentat(body: dict) -> str:
    answer = notify.plain(str(body.get("answer_md") or "")).strip()
    if not answer:
        raise AskError("mentat sent an empty answer.")
    labels = [str(s.get("label") or "").strip() for s in body.get("sources") or [] if isinstance(s, dict)]
    labels = [label[:120] for label in labels if label][:_MAX_SOURCES]
    tail = []
    if body.get("degraded"):
        tail.append("Answered by mentat's local model (its synthesis model wasn't available or its budget is used up).")
    if labels:
        tail.append("Sources:\n" + "\n".join(f"[{i}] {label}" for i, label in enumerate(labels, 1)))
    return answer[:_MAX_ANSWER] + ("\n\n" + "\n\n".join(tail) if tail else "")


_NEXUS_SOURCE = {"frontier": "its frontier model", "llm": "the local model",
                 "fallback": "a plain digest of its evidence (no model answered)"}


def _nexus(body: dict) -> str:
    answer = notify.plain(str(body.get("answer") or "")).strip()
    if not answer:
        raise AskError("nexus sent an empty answer.")
    tail = []
    conf = body.get("confidence")
    how = _NEXUS_SOURCE.get(str(body.get("answer_source")), "")
    if isinstance(conf, (int, float)):
        tail.append(f"Confidence {round(float(conf) * 100)}%" + (f", from {how}" if how else "") + ".")
    elif how:
        tail.append(f"From {how}.")
    caveats = body.get("caveats")
    if isinstance(caveats, list):
        caveats = "; ".join(str(c) for c in caveats if c)
    if isinstance(caveats, str) and caveats.strip():
        tail.append(f"Caveats: {notify.plain(caveats.strip())[:600]}")
    return answer[:_MAX_ANSWER] + ("\n\n" + "\n".join(tail) if tail else "")


# kind → (path, request body, reply → text)
KINDS: dict[str, tuple[str, Callable[[str], dict], Callable[[dict], str]]] = {
    "mentat": ("/ask", lambda q: {"question": q}, _mentat),
    "nexus": ("/api/ask", lambda q: {"question": q}, _nexus),
}


def askable(apps: list[hub.App] | None = None) -> dict[str, hub.App]:
    """id → app, for the apps fleet.yaml lets you ask."""
    listed = hub.load_fleet() if apps is None else apps
    return {a.id: a for a in listed
            if a.privacy == "home" and a.ask in KINDS and hub._safe_url(a.summary.get("base"))}


async def _post_json(client: httpx.AsyncClient, url: str, payload: dict) -> Any:
    if not hub._safe_url(url):
        raise AskError("That app's address isn't allowed.")
    async with client.stream("POST", url, json=payload) as r:
        if not r.is_success:
            raise AskError(f"It answered with an error ({r.status_code}).")
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk
            if len(body) > _MAX_BYTES:
                raise AskError("Its answer was too large.")
    return json.loads(body)


async def ask(app: hub.App, question: str, client: httpx.AsyncClient | None = None) -> str:
    """The app's answer, ready to send. Failures come back as a sentence, never raised."""
    path, build, read = KINDS[app.ask or ""]
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=5.0), follow_redirects=False)
    try:
        body = await _post_json(client, f"{app.summary['base']}{path}", build(question))
        if not isinstance(body, dict):
            raise AskError("Its answer couldn't be read.")
        return f"{app.title}:\n\n{read(body)}"
    except AskError as e:
        return f"{app.title} couldn't answer. {e}"
    except httpx.TimeoutException:
        return f"{app.title} took too long to answer."
    except (httpx.HTTPError, ValueError) as e:
        log.info("ask %s: %s", app.id, type(e).__name__)
        return f"{app.title} couldn't be reached."
    finally:
        if own:
            await client.aclose()

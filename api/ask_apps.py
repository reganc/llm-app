"""Ask another app a question: `/mentat …` and `/nexus …` in Telegram.

The question goes to that app's own ask endpoint and its answer comes back as
plain text, labelled with what the app says about it (sources, confidence,
which model answered). No model here: the app answers, this only carries it.

Which apps can be asked is fleet.yaml's `ask:` (home apps only, never private
ones); the address is the entry's `summary.base`, so only host.docker.internal
is ever called, without redirects. Asking records the question in that app (its
own history: mentat's conversations, nexus's asks table) and may use that app's
paid model within its own budget. Nothing else is changed anywhere.

`/research …` asks mentat's research pipeline instead (POST /research): it
reads sources, writes a report, checks every citation resolves to a source it
gathered, and keeps the report in mentat (with a line in its audit log). The
reply relays the report, that check, and the distinct sources (one local
GET /research/{id}/cite/{claim} per sourced claim).
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
RESEARCH_TIMEOUT_S = 900.0   # ~40 s measured; revisions and a busy GPU take longer
_MAX_REPORT = 6000           # Telegram gets at most 3 parts
_MAX_CLAIM_LOOKUPS = 30
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


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    if not hub._safe_url(url):
        raise AskError("That app's address isn't allowed.")
    r = await client.get(url)
    if not r.is_success or len(r.content) > _MAX_BYTES:
        raise AskError(f"It answered with an error ({r.status_code}).")
    return r.json()


def _check_line(report: dict) -> str:
    """How the report was checked, in mentat's own terms."""
    claims = [c for c in report.get("claims") or [] if isinstance(c, dict)]
    sourced = sum(1 for c in claims if c.get("sourced"))
    review = report.get("review") if isinstance(report.get("review"), dict) else None
    parts = [f"{len(claims)} claims, {sourced} cite a source it read"
             + (f", {len(claims) - sourced} marked as inference" if len(claims) > sourced else "")]
    if review:
        share = review.get("provenance_completeness")
        if isinstance(share, (int, float)):
            parts.append(f"{round(share * 100)}% of the citations check out")
        if not review.get("passed"):
            parts.append("it did NOT pass its review (the revision limit was reached)")
        if review.get("independent_reviewer") is False:
            parts.append("reviewed by the same local model that wrote it")
    if report.get("degraded"):
        parts.append("written by mentat's local model")
    return "Checked: " + "; ".join(parts) + "."


async def _sources(client: httpx.AsyncClient, base: str, report: dict) -> list[str]:
    seen: dict[str, str] = {}
    claims = [c for c in report.get("claims") or [] if isinstance(c, dict) and c.get("cite_available")]
    for c in claims[:_MAX_CLAIM_LOOKUPS]:
        try:
            ev = await _get_json(client, f"{base}/research/{report['id']}/cite/{c.get('id')}")
        except (AskError, httpx.HTTPError, ValueError):
            continue
        title = notify.plain(str(ev.get("source_title") or "")).strip()[:120]
        url = str(ev.get("source_url") or "")
        if title and title not in seen:
            seen[title] = url if url.startswith(("http://", "https://")) else ""
    return [f"{t} {u}".strip() for t, u in seen.items()][:_MAX_SOURCES + 3]


async def research(app: hub.App, topic: str, client: httpx.AsyncClient | None = None) -> str:
    """mentat's research report on `topic`, ready to send. Failures are a sentence, never raised."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(RESEARCH_TIMEOUT_S, connect=5.0),
                                         follow_redirects=False)
    base = app.summary["base"]
    try:
        report = await _post_json(client, f"{base}/research", {"query": topic, "max_sources": 5})
        if not isinstance(report, dict) or not report.get("id"):
            raise AskError("Its report couldn't be read.")
        body = notify.plain(str(report.get("body_md") or "")).strip()
        if not body:
            raise AskError("It wrote an empty report.")
        sources = await _sources(client, base, report)
        tail = [_check_line(report)]
        if sources:
            tail.append("Sources:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1)))
        tail.append(f"Report {report['id']} is kept in {app.title}.")
        return f"{app.title} research: {topic}\n\n{body[:_MAX_REPORT]}\n\n" + "\n\n".join(tail)
    except AskError as e:
        return f"{app.title} couldn't research that. {e}"
    except httpx.TimeoutException:
        return f"{app.title} took too long; the report may still appear in {app.title}."
    except (httpx.HTTPError, ValueError) as e:
        log.info("research %s: %s", app.id, type(e).__name__)
        return f"{app.title} couldn't be reached."
    finally:
        if own:
            await client.aclose()


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

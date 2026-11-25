"""X/Twitter search via x-cli.

x-cli drives X's private web API with a saved browser login, so it fails in
ways that look like success: an expired login returns exit code 0 and a GraphQL
`errors` body, and `x-cli auth status` only reads the local file. Every failure
is therefore classified here (`XSearchError`) and surfaced, instead of being
reported as "no results".
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import memory as mem
from config import CFG

log = logging.getLogger("llm-api.x")

CREDS_PATH = os.path.expanduser("~/.x-cli/credentials.json")
TIMEOUT_S = 30.0
# While the login is known to be dead, background searches skip X for this
# long before trying again. Explicit X searches always try.
AUTH_RETRY_S = 900.0

_AUTH_RE = re.compile(r"authentication failed|not logged in|DenyLoggedOutUsers|HTTP 40[13]",
                      re.IGNORECASE)
_RATE_RE = re.compile(r"rate limit", re.IGNORECASE)
_AUTH_HELP = "X login has expired. Run `x-cli auth login` on the host, then try again."


class XSearchError(Exception):
    """kind: auth | rate_limit | timeout | error. The message is user-facing."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class _State:
    last_error: str | None = None
    auth_failed_creds: float | None = None  # credentials mtime that was rejected
    auth_failed_at: float = 0.0             # monotonic
    worked: bool = False                    # the last search succeeded


_state = _State()


def reset_state() -> None:
    global _state
    _state = _State()


def _creds_mtime() -> float | None:
    try:
        return os.stat(CREDS_PATH).st_mtime
    except OSError:
        return None


def _login_known_dead() -> bool:
    """X rejected exactly the credentials file that is still in place. A new
    login rewrites the file, which clears this without a restart."""
    return (_state.auth_failed_creds is not None
            and _state.auth_failed_creds == _creds_mtime())


def status() -> dict:
    binary_found = bool(shutil.which(CFG.x_cli_path) or os.path.isfile(CFG.x_cli_path))
    creds_found = os.path.isfile(CREDS_PATH)
    dead = _login_known_dead()
    return {
        "enabled": CFG.x_search_enabled,
        "binary_found": binary_found,
        "creds_found": creds_found,
        "auth_ok": False if dead else (True if _state.worked else None),  # None = not checked
        "last_error": _state.last_error,
        "ready": CFG.x_search_enabled and binary_found and creds_found and not dead,
    }


def _parse(data: dict) -> list[dict]:
    node: object = data
    for key in ("data", "search_by_raw_query", "search_timeline", "timeline"):
        node = node.get(key) if isinstance(node, dict) else None
    instructions = (node.get("instructions") if isinstance(node, dict) else None) or []
    out = []
    for inst in instructions:
        if inst.get("type") != "TimelineAddEntries":
            continue
        for entry in inst.get("entries", []):
            try:
                item = entry["content"]["itemContent"]
                if item.get("itemType") != "TimelineTweet":
                    continue
                tweet = item["tweet_results"]["result"]
                if tweet.get("__typename") != "Tweet":
                    continue
                legacy = tweet.get("legacy", {})
                ucore = (tweet.get("core", {})
                              .get("user_results", {})
                              .get("result", {})
                              .get("core", {}))
                screen = ucore.get("screen_name", "unknown")
                name = ucore.get("name", screen)
                tweet_id = tweet.get("rest_id", "")
                full_text = legacy.get("full_text", "").strip()
                if not full_text or not tweet_id:
                    continue
                created = legacy.get("created_at", "")
                likes = legacy.get("favorite_count", 0)
                rts = legacy.get("retweet_count", 0)
                views = tweet.get("views", {}).get("count", "0")
                url = f"https://x.com/{screen}/status/{tweet_id}"
                stats = f"♥ {likes}  RT {rts}  views {views}  · {created}"
                text = f"{full_text}\n\n{stats}"
                out.append({
                    "title": f"@{screen} ({name})",
                    "url": url,
                    "snippet": full_text,
                    "text": text,
                    "engine": "x_twitter",
                    "char_count": len(text),
                    "created_at": created,
                    "likes": int(likes),
                    "retweets": int(rts),
                })
            except (KeyError, TypeError):
                continue
    return out


def _classify(detail: str) -> XSearchError:
    if _AUTH_RE.search(detail):
        return XSearchError("auth", _AUTH_HELP)
    if _RATE_RE.search(detail):
        return XSearchError("rate_limit", "X is rate limiting this account. Try again later.")
    return XSearchError("error", f"X search failed: {detail.strip()[:200] or 'unknown error'}")


async def _run_cli(query: str, count: int) -> list[dict]:
    try:
        proc = await asyncio.create_subprocess_exec(
            # "--" so a query starting with "-" is never read as a flag.
            CFG.x_cli_path, "search", "--type", "latest",
            "--json", "--count", str(count), "--", query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise XSearchError("error", f"x-cli could not be started: {e}") from e
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise XSearchError("timeout", f"X search timed out after {TIMEOUT_S:.0f}s.") from e
    if proc.returncode != 0:
        # Cobra prints "Error: …" first, then the usage text.
        lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise _classify(lines[0] if lines else "")
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise XSearchError("error", "X search returned an unreadable response.") from e
    if not isinstance(data, dict):
        raise XSearchError("error", "X search returned an unexpected response.")
    posts = _parse(data)
    errors = data.get("errors")
    if errors and not posts:  # exit code 0, but nothing usable came back
        raise _classify("; ".join(str(e.get("message", "")) if isinstance(e, dict) else str(e)
                                  for e in errors))
    return posts


async def fetch_x(query: str, count: int | None = None) -> list[dict]:
    """Search X. Returns posts (possibly none); raises XSearchError on failure."""
    global _state
    creds = _creds_mtime()  # before the call: a login during it must not be blamed
    try:
        posts = await _run_cli(query, count or CFG.x_search_count)
    except XSearchError as e:
        if e.kind == "auth":
            _state = _State(str(e), creds, time.monotonic())
        else:
            _state = _State(str(e), _state.auth_failed_creds, _state.auth_failed_at)
        log.warning("x search failed (%s): %s", e.kind, e)
        raise
    _state = _State(worked=True)
    return posts


async def search_x(query: str, count: int | None = None) -> list[dict]:
    """Best-effort variant that augments a web search: never raises."""
    if not CFG.x_search_enabled:
        return []
    if _login_known_dead() and time.monotonic() - _state.auth_failed_at < AUTH_RETRY_S:
        return []
    try:
        return await fetch_x(query, count)
    except XSearchError:
        return []  # logged and recorded in status() by fetch_x


async def search_x_and_ingest(query: str, *, store_memory: bool = True) -> dict:
    if not CFG.x_search_enabled:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "X search disabled"}
    try:
        items = await fetch_x(query)
    except XSearchError as e:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": str(e), "error_kind": e.kind}
    if not items:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "No X results"}
    stored = 0
    if store_memory:
        tasks = [
            mem.store_knowledge(
                text=r["text"][:15000],
                title=r["title"],
                source_type="x_twitter",
                identifier=r["url"],
                extra={"search_query": query[:200]},
            )
            for r in items if len(r.get("text", "")) > 10
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stored = sum(1 for r in results
                     if isinstance(r, dict) and r.get("chunks_stored", 0) > 0)

    lines = [f"── X/TWITTER SEARCH: {query} ──"]
    for i, r in enumerate(items[:10], 1):
        preview = r.get("text", "")[:400]
        lines.append(f"[X{i}] {r['title']}\nURL: {r['url']}\n{preview}")
    lines.append("──")
    return {
        "query": query,
        "results": [{"title": r["title"], "url": r["url"],
                     "char_count": r.get("char_count", 0),
                     "engine": "x_twitter"} for r in items],
        "context_text": "\n\n".join(lines),
        "stored": stored,
        "source": "x_twitter",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

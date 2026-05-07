"""X/Twitter search via x-cli."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone

import memory as mem
from config import CFG

log = logging.getLogger("llm-api.x")


def status() -> dict:
    binary_found = bool(shutil.which(CFG.x_cli_path) or os.path.isfile(CFG.x_cli_path))
    creds = os.path.expanduser("~/.x-cli/credentials.json")
    creds_found = os.path.isfile(creds)
    return {
        "enabled": CFG.x_search_enabled,
        "binary_found": binary_found,
        "creds_found": creds_found,
        "ready": CFG.x_search_enabled and binary_found and creds_found,
    }


def _parse(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    instructions = (
        data.get("data", {})
            .get("search_by_raw_query", {})
            .get("search_timeline", {})
            .get("timeline", {})
            .get("instructions", [])
    )
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


async def search_x(query: str, count: int | None = None) -> list[dict]:
    if not CFG.x_search_enabled:
        return []
    count = count or CFG.x_search_count
    try:
        proc = await asyncio.create_subprocess_exec(
            CFG.x_cli_path, "search", query, "--type", "latest",
            "--json", "--count", str(count),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        if proc.returncode != 0:
            return []
        return _parse(stdout.decode("utf-8", errors="replace"))
    except Exception as e:
        log.warning("x search failed: %s", e)
        return []


async def search_x_and_ingest(query: str, *, store_memory: bool = True) -> dict:
    if not CFG.x_search_enabled:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "X search disabled"}
    items = await search_x(query)
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
        lines.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{preview}")
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

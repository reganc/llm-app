"""Web search: SearXNG → DuckDuckGo fallback, plus auxiliary sites and auto-search decision."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import httpx

import memory as mem
import ollama as oll
from config import CFG, AUX_SITES_FILE
from extract import _firecrawl, _trafilatura

log = logging.getLogger("llm-api.search")

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; llm-research-bot/1.0)"}
_SEARXNG_HEADERS = {**_HEADERS, "X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"}


# ── Auxiliary sites ──────────────────────────────────────────────────────────
def _load_aux() -> list[dict]:
    env_seeds = [
        {"url": u.strip(), "label": u.strip()}
        for u in (
            __import__("os").getenv("AUXILIARY_SITES", "").split(",")
        ) if u.strip()
    ]
    try:
        saved = json.loads(AUX_SITES_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        saved = []
    urls = {s["url"] for s in saved}
    return saved + [s for s in env_seeds if s["url"] not in urls]


_aux_sites: list[dict] = _load_aux()


def _save_aux() -> None:
    AUX_SITES_FILE.write_text(json.dumps(_aux_sites, indent=2))


def list_aux_sites() -> list[dict]:
    return list(_aux_sites)


def add_aux_site(url: str, label: str = "") -> dict:
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    for s in _aux_sites:
        if s["url"] == url:
            return s
    entry = {"url": url, "label": label.strip() or url}
    _aux_sites.append(entry)
    _save_aux()
    return entry


def remove_aux_site(url: str) -> bool:
    url = url.strip()
    before = len(_aux_sites)
    _aux_sites[:] = [s for s in _aux_sites if s["url"] != url]
    removed = len(_aux_sites) < before
    if removed:
        _save_aux()
    return removed


# ── SearXNG / DDG ────────────────────────────────────────────────────────────
async def searxng_search(query: str, num: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_SEARXNG_HEADERS) as client:
            r = await client.get(
                f"{CFG.searxng_url}/search",
                params={"q": query, "format": "json",
                        "engines": "google,bing,duckduckgo", "lang": "en"},
            )
            r.raise_for_status()
            results = r.json().get("results", [])[:num]
            return [
                {"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("content", ""),
                 "engine": x.get("engine", "searxng")}
                for x in results if x.get("url")
            ]
    except Exception as e:
        log.warning("searxng failed: %s — falling back to DDG", e)
        return await _ddg(query, num)


async def _ddg(query: str, num: int) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_HEADERS,
                                     follow_redirects=True) as client:
            r = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
            r.raise_for_status()
            html = r.text
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)
        out = []
        for i, (url, title) in enumerate(links[:num]):
            if "uddg=" in url:
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    url = urllib.parse.unquote(m.group(1))
            out.append({"title": title.strip(), "url": url,
                        "snippet": snippets[i].strip() if i < len(snippets) else "",
                        "engine": "duckduckgo_fallback"})
        return out
    except Exception as e:
        log.warning("ddg fallback failed: %s", e)
        return []


# ── Page fetch ───────────────────────────────────────────────────────────────
async def _fetch_text(url: str) -> Optional[dict]:
    return await _firecrawl(url) or await _trafilatura(url)


async def fetch_aux_sites() -> list[dict]:
    if not _aux_sites:
        return []
    sem = asyncio.Semaphore(3)

    async def one(site):
        async with sem:
            r = await _fetch_text(site["url"])
            if r and r.get("text"):
                return {"title": r["title"] or site.get("label", site["url"]),
                        "url": site["url"], "snippet": r["text"][:300],
                        "text": r["text"], "char_count": r["char_count"],
                        "engine": "auxiliary_site"}
            return None

    items = await asyncio.gather(*[one(s) for s in _aux_sites])
    return [i for i in items if i]


# ── Main pipeline ────────────────────────────────────────────────────────────
async def search_and_ingest(query: str, *, num_results: int | None = None,
                            store_memory: bool = True,
                            include_x: bool = True) -> dict:
    if not CFG.search_enabled:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "Search disabled"}

    num = num_results or CFG.search_results
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    from x_search import search_x  # local import to avoid cycle

    raw, aux, x_items = await asyncio.gather(
        searxng_search(query, num),
        fetch_aux_sites(),
        search_x(query) if (include_x and CFG.x_search_enabled) else _empty(),
    )

    if not (raw or aux or x_items):
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "No search results returned"}

    sem = asyncio.Semaphore(3)

    async def enrich(r):
        async with sem:
            f = await _fetch_text(r["url"])
            if f:
                r["text"] = f["text"]
                r["title"] = f["title"] or r["title"]
                r["char_count"] = f["char_count"]
            else:
                r["text"] = r.get("snippet", "")
                r["char_count"] = len(r["text"])
            return r

    web = [r for r in await asyncio.gather(*[enrich(r) for r in raw]) if r.get("text", "").strip()]
    web += aux

    stored = 0
    if store_memory and (web or x_items):
        tasks = [
            mem.store_knowledge(
                text=r["text"][:15000],
                title=r["title"],
                source_type=r.get("engine", "web_search"),
                identifier=r["url"],
                extra={"search_query": query[:200]},
            )
            for r in (web + x_items) if len(r.get("text", "")) > 100
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stored = sum(1 for r in results
                     if isinstance(r, dict) and r.get("chunks_stored", 0) > 0)

    # Build context block
    query_items = [r for r in web if r.get("engine") != "auxiliary_site"]
    aux_only = [r for r in web if r.get("engine") == "auxiliary_site"]

    lines = [f"── LIVE SEARCH RESULTS for: {query} (retrieved {retrieved_at}) ──"]
    if query_items:
        lines.append("[ WEB SEARCH RESULTS ]")
        for i, r in enumerate(query_items[:6], 1):
            preview = r.get("text", "")[:1500]
            tail = "…" if len(r.get("text", "")) > 1500 else ""
            lines.append(f"[W{i}] {r['title']}\nURL: {r['url']}\n{preview}{tail}")
    if x_items:
        lines.append("[ X / TWITTER — LATEST POSTS ]")
        for i, r in enumerate(x_items[:6], 1):
            preview = r.get("text", r.get("snippet", ""))[:400]
            lines.append(f"[X{i}] {r['title']}\nURL: {r['url']}\n{preview}")
    if aux_only:
        lines.append("[ AUXILIARY SITES ]")
        for i, r in enumerate(aux_only[:3], 1):
            preview = r.get("text", "")[:300]
            lines.append(f"[A{i}] {r['title']}\nURL: {r['url']}\n{preview}")
    lines.append("── END SEARCH RESULTS ──")
    context_text = "\n\n".join(lines)

    all_results = web + x_items
    sources = []
    if web:
        sources.append("duckduckgo_fallback" if any("fallback" in r.get("engine", "") for r in web) else "searxng")
    if x_items:
        sources.append("x_twitter")

    return {
        "query": query,
        "results": [{"title": r["title"], "url": r["url"],
                     "char_count": r.get("char_count", 0),
                     "engine": r.get("engine", "web")}
                    for r in all_results],
        "context_text": context_text,
        "stored": stored,
        "source": "+".join(sources) if sources else "unknown",
        "retrieved_at": retrieved_at,
    }


async def _empty() -> list:
    return []


# ── Auto-search decision ─────────────────────────────────────────────────────
_CLASSIFIER_PROMPT = """\
You are a query router. Decide whether the question below requires CURRENT information \
(data that changes over time and would be stale if more than a day or two old) or can be \
answered from HISTORICAL knowledge.

Rules:
- Prices, rates, scores, rankings, market data → CURRENT
- Ongoing events, missions, elections, conflicts → CURRENT
- Schedules, availability, "will X happen" → CURRENT
- Roles/positions of living people → CURRENT
- News, recent announcements → CURRENT
- Historical events, definitions, science, math → HISTORICAL

Respond with JSON only: {{"needs_current": true_or_false}}

Question: {query}"""


_HEDGE_RE = re.compile(
    r'\b(as of my (last|latest|most recent) (update|training|knowledge)|'
    r'my (knowledge|training) (cutoff|data)|'
    r'I (don\'t|do not) have (access to )?(real.?time|current|live|up.to.date)|'
    r'I (can\'t|cannot|may not) (verify|confirm|guarantee)|'
    r'(please|you (should|may want to)) (verify|check|confirm|consult)|'
    r'(may|might) not be (accurate|current|up.to.date)|'
    r'I (am|\'m) not (sure|certain|confident))\b',
    re.IGNORECASE,
)


async def _classify_freshness(query: str) -> bool:
    payload = {
        "model": oll.normalize_model(CFG.default_model),
        "messages": [{"role": "user", "content": _CLASSIFIER_PROMPT.format(query=query.strip())}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 60},
    }
    try:
        async with httpx.AsyncClient(timeout=CFG.classifier_timeout) as client:
            r = await client.post(f"{CFG.ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            text = r.json()["message"]["content"].strip()
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return bool(json.loads(m.group()).get("needs_current", True))
    except Exception:
        pass
    return True


async def _probe_hedging(query: str) -> bool:
    payload = {
        "model": oll.normalize_model(CFG.default_model),
        "prompt": query.strip(),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 50},
    }
    try:
        async with httpx.AsyncClient(timeout=CFG.classifier_timeout) as client:
            r = await client.post(f"{CFG.ollama_url}/api/generate", json=payload)
            r.raise_for_status()
            return bool(_HEDGE_RE.search(r.json().get("response", "")))
    except Exception:
        return False


async def should_auto_search(memory_chunks: list[dict], query: str = "") -> bool:
    if not CFG.search_enabled:
        return False
    if CFG.search_always:
        return True
    if not query:
        return not memory_chunks
    if not memory_chunks:
        c, p = await asyncio.gather(_classify_freshness(query), _probe_hedging(query))
        return c or p
    best = max((c.get("score", 0) for c in memory_chunks), default=0)
    if best < CFG.auto_search_threshold:
        return True
    c, p = await asyncio.gather(_classify_freshness(query), _probe_hedging(query))
    return c or p


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{CFG.searxng_url}/healthz")
            return r.status_code == 200
    except Exception:
        return False

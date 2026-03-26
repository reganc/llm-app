"""
web_search.py — Agentic web search for the LLM API.

Uses self-hosted SearXNG (private, unlimited, no API key).
Falls back to DuckDuckGo HTML scraping if SearXNG is unreachable.

Flow:
  search_and_ingest(query) →
    1. SearXNG/DDG → list of URLs + snippets
    2. Fetch full article text via trafilatura (parallel)
    3. Store in ChromaDB memory
    4. Return context_text ready for prompt injection

Auto-trigger: if best memory chunk score < AUTO_SEARCH_THRESHOLD → search fires.
Manual trigger: user prefixes message with /search
"""

import asyncio
import os
import re
import urllib.parse
from typing import Optional

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
SEARXNG_URL           = os.getenv("SEARXNG_URL",           "http://searxng:8080")
AUTO_SEARCH_THRESHOLD = float(os.getenv("AUTO_SEARCH_THRESHOLD", "0.35"))
SEARCH_RESULTS        = int(os.getenv("SEARCH_RESULTS",    "5"))
SEARCH_FETCH_TIMEOUT  = int(os.getenv("SEARCH_FETCH_TIMEOUT", "15"))
SEARCH_ENABLED        = os.getenv("SEARCH_ENABLED", "true").lower() == "true"
FIRECRAWL_URL         = os.getenv("FIRECRAWL_URL", "").rstrip("/")
FIRECRAWL_API_KEY     = os.getenv("FIRECRAWL_API_KEY", "fc-local")

# Auxiliary sites — always fetched on every search alongside normal results.
# Populated at runtime via the API (/v1/search/auxiliary-sites).
# Can be pre-seeded with a comma-separated env var: AUXILIARY_SITES=https://a.com,https://b.com
_auxiliary_sites: list[dict] = [
    {"url": u.strip(), "label": u.strip()}
    for u in os.getenv("AUXILIARY_SITES", "").split(",")
    if u.strip()
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; llm-research-bot/1.0)"}

# SearXNG bot-detection requires these headers to be present
_SEARXNG_HEADERS = {
    **_HEADERS,
    "X-Forwarded-For": "127.0.0.1",
    "X-Real-IP":       "127.0.0.1",
}


# ── SearXNG ───────────────────────────────────────────────────────────────────
async def searxng_search(query: str, num: int = SEARCH_RESULTS) -> list[dict]:
    """Query SearXNG, return list of {title, url, snippet, engine}."""
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_SEARXNG_HEADERS) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json",
                        "engines": "google,bing,duckduckgo", "lang": "en"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:num]
            return [
                {"title":   r.get("title", ""),
                 "url":     r.get("url", ""),
                 "snippet": r.get("content", ""),
                 "engine":  r.get("engine", "searxng")}
                for r in results if r.get("url")
            ]
    except Exception:
        return await _ddg_fallback(query, num)


async def _ddg_fallback(query: str, num: int = 5) -> list[dict]:
    """DuckDuckGo HTML fallback — no key, rate-limited."""
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_HEADERS,
                                      follow_redirects=True) as client:
            resp = await client.get("https://html.duckduckgo.com/html/",
                                     params={"q": query})
            resp.raise_for_status()
            html = resp.text

        results, snippets = [], []
        links    = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)

        for i, (url, title) in enumerate(links[:num]):
            if "uddg=" in url:
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    url = urllib.parse.unquote(m.group(1))
            results.append({
                "title":   title.strip(),
                "url":     url,
                "snippet": snippets[i].strip() if i < len(snippets) else "",
                "engine":  "duckduckgo_fallback",
            })
        return results
    except Exception:
        return []


# ── Auxiliary sites management ────────────────────────────────────────────────
def get_auxiliary_sites() -> list[dict]:
    return list(_auxiliary_sites)

def add_auxiliary_site(url: str, label: str = "") -> dict:
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    # Deduplicate
    for site in _auxiliary_sites:
        if site["url"] == url:
            return site
    entry = {"url": url, "label": label.strip() or url}
    _auxiliary_sites.append(entry)
    return entry

def remove_auxiliary_site(url: str) -> bool:
    url = url.strip()
    before = len(_auxiliary_sites)
    _auxiliary_sites[:] = [s for s in _auxiliary_sites if s["url"] != url]
    return len(_auxiliary_sites) < before


async def fetch_auxiliary_sites(sites: list[dict] | None = None) -> list[dict]:
    """
    Fetch the current auxiliary sites list and return them as search result dicts.
    Each site is fetched with full text extraction, just like a normal search result.
    """
    targets = sites if sites is not None else _auxiliary_sites
    if not targets:
        return []

    sem = asyncio.Semaphore(3)

    async def fetch_one(site: dict) -> dict | None:
        async with sem:
            result = await fetch_url_text(site["url"])
            if result:
                return {
                    "title":      result["title"] or site.get("label", site["url"]),
                    "url":        site["url"],
                    "snippet":    result["text"][:300],
                    "text":       result["text"],
                    "char_count": result["char_count"],
                    "engine":     "auxiliary_site",
                }
            # Return a stub with label even if fetch failed
            return {
                "title":   site.get("label", site["url"]),
                "url":     site["url"],
                "snippet": "",
                "text":    "",
                "engine":  "auxiliary_site",
            }

    results = await asyncio.gather(*[fetch_one(s) for s in targets])
    # Only keep sites where text was actually extracted
    return [r for r in results if r and r.get("text", "").strip()]



async def _firecrawl_scrape(url: str) -> Optional[dict]:
    """Scrape a URL via local Firecrawl — handles JS, bot protection, paywalls."""
    if not FIRECRAWL_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=SEARCH_FETCH_TIMEOUT + 10) as client:
            resp = await client.post(
                f"{FIRECRAWL_URL}/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                         "Content-Type": "application/json"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", {})
            text = data.get("markdown", "")
            if not text or len(text) < 100:
                return None
            meta  = data.get("metadata", {})
            title = (meta.get("title") or meta.get("ogTitle") or url)[:200]
            return {"url": url, "title": title, "text": text, "char_count": len(text)}
    except Exception:
        return None


async def fetch_url_text(url: str) -> Optional[dict]:
    """Fetch URL and extract clean text. Tries Firecrawl first, falls back to trafilatura."""
    # Try Firecrawl (better JS rendering, handles bot protection)
    result = await _firecrawl_scrape(url)
    if result:
        return result

    # Fallback: trafilatura
    try:
        import trafilatura
        async with httpx.AsyncClient(timeout=SEARCH_FETCH_TIMEOUT,
                                      follow_redirects=True,
                                      headers=_HEADERS) as client:
            resp = await client.get(url)
            if resp.status_code not in (200, 203):
                return None
            html = resp.text

        text = trafilatura.extract(html, include_comments=False,
                                    include_tables=True, no_fallback=False)
        if not text or len(text) < 100:
            return None

        meta  = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else url)[:200]
        return {"url": url, "title": title, "text": text, "char_count": len(text)}
    except Exception:
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def search_and_ingest(
    query: str,
    num_results: int = SEARCH_RESULTS,
    store_memory: bool = True,
) -> dict:
    """
    Full search → fetch → store pipeline.
    Returns {query, results, context_text, stored, source, error?}
    context_text is ready to inject directly into a prompt.
    """
    if not SEARCH_ENABLED:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "Search disabled (SEARCH_ENABLED=false)"}

    # 1 — search (web + auxiliary sites in parallel)
    raw, aux_items = await asyncio.gather(
        searxng_search(query, num=num_results),
        fetch_auxiliary_sites(),
    )
    if not raw and not aux_items:
        return {"query": query, "results": [], "context_text": "",
                "stored": 0, "error": "No search results returned"}

    # 2 — fetch full text in parallel (max 3 concurrent to be polite)
    sem = asyncio.Semaphore(3)

    async def guarded(r):
        async with sem:
            fetched = await fetch_url_text(r["url"])
            if fetched:
                r["text"]  = fetched["text"]
                r["title"] = fetched["title"] or r["title"]
                r["char_count"] = fetched["char_count"]
            else:
                r["text"]  = r["snippet"]
                r["char_count"] = len(r["snippet"])
            return r

    items = await asyncio.gather(*[guarded(r) for r in raw])
    items = [r for r in items if r.get("text", "").strip()]

    # Merge auxiliary site results (already fetched — prepend so they appear first)
    items = aux_items + items

    # 3 — store in ChromaDB
    stored = 0
    if store_memory and items:
        try:
            import memory as mem
            tasks = [
                mem.store_knowledge(
                    text        = r["text"][:15000],
                    title       = r["title"],
                    source_type = "web_search",
                    identifier  = r["url"],
                    extra_meta  = {"search_query": query[:200]},
                )
                for r in items if len(r.get("text", "")) > 100
            ]
            res = await asyncio.gather(*tasks, return_exceptions=True)
            stored = sum(
                1 for r in res
                if isinstance(r, dict) and r.get("chunks_stored", 0) > 0
            )
        except Exception:
            pass

    # 4 — build prompt context block
    lines = [f"── WEB SEARCH: {query} ──────────────────────────────────────"]
    for i, r in enumerate(items[:5], 1):
        preview = r.get("text", r.get("snippet", ""))[:600]
        lines.append(
            f"[{i}] {r['title']}\nURL: {r['url']}\n{preview}"
            + ("…" if len(r.get("text","")) > 600 else "")
        )
    lines.append("─────────────────────────────────────────────────────────")
    context_text = "\n\n".join(lines)

    engine = items[0].get("engine", "") if items else ""
    return {
        "query":        query,
        "results":      [{"title": r["title"], "url": r["url"],
                           "char_count": r.get("char_count", 0)} for r in items],
        "context_text": context_text,
        "stored":       stored,
        "source":       "duckduckgo_fallback" if "fallback" in engine else "searxng",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
_TIME_SENSITIVE_RE = re.compile(
    r'\b(today|current(ly)?|right now|live|latest|recent(ly)?|just (happened|announced|released|broke)|'
    r'this (week|month|year)|breaking|now|price|stock|weather|score|news)\b',
    re.IGNORECASE,
)

def is_time_sensitive(query: str) -> bool:
    """True when the query clearly needs fresh/current data."""
    return bool(_TIME_SENSITIVE_RE.search(query))


def should_auto_search(memory_chunks: list[dict], query: str = "") -> bool:
    """True when memory is too weak to answer and web search would help."""
    if not SEARCH_ENABLED:
        return False
    if not memory_chunks:
        return True
    if query and is_time_sensitive(query):
        return True
    best = max((c.get("score", 0) for c in memory_chunks), default=0)
    return best < AUTO_SEARCH_THRESHOLD


async def is_available() -> bool:
    """Ping SearXNG healthcheck."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{SEARXNG_URL}/healthz")
            return r.status_code == 200
    except Exception:
        return False

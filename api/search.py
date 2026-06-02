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


# ── Query-focused excerpting ────────────────────────────────────────────────
# Stop-words we don't count when scoring paragraph relevance.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "of", "to", "in", "on", "at",
    "by", "for", "with", "as", "from", "up", "down", "and", "or", "but",
    "if", "then", "than", "this", "that", "these", "those", "what", "when",
    "where", "why", "how", "which", "who", "whom", "i", "you", "we", "they",
    "it", "its", "his", "her", "their", "my", "our", "your", "would",
    "could", "should", "will", "shall", "may", "might", "can", "today",
    "current", "currently", "latest", "now",
}

# Words in the user's query that signal they want a numeric / factual datum
# — when present, paragraphs containing currency symbols, units, or digit
# clusters are heavily up-weighted.
_NUMERIC_INTENT_RE = re.compile(
    r"\b("
    r"price|cost|rate|spot|level|value|worth|"
    r"yield|return|change|"
    r"close|open|high|low|"
    r"score|count|share|stake|cap(italization)?|"
    r"how (much|many|long|tall|big|deep|fast|hot|cold)|"
    r"what(?:'?s|s| is| was| were) the"
    r")\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CURRENCY_RE = re.compile(
    r"[$€£¥₹]|\bUSD\b|\bEUR\b|\bGBP\b|\bJPY\b|\bCNY\b|"
    r"\b(?:per|/)\s*(?:oz|ounce|gram|kg|lb|barrel|share|unit)\b",
    re.IGNORECASE,
)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")

# Phrases that mark a paragraph as describing CURRENT state (boost when the
# query has numeric intent, since we want the latest price, not a record).
_PRESENT_TENSE_RE = re.compile(
    r"\b(now|today|currently|current|spot|live|real[-\s]?time|"
    r"as of (?:today|now|\d{1,2})|right now|latest|"
    r"intraday|moments? ago|minutes? ago|hour[s]? ago)\b",
    re.IGNORECASE,
)
# Phrases that mark a paragraph as describing HISTORICAL data (penalize for
# numeric-intent queries — the user asked "what is X now", not "what was X").
# NOTE: bare year references (e.g. "in 2026") are matched separately at scoring
# time so we can compare against the CURRENT year — penalizing only past years.
_HISTORY_RE = re.compile(
    r"\b("
    r"back in \d{4}|"
    r"during the \d{4}s|"
    r"\bhistory\b|historical|historically|"
    r"all[-\s]?time (?:high|low|record)|"
    r"nominal (?:high|low|record)|"
    r"milestone|previously|past (?:year|decade|century)|"
    r"\d+ years? ago|years? ago, (?:gold|silver|platinum|the price)"
    r")\b",
    re.IGNORECASE,
)
# Bare years — checked against current year, penalty only when the year is
# older than today. "In 2026" on a 2026-current page is fine; "in 2020" is
# clearly retrospective.
_BARE_YEAR_RE = re.compile(r"\b(?:in|during|back in|since)\s+((?:19|20)\d{2})\b",
                           re.IGNORECASE)


def _query_terms(query: str) -> list[str]:
    """Significant lowercase query terms (≥2 chars, non-stopword)."""
    if not query:
        return []
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", query.lower())
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def _current_year() -> int:
    """Today's year, computed lazily so the value is fresh per call."""
    return datetime.now(timezone.utc).year


def _focused_excerpt(text: str, query: str, target_chars: int) -> str:
    """Return a relevance-ranked excerpt of ``text`` ≤ ``target_chars`` chars.

    Strategy: split into paragraphs; score each by query-term overlap and
    (when the query has numeric intent) by digit/currency/unit density;
    greedily pick highest scorers, then re-emit them in source order with
    " […] " between non-adjacent picks. This avoids the head-truncation
    pathology where nav, cookie banners, and marketing copy occupy the first
    1500 chars and the actual datum (price, level, status) sits below the
    fold.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= target_chars:
        return text

    paras = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    if len(paras) <= 1:
        # Fallback: split on single newlines so single-block pages still excerpt.
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not paras:
        return text[:target_chars]

    terms = _query_terms(query)
    numeric_intent = bool(_NUMERIC_INTENT_RE.search(query)) if query else False
    this_year = _current_year() if numeric_intent else 0

    scored: list[tuple[int, float, str]] = []
    for idx, p in enumerate(paras):
        plow = p.lower()
        # Overlap: count distinct query terms present (not raw frequency, so
        # a paragraph repeating one keyword doesn't drown out one that
        # mentions several).
        hit_terms = sum(1 for t in terms if t in plow)
        score = hit_terms * 3.0
        if numeric_intent:
            n_numbers = len(_NUMBER_RE.findall(p))
            n_currency = len(_CURRENCY_RE.findall(p))
            score += min(n_numbers, 6) * 1.0
            score += n_currency * 2.5
            # Recency bias: when the user wants the LATEST/CURRENT value,
            # heavily favor paragraphs marked as present-tense and demote
            # paragraphs that read as historical narrative. Live-price
            # tracker pages bury the live number in JS-rendered tables,
            # so trafilatura's static-HTML extract is dominated by
            # historical prose. Without this bias, those win on numeric
            # density and the model gets fed only old data.
            if _PRESENT_TENSE_RE.search(p):
                score += 4.0
            history_hits = len(_HISTORY_RE.findall(p))
            if history_hits:
                score -= 2.0 * history_hits
            # Year-based penalty — only when the year is OLDER than today.
            # "in 2026" on a 2026-current page is the live context, not a
            # retrospective; we don't want to bury current-year paragraphs.
            for ym in _BARE_YEAR_RE.finditer(p):
                year = int(ym.group(1))
                if year < this_year:
                    # Older years scale up the penalty linearly with age.
                    score -= 1.5 + min(this_year - year, 30) * 0.15
        # Slight bias toward shorter paragraphs (more datum, less prose).
        if 40 <= len(p) <= 600:
            score += 0.5
        # Light recency-of-position bias to break ties: earlier paras
        # win when scores tie, since news pages put the lede up top.
        score -= idx * 0.01
        scored.append((idx, score, p))

    # Always include any paragraph that scored > 0; if budget allows, fill
    # with high-scoring remainder. If NOTHING scored, fall back to the head.
    positive = [s for s in scored if s[1] > 0.5]
    if not positive:
        return text[:target_chars]
    positive.sort(key=lambda s: s[1], reverse=True)

    picked: dict[int, str] = {}
    used = 0
    for idx, _score, p in positive:
        # +6 accounts for joiner " […] " between non-adjacent.
        cost = len(p) + 6
        if used + cost > target_chars and picked:
            break
        picked[idx] = p
        used += cost
        if used >= target_chars:
            break

    if not picked:
        return text[:target_chars]
    out_parts: list[str] = []
    last_idx = -2
    for idx in sorted(picked):
        if last_idx >= 0 and idx != last_idx + 1:
            out_parts.append("[…]")
        out_parts.append(picked[idx])
        last_idx = idx
    excerpt = "\n\n".join(out_parts)
    if len(excerpt) > target_chars:
        excerpt = excerpt[:target_chars].rstrip() + "…"
    return excerpt


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

    # Per-source budgets, tuned for the typical 8K-token model context.
    # 9 web × 2800 + 10 X × 500 + 3 aux × 700 ≈ 32K chars (~8K tokens) of
    # raw content; ``_focused_excerpt`` skips nav/marketing copy so the kept
    # text is high-density. Bumped from the prior head-truncation defaults
    # (1500/400/300 × 6/6/3) which dropped mid-page price/datum tables off
    # the cliff.
    lines = [f"── LIVE SEARCH RESULTS for: {query} (retrieved {retrieved_at}) ──"]
    if query_items:
        lines.append("[ WEB SEARCH RESULTS ]")
        for i, r in enumerate(query_items[:9], 1):
            preview = _focused_excerpt(r.get("text", ""), query, 2800)
            lines.append(f"[W{i}] {r['title']}\nURL: {r['url']}\n{preview}")
    if x_items:
        lines.append("[ X / TWITTER — LATEST POSTS ]")
        for i, r in enumerate(x_items[:10], 1):
            preview = _focused_excerpt(r.get("text", r.get("snippet", "")) or "",
                                       query, 500)
            lines.append(f"[X{i}] {r['title']}\nURL: {r['url']}\n{preview}")
    if aux_only:
        lines.append("[ AUXILIARY SITES ]")
        for i, r in enumerate(aux_only[:3], 1):
            preview = _focused_excerpt(r.get("text", ""), query, 700)
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


# ── Intent detection (deterministic, no LLM) ────────────────────────────────
_FORCE_SEARCH_RE = re.compile(
    r"\b("
    r"search (the web )?for|search up|look (it |that |this )?up|google (it|for|this)|"
    r"find (me )?(the )?(latest|current|recent|most recent|newest)|"
    r"latest|current|currently|today|today'?s|tonight|tonight'?s|right now|live|breaking|"
    r"recent|recently|this (week|month|hour|morning|afternoon|evening)|"
    r"what'?s (happening|new|going on|the latest)|"
    r"in the news|news about|news on|news of|"
    r"as of (now|today)|up.to.date|real.?time|"
    r"happening (now|today|right now)|"
    r"price(s)? (of|for|right now|today)|stock (price|quote)"
    r")\b",
    re.IGNORECASE,
)

_FORCE_X_RE = re.compile(
    r"\b("
    r"tweet(s|ed|ing)?|twitter|"
    r"x post|x posts|x update|x updates|x thread|x threads|"
    r"on x\b|posted on x|said on x|"
    r"@[a-zA-Z0-9_]{2,15}\b|"
    r"x\.com/"
    r")",
    re.IGNORECASE,
)


def detect_intent(query: str) -> dict:
    """Fast, deterministic signal extraction. No LLM. Returns flags + matched signals."""
    if not query or not query.strip():
        return {"force_search": False, "force_x": False, "signals": []}
    signals: list[str] = []
    force_search = False
    force_x = False
    for m in _FORCE_SEARCH_RE.finditer(query):
        signals.append(m.group(0).lower())
        force_search = True
    for m in _FORCE_X_RE.finditer(query):
        signals.append(m.group(0).lower())
        force_x = True
    if force_x:
        force_search = True  # X queries always need a fresh fetch
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return {"force_search": force_search, "force_x": force_x, "signals": unique[:6]}


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

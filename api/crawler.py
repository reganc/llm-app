"""Recursive site crawling.

Two strategies, picked per-request:

1. **Crawl4AI** (default) — embedded Chromium via Playwright. Renders JS,
   handles SPAs, returns clean Markdown. Runs in-process; no external service.
2. **Local httpx + html.parser BFS** — same-origin breadth-first, no JS.
   Tiny, fast, and used when the user passes ``force_local=True`` or when
   Crawl4AI isn't importable (e.g. dev container without browsers installed).

Crawl jobs are tracked in-process: started → polled → terminal state.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from config import CFG
from extract import _HEADERS, _trafilatura, truncate

log = logging.getLogger("llm-api.crawler")

# In-memory hot cache for *active* jobs. Persistence happens via memory.py
# (Postgres) so jobs survive uvicorn --reload and container restarts; the
# SPA polling loop hits the DB-backed view via get_job/list_jobs whenever
# the in-process registry is cold (e.g. immediately after a reload).
_JOBS: dict[str, "CrawlJob"] = {}
_JOB_LOCK = asyncio.Lock()

# Hard caps to prevent runaway crawls.
_MAX_PAGES_CAP = 500
_MAX_DEPTH_CAP = 6

# Detect Crawl4AI lazily so missing-browser dev environments still boot the
# API; the actual crawl call surfaces a clear error if it fails.
try:  # pragma: no cover — import-time check
    import crawl4ai  # noqa: F401
    _CRAWL4AI_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    log.warning("crawl4ai not importable: %s", _exc)
    _CRAWL4AI_AVAILABLE = False


@dataclass
class CrawlPage:
    url: str
    title: str
    text: str
    char_count: int
    source_type: str  # "crawl4ai" | "web"
    error: Optional[str] = None


@dataclass
class CrawlJob:
    job_id: str
    seed_url: str
    max_pages: int
    max_depth: int
    strategy: str  # "crawl4ai" | "local"
    status: str = "running"  # running | completed | failed
    pages: list[CrawlPage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    chunks_stored: int = 0
    pages_ingested: int = 0
    resumed_from: int = 0  # URLs already in memory for this seed at job start
    pages_skipped: int = 0  # URLs encountered this run that matched the skip set

    def to_status(self) -> dict:
        out = {
            "job_id": self.job_id,
            "seed_url": self.seed_url,
            "status": self.status,
            "strategy": self.strategy,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "pages_crawled": len(self.pages),
            "pages_ingested": self.pages_ingested,
            "pages_skipped": self.pages_skipped,
            "resumed_from": self.resumed_from,
            "chunks_stored": self.chunks_stored,
            "errors": self.errors[-10:],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(
                (self.finished_at or time.time()) - self.started_at, 2
            ),
            "pages": [
                {
                    "url": p.url,
                    "title": p.title,
                    "char_count": p.char_count,
                    "error": p.error,
                }
                for p in self.pages
            ],
        }
        # ISO timestamps for the persistence layer. Kept separate from the
        # epoch fields the SPA already consumes.
        out["started_at_iso"] = (
            datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
            if self.started_at else None
        )
        out["finished_at_iso"] = (
            datetime.fromtimestamp(self.finished_at, tz=timezone.utc).isoformat()
            if self.finished_at else None
        )
        return out


async def _persist(job: "CrawlJob") -> None:
    """Best-effort write-through to Postgres. Logged + swallowed on
    failure — a DB hiccup must not abort an in-flight crawl."""
    try:
        import memory as mem  # lazy: avoid circular import on module load
        await mem.save_crawl_job(job.to_status())
    except Exception as e:
        log.debug("crawl_jobs persist failed: %s", e)


async def get_job(job_id: str) -> Optional[dict]:
    """Return the status dict for a job — hot cache first, DB fallback.

    Returns None if the job doesn't exist anywhere; callers turn that into
    a 404. The DB fallback is what makes the SPA polling loop survive a
    reload mid-crawl: in-memory registry is empty in the new process,
    but the row is still there.
    """
    job = _JOBS.get(job_id)
    if job is not None:
        return job.to_status()
    try:
        import memory as mem
        return await mem.load_crawl_job(job_id)
    except Exception:
        return None


async def list_jobs(limit: int = 20) -> list[dict]:
    try:
        import memory as mem
        rows = await mem.list_crawl_jobs(limit=limit)
    except Exception:
        rows = []
    if rows:
        return rows
    # Fallback if memory disabled — return whatever's in-process.
    items = sorted(_JOBS.values(), key=lambda j: j.started_at, reverse=True)
    return [j.to_status() for j in items[:limit]]


# ── Crawl4AI strategy (in-process Chromium) ─────────────────────────────────

def _build_skip_filter_chain(skip_urls: set[str]):
    """FilterChain that rejects URLs already crawled in a prior run.

    Returns None when filter classes aren't importable; the caller falls
    back to post-filtering (which still works but wastes browser renders).
    """
    if not skip_urls:
        return None
    try:
        from crawl4ai.deep_crawling.filters import FilterChain, URLFilter
    except Exception as e:  # pragma: no cover — version drift
        log.warning("crawl4ai FilterChain unavailable (%s); using post-filter", e)
        return None

    class _SkipURLFilter(URLFilter):
        def __init__(self, exclude: set[str]) -> None:
            super().__init__(name="skip-previously-crawled")
            self._exclude = exclude

        def apply(self, url: str) -> bool:
            passed = _normalize(url) not in self._exclude
            self._update_stats(passed)
            return passed

    return FilterChain([_SkipURLFilter(skip_urls)])


async def _crawl4ai_run(seed_url: str, max_pages: int, max_depth: int,
                        errors: list[str], *,
                        skip_urls: set[str] | None = None) -> tuple[list[CrawlPage], int]:
    """Run an embedded Crawl4AI deep crawl. Returns (pages, skipped_count).

    Uses BFSDeepCrawlStrategy with a same-origin filter and stream=True so
    pages flow back as they finish. Falls back to a single-page run if the
    deep-crawl import shape isn't available in this Crawl4AI version.

    skip_urls (normalized) excludes pages already ingested from a prior
    crawl with the same seed; the filter chain prevents Chromium from
    rendering them at all when supported.
    """
    skip_urls = skip_urls or set()
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    try:
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
        deep_crawl_supported = True
    except Exception as e:  # noqa: BLE001
        log.warning("crawl4ai deep-crawl import failed (%s); single-page mode", e)
        BFSDeepCrawlStrategy = None  # type: ignore[assignment]
        deep_crawl_supported = False

    browser = BrowserConfig(
        headless=True,
        verbose=False,
    )

    run_kwargs: dict = {
        "word_count_threshold": 30,
        "exclude_external_links": True,
        "exclude_social_media_links": True,
        "remove_overlay_elements": True,
        "verbose": False,
        "stream": True,
    }

    # include_external=False already constrains to the seed's origin —
    # no DomainFilter needed (and DomainFilter's exact-match semantics
    # turned out to be over-restrictive in practice).
    if deep_crawl_supported and max_depth > 0:
        bfs_kwargs: dict = {
            "max_depth": max_depth,
            "max_pages": max_pages,
            "include_external": False,
        }
        filter_chain = _build_skip_filter_chain(skip_urls)
        if filter_chain is not None:
            bfs_kwargs["filter_chain"] = filter_chain
        run_kwargs["deep_crawl_strategy"] = BFSDeepCrawlStrategy(**bfs_kwargs)

    config = CrawlerRunConfig(**run_kwargs)

    pages: list[CrawlPage] = []
    skipped = 0
    try:
        async with AsyncWebCrawler(config=browser) as crawler:
            result_iter = await crawler.arun(url=seed_url, config=config)

            # arun returns either an async iterator (stream=True) or a single
            # CrawlResult-like object depending on whether deep crawl is set.
            if hasattr(result_iter, "__aiter__"):
                async for result in result_iter:
                    page = _crawl4ai_to_page(result, errors)
                    if not page:
                        continue
                    # Belt-and-suspenders: even with FilterChain the seed
                    # itself bypasses filters, so guard here too.
                    if skip_urls and _normalize(page.url) in skip_urls:
                        skipped += 1
                        continue
                    pages.append(page)
                    if len(pages) >= max_pages:
                        break
            else:
                page = _crawl4ai_to_page(result_iter, errors)
                if page and not (skip_urls and _normalize(page.url) in skip_urls):
                    pages.append(page)
                elif page:
                    skipped += 1
    except Exception as e:  # noqa: BLE001
        errors.append(f"crawl4ai exception: {type(e).__name__}: {e}")
        log.exception("crawl4ai run failed")

    return pages, skipped


def _crawl4ai_to_page(result, errors: list[str]) -> Optional[CrawlPage]:
    """Convert a Crawl4AI CrawlResult into our CrawlPage. Returns None if empty."""
    try:
        url = getattr(result, "url", "") or ""
        success = getattr(result, "success", True)
        if not success:
            err = getattr(result, "error_message", None) or "unknown"
            errors.append(f"{url or 'page'}: {err}")
            return None

        # Crawl4AI exposes either .markdown (string in some versions) or
        # .markdown.fit_markdown / .markdown.raw_markdown (object). Try both.
        md_obj = getattr(result, "markdown", None)
        text = ""
        if isinstance(md_obj, str):
            text = md_obj
        elif md_obj is not None:
            text = (
                getattr(md_obj, "fit_markdown", None)
                or getattr(md_obj, "raw_markdown", None)
                or ""
            )
        if not text:
            text = getattr(result, "cleaned_html", "") or ""
        if not text or len(text) < 100:
            return None

        meta = getattr(result, "metadata", None) or {}
        title = (
            (meta.get("title") if isinstance(meta, dict) else None)
            or getattr(result, "title", None)
            or url
        )[:200]

        return CrawlPage(
            url=url,
            title=title,
            text=text,
            char_count=len(text),
            source_type="crawl4ai",
        )
    except Exception as e:  # noqa: BLE001
        errors.append(f"crawl4ai result parse: {type(e).__name__}: {e}")
        return None


# ── Local BFS fallback (no JS) ──────────────────────────────────────────────

class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        for k, v in attrs:
            if k == "href" and v:
                self.hrefs.append(v)


def _same_origin(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _normalize(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/")


async def _fetch_html(client: httpx.AsyncClient, url: str) -> tuple[Optional[str], str]:
    """Returns (html_or_None, reason). Reason is human-readable on failure."""
    try:
        r = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=20.0)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype.lower():
            return None, f"non-html content-type: {ctype or 'unknown'}"
        return r.text, "ok"
    except Exception as e:
        return None, f"fetch error: {type(e).__name__}: {e}"


def _discover_links(html: str, base_url: str) -> list[str]:
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    out: list[str] = []
    for href in parser.hrefs:
        if href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        try:
            absolute = urljoin(base_url, href)
        except Exception:
            continue
        if not absolute.startswith(("http://", "https://")):
            continue
        out.append(_normalize(absolute))
    return out


async def _local_crawl(seed_url: str, max_pages: int, max_depth: int,
                       errors: list[str], *,
                       skip_urls: set[str] | None = None) -> list[CrawlPage]:
    seed = _normalize(seed_url)
    skip_urls = skip_urls or set()
    # Pre-populate `seen` with previously-crawled URLs so they are never
    # enqueued; the seed itself is always processed (its links are how we
    # discover anything we missed last run).
    seen: set[str] = {seed} | skip_urls
    queue: list[tuple[str, int]] = [(seed, 0)]
    pages: list[CrawlPage] = []
    attempts = 0

    async with httpx.AsyncClient() as client:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            attempts += 1
            is_seed = (url == seed)

            html, fetch_reason = await _fetch_html(client, url)

            extracted = await _trafilatura(url)
            if extracted and not extracted.get("error") and extracted.get("text"):
                text, _ = truncate(extracted["text"])
                pages.append(CrawlPage(
                    url=extracted.get("url") or url,
                    title=extracted.get("title") or url,
                    text=text,
                    char_count=extracted.get("char_count") or len(text),
                    source_type="web",
                ))
            elif is_seed:
                if extracted and extracted.get("error"):
                    errors.append(f"seed extraction: {extracted['error']}")
                elif fetch_reason != "ok":
                    errors.append(f"seed fetch: {fetch_reason}")
                elif html and len(html) < 500:
                    errors.append(
                        f"seed returned only {len(html)} bytes of HTML — "
                        "likely a JS-rendered SPA or anti-bot challenge",
                    )
                else:
                    errors.append(
                        "seed extracted no text (paywall, login wall, "
                        "JS-rendered, or content under the 100-char threshold)",
                    )

            if depth >= max_depth:
                continue
            if not html:
                if is_seed:
                    errors.append(f"seed link discovery skipped: {fetch_reason}")
                continue

            new_links = 0
            for link in _discover_links(html, url):
                if link in seen:
                    continue
                if not _same_origin(seed, link):
                    continue
                seen.add(link)
                queue.append((link, depth + 1))
                new_links += 1
            if is_seed and new_links == 0:
                errors.append(
                    "no same-origin links found on seed page — site may be "
                    "JS-rendered or use external/relative links only",
                )

    if attempts and not pages:
        errors.append(f"attempted {attempts} URL(s); none yielded extractable text")
    return pages


# ── Public entrypoints ──────────────────────────────────────────────────────

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


async def _load_skip_urls(seed_url: str) -> set[str]:
    """Normalized set of URLs already ingested for this seed.

    Returns empty set on any failure (memory disabled, query error, etc.) —
    a failed lookup means we crawl as if fresh, not that we abort.
    """
    try:
        import memory as mem  # lazy: avoid circular import on module load
        raw = await mem.crawled_identifiers_for_seed(seed_url)
    except Exception as e:
        log.warning("resume lookup failed for %s: %s", seed_url, e)
        return set()
    return {_normalize(u) for u in raw if u}


async def start_crawl(seed_url: str, max_pages: int, max_depth: int,
                      *, force_local: bool = False,
                      resume: bool = True) -> CrawlJob:
    """Create and start a crawl job. Returns immediately; runs in background.

    When ``resume=True`` (default), URLs previously ingested under this seed
    are loaded from memory and skipped, letting consecutive runs walk past
    the per-job page cap to pick up the long tail of a large site.
    """
    max_pages = _clamp(max_pages, 1, _MAX_PAGES_CAP)
    max_depth = _clamp(max_depth, 0, _MAX_DEPTH_CAP)
    job_id = uuid.uuid4().hex[:12]

    skip_urls: set[str] = (await _load_skip_urls(seed_url)) if resume else set()
    use_crawl4ai = _CRAWL4AI_AVAILABLE and not force_local
    job = CrawlJob(
        job_id=job_id, seed_url=seed_url,
        max_pages=max_pages, max_depth=max_depth,
        strategy="crawl4ai" if use_crawl4ai else "local",
        resumed_from=len(skip_urls),
    )
    async with _JOB_LOCK:
        _JOBS[job_id] = job

    await _persist(job)
    asyncio.create_task(_run_job(job, skip_urls))
    return job


async def _run_job(job: CrawlJob, skip_urls: set[str]) -> None:
    import memory as mem  # lazy: avoid circular import on module load

    try:
        pages: list[CrawlPage] = []

        if job.strategy == "crawl4ai":
            pages, skipped = await _crawl4ai_run(
                job.seed_url, job.max_pages, job.max_depth, job.errors,
                skip_urls=skip_urls,
            )
            job.pages_skipped += skipped
            if not pages:
                job.errors.append("crawl4ai returned no pages; falling back to local crawl")
                job.strategy = "local"
                await _persist(job)

        if job.strategy == "local":
            pages = await _local_crawl(
                job.seed_url, job.max_pages, job.max_depth, job.errors,
                skip_urls=skip_urls,
            )

        job.pages = pages
        await _persist(job)

        if CFG.memory_enabled:
            for page in pages:
                if page.error or not page.text:
                    continue
                try:
                    result = await mem.store_knowledge(
                        text=page.text,
                        title=page.title or page.url,
                        source_type="web",
                        identifier=page.url,
                        extra={"crawl_seed": job.seed_url, "crawl_job": job.job_id},
                    )
                    job.chunks_stored += int(result.get("chunks_stored", 0))
                    if result.get("chunks_stored", 0) > 0:
                        job.pages_ingested += 1
                except Exception as e:
                    job.errors.append(f"store_knowledge failed for {page.url}: {e}")
                # Periodic write-through so the SPA's progress bar isn't
                # stuck at 0/N for the whole ingest phase.
                if job.pages_ingested and job.pages_ingested % 10 == 0:
                    await _persist(job)

        job.status = "completed"
    except Exception as e:
        log.exception("crawl job %s failed", job.job_id)
        job.errors.append(f"job exception: {e}")
        job.status = "failed"
    finally:
        job.finished_at = time.time()
        await _persist(job)

"""Extract clean text from URLs, files, YouTube, PDFs."""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional

import httpx
from fastapi import UploadFile

from config import CFG

log = logging.getLogger("llm-api.extract")

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; llm-research-bot/1.0)"}


def truncate(text: str, limit: int | None = None) -> tuple[str, bool]:
    limit = limit or CFG.context_char_limit
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    last_period = cut.rfind(". ")
    if last_period > limit * 0.8:
        cut = cut[:last_period + 1]
    return cut, True


async def _firecrawl(url: str) -> Optional[dict]:
    if not CFG.firecrawl_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{CFG.firecrawl_url}/v1/scrape",
                headers={
                    "Authorization": f"Bearer {CFG.firecrawl_api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
        if r.status_code != 200:
            return None
        data = r.json().get("data", {})
        text = data.get("markdown", "")
        if not text or len(text) < 100:
            return None
        meta = data.get("metadata", {})
        title = (meta.get("title") or meta.get("ogTitle") or url)[:200]
        return {"url": url, "title": title, "text": text,
                "source_type": "firecrawl", "char_count": len(text)}
    except Exception as e:
        log.warning("firecrawl failed for %s: %s", url, e)
        return None


async def _trafilatura(url: str) -> Optional[dict]:
    try:
        import trafilatura
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                     headers=_HEADERS) as client:
            r = await client.get(url)
            r.raise_for_status()
            html = r.text
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=True, no_fallback=False)
        if not text or len(text) < 100:
            return None
        meta = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else url)[:200]
        return {"url": url, "title": title, "text": text,
                "source_type": "web", "char_count": len(text)}
    except httpx.HTTPStatusError as e:
        return {"url": url, "title": "", "text": "",
                "source_type": "web", "char_count": 0,
                "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        log.warning("trafilatura failed for %s: %s", url, e)
        return None


async def extract_url(url: str) -> dict:
    """Returns dict with text/title/url/source_type/truncated/char_count/error."""
    base = {"url": url, "title": "", "text": "", "source_type": "web",
            "truncated": False, "char_count": 0, "error": None}

    # YouTube
    yt = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if yt:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            video_id = yt.group(1)
            tx = YouTubeTranscriptApi.get_transcript(video_id)
            raw = " ".join(t["text"] for t in tx)
            base["source_type"] = "youtube_transcript"
            base["title"] = f"YouTube video {video_id}"
            base["text"], base["truncated"] = truncate(raw)
            base["char_count"] = len(raw)
            return base
        except Exception as e:
            base["error"] = f"YouTube transcript unavailable: {e}"
            return base

    # Firecrawl → trafilatura
    result = await _firecrawl(url) or await _trafilatura(url)
    if not result:
        base["error"] = "Could not extract content (paywall, JS-rendered, or blocked)"
        return base
    if result.get("error"):
        base["error"] = result["error"]
        return base

    base["title"] = result["title"]
    base["text"], base["truncated"] = truncate(result["text"])
    base["char_count"] = result["char_count"]
    base["source_type"] = result["source_type"]
    return base


async def extract_file(file: UploadFile) -> dict:
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    base = {"filename": filename, "title": filename, "text": "",
            "source_type": ext.lstrip("."), "truncated": False,
            "char_count": 0, "error": None}
    raw = await file.read()
    try:
        if ext in (".txt", ".md", ".csv", ".log", ".json", ".xml", ".html"):
            text = raw.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
            if not text:
                base["error"] = "PDF appears scanned/image-only — no extractable text"
                return base
        elif ext == ".docx":
            import docx
            doc = docx.Document(io.BytesIO(raw))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".rtf":
            text = raw.decode("utf-8", errors="replace")
            text = re.sub(r"\\[a-z]+\d* ?", " ", text)
            text = re.sub(r"[{}\\]", "", text).strip()
        else:
            base["error"] = f"Unsupported file type: {ext}. Try txt/md/csv/pdf/docx/rtf"
            return base
        base["text"], base["truncated"] = truncate(text)
        base["char_count"] = len(text)
    except Exception as e:
        base["error"] = f"Failed to extract file: {e}"
    return base

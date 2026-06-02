"""Ingest URLs and documents into RAG memory + answer questions."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import conversations as conv_store
import crawler
import memory as mem
import ollama as oll
from auth import verify_api_key
from config import CFG, get_active_model
from extract import extract_file, extract_url
from prompts import get_system_prompt

log = logging.getLogger("llm-api.ingest")

router = APIRouter(prefix="/v1/ingest", tags=["ingest"], dependencies=[Depends(verify_api_key)])


def _build_messages(extracted: dict, question: str, system_key: str | None) -> list[dict]:
    source_label = extracted.get("url") or extracted.get("filename") or "document"
    title = extracted.get("title") or source_label
    note = ("\n\n[NOTE: Source text was truncated to fit context window.]"
            if extracted.get("truncated") else "")
    content = (
        f"SOURCE: {source_label}\n"
        f"TITLE: {title}\n"
        f"{'─' * 60}\n"
        f"{extracted['text']}{note}\n"
        f"{'─' * 60}\n\n"
        f"QUESTION: {question}"
    )
    return [
        {"role": "system", "content": get_system_prompt(system_key)},
        {"role": "user", "content": content},
    ]


class UrlIngestRequest(BaseModel):
    url: str
    question: str = "Summarise the key points of this article."
    model: str = Field(default_factory=get_active_model)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = 3000
    system_prompt_key: str | None = None


class UrlConvRequest(BaseModel):
    url: str
    conversation_name: str | None = None
    system_prompt_key: str = "default"
    model: str = Field(default_factory=get_active_model)


@router.post("/url")
async def ingest_url_route(req: UrlIngestRequest):
    extracted = await extract_url(req.url)
    if extracted["error"]:
        raise HTTPException(422, extracted["error"])

    mem_task = None
    if CFG.memory_enabled and extracted.get("text"):
        mem_task = asyncio.create_task(mem.store_knowledge(
            text=extracted["text"], title=extracted["title"] or req.url,
            source_type=extracted["source_type"], identifier=req.url,
        ))

    messages = _build_messages(extracted, req.question, req.system_prompt_key)
    payload = oll.build_payload(messages, req.model,
                                temperature=req.temperature,
                                max_tokens=min(req.max_tokens, CFG.max_tokens))
    data = await oll.chat(payload)
    reply = data.get("message", {}).get("content", "")

    mem_result = None
    if mem_task:
        try:
            mem_result = await asyncio.wait_for(mem_task, timeout=30.0)
        except Exception:
            mem_result = {"error": "memory store timeout"}

    return {
        "answer": reply,
        "source": {
            "url": extracted.get("url"),
            "title": extracted["title"],
            "source_type": extracted["source_type"],
            "char_count": extracted["char_count"],
            "truncated": extracted["truncated"],
        },
        "memory": mem_result,
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }


@router.post("/url/conversation")
async def ingest_url_to_conversation(req: UrlConvRequest):
    extracted = await extract_url(req.url)
    if extracted["error"]:
        raise HTTPException(422, extracted["error"])
    title = extracted.get("title") or req.url
    name = req.conversation_name or f"Article: {title[:50]}"
    note = ("\n\n[NOTE: Content was truncated to fit context window.]"
            if extracted.get("truncated") else "")
    body = (
        f"The following article has been loaded for analysis.\n\n"
        f"SOURCE: {extracted.get('url', req.url)}\n"
        f"TITLE: {title}\n{'─' * 60}\n"
        f"{extracted['text']}{note}\n{'─' * 60}\n\n"
        f"I'm ready to answer questions about this content."
    )
    conv = conv_store.create(
        name=name,
        system_prompt_key=req.system_prompt_key,
        extra={"source": {
            "url": extracted.get("url"),
            "title": title,
            "source_type": extracted["source_type"],
            "char_count": extracted["char_count"],
            "truncated": extracted["truncated"],
        }},
    )
    conv_store.append_message(conv["id"], "user",
                              f"Please load this article:\n\nSOURCE: {extracted.get('url', req.url)}\n"
                              f"TITLE: {title}\n\n{extracted['text']}{note}")
    conv_store.append_message(conv["id"], "assistant", body)
    if CFG.memory_enabled:
        asyncio.create_task(mem.store_knowledge(
            text=extracted["text"], title=title,
            source_type=extracted["source_type"], identifier=req.url,
        ))
    return {"conversation_id": conv["id"], "name": name,
            "source": conv.get("source")}


class SiteIngestRequest(BaseModel):
    url: str
    max_pages: int = Field(default=25, ge=1, le=500)
    max_depth: int = Field(default=2, ge=0, le=6)
    force_local: bool = False
    # When true, URLs already ingested under this seed are skipped so
    # consecutive runs walk past the per-job page cap on large sites.
    resume: bool = True


@router.post("/site")
async def ingest_site_route(req: SiteIngestRequest):
    if not CFG.memory_enabled:
        raise HTTPException(409, "MEMORY_ENABLED=false; cannot ingest a site")
    job = await crawler.start_crawl(
        seed_url=req.url,
        max_pages=req.max_pages,
        max_depth=req.max_depth,
        force_local=req.force_local,
        resume=req.resume,
    )
    return job.to_status()


@router.get("/site/{job_id}")
async def ingest_site_status(job_id: str):
    status = await crawler.get_job(job_id)
    if not status:
        raise HTTPException(404, f"unknown crawl job: {job_id}")
    return status


@router.get("/site")
async def ingest_site_list(limit: int = 20):
    return {"jobs": await crawler.list_jobs(limit=limit)}


@router.post("/document")
async def ingest_document_route(
    file: UploadFile = File(...),
    question: str = Form("Summarise the key points of this document."),
    model: str | None = Form(None),
    temperature: float = Form(0.3),
    max_tokens: int = Form(3000),
    system_prompt_key: str = Form("default"),
):
    model = model or get_active_model()
    extracted = await extract_file(file)
    if extracted["error"]:
        raise HTTPException(422, extracted["error"])
    if not extracted["text"].strip():
        raise HTTPException(422, "No text could be extracted")

    messages = _build_messages(extracted, question, system_prompt_key)
    payload = oll.build_payload(messages, model,
                                temperature=temperature,
                                max_tokens=min(max_tokens, CFG.max_tokens))

    mem_task = None
    if CFG.memory_enabled:
        mem_task = asyncio.create_task(mem.store_knowledge(
            text=extracted["text"], title=extracted["filename"],
            source_type=extracted["source_type"], identifier=extracted["filename"],
        ))

    data = await oll.chat(payload)
    reply = data.get("message", {}).get("content", "")
    mem_result = None
    if mem_task:
        try:
            mem_result = await asyncio.wait_for(mem_task, timeout=30.0)
        except Exception:
            mem_result = {"error": "memory store timeout"}

    return {
        "answer": reply,
        "source": {
            "filename": extracted["filename"],
            "source_type": extracted["source_type"],
            "char_count": extracted["char_count"],
            "truncated": extracted["truncated"],
        },
        "memory": mem_result,
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        },
    }


@router.post("/document/conversation")
async def ingest_document_to_conversation(
    file: UploadFile = File(...),
    conversation_name: str = Form(""),
    system_prompt_key: str = Form("default"),
    model: str | None = Form(None),
):
    extracted = await extract_file(file)
    if extracted["error"]:
        raise HTTPException(422, extracted["error"])
    if not extracted["text"].strip():
        raise HTTPException(422, "No text could be extracted")

    filename = extracted["filename"]
    name = conversation_name or f"Doc: {filename}"
    note = ("\n\n[NOTE: Document was truncated to fit context window.]"
            if extracted.get("truncated") else "")
    doc_block = (
        f"FILE: {filename}\n{'─' * 60}\n{extracted['text']}{note}\n{'─' * 60}"
    )
    conv = conv_store.create(
        name=name, system_prompt_key=system_prompt_key,
        extra={"source": {
            "filename": filename,
            "source_type": extracted["source_type"],
            "char_count": extracted["char_count"],
            "truncated": extracted["truncated"],
        }},
    )
    conv_store.append_message(conv["id"], "user", f"Please load this document:\n\n{doc_block}")
    conv_store.append_message(
        conv["id"], "assistant",
        f"Document loaded: {filename} ({extracted['char_count']:,} chars"
        f"{', truncated' if extracted['truncated'] else ''}). Ready for questions.",
    )
    if CFG.memory_enabled:
        asyncio.create_task(mem.store_knowledge(
            text=extracted["text"], title=filename,
            source_type=extracted["source_type"], identifier=filename,
        ))
    return {"conversation_id": conv["id"], "name": name, "source": conv.get("source")}

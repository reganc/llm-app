"""Search, memory, settings, system-prompt, model-listing endpoints."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import memory as mem
import ollama as oll
import search
import x_search
from auth import verify_api_key
from config import (CFG, get_active_model, get_default_system_prompt_key,
                    set_setting)
from extract import extract_file
from prompts import PROMPT_DESCRIPTIONS, SYSTEM_PROMPTS, get_system_prompt

log = logging.getLogger("llm-api.admin")

router = APIRouter(prefix="/v1", tags=["admin"], dependencies=[Depends(verify_api_key)])


# ── Models ───────────────────────────────────────────────────────────────────
@router.get("/models")
async def list_models():
    items = await oll.list_models()
    return {
        "object": "list",
        "data": [{"id": m["name"], "object": "model",
                  "created": int(time.time()), "owned_by": "llm-api",
                  "size": m.get("size"), "modified_at": m.get("modified_at")}
                 for m in items],
    }


# ── Settings ─────────────────────────────────────────────────────────────────
class SettingsPatch(BaseModel):
    default_model: str | None = None
    default_system_prompt_key: str | None = None
    search_always: bool | None = None


@router.get("/settings")
async def get_settings():
    return {
        "default_model": get_active_model(),
        "env_default_model": CFG.default_model,
        "default_system_prompt_key": get_default_system_prompt_key(),
        "search_enabled": CFG.search_enabled,
        "memory_enabled": CFG.memory_enabled,
        "x_search": x_search.status(),
    }


@router.patch("/settings")
async def patch_settings(body: SettingsPatch):
    if body.default_model is not None:
        set_setting("default_model", body.default_model)
    if body.default_system_prompt_key is not None:
        if body.default_system_prompt_key not in SYSTEM_PROMPTS:
            raise HTTPException(400, "Unknown system_prompt_key")
        set_setting("default_system_prompt_key", body.default_system_prompt_key)
    if body.search_always is not None:
        set_setting("search_always", body.search_always)
    return await get_settings()


# ── System prompts ───────────────────────────────────────────────────────────
@router.get("/system-prompts")
async def list_system_prompts():
    return {
        "default_key": get_default_system_prompt_key(),
        "prompts": {
            k: {"description": PROMPT_DESCRIPTIONS.get(k, ""),
                "preview": v[:140] + ("…" if len(v) > 140 else "")}
            for k, v in SYSTEM_PROMPTS.items()
        },
    }


# ── Search ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=10)
    store_memory: bool = True
    answer: bool = True
    model: str = Field(default_factory=get_active_model)
    system_prompt_key: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048)


@router.post("/search")
async def explicit_search(req: SearchRequest):
    result = await search.search_and_ingest(
        req.query, num_results=req.num_results,
        store_memory=req.store_memory and CFG.memory_enabled,
    )
    if result.get("error"):
        raise HTTPException(422, result["error"])
    out = {
        "query": req.query, "results": result["results"],
        "stored": result.get("stored", 0),
        "source": result.get("source", "unknown"),
        "answer": None,
    }
    if req.answer and result.get("context_text"):
        messages = [
            {"role": "system", "content": get_system_prompt(req.system_prompt_key)},
            {"role": "user", "content":
                f"{result['context_text']}\n\nBased on these results, answer: {req.query}"},
        ]
        payload = oll.build_payload(messages, req.model,
                                    temperature=req.temperature, max_tokens=req.max_tokens)
        try:
            data = await oll.chat(payload)
            out["answer"] = data.get("message", {}).get("content", "")
        except Exception as e:
            out["answer_error"] = str(e)
    return out


# ── Auxiliary sites ──────────────────────────────────────────────────────────
class AuxSiteRequest(BaseModel):
    url: str
    label: str = ""


@router.get("/search/auxiliary-sites")
async def aux_list():
    sites = search.list_aux_sites()
    return {"sites": sites, "count": len(sites)}


@router.post("/search/auxiliary-sites")
async def aux_add(req: AuxSiteRequest):
    try:
        entry = search.add_aux_site(req.url, req.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"added": entry, "sites": search.list_aux_sites()}


@router.delete("/search/auxiliary-sites")
async def aux_remove(url: str):
    if not search.remove_aux_site(url):
        raise HTTPException(404, "Site not found")
    return {"removed": url, "sites": search.list_aux_sites()}


# ── X / Twitter ──────────────────────────────────────────────────────────────
class XSearchRequest(BaseModel):
    query: str
    count: int = Field(default=10, ge=1, le=50)
    store_memory: bool = True
    answer: bool = True
    model: str = Field(default_factory=get_active_model)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048)


@router.get("/search/x/status")
async def x_status():
    return x_search.status()


@router.post("/search/x")
async def x_search_route(req: XSearchRequest):
    result = await x_search.search_x_and_ingest(req.query,
                                                store_memory=req.store_memory and CFG.memory_enabled)
    if result.get("error"):
        raise HTTPException(422, result["error"])
    out = {"query": req.query, "results": result["results"],
           "stored": result.get("stored", 0), "source": "x_twitter", "answer": None}
    if req.answer and result.get("context_text"):
        messages = [
            {"role": "system", "content": get_system_prompt(None)},
            {"role": "user", "content":
                f"{result['context_text']}\n\nBased on these X posts, answer: {req.query}"},
        ]
        payload = oll.build_payload(messages, req.model,
                                    temperature=req.temperature, max_tokens=req.max_tokens)
        try:
            data = await oll.chat(payload)
            out["answer"] = data.get("message", {}).get("content", "")
        except Exception as e:
            out["answer_error"] = str(e)
    return out


# ── Memory admin ─────────────────────────────────────────────────────────────
@router.get("/memory/stats")
async def memory_stats():
    return mem.get_stats()


@router.get("/memory/search")
async def memory_search(q: str, top_k: int = 5):
    if not CFG.memory_enabled:
        raise HTTPException(503, "Memory not enabled")
    chunks = await mem.retrieve(q, top_k=top_k)
    return {"query": q, "results": chunks, "count": len(chunks)}


@router.get("/memory/sources")
async def memory_sources(q: str = "", source_type: str = "",
                         limit: int = 200, offset: int = 0):
    if not CFG.memory_enabled:
        raise HTTPException(503, "Memory not enabled")
    return mem.list_sources(query=q, source_type=source_type,
                            limit=limit, offset=offset)


@router.get("/memory/source/{source_id}")
async def memory_get_source(source_id: str):
    if not CFG.memory_enabled:
        raise HTTPException(503, "Memory not enabled")
    res = mem.get_source(source_id)
    if not res.get("available"):
        raise HTTPException(503, res.get("error", "Memory unavailable"))
    if not res.get("found"):
        raise HTTPException(404, "Source not found")
    return res


@router.delete("/memory/source/{source_id}")
async def memory_delete(source_id: str):
    if not CFG.memory_enabled:
        raise HTTPException(503, "Memory not enabled")
    return mem.delete_source(source_id)


@router.post("/memory/ingest")
async def memory_ingest(
    title: str = Form(...),
    source_type: str = Form("manual"),
    identifier: str = Form(""),
    file: UploadFile | None = File(None),
    text: str = Form(""),
):
    if not CFG.memory_enabled:
        raise HTTPException(503, "Memory not enabled")
    if file and file.filename:
        extracted = await extract_file(file)
        if extracted["error"]:
            raise HTTPException(422, extracted["error"])
        content = extracted["text"]
        identifier = identifier or extracted["filename"]
    elif text.strip():
        content = text
    else:
        raise HTTPException(400, "Provide either a file or text")
    return await mem.store_knowledge(
        text=content, title=title, source_type=source_type,
        identifier=identifier or title,
    )

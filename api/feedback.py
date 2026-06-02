"""Per-message feedback (thumbs up/down) used to re-rank RAG retrieval."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import conversations as conv_store
import memory as mem
from auth import verify_api_key

log = logging.getLogger("llm-api.feedback")

router = APIRouter(prefix="/v1", tags=["feedback"], dependencies=[Depends(verify_api_key)])


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=4, max_length=64)
    rating: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=2000)
    conversation_id: str | None = None


class FeedbackClearRequest(BaseModel):
    message_id: str = Field(..., min_length=4, max_length=64)
    conversation_id: str | None = None


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    chunks_updated = await mem.update_feedback(req.message_id, req.rating)
    conv_updated = False
    if req.conversation_id:
        conv_updated = conv_store.set_feedback(
            req.conversation_id, req.message_id, req.rating, req.note,
        )
    if chunks_updated == 0 and not conv_updated:
        raise HTTPException(404, "message_id not found in memory or conversation")
    return {
        "ok": True,
        "message_id": req.message_id,
        "rating": req.rating,
        "chunks_updated": chunks_updated,
        "conversation_updated": conv_updated,
    }


@router.delete("/feedback")
async def clear_feedback(req: FeedbackClearRequest):
    chunks_updated = await mem.update_feedback(req.message_id, None)
    conv_updated = False
    if req.conversation_id:
        conv_updated = conv_store.set_feedback(
            req.conversation_id, req.message_id, None, None,
        )
    return {
        "ok": True,
        "message_id": req.message_id,
        "chunks_updated": chunks_updated,
        "conversation_updated": conv_updated,
    }


@router.get("/feedback/stats")
async def feedback_stats():
    return await mem.feedback_stats()

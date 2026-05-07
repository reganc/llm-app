"""Bearer token auth dependency."""
from __future__ import annotations

from fastapi import HTTPException, Request

from config import CFG


async def verify_api_key(request: Request) -> None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != CFG.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

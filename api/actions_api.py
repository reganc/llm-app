"""/v1/actions: the web panel's view of proposed actions.

List and reject only. There is deliberately no approve endpoint: MCP clients
hold the same API key, so an approval here would let a client approve its own
proposal. Approving is a tap on the Telegram card (action_cards.py).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import action_cards
import actions
from auth import verify_api_key

router = APIRouter(prefix="/v1/actions", tags=["actions"], dependencies=[Depends(verify_api_key)])


@router.get("")
async def list_actions(limit: int = 30):
    for record in actions.sweep():          # expire first, and say so on the cards
        await action_cards.refresh(record)
    rows = actions.recent(max(1, min(limit, 100)))
    return {"waiting": sum(1 for r in rows if r["status"] in actions.OPEN),
            "actions": [actions.view(r) for r in rows]}


@router.post("/{action_id}/reject")
async def reject(action_id: str):
    if actions.get(action_id) is None:
        raise HTTPException(404, "No proposal has that id.")
    try:
        record = actions.reject(action_id, by="web")
    except actions.Refused as e:
        raise HTTPException(409, str(e)) from None
    await action_cards.refresh(record)       # the phone's card stops offering Approve
    return actions.view(record)

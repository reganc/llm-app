"""/v1/routines — saved tasks that run on a schedule, and the inbox of results."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

import routine_draft
import routine_query
import routine_runner as runner
import routine_store as store
import skills
from auth import verify_api_key
from config import CFG
from routine_schedule import Schedule, valid_timezone

router = APIRouter(prefix="/v1/routines", tags=["routines"],
                   dependencies=[Depends(verify_api_key)])

MAX_ROUTINES = 25

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
DraftText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Notify = Literal["off", "headline", "full"]
SkillName = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, max_length=40)]
SearchQuery = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]


class Sources(BaseModel):
    web: bool = True
    library: bool = False
    news: bool = False  # search recent news stories instead of the general web
    extra_sites: bool = False  # with news: also read the always-included sites

    model_config = {"extra": "forbid"}


def _check_timezone(value: str | None) -> str | None:
    if value is not None and not valid_timezone(value):
        raise ValueError("unknown timezone")
    return value


class RoutineCreate(BaseModel):
    name: Name
    prompt: Prompt
    query: SearchQuery | None = None  # what to search for; worked out from the prompt if empty
    schedule: Schedule
    timezone: str = Field(default_factory=lambda: CFG.routine_timezone)
    sources: Sources = Field(default_factory=Sources)
    only_new: bool = False  # report only what earlier runs did not
    notify: Notify = "off"  # Telegram, for scheduled runs
    skill: SkillName | None = None  # saved instructions to follow
    enabled: bool = True

    model_config = {"extra": "forbid"}
    _tz = field_validator("timezone")(_check_timezone)


class RoutineUpdate(BaseModel):
    name: Name | None = None
    prompt: Prompt | None = None
    query: SearchQuery | None = None
    schedule: Schedule | None = None
    timezone: str | None = None
    sources: Sources | None = None
    only_new: bool | None = None
    notify: Notify | None = None
    skill: SkillName | None = None
    enabled: bool | None = None

    model_config = {"extra": "forbid"}
    _tz = field_validator("timezone")(_check_timezone)

    @model_validator(mode="after")
    def _no_nulls(self) -> "RoutineUpdate":
        for name in self.model_fields_set - {"query", "skill"}:  # only these can be cleared
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self


def _summary(routine: dict) -> dict:
    """A routine without its run bodies, plus its latest run's headline."""
    latest = routine["runs"][0] if routine["runs"] else None
    keys = ("id", "status", "trigger", "started_at", "finished_at")
    memory = routine.get("memory") or {}
    return {"only_new": False, "notify": "off", "skill": None,
            **{k: v for k, v in routine.items() if k not in ("runs", "memory")},
            "search_phrase": routine_query.effective(routine),
            "remembers": len(memory.get("urls") or []),
            "last_run": {k: latest[k] for k in keys} if latest else None}


def _check_skill(fields: dict) -> None:
    if "skill" in fields:
        fields["skill"] = fields["skill"] or None
        if fields["skill"] and skills.get(fields["skill"]) is None:
            raise HTTPException(422, "skill not found")


def _require(routine_id: str) -> dict:
    routine = store.get(routine_id)
    if routine is None:
        raise HTTPException(404, "routine not found")
    return routine


@router.get("")
async def list_routines():
    return {"routines": [_summary(r) for r in store.list_all()],
            "unread": store.unread_count(), "scheduler_enabled": CFG.routines_enabled,
            "default_timezone": CFG.routine_timezone}


@router.post("", status_code=201)
async def create_routine(req: RoutineCreate):
    if store.count() >= MAX_ROUTINES:
        raise HTTPException(409, f"limit of {MAX_ROUTINES} routines reached")
    fields = req.model_dump()
    fields["query"] = fields["query"] or None
    _check_skill(fields)
    return _summary(store.create({**fields, "next_run_at": runner.upcoming(fields)}))


class DraftRequest(BaseModel):
    text: DraftText

    model_config = {"extra": "forbid"}


@router.post("/draft")
async def draft_routine(req: DraftRequest):
    """A routine filled in from plain language, for the user to check and save."""
    return await routine_draft.draft(req.text)


@router.get("/inbox")
async def inbox(limit: int = Query(default=30, ge=1, le=100)):
    return {"runs": store.inbox(limit), "unread": store.unread_count()}


@router.post("/inbox/read")
async def mark_all_read():
    return {"marked": store.mark_all_read()}


@router.get("/{routine_id}")
async def get_routine(routine_id: str):
    routine = _require(routine_id)
    return {**_summary(routine), "runs": routine["runs"]}


@router.patch("/{routine_id}")
async def update_routine(routine_id: str, req: RoutineUpdate):
    current = _require(routine_id)
    fields = req.model_dump(exclude_unset=True)
    if "query" in fields:
        fields["query"] = fields["query"] or None
    _check_skill(fields)
    if fields.keys() & {"schedule", "timezone", "enabled"}:
        fields["next_run_at"] = runner.upcoming({**current, **fields})
    # Filled in first: a routine saved before a source option existed is not a new task.
    before = {**current, "sources": Sources(**current["sources"]).model_dump()}
    if any(f in fields and fields[f] != before.get(f) for f in runner.TASK_FIELDS):
        fields["memory"] = None
    updated = store.update(routine_id, fields)
    if updated is None:
        raise HTTPException(404, "routine not found")
    return _summary(updated)


@router.delete("/{routine_id}")
async def delete_routine(routine_id: str):
    if not store.delete(routine_id):
        raise HTTPException(404, "routine not found")
    return {"deleted": routine_id}


@router.post("/{routine_id}/run", status_code=202)
async def run_now(routine_id: str):
    _require(routine_id)
    if not runner.start_manual(routine_id):
        raise HTTPException(409, "this routine is already running")
    return {"started": routine_id}


@router.post("/{routine_id}/forget")
async def forget(routine_id: str):
    """Start comparing from scratch."""
    _require(routine_id)
    store.set_memory(routine_id, None)
    return {"forgotten": routine_id}


@router.post("/{routine_id}/runs/{run_id}/read")
async def mark_read(routine_id: str, run_id: str):
    if not store.mark_read(routine_id, run_id):
        raise HTTPException(404, "run not found")
    return {"read": run_id}

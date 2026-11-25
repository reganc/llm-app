"""/v1/watches — pages checked for changes."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StringConstraints, model_validator

import safe_fetch
import watch_check
import watch_runner as runner
import watch_store as store
from auth import verify_api_key
from config import CFG

router = APIRouter(prefix="/v1/watches", tags=["watches"], dependencies=[Depends(verify_api_key)])

MAX_WATCHES = 25

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Url = Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=1000)]
Words = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Filter = Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)]
EveryMinutes = Literal[15, 30, 60, 180, 360, 720, 1440]


class _Strict(BaseModel):
    model_config = {"extra": "forbid"}


class NumberRule(_Strict):
    kind: Literal["number"]
    label: Words
    when: Literal["changes", "below", "above"] = "changes"
    limit: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _limit_needed(self) -> "NumberRule":
        if (self.when == "changes") != (self.limit is None):
            raise ValueError("a limit goes with below or above, and only with them")
        return self


class PhraseRule(_Strict):
    kind: Literal["phrase"]
    phrase: Words
    when: Literal["appears", "disappears", "either"] = "appears"


class ChangeRule(_Strict):
    kind: Literal["change"]
    only: Filter = ""  # watch only lines containing this


Rule = Annotated[Union[NumberRule, PhraseRule, ChangeRule], Field(discriminator="kind")]


class Look(_Strict):
    url: Url
    rule: Rule


class WatchCreate(Look):
    name: Name
    approved_host: str = Field(max_length=255)  # the site the user agreed to have read
    every_minutes: EveryMinutes = 60
    notify: bool = True
    enabled: bool = True


class WatchUpdate(_Strict):
    name: Name | None = None
    url: Url | None = None
    approved_host: str | None = Field(default=None, max_length=255)
    rule: Rule | None = None
    every_minutes: EveryMinutes | None = None
    notify: bool | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def _no_nulls(self) -> "WatchUpdate":
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self


def _site(url: str) -> str:
    try:
        return safe_fetch.host_of(url)
    except safe_fetch.FetchError as e:
        raise HTTPException(422, str(e)) from None


def _approve(url: str, approved_host: str) -> str:
    host = _site(url)
    if approved_host.lower() != host:
        raise HTTPException(422, f"Approve reading {host} to watch this page.")
    return host


def _summary(watch: dict) -> dict:
    return {**{k: v for k, v in watch.items() if k != "state"},
            "unread": sum(not a["read"] for a in watch["alerts"])}


def _require(watch_id: str) -> dict:
    watch = store.get(watch_id)
    if watch is None:
        raise HTTPException(404, "watch not found")
    return watch


@router.get("")
async def list_watches():
    return {"watches": [_summary(w) for w in store.list_all()], "unread": store.unread_count(),
            "scheduler_enabled": CFG.watches_enabled}


@router.post("/preview")
async def preview(req: Look):
    """What a watch would see now. Nothing is saved."""
    host, rule = _site(req.url), req.rule.model_dump()
    try:
        page_lines = watch_check.lines(await safe_fetch.fetch(req.url, host))
        result = watch_check.compare(rule, page_lines, None)
    except safe_fetch.FetchError as e:
        raise HTTPException(422, str(e)) from None
    return {"site": host, "shown": result.shown, "sample": result.lines[:12]}


@router.post("", status_code=201)
async def create_watch(req: WatchCreate):
    if store.count() >= MAX_WATCHES:
        raise HTTPException(409, f"limit of {MAX_WATCHES} watches reached")
    fields = req.model_dump()
    fields["approved_host"] = _approve(req.url, req.approved_host)
    fields["next_check_at"] = store.now() if req.enabled else None
    return _summary(store.create(fields))


@router.get("/{watch_id}")
async def get_watch(watch_id: str):
    return _summary(_require(watch_id))


@router.patch("/{watch_id}")
async def update_watch(watch_id: str, req: WatchUpdate):
    current, fields = _require(watch_id), req.model_dump(exclude_unset=True)
    if "url" in fields or "approved_host" in fields:
        fields["approved_host"] = _approve(fields.get("url", current["url"]),
                                           fields.get("approved_host", current["approved_host"]))
    if any(k in fields and fields[k] != current[k] for k in ("url", "rule")):
        fields.update(state=None, shown="", error=None, failures=0)
    merged = {**current, **fields}
    if not merged["enabled"]:
        fields["next_check_at"] = None
    elif "state" in fields or not current["enabled"]:
        fields["next_check_at"] = store.now()
    elif "every_minutes" in fields:
        fields["next_check_at"] = runner.next_check(merged)
    return _summary(store.update(watch_id, fields) or _require(watch_id))


@router.delete("/{watch_id}")
async def delete_watch(watch_id: str):
    if not store.delete(watch_id):
        raise HTTPException(404, "watch not found")
    return {"deleted": watch_id}


@router.post("/{watch_id}/check")
async def check_now(watch_id: str):
    _require(watch_id)
    await runner.check(watch_id)
    return _summary(_require(watch_id))


@router.post("/{watch_id}/read")
async def mark_read(watch_id: str):
    if not store.mark_read(watch_id):
        raise HTTPException(404, "watch not found")
    return {"read": watch_id}

"""Saved skills: reusable instructions for chat (`/name …`), routines and Telegram.

A skill is text added to the system prompt. It gives the model no tools and no
access it did not already have, and the rules about sources still apply.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints, field_validator

import routine_store
from auth import verify_api_key
from config import DATA_DIR

log = logging.getLogger("llm-api.skills")

MAX_SKILLS = 50
# Names that already mean something after a slash, in chat or Telegram.
RESERVED = {"search", "x", "library", "lib", "routine", "routines", "watch", "watches", "run",
            "pause", "resume", "new", "more", "help", "start", "skills", "status", "stop"}
_COMMAND = re.compile(r"^/([a-z0-9][a-z0-9-]*)(?:\s+([\s\S]*))?$", re.IGNORECASE)
_ALONE = "Do this now, using the conversation so far."

_path: Path = DATA_DIR / "skills.json"
_skills: dict[str, dict] = {}


def load(path: Path | None = None) -> None:
    global _path, _skills
    _path = path or _path
    try:
        data = json.loads(_path.read_text())
        _skills = {s["name"]: s for s in data if isinstance(s["instructions"], str)}
    except FileNotFoundError:
        _skills = {}
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.error("skills file is unreadable and is left as it is: %s", type(e).__name__)
        _skills = {}


def _save(skills: dict[str, dict]) -> None:
    global _skills
    tmp = _path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(list(skills.values()), f, indent=1)
    os.replace(tmp, _path)
    _skills = skills


def get(name: str | None) -> dict | None:
    found = _skills.get((name or "").lower())
    return dict(found) if found else None


def list_all() -> list[dict]:
    return [dict(_skills[n]) for n in sorted(_skills)]


def split(text: str) -> tuple[str, dict | None]:
    """("the rest", skill) for "/name the rest" when that skill exists."""
    match = _COMMAND.match((text or "").strip())
    skill = get(match.group(1)) if match else None
    if not match or skill is None:
        return text, None
    return (match.group(2) or "").strip() or _ALONE, skill


def prompt(skill: dict | None) -> str:
    """What a skill adds to the system prompt."""
    if not skill:
        return ""
    return (f"\n\nFollow the user's saved skill “{skill['name']}” for this answer. It sets the "
            "steps and the format; the rules about tools, sources and citations still apply.\n"
            f"--- skill ---\n{skill['instructions']}\n--- end of skill ---")


def apply(messages: list[dict], skill: dict | None) -> list[dict]:
    if not skill:
        return messages
    return [{**m, "content": m["content"] + prompt(skill)} if m["role"] == "system" else m
            for m in messages]


# ── API ──────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/skills", tags=["skills"], dependencies=[Depends(verify_api_key)])

Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)]
Instructions = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class Skill(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True,
                                           pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{0,39}$")]
    description: Description = ""
    instructions: Instructions

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def _free(cls, value: str) -> str:
        if value.lower() in RESERVED:
            raise ValueError("that name is used by a built-in command")
        return value.lower()


def _used_by(name: str) -> list[str]:
    return [r["name"] for r in routine_store.list_all() if r.get("skill") == name]


@router.get("")
async def list_skills():
    return {"skills": [{**s, "used_by": _used_by(s["name"])} for s in list_all()]}


@router.post("", status_code=201)
async def create_skill(req: Skill):
    if req.name in _skills:
        raise HTTPException(409, "a skill with that name already exists")
    if len(_skills) >= MAX_SKILLS:
        raise HTTPException(409, f"limit of {MAX_SKILLS} skills reached")
    _save({**_skills, req.name: req.model_dump()})
    return req.model_dump()


@router.put("/{name}")
async def update_skill(name: str, req: Skill):
    if name not in _skills:
        raise HTTPException(404, "skill not found")
    if req.name != name:
        raise HTTPException(422, "a skill can't be renamed")
    _save({**_skills, name: req.model_dump()})
    return req.model_dump()


@router.delete("/{name}")
async def delete_skill(name: str):
    if name not in _skills:
        raise HTTPException(404, "skill not found")
    if used := _used_by(name):
        raise HTTPException(409, f"used by: {', '.join(used)}")
    _save({n: s for n, s in _skills.items() if n != name})
    return {"deleted": name}


load()

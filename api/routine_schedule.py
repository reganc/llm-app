"""When a routine runs: daily wall-clock times on chosen weekdays, or every N hours."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, model_validator

from scheduler import parse_hhmm

MAX_INTERVAL_HOURS = 168  # one week


class Schedule(BaseModel):
    kind: Literal["daily", "interval"]
    time: str | None = None                        # daily: HH:MM, 24h
    days: list[int] = [0, 1, 2, 3, 4, 5, 6]        # daily: 0 = Monday
    hours: int | None = None                       # interval

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> "Schedule":
        if self.kind == "daily":
            if parse_hhmm(self.time or "") is None:
                raise ValueError("time must be HH:MM (24-hour)")
            if not self.days or any(d not in range(7) for d in self.days):
                raise ValueError("days must be one or more of 0 (Monday) to 6 (Sunday)")
            self.days = sorted(set(self.days))
        elif self.hours is None or not 1 <= self.hours <= MAX_INTERVAL_HOURS:
            raise ValueError(f"hours must be between 1 and {MAX_INTERVAL_HOURS}")
        return self


def valid_timezone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (KeyError, ValueError, OSError):  # unknown name, or a path-like one
        return False
    return bool(name)


def next_run(schedule: Schedule, after: datetime, tz_name: str) -> datetime:
    """First run strictly after `after` (aware UTC). Daily times keep their
    wall clock across DST; a time skipped by the spring change runs once just
    after it, and a time repeated in autumn runs once."""
    if schedule.kind == "interval":
        return after + timedelta(hours=schedule.hours or 1)
    tz = ZoneInfo(tz_name)
    hour, minute = parse_hhmm(schedule.time or "") or (0, 0)
    today = after.astimezone(tz).date()
    for offset in range(9):  # 8 days covers "same weekday next week"
        day = today + timedelta(days=offset)
        if day.weekday() not in schedule.days:
            continue
        at = datetime.combine(day, time(hour, minute), tzinfo=tz).astimezone(timezone.utc)
        if at > after:
            return at
    raise AssertionError("unreachable: a weekday always recurs within 8 days")

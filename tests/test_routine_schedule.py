"""Routine schedules: wall-clock times, weekdays, DST and intervals."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from routine_schedule import Schedule, next_run, valid_timezone

TZ = "America/Chicago"


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def daily(time="07:00", days=(0, 1, 2, 3, 4, 5, 6)):
    return Schedule(kind="daily", time=time, days=list(days))


def test_daily_later_today_then_tomorrow():
    # Wed 2026-11-25, CST (UTC-6): 07:00 local = 13:00Z
    assert next_run(daily(), utc(2026, 11, 25, 12, 0), TZ) == utc(2026, 11, 25, 13, 0)
    assert next_run(daily(), utc(2026, 11, 25, 13, 0), TZ) == utc(2026, 11, 26, 13, 0)


def test_weekdays_skip_the_weekend():
    # After Friday's run the next is Monday.
    assert next_run(daily(days=(0, 1, 2, 3, 4)), utc(2026, 11, 27, 14, 0), TZ) == utc(2026, 11, 30, 13, 0)


def test_single_weekday_wraps_a_full_week():
    assert next_run(daily(days=(2,)), utc(2026, 11, 25, 13, 0), TZ) == utc(2026, 12, 2, 13, 0)


def test_wall_clock_is_kept_across_dst():
    # CDT before 2026-11-01, CST after.
    assert next_run(daily(), utc(2026, 10, 31, 11, 0), TZ) == utc(2026, 10, 31, 12, 0)
    assert next_run(daily(), utc(2026, 10, 31, 12, 0), TZ) == utc(2026, 11, 1, 13, 0)


def test_nonexistent_time_runs_once_after_the_jump():
    # 02:30 does not exist on 2027-03-14 in Chicago.
    first = next_run(daily("02:30"), utc(2027, 3, 14, 0, 0), TZ)
    assert first == utc(2027, 3, 14, 8, 30)
    assert next_run(daily("02:30"), first, TZ) == utc(2027, 3, 15, 7, 30)


def test_repeated_time_runs_once():
    # 01:30 happens twice on 2026-11-01.
    first = next_run(daily("01:30"), utc(2026, 11, 1, 0, 0), TZ)
    assert first == utc(2026, 11, 1, 6, 30)
    assert next_run(daily("01:30"), first, TZ) == utc(2026, 11, 2, 7, 30)


def test_interval():
    s = Schedule(kind="interval", hours=6)
    assert next_run(s, utc(2026, 11, 25, 1, 0), TZ) == utc(2026, 11, 25, 7, 0)


@pytest.mark.parametrize("bad", [
    {"kind": "daily", "time": "7am"}, {"kind": "daily", "time": "24:00"}, {"kind": "daily"},
    {"kind": "daily", "time": "07:00", "days": []}, {"kind": "daily", "time": "07:00", "days": [7]},
    {"kind": "interval"}, {"kind": "interval", "hours": 0}, {"kind": "interval", "hours": 169},
    {"kind": "cron", "time": "07:00"}, {"kind": "daily", "time": "07:00", "extra": 1},
])
def test_invalid_schedules_are_rejected(bad):
    with pytest.raises(ValidationError):
        Schedule(**bad)


def test_days_are_deduped_and_sorted():
    assert daily(days=(4, 0, 4)).days == [0, 4]


def test_timezone_validation():
    assert valid_timezone("America/Chicago") and valid_timezone("UTC")
    for bad in ("Mars/Base", "", "../etc/passwd", "/etc/passwd"):
        assert not valid_timezone(bad)

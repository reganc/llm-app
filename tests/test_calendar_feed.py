"""Today's events from an iCal feed: zones, all-day, repeats, exceptions, odd input."""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from zoneinfo import ZoneInfo

import httpx
import pytest

import calendar_feed as cf

CHI = ZoneInfo("America/Chicago")
DAY = date(2026, 11, 26)   # a Thursday
URL = "https://calendar.google.com/calendar/ical/me%40gmail.com/private-0123456789abcdef/basic.ics"


def ics(*events, zone="America/Chicago"):
    body = "\r\n".join(f"BEGIN:VEVENT\r\n{e.strip()}\r\nEND:VEVENT" for e in events)
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nX-WR-TIMEZONE:{zone}\r\n{body}\r\nEND:VCALENDAR\r\n"


def got(*events, day=DAY, **kw):
    return [(("all day" if e.all_day else e.start.strftime("%H:%M")), e.title)
            for e in cf.events_on(ics(*events, **kw), day, CHI)]


def test_zoned_utc_floating_and_all_day_in_order():
    assert got(
        "UID:a\nSUMMARY:Dentist\nDTSTART;TZID=America/Chicago:20261126T090000\nDTEND;TZID=America/Chicago:20261126T100000",
        "UID:b\nSUMMARY:Call\nDTSTART:20261126T200000Z\nDTEND:20261126T203000Z",
        "UID:c\nSUMMARY:Floating\nDTSTART:20261126T070000\nDURATION:PT1H",
        "UID:d\nSUMMARY:Thanksgiving\nDTSTART;VALUE=DATE:20261126\nDTEND;VALUE=DATE:20261127",
        "UID:e\nSUMMARY:Tomorrow\nDTSTART;TZID=America/Chicago:20261127T090000\nDTEND;TZID=America/Chicago:20261127T100000",
    ) == [("all day", "Thanksgiving"), ("07:00", "Floating"), ("09:00", "Dentist"), ("14:00", "Call")]


def test_multi_day_and_overnight_events():
    assert got(
        "UID:a\nSUMMARY:Trip\nDTSTART;VALUE=DATE:20261125\nDTEND;VALUE=DATE:20261128",
        "UID:b\nSUMMARY:Ended yesterday\nDTSTART;VALUE=DATE:20261125\nDTEND;VALUE=DATE:20261126",
        "UID:c\nSUMMARY:Overnight\nDTSTART;TZID=America/Chicago:20261125T230000\nDTEND;TZID=America/Chicago:20261126T010000",
        "UID:d\nSUMMARY:Ends at midnight\nDTSTART;TZID=America/Chicago:20261125T230000\nDTEND;TZID=America/Chicago:20261126T000000",
        "UID:e\nSUMMARY:Reminder\nDTSTART;TZID=America/Chicago:20261126T000000",
    ) == [("all day", "Trip"), ("23:00", "Overnight"), ("00:00", "Reminder")]


def test_weekly_repeat_with_skips_moves_and_cancellations():
    weekly = "UID:w\nSUMMARY:Standup\nDTSTART;TZID=America/Chicago:20261001T093000\nDURATION:PT15M\nRRULE:FREQ=WEEKLY;BYDAY=TH"
    assert got(weekly) == [("09:30", "Standup")]
    assert got(weekly + "\nEXDATE;TZID=America/Chicago:20261126T093000") == []
    moved = "UID:w\nRECURRENCE-ID;TZID=America/Chicago:20261126T093000\nSUMMARY:Standup (late)\nDTSTART;TZID=America/Chicago:20261126T110000\nDURATION:PT15M"
    assert got(weekly, moved) == [("11:00", "Standup (late)")]
    cancelled = "UID:w\nRECURRENCE-ID;TZID=America/Chicago:20261126T093000\nSTATUS:CANCELLED\nDTSTART;TZID=America/Chicago:20261126T093000"
    assert got(weekly, cancelled) == []
    other_week = moved.replace("20261126T", "20261119T")
    assert got(weekly, other_week) == [("09:30", "Standup")]


def test_repeats_keep_local_time_across_daylight_saving():
    assert got("UID:w\nSUMMARY:Gym\nDTSTART;TZID=America/Chicago:20260701T063000\nRRULE:FREQ=DAILY") == [("06:30", "Gym")]
    assert got("UID:w\nSUMMARY:UTC call\nDTSTART:20260701T120000Z\nRRULE:FREQ=WEEKLY;BYDAY=TH") == [("06:00", "UTC call")]


@pytest.mark.parametrize("until,want", [("20261126T235959Z", 1), ("20261125T235959Z", 0), ("20261126", 1),
                                        ("20261126T093000", 1), ("20261120", 0)])
def test_until_in_its_different_forms(until, want):
    rule = f"UID:w\nSUMMARY:S\nDTSTART;TZID=America/Chicago:20261001T093000\nRRULE:FREQ=DAILY;UNTIL={until}"
    assert len(got(rule)) == want


def test_count_monthly_and_yearly():
    assert got("UID:a\nSUMMARY:Rent\nDTSTART;VALUE=DATE:20260126\nRRULE:FREQ=MONTHLY") == [("all day", "Rent")]
    assert got("UID:a\nSUMMARY:Bday\nDTSTART;VALUE=DATE:20101126\nRRULE:FREQ=YEARLY") == [("all day", "Bday")]
    assert got("UID:a\nSUMMARY:Few\nDTSTART;VALUE=DATE:20261120\nRRULE:FREQ=DAILY;COUNT=3") == []


def test_cancelled_folded_escaped_and_untitled():
    assert got(
        "UID:a\nSTATUS:CANCELLED\nSUMMARY:Off\nDTSTART;TZID=America/Chicago:20261126T080000",
        "UID:b\nSUMMARY:Lunch\\, with Ann\; and\\nBob at \\\"Joe's\\\"\nDTSTART;TZID=America/Chicago:20261126T12\n 0000",
        "UID:c\nDTSTART;TZID=America/Chicago:20261126T130000",
    ) == [("12:00", "Lunch, with Ann; and Bob at \\\"Joe's\\\""), ("13:00", "(no title)")]


def test_other_zones_and_unknown_zones():
    assert got("UID:a\nSUMMARY:NY\nDTSTART;TZID=America/New_York:20261126T100000") == [("09:00", "NY")]
    assert got("UID:a\nSUMMARY:Win\nDTSTART;TZID=Central Standard Time:20261126T100000") == [("10:00", "Win")]
    assert got("UID:a\nSUMMARY:Odd\nDTSTART;TZID=\"Nowhere/Land\":20261126T100000") == [("10:00", "Odd")]
    assert got("UID:a\nSUMMARY:Float\nDTSTART:20261126T100000", zone="America/New_York") == [("09:00", "Float")]


@pytest.mark.parametrize("bad", [
    "UID:a\nSUMMARY:x\nDTSTART:garbage", "UID:a\nSUMMARY:x\nDTSTART;TZID=America/Chicago:20261126T090000\nRRULE:FREQ=NOPE",
    "UID:a\nSUMMARY:no start", "UID:a\nSUMMARY:x\nDTSTART:20261126T090000\nDURATION:soon",
])
def test_a_bad_event_is_skipped_and_the_rest_kept(bad):
    good = "UID:g\nSUMMARY:Good\nDTSTART;TZID=America/Chicago:20261126T090000"
    assert ("09:00", "Good") in got(bad, good)


@pytest.mark.parametrize("text", ["", "not a calendar", "BEGIN:VEVENT", "END:VEVENT\nBEGIN:VCALENDAR", "\x00" * 50])
def test_junk_never_raises(text):
    assert cf.events_on(text, DAY, CHI) == []


def test_the_same_event_listed_twice_shows_once():
    e = "UID:a\nSUMMARY:Dup\nDTSTART;TZID=America/Chicago:20261126T090000"
    assert got(e, e.replace("UID:a", "UID:b")) == [("09:00", "Dup")]


def test_bounded_title_and_count():
    events = [f"UID:{i}\nSUMMARY:{i}{'x' * 500}\nDTSTART;TZID=America/Chicago:20261126T0{i % 10}0000" for i in range(40)]
    found = cf.events_on(ics(*events), DAY, CHI)
    assert len(found) == 20 and max(len(e.title) for e in found) == 120


# ── fetching ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", ["http://calendar.google.com/calendar/ical/x/basic.ics", "https://evil.example/calendar/ical/x",
                                 "https://calendar.google.com/calendar/embed?src=x", "file:///etc/passwd", ""])
def test_only_google_secret_addresses(url):
    with pytest.raises(cf.CalendarError):
        cf.check_address(url)


def _serve(monkeypatch, response):
    real = httpx.AsyncClient

    def handler(request):
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(cf.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.parametrize("response,word", [
    (httpx.Response(404), "404"), (httpx.Response(302, headers={"location": "https://x"}), "302"),
    (httpx.ConnectError(URL), "ConnectError"), (httpx.Response(200, content=b"x" * (cf._MAX_BYTES + 1)), "too large"),
])
def test_fetch_failures_never_show_the_address(monkeypatch, response, word):
    _serve(monkeypatch, response)
    with pytest.raises(cf.CalendarError) as e:
        asyncio.run(cf.fetch(URL))
    assert word in str(e.value) and "private-" not in str(e.value)


def test_fetch_and_the_address_is_redacted_from_logs(monkeypatch, caplog):
    _serve(monkeypatch, httpx.Response(200, text=ics("UID:a\nSUMMARY:Hi\nDTSTART;VALUE=DATE:20261126")))
    with caplog.at_level(logging.INFO, logger="httpx"):
        assert "Hi" in asyncio.run(cf.fetch(URL))
    record = logging.LogRecord("httpx", logging.INFO, "f", 1, "GET %s", (URL,), None)
    assert cf.RedactCalendar().filter(record) and "private-" not in record.getMessage()
    assert "private-" not in caplog.text

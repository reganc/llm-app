"""Today's events from a calendar's private iCal address (Google: "Secret address in
iCal format"). Read-only, parsed in code; repeating events use dateutil.

The address itself is a password: anyone with it can read the calendar. It is never
logged (httpx logs URLs, so `RedactCalendar` filters them) and errors never carry it.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from dateutil import rrule

log = logging.getLogger("llm-api.calendar")

_MAX_BYTES = 5_000_000
_MAX_EVENTS = 20
_TITLE_CHARS = 120
_HOSTS = ("calendar.google.com",)
_SECRET_RE = re.compile(r"(/calendar/ical/)[^\s\"']+")
# Outlook invites saved into Google sometimes keep Windows zone names.
_WINDOWS = {"Eastern Standard Time": "America/New_York", "Central Standard Time": "America/Chicago",
            "Mountain Standard Time": "America/Denver", "Pacific Standard Time": "America/Los_Angeles",
            "GMT Standard Time": "Europe/London", "W. Europe Standard Time": "Europe/Berlin",
            "UTC": "UTC", "Coordinated Universal Time": "UTC"}


class CalendarError(Exception):
    """Safe to show: never contains the address."""


class RedactCalendar(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if _SECRET_RE.search(text):
            record.msg, record.args = _SECRET_RE.sub(r"\1<redacted>", text), ()
        return True


def install_redaction() -> None:
    for target in (logging.getLogger("httpx"), *logging.getLogger().handlers):
        if not any(isinstance(f, RedactCalendar) for f in target.filters):
            target.addFilter(RedactCalendar())


install_redaction()


@dataclass(frozen=True)
class Event:
    title: str
    start: datetime          # local time; midnight for all-day events
    all_day: bool


def check_address(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in _HOSTS or "/ical/" not in parts.path:
        raise CalendarError("The calendar address must be Google Calendar's “Secret address in iCal format”.")


async def fetch(url: str) -> str:
    check_address(url)
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise CalendarError(f"Google Calendar answered with an error ({response.status_code}). "
                                        "If you reset the secret address, put the new one in .env.")
                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > _MAX_BYTES:
                        raise CalendarError("The calendar is too large to read.")
    except httpx.HTTPError as e:
        raise CalendarError(f"Google Calendar could not be reached ({type(e).__name__}).") from None
    return body.decode("utf-8", errors="replace")


# ── parsing ──────────────────────────────────────────────────────────────────

def _unfold(text: str) -> list[str]:
    out: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        elif line:
            out.append(line)
    return out


def _split(line: str) -> tuple[str, dict[str, str], str]:
    """("DTSTART", {"TZID": "America/Chicago"}, "20261126T090000")."""
    quoted, at = False, -1
    for i, char in enumerate(line):
        if char == '"':
            quoted = not quoted
        elif char == ":" and not quoted:
            at = i
            break
    head, value = (line[:at], line[at + 1:]) if at >= 0 else (line, "")
    name, *params = head.split(";")
    found = {}
    for p in params:
        key, _, val = p.partition("=")
        found[key.upper()] = val.strip('"')
    return name.upper(), found, value


def _text(value: str) -> str:
    out = re.sub(r"\\([\;,nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), value)
    return " ".join(out.split())


def _zone(name: str | None, fallback: ZoneInfo) -> ZoneInfo:
    if not name:
        return fallback
    try:
        return ZoneInfo(_WINDOWS.get(name, name))
    except (ZoneInfoNotFoundError, ValueError):
        return fallback


def _when(value: str, params: dict[str, str], tz: ZoneInfo) -> tuple[datetime, bool]:
    """(aware datetime, all day)."""
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        d = datetime.strptime(value[:8], "%Y%m%d")
        return d.replace(tzinfo=tz), True
    if value.endswith("Z"):
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc), False
    return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_zone(params.get("TZID"), tz)), False


_DURATION = re.compile(r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def _duration(value: str) -> timedelta:
    m = _DURATION.match(value.strip())
    if not m:
        return timedelta(0)
    w, d, h, mi, s = (int(x or 0) for x in m.groups()[1:])
    span = timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)
    return -span if m.group(1) == "-" else span


def _events(lines: list[str]) -> tuple[list[dict], str | None]:
    """Raw VEVENTs and the calendar's own zone."""
    events: list[dict] = []
    current: dict | None = None
    depth, cal_zone = 0, None
    for line in lines:
        name, params, value = _split(line)
        if name == "BEGIN":
            depth += 1
            if value.upper() == "VEVENT":
                current = {"props": []}
            continue
        if name == "END":
            depth -= 1
            if value.upper() == "VEVENT" and current is not None:
                events.append(current)
                current = None
            continue
        if current is not None:
            current["props"].append((name, params, value))
        elif name == "X-WR-TIMEZONE":
            cal_zone = value.strip()
    return events, cal_zone


def _instances(raw: dict, tz: ZoneInfo, start: datetime, end: datetime) -> tuple[str, str | None, datetime | None, list[tuple[datetime, timedelta, bool]]]:
    """(uid, title or None if cancelled, recurrence id, [(start, length, all day)]) for
    occurrences near start..end; the caller keeps the ones on the day."""
    props: dict[str, list[tuple[dict, str]]] = {}
    for name, params, value in raw["props"]:
        props.setdefault(name, []).append((params, value))

    def one(key):
        return props.get(key, [(None, None)])[0]
    uid = (one("UID")[1] or "").strip()
    if (one("STATUS")[1] or "").strip().upper() == "CANCELLED":
        title = None
    else:
        title = _text(one("SUMMARY")[1] or "") or "(no title)"
    params, value = one("DTSTART")
    if value is None:
        return uid, title, None, []
    first, all_day = _when(value, params, tz)
    end_params, end_value = one("DTEND")
    if end_value is not None:
        length = _when(end_value, end_params, tz)[0] - first
    elif one("DURATION")[1]:
        length = _duration(one("DURATION")[1])
    else:
        length = timedelta(days=1) if all_day else timedelta(0)
    rid = None
    if one("RECURRENCE-ID")[1]:
        rid = _when(one("RECURRENCE-ID")[1], one("RECURRENCE-ID")[0], tz)[0]
    starts = [first]
    rule = one("RRULE")[1]
    if rule and not rid:
        starts = _expand(rule, first, start - max(length, timedelta(0)) - timedelta(days=2),
                         end + timedelta(days=2))
        for p, v in props.get("RDATE", []):
            starts += [_when(x, p, tz)[0] for x in v.split(",") if x.strip()]
        excluded = {_when(x, p, tz)[0] for p, v in props.get("EXDATE", []) for x in v.split(",") if x.strip()}
        starts = [s for s in starts if s not in excluded]
    return uid, title, rid, [(s, length, all_day) for s in starts]


def _expand(rule: str, first: datetime, after: datetime, before: datetime) -> list[datetime]:
    try:
        r = rrule.rrulestr(rule, dtstart=first)
    except (ValueError, TypeError):
        # UNTIL in floating time with a zoned start: read it in the start's zone.
        fixed = re.sub(r"UNTIL=(\d{8}(?:T\d{6})?)(?!Z)",
                       lambda m: "UNTIL=" + (m.group(1) if "T" in m.group(1) else m.group(1) + "T235959"), rule)
        naive = rrule.rrulestr(fixed, dtstart=first.replace(tzinfo=None))
        return [d.replace(tzinfo=first.tzinfo) for d in naive.between(after.astimezone(first.tzinfo).replace(tzinfo=None),
                                                                       before.astimezone(first.tzinfo).replace(tzinfo=None), inc=True)]
    return list(r.between(after, before, inc=True))


def _on_day(s: datetime, length: timedelta, all_day: bool, day: date, start: datetime, end: datetime) -> bool:
    if all_day:  # dates, whatever the zones: "the 26th" is the 26th
        last = (s + max(length, timedelta(days=1))).date()
        return s.date() <= day < last
    if length <= timedelta(0):
        return start <= s < end
    return s < end and s + length > start


def events_on(ics: str, day: date, tz: ZoneInfo) -> list[Event]:
    """Events overlapping `day` in `tz`, all-day ones first, then by start time."""
    start = datetime.combine(day, time(0), tz)
    end = start + timedelta(days=1)
    raw, cal_zone = _events(_unfold(ics))
    zone = _zone(cal_zone, tz)
    moved: dict[str, set[datetime]] = {}
    found: list[tuple[str, Event]] = []
    parsed = []
    for r in raw:
        try:
            parsed.append(_instances(r, zone, start, end))
        except (ValueError, TypeError, OverflowError) as e:
            log.info("calendar: skipped an event that could not be read (%s)", type(e).__name__)
    for uid, _title, rid, _hits in parsed:
        if rid is not None:
            moved.setdefault(uid, set()).add(rid)
    for uid, title, rid, hits in parsed:
        for s, length, all_day in hits:
            if rid is None and s in moved.get(uid, set()):
                continue  # this occurrence was changed or cancelled separately
            if title is None or not _on_day(s, length, all_day, day, start, end):
                continue
            local = datetime.combine(day, time(0), tz) if all_day else s.astimezone(tz)
            found.append((uid, Event(title[:_TITLE_CHARS], local, all_day)))
    unique = list(dict.fromkeys(e for _, e in found))
    return sorted(unique, key=lambda e: (not e.all_day, e.start, e.title))[:_MAX_EVENTS]

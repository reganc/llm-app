"""Read a watched page and compare it with last time. No model: alerts quote the page.

Three kinds of watch:
  * number — the first number after a label, alerting on any change or on
    crossing a limit;
  * phrase — whether some words are on the page;
  * change — lines added or removed. A line only one of two loads a few seconds
    apart has (ads, counters, clocks) is not reported, and one that does this at
    two checks never is.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from safe_fetch import FetchError, Page

_MAX_LINES, _LINE_CHARS = 4000, 400
_LABEL_REACH = 200      # a number must follow its label within this many characters
_SHOW_LINES = 8
# Seen to differ between two loads at this many checks: ignored for good.
_ALWAYS_NOISY = 2
_MAX_NOISY = 500
# Replacing more of the page than this looks like a redesign or an error page.
_MOSTLY = 0.6

_SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe"}
_INLINE = {"a", "b", "i", "em", "strong", "span", "small", "sup", "sub", "u", "s", "mark", "abbr",
           "code", "time", "label", "font", "data", "bdi", "q", "cite", "wbr", "ins", "del"}
_VOID = {"br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "base", "col", "embed",
         "track", "param"}
_NOISE = re.compile(
    r"^(?:(?:updated|posted|published|last updated|as of)\W*)?"
    r"(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?(?:\s*[a-z]{2,4})?|"
    r"(?:a|an|\d+)\s+(?:second|sec|minute|min|hour|hr)s?\s+ago|just now|[\d,.\s]+(?:views?|viewers?)?)$",
    re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.,])([-−]?)\s?[$€£¥₹]?\s?([-−]?)(\d[\d.,  ']*)")


# Dots that can't be thousands marks: some part after the first isn't three digits.
_VERSION = re.compile(r"(?!\d{1,3}(?:\.\d{3})+$)(\d+\.\d+)\.\d+(?:\.\d+)*")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP and tag not in _VOID:
            self.skipping.append(tag)
        if tag not in _INLINE:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skipping and tag == self.skipping[-1]:
            self.skipping.pop()
        if tag not in _INLINE:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(re.sub(r"\s+", " ", data))


def lines(page: Page) -> list[str]:
    """The page's visible text, one block per line."""
    text = page.text
    if "json" in page.content_type:
        try:
            text = json.dumps(json.loads(text), indent=0, ensure_ascii=False)
        except ValueError:
            pass
    elif "html" in page.content_type or text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        parser = _Text()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    out = []
    for line in text.splitlines():
        line = " ".join(line.split())[:_LINE_CHARS]
        if line:
            out.append(line)
            if len(out) >= _MAX_LINES:
                break
    if not out:
        raise FetchError("The page has no readable text. It may need a browser to load.")
    return out


# ── numbers ──────────────────────────────────────────────────────────────────

def parse_number(text: str) -> float | None:
    """"1,234.50", "1.234,50", "1 234" → a float. The last separator is the decimal
    point unless it is followed by exactly three digits and could be a thousands mark.
    A version ("3.15.2") gives its first two parts."""
    raw = re.sub(r"[  '\s]", "", text).rstrip(".,")
    if not re.fullmatch(r"\d[\d.,]*", raw):
        return None
    if version := _VERSION.fullmatch(raw):
        return float(version.group(1))
    last = max(raw.rfind("."), raw.rfind(","))
    if last >= 0:
        sep, other = raw[last], "," if raw[last] == "." else "."
        head, tail = raw[:last], raw[last + 1:]
        thousands = raw.count(sep) > 1 or (len(tail) == 3 and other not in raw and sep == ","
                                           and not head.startswith("0"))
        raw = re.sub(r"[.,]", "", raw) if thousands else f"{re.sub(r'[.,]', '', head)}.{tail}"
    try:
        return float(raw)
    except ValueError:
        return None


def find_number(page_lines: list[str], label: str) -> tuple[str, float, str]:
    """(as written, value, where it was found)."""
    text = "\n".join(page_lines)
    start = 0
    low, want = text.lower(), " ".join(label.lower().split())
    while (at := low.find(want, start)) >= 0:
        after = text[at + len(want):at + len(want) + _LABEL_REACH]
        for m in _NUMBER.finditer(after):
            value = parse_number(m.group(3))
            if value is None:
                continue
            negative = bool(m.group(1) or m.group(2))
            written = m.group(0).strip().rstrip(".,").strip()
            end = at + len(want) + m.end()
            where = " ".join(text[at:min(end + 20, len(text))].split())
            return written, (-value if negative else value), where[:200]
        start = at + len(want)
    raise FetchError(f"No number was found after “{label}”.")


# ── comparing ────────────────────────────────────────────────────────────────

@dataclass
class Result:
    state: dict                              # what to remember
    alert: str | None = None                 # exact text, or None if nothing to report
    shown: str = ""                          # the current reading, for the UI
    confirm: bool = False                    # load again before believing this
    lines: list[str] = field(default_factory=list)


def _number(rule: dict, now: list[str], before: dict | None) -> Result:
    written, value, where = find_number(now, rule["label"])
    when, limit = rule["when"], rule.get("limit") or 0.0
    met = (when == "below" and value < limit) or (when == "above" and value > limit)
    version = re.sub(r"[^\d.]", "", written)
    state = {"written": written, "value": value, "met": met,
             "version": version if _VERSION.fullmatch(version) else ""}
    out = Result(state, shown=where)
    if before is None:
        return out
    if when == "changes" and (value, state["version"]) != (before["value"], before["version"]):
        out.alert = f"{rule['label']}: {before['written']} → {written}"
    elif when in ("below", "above") and met and not before["met"]:
        out.alert = f"{rule['label']}: {before['written']} → {written} ({when} {limit:g})"
    return out


def _phrase(rule: dict, now: list[str], before: dict | None) -> Result:
    present = " ".join(rule["phrase"].lower().split()) in " ".join(now).lower()
    shown = f"“{rule['phrase']}” is {'' if present else 'not '}on the page"
    out = Result({"present": present}, shown=shown)
    if before is not None and present != before["present"]:
        wanted = {"appears": present, "disappears": not present}.get(rule["when"], True)
        if wanted:
            out.alert = f"“{rule['phrase']}” {'now appears' if present else 'is no longer'} on the page"
    return out


def _watched_lines(rule: dict, now: list[str]) -> list[str]:
    only = (rule.get("only") or "").lower()
    return list(dict.fromkeys(x for x in now if not _NOISE.match(x) and (not only or only in x.lower())))


def _cut(text: str) -> str:
    return text if len(text) <= 160 else text[:159] + "…"


def _change(rule: dict, now: list[str], before: dict | None) -> Result:
    current = _watched_lines(rule, now)
    counts = (before or {}).get("noisy", {})
    noisy = {x for x, n in counts.items() if n >= _ALWAYS_NOISY}
    out = Result({"lines": current, "noisy": counts}, lines=current,
                 shown=f"{len(current)} line{'' if len(current) == 1 else 's'} watched")
    if before is None:
        return out
    old = before["lines"]
    old_set, new_set = set(old), set(current)
    added = [x for x in current if x not in old_set and x not in noisy]
    removed = [x for x in old if x not in new_set and x not in noisy]
    if not added and not removed:
        return out
    if old and len(removed) > max(5, _MOSTLY * len(old)):
        out.alert = (f"Most of the page changed ({len(removed)} of {len(old)} lines). "
                     "It may have been redesigned, or the site may be showing an error page.")
        return out
    shown = [f"+ {_cut(x)}" for x in added[:_SHOW_LINES]] + [f"− {_cut(x)}" for x in removed[:_SHOW_LINES]]
    more = max(0, len(added) - _SHOW_LINES) + max(0, len(removed) - _SHOW_LINES)
    out.alert = "\n".join(shown + ([f"and {more} more"] if more else []))
    return out


_KINDS = {"number": _number, "phrase": _phrase, "change": _change}


def compare(rule: dict, now: list[str], before: dict | None) -> Result:
    """`before` is the remembered state, None on the first look."""
    result = _KINDS[rule["kind"]](rule, now, before)
    result.confirm = result.alert is not None
    return result


def confirmed(rule: dict, first: list[str], second: list[str], before: dict | None) -> Result:
    """Believe only what two loads a few seconds apart agree on."""
    a, b = compare(rule, first, before), compare(rule, second, before)
    if rule["kind"] != "change":
        same = a.state == b.state
        if same:
            return b
        # Still settling: keep what was known and look again next time.
        return Result(before if before is not None else b.state, shown=b.shown)
    unsteady = set(a.lines) ^ set(b.lines)
    noisy = dict((before or {}).get("noisy", {}))
    for line in unsteady:
        noisy[line] = noisy.get(line, 0) + 1
    noisy = dict(list(noisy.items())[-_MAX_NOISY:])
    # A line only one load had says nothing: it keeps whatever it was last time.
    old = set((before or {}).get("lines", []))
    steady = [x for x in b.lines if x not in unsteady] + sorted(unsteady & old)
    result = compare(rule, steady, {**before, "noisy": noisy} if before is not None else None)
    result.state = {"lines": steady, "noisy": noisy}
    return result

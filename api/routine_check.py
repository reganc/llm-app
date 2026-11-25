"""Keep only the points of an answer that its sources back.

Deterministic, per point: a list item (with its sub-items) or a paragraph must
cite a source, and enough of its words must appear in the text the model was
given for that source. This catches points with no source and points pinned to
an unrelated page. It can't tell a subtly wrong claim from a right one,
and it can't match a point to a page in another language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import agent

# Share of a point's words its cited text must contain.
_MIN_OVERLAP = 0.5
_STEM = 5
_MAX_REMOVED, _REMOVED_CHARS = 20, 400

_ITEM = re.compile(r"^(\s*)(?:[-*•+]|\d{1,2}[.)])\s+\S")
_FRAME = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*]+\*\*:?\s*$|[-*_]{3,}\s*$)")
_STOP = {"that", "this", "with", "from", "have", "been", "were", "will", "would", "could", "should",
         "their", "there", "they", "them", "than", "then", "into", "over", "after", "before", "about",
         "which", "while", "where", "when", "what", "also", "more", "most", "some", "such", "other",
         "according", "including", "reported", "reportedly", "report", "says", "said", "amid",
         "source", "sources", "news", "latest", "recent", "today", "potentially", "indicating",
         "suggesting", "highlighting", "indicates", "suggests", "potential", "a", "an", "the"}


@dataclass
class _Block:
    kind: str                      # "item" | "text" | "frame"
    lines: list[str] = field(default_factory=list)
    indent: int = 0
    reason: str | None = None      # why it is removed

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Checked:
    content: str                   # the kept points ("" if none)
    kept: int
    total: int
    removed: list[dict]            # {"text", "reason"}

    @property
    def enough(self) -> bool:
        return self.kept > 0 and self.kept * 2 >= self.total


def _blocks(content: str) -> list[_Block]:
    out: list[_Block] = []
    gap = True
    for line in content.splitlines():
        if not line.strip():
            gap = True
            continue
        item = _ITEM.match(line)
        indent = len(line) - len(line.lstrip())
        last = out[-1] if out else None
        if item and not (last and last.kind == "item" and indent > last.indent + 1):
            out.append(_Block("item", [line], indent))
        elif last and last.kind == "item" and (not gap or indent > last.indent + 1):
            last.lines.append(line)          # sub-item or wrapped line
        elif _FRAME.match(line) and not agent.cited_markers(line):
            out.append(_Block("frame", [line]))
        elif last and last.kind == "text" and not gap:
            last.lines.append(line)
        else:
            out.append(_Block("text", [line]))
        gap = False
    for b in out:                            # "Here are the stories:"
        if b.kind == "text" and b.text.rstrip("* ").endswith(":") and not agent.cited_markers(b.text):
            b.kind = "frame"
    return out


def _tokens(text: str) -> set[str]:
    """Word stems, numbers and short all-caps names (US, AI)."""
    out = set()
    for word in re.findall(r"\d[\d.,]*|[^\W\d_]+", text):
        if word[0].isdigit():
            out.add(word.replace(",", "").rstrip("."))
        elif word.lower() in _STOP:
            continue
        elif len(word) > 3 or (len(word) > 1 and word.isupper()):
            out.add(word.lower()[:_STEM])
    return out


def _plain(text: str) -> str:
    return re.sub(r"https?://\S+", "", agent.strip_markers(text))


def _backed(text: str, cited: list[str], texts: dict[str, str]) -> bool:
    source = " ".join(texts.get(m, "") for m in cited)
    if not source.strip():
        return False
    claim = _tokens(_plain(text))
    return not claim or len(claim & _tokens(source)) / len(claim) >= _MIN_OVERLAP


def check(content: str, texts: dict[str, str]) -> Checked:
    """`texts` is marker -> the text the model was given for it."""
    blocks = _blocks(content)
    points = [b for b in blocks if b.kind != "frame"]
    for b in points:
        cited = agent.cited_markers(b.text)
        if not cited:
            b.reason = "no source cited"
        elif not _backed(b.text, cited, texts):
            b.reason = "could not be matched to the cited source"
    kept: list[_Block] = []
    for i, b in enumerate(blocks):
        if b.kind == "frame":                # a heading stays if something under it does
            under = []
            for later in blocks[i + 1:]:
                if later.kind == "frame":
                    break
                under.append(later)
            if any(p.reason is None for p in under):
                kept.append(b)
        elif b.reason is None:
            kept.append(b)
    removed = [{"text": " ".join(_plain(b.text).split())[:_REMOVED_CHARS], "reason": b.reason}
               for b in points if b.reason]
    n = len(points) - len(removed)
    return Checked(_join(kept) if n else "", n, len(points), removed[:_MAX_REMOVED])


def _join(blocks: list[_Block]) -> str:
    out = ""
    for prev, b in zip([None, *blocks], blocks):
        tight = prev is not None and prev.kind == b.kind == "item"
        out += ("" if prev is None else "\n" if tight else "\n\n") + b.text
    return out

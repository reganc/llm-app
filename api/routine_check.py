"""Keep only the points of an answer that its sources back.

Deterministic, per point: a list item (with its sub-items) or a paragraph must
cite a source, and enough of its words must appear in the text the model was
given for that source. This catches points with no source and points pinned to
an unrelated page. Quotation marks must hold the source's exact words: a long quote
that doesn't is removed with its point, a short one loses its marks. Points resting
only on search previews are labelled.

It can't tell a subtly wrong claim from a right one,
and it can't match a point to a page in another language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import agent

# Share of a point's words its cited text must contain.
_MIN_OVERLAP = 0.5
_STEM = 5
# A wrong quote this short is wording ("flimsy guardrails"), not a claimed statement.
_SHORT_QUOTE_WORDS = 4
_PREVIEW_LABEL = "(search preview only)"
_MAX_REMOVED, _REMOVED_CHARS = 20, 400

_ITEM = re.compile(r"^(\s*)(?:[-*•+]|\d{1,2}[.)])\s+\S")
_QUOTE = re.compile(r'"([^"\n]{1,600})"|“([^”\n]{1,600})”|«([^»\n]{1,600})»')
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


def _flat(text: str) -> str:
    """For comparing wording: case, spacing, emphasis, quote and dash style aside."""
    text = agent.strip_markers(text).lower()
    text = re.sub(r"[\"'“”‘’«»*_`]", "", text)
    text = re.sub(r"[‐‑–—]", "-", text)
    return " ".join(text.split())


def _verbatim(quote: str, source: str) -> bool:
    pieces = [_flat(p).strip(" .,;:!?-") for p in re.split(r"\[?(?:…|\.\.\.)\]?", quote)]
    return all(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", source) for p in pieces if p)


def _check_quotes(b: _Block, source: str) -> None:
    """Unquote short wrong quotes; mark the point for removal on a long one."""
    source = _flat(source)

    def fix(m: re.Match) -> str:
        quote = next(g for g in m.groups() if g is not None)
        if _verbatim(quote, source):
            return m.group(0)
        if len(quote.split()) > _SHORT_QUOTE_WORDS:
            b.reason = "quotes words its source does not contain"
        return quote

    fixed = _QUOTE.sub(fix, b.text)
    if b.reason is None:
        b.lines = fixed.splitlines()


def check(content: str, texts: dict[str, str],
          previews: set[str] | frozenset[str] = frozenset()) -> Checked:
    """`texts` is marker -> the text the model was given for it; `previews` the
    markers where that is only a search preview."""
    blocks = _blocks(content)
    points = [b for b in blocks if b.kind != "frame"]
    for b in points:
        cited = agent.cited_markers(b.text)
        if not cited:
            b.reason = "no source cited"
        elif not _backed(b.text, cited, texts):
            b.reason = "could not be matched to the cited source"
        else:
            _check_quotes(b, "\n".join(texts.get(m, "") for m in cited))
            if b.reason is None and all(m in previews for m in cited):
                b.lines = [*b.lines[:-1], f"{b.lines[-1].rstrip()} {_PREVIEW_LABEL}"]
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

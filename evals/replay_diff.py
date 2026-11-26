"""Pure comparison for model replays (`replay.py`): no I/O, unit-tested offline.

Two models answer the same real question. Nothing here judges which answer is
better: that takes a person reading both. Code only records what it can check
(the idea of Anthropic's Messages API migration, 2026-11-25: the checker is
deterministic, and differences are grouped by kind so they can be counted).

A *flaw* is a problem in one answer on its own: a regression when only the
candidate has it, a fix when only the baseline does. A *difference* is neutral,
a pointer to the answers worth reading side by side.
"""
from __future__ import annotations

import re
import statistics

import scoring

FLAWS = ("error", "empty", "truncated", "bad_citations", "uncited_web", "repaired")
DIFFERENCES = ("search_differs", "numbers_differ", "length_differs", "slower", "faster")

# Bigger than this ratio between the two answers' lengths is worth a look.
LENGTH_RATIO = 2.5
# Slower/faster only when the gap is both relative and absolute.
LATENCY_RATIO, LATENCY_MIN_S = 2.0, 5.0

_MARKER = re.compile(r"\[\s*[WXAL]\d+(?:\s*[,;]\s*[WXAL]\d+)*\s*\]")
_LIST_NUMBER = re.compile(r"(?m)^\s*\d+[.)]\s")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def numbers(text: str) -> set[str]:
    """Figures stated in an answer, without citation markers or list numbering."""
    clean = _LIST_NUMBER.sub(" ", _MARKER.sub(" ", text or ""))
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(clean)}


def facts(obs: dict) -> dict:
    """What code can see in one answer."""
    if obs.get("error"):
        return {"flaws": ["error"], "detail": obs["error"][:200],
                "latency_s": obs.get("latency_s"), "searched": False, "chars": 0, "numbers": []}
    content = obs.get("content") or ""
    cited = scoring.extract_citations(content)
    shown = scoring.available_markers(obs.get("web_search"), obs.get("memory"))
    bad = sorted(f"{k}{n}" for k, nums in cited.items() for n in nums
                 if n < 1 or n > shown.get(k, 0))
    agent = obs.get("agent") or {}
    flaws = []
    if not content.strip():
        flaws.append("empty")
    if obs.get("finish_reason") == "length":
        flaws.append("truncated")
    if bad:
        flaws.append("bad_citations")
    if shown.get("W", 0) and content.strip() and not cited:
        flaws.append("uncited_web")
    if agent.get("citation_repair"):
        flaws.append("repaired")       # the gateway had to fix its citations
    return {
        "flaws": flaws,
        "detail": f"cited {', '.join(bad)}; shown {shown}" if bad else "",
        "latency_s": obs.get("latency_s"),
        "searched": bool((obs.get("web_search") or {}).get("triggered")),
        "tool_calls": [f"{t.get('name')}({t.get('query')!r})"
                       for t in agent.get("tool_calls") or []],
        "chars": len(content),
        "numbers": sorted(numbers(content)),
    }


def differences(a: dict, b: dict) -> list[str]:
    """Neutral ways the candidate's answer (b) differs from the baseline's (a)."""
    if "error" in a["flaws"] or "error" in b["flaws"]:
        return []
    out = []
    if a["searched"] != b["searched"]:
        out.append("search_differs")
    if a["numbers"] and b["numbers"] and set(a["numbers"]) != set(b["numbers"]):
        out.append("numbers_differ")
    short, long_ = sorted((a["chars"], b["chars"]))
    if short and long_ / short > LENGTH_RATIO:
        out.append("length_differs")
    la, lb = a.get("latency_s"), b.get("latency_s")
    if la is not None and lb is not None:
        if lb > la * LATENCY_RATIO and lb - la >= LATENCY_MIN_S:
            out.append("slower")
        elif la > lb * LATENCY_RATIO and la - lb >= LATENCY_MIN_S:
            out.append("faster")
    return out


def compare(base_obs: dict, cand_obs: dict) -> dict:
    a, b = facts(base_obs), facts(cand_obs)
    return {
        "baseline": a, "candidate": b,
        "regressed": [f for f in b["flaws"] if f not in a["flaws"]],
        "fixed": [f for f in a["flaws"] if f not in b["flaws"]],
        "differs": differences(a, b),
    }


def _p(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def summarize(rows: list[dict]) -> dict:
    """rows: [{"id", "compare": compare(...)}] → counts per kind, for each model."""
    def count(key: str, kinds) -> dict[str, int]:
        return {k: sum(k in r["compare"][key] for r in rows) for k in kinds}

    def side(name: str) -> dict:
        lat = [r["compare"][name]["latency_s"] for r in rows
               if r["compare"][name].get("latency_s") is not None
               and "error" not in r["compare"][name]["flaws"]]
        return {"flaws": {k: sum(k in r["compare"][name]["flaws"] for r in rows) for k in FLAWS},
                "clean": sum(not r["compare"][name]["flaws"] for r in rows),
                "latency_p50_s": _p(lat, 50), "latency_p95_s": _p(lat, 95)}

    return {
        "questions": len(rows),
        "baseline": side("baseline"), "candidate": side("candidate"),
        "regressed": count("regressed", FLAWS), "fixed": count("fixed", FLAWS),
        "differs": count("differs", DIFFERENCES),
        "regressed_ids": [r["id"] for r in rows if r["compare"]["regressed"]],
        "fixed_ids": [r["id"] for r in rows if r["compare"]["fixed"]],
    }


def reading_order(rows: list[dict]) -> list[dict]:
    """Regressions first, then fixes, then the most differences: what to read first."""
    return sorted(rows, key=lambda r: (-len(r["compare"]["regressed"]),
                                       -len(r["compare"]["fixed"]),
                                       -len(r["compare"]["differs"])))

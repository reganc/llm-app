"""Pure scoring for gateway evals — no I/O, so it is unit-tested offline.

An *observation* is what the runner saw for one case:

    {"content": str, "finish_reason": str | None, "prompt_tokens": int | None,
     "web_search": dict | None, "memory": dict | None, "error": str | None,
     "latency_s": float}

`finish_reason` and `prompt_tokens` are None in streaming mode (the SSE path
does not report them); checks that depend on them are skipped, not failed.

Each check returns ``Check(name, passed, detail)`` where ``passed`` is True,
False, or None (skipped — not observable or not asked for by the case).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

# [W1] · [L2] · [W1, W3] · [W1; X2] · [ X3 ] (models sometimes pad)
_CITE_GROUP_RE = re.compile(r"\[\s*((?:[WXAL]\d+)(?:\s*[,;]\s*[WXAL]\d+)*)\s*\]")
_CITE_ONE_RE = re.compile(r"([WXAL])(\d+)")

# Mirrors search.search_and_ingest's context block caps.
_WEB_CAP, _X_CAP, _AUX_CAP = 9, 10, 3

# Ollama silently drops the start of the prompt past num_ctx, so a prompt that
# lands within this many tokens of the window was almost certainly truncated.
CONTEXT_MARGIN = 64


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None
    detail: str = ""


def extract_citations(text: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for group in _CITE_GROUP_RE.findall(text or ""):
        for kind, num in _CITE_ONE_RE.findall(group):
            out.setdefault(kind, set()).add(int(num))
    return out


def available_markers(web_search: dict | None, memory: dict | None) -> dict[str, int]:
    """How many of each marker the model was actually shown."""
    reported = (web_search or {}).get("markers")
    if reported:  # agent mode renumbers across calls and reports the totals
        return {"W": reported.get("W", 0), "X": reported.get("X", 0),
                "A": reported.get("A", 0), "L": int((memory or {}).get("used") or 0)}
    results = (web_search or {}).get("results") or []
    engines = [r.get("engine", "") for r in results]
    return {
        "W": min(sum(e not in ("x_twitter", "auxiliary_site") for e in engines), _WEB_CAP),
        "X": min(sum(e == "x_twitter" for e in engines), _X_CAP),
        "A": min(sum(e == "auxiliary_site" for e in engines), _AUX_CAP),
        "L": int((memory or {}).get("used") or 0),
    }


def _norm(text: str) -> str:
    # Case-, apostrophe- and accent-light normalisation for substring checks.
    return (text or "").lower().replace("’", "'").replace("²", "^2")


def _contains(content: str, needle: str) -> bool:
    return _norm(needle) in _norm(content)


def _check_citations(obs: dict) -> Check:
    cited = extract_citations(obs.get("content", ""))
    if not cited:
        return Check("citations_valid", None, "no citations")
    avail = available_markers(obs.get("web_search"), obs.get("memory"))
    bad = sorted(f"{k}{n}" for k, nums in cited.items() for n in nums
                 if n < 1 or n > avail.get(k, 0))
    if bad:
        return Check("citations_valid", False,
                     f"cited {', '.join(bad)}; shown {avail}")
    return Check("citations_valid", True)


def _check_memory_source(expect: dict, obs: dict) -> Check:
    needle = expect.get("memory_source_contains")
    if needle is None:
        return Check("memory_source", None)
    items = (obs.get("memory") or {}).get("items") or []
    for item in items:
        if _contains(f"{item.get('identifier', '')} {item.get('title', '')}", needle):
            return Check("memory_source", True)
    got = [i.get("identifier") or i.get("title") for i in items[:5]]
    return Check("memory_source", False, f"wanted {needle!r}; got {got}")


def score(case: dict, obs: dict, *, num_ctx: int) -> list[Check]:
    expect = case.get("expect", {})
    content = obs.get("content") or ""
    checks: list[Check] = []

    if obs.get("error"):
        return [Check("ok", False, obs["error"])]
    checks.append(Check("ok", True))
    checks.append(Check("non_empty", bool(content.strip())))

    fr = obs.get("finish_reason")
    checks.append(Check("not_truncated", None if fr is None else fr != "length",
                        "" if fr is None else f"finish_reason={fr}"))

    pt = obs.get("prompt_tokens")
    checks.append(Check("context_fit",
                        None if pt is None else pt < num_ctx - CONTEXT_MARGIN,
                        "" if pt is None else f"prompt_tokens={pt}/{num_ctx}"))

    checks.append(_check_citations(obs))

    if "web_search" in expect:
        searched = bool((obs.get("web_search") or {}).get("triggered"))
        checks.append(Check("route_web_search", searched == expect["web_search"],
                            f"searched={searched}"))
    if "memory_mode" in expect:
        mode = (obs.get("memory") or {}).get("mode")
        checks.append(Check("route_memory_mode", mode == expect["memory_mode"],
                            f"mode={mode}"))
    checks.append(_check_memory_source(expect, obs))

    if expect.get("must_cite"):
        checks.append(Check("cites_sources", bool(extract_citations(content))))
    if any_of := expect.get("contains_any"):
        hit = any(_contains(content, n) for n in any_of)
        checks.append(Check("contains_any", hit, "" if hit else f"none of {any_of}"))
    for needle in expect.get("contains_all", []):
        checks.append(Check(f"contains:{needle}", _contains(content, needle)))
    for needle in expect.get("not_contains", []):
        checks.append(Check(f"not_contains:{needle}", not _contains(content, needle)))

    return checks


def case_passed(checks: list[Check]) -> bool:
    """One run passes when no check failed (skipped checks don't count)."""
    return all(c.passed is not False for c in checks)


def case_result(runs: list[list[Check]]) -> dict:
    """Aggregate repeated runs of one case: strict majority must pass."""
    wins = sum(case_passed(r) for r in runs)
    rate = round(wins / len(runs), 2) if runs else 0.0
    return {"pass_rate": rate, "passed": wins * 2 > len(runs)}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(rows: list[dict]) -> dict:
    """rows: [{"id", "category", "runs": [{"checks": [Check], "latency_s"}]}]."""
    by_check: dict[str, list[bool]] = {}
    by_cat: dict[str, list[bool]] = {}
    flaky: list[str] = []
    latencies: list[float] = []
    for row in rows:
        result = case_result([run["checks"] for run in row["runs"]])
        by_cat.setdefault(row["category"], []).append(result["passed"])
        if 0 < result["pass_rate"] < 1:
            flaky.append(row["id"])
        for run in row["runs"]:
            if run.get("latency_s") is not None:
                latencies.append(run["latency_s"])
            for c in run["checks"]:
                if c.passed is None:
                    continue
                by_check.setdefault(c.name.split(":", 1)[0], []).append(c.passed)
    return {
        "cases": len(rows),
        "passed": sum(v for vals in by_cat.values() for v in vals),
        "runs_per_case": max((len(r["runs"]) for r in rows), default=0),
        "checks": {k: {"passed": sum(v), "total": len(v)} for k, v in sorted(by_check.items())},
        "categories": {k: {"passed": sum(v), "total": len(v)} for k, v in sorted(by_cat.items())},
        "flaky": flaky,
        "latency_p50_s": _pct(latencies, 50),
        "latency_p95_s": _pct(latencies, 95),
    }


def _rates(result: dict) -> dict[str, tuple[bool, float]]:
    # Results saved before --repeat existed carry only `passed`.
    return {r["id"]: (r["passed"], r.get("pass_rate", 1.0 if r["passed"] else 0.0))
            for r in result.get("rows", [])}


def compare(baseline: dict, current: dict) -> dict:
    """Per-case pass/fail flips and pass-rate changes between two saved runs."""
    base, cur = _rates(baseline), _rates(current)
    shared = sorted(base.keys() & cur.keys())
    return {
        "fixed": [i for i in shared if cur[i][0] and not base[i][0]],
        "regressed": [i for i in shared if base[i][0] and not cur[i][0]],
        "new": sorted(cur.keys() - base.keys()),
        "removed": sorted(base.keys() - cur.keys()),
        "rate_changed": {i: [base[i][1], cur[i][1]] for i in shared
                         if base[i][1] != cur[i][1]},
    }

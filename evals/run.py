"""Run the golden eval set against a live gateway and score it.

    python evals/run.py                      # all cases, JSON mode
    python evals/run.py --mode stream        # the SPA's streaming path
    python evals/run.py --only followup      # category or case-id substring
    python evals/run.py --compare evals/results/<baseline>.json

Every request is sent with `store: false`, so evals read memory but never
write to it — otherwise each run's turns would be recalled by the next.

Cases run sequentially: there is one GPU, and concurrent requests would
distort latency. Results land in evals/results/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

import scoring

ROOT = Path(__file__).resolve().parent
CASES_FILE = ROOT / "cases.json"
RESULTS_DIR = ROOT / "results"


def _dotenv(name: str) -> str:
    """Value of NAME in llm-app/.env — the file the gateway container reads."""
    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def _api_key() -> str:
    key = os.getenv("LLM_API_KEY") or _dotenv("API_KEY")
    if not key:
        sys.exit("no API key: set LLM_API_KEY or API_KEY in llm-app/.env")
    return key


def _messages(case: dict) -> list[dict]:
    return case.get("messages") or [{"role": "user", "content": case["prompt"]}]


def _observe_json(client: httpx.Client, url: str, body: dict) -> dict:
    r = client.post(url, json=body)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    choice = data["choices"][0]
    return {
        "content": choice["message"].get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
        "web_search": data.get("web_search"),
        "memory": data.get("memory"),
        "retrieval_query": data.get("retrieval_query"),
        "agent": data.get("agent"),
    }


def _observe_stream(client: httpx.Client, url: str, body: dict) -> dict:
    out: dict = {"content": "", "finish_reason": None, "prompt_tokens": None,
                 "web_search": None, "memory": None, "retrieval_query": None,
                 "agent": None}
    parts: list[str] = []
    event = None
    with client.stream("POST", url, json={**body, "stream": True}) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
                continue
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                event = None
                continue
            obj = json.loads(payload)
            if event == "llm.memory":
                out["memory"] = obj
            elif event == "llm.web_search":
                out["web_search"] = obj
            elif event == "llm.agent":
                out["agent"] = obj
            elif event == "llm.replace":  # server-corrected answer supersedes tokens
                parts = [obj.get("content", "")]
            elif event == "llm.retrieval_query":
                out["retrieval_query"] = obj.get("query")
            elif event is None:
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                parts.append(delta.get("content", ""))
            event = None
    out["content"] = "".join(parts)
    return out


def run_case(client: httpx.Client, url: str, case: dict, mode: str,
             temperature: float | None, use_agent: bool = False) -> dict:
    body = {"model": "default", "messages": _messages(case), "store": False}
    if use_agent:
        body["agent"] = True
    if temperature is not None:
        body["temperature"] = temperature
    started = time.monotonic()
    try:
        observe = _observe_stream if mode == "stream" else _observe_json
        obs = observe(client, url, body)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        obs = {"error": f"{type(e).__name__}: {e}"}
    obs["latency_s"] = round(time.monotonic() - started, 2)
    return obs


def _select(cases: list[dict], only: str | None) -> list[dict]:
    if not only:
        return cases
    return [c for c in cases if only == c["category"] or only in c["id"]]


def _print_case(row: dict) -> None:
    mark = "PASS" if row["passed"] else "FAIL"
    rate = f"  {row['pass_rate']:.0%}" if len(row["runs"]) > 1 else ""
    latency = statistics.median(r["latency_s"] for r in row["runs"])
    print(f"  {mark}  {row['id']:<34} {latency:>6.1f}s{rate}")
    for n, run in enumerate(row["runs"], 1):
        tag = f"run {n}: " if len(row["runs"]) > 1 else ""
        if rq := run["observation"].get("retrieval_query"):
            print(f"          ↳ {tag}retrieval query: {rq}")
        ag = run["observation"].get("agent") or {}
        for tc in ag.get("tool_calls") or []:
            print(f"          ⚙ {tag}{tc['name']}({tc['query']!r}) {tc['status']}")
        if rp := ag.get("citation_repair"):
            print(f"          ✎ {tag}citation repair: {rp}")
        for c in run["checks"]:
            if c["passed"] is False:
                print(f"          ✗ {tag}{c['name']}: {c['detail']}")
    sys.stdout.flush()  # progress is visible when output goes to a file


def _print_summary(summary: dict) -> None:
    runs = summary["runs_per_case"]
    print(f"\n{summary['passed']}/{summary['cases']} cases passed"
          f"{f' (majority of {runs} runs)' if runs > 1 else ''}   "
          f"latency p50 {summary['latency_p50_s']}s  p95 {summary['latency_p95_s']}s")
    if summary["flaky"]:
        print(f"flaky: {', '.join(summary['flaky'])}")
    print("\nby category:")
    for k, v in summary["categories"].items():
        print(f"  {k:<20} {v['passed']}/{v['total']}")
    print("\nby check (all runs):")
    for k, v in summary["checks"].items():
        print(f"  {k:<20} {v['passed']}/{v['total']}")


def _run_all(cases: list[dict], args, url: str, headers: dict) -> list[dict]:
    rows: list[dict] = []
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        for case in cases:
            runs = []
            for _ in range(args.repeat):
                obs = run_case(client, url, case, args.mode, args.temperature,
                               args.agent)
                checks = scoring.score(case, obs, num_ctx=args.num_ctx)
                runs.append({"checks": checks, "latency_s": obs["latency_s"],
                             "observation": obs})
            result = scoring.case_result([r["checks"] for r in runs])
            row = {"id": case["id"], "category": case["category"], **result,
                   "runs": runs}
            _print_case({**row, "runs": [{**r, "checks": [c.__dict__ for c in r["checks"]]}
                                         for r in runs]})
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.getenv("LLM_BASE_URL", "http://localhost:8030/v1"))
    ap.add_argument("--mode", choices=["json", "stream"], default="json")
    ap.add_argument("--only", help="category name or case-id substring")
    ap.add_argument("--agent", action="store_true",
                    help="send agent:true (native tool-calling loop)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per case; a case passes on a strict majority")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (default 0 for repeatable runs; "
                         "pass a negative value to use the server default)")
    ap.add_argument("--num-ctx", type=int,
                    default=int(os.getenv("NUM_CTX") or _dotenv("NUM_CTX") or "8192"),
                    help="the gateway's NUM_CTX (default: from llm-app/.env)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--compare", type=Path, help="earlier results file to diff against")
    ap.add_argument("--label", default="", help="tag appended to the results filename")
    args = ap.parse_args()
    if args.repeat < 1:
        sys.exit("--repeat must be >= 1")
    if args.temperature < 0:
        args.temperature = None

    cases = _select(json.loads(CASES_FILE.read_text()), args.only)
    if not cases:
        sys.exit(f"no cases match {args.only!r}")
    url = args.url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {_api_key()}"}

    temp = "server default" if args.temperature is None else args.temperature
    agent_tag = " · agent" if args.agent else ""
    print(f"{len(cases)} cases × {args.repeat} · {args.mode}{agent_tag} · "
          f"temperature {temp} · {url}\n")
    rows = _run_all(cases, args, url, headers)
    summary = scoring.summarize(rows)
    _print_summary(summary)

    serial = [{**r, "runs": [{**run, "checks": [c.__dict__ for c in run["checks"]]}
                             for run in r["runs"]]} for r in rows]
    result = {"started_at": datetime.now().isoformat(timespec="seconds"),
              "mode": args.mode, "url": url, "num_ctx": args.num_ctx,
              "repeat": args.repeat, "temperature": args.temperature,
              "agent": args.agent,
              "summary": summary, "rows": serial}
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.label}" if args.label else ""
    out = RESULTS_DIR / f"{stamp}-{args.mode}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out.relative_to(ROOT.parent)}")

    if args.compare:
        diff = scoring.compare(json.loads(args.compare.read_text()), result)
        print("\nvs baseline:", json.dumps(diff))


if __name__ == "__main__":
    main()

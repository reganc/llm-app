"""Replay real questions through the active model and a candidate, and compare.

    python evals/replay.py --candidate <ollama tag>            # newest 40 questions
    python evals/replay.py --candidate <tag> --limit 10 --label gemma

The golden set (`run.py`) checks 29 cases someone wrote; this asks the questions
actually put to the gateway: the SPA's saved conversations and the Telegram
chat's remembered turns, each with the history it had. Both models answer every
question through the gateway (`store: false`, so nothing is saved or recalled
later), and `replay_diff` records what code can check: empty, truncated, bad or
missing citations, a citation repair, whether it searched, which figures it
gave, how long it took. Which answer is *better* is for you to read: the report
puts regressions first.

- **Same conditions.** Questions alternate which model goes first, and each
  model is loaded before its timed call, so latency is answering, not loading.
  Web results can still differ between the two calls; `search_differs` and
  `numbers_differ` point at where to look.
- **The GPU.** Both models don't fit in 12 GB, so every question swaps them.
  Other apps wait during the run: run it when nothing else is busy. The active
  model is loaded again at the end.
- **Telegram turns** are replayed through the gateway, not the bot's own code
  path (its style prompt and `propose_action` tool aren't there), with memory
  off as in the bot, and its 1024-token budget.
- **Private.** The questions and answers are yours: they are written only to
  evals/results/ (gitignored, mode 0600) and never printed; the console shows
  question ids and kinds.

Needs the stack up (`:8030`), `API_KEY` in llm-app/.env, Docker access (to read
the Telegram history), and the candidate pulled first:
`docker exec ollama ollama pull <tag>`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

import replay_diff
import run as evalrun

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TELEGRAM_FILE = "/app/data/telegram_history.json"   # inside llm-api
HISTORY_MESSAGES = 12          # SPA: earlier messages sent with a question
TELEGRAM_TTL_S = 24 * 3600     # matches chat_memory.TTL_S
MIN_CHARS = 8                  # "thanks", "ok" aren't worth two model calls


def _usable(text) -> bool:
    return isinstance(text, str) and len(text.strip()) >= MIN_CHARS and not text.lstrip().startswith("/")


def from_conversation(conv: dict) -> list[dict]:
    """Every user message in a saved SPA conversation, with what came before it."""
    msgs = [m for m in conv.get("messages") or []
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    out = []
    for i, m in enumerate(msgs):
        if m["role"] == "user" and _usable(m["content"]):
            history = [{"role": h["role"], "content": h["content"]} for h in msgs[:i]][-HISTORY_MESSAGES:]
            out.append({"id": f"spa-{str(conv.get('id', ''))[:8]}-{i}", "source": "spa",
                        "at": conv.get("updated_at") or conv.get("created_at") or "",
                        "messages": history + [{"role": "user", "content": m["content"]}]})
    return out


def from_telegram(data: dict) -> list[dict]:
    """Each remembered Telegram question, with the turns the bot would have sent with it."""
    turns = [t for t in (data or {}).get("turns") or []
             if isinstance(t, dict) and isinstance(t.get("at"), (int, float))]
    out = []
    for k, t in enumerate(turns):
        if not _usable(t.get("q")):
            continue
        history = [m for p in turns[:k] if t["at"] - p["at"] <= TELEGRAM_TTL_S
                   for m in ({"role": "user", "content": p["q"]}, {"role": "assistant", "content": p["a"]})]
        out.append({"id": f"tg-{k}", "source": "telegram",
                    "at": datetime.fromtimestamp(t["at"]).isoformat(timespec="seconds"),
                    "messages": history + [{"role": "user", "content": t["q"]}]})
    return out


def newest(questions: list[dict], limit: int) -> list[dict]:
    return sorted(questions, key=lambda q: q["at"], reverse=True)[:limit]


def body_for(question: dict, model: str, temperature: float) -> dict:
    body = {"model": model, "messages": question["messages"], "store": False,
            "temperature": temperature}
    if question["source"] == "telegram":
        body |= {"memory": False, "max_tokens": 1024}
    return body


# ── talking to the stack ────────────────────────────────────────────────────

def _load_questions(client: httpx.Client, base: str) -> tuple[list[dict], list[str]]:
    notes, found = [], []
    convs = client.get(f"{base}/conversations").json().get("conversations", [])
    for c in convs:
        r = client.get(f"{base}/conversations/{c['id']}")
        if r.status_code == 200:
            found += from_conversation(r.json())
    try:
        raw = subprocess.run(["docker", "exec", "llm-api", "cat", TELEGRAM_FILE],
                             capture_output=True, text=True, timeout=20)
        if raw.returncode == 0:
            found += from_telegram(json.loads(raw.stdout))
        else:
            notes.append("no Telegram history yet")
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        notes.append(f"Telegram history unreadable ({type(e).__name__})")
    return found, notes


def _load_model(ollama: str, model: str) -> float:
    """Load a model before its timed call; returns the seconds it took."""
    started = time.monotonic()
    httpx.post(f"{ollama}/api/generate", json={"model": model}, timeout=300).raise_for_status()
    return round(time.monotonic() - started, 1)


def _models(ollama: str) -> set[str]:
    tags = httpx.get(f"{ollama}/api/tags", timeout=10).json().get("models", [])
    return {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}


def _ask(client, url, ollama, question, model, warm_name, temperature) -> dict:
    load_s = _load_model(ollama, warm_name)
    started = time.monotonic()
    try:
        obs = evalrun._observe_json(client, url, body_for(question, model, temperature))
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        obs = {"error": f"{type(e).__name__}: {e}"}
    obs["latency_s"] = round(time.monotonic() - started, 2)
    obs["load_s"] = load_s
    return obs


# ── output ──────────────────────────────────────────────────────────────────

def _private_write(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _facts_line(f: dict) -> str:
    bits = [f"{f.get('latency_s')}s", f"{f.get('chars', 0)} chars",
            "searched" if f.get("searched") else "no search"]
    bits += f.get("tool_calls") or []
    if f["flaws"]:
        bits.append("**" + ", ".join(f["flaws"]) + "**")
    if f.get("detail"):
        bits.append(f["detail"])
    return " · ".join(bits)


def report_md(meta: dict, summary: dict, rows: list[dict]) -> str:
    b, c = summary["baseline"], summary["candidate"]
    lines = [f"# Replay: {meta['baseline']} vs {meta['candidate']}", "",
             f"{meta['started_at']} · {summary['questions']} questions · temperature {meta['temperature']}",
             "", "| | baseline | candidate |", "|---|---|---|",
             f"| no flaws | {b['clean']} | {c['clean']} |",
             f"| latency p50 / p95 | {b['latency_p50_s']}s / {b['latency_p95_s']}s "
             f"| {c['latency_p50_s']}s / {c['latency_p95_s']}s |"]
    lines += [f"| {k} | {b['flaws'][k]} | {c['flaws'][k]} |" for k in replay_diff.FLAWS]
    lines += ["", "Differences (neutral, worth reading): "
              + ", ".join(f"{k} {v}" for k, v in summary["differs"].items()), ""]
    for row in replay_diff.reading_order(rows):
        cmp = row["compare"]
        tags = ([f"regressed: {', '.join(cmp['regressed'])}"] if cmp["regressed"] else []) \
            + ([f"fixed: {', '.join(cmp['fixed'])}"] if cmp["fixed"] else []) \
            + ([f"differs: {', '.join(cmp['differs'])}"] if cmp["differs"] else [])
        q = row["question"]
        lines += ["---", "", f"## {row['id']} ({q['source']}, {q['at']})" + (f" — {'; '.join(tags)}" if tags else ""),
                  "", f"**Question** ({len(q['messages']) - 1} earlier messages sent with it):", "",
                  "> " + q["messages"][-1]["content"].replace("\n", "\n> "), ""]
        for name, model in (("baseline", meta["baseline"]), ("candidate", meta["candidate"])):
            obs = row[name]
            lines += [f"### {name}: {model}", "", _facts_line(cmp[name]), "",
                      (obs.get("content") or obs.get("error") or "(empty)").strip(), ""]
    return "\n".join(lines) + "\n"


def _print_summary(summary: dict) -> None:
    b, c = summary["baseline"], summary["candidate"]
    print(f"\n{summary['questions']} questions   no flaws: baseline {b['clean']}, candidate {c['clean']}")
    print(f"latency p50/p95: baseline {b['latency_p50_s']}s/{b['latency_p95_s']}s, "
          f"candidate {c['latency_p50_s']}s/{c['latency_p95_s']}s")
    print(f"{'':<16}{'baseline':>9}{'candidate':>11}{'regressed':>11}{'fixed':>7}")
    for k in replay_diff.FLAWS:
        print(f"  {k:<14}{b['flaws'][k]:>9}{c['flaws'][k]:>11}"
              f"{summary['regressed'][k]:>11}{summary['fixed'][k]:>7}")
    print("differences:", ", ".join(f"{k} {v}" for k, v in summary["differs"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True, help="Ollama tag to compare (pull it first)")
    ap.add_argument("--limit", type=int, default=40, help="newest N questions (default 40)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="both models (default 0, so differences are the model's, not sampling)")
    ap.add_argument("--url", default=os.getenv("LLM_BASE_URL", "http://localhost:8030/v1"))
    ap.add_argument("--ollama", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {evalrun._api_key()}"}
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        baseline = client.get(f"{base}/settings").json()["default_model"]
        if args.candidate in (baseline, "default"):
            sys.exit("the candidate is the active model: nothing to compare")
        if args.candidate not in _models(args.ollama):
            sys.exit(f"{args.candidate} isn't pulled: docker exec ollama ollama pull {args.candidate}")
        found, notes = _load_questions(client, base)
        questions = newest(found, args.limit)
        if not questions:
            sys.exit("no questions to replay")
        bysrc = {s: sum(q["source"] == s for q in questions) for s in ("spa", "telegram")}
        print(f"{len(questions)} of {len(found)} questions (SPA {bysrc['spa']}, Telegram {bysrc['telegram']})"
              f"{' · ' + '; '.join(notes) if notes else ''}\n{baseline}  vs  {args.candidate}\n"
              "Both models are swapped on the GPU for every question: other apps wait meanwhile.\n")

        url, rows = f"{base}/chat/completions", []
        try:
            for i, q in enumerate(questions):
                order = [("baseline", baseline, "default"), ("candidate", args.candidate, args.candidate)]
                answers = {name: _ask(client, url, args.ollama, q, model, warm, args.temperature)
                           for name, model, warm in (order if i % 2 == 0 else order[::-1])}
                cmp = replay_diff.compare(answers["baseline"], answers["candidate"])
                rows.append({"id": q["id"], "question": q, "compare": cmp, **answers})
                tags = [f"regressed {','.join(cmp['regressed'])}"] * bool(cmp["regressed"]) \
                    + [f"fixed {','.join(cmp['fixed'])}"] * bool(cmp["fixed"]) \
                    + [f"differs {','.join(cmp['differs'])}"] * bool(cmp["differs"])
                print(f"  [{i + 1}/{len(questions)}] {q['id']:<20} "
                      f"{answers['baseline']['latency_s']:>6.1f}s {answers['candidate']['latency_s']:>6.1f}s"
                      f"  {'  '.join(tags)}", flush=True)
        finally:
            try:
                _load_model(args.ollama, "default")       # give the fleet its model back
            except httpx.HTTPError:
                print("could not reload the active model; the next request will load it")

    summary = replay_diff.summarize(rows)
    _print_summary(summary)
    meta = {"started_at": datetime.now().isoformat(timespec="seconds"), "baseline": baseline,
            "candidate": args.candidate, "temperature": args.temperature, "url": url}
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-replay{'-' + args.label if args.label else ''}"
    _private_write(RESULTS_DIR / f"{stem}.json", json.dumps({**meta, "summary": summary, "rows": rows}, indent=2))
    _private_write(RESULTS_DIR / f"{stem}.md", report_md(meta, summary, rows))
    print(f"\nread evals/results/{stem}.md (regressions first; private, not in git)")


if __name__ == "__main__":
    main()

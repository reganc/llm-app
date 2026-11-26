# Gateway evals

A golden set for the gateway's *agent* behaviour — routing, grounding, and
reply health — run against the live stack. Use it to baseline before changing
retrieval, search routing, prompts or the model, then diff after.

```bash
python evals/run.py                                  # JSON mode (two-pass path)
python evals/run.py --mode stream                    # SPA streaming path
python evals/run.py --only followup                  # one category / id substring
python evals/run.py --repeat 3                       # 3 runs/case, strict majority passes
python evals/run.py --legacy                         # legacy router path (agent:false)
python evals/run.py --agent                          # force agent:true
python evals/run.py --label tool-loop --compare evals/results/<baseline>.json
```

Runs default to **temperature 0** so results are repeatable; `--temperature
-1` uses the server default (0.7) — pair it with
`--repeat` to measure real-world variance. With `--repeat N` a case passes on
a strict majority, cases that sometimes fail are listed as `flaky`, and
`--compare` reports every pass-rate change, not just pass/fail flips. Compare
only runs made with the same mode, temperature and repeat count.

Needs the stack up (`:8030`) and `API_KEY` in `llm-app/.env` (or `LLM_API_KEY`).
Cases run sequentially (one GPU). Results go to `evals/results/` (gitignored).

**Evals never write memory.** Every request sends `store: false`, which skips
storing the turn and ingesting search pages while leaving retrieval on.
Without it each run's answers would be recalled by the next run.

## What is scored

Every case: `ok` (no transport error), `non_empty`, `not_truncated`,
`context_fit` (prompt below `num_ctx` — Ollama silently drops the start of an
over-long prompt), and `citations_valid` (every `[W#]/[X#]/[A#]/[L#]` refers
to a source the model was actually shown).

Per case, from `expect` in `cases.json`:

| Key | Check |
|---|---|
| `web_search: bool` | whether a web search fired (auto, forced, or model-requested) |
| `memory_mode: "library"` | routed to library mode |
| `memory_source_contains` | a retrieved memory item's id/title contains this |
| `must_cite` | reply cites at least one source |
| `contains_any` / `contains_all` / `not_contains` | case-insensitive substring checks |

Streaming mode can't observe `finish_reason` or token usage, so
`not_truncated` / `context_fit` are skipped there, not failed.

## Categories

- `no_search` — timeless facts; searching is wasted latency.
- `search_implicit` — needs fresh data but has no regex trigger word; exercises
  the LLM router.
- `search_explicit` — regex-forced search.
- `library_explicit` / `library_implicit` — anchored on documents saved in this
  instance's library (Kybalion, Marcuse, Abramelin). They are data-dependent:
  edit them if the library changes.
- `followup` — multi-turn; the last message alone is ambiguous ("when was he
  born?").

Time-sensitive cases check routing and citations only, never the answer text.

## Replaying real questions (`replay.py`)

Before swapping the model, ask the new one the questions you actually asked,
not just the 29 written cases. `replay.py` takes the newest saved SPA
conversations and remembered Telegram turns, each with the history it had, and
sends every one to the active model and to a candidate through the gateway
(`store: false`).

```bash
docker exec ollama ollama pull <tag>                       # the candidate, first
python evals/replay.py --candidate <tag>                   # newest 40 questions
python evals/replay.py --candidate <tag> --limit 10 --label quick
```

Code records only what it can check (`replay_diff.py`). **Flaws** are problems
in one answer: `error`, `empty`, `truncated`, `bad_citations` (a marker it was
never shown), `uncited_web` (web results shown, nothing cited), `repaired` (the
gateway had to fix its citations). A flaw only the candidate has is a
regression; one only the baseline has is a fix. **Differences** are neutral
pointers to answers worth reading: `search_differs`, `numbers_differ` (the
figures given aren't the same), `length_differs` (over 2.5×), `slower`/`faster`
(2× and at least 5 s). Which answer is better is for a person to read: the
report lists regressions first, then fixes, then the most differences.

- Questions alternate which model goes first, and each model is loaded before
  its timed call, so latency is answering, not loading. Web results can still
  differ between the two calls.
- Both models don't fit in 12 GB together, so every question swaps them and
  other apps wait: run it when the fleet is quiet. The active model is loaded
  again at the end.
- Telegram turns go through the gateway, not the bot's own code path (no style
  prompt or `propose_action`), with memory off and 1024 tokens, as in the bot.
- The questions and answers are private: written only to `evals/results/`
  (gitignored, mode 0600) as `.json` and a readable `.md`, never printed.
- A document or article conversation's first message ("Please load this
  document: …") is history, never a question: the app stores a fixed reply to
  it. Replayed as a question it looked like a citation problem (all three
  needed repair); the 17 real follow-ups in those conversations needed none.
- Calibrated 2026-11-26 with the active model against itself (`--candidate
  default:latest`): at temperature 0 the answers were identical, with no
  differences flagged; latency for the same answer varied up to 2×.

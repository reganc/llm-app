# Gateway evals

A golden set for the gateway's *agent* behaviour — routing, grounding, and
reply health — run against the live stack. Use it to baseline before changing
retrieval, search routing, prompts or the model, then diff after.

```bash
python evals/run.py                                  # JSON mode (two-pass path)
python evals/run.py --mode stream                    # SPA streaming path
python evals/run.py --only followup                  # one category / id substring
python evals/run.py --repeat 3                       # 3 runs/case, strict majority passes
python evals/run.py --agent                          # native tool-calling loop (agent:true)
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

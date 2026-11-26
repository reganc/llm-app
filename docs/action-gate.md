# Action gate: the bot proposes, you approve, code acts

Status: steps 1–2 built (2026-11-26): the gate, its log, Telegram cards, the three
llm-app actions, and `rerun_summary` (with Mentat's `?replace=true`). Steps 3–5 not
started.

Decided: restarts for mentat, nexus, delphi and trading (not llm-app or Ollama);
high risk keeps the second tap; MCP proposals allowed when step 4 lands.

## Why

Today the bot only reads. Your typed Telegram commands (`/run`, `/pause`,
`/resume`) act directly, because you wrote them. This adds a second path: the
model may **propose** an action (in a Telegram conversation, or from an MCP
client such as Claude Code), and nothing happens until you approve that exact
action. That is the fleet's boundary rule: authorize → policy check → execute →
immutable log, with a person confirming.

The gate is code. The model has no tool that executes anything, and no way to
produce an approval.

## The four actions

| Action | Arguments | Runs in | Risk | Notes |
|---|---|---|---|---|
| `run_routine` | routine name | llm-app (`routine_runner.start_manual`, a manual run) | low | routines only read; a manual run doesn't feed only-what's-new memory |
| `check_watch` | watched page name | llm-app (the `/v1/watches/{id}/check` code path) | low | |
| `set_routine` | routine name, `paused` / `active` | llm-app (same code as `/pause`, `/resume`) | low | |
| `rerun_summary` | `daily` / `weekly` | mentat (`POST /summaries/{kind}`) | low | needs the mentat change below |
| `restart_app` | fleet app id | executor (new, see below) | **high** | only apps that opt in via fleet.yaml `restart:` |

(`run_routine` and `check_watch` are the one "run something now" item, split so
each approval names one kind of thing.)

Never offered, whatever is asked: anything for a `privacy: private` app
(tax-planner), deleting anything, settings, secrets, sending messages to anyone
else, money. Adding an action means a code change and a test, never config
alone.

## Flow

```
model ──propose_action(action, args, reason)──▶ llm-app actions.py
                                                   │ 1. validate: known action, args valid,
                                                   │    target exists, policy (cooldown,
                                                   │    limits, never-list)
                                                   │ 2. store pending (exact args, hash,
                                                   │    expires in 15 min); log "proposed"
                                                   ▼
                          Telegram: card with [Approve] [Reject]   (and the SPA's list)
                                                   │ you tap Approve
                                                   ▼
                                    3. check: your chat, your user id, not expired,
                                       not used, args hash unchanged, policy again
                                       (high risk: second tap "Yes, restart …")
                                                   │
                         ┌─────────────────────────┴───────────────┐
                    llm-app actions                     restart_app → executor
                         │                                        │
                    4. run in code, log result, edit the card with the outcome
```

The model gets back "waiting for your approval (id …)". It can see the outcome
later via `action_status`, but nothing it says or does changes the state.

## What you see on the card

```
Proposed action  · high risk
Restart Mentat: containers mentat-api-1, mentat-worker-1, mentat-beat-1
About 20 s of downtime. Health check afterwards: http://…:8400/health

Why (the model's words): "the worker stopped picking up the hourly ingest"
Proposed from: Telegram chat, after reading 2 web pages
Expires 16:42

[Approve]  [Reject]
```

- The first lines are written by code from the stored arguments; this is
  what will run. The model's reason is shown separately and labelled.
- "After reading web pages" appears when the conversation turn included web or
  library results: that's where injected instructions would come from.
- After a decision the card is edited to say approved/rejected/expired and,
  later, the result, so old buttons can't be pressed again.

## Rules (all in code)

- **Approval:** a Telegram callback from the chosen chat *and* the chosen user
  id, or the SPA (which needs the API key). One approval = one action; it
  covers the stored arguments' hash and is consumed when used.
- **Expiry:** 15 minutes. Expired and rejected proposals can't be revived;
  the model must propose again.
- **Limits:** at most 3 pending at once, 10 proposals an hour; an identical
  pending proposal is not repeated. `restart_app`: one per app every 10
  minutes, 6 a day overall.
- **Unattended:** routines and the briefing never get `propose_action`; they
  stay read-only. Proposals only come from a conversation or an MCP client.
- **Re-checked at execution:** the target still exists, the app still opts in,
  cooldowns still hold. If not, it fails with the reason.
- **Log:** `DATA_DIR/actions/audit.jsonl`, hash-chained like mentat's research
  log: proposed, approved, rejected, expired, executed, failed, each with the
  action, arguments, source and times. No message text, no secrets. Pending
  proposals live in `DATA_DIR/actions/pending.json` (atomic writes, like
  routines).

## Restarts: a separate executor

llm-app faces the internet and stays without Docker access (the Docker socket
is root on this machine). Restarts go through a small new service, the
**executor**:

- One file, no model, runs in a container with the Docker socket, ports bound
  to `127.0.0.1` and `172.17.0.1` only (the port lockdown pattern).
- One endpoint: `POST /restart {app, action_id}` with a bearer token
  (`ACTIONS_EXECUTOR_TOKEN`, in both .env files).
- It reads fleet.yaml itself (read-only mount) and acts only on an app's
  `restart:` entry, e.g.

  ```yaml
  - id: mentat
    restart: {project: mentat, services: [api, worker, beat]}
  ```

  It restarts containers matching those compose labels, then polls the app's
  `health` URL for up to 90 s and returns ok/failed with the time taken.
- It keeps its own log and cooldowns, so even a compromised llm-app can only
  restart opted-in apps, at most once per 10 minutes each.

Honest limit: the executor can't verify that a person approved; it trusts
llm-app's token. What it guarantees is the blast radius: a bad request can
only restart listed apps, slowly. Opt-in to start with: mentat, nexus, delphi,
trading. Not llm-app itself or Ollama at first (restarting the bot from the
bot loses the card's outcome; Ollama is shared by everything).

## MCP

llm-app serves an MCP endpoint (`/mcp`, streamable HTTP, same bearer key)
with four tools:

- `fleet_status` — the hub (`/v1/hub`), read-only
- `list_actions` — what can be proposed and its arguments
- `propose_action` — as above; returns the pending id
- `action_status` — pending / approved / rejected / expired / done / failed

No execute tool. Claude Code (or any MCP client) proposes, and you approve on
your phone or in the SPA like any other proposal; the card says which client
proposed it. The same registry backs the bot's own agent tool, so there is one
list of actions and one gate.

## mentat change (done)

A day's or week's report that exists is kept with no model call and
`saved: "kept"`; `?replace=true` replaces it with its claims, and results say
"created", "replaced" or "kept". Scheduled runs never replace.

## Build order

1. **Registry, pending store, log, Telegram cards** with the three llm-app
   actions. Telegram polling adds `callback_query`. Tests: validation,
   expiry, one use, hash binding, wrong chat/user, limits, never-list, log
   chain, the model never reaching execution.
2. **mentat re-run fix**, then `rerun_summary`.
3. **Executor + `restart_app`** with fleet.yaml `restart:`; tests with a fake
   Docker client; then one real restart of mentat, approved by you.
4. **MCP endpoint**, tried from Claude Code.
5. SPA "Pending actions" panel (Telegram covers approval until then).

Each step is its own PR with the boundary text added to the app's CLAUDE.md.

## Open questions

1. Which apps may be restarted (proposed: mentat, nexus, delphi, trading)?
2. Should the high-risk second tap stay, or is one tap enough on your own phone?
3. Should MCP proposals from Claude Code be allowed at all, or Telegram only at
   first?

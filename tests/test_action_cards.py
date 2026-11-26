"""Proposals from Telegram: the model only gets a card sent; the user's tap on
that card is the only thing that runs it."""
from __future__ import annotations

import time

import pytest

import action_cards
import actions
import routine_store as store
import telegram_chat as tc
from test_telegram_chat import ME, env, msg, routine, run, say  # noqa: F401 (fixture)

_ids = iter(range(50_000, 60_000))


@pytest.fixture
def act(env, tmp_path, monkeypatch):  # noqa: F811
    actions.load(tmp_path / "a")
    monkeypatch.setattr(action_cards, "get_setting", lambda k, d=None: env.settings.get(k, d))
    env.card_ok = True

    async def call(method, payload, timeout=15.0):
        env.calls.append((method, payload))
        if method == "getUpdates":
            return env.updates
        if method == "sendMessage":
            if not env.card_ok:
                raise tc.notify.NotifyError("Could not reach Telegram.")
            return {"message_id": 700 + len(env.calls)}
        return True
    monkeypatch.setattr(tc.notify, "call", call)
    return env


def propose_call(action="set_routine", target="Morning AI News", **args):
    return {"content": "", "tool_calls": [{"function": {"name": "propose_action", "arguments": {
        "action": action, "target": target, "reason": "you asked", **args}}}]}


def cards(env):
    return [p for m, p in env.calls if m == "sendMessage" and "reply_markup" in p]


def edits(env):
    return [p for m, p in env.calls if m == "editMessageText"]


def tap(record, verb="approve", nonce=None, *, sender=ME, chat=ME, message_id=None):
    return {"update_id": next(_ids), "callback_query": {
        "id": "cb1", "from": {"id": sender},
        "message": {"message_id": record["message_id"] if message_id is None else message_id,
                    "chat": {"id": chat, "type": "private"}},
        "data": f"act:{record['id']}:{verb}:{record['nonce'] if nonce is None else nonce}"}}


def only_record():
    (record,) = actions._records.values()
    return actions.get(record["id"])


def test_model_proposal_sends_a_card_and_changes_nothing(act):
    r = routine()
    act.replies = [propose_call(state="paused"), "I've paused it for you."]
    out = say(act, msg("please pause my morning news routine"))
    (card,) = cards(act)
    assert card["text"].startswith("Proposed action · low risk\nPause the routine “Morning AI News”")
    labels = [b["text"] for b in card["reply_markup"]["inline_keyboard"][0]]
    assert labels == ["Approve", "Reject"]
    assert store.get(r["id"])["enabled"] is True                       # nothing ran
    tool_result = act.payloads[1]["messages"][-1]
    assert tool_result["role"] == "tool" and tool_result["content"].startswith("Proposed: Pause the routine")
    assert out == ["I've paused it for you." + tc._WAITS]               # the claim is corrected
    assert only_record()["message_id"] is not None and only_record()["status"] == "pending"


def test_refused_proposal_tells_the_model_and_sends_no_card(act):
    routine()
    act.replies = [propose_call(action="run_routine", target="Weather"), "There is no such routine."]
    say(act, msg("run my weather routine"))
    assert cards(act) == []
    assert act.payloads[1]["messages"][-1]["content"].startswith("Error: There is no routine called “Weather”")


def test_undelivered_card_voids_the_proposal(act):
    routine()
    act.card_ok = False
    act.replies = [propose_call(action="run_routine"), "Couldn't."]
    say(act, msg("run my morning news"))
    assert act.payloads[1]["messages"][-1]["content"] == (
        "Error: the approval card couldn't be sent, so nothing was proposed.")
    assert only_record()["status"] == "undelivered"


def test_reading_web_results_before_proposing_is_shown_on_the_card(act):
    routine()
    act.replies = [{"content": "", "tool_calls": [{"function": {"name": "web_search",
                                                                "arguments": {"query": "news"}}}]},
                   propose_call(action="run_routine"), "Proposed."]
    say(act, msg("check the news and run my morning routine"))
    assert "after reading web pages or your library" in cards(act)[0]["text"]


def test_approve_tap_runs_it_and_the_card_says_so(act):
    r = routine()
    act.replies = [propose_call(state="paused"), "Waiting for you."]
    say(act, msg("pause my morning news routine"))
    record = only_record()
    say(act, tap(record))
    assert ("answerCallbackQuery", {"callback_query_id": "cb1", "text": "Approved."}) in act.calls
    assert store.get(r["id"])["enabled"] is False
    last = edits(act)[-1]
    assert last["message_id"] == record["message_id"]
    assert last["text"].startswith("Done.\nPause the routine “Morning AI News”")
    assert last["reply_markup"] == {"inline_keyboard": [[]]}            # no buttons left
    say(act, tap(record))                                               # the same button again
    assert act.calls[-2] == ("answerCallbackQuery", {"callback_query_id": "cb1", "text": "Already done."})


def test_taps_from_anyone_else_or_another_message_do_nothing(act):
    r = routine()
    act.replies = [propose_call(state="paused"), "Waiting."]
    say(act, msg("pause my morning news routine"))
    record = only_record()
    n = len(act.calls)
    say(act, tap(record, sender=7), tap(record, chat=7))
    assert act.calls[n:] == [] and only_record()["status"] == "pending"   # silently ignored
    say(act, tap(record, message_id=1))
    assert act.calls[-1] == ("answerCallbackQuery", {"callback_query_id": "cb1",
                                                     "text": "That button is out of date."})
    say(act, tap(record, nonce="00000000"))
    assert act.calls[-2][1]["text"] == "That button is out of date."
    assert store.get(r["id"])["enabled"] is True


def test_reject_tap(act):
    r = routine()
    act.replies = [propose_call(state="paused"), "Waiting."]
    say(act, msg("pause my morning news routine"))
    say(act, tap(only_record(), "reject"))
    assert store.get(r["id"])["enabled"] is True
    assert edits(act)[-1]["text"].startswith("Rejected. Nothing was changed.")


def test_polling_asks_for_button_taps_and_expires_old_cards(act, monkeypatch):
    routine()
    act.replies = [propose_call(action="run_routine"), "Waiting."]
    say(act, msg("run my morning news"))
    later = time.time() + actions.TTL_S + 1
    monkeypatch.setattr(actions, "_now", lambda: later)
    act.updates = []
    run(tc.poll_once())
    (poll,) = [p for m, p in act.calls if m == "getUpdates"]
    assert poll["allowed_updates"] == ["message", "callback_query"]
    assert edits(act)[-1]["text"].startswith("Expired. Nothing was changed.")
    assert only_record()["status"] == "expired"


def test_approved_rerun_runs_in_the_background_and_the_card_reports_it(act, monkeypatch):
    import ask_apps
    import hub
    monkeypatch.setattr(hub, "load_fleet", lambda *a: [hub.App(
        id="mentat", title="Mentat", privacy="home", summary={"base": "http://host.docker.internal:8400"})])

    async def post(client, url, payload):
        return {"status": "ok", "saved": "replaced", "headline": "Calm week", "check": {"points": 4, "kept": 4}}
    monkeypatch.setattr(ask_apps, "_post_json", post)
    act.replies = [propose_call(action="rerun_summary", target="weekly"), "Waiting for you."]
    say(act, msg("redo mentat's weekly report"))
    say(act, tap(only_record()))
    texts = [e["text"].splitlines()[0] for e in edits(act)]
    assert texts == ["Approved. Running…", "Done."]
    assert "Weekly synthesis replaced: 4 of 4 points held up.\nCalm week" in edits(act)[-1]["text"]

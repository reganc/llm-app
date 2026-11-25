"""X search: failures are reported as what they are, never as "no results".

x-cli exits 0 and prints a GraphQL `errors` body when X treats the session as
logged out, so the old code showed an expired login as "No X results" and
status() said ready.
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest

import search
import x_search

LOGGED_OUT = {"data": {"search_by_raw_query": {"search_timeline": {}}},
              "errors": [{"message": "InvalidArgument: ProductDisabled: Closed gate stopped "
                                     "execution of the pipeline (TopTabSrpDenyLoggedOutUsers)"}]}


def tweets_body(n=1):
    entries = [{"content": {"itemContent": {
        "itemType": "TimelineTweet",
        "tweet_results": {"result": {
            "__typename": "Tweet", "rest_id": str(100 + i),
            "legacy": {"full_text": f"post {i} about nvidia", "created_at": "Wed"},
            "core": {"user_results": {"result": {"legacy": {},
                                                 "core": {"screen_name": "nv", "name": "NV"}}}},
        }}}}} for i in range(n)]
    return {"data": {"search_by_raw_query": {"search_timeline": {"timeline": {
        "instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}}}


class Proc:
    def __init__(self, rc=0, out=b"", err=b"", hang=False):
        self.returncode, self._o, self._e, self.hang = rc, out, err, hang
        self.killed = False

    async def communicate(self):
        if self.hang:
            import asyncio
            await asyncio.sleep(60)
        return self._o, self._e

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


@pytest.fixture
def xenv(monkeypatch, tmp_path):
    creds = tmp_path / "credentials.json"
    creds.write_text("{}")
    binary = tmp_path / "x-cli"
    binary.write_text("")
    monkeypatch.setattr(x_search, "CREDS_PATH", str(creds))
    monkeypatch.setattr(x_search, "CFG", dataclasses.replace(
        x_search.CFG, x_search_enabled=True, x_cli_path=str(binary)))
    x_search.reset_state()
    state = {"procs": [], "calls": 0, "creds": creds}

    async def fake_exec(*_a, **_k):
        state["calls"] += 1
        return state["procs"].pop(0)
    monkeypatch.setattr(x_search.asyncio, "create_subprocess_exec", fake_exec)
    yield state
    x_search.reset_state()


def run(coro):
    import asyncio
    return asyncio.run(coro)


@pytest.mark.parametrize("proc,kind", [
    (Proc(0, json.dumps(LOGGED_OUT).encode()), "auth"),
    (Proc(1, b"", b"Error: search: authentication failed (HTTP 401) - try: x-cli auth login"), "auth"),
    (Proc(1, b"", b"Error: rate limited - resets at 10:00:00"), "rate_limit"),
    (Proc(1, b"", b"Error: API error (HTTP 500): boom"), "error"),
    (Proc(0, b"not json"), "error"),
    (Proc(0, b'{"errors":[{"message":"something odd"}]}'), "error"),
])
def test_failures_are_typed(xenv, proc, kind):
    xenv["procs"] = [proc]
    with pytest.raises(x_search.XSearchError) as e:
        run(x_search.fetch_x("nvidia"))
    assert e.value.kind == kind


def test_timeout_kills_the_process(xenv, monkeypatch):
    monkeypatch.setattr(x_search, "TIMEOUT_S", 0.01)
    p = Proc(hang=True)
    xenv["procs"] = [p]
    with pytest.raises(x_search.XSearchError) as e:
        run(x_search.fetch_x("nvidia"))
    assert e.value.kind == "timeout" and p.killed


def test_empty_result_is_not_an_error(xenv):
    xenv["procs"] = [Proc(0, json.dumps(tweets_body(0)).encode())]
    assert run(x_search.fetch_x("nvidia")) == []
    assert x_search.status()["ready"] is True and x_search.status()["auth_ok"] is True


def test_partial_errors_with_posts_still_return_posts(xenv):
    xenv["procs"] = [Proc(0, json.dumps({**tweets_body(2), **{"errors": [{"message": "x"}]}}).encode())]
    assert len(run(x_search.fetch_x("nvidia"))) == 2


def test_status_not_ready_after_auth_failure_until_new_login(xenv):
    assert x_search.status()["ready"] is True and x_search.status()["auth_ok"] is None
    xenv["procs"] = [Proc(0, json.dumps(LOGGED_OUT).encode())]
    with pytest.raises(x_search.XSearchError):
        run(x_search.fetch_x("nvidia"))
    st = x_search.status()
    assert st["ready"] is False and st["auth_ok"] is False
    assert "x-cli auth login" in st["last_error"]
    # `x-cli auth login` rewrites the credentials file.
    os.utime(xenv["creds"], (2_000_000_000, 2_000_000_000))
    assert x_search.status()["ready"] is True


def test_other_errors_do_not_mark_the_login_bad(xenv):
    xenv["procs"] = [Proc(1, b"", b"rate limited")]
    with pytest.raises(x_search.XSearchError):
        run(x_search.fetch_x("nvidia"))
    st = x_search.status()
    assert st["auth_ok"] is not False and "rate" in st["last_error"].lower()


def test_background_search_x_never_raises_and_skips_known_dead_login(xenv):
    xenv["procs"] = [Proc(0, json.dumps(LOGGED_OUT).encode())]
    assert run(x_search.search_x("a")) == []
    assert run(x_search.search_x("b")) == []      # not called again
    assert xenv["calls"] == 1


def test_explicit_search_retries_even_when_login_was_bad(xenv):
    xenv["procs"] = [Proc(0, json.dumps(LOGGED_OUT).encode()),
                     Proc(0, json.dumps(tweets_body(1)).encode())]
    run(x_search.search_x("a"))
    out = run(x_search.search_x_and_ingest("b", store_memory=False))
    assert len(out["results"]) == 1 and not out.get("error")
    assert x_search.status()["ready"] is True


def test_ingest_reports_reason_and_uses_x_markers(xenv):
    xenv["procs"] = [Proc(0, json.dumps(LOGGED_OUT).encode()),
                     Proc(0, json.dumps(tweets_body(2)).encode())]
    bad = run(x_search.search_x_and_ingest("a", store_memory=False))
    assert "x-cli auth login" in bad["error"] and bad["context_text"] == ""
    good = run(x_search.search_x_and_ingest("a", store_memory=False))
    assert "[X1] @nv" in good["context_text"] and "[X2]" in good["context_text"]


def test_status_disabled(monkeypatch):
    monkeypatch.setattr(x_search, "CFG", dataclasses.replace(x_search.CFG, x_search_enabled=False))
    assert x_search.status()["ready"] is False


def test_x_command_falls_back_to_web_and_says_so(monkeypatch):
    import chat

    async def dead(query, **_):
        return {"query": query, "results": [], "context_text": "", "stored": 0,
                "error": "X login expired"}

    async def web(query, *, store_memory=True, include_x=True, **_):
        assert include_x is False
        return {"query": query, "results": [{"url": "https://e.com"}],
                "context_text": "[W1] t\nURL: https://e.com", "stored": 0, "source": "searxng"}
    monkeypatch.setattr(x_search, "search_x_and_ingest", dead)
    monkeypatch.setattr(search, "search_and_ingest", web)
    r = run(chat._x_or_web("q", store=False))
    assert "X login expired" in r["context_text"] and "[W1]" in r["context_text"]
    assert r["x_error"] == "X login expired"


def test_x_command_with_no_fallback_still_tells_the_model(monkeypatch):
    import chat

    async def dead(query, **_):
        return {"query": query, "results": [], "context_text": "", "stored": 0, "error": "X down"}

    async def noweb(query, **_):
        return {"query": query, "results": [], "context_text": "", "stored": 0, "error": "none"}
    monkeypatch.setattr(x_search, "search_x_and_ingest", dead)
    monkeypatch.setattr(search, "search_and_ingest", noweb)
    r = run(chat._x_or_web("q", store=False))
    assert "X down" in r["context_text"] and "Do not describe" in r["context_text"]


# ── the notice is added in code: the model ignored the instruction ───────────

FALLBACK = {"query": "q", "results": [], "context_text": "[W1] t", "x_error": "X login expired."}


def test_notice_is_prefixed_to_the_answer():
    import chat
    data = {"message": {"content": "People say things.", "thinking": "t"}, "done_reason": "stop"}
    out = chat._with_x_notice(data, FALLBACK)
    text = out["message"]["content"]
    assert text.index("X login expired.") < text.index("People say things.")
    assert "Web results" in text and out["message"]["thinking"] == "t"
    assert data["message"]["content"] == "People say things."   # input untouched


def test_no_notice_without_a_failure_or_an_answer_or_in_json_mode():
    import chat
    data = {"message": {"content": "hi"}}
    assert chat._with_x_notice(data, None) is data
    assert chat._with_x_notice(data, {"results": []}) is data
    assert chat._with_x_notice(data, FALLBACK, fmt="json") is data
    empty = {"message": {"content": "", "thinking": "t"}}   # reasoning fallback must still apply
    assert chat._with_x_notice(empty, FALLBACK) is empty


@pytest.mark.parametrize("fmt,noticed", [(None, True), ("json", False)])
def test_stream_emits_and_records_the_notice_first(monkeypatch, fmt, noticed):
    import chat
    import ollama as oll

    async def fake_stream(payload, on_token=None, **_):
        on_token("Hello")
        yield oll._sse("c", "m", delta="Hello", done=False)
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(oll, "stream_chat", fake_stream)
    buf: list[str] = []
    payload = {"model": "m", **({"format": fmt} if fmt else {})}

    async def go():
        return [c async for c in chat._stream_with_summary(
            payload, search_result=FALLBACK, command="x", message_id="i", on_token=buf.append)]
    out = run(go())
    body = [c for c in out if c.startswith("data: {")]
    assert ("X login expired." in body[0]) is noticed
    assert ("X login expired." in "".join(buf)) is noticed
    assert "".join(buf).endswith("Hello")


def test_stream_records_nothing_if_the_model_said_nothing(monkeypatch):
    import chat
    import ollama as oll

    async def fake_stream(payload, on_token=None, **_):
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(oll, "stream_chat", fake_stream)
    buf: list[str] = []

    async def go():
        return [c async for c in chat._stream_with_summary(
            {"model": "m"}, search_result=FALLBACK, command="x", message_id="i", on_token=buf.append)]
    run(go())
    assert buf == []


@pytest.mark.parametrize("enabled,posts", [(False, None), (True, 0)])
def test_notice_reads_as_sentences(xenv, monkeypatch, enabled, posts):
    import chat
    monkeypatch.setattr(x_search, "CFG",
                        dataclasses.replace(x_search.CFG, x_search_enabled=enabled))
    if posts is not None:
        xenv["procs"] = [Proc(0, json.dumps(tweets_body(posts)).encode())]
    error = run(x_search.search_x_and_ingest("q", store_memory=False))["error"]
    assert error.endswith(".")
    notice = chat._x_notice({"x_error": error})
    assert f"{error} " in notice and "unavailable" not in notice


def test_notice_punctuates_raw_errors_once():
    import chat
    assert "failed: HTTP 500. Web" in chat._x_notice({"x_error": "X search failed: HTTP 500"})
    assert ".." not in chat._x_notice({"x_error": "Done."})

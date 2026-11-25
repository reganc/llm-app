"""Images through /v1/chat/completions and conversations."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat
import conversations
from test_agent_endpoint import AUTH, URL, env  # noqa: F401  (fixture)
from test_images import png, url


@pytest.fixture
def client(env):  # noqa: F811
    app = FastAPI()
    app.include_router(chat.router)
    return TestClient(app)


def parts(text, *urls):
    return [{"type": "text", "text": text}, *({"type": "image_url", "image_url": {"url": u}} for u in urls)]


def last_sent(env):  # noqa: F811
    return (env.chat_payloads or env.stream_payloads)[-1]["messages"]


@pytest.mark.parametrize("extra", [{"agent": True}, {"agent": True, "stream": True}, {"agent": False, "search": False},
                                   {"raw": True}])
def test_image_reaches_the_model_on_the_question(client, env, extra):  # noqa: F811
    body = {"model": "default", "store": False, **extra, "messages": [{"role": "user", "content": parts("what is this?", url(png()))}]}
    assert client.post(URL, json=body, headers=AUTH).status_code == 200
    m = last_sent(env)
    assert m[-1]["content"] == "what is this?" and len(m[-1]["images"]) == 1
    assert "isn't shown" in m[0]["content"]


def test_follow_up_keeps_the_latest_image_on_the_new_question(client, env):  # noqa: F811
    msgs = [{"role": "user", "content": parts("read this", url(png()))},
            {"role": "assistant", "content": "It says hi."}, {"role": "user", "content": "and the colour?"}]
    client.post(URL, json={"model": "default", "agent": True, "store": False, "messages": msgs}, headers=AUTH)
    m = last_sent(env)
    assert m[-1]["content"] == "and the colour?" and m[-1]["images"] and "images" not in m[1]


def test_image_only_message(client, env):  # noqa: F811
    body = {"model": "default", "agent": True, "store": False, "messages": [{"role": "user", "content": parts("", url(png()))}]}
    client.post(URL, json=body, headers=AUTH)
    assert last_sent(env)[-1]["content"] == "Describe this image."


@pytest.mark.parametrize("bad,word", [("https://example.com/a.png", "not links"), (url(b"nope"), "could not be read")])
def test_bad_images_are_refused_before_the_model(client, env, bad, word):  # noqa: F811
    body = {"model": "default", "messages": [{"role": "user", "content": parts("x", bad)}]}
    r = client.post(URL, json=body, headers=AUTH)
    assert r.status_code == 422 and word in r.json()["detail"] and env.chat_payloads == []


def test_reasoning_mode_says_no_to_images(client):
    body = {"model": "default", "messages": [{"role": "user", "content": parts("x", url(png()))}]}
    r = client.post("/v1/chat/reasoning", json=body, headers=AUTH)
    assert r.status_code == 422 and "reasoning" in r.json()["detail"]


def test_conversation_turn_with_an_image_is_not_stored(client, env, tmp_path, monkeypatch):  # noqa: F811
    async def nothing():
        return None
    monkeypatch.setattr(chat.mem, "store_conversation_turn", lambda *a, **k: nothing())
    conv = conversations.create("t", None, None)
    r = client.post(f"/v1/conversations/{conv['id']}/messages", headers=AUTH,
                    json={"content": "what is it?", "images": [url(png())]})
    assert r.status_code == 200
    stored = conversations.get(conv["id"])["messages"]
    assert stored[-2]["content"] == "what is it? (with an image)" and "data:image" not in str(stored)
    assert last_sent(env)[-1]["images"]
    bad = client.post(f"/v1/conversations/{conv['id']}/messages", headers=AUTH,
                      json={"content": "x", "images": ["https://x.example/a.png"]})
    assert bad.status_code == 422
    conversations.delete(conv["id"])


@pytest.mark.parametrize("text,searched", [("What's the total?", False), ("what is this plant? search the web", True),
                                          ("Is this the latest model?", True), ("Describe it", False)])
def test_with_an_image_the_web_is_searched_only_when_asked(client, env, text, searched):  # noqa: F811
    body = {"model": "default", "agent": True, "store": False, "messages": [{"role": "user", "content": parts(text, url(png()))}]}
    client.post(URL, json=body, headers=AUTH)
    offered = [t["function"]["name"] for t in last_payload(env).get("tools", [])]
    assert ("web_search" in offered) is searched


def last_payload(env):  # noqa: F811
    return env.chat_payloads[-1]


def test_with_an_image_the_library_is_searched_only_when_asked(client, env):  # noqa: F811
    def offered(text):
        body = {"model": "default", "agent": True, "store": False, "messages": [{"role": "user", "content": parts(text, url(png()))}]}
        client.post(URL, json=body, headers=AUTH)
        return [t["function"]["name"] for t in env.chat_payloads[-1].get("tools", [])]
    assert "library_search" not in offered("What is the total?")
    before = len(env.retrieve_kwargs)
    offered("Is this in my library?")          # library mode: the library is read up front
    assert len(env.retrieve_kwargs) > before

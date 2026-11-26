"""Telegram voice notes: transcribed locally, shown back, answered like a typed
question; never a command, never logged, one at a time, long notes refused."""
from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

import routine_store as store
import telegram_chat as tc
import voice
from test_telegram_chat import ME, env, msg, routine, run, say  # noqa: F401 (fixture)

SECRET = "my account number is 4417"


def note(duration=4, file_id="v1", **kw):
    m = msg(**kw)
    m["message"].pop("text")
    m["message"]["voice"] = {"file_id": file_id, "duration": duration, "mime_type": "audio/ogg"}
    return m


@pytest.fixture
def heard(env, monkeypatch):  # noqa: F811
    env.heard, env.downloads = "What is the weather in Austin?", []

    async def download(file_id):
        env.downloads.append(file_id)
        return b"OggS-fake-audio"

    def transcribe_sync(audio):
        assert audio == b"OggS-fake-audio"
        if isinstance(env.heard, Exception):
            raise env.heard
        return env.heard
    monkeypatch.setattr(tc.notify, "download", download)
    monkeypatch.setattr(voice, "_transcribe_sync", transcribe_sync)
    monkeypatch.setattr(voice, "CFG", dataclasses.replace(voice.CFG, voice_enabled=True))
    return env


def test_voice_note_is_shown_back_then_answered(heard):
    heard.replies = ["Sunny and 75."]
    out = say(heard, note())
    assert out == ["Heard: “What is the weather in Austin?”", "Sunny and 75."]
    assert heard.downloads == ["v1"]
    assert heard.payloads[-1]["messages"][-1]["content"] == "What is the weather in Austin?"


def test_spoken_commands_are_questions_not_commands(heard):
    r = routine()
    heard.heard = "/pause 1"
    heard.replies = ["I can propose pausing it."]
    out = say(heard, note())
    assert out[0] == "Heard: “/pause 1”" and store.get(r["id"])["enabled"] is True
    assert heard.payloads, "it went to the model as a question"


def test_spoken_requests_to_act_get_the_fixed_reply(heard):
    heard.heard = "Please delete my emails"
    assert say(heard, note()) == ["Heard: “Please delete my emails”", tc._REFUSE]
    assert heard.payloads == []


def test_long_notes_are_refused_before_download(heard):
    assert say(heard, note(duration=voice.MAX_SECONDS + 1)) == [
        "That voice note is too long; keep it under 5 minutes."]
    assert heard.downloads == []


def test_silence_and_failures_say_so_without_the_audio_or_text(heard, caplog):
    heard.heard = ""
    assert say(heard, note()) == ["I didn't hear any speech in that voice note."]
    heard.heard = RuntimeError(SECRET)
    with caplog.at_level(logging.DEBUG):
        assert say(heard, note()) == ["I couldn't make out that voice note."]
    assert SECRET not in caplog.text


def test_transcripts_are_never_logged(heard, caplog):
    heard.heard = SECRET
    heard.replies = ["Noted."]
    with caplog.at_level(logging.DEBUG):
        say(heard, note())
    assert SECRET not in caplog.text


def test_switched_off(heard, monkeypatch):
    monkeypatch.setattr(voice, "CFG", dataclasses.replace(voice.CFG, voice_enabled=False))
    assert say(heard, note()) == ["Voice notes are switched off (VOICE_ENABLED=false)."]


def test_one_transcription_at_a_time(monkeypatch):
    running, peak = [0], [0]

    def slow(audio):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        import time
        time.sleep(0.05)
        running[0] -= 1
        return "hi"
    monkeypatch.setattr(voice, "_transcribe_sync", slow)
    monkeypatch.setattr(voice, "CFG", dataclasses.replace(voice.CFG, voice_enabled=True))

    async def go():
        voice._one_at_a_time = asyncio.Semaphore(1)
        return await asyncio.gather(*(voice.transcribe(b"x") for _ in range(3)))
    assert run(go()) == ["hi", "hi", "hi"] and peak[0] == 1


def test_others_voice_notes_are_ignored(heard):
    m = note()
    m["message"]["from"] = {"id": 7}
    assert say(heard, m) == [] and heard.downloads == []

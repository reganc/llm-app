"""The chat box offers a routine only for text that is clearly a schedule."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "api" / "static" / "routine-phrases.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def ask(text: str):
    script = (f"import {{ routineRequest }} from {json.dumps(JS.as_uri())};"
              f"process.stdout.write(JSON.stringify(routineRequest({json.dumps(text)})));")
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("text", [
    "every weekday at 7, summarize AI news", "Every day at 6:30pm give me the weather",
    "each morning, list new papers", "daily at 18:30 check nvidia", "every 6 hours, check the status page",
    "every monday and thursday at 9am list new papers", "summarize AI news every weekday at 7am",
    "weekly on friday at 4pm, summarize the week", "every Tue at noon do x",
])
def test_clear_schedules(text):
    assert ask(text) == {"text": text, "explicit": False}


@pytest.mark.parametrize("text", [
    "Every day I feel tired, why?", "Why does it rain every day?", "what happened every monday in history",
    "Each of these is wrong, explain", "Everyone says that, true?", "How do I run a cron every day at 7am?",
    "Explain what happens every day at 7am in a bakery?", "daily news", "every", "hello",
    "is the market open every weekday at 9am?",
])
def test_ordinary_messages(text):
    assert ask(text) is None


def test_command():
    assert ask("/routine summarize AI news") == {"text": "summarize AI news", "explicit": True}
    assert ask("/ROUTINE  x ") == {"text": "x", "explicit": True}
    assert ask("/routine") == {"text": "", "explicit": True}
    assert ask("/routines x") is None

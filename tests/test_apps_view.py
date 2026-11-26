"""The SPA's Apps panel: which colour and words each /v1/hub entry gets.

Runs the real module under node, like test_render_md.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APPS = Path(__file__).resolve().parent.parent / "api" / "static" / "apps.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def state(app: dict) -> dict:
    entry = {"up": None, "level": "info", "items": [], "problem": None, "withheld": 0, **app}
    script = (f"import {{ appState }} from {json.dumps(APPS.as_uri())};"
              f"process.stdout.write(JSON.stringify(appState({json.dumps(entry)})));")
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_down_beats_everything():
    assert state({"up": False, "level": "alert"}) == {"cls": "alert", "text": "Down"}


def test_alert_and_warning_items():
    assert state({"up": True, "level": "alert", "items": [{"title": "x", "severity": "alert"}]})["cls"] == "alert"
    assert state({"up": True, "level": "warn", "items": [{"title": "x", "severity": "warn"}]}) == \
        {"cls": "warn", "text": "Warning"}


def test_unreadable_card_shows_why():
    assert state({"level": "warn", "problem": "Its card is out of date."}) == \
        {"cls": "warn", "text": "Its card is out of date."}


def test_healthy_and_unchecked():
    assert state({"up": True}) == {"cls": "ok", "text": "OK"}
    assert state({"items": [{"title": "3 deadlines", "severity": "info"}]})["cls"] == "ok"
    assert state({}) == {"cls": "none", "text": "No health check"}

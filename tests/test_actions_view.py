"""The SPA's Actions panel: words and colour per status, and where a proposal came from (needs node)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ACTIONS = Path(__file__).resolve().parent.parent / "api" / "static" / "actions.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def js(fn: str, arg: dict):
    script = (f"import {{ {fn} }} from {json.dumps(ACTIONS.as_uri())};"
              f"process.stdout.write(JSON.stringify({fn}({json.dumps(arg)})));")
    return json.loads(subprocess.run(["node", "--input-type=module", "-e", script],
                                     capture_output=True, text=True, check=True).stdout)


def test_open_proposals_wait_for_you():
    assert js("actionState", {"status": "pending"}) == {"cls": "warn", "text": "Waiting for you", "open": True}
    assert js("actionState", {"status": "confirming"})["text"] == "Waiting for your second tap"


def test_finished_states():
    assert js("actionState", {"status": "done"}) == {"cls": "ok", "text": "Done", "open": False}
    assert js("actionState", {"status": "failed"})["cls"] == "alert"
    assert js("actionState", {"status": "expired"}) == {"cls": "none", "text": "Expired", "open": False}


def test_source_text():
    assert js("sourceText", {"source": "mcp", "client": "claude-code"}) == \
        "MCP client “claude-code” (what it read before is unknown)"
    assert js("sourceText", {"source": "telegram", "after_reading": True}) == \
        "Telegram chat, after reading web pages or your library"

"""The SPA's markdown renderer must never emit HTML that came from the text.

Answers quote web pages, and the API key lives in localStorage, so unescaped
HTML would let a page run script with it. Runs the real module under node.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MD = Path(__file__).resolve().parent.parent / "api" / "static" / "md.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def render(text: str) -> str:
    script = (f"import {{ renderMd }} from {json.dumps(MD.as_uri())};"
              f"process.stdout.write(renderMd({json.dumps(text)}));")
    return subprocess.run(["node", "--input-type=module", "-e", script],
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("text", [
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    "# <svg onload=alert(1)>",
    "- <b onclick=x>item</b>",
    "> <iframe src=//evil></iframe>",
    "**<u>x</u>**",
    "`<img src=x>`",
    "```html\n<img src=x>\n```",
    "```\n<img src=x>\n```",
    "\u0000CODE0\u0000<img>",
    "[<img src=x>](https://a.example)",
])
def test_html_is_escaped(text):
    out = render(text)
    for tag in ("<img", "<script", "<svg", "<b ", "<iframe", "<u>"):
        assert tag not in out


@pytest.mark.parametrize("text", [
    '[x](https://a.example/"onmouseover="alert(1))',
    'https://a.example/"onmouseover="alert(1)',
    "[x](javascript:alert(1))",
])
def test_links_cannot_add_attributes_or_schemes(text):
    out = render(text)
    assert '"onmouseover' not in out and 'href="javascript' not in out


def test_markdown_still_renders():
    out = render("# T\n\n**b** and *i* and `c<d`\n\n> quote\n\n- one\n- two\n\n1. a\n\n"
                 "[l](https://a.example/?a=1&b=2) https://b.example/x\n\n```py\nx < 1\n```")
    for want in ("<h1>T</h1>", "<strong>b</strong>", "<em>i</em>", "<code>c&lt;d</code>",
                 "<blockquote>quote</blockquote>", "<ul><li>one</li>", "<ol><li>a</li>",
                 'href="https://a.example/?a=1&amp;b=2"', 'href="https://b.example/x"',
                 "x &lt; 1"):
        assert want in out
    assert "&amp;lt;" not in out


def test_dollar_and_empty():
    assert render("") == ""
    assert "$1 and $&amp;" in render("```\n$1 and $&\n```")


def test_code_placeholder_out_of_range_is_dropped():
    assert "undefined" not in render("\u0000CODE7\u0000")

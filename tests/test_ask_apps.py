"""Asking another app: which apps, what is sent, how answers and failures read."""
from __future__ import annotations

import asyncio
import json

import httpx

import ask_apps
import hub

BASE = "http://host.docker.internal:8400"
MENTAT = hub.App(id="mentat", title="Mentat", privacy="home", ask="mentat", summary={"base": BASE})
NEXUS = hub.App(id="nexus", title="Nexus", privacy="home", ask="nexus",
                summary={"base": "http://host.docker.internal:9000"})


def test_only_home_apps_with_a_known_kind_and_an_address(tmp_path):
    f = tmp_path / "fleet.yaml"
    f.write_text("""
apps:
  - {id: mentat, privacy: home, ask: mentat, summary: {base: "http://host.docker.internal:8400"}}
  - {id: tax-planner, privacy: private, ask: mentat, summary: {base: "http://host.docker.internal:8475"}}
  - {id: odd, privacy: home, ask: shell, summary: {base: "http://host.docker.internal:1"}}
  - {id: far, privacy: home, ask: nexus, summary: {base: "http://10.0.0.5:9000"}}
  - {id: nobase, privacy: home, ask: nexus}
""")
    assert list(ask_apps.askable(hub.load_fleet(f))) == ["mentat"]


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2)


def _ask(app, handler):
    async def go():
        async with _client(handler) as c:
            return await ask_apps.ask(app, "What links the weather and markets?", c)
    return asyncio.run(go())


def test_mentat_answer_with_sources_and_degraded_note():
    seen = []

    def handler(req):
        seen.append((req.method, str(req.url), json.loads(req.content)))
        return httpx.Response(200, json={
            "answer_md": "## Short version\n**Rain** moves *nothing* [1].", "degraded": True,
            "sources": [{"n": 1, "label": "nexus episode 12", "text": "…"}, {"n": 2, "label": ""}]})
    text = _ask(MENTAT, handler)
    assert seen == [("POST", f"{BASE}/ask", {"question": "What links the weather and markets?"})]
    assert text.startswith("Mentat:\n\nShort version\nRain moves *nothing* [1].")
    assert "local model" in text and "[1] nexus episode 12" in text and "[2]" not in text


def test_nexus_answer_with_confidence_and_caveats():
    text = _ask(NEXUS, lambda r: httpx.Response(200, json={
        "answer": "Weak link.", "confidence": 0.42, "answer_source": "llm", "caveats": ["Few episodes"]}))
    assert text == "Nexus:\n\nWeak link.\n\nConfidence 42%, from the local model.\nCaveats: Few episodes"


def test_failures_are_sentences():
    assert _ask(NEXUS, lambda r: httpx.Response(500)) == "Nexus couldn't answer. It answered with an error (500)."
    assert _ask(NEXUS, lambda r: httpx.Response(200, json={"answer": ""})) == \
        "Nexus couldn't answer. nexus sent an empty answer."
    assert _ask(NEXUS, lambda r: httpx.Response(200, json=[1])) == "Nexus couldn't answer. Its answer couldn't be read."
    assert _ask(NEXUS, lambda r: httpx.Response(200, content=b"x" * 1_000_001)) == \
        "Nexus couldn't answer. Its answer was too large."

    def refuse(r):
        raise httpx.ConnectError("refused")
    assert _ask(NEXUS, refuse) == "Nexus couldn't be reached."

    def slow(r):
        raise httpx.ReadTimeout("slow")
    assert _ask(NEXUS, slow) == "Nexus took too long to answer."


def test_redirects_are_not_followed():
    text = _ask(NEXUS, lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/"}))
    assert text == "Nexus couldn't answer. It answered with an error (302)."


def test_an_address_outside_the_rule_is_never_called():
    app = hub.App(id="x", title="X", privacy="home", ask="nexus", summary={"base": "http://example.com"})

    def handler(r):
        raise AssertionError("called")
    assert _ask(app, handler) == "X couldn't answer. That app's address isn't allowed."


# ── /research: mentat's research pipeline ────────────────────────────────────

REPORT = {
    "id": "16b1e04ae0d1", "query": "speculative decoding",
    "body_md": "# Speculative Decoding\n\n**Drafts** are verified in parallel.",
    "claims": [{"id": "c1", "sourced": True, "cite_available": True},
               {"id": "c2", "sourced": True, "cite_available": True},
               {"id": "c3", "sourced": False, "cite_available": False}],
    "degraded": True, "revision_count": 0,
    "review": {"passed": True, "provenance_completeness": 1.0, "independent_reviewer": False},
}
EVIDENCE = {"c1": {"source_title": "Hybrid Spark", "source_url": "http://arxiv.org/abs/2601.1"},
            "c2": {"source_title": "Hybrid Spark", "source_url": "http://arxiv.org/abs/2601.1"}}


def _research(handler, topic="speculative decoding"):
    async def go():
        async with _client(handler) as c:
            return await ask_apps.research(MENTAT, topic, c)
    return asyncio.run(go())


def test_research_relays_the_report_its_check_and_distinct_sources():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.method == "POST":
            assert json.loads(req.content) == {"query": "speculative decoding", "max_sources": 5}
            return httpx.Response(200, json=REPORT)
        return httpx.Response(200, json=EVIDENCE[req.url.path.rsplit("/", 1)[1]])
    text = _research(handler)
    assert seen == [("POST", "/research"), ("GET", "/research/16b1e04ae0d1/cite/c1"),
                    ("GET", "/research/16b1e04ae0d1/cite/c2")]
    assert text.startswith("Mentat research: speculative decoding\n\nSpeculative Decoding\n\nDrafts are verified")
    assert ("Checked: 3 claims, 2 cite a source it read, 1 marked as inference; 100% of the citations "
            "check out; reviewed by the same local model that wrote it; written by mentat's local model.") in text
    assert "Sources:\n[1] Hybrid Spark http://arxiv.org/abs/2601.1" in text and "[2]" not in text
    assert text.endswith("Report 16b1e04ae0d1 is kept in Mentat.")


def test_a_failed_review_is_said_plainly():
    report = {**REPORT, "claims": [], "degraded": False,
              "review": {"passed": False, "provenance_completeness": 0.5, "independent_reviewer": True}}
    text = _research(lambda r: httpx.Response(200, json=report))
    assert "Checked: 0 claims, 0 cite a source it read; 50% of the citations check out; it did NOT pass its review" in text


def test_research_failures_are_sentences():
    assert _research(lambda r: httpx.Response(500)) == "Mentat couldn't research that. It answered with an error (500)."
    assert _research(lambda r: httpx.Response(200, json={**REPORT, "body_md": ""})) == \
        "Mentat couldn't research that. It wrote an empty report."

    def slow(r):
        raise httpx.ReadTimeout("slow")
    assert "took too long; the report may still appear" in _research(slow)


def test_a_failing_cite_lookup_only_drops_that_source():
    def handler(req):
        if req.method == "POST":
            return httpx.Response(200, json=REPORT)
        return httpx.Response(404) if req.url.path.endswith("c1") else httpx.Response(200, json=EVIDENCE["c2"])
    assert "[1] Hybrid Spark" in _research(handler)

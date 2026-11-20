"""Stale citation markers inside stored library text.

Past assistant answers are saved to the library with their citations intact
("...reverse its vibration [W3][L2]."). Rendered back into a prompt, those
markers point at another turn's sources. The model copied "[W3]" into a
library-only answer, and `Trace.renumber` counted the stray marker as a shown
web result, so citation repair accepted it. `build_context_block` removes them;
the block's own [L#] headers are the only markers that should reach the model.
"""
from __future__ import annotations

import agent
import memory as mem


def chunk(text, **kw):
    return {"score": 0.9, "text": text, "title": "Doc: kybalion.pdf",
            "identifier": "c0f80e9dc9de", "source_type": "conversation", **kw}


def test_markers_in_chunk_text_are_removed():
    block = mem.build_context_block(
        [chunk("One can reverse its vibration [W3][L2]. See also [W1, X2] and [ A4 ].")])
    assert "[W3]" not in block and "[L2]" not in block
    assert "[W1, X2]" not in block and "[ A4 ]" not in block
    assert "One can reverse its vibration." in block
    assert block.count("[L1]") == 1  # the block's own header marker survives


def test_markers_in_refined_summary_are_removed():
    block = mem.build_context_block([chunk(
        "Body text.",
        summary={"tl_dr": "Rhythm is a pendulum [W2].",
                 "key_claims": ["Polarity reverses states [L5]"]})])
    assert "[W2]" not in block and "[L5]" not in block
    assert "Rhythm is a pendulum." in block


def test_ordinary_brackets_are_kept():
    text = "Paper size [A4 sheet], footnote [1], array a[W] and [x] stay."
    assert text in mem.build_context_block([chunk(text)])


def test_renumber_counts_only_the_blocks_own_markers():
    trace = agent.Trace()
    trace.renumber(mem.build_context_block(
        [chunk("Claim [W3][L2]."), chunk("Other claim [X7].", identifier="b")]))
    assert trace.markers == {"W": 0, "X": 0, "A": 0, "L": 2}
    # So a copied stale marker is now caught as invalid.
    assert agent.invalid_citations("Claim [W3][L1].", trace.markers) == ["W3"]

"""Only points their cited sources back are kept."""
from __future__ import annotations

import time

import pytest

import routine_check as rc

ISL = "ISL points table 2026/27 Indian Super League standings Bengaluru FC Mumbai City 2-0 November 20"
NV = "Nvidia shares fell 4% on Tuesday as investors sold AI chip stocks, wiping out $200 billion"
TEXTS = {"W1": ISL, "W2": NV}


def test_the_bad_run_that_prompted_this():
    out = rc.check(
        "*   **Hacker News** is buzzing with discussions on **GitHub's OpenAI repository**.\n"
        "*   **Nvidia's legacy hardware** is experiencing a revival.\n"
        "*   **Microsoft has issued a \"Code Red\" declaration**.\n"
        "*   **The Indian Super League (ISL)** standings show **Bengaluru FC** beat Mumbai City on November 20 [W1].\n"
        "*   **Global bond markets** are favoring \"safe\" US paper over stocks, with yields dropping as "
        "risk-fleeing capital moves into government debt [W1].", TEXTS)
    assert (out.kept, out.total, out.enough) == (1, 5, False)
    assert "Bengaluru" in out.content and "bond" not in out.content


def test_backed_points_are_unchanged():
    text = "1. Nvidia shares fell 4% as investors sold chip stocks [W2].\n2. Bengaluru lead the Indian standings [W1]."
    out = rc.check(text, TEXTS)
    assert out.content == text and out.removed == [] and out.enough


def test_right_marker_is_required():
    out = rc.check("- Nvidia shares fell 4% as investors sold chip stocks [W1].", TEXTS)
    assert out.kept == 0 and out.content == "" and out.removed[0]["reason"].startswith("could not be matched")


def test_any_of_several_cited_sources_may_back_a_point():
    assert rc.check("- Nvidia shares fell; investors sold [W1, W2].", TEXTS).kept == 1


@pytest.mark.parametrize("marker", ["W9", "L1"])
def test_marker_without_text_backs_nothing(marker):
    assert rc.check(f"- Nvidia shares fell [{marker}].", TEXTS).kept == 0


def test_sub_items_and_wrapped_lines_belong_to_their_point():
    out = rc.check("1. **Nvidia**\n   - Nvidia shares fell 4% [W2]\n   - investors sold chip stocks\n"
                   "2. Unrelated claim\n   continues here", TEXTS)
    assert (out.kept, out.total) == (1, 2)
    assert "investors sold" in out.content and "continues" not in out.content


def test_headings_and_intros_stay_only_with_a_kept_point():
    out = rc.check("Here are the stories:\n\n## Chips\n- Nvidia shares fell 4% [W2].\n\n## Bonds\n- Yields dropped.", TEXTS)
    assert "Chips" in out.content and "Bonds" not in out.content
    assert (out.kept, out.total) == (1, 2)


def test_prose_paragraphs_are_checked_too():
    out = rc.check("Nvidia shares fell 4% as investors sold [W2].\n\nBond yields dropped everywhere.", TEXTS)
    assert out.content.startswith("Nvidia") and out.total == 2 and out.enough


def test_half_is_enough_and_less_is_not():
    good, bad = "- Nvidia shares fell 4% [W2].", "- Yields dropped."
    assert rc.check(f"{good}\n{bad}", TEXTS).enough
    assert not rc.check(f"{good}\n{bad}\n{bad}", TEXTS).enough


def test_numbers_are_compared_without_separators():
    assert rc.check("- Price: 1,200 [W1].", {"W1": "price 1200"}).kept == 1


def test_other_languages_are_compared():
    assert rc.check("- Les actions de Nvidia chutent [W1].", {"W1": "Les actions de Nvidia chutent mardi"}).kept == 1


@pytest.mark.parametrize("text", ["", "\n\n", "NOTHING", "#", "- ", "***", "[W1]", "| a | b |\n|---|---|\n| x [W2] | y |"])
def test_odd_content_never_raises(text):
    rc.check(text, TEXTS)
    rc.check(text, {})


def test_removed_list_is_bounded():
    out = rc.check("\n".join(f"- claim {i} " + "word " * 300 for i in range(60)), TEXTS)
    assert len(out.removed) <= 20 and all(len(r["text"]) <= 400 for r in out.removed)
    assert out.total == 60


def test_large_answers_are_fast():
    start = time.perf_counter()
    rc.check("\n".join(f"- Nvidia shares fell {i} [W2]" for i in range(400)), {"W2": NV * 40})
    assert time.perf_counter() - start < 2

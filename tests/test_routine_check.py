"""Only points their cited sources back are kept."""
from __future__ import annotations

import time

import pytest

import routine_check as rc
import search

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


# ── quotes ───────────────────────────────────────────────────────────────────

TOY = {"W1": "PIRG tested AI-powered toys and found that many come with flimsy safety guardrails. "
             "“The rush to sell toys to young children puts profits before kids’ safety,” the senator said."}


def test_exact_quotes_are_kept_whatever_the_quote_style():
    for text in ('- Many AI toys come with "flimsy safety guardrails" [W1].',
                 "- The senator said “puts profits before kids' safety” about toys [W1].",
                 '- The senator said "The rush to sell toys to **young children** puts profits before kids’ safety," [W1].',
                 '- The senator said "The rush to sell toys … puts profits before kids’ safety." [W1]'):
        assert rc.check(text, TOY).content == text


def test_a_short_altered_quote_loses_its_marks():
    out = rc.check('- Many AI toys come with "flimsy guardrails" [W1].', TOY)
    assert out.content == "- Many AI toys come with flimsy guardrails [W1]." and out.removed == []


def test_a_long_invented_quote_removes_the_point():
    out = rc.check('- The senator said "toys are a danger to all young children everywhere" [W1].', TOY)
    assert out.kept == 0 and "quotes words" in out.removed[0]["reason"]


def test_a_quote_must_be_in_the_source_it_cites():
    texts = {**TOY, "W2": "The senator said toys puts profits before kids safety? No: young children rush to sell."}
    out = rc.check('- The senator said "The rush to sell toys to young children" [W2].', texts)
    assert out.kept == 0


def test_quotes_match_whole_words_and_survive_odd_marks():
    assert rc.check('- AI toys have "flimsy safe" parts, many come with guardrails [W1].', TOY).content.count('"') == 0
    for text in ('- AI toys "flimsy [W1].', '- AI toys "" [W1].', '- AI toys "..." [W1].', '- toys "(((" [W1]'):
        rc.check(text, TOY)


# ── previews ─────────────────────────────────────────────────────────────────

def test_points_resting_only_on_previews_are_labelled():
    text = "- Nvidia shares fell 4% [W2].\n- Bengaluru lead the Indian standings [W1]."
    out = rc.check(text, TEXTS, {"W2"})
    assert out.content == ("- Nvidia shares fell 4% [W2]. (search preview only)\n"
                           "- Bengaluru lead the Indian standings [W1].")


def test_a_point_with_one_readable_source_is_not_labelled():
    assert "preview" not in rc.check("- Nvidia shares fell; investors sold [W1, W2].", TEXTS, {"W2"}).content


# ── news: one page's share, and invented links to the topic ─────────────────

PAGES = {"W1": "a.example/1", "W2": "b.example/2", "W3": "a.example/1"}


def test_a_page_backs_at_most_two_points_even_under_two_markers():
    texts = {"W1": NV, "W3": NV, "W2": ISL}
    text = ("- Nvidia shares fell 4% [W1].\n- Nvidia investors sold chip stocks [W1].\n"
            "- Nvidia lost $200 billion [W3].\n- Bengaluru lead the Indian standings [W2].")
    out = rc.check(text, texts, pages=PAGES)
    assert "$200 billion" not in out.content and "Bengaluru" in out.content
    assert (out.kept, out.total, out.extra, out.enough) == (3, 3, 1, True)
    assert out.removed[0]["reason"] == "more than 2 points from one page"
    assert rc.check(text, texts).kept == 4                        # not for other routines


def test_a_point_with_another_page_behind_it_is_not_extra():
    texts = {"W1": NV, "W2": NV}
    text = "- Nvidia fell [W1].\n- Nvidia shares fell [W1].\n- Nvidia stocks fell [W1][W2]."
    assert rc.check(text, texts, pages=PAGES).extra == 0


CBS = """Iran War Updates: Tehran denounces U.S. for using AI in strikes
- Iran's U.N. ambassador criticized the U.S. military's use of artificial intelligence systems in the war, claiming American forces leveraged the technology to carry out some 2,000 attacks over four days.
- The Iran-backed Houthi group claimed an attack on the financial district of Saudi Arabia's capital on Wednesday, as well on a sprawling oil refinery in the country's east.
- Israeli Defense Minister Israel Katz said Iran is developing chemical weapons and vowed to stop Tehran from obtaining them.
Iran has repeatedly denied it is developing chemical weapons.
Britain's air force is moving planes into hangars and hardened shelters overnight to protect them from drone threats.
The Royal Air Force is planning to install anti-drone nets around warplanes that remain exposed.
Iran's First Vice President said Tehran is not concerned about the naval blockade of Iranian ports.
Israel was trying to drag the United States into another war, the Iranian foreign minister said."""
AI_TOPIC = search.topic_pattern("AI news")


def _ai(text, source=CBS):
    return rc.check(text, {"W1": source}, topic=AI_TOPIC)


@pytest.mark.parametrize("text", [
    "* **The Iran-backed Houthi group claimed an attack** on Saudi Arabia's financial district and a sprawling "
    "oil refinery, adding to the regional tension fueled by AI-driven precision strikes [W1].",
    "* **Britain's air force is moving planes into shelters** and planning anti-drone nets to protect warplanes "
    "from drone threats, a defensive move against the AI-enabled aerial assaults [W1].",
])
def test_links_to_the_topic_that_the_page_does_not_make_are_removed(text):
    out = _ai(text)
    assert out.kept == 0 and "links it to the topic" in out.removed[0]["reason"]


@pytest.mark.parametrize("text", [
    "* Iran's U.N. ambassador criticized the U.S. military for using artificial intelligence systems in the war [W1].",
    "* The Iran-backed Houthi group claimed an attack on the financial district of Saudi Arabia's capital [W1].",
])
def test_links_the_page_makes_and_points_without_one_stay(text):
    assert _ai(text).kept == 1


def test_pages_about_the_topic_are_trusted_to_be_about_it():
    page = "AI toys fail safety tests\nAI toys told children odd things.\nPIRG tested AI plush toys.\nParents worry about bedtime."
    assert _ai("* Parents worry about bedtime, as AI plush companions enter homes everywhere [W1].", page).kept == 1


def test_no_topic_means_no_link_check():
    assert rc.check("* Houthi group claimed an attack on an oil refinery, fueled by AI precision strikes [W1].",
                    {"W1": CBS}).kept == 1
    assert search.topic_pattern("latest news") is None

"""Reading a watched page and comparing it with last time."""
from __future__ import annotations

import time

import pytest

import watch_check as wc
from safe_fetch import FetchError, Page


def page(html, kind="text/html"):
    return wc.lines(Page("https://s.example", kind, html))


NUM = {"kind": "number", "label": "Price", "when": "changes", "limit": None}


@pytest.mark.parametrize("text,value", [
    ("1,234.50", 1234.5), ("1.234,50", 1234.5), ("1 234", 1234), ("1,234", 1234), ("12,5", 12.5),
    ("1,234,567", 1234567), ("1.234.567", 1234567), ("0.5", 0.5), ("0,123", 0.123), ("88", 88),
    ("3.1415", 3.1415), ("1 299,00", 1299), ("99.", 99), ("3.15.2", 3.15), ("0.32.14", 0.32), ("1.2.3.4", 1.2), ("10.0.100", 10.0), ("abc", None), ("", None), (".", None),
])
def test_parse_number(text, value):
    assert wc.parse_number(text) == value


def test_visible_text_only():
    got = page("<html><head><title>T</title></head><body><script>var x=1</script><style>a{}</style>"
               "<h1>Hi &amp; bye</h1><p>One <b>two</b> three</p><noscript>no</noscript><div>Four</div></body>")
    assert got == ["Hi & bye", "One two three", "Four"]


def test_json_is_split_into_lines():
    assert '"price": 5,' in page('{"price": 5, "x": [1]}', "application/json")


@pytest.mark.parametrize("html", ["", "<html><script>x</script></html>", "   "])
def test_empty_pages_are_an_error(html):
    with pytest.raises(FetchError, match="no readable text"):
        page(html)


@pytest.mark.parametrize("html", ["<div><script>", "<<<>>><p", "<p>" * 5000 + "x", "&#xZZ; ok", "<svg><p>a</svg>b"])
def test_broken_html_never_raises_anything_else(html):
    try:
        page(html)
    except FetchError:
        pass


def test_large_pages_are_bounded_and_quick():
    start = time.perf_counter()
    got = page("".join(f"<p>line {i} {'x' * 600}</p>" for i in range(9000)))
    assert len(got) == 4000 and max(map(len, got)) <= 400 and time.perf_counter() - start < 3


# ── number ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("html,written,value", [
    ("<p>Price: $1,299.00 today</p>", "$1,299.00", 1299),
    ("<p>price</p><p>€ 49,90</p>", "€ 49,90", 49.9),
    ("<p>Price change -3.5%</p>", "-3.5", -3.5),
    ("<p>PRICE</p><span>88</span><span>%</span>", "88", 88),
])
def test_number_after_its_label(html, written, value):
    got = wc.compare(NUM, page(html), None)
    assert (got.state["written"], got.state["value"]) == (written, value)


def test_number_skips_a_label_with_no_number_near_it():
    html = "<p>Price guide</p>" + "<p>word word word</p>" * 40 + "<p>Price 12</p>"
    assert wc.compare(NUM, page(html), None).state["value"] == 12


@pytest.mark.parametrize("html", ["<p>Nothing here</p>", "<p>Price on request</p>"])
def test_missing_number_is_an_error_not_a_change(html):
    with pytest.raises(FetchError, match="No number"):
        wc.compare(NUM, page(html), {"written": "1", "value": 1, "met": False, "version": ""})


def step(rule, *pages):
    """Alerts for a series of looks, the first being the starting point."""
    state, out = None, []
    for html in pages:
        r = wc.compare(rule, page(html), state)
        state = r.state
        out.append(r.alert)
    return out


def test_first_look_never_alerts_and_changes_are_exact():
    assert step(NUM, "<p>Price $10</p>", "<p>Price $10.00</p>", "<p>Price $12</p>") == [
        None, None, "Price: $10.00 → $12"]


def test_below_alerts_once_per_crossing():
    rule = {**NUM, "when": "below", "limit": 100}
    pages = [f"<p>Price {n}</p>" for n in (120, 99, 95, 101, 98)]
    assert step(rule, *pages) == [None, "Price: 120 → 99 (below 100)", None, None, "Price: 101 → 98 (below 100)"]


def test_already_below_at_the_start_is_not_news():
    assert step({**NUM, "when": "below", "limit": 100}, "<p>Price 50</p>", "<p>Price 49</p>") == [None, None]


def test_above_and_a_limit_of_zero():
    assert step({**NUM, "when": "above", "limit": 0}, "<p>Price -1</p>", "<p>Price 1</p>")[1] == "Price: -1 → 1 (above 0)"


def test_versions_change_and_are_not_thousands():
    rule = {**NUM, "label": "Download Python"}
    assert step(rule, "<p>Download Python 3.15.2</p>", "<p>Download Python 3.15.3</p>")[1] == "Download Python: 3.15.2 → 3.15.3"
    above = {**rule, "when": "above", "limit": 3.15}
    assert step(above, "<p>Download Python 3.15.2</p>", "<p>Download Python 3.16.0</p>")[1].endswith("(above 3.15)")


# ── phrase ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("when,want", [("appears", [None, True, None]), ("disappears", [None, None, True]),
                                       ("either", [None, True, True])])
def test_phrase(when, want):
    rule = {"kind": "phrase", "phrase": "In  STOCK", "when": when}
    got = step(rule, "<p>Sold out</p>", "<p>Now in</p><p>stock</p>", "<p>Sold out</p>")
    assert [bool(a) if a else None for a in got] == want


# ── change ───────────────────────────────────────────────────────────────────

CH = {"kind": "change", "only": ""}
BASE = "".join(f"<p>Item {i}</p>" for i in range(20))


def test_added_and_removed_lines_are_quoted():
    assert step(CH, BASE, BASE.replace("<p>Item 3</p>", "<p>Item three</p>"))[1] == "+ Item three\n− Item 3"


def test_clocks_and_counters_are_ignored():
    noise = ["12:30 PM", "Updated 5 minutes ago", "1,204 views", "just now", "Posted: 2 hours ago", "14:02:11 UTC"]
    assert step(CH, BASE, BASE + "".join(f"<p>{n}</p>" for n in noise))[1] is None
    assert step(CH, BASE, BASE + "<p>12 items left</p>")[1] == "+ 12 items left"


def test_only_lines_with_a_phrase():
    rule = {"kind": "change", "only": "release"}
    assert step(rule, BASE, BASE + "<p>other</p>", BASE + "<p>New RELEASE 2</p>") == [None, None, "+ New RELEASE 2"]


def test_moved_lines_are_not_changes():
    assert step(CH, "<p>a1</p><p>b1</p>", "<p>b1</p><p>a1</p>")[1] is None


def test_a_replaced_page_is_summed_up_not_dumped():
    out = step(CH, BASE, "<p>Please verify you are human</p>")[1]
    assert out.startswith("Most of the page changed") and "Item" not in out


def test_long_changes_are_cut():
    out = step(CH, BASE, BASE + "".join(f"<p>New {i} {'x' * 300}</p>" for i in range(30)))[1]
    assert "and 22 more" in out and max(map(len, out.splitlines())) <= 163


# ── two loads ────────────────────────────────────────────────────────────────

def both(rule, before, one, two):
    return wc.confirmed(rule, page(one), page(two), before)


def test_a_line_only_one_load_has_is_not_reported():
    start = both(CH, None, BASE + "<p>Ad A</p>", BASE + "<p>Ad B</p>")
    assert "Ad A" not in start.state["lines"] and start.alert is None
    again = both(CH, start.state, BASE + "<p>Ad C</p>", BASE + "<p>Ad D</p>")
    assert again.alert is None


def test_a_real_change_both_loads_have_is_reported():
    start = both(CH, None, BASE, BASE)
    assert both(CH, start.state, BASE + "<p>News!</p>", BASE + "<p>News!</p>").alert == "+ News!"


def test_a_change_landing_between_the_two_loads_is_reported_next_time():
    start = both(CH, None, BASE, BASE)
    mid = both(CH, start.state, BASE, BASE + "<p>News!</p>")
    assert mid.alert is None
    assert both(CH, mid.state, BASE + "<p>News!</p>", BASE + "<p>News!</p>").alert == "+ News!"


def test_a_line_that_keeps_flickering_is_ignored_for_good():
    state = both(CH, None, BASE + "<p>Ad A</p>", BASE).state
    state = both(CH, state, BASE, BASE + "<p>Ad A</p>").state
    assert both(CH, state, BASE + "<p>Ad A</p>", BASE + "<p>Ad A</p>").alert is None


def test_a_known_line_missing_from_one_load_is_not_removed():
    start = both(CH, None, BASE, BASE)
    out = both(CH, start.state, BASE.replace("<p>Item 3</p>", ""), BASE)
    assert out.alert is None and "Item 3" in out.state["lines"]


def test_numbers_must_agree_twice():
    start = both(NUM, None, "<p>Price 10</p>", "<p>Price 10</p>").state
    assert both(NUM, start, "<p>Price 0</p>", "<p>Price 10</p>").alert is None
    flicker = both(NUM, start, "<p>Price 11</p>", "<p>Price 12</p>")
    assert flicker.alert is None and flicker.state == start
    assert both(NUM, start, "<p>Price 12</p>", "<p>Price 12</p>").alert == "Price: 10 → 12"


def test_noise_memory_is_bounded():
    state = None
    for k in range(3):
        state = both(CH, state, BASE + "".join(f"<p>r{k}-{i}</p>" for i in range(400)), BASE).state
    assert len(state["noisy"]) <= 500

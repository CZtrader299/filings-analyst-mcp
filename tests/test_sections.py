"""Tests for the 10-K section extractor."""

from __future__ import annotations

from pathlib import Path

from filings_analyst import sections


FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


def test_extract_sections_finds_all_four():
    html = FIXTURE.read_text(encoding="utf-8")
    result = sections.extract_sections(html)
    assert set(result.keys()) == set(sections.SECTION_NAMES)
    # Each present section should have at least some text.
    assert "hypothetical entity" in result["Business"]
    assert "high degree of fictional risk" in result["Risk Factors"]
    assert "Results of Operations" in result["MD&A"]
    # Financial Statements section has body text in our fixture.
    assert "consolidated financial statements" in result["Financial Statements"].lower()


def test_risk_factors_does_not_leak_into_business():
    html = FIXTURE.read_text(encoding="utf-8")
    result = sections.extract_sections(html)
    # The Risk Factors marker phrase must not appear inside Business.
    assert "high degree of fictional risk" not in result["Business"]


def test_extract_sections_returns_empty_strings_for_missing():
    # Minimal HTML with no Item headers at all.
    html = "<html><body><p>nothing here</p></body></html>"
    result = sections.extract_sections(html)
    assert set(result.keys()) == set(sections.SECTION_NAMES)
    for value in result.values():
        assert value == ""


def test_extract_sections_handles_empty_input():
    assert sections.extract_sections("") == {n: "" for n in sections.SECTION_NAMES}


def test_clean_section_text_strips_artifacts():
    raw = "Line one  with   spaces\n  \n\n\n12\n-3-\nLine two (continued) end."
    cleaned = sections.clean_section_text(raw)
    # Bare page-number lines should be dropped.
    assert "\n12\n" not in cleaned
    assert "-3-" not in cleaned
    # (continued) marker should be removed.
    assert "(continued)" not in cleaned.lower()
    assert "Line one" in cleaned
    assert "Line two" in cleaned


def test_clean_section_text_handles_empty():
    assert sections.clean_section_text("") == ""
    assert sections.clean_section_text("   \n\n  ") == ""


# --- Heading-variant regression tests ----------------------------------
#
# These exercise the formattings observed in real filings (Apple 2025
# 10-K and similar) that an earlier version of the regex missed.


def _build_aapl_style_html() -> str:
    """Synthetic 10-K mimicking Apple's heading style.

    Uses ``&#160;`` (non-breaking space) between the item number and
    title, and ``&#8217;`` (right single quote) for the apostrophe in
    "Management's". This is exactly what tripped the prior MD&A regex.
    No real SEC content is shipped — body text is synthetic.
    """
    return """
    <html><body>
    <h2>Table of Contents</h2>
    <p>Item 1.&#160;Business</p>
    <p>Item 1A.&#160;Risk Factors</p>
    <p>Item 7.&#160;Management&#8217;s Discussion and Analysis</p>
    <p>Item 8.&#160;Financial Statements</p>

    <h3>Item 1.&#160;&#160;&#160;&#160;Business</h3>
    <p>We make synthetic widgets in this Apple-style fixture.</p>

    <h3>Item 1A.&#160;&#160;&#160;&#160;Risk Factors</h3>
    <p>Curly-quote risk factors body content goes here.</p>

    <h3>Item 7.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</h3>
    <p>This MD&amp;A discusses synthetic results of operations.</p>

    <h3>Item 8.&#160;&#160;&#160;&#160;Financial Statements and Supplementary Data</h3>
    <p>The synthetic consolidated financial statements follow.</p>
    </body></html>
    """


def test_extract_sections_aapl_style_curly_apostrophe_and_nbsp():
    html = _build_aapl_style_html()
    result = sections.extract_sections(html)
    # The body MD&A must be found despite the curly apostrophe and nbsp.
    assert "synthetic results of operations" in result["MD&A"]
    # Body Business / Risk Factors / Financial Statements likewise.
    assert "synthetic widgets" in result["Business"]
    assert "Curly-quote risk factors body content" in result["Risk Factors"]
    assert "synthetic consolidated financial statements" in result["Financial Statements"]


def test_extract_sections_all_caps_no_trailing_period():
    """ALL CAPS heading with no period after the item number."""
    html = """
    <html><body>
    <p>ITEM 1 BUSINESS</p>
    <p>All caps business body.</p>
    <p>ITEM 1A RISK FACTORS</p>
    <p>All caps risk body.</p>
    <p>ITEM 7 MANAGEMENT DISCUSSION AND ANALYSIS</p>
    <p>All caps MD&amp;A body content.</p>
    <p>ITEM 8 FINANCIAL STATEMENTS</p>
    <p>All caps financials body.</p>
    </body></html>
    """
    result = sections.extract_sections(html)
    assert "All caps business body" in result["Business"]
    assert "All caps risk body" in result["Risk Factors"]
    assert "All caps MD&A body content" in result["MD&A"]
    assert "All caps financials body" in result["Financial Statements"]


def test_extract_sections_mda_title_only_fallback():
    """If no Item-7 anchor is present but the title appears, MD&A still found."""
    html = """
    <html><body>
    <p>Some preamble.</p>
    <h3>Management's Discussion and Analysis of Financial Condition</h3>
    <p>Title-only MD&amp;A body.</p>
    </body></html>
    """
    result = sections.extract_sections(html)
    assert "Title-only MD&A body" in result["MD&A"]


def test_normalize_heading_line_folds_curly_quotes_and_nbsp():
    line = "Item 7.\xa0\xa0Management’s Discussion"
    normalized = sections._normalize_heading_line(line)
    assert "\xa0" not in normalized
    assert "’" not in normalized
    assert "Management's" in normalized

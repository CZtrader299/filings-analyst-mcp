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


# --- Section-title fallback regression tests ----------------------------
#
# Three real failure modes observed in our corpus (JPM, BAC, XOM 10-Ks).
# Each fixture below mimics the structural pattern of one ticker without
# bundling any real SEC content.


def _jpm_style_forward_reference_html() -> str:
    """JPM-style: Item 7 anchor is a forward-reference placeholder.

    Mimics the 2025 JPMorgan 10-K layout: the literal ``Item 7.`` header
    is followed by a short placeholder paragraph saying MD&A appears
    elsewhere, then ``Item 7A.`` immediately. The actual MD&A content
    is in a separately-titled ``Management's discussion and analysis``
    section much later in the document.
    """
    # The "real MD&A body" line is repeated to push the cleaned-text
    # length above the 3000-char reasonable-MD&A floor and the prose-
    # density floor in the fallback.
    body_para = (
        "This section discusses the financial condition and results of "
        "operations of the firm for the year ended. Management evaluates "
        "performance across the consumer banking, commercial banking, "
        "investment banking, and asset management segments. Revenues "
        "grew across all segments driven by higher net interest income "
        "and stronger trading results. "
    )
    real_body = ("<p>" + body_para + "</p>\n") * 60
    return f"""
    <html><body>
    <h2>Table of Contents</h2>
    <p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>

    <h3>Item 1. Business</h3>
    <p>The Firm is a financial holding company.</p>

    <h3>Item 1A. Risk Factors</h3>
    <p>The Firm faces a variety of risks.</p>

    <h3>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations.</h3>
    <p>Management's discussion and analysis of financial condition and results of operations, entitled "Management's discussion and analysis," appears on pages 46-160. Such information should be read in conjunction with the consolidated financial statements.</p>
    <h3>Item 7A. Quantitative and Qualitative Disclosures About Market Risk.</h3>
    <p>Refer to the Market Risk Management section of MD&amp;A on pages 133-142.</p>
    <h3>Item 8. Financial Statements and Supplementary Data.</h3>
    <p>Refer to the consolidated financial statements.</p>

    <h2>Management's discussion and analysis</h2>
    <p>The following is Management's discussion and analysis of the financial condition and results of operations of the firm.</p>
    {real_body}

    <h2>Report of Independent Registered Public Accounting Firm</h2>
    <p>To the Board of Directors and Shareholders.</p>
    </body></html>
    """


def test_extract_sections_jpm_forward_reference_fallback():
    html = _jpm_style_forward_reference_html()
    result = sections.extract_sections(html)
    mda = result["MD&A"]
    # The fallback should land in the *real* body, not the short
    # forward-reference paragraph. The real body is many KB; the
    # forward-reference paragraph is a few hundred chars.
    assert len(mda) > sections._MDA_MIN_REASONABLE_CHARS
    assert "following is Management's discussion and analysis" in mda
    # The forward-reference placeholder language should NOT be the
    # body we picked (its sentinel phrase doesn't appear in the real
    # body of this fixture).
    assert "appears on pages 46-160" not in mda
    # End marker should have cut at the auditor's report.
    assert "To the Board of Directors" not in mda


def _xom_style_toc_anchor_html() -> str:
    """XOM-style: multiple ``Item 7.`` matches, repeating page headers.

    Mimics the ExxonMobil layout: the document contains an Item 7
    forward-reference in the cover section, then the actual MD&A body
    which uses repeated ALL-CAPS page headers ("MANAGEMENT'S DISCUSSION
    AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS") at
    the top of every page. The existing Item-anchor 'last match'
    heuristic landed on a TOC repeat; we need the title fallback
    to land in the dense body instead.
    """
    page_body = (
        "Earnings increased compared to the prior year reflecting higher "
        "realizations, growing advantaged volumes, and structural cost "
        "savings. Upstream earnings were driven by higher liquids volumes "
        "from advantaged assets. Energy products earnings reflected higher "
        "industry refining margins. "
    )
    # Build several "pages" each prefixed by the repeating ALL-CAPS header.
    pages = ""
    for _ in range(40):
        pages += (
            "<p>Financial Table of Contents</p>\n"
            "<p>MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>\n"
            f"<p>{page_body}</p>\n" * 4
        )
    return f"""
    <html><body>
    <a href="#mda"><p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</p></a>
    <a href="#mda7a"><p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p></a>

    <h3>Item 1. Business</h3>
    <p>ExxonMobil is an integrated oil and gas company.</p>

    <h3>Item 1A. Risk Factors</h3>
    <p>The company faces commodity-price risk.</p>

    <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
    <p>Reference is made to the section entitled Management's Discussion and Analysis of Financial Condition and Results of Operations in the Financial Section of this report.</p>
    <p>ITEM 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
    <p>Reference is made to the Financial Section.</p>
    <p>ITEM 8. Financial Statements and Supplementary Data</p>
    <p>Reference is made to the Financial Section.</p>

    {pages}

    <h2>Report of Independent Registered Public Accounting Firm</h2>
    <p>To the Shareholders.</p>
    </body></html>
    """


def test_extract_sections_xom_toc_anchor_fallback():
    html = _xom_style_toc_anchor_html()
    result = sections.extract_sections(html)
    mda = result["MD&A"]
    # Real MD&A body should be picked up despite the cover Item 7
    # forward-reference and the repeating ALL-CAPS page headers.
    assert len(mda) > sections._MDA_MIN_REASONABLE_CHARS
    assert "Earnings increased compared to the prior year" in mda
    # Auditor's report should NOT be in MD&A.
    assert "To the Shareholders" not in mda


def test_extract_sections_aapl_style_does_not_trigger_fallback():
    """Regression: the Apple-style fixture already produces a long MD&A.

    The fallback path must not fire here — verifies the threshold is
    well above the synthetic body length and the unchanged path still
    works for filings the original heuristic handled correctly.
    """
    html = _build_aapl_style_html()
    # Pad the AAPL fixture's MD&A body so it clears the reasonable
    # threshold. (The synthetic body in the original fixture is short
    # because it's a test fixture; real Apple MD&A is 20K+ chars.)
    body_para = (
        "Net sales increased compared to the prior year driven by higher "
        "iPhone and services revenues across all geographic segments. "
    ) * 200
    html = html.replace(
        "<p>This MD&amp;A discusses synthetic results of operations.</p>",
        f"<p>This MD&amp;A discusses synthetic results of operations.</p>\n<p>{body_para}</p>",
    )
    result = sections.extract_sections(html)
    # Item-anchor path handles this fixture; the fallback wasn't needed.
    assert "synthetic results of operations" in result["MD&A"]
    assert "Net sales increased" in result["MD&A"]

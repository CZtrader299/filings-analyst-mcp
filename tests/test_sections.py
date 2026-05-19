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

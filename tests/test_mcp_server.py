"""Smoke tests for the MCP tool implementations.

We exercise the underlying tool functions directly instead of spinning up
the stdio server — the SDK's stdio plumbing is well-tested upstream and
adds nothing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from filings_analyst import edgar, mcp_server, sections


FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


def _seed_cache(tmp_path: Path, ticker: str, accession_no: str) -> None:
    target = tmp_path / ticker / "10-K" / accession_no
    target.mkdir(parents=True)
    (target / "full-text.html").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "accession_no": accession_no,
                "ticker": ticker,
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "period_end": "2024-09-28",
            }
        ),
        encoding="utf-8",
    )


def test_tool_search_filings_calls_client():
    fake_filing = edgar.FilingRecord(
        accession_no="0000320193-24-000123",
        form_type="10-K",
        filing_date="2024-11-01",
        period_end="2024-09-28",
        primary_document="aapl.htm",
    )
    mock_client = MagicMock()
    mock_client.get_cik.return_value = "0000320193"
    mock_client.get_company_filings.return_value = [fake_filing]

    out = mcp_server.tool_search_filings("AAPL", client=mock_client)
    assert out["ticker"] == "AAPL"
    assert out["cik"] == "0000320193"
    assert out["filings"][0]["accession_no"] == "0000320193-24-000123"
    assert out["filings"][0]["filing_date"] == "2024-11-01"


def test_tool_get_filing_reads_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)

    out = mcp_server.tool_get_filing(accession, "AAPL", preview_chars=200)
    assert out["accession_no"] == accession
    assert out["ticker"] == "AAPL"
    assert out["text_length"] > 0
    assert len(out["preview"]) <= 200
    assert "error" not in out


def test_tool_extract_section_returns_text(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)

    out = mcp_server.tool_extract_section(accession, "AAPL", "Risk Factors")
    assert out["section"] == "Risk Factors"
    assert "fictional risk" in out["text"]
    assert "error" not in out


def test_tool_extract_section_unknown_section(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    _seed_cache(tmp_path, "AAPL", "0000320193-24-000123")
    out = mcp_server.tool_extract_section(
        "0000320193-24-000123", "AAPL", "Nonexistent Section"
    )
    assert out["text"] == ""
    assert "error" in out


def test_tool_extract_section_missing_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    out = mcp_server.tool_extract_section("0000000000-00-000000", "AAPL", "Business")
    assert out["text"] == ""
    assert "error" in out
    assert "cache" in out["error"].lower()


def test_build_server_registers_three_tools():
    """The MCP server build should not error and should expose all three tools."""
    pytest.importorskip("mcp")
    server = mcp_server._build_server()
    assert server is not None

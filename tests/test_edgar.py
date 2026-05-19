"""Tests for the SEC EDGAR client.

All HTTP is mocked via the ``responses`` library so these never touch
the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import responses

from filings_analyst import edgar


COMPANY_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


SUBMISSIONS_PAYLOAD = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000123", "0000320193-23-000106"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2024-11-01", "2023-11-03"],
            "reportDate": ["2024-09-28", "2023-09-30"],
            "primaryDocument": ["aapl-20240928.htm", "aapl-20230930.htm"],
        }
    },
}


@responses.activate
def test_get_cik_resolves_ticker():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=COMPANY_TICKERS_PAYLOAD,
        status=200,
    )
    client = edgar.EdgarClient()
    assert client.get_cik("AAPL") == "0000320193"
    # case-insensitive
    assert client.get_cik("aapl") == "0000320193"


@responses.activate
def test_get_cik_raises_on_unknown_ticker():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=COMPANY_TICKERS_PAYLOAD,
        status=200,
    )
    client = edgar.EdgarClient()
    with pytest.raises(ValueError):
        client.get_cik("NOPE")


@responses.activate
def test_get_company_filings_parses_recent_index():
    responses.add(
        responses.GET,
        "https://data.sec.gov/submissions/CIK0000320193.json",
        json=SUBMISSIONS_PAYLOAD,
        status=200,
    )
    client = edgar.EdgarClient()
    filings = client.get_company_filings("0000320193", form_type="10-K", count=5)
    assert len(filings) == 2
    assert filings[0].accession_no == "0000320193-24-000123"
    assert filings[0].form_type == "10-K"
    assert filings[0].filing_date == "2024-11-01"
    assert filings[0].period_end == "2024-09-28"
    assert filings[0].primary_document == "aapl-20240928.htm"


@responses.activate
def test_download_10k_caches_and_hits_cache(tmp_path: Path, monkeypatch):
    # Point the module-level cache at tmp_path.
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)

    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=COMPANY_TICKERS_PAYLOAD,
        status=200,
    )
    responses.add(
        responses.GET,
        "https://data.sec.gov/submissions/CIK0000320193.json",
        json=SUBMISSIONS_PAYLOAD,
        status=200,
    )
    responses.add(
        responses.GET,
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        body="<html><body>synthetic filing</body></html>",
        status=200,
        content_type="text/html",
    )

    # First call: hits the network (mocked) and writes cache.
    meta = edgar.download_10k("AAPL", cache_dir=tmp_path)
    assert meta["accession_no"] == "0000320193-24-000123"
    assert meta["ticker"] == "AAPL"
    html_path = Path(meta["html_path"])
    assert html_path.exists()
    assert "synthetic filing" in html_path.read_text(encoding="utf-8")

    # Second call: must skip the expensive document fetch. We re-issue the
    # cheap CIK + submissions JSON lookups (those would need a separate
    # cache layer we haven't built yet) but the document URL must NOT be
    # re-fetched.
    pre_count = len(responses.calls)
    meta2 = edgar.download_10k("AAPL", cache_dir=tmp_path)
    assert meta2 == meta
    document_calls = [
        c for c in responses.calls
        if "aapl-20240928.htm" in c.request.url
    ]
    # Only the first call should have hit the document URL.
    assert len(document_calls) == 1
    # And the second call added at most two cheap index lookups.
    assert len(responses.calls) - pre_count <= 2


def test_load_cached_helpers_raise_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        edgar.load_cached_filing_text("AAPL", "0000000000-00-000000", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        edgar.load_cached_metadata("AAPL", "0000000000-00-000000", cache_dir=tmp_path)


def test_load_cached_helpers_read_written_files(tmp_path: Path):
    target = tmp_path / "AAPL" / "10-K" / "0000320193-24-000123"
    target.mkdir(parents=True)
    (target / "full-text.html").write_text("hello", encoding="utf-8")
    (target / "metadata.json").write_text(
        json.dumps({"accession_no": "0000320193-24-000123", "ticker": "AAPL"}),
        encoding="utf-8",
    )
    assert (
        edgar.load_cached_filing_text("AAPL", "0000320193-24-000123", cache_dir=tmp_path)
        == "hello"
    )
    meta = edgar.load_cached_metadata("AAPL", "0000320193-24-000123", cache_dir=tmp_path)
    assert meta["ticker"] == "AAPL"

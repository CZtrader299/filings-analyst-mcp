"""SEC EDGAR client.

Two layers:

* ``EdgarClient`` — low-level wrapper around ``requests.Session`` with the
  mandated User-Agent and a small rate-limit sleep between requests.
* Module-level helpers — ``download_10k`` and
  ``download_10ks_for_starter_universe`` — handle on-disk caching so we
  don't hammer EDGAR for filings we've already fetched.

Cache layout::

    {cache_dir}/{ticker}/{form_type}/{accession_no}/full-text.html
    {cache_dir}/{ticker}/{form_type}/{accession_no}/metadata.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from . import config


SEC_HOST = "https://www.sec.gov"
DATA_HOST = "https://data.sec.gov"

# SEC's published cap is 10 requests/second. We sleep 0.11s between requests
# to leave headroom and stay polite.
RATE_LIMIT_SLEEP = 0.11


def _accession_no_clean(accession_no: str) -> str:
    """Convert 0000320193-24-000123 -> 000032019324000123."""
    return accession_no.replace("-", "")


@dataclass
class FilingRecord:
    """One row from the SEC submissions index."""

    accession_no: str
    form_type: str
    filing_date: str
    period_end: str = ""
    primary_document: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession_no": self.accession_no,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "period_end": self.period_end,
            "primary_document": self.primary_document,
        }


class EdgarClient:
    """Low-level SEC EDGAR HTTP client.

    Keeps a ``requests.Session`` with the SEC-mandated identifying
    User-Agent header. Adds a small sleep between requests so we stay
    well under SEC's 10 req/s cap.
    """

    def __init__(self, user_agent: str | None = None, session: requests.Session | None = None):
        self.user_agent = user_agent or config.USER_AGENT
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": None,  # let requests fill per-URL
            }
        )

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        # Rate-limit before each call. Tests that mock ``requests`` patch
        # at the requests level so the sleep is a near-no-op there.
        time.sleep(RATE_LIMIT_SLEEP)
        timeout = kwargs.pop("timeout", config.REQUEST_TIMEOUT)
        resp = self.session.get(url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp

    # --- CIK lookup ------------------------------------------------------

    def get_cik(self, ticker: str) -> str:
        """Look up the zero-padded 10-digit CIK for a ticker.

        Uses SEC's ``company_tickers.json`` index. Raises ``ValueError`` if
        the ticker is not in the index.
        """
        url = f"{SEC_HOST}/files/company_tickers.json"
        data = self._get(url).json()
        ticker_upper = ticker.upper()
        # The index is a dict of {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}
        for row in data.values():
            if row.get("ticker", "").upper() == ticker_upper:
                return str(row["cik_str"]).zfill(10)
        raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers index")

    # --- Filing list ----------------------------------------------------

    def get_company_filings(
        self,
        cik: str,
        form_type: str = "10-K",
        count: int = 10,
    ) -> list[FilingRecord]:
        """Return the most recent ``count`` filings of ``form_type``."""
        cik_padded = str(cik).zfill(10)
        url = f"{DATA_HOST}/submissions/CIK{cik_padded}.json"
        data = self._get(url).json()
        recent = data.get("filings", {}).get("recent", {})

        accession_nos = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        period_ends = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        out: list[FilingRecord] = []
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            out.append(
                FilingRecord(
                    accession_no=accession_nos[i],
                    form_type=form,
                    filing_date=filing_dates[i] if i < len(filing_dates) else "",
                    period_end=period_ends[i] if i < len(period_ends) else "",
                    primary_document=primary_docs[i] if i < len(primary_docs) else "",
                )
            )
            if len(out) >= count:
                break
        return out

    # --- Document fetch -------------------------------------------------

    def get_filing_text(self, accession_number: str, cik: str, primary_document: str = "") -> str:
        """Fetch the primary document of a filing as raw HTML/text."""
        cik_int = str(int(cik))  # archives URLs use un-padded CIK
        accession_clean = _accession_no_clean(accession_number)
        if primary_document:
            url = f"{SEC_HOST}/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_document}"
        else:
            # Fall back to the .txt full-submission, which is always present.
            url = f"{SEC_HOST}/Archives/edgar/data/{cik_int}/{accession_clean}/{accession_number}.txt"
        return self._get(url).text


# --- High-level cached helpers ------------------------------------------------


def _filing_cache_dir(ticker: str, form_type: str, accession_no: str) -> Path:
    return config.CACHE_DIR / ticker.upper() / form_type / accession_no


def download_10k(
    ticker: str,
    year: int | None = None,
    client: EdgarClient | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Download (or fetch from cache) the most recent 10-K for ``ticker``.

    If ``year`` is given, the most recent filing with a ``period_end``
    starting with that year is returned. Returns a metadata dict that
    includes the on-disk path to the cached HTML.
    """
    client = client or EdgarClient()
    base_cache = Path(cache_dir) if cache_dir else config.CACHE_DIR

    cik = client.get_cik(ticker)
    filings = client.get_company_filings(cik, form_type="10-K", count=10)
    if not filings:
        raise ValueError(f"No 10-K filings found for {ticker}")

    if year is not None:
        filings = [f for f in filings if f.period_end.startswith(str(year))]
        if not filings:
            raise ValueError(f"No 10-K for {ticker} in year {year}")

    chosen = filings[0]
    target_dir = base_cache / ticker.upper() / "10-K" / chosen.accession_no
    html_path = target_dir / "full-text.html"
    meta_path = target_dir / "metadata.json"

    if html_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    target_dir.mkdir(parents=True, exist_ok=True)
    text = client.get_filing_text(chosen.accession_no, cik, chosen.primary_document)
    html_path.write_text(text, encoding="utf-8")

    meta = {
        "ticker": ticker.upper(),
        "cik": cik,
        "accession_no": chosen.accession_no,
        "form_type": chosen.form_type,
        "filing_date": chosen.filing_date,
        "period_end": chosen.period_end,
        "primary_document": chosen.primary_document,
        "html_path": str(html_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def download_10ks_for_starter_universe(
    client: EdgarClient | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience: download the most recent 10-K for each starter ticker."""
    client = client or EdgarClient()
    results: list[dict[str, Any]] = []
    for ticker in config.STARTER_TICKERS:
        try:
            results.append(download_10k(ticker, client=client, cache_dir=cache_dir))
        except Exception as exc:  # noqa: BLE001 — log-and-continue is the goal
            results.append({"ticker": ticker, "error": str(exc)})
    return results


def load_cached_filing_text(
    ticker: str, accession_no: str, cache_dir: Path | None = None
) -> str:
    """Read the cached HTML text for a filing, or raise FileNotFoundError."""
    base = Path(cache_dir) if cache_dir else config.CACHE_DIR
    html_path = base / ticker.upper() / "10-K" / accession_no / "full-text.html"
    if not html_path.exists():
        raise FileNotFoundError(f"No cached filing at {html_path}")
    return html_path.read_text(encoding="utf-8")


def load_cached_metadata(
    ticker: str, accession_no: str, cache_dir: Path | None = None
) -> dict[str, Any]:
    base = Path(cache_dir) if cache_dir else config.CACHE_DIR
    meta_path = base / ticker.upper() / "10-K" / accession_no / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No cached metadata at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

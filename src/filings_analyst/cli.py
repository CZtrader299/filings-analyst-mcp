"""Command-line interface for filings-analyst.

Subcommands:

* ``ingest`` — download 10-Ks into the local cache.
* ``show-sections`` — print extracted sections for a cached filing.
* ``serve-mcp`` — launch the MCP server over stdio.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import config, edgar, sections


def _cmd_ingest(args: argparse.Namespace) -> int:
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = config.STARTER_TICKERS

    print(f"Cache dir: {config.CACHE_DIR}")
    print(f"User-Agent: {config.USER_AGENT}")
    print(f"Tickers: {', '.join(tickers)}")
    if args.year:
        print(f"Year filter: {args.year}")
    print()

    client = edgar.EdgarClient()
    any_failed = False
    for ticker in tickers:
        try:
            meta = edgar.download_10k(ticker, year=args.year, client=client)
            print(
                f"  OK {ticker}: {meta['accession_no']} "
                f"(filed {meta['filing_date']}, period {meta['period_end']})"
            )
        except Exception as exc:  # noqa: BLE001
            any_failed = True
            print(f"  FAIL {ticker}: {exc}")
    return 1 if any_failed else 0


def _cmd_show_sections(args: argparse.Namespace) -> int:
    try:
        html = edgar.load_cached_filing_text(args.ticker, args.accession_no)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Hint: run `filings-analyst ingest --tickers <TICKER>` first.", file=sys.stderr)
        return 2

    extracted = sections.extract_sections(html)
    for name in sections.SECTION_NAMES:
        text = extracted.get(name, "")
        print(f"=== {name} ({len(text)} chars) ===")
        if text:
            print(text[:200].rstrip())
        else:
            print("(not found)")
        print()
    return 0


def _cmd_serve_mcp(_args: argparse.Namespace) -> int:
    # Imported lazily so `filings-analyst --help` works even if mcp isn't installed.
    from . import mcp_server

    mcp_server.main()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filings-analyst",
        description="MCP server + ingestion tools for SEC EDGAR 10-K analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download 10-Ks into the local cache.")
    p_ingest.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker list (default: starter universe).",
    )
    p_ingest.add_argument(
        "--year",
        type=int,
        default=None,
        help="Only download filings whose period_end starts with this year.",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_show = sub.add_parser(
        "show-sections",
        help="Print extracted sections (head only) for a cached filing.",
    )
    p_show.add_argument("accession_no", help="Accession number, e.g. 0000320193-24-000123")
    p_show.add_argument("ticker", help="Ticker (used to locate the cache dir)")
    p_show.set_defaults(func=_cmd_show_sections)

    p_serve = sub.add_parser("serve-mcp", help="Run the MCP server over stdio.")
    p_serve.set_defaults(func=_cmd_serve_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

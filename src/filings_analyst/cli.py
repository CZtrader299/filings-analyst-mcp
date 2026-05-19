"""Command-line interface for filings-analyst.

Subcommands:

* ``ingest`` — download 10-Ks into the local cache, then chunk + embed
  + store unless ``--no-embed`` is passed.
* ``show-sections`` — print extracted sections for a cached filing.
* ``ask`` — RAG Q&A against one ingested filing.
* ``ask-corpus`` — RAG Q&A across every ingested filing.
* ``serve-mcp`` — launch the MCP server over stdio.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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
    if args.no_embed:
        print("Embed: disabled (--no-embed)")
    print()

    client = edgar.EdgarClient()
    any_failed = False

    # Defer RAG construction so a download-only run doesn't require the
    # embeddings extras to be installed.
    rag = None
    if not args.no_embed:
        try:
            from .rag import FilingRAG

            rag = FilingRAG()
        except ImportError as exc:
            print(
                f"  Warning: embeddings extras not installed ({exc}). "
                "Falling back to download-only."
            )
            rag = None

    for ticker in tickers:
        try:
            meta = edgar.download_10k(ticker, year=args.year, client=client)
        except Exception as exc:  # noqa: BLE001
            any_failed = True
            print(f"  FAIL {ticker}: {exc}")
            continue

        print(
            f"  OK {ticker}: {meta['accession_no']} "
            f"(filed {meta['filing_date']}, period {meta['period_end']})"
        )

        if rag is None:
            continue

        try:
            summary = rag.ingest_filing(meta["accession_no"], ticker)
        except Exception as exc:  # noqa: BLE001
            any_failed = True
            print(f"    embed FAIL: {exc}")
            continue

        if "error" in summary:
            any_failed = True
            print(f"    embed FAIL: {summary['error']}")
            continue

        print(
            f"    embedded {summary['chunks_added']} chunks in "
            f"{summary['elapsed_sec']}s "
            f"(dim={summary['embedding_dim']}, "
            f"provider={summary['embedding_provider']})"
        )
        for section_name, count in summary.get("chunks_by_section", {}).items():
            print(f"      - {section_name}: {count}")

    if rag is not None:
        rag.close()
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


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        from .rag import FilingRAG
    except ImportError as exc:
        print(f"Error: RAG dependencies not installed: {exc}", file=sys.stderr)
        print('Hint: run `pip install -e ".[embeddings]"`.', file=sys.stderr)
        return 2

    rag = FilingRAG()
    result = rag.ask_filing(
        args.question,
        accession_no=args.accession_no,
        ticker=args.ticker,
        k=args.k,
    )

    print(f"Q: {result['question']}\n")
    print("A:")
    if result.get("answer"):
        wrapped = textwrap.fill(
            result["answer"], width=88, replace_whitespace=False
        )
        print(wrapped)
    else:
        print("(no answer)")
        if "error" in result:
            print(f"\nError: {result['error']}")
    print()
    print(f"Provider: {result.get('provider', 'unknown')}")
    cited = result.get("cited_chunks") or []
    if cited:
        print(f"\nCited chunks ({len(cited)}):")
        for c in cited:
            snippet = c["text"][:80].replace("\n", " ")
            print(
                f"  - [{c['section']} §{c['chunk_idx']}] "
                f"score={c['score']:.4f}  {snippet}..."
            )

    rag.close()
    return 0 if result.get("answer") else 1


def _cmd_ask_corpus(args: argparse.Namespace) -> int:
    try:
        from .rag import FilingRAG
    except ImportError as exc:
        print(f"Error: RAG dependencies not installed: {exc}", file=sys.stderr)
        print('Hint: run `pip install -e ".[embeddings]"`.', file=sys.stderr)
        return 2

    tickers: list[str] | None = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    accession_nos: list[str] | None = None
    if args.accession_nos:
        accession_nos = [
            a.strip() for a in args.accession_nos.split(",") if a.strip()
        ]

    rag = FilingRAG()
    result = rag.ask_corpus(
        args.question,
        tickers=tickers,
        accession_nos=accession_nos,
        k=args.k,
    )

    print(f"Q: {result['question']}\n")
    print("A:")
    if result.get("answer"):
        wrapped = textwrap.fill(
            result["answer"], width=88, replace_whitespace=False
        )
        print(wrapped)
    else:
        print("(no answer)")
        if "error" in result:
            print(f"\nError: {result['error']}")
    print()
    print(f"Provider: {result.get('provider', 'unknown')}")

    searched = result.get("filings_searched") or []
    if searched:
        print(f"\nFilings searched ({len(searched)}):")
        for f in searched:
            date = f.get("filing_date", "")
            date_suffix = f" (filed {date})" if date else ""
            print(f"  - {f['ticker']}  {f['accession_no']}{date_suffix}")

    cited = result.get("cited_chunks") or []
    if cited:
        print(f"\nCited chunks ({len(cited)}):")
        for c in cited:
            snippet = c["text"][:80].replace("\n", " ")
            print(
                f"  - [{c['ticker']} {c['section']} §{c['chunk_idx']}] "
                f"score={c['score']:.4f}  {snippet}..."
            )

    rag.close()
    return 0 if result.get("answer") else 1


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

    p_ingest = sub.add_parser(
        "ingest",
        help="Download 10-Ks and embed their sections into the vector store.",
    )
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
    p_ingest.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip chunking + embedding; download to cache only.",
    )
    # Positional tickers convenience (`filings-analyst ingest AAPL`).
    p_ingest.add_argument(
        "ticker_pos",
        nargs="?",
        default=None,
        help="Optional positional ticker (shortcut for --tickers).",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_show = sub.add_parser(
        "show-sections",
        help="Print extracted sections (head only) for a cached filing.",
    )
    p_show.add_argument("accession_no", help="Accession number, e.g. 0000320193-24-000123")
    p_show.add_argument("ticker", help="Ticker (used to locate the cache dir)")
    p_show.set_defaults(func=_cmd_show_sections)

    p_ask = sub.add_parser(
        "ask",
        help="Ask a question about an ingested filing (RAG).",
    )
    p_ask.add_argument("accession_no", help="Accession number of the ingested filing")
    p_ask.add_argument("ticker", help="Ticker (uppercased internally)")
    p_ask.add_argument("question", help="Natural-language question in quotes")
    p_ask.add_argument(
        "--k", type=int, default=6, help="Top-k chunks to retrieve (default 6)"
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_ask_corpus = sub.add_parser(
        "ask-corpus",
        help="Ask a question across every ingested filing (multi-filing RAG).",
    )
    p_ask_corpus.add_argument(
        "question", help="Natural-language question in quotes"
    )
    p_ask_corpus.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker filter (default: search all ingested filings).",
    )
    p_ask_corpus.add_argument(
        "--accession-nos",
        type=str,
        default="",
        dest="accession_nos",
        help="Comma-separated accession-number filter (default: no filter).",
    )
    p_ask_corpus.add_argument(
        "--k", type=int, default=8, help="Top-k chunks to retrieve (default 8)"
    )
    p_ask_corpus.set_defaults(func=_cmd_ask_corpus)

    p_serve = sub.add_parser("serve-mcp", help="Run the MCP server over stdio.")
    p_serve.set_defaults(func=_cmd_serve_mcp)

    return parser


def _normalize_ingest_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fold positional ticker shortcut into --tickers."""
    if getattr(args, "ticker_pos", None) and not args.tickers:
        args.tickers = args.ticker_pos
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        args = _normalize_ingest_args(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

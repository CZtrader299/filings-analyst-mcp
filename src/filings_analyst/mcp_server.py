"""MCP server exposing 10-K query tools over stdio.

Five tools today:

* ``search_filings`` — list available filings for a ticker.
* ``get_filing`` — return metadata + preview of a filing's full text.
* ``extract_section`` — return a named section from a filing.
* ``ask_filing`` — RAG-backed Q&A over a single ingested filing.
* ``ask_corpus`` — RAG-backed Q&A across every ingested filing
  (optionally filtered by ticker or accession). Returns ticker-tagged
  citations so the reader knows which filing each excerpt came from.

Run with::

    filings-analyst serve-mcp
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from . import edgar, sections


# --- Tool implementations (pure functions so tests can call them directly) ---


def tool_search_filings(
    ticker: str,
    form_type: str = "10-K",
    year: int | None = None,
    client: edgar.EdgarClient | None = None,
) -> dict[str, Any]:
    """List recent filings for ``ticker``.

    Returns ``{"ticker": ..., "filings": [{"accession_no": ..., "form_type": ...,
    "filing_date": ..., "period_end": ...}, ...]}``.
    """
    client = client or edgar.EdgarClient()
    cik = client.get_cik(ticker)
    filings = client.get_company_filings(cik, form_type=form_type, count=20)
    if year is not None:
        filings = [f for f in filings if f.period_end.startswith(str(year))]
    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "filings": [f.to_dict() for f in filings],
    }


def tool_get_filing(
    accession_no: str,
    ticker: str,
    preview_chars: int = 2000,
) -> dict[str, Any]:
    """Return cached filing metadata + a short preview.

    Doesn't return the full HTML — MCP clients shouldn't be force-fed
    1MB documents. Use ``extract_section`` for targeted content.
    """
    try:
        meta = edgar.load_cached_metadata(ticker, accession_no)
        text = edgar.load_cached_filing_text(ticker, accession_no)
    except FileNotFoundError:
        # Auto-download on miss. The CLI ingest command is the documented
        # path, but a fresh client invoking get_filing directly should
        # still work.
        meta = edgar.download_10k(ticker)
        if meta.get("accession_no") != accession_no:
            return {
                "accession_no": accession_no,
                "ticker": ticker.upper(),
                "error": (
                    f"Requested accession_no {accession_no} not in cache; "
                    f"latest available is {meta.get('accession_no')}. "
                    "Run `filings-analyst ingest` for a wider range."
                ),
            }
        text = edgar.load_cached_filing_text(ticker, accession_no)

    return {
        "accession_no": accession_no,
        "ticker": ticker.upper(),
        "form_type": meta.get("form_type", ""),
        "filing_date": meta.get("filing_date", ""),
        "period_end": meta.get("period_end", ""),
        "text_length": len(text),
        "preview": text[:preview_chars],
    }


def tool_extract_section(
    accession_no: str,
    ticker: str,
    section: str,
) -> dict[str, Any]:
    """Return one named section from a cached filing."""
    if section not in sections.SECTION_NAMES:
        return {
            "accession_no": accession_no,
            "section": section,
            "text": "",
            "error": (
                f"Unknown section {section!r}. "
                f"Supported: {list(sections.SECTION_NAMES)}"
            ),
        }
    try:
        html = edgar.load_cached_filing_text(ticker, accession_no)
    except FileNotFoundError as exc:
        return {
            "accession_no": accession_no,
            "section": section,
            "text": "",
            "error": f"Filing not in cache: {exc}",
        }
    extracted = sections.extract_sections(html)
    text = extracted.get(section, "")
    out: dict[str, Any] = {
        "accession_no": accession_no,
        "ticker": ticker.upper(),
        "section": section,
        "text": text,
    }
    if not text:
        out["error"] = "Section not found in this filing"
    return out


def tool_ask_corpus(
    question: str,
    tickers: Optional[list[str]] = None,
    accession_nos: Optional[list[str]] = None,
    k: int = 8,
    rag: Optional[Any] = None,
) -> dict[str, Any]:
    """RAG-backed Q&A across every ingested filing.

    Args:
        question: Natural-language question.
        tickers: Optional list of tickers to restrict retrieval to.
        accession_nos: Optional list of accession numbers to restrict
            retrieval to.
        k: Top-k chunks to retrieve.
        rag: Test injection hook; production callers pass ``None``.
    """
    if rag is None:
        from .rag import FilingRAG

        try:
            rag = FilingRAG()
        except ImportError as exc:
            return {
                "question": question,
                "error": (
                    f"RAG dependencies not installed: {exc}. "
                    'Run `pip install -e ".[embeddings]"`.'
                ),
            }

    try:
        return rag.ask_corpus(
            question,
            tickers=tickers,
            accession_nos=accession_nos,
            k=k,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "question": question,
            "error": f"ask_corpus failed: {exc}",
        }


def tool_ask_filing(
    accession_no: str,
    ticker: str,
    question: str,
    k: int = 6,
    rag: Optional[Any] = None,
) -> dict[str, Any]:
    """RAG-backed Q&A over a single ingested filing.

    The ``rag`` arg is for test injection; production callers leave it
    None so a fresh ``FilingRAG`` is constructed per call.
    """
    if rag is None:
        # Local import: keeps mcp_server importable even when the
        # embeddings extras aren't installed (tests for the deterministic
        # tools should still pass on a minimal env).
        from .rag import FilingRAG

        try:
            rag = FilingRAG()
        except ImportError as exc:
            return {
                "accession_no": accession_no,
                "ticker": ticker.upper(),
                "question": question,
                "error": (
                    f"RAG dependencies not installed: {exc}. "
                    'Run `pip install -e ".[embeddings]"`.'
                ),
            }

    try:
        result = rag.ask_filing(
            question,
            accession_no=accession_no,
            ticker=ticker,
            k=k,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "accession_no": accession_no,
            "ticker": ticker.upper(),
            "question": question,
            "error": f"ask_filing failed: {exc}",
        }
    # Inject identifying fields so MCP clients always see them.
    result.setdefault("accession_no", accession_no)
    result.setdefault("ticker", ticker.upper())
    return result


# --- MCP server wiring -------------------------------------------------------


def _build_server():
    """Construct the MCP Server with the three tools registered.

    Imported lazily so ``import filings_analyst.mcp_server`` doesn't fail
    in environments where the ``mcp`` SDK isn't installed (e.g., minimal
    test envs).
    """
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("filings-analyst")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_filings",
                description=(
                    "List recent SEC EDGAR filings for a ticker. "
                    "Defaults to 10-K. Optional `year` filters by period end."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "form_type": {"type": "string", "default": "10-K"},
                        "year": {"type": ["integer", "null"]},
                    },
                    "required": ["ticker"],
                },
            ),
            Tool(
                name="get_filing",
                description=(
                    "Return metadata and a short preview of a filing's full "
                    "text. Filing must already be in the local cache."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "accession_no": {"type": "string"},
                        "ticker": {"type": "string"},
                    },
                    "required": ["accession_no", "ticker"],
                },
            ),
            Tool(
                name="ask_filing",
                description=(
                    "Answer a natural-language question about a single 10-K "
                    "filing using RAG over its sections. Filing must already "
                    "be ingested into the vector store via "
                    "`filings-analyst ingest`. Returns the answer plus the "
                    "cited chunks used to ground it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "accession_no": {"type": "string"},
                        "ticker": {"type": "string"},
                        "question": {"type": "string"},
                        "k": {"type": "integer", "default": 6},
                    },
                    "required": ["accession_no", "ticker", "question"],
                },
            ),
            Tool(
                name="ask_corpus",
                description=(
                    "Answer a natural-language question across every "
                    "ingested 10-K filing (optionally filtered by ticker "
                    "or accession number). Returns the answer plus the "
                    "cited chunks — each tagged with the ticker, "
                    "accession_no, and filing_date so the reader knows "
                    "which filing each excerpt came from — and a "
                    "filings_searched manifest of the filings whose "
                    "chunks actually contributed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "accession_nos": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "k": {"type": "integer", "default": 8},
                    },
                    "required": ["question"],
                },
            ),
            Tool(
                name="extract_section",
                description=(
                    "Return one named section from a cached 10-K. "
                    "Supported sections: Business, Risk Factors, MD&A, "
                    "Financial Statements."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "accession_no": {"type": "string"},
                        "ticker": {"type": "string"},
                        "section": {
                            "type": "string",
                            "enum": list(sections.SECTION_NAMES),
                        },
                    },
                    "required": ["accession_no", "ticker", "section"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "search_filings":
            result = tool_search_filings(
                ticker=arguments["ticker"],
                form_type=arguments.get("form_type", "10-K"),
                year=arguments.get("year"),
            )
        elif name == "get_filing":
            result = tool_get_filing(
                accession_no=arguments["accession_no"],
                ticker=arguments["ticker"],
            )
        elif name == "extract_section":
            result = tool_extract_section(
                accession_no=arguments["accession_no"],
                ticker=arguments["ticker"],
                section=arguments["section"],
            )
        elif name == "ask_filing":
            result = tool_ask_filing(
                accession_no=arguments["accession_no"],
                ticker=arguments["ticker"],
                question=arguments["question"],
                k=arguments.get("k", 6),
            )
        elif name == "ask_corpus":
            result = tool_ask_corpus(
                question=arguments["question"],
                tickers=arguments.get("tickers"),
                accession_nos=arguments.get("accession_nos"),
                k=arguments.get("k", 8),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    server = _build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for ``filings-analyst serve-mcp``."""
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()

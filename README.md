# filings-analyst-mcp

An MCP server for querying SEC EDGAR filings. Built for AI agents and humans.

## Status

Active development — currently week 1 of 4. Today's scaffold ships ingestion, section extraction, and three deterministic MCP tools. RAG and formal evaluation land in subsequent commits.

## Why this exists

This is a portfolio project demonstrating three things at once: a working MCP server implementation, a real RAG pipeline with a formal evaluation harness (real precision/recall metrics, not vibes), and the same multi-provider AI architecture used in my LSE Buyback Scraper at [github.com/CZtrader299/lse-buyback-scraper](https://github.com/CZtrader299/lse-buyback-scraper). The differentiator over the average "RAG over 10-Ks" portfolio project is the eval harness — anyone can wire up a vector store, far fewer projects measure their retrieval quality honestly.

## Architecture (current)

```
src/filings_analyst/
  config.py        # env-driven constants (User-Agent, cache dir, model defaults)
  edgar.py         # SEC EDGAR client + on-disk cache helpers
  sections.py      # 10-K section extractor (Business / Risk Factors / MD&A / Financials)
  providers.py     # LLM provider router (Anthropic, OpenAI, Claude CLI)
  mcp_server.py    # MCP server: search_filings, get_filing, extract_section
  cli.py           # ingest / show-sections / serve-mcp subcommands
```

## MCP tools available now

- `search_filings(ticker, form_type="10-K", year=None)` — list recent filings for a ticker from SEC EDGAR.
- `get_filing(accession_no, ticker)` — return metadata and a short preview of a filing's full text (filing must be in the local cache).
- `extract_section(accession_no, ticker, section)` — return one named section ("Business", "Risk Factors", "MD&A", "Financial Statements") from a cached 10-K.

## Roadmap

- Week 2 — embeddings + sqlite-vec + `ask_filing` tool (single-document RAG).
- Week 3 — `ask_corpus` tool with retrieval across the starter universe.
- Week 4 — ragas eval harness + hand-curated golden set + metrics published in this README.

## Quick start

```
pip install -e .
filings-analyst ingest --tickers AAPL
filings-analyst show-sections <accession_no> AAPL
```

The first `ingest` run will fetch one or more 10-Ks from SEC EDGAR and cache them under `~/.filings_analyst_cache/`. The SEC requires every client to identify itself with a real email — the default User-Agent uses mine. Override via `FILINGS_ANALYST_USER_AGENT` if you fork.

## Connecting to Claude Desktop / Cursor

Register the server in your MCP client config. For Claude Desktop, add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "filings-analyst": {
      "command": "filings-analyst",
      "args": ["serve-mcp"]
    }
  }
}
```

Cursor uses an equivalent `mcpServers` block in its settings. The server speaks the standard MCP stdio transport.

## AI providers

Five backends are configurable via the `LLM_PROVIDER` env var (deterministic tools in week 1 don't use the LLM yet; this matters once RAG lands):

- `auto` (default) — tries Anthropic API → OpenAI API → Claude CLI in order.
- `anthropic_api` — requires `ANTHROPIC_API_KEY`. Uses `claude-haiku-4-5` by default (`ANTHROPIC_MODEL` overrides).
- `openai_api` — requires `OPENAI_API_KEY`. Uses `gpt-4o-mini` by default (`OPENAI_MODEL` overrides).
- `claude_cli` — uses the local `claude -p` binary. On a Claude Max plan, the $100/month Agent SDK credit (activating 2026-06-15) covers exactly this invocation.
- `none` — disable generation entirely; deterministic tools still work.

Same pluggable-provider pattern as my LSE Buyback Scraper.

## Testing

```
pip install -e ".[dev]"
pytest tests/ -v
```

All tests run offline. SEC EDGAR HTTP calls are mocked via `responses`; subprocess and requests calls for the LLM providers are mocked too. The test suite does not touch the network.

## License

MIT.

## Author

Dan Krawczun. [krawczun.com](https://krawczun.com).

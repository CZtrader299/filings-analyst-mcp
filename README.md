# filings-analyst-mcp

An MCP server for querying SEC EDGAR filings. Built for AI agents and humans.

## Status

Active development — currently **week 2 of 4**. The RAG pipeline now works end-to-end against single filings: ingest a 10-K, embed its sections locally, retrieve top-k chunks for a question, and synthesize a grounded answer with inline citations. Multi-filing retrieval and a formal eval harness land in the next two weeks.

## Why this exists

This is a portfolio project demonstrating three things at once: a working MCP server implementation, a real RAG pipeline with a formal evaluation harness (real precision/recall metrics, not vibes), and the same multi-provider AI architecture used in my LSE Buyback Scraper at [github.com/CZtrader299/lse-buyback-scraper](https://github.com/CZtrader299/lse-buyback-scraper). The differentiator over the average "RAG over 10-Ks" portfolio project is the eval harness — anyone can wire up a vector store, far fewer projects measure their retrieval quality honestly.

## Architecture (current)

```
src/filings_analyst/
  config.py        # env-driven constants (User-Agent, cache dir, model defaults)
  edgar.py         # SEC EDGAR client + on-disk cache helpers
  sections.py      # 10-K section extractor (Business / Risk Factors / MD&A / Financials)
  chunking.py      # deterministic sentence-aware text chunker with overlap
  embeddings.py    # pluggable embedding provider (local sentence-transformers or OpenAI)
  vectorstore.py   # sqlite-vec wrapper for chunk storage + similarity search
  rag.py           # orchestrator: ingest + ask_filing (retrieval + grounded synthesis)
  providers.py     # LLM provider router (Anthropic, OpenAI, Claude CLI)
  mcp_server.py    # MCP server: search_filings, get_filing, extract_section, ask_filing
  cli.py           # ingest / show-sections / ask / serve-mcp subcommands
```

## MCP tools available now

- `search_filings(ticker, form_type="10-K", year=None)` — list recent filings for a ticker from SEC EDGAR.
- `get_filing(accession_no, ticker)` — return metadata and a short preview of a filing's full text (filing must be in the local cache).
- `extract_section(accession_no, ticker, section)` — return one named section ("Business", "Risk Factors", "MD&A", "Financial Statements") from a cached 10-K.
- `ask_filing(accession_no, ticker, question, k=6)` — RAG-backed Q&A over a single ingested filing. Returns a grounded answer plus the top-k cited chunks (section, chunk index, similarity score, excerpt). Filing must be ingested first via `filings-analyst ingest`.

## Roadmap

- Week 2 (done) — chunking + embeddings + sqlite-vec + `ask_filing` tool (single-document RAG).
- Week 3 — `ask_corpus` tool for multi-filing retrieval across the starter universe, plus a provider-quality comparison (Anthropic / OpenAI / Claude CLI on the same prompts).
- Week 4 — ragas eval harness + hand-curated golden set + metrics published in this README.

## Quick start

```
pip install -e ".[embeddings]"
filings-analyst ingest AAPL
# Note the accession_no printed by ingest, then:
filings-analyst ask <accession_no> AAPL "What did management say about AI in this year's MD&A?"
```

The first `ingest` run downloads the most recent AAPL 10-K from SEC EDGAR, extracts its named sections, chunks them, and embeds the chunks locally with `all-MiniLM-L6-v2` (~80MB; downloaded once into the user-level Hugging Face cache, not into this repo). Subsequent runs reuse both caches.

Filings are cached under `~/.filings_analyst_cache/`. The embedded chunk vectors live in `~/.filings_analyst_cache/vectors.db` (sqlite-vec). The SEC requires every client to identify itself with a real email — the default User-Agent uses mine. Override via `FILINGS_ANALYST_USER_AGENT` if you fork.

## How RAG works here

The pipeline is intentionally boring and inspectable: each cached 10-K is parsed into four named sections (Business / Risk Factors / MD&A / Financial Statements), each section is split into ~500-token chunks with ~50-token overlap at sentence boundaries, and each chunk is embedded with `all-MiniLM-L6-v2` running locally. Chunks land in a `sqlite-vec` virtual table with metadata in a sibling sqlite table. At query time, the question is embedded with the same model, the top-k chunks for that filing are pulled via cosine similarity, and a grounded synthesis prompt asks the LLM to answer using only those excerpts and to cite them inline as `[Section §chunk_idx]` markers a reader can verify against the returned `cited_chunks` list. The default LLM is the local Claude CLI (`claude -p`), matching the multi-provider pattern from my LSE scraper; Anthropic or OpenAI API keys take priority when present. No API key is required for the default local-embedding path — the entire ingest + retrieval flow runs offline once the model is downloaded.

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

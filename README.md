# filings-analyst-mcp

An MCP server for querying SEC EDGAR filings. Built for AI agents and humans.

## Status

Active development — currently **week 3 of 4 — corpus-wide retrieval working across multiple filings.** The RAG pipeline runs end-to-end against both single filings (`ask_filing`) and the entire ingested corpus (`ask_corpus`). The starter corpus spans AAPL, MSFT, JPM, BAC, and XOM. A formal evaluation harness lands in week 4.

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
  mcp_server.py    # MCP server: search_filings, get_filing, extract_section, ask_filing, ask_corpus
  cli.py           # ingest / show-sections / ask / ask-corpus / serve-mcp subcommands
```

## MCP tools available now

- `search_filings(ticker, form_type="10-K", year=None)` — list recent filings for a ticker from SEC EDGAR.
- `get_filing(accession_no, ticker)` — return metadata and a short preview of a filing's full text (filing must be in the local cache).
- `extract_section(accession_no, ticker, section)` — return one named section ("Business", "Risk Factors", "MD&A", "Financial Statements") from a cached 10-K.
- `ask_filing(accession_no, ticker, question, k=6)` — RAG-backed Q&A over a single ingested filing. Returns a grounded answer plus the top-k cited chunks (section, chunk index, similarity score, excerpt). Filing must be ingested first via `filings-analyst ingest`.
- `ask_corpus(question, tickers=None, accession_nos=None, k=8)` — RAG-backed Q&A across every ingested filing, optionally filtered by ticker or accession. Citations are ticker-tagged (`[AAPL Risk Factors §3]`) so the reader can tell which filing each excerpt came from, and the response includes a `filings_searched` manifest of the filings whose chunks actually contributed.

## Roadmap

- Week 2 (done) — chunking + embeddings + sqlite-vec + `ask_filing` tool (single-document RAG).
- Week 3 (done) — `ask_corpus` tool for multi-filing retrieval across the starter universe (AAPL/MSFT/JPM/BAC/XOM); MD&A section extraction hardened for real-world heading variants (Apple-style curly apostrophes and non-breaking spaces, Microsoft-style cross-line splits); honest provider comparison documented below.
- Week 4 — ragas eval harness + hand-curated golden set + metrics published in this README.

## Quick start

```
pip install -e ".[embeddings]"
filings-analyst ingest AAPL
# Note the accession_no printed by ingest, then ask a question
# against one filing:
filings-analyst ask <accession_no> AAPL "What did management say about AI in this year's MD&A?"

# Or ingest the full starter corpus and ask a cross-filing question:
filings-analyst ingest --tickers AAPL,MSFT,JPM,BAC,XOM
filings-analyst ask-corpus "Which of these companies discusses AI risks most prominently in their 10-K filings?"
```

The first `ingest` run downloads the most recent AAPL 10-K from SEC EDGAR, extracts its named sections, chunks them, and embeds the chunks locally with `all-MiniLM-L6-v2` (~80MB; downloaded once into the user-level Hugging Face cache, not into this repo). Subsequent runs reuse both caches.

Filings are cached under `~/.filings_analyst_cache/`. The embedded chunk vectors live in `~/.filings_analyst_cache/vectors.db` (sqlite-vec). The SEC requires every client to identify itself with a real email — the default User-Agent uses mine. Override via `FILINGS_ANALYST_USER_AGENT` if you fork.

## How RAG works here

The pipeline is intentionally boring and inspectable: each cached 10-K is parsed into four named sections (Business / Risk Factors / MD&A / Financial Statements), each section is split into ~500-token chunks with ~50-token overlap at sentence boundaries, and each chunk is embedded with `all-MiniLM-L6-v2` running locally. Chunks land in a `sqlite-vec` virtual table with metadata in a sibling sqlite table. At query time, the question is embedded with the same model, top-k chunks are pulled via cosine similarity, and a grounded synthesis prompt asks the LLM to answer using only those excerpts and to cite them inline so a reader can verify against the returned `cited_chunks` list. Two retrieval modes are supported:

- **Single-filing** (`ask_filing`) — restricts retrieval to one accession number; citations look like `[Section §chunk_idx]`.
- **Corpus-wide** (`ask_corpus`) — retrieves across every ingested filing (optionally filtered by ticker or accession); citations are ticker-tagged like `[AAPL Risk Factors §3]` and the response includes a `filings_searched` manifest of the filings whose chunks actually contributed.

The default LLM is the local Claude CLI (`claude -p`), matching the multi-provider pattern from my LSE scraper; Anthropic or OpenAI API keys take priority when present. No API key is required for the default local-embedding path — the entire ingest + retrieval flow runs offline once the model is downloaded.

## Provider comparison

The same question, answered by three different providers against the same retrieved context. Run yourself to verify — these are real, reproducible outputs.

Question: **"What did Apple say about AI risks in its most recent 10-K?"** Same retrieval (k=6 over the AAPL 2025 10-K, accession `0000320193-25-000079`) for all three rows.

| Provider | Answer (first ~150 chars) | Latency | Cost |
|----------|---------------------------|---------|------|
| `claude_cli` (Haiku via `claude -p`) | "The provided excerpts do not contain that information." | 6.62s | via Claude Max subscription credit |
| `openai_api` (gpt-4o-mini, temp 0) | **TBD — set `OPENAI_API_KEY` to populate** | n/a | n/a |
| `regex` (no synthesis, retrieved chunks only) | "Here are the most relevant excerpts:\n\n- [Risk Factors chunk_idx=30] Efforts by the Company to advance its business and values, or achieve its goals an..." | <1ms | $0 |

Methodology: same retrieval (k=6) for all three rows. Claude via `claude -p` on a Max 5x subscription (Agent SDK credit activates 2026-06-15; until then the call goes against the standard subscription quota). OpenAI via `gpt-4o-mini` at temperature 0 would compute actual token cost; this dev box has no `OPENAI_API_KEY` set, so that row is honestly blank rather than fabricated. Regex baseline returns the retrieved chunks verbatim with no synthesis at all (a `Here are the most relevant excerpts:` header + bullet list).

What the comparison shows, honestly: on this question, Claude correctly refuses to answer ("the provided excerpts do not contain that information") because the top-6 retrieved chunks from local MiniLM embeddings don't actually contain Apple's AI-specific risk language — the most-similar Risk Factors chunks were about competition and IP licensing, not AI per se. The regex baseline is faster and free but offloads all the reading to the human. This is a useful real-world finding: at k=6 with MiniLM-L6-v2 embeddings, retrieval quality is the binding constraint, not the LLM. The week-4 eval harness will quantify that on a golden set rather than leaving it as a single-question anecdote.

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

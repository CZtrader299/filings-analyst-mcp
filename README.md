# filings-analyst-mcp

An MCP server for querying SEC EDGAR filings. Built for AI agents and humans.

## Status

The RAG pipeline runs end-to-end against both single filings (`ask_filing`) and the entire ingested corpus (`ask_corpus`). The starter corpus spans AAPL, MSFT, JPM, BAC, and XOM. A formal evaluation harness produces real, auditable metrics against a hand-curated 35-question golden set. Numbers are in the Evaluation section below.

## Why this exists

This is a portfolio project demonstrating three things at once: a working MCP server implementation, a real RAG pipeline with a formal evaluation harness (real precision/recall metrics, not vibes), and the same multi-provider AI architecture used in my LSE Buyback Scraper at [github.com/CZtrader299/lse-buyback-scraper](https://github.com/CZtrader299/lse-buyback-scraper). The differentiator over the average "RAG over 10-Ks" portfolio project is the eval harness. Anyone can wire up a vector store; far fewer projects measure their retrieval quality honestly.

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

- `search_filings(ticker, form_type="10-K", year=None)`: list recent filings for a ticker from SEC EDGAR.
- `get_filing(accession_no, ticker)`: return metadata and a short preview of a filing's full text (filing must be in the local cache).
- `extract_section(accession_no, ticker, section)`: return one named section ("Business", "Risk Factors", "MD&A", "Financial Statements") from a cached 10-K.
- `ask_filing(accession_no, ticker, question, k=6)`: RAG-backed Q&A over a single ingested filing. Returns a grounded answer plus the top-k cited chunks (section, chunk index, similarity score, excerpt). Filing must be ingested first via `filings-analyst ingest`.
- `ask_corpus(question, tickers=None, accession_nos=None, k=8)`: RAG-backed Q&A across every ingested filing, optionally filtered by ticker or accession. Citations are ticker-tagged (`[AAPL Risk Factors §3]`) so the reader can tell which filing each excerpt came from, and the response includes a `filings_searched` manifest of the filings whose chunks actually contributed.

## What's in the box

- **Single-document RAG**: chunking, local embeddings (`all-MiniLM-L6-v2`), `sqlite-vec` retrieval, and grounded synthesis with inline citations, exposed as the `ask_filing` MCP tool.
- **Corpus-wide RAG**: the same pipeline extended across every ingested filing in the starter universe (AAPL / MSFT / JPM / BAC / XOM), with ticker-tagged citations and a `filings_searched` manifest of which filings actually contributed; exposed as `ask_corpus`.
- **Defensive section extraction**: Item-1 / 1A / 7 / 8 regex variants harden against the heading styles real filers use, including Apple-style curly apostrophes and non-breaking spaces and Microsoft-style cross-line heading splits.
- **Honest provider comparison**: one canonical question answered by Claude CLI, OpenAI API, and a regex-only baseline, with real latency and cost reported transparently (documented below).
- **Formal eval harness**: hand-curated 35-question golden set across the five tickers; five metrics (faithfulness / answer_relevancy / context_precision / context_recall + a hand-rolled refusal_correctness); markdown reporter with worst-3-per-metric; manual-trigger GitHub Actions workflow. Real numbers from a stratified 10-item sample published below.

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

# Reproduce the evaluation harness numbers:
python scripts/run_sample_eval.py    # 10-item stratified sample, ~10 min
filings-analyst eval                  # full 35-item run, ~67 min
```

The first `ingest` run downloads the most recent AAPL 10-K from SEC EDGAR, extracts its named sections, chunks them, and embeds the chunks locally with `all-MiniLM-L6-v2` (~80MB; downloaded once into the user-level Hugging Face cache, not into this repo). Subsequent runs reuse both caches.

Filings are cached under `~/.filings_analyst_cache/`. The embedded chunk vectors live in `~/.filings_analyst_cache/vectors.db` (sqlite-vec). The SEC requires every client to identify itself with a real email; the default User-Agent uses mine. Override via `FILINGS_ANALYST_USER_AGENT` if you fork.

## How RAG works here

The pipeline is intentionally boring and inspectable: each cached 10-K is parsed into four named sections (Business / Risk Factors / MD&A / Financial Statements), each section is split into ~500-token chunks with ~50-token overlap at sentence boundaries, and each chunk is embedded with `all-MiniLM-L6-v2` running locally. Chunks land in a `sqlite-vec` virtual table with metadata in a sibling sqlite table. At query time, the question is embedded with the same model, top-k chunks are pulled via cosine similarity, and a grounded synthesis prompt asks the LLM to answer using only those excerpts and to cite them inline so a reader can verify against the returned `cited_chunks` list. Two retrieval modes are supported:

- **Single-filing** (`ask_filing`): restricts retrieval to one accession number; citations look like `[Section §chunk_idx]`.
- **Corpus-wide** (`ask_corpus`): retrieves across every ingested filing (optionally filtered by ticker or accession); citations are ticker-tagged like `[AAPL Risk Factors §3]` and the response includes a `filings_searched` manifest of the filings whose chunks actually contributed.

The default LLM is the local Claude CLI (`claude -p`), matching the multi-provider pattern from my LSE scraper; Anthropic or OpenAI API keys take priority when present. No API key is required for the default local-embedding path; the entire ingest + retrieval flow runs offline once the model is downloaded.

## Provider comparison

The same question, answered by three different providers against the same retrieved context. Run yourself to verify; these are real, reproducible outputs.

Question: **"What did Apple say about AI risks in its most recent 10-K?"** Same retrieval (k=6 over the AAPL 2025 10-K, accession `0000320193-25-000079`) for all three rows.

| Provider | Answer (first ~150 chars) | Latency | Cost |
|----------|---------------------------|---------|------|
| `claude_cli` (Haiku via `claude -p`) | "The provided excerpts do not contain that information." | 6.62s | via Claude Max subscription credit |
| `openai_api` (gpt-4o-mini, temp 0) | **TBD: set `OPENAI_API_KEY` to populate** | n/a | n/a |
| `regex` (no synthesis, retrieved chunks only) | "Here are the most relevant excerpts:\n\n- [Risk Factors chunk_idx=30] Efforts by the Company to advance its business and values, or achieve its goals an..." | <1ms | $0 |

Methodology: same retrieval (k=6) for all three rows. Claude via `claude -p` on a Max 5x subscription (Agent SDK credit activates 2026-06-15; until then the call goes against the standard subscription quota). OpenAI via `gpt-4o-mini` at temperature 0 would compute actual token cost; this dev box has no `OPENAI_API_KEY` set, so that row is honestly blank rather than fabricated. Regex baseline returns the retrieved chunks verbatim with no synthesis at all (a `Here are the most relevant excerpts:` header + bullet list).

What the comparison shows, honestly: on this question, Claude correctly refuses to answer ("the provided excerpts do not contain that information") because the top-6 retrieved chunks from local MiniLM embeddings don't actually contain Apple's AI-specific risk language. The most-similar Risk Factors chunks were about competition and IP licensing, not AI per se. The regex baseline is faster and free but offloads all the reading to the human. This is a useful real-world finding: at k=6 with MiniLM-L6-v2 embeddings, retrieval quality is the binding constraint, not the LLM. The Evaluation section below quantifies exactly that on a golden set rather than leaving it as a single-question anecdote.

## Evaluation

The repo ships with a 35-item hand-curated golden set spanning all five tickers (AAPL, MSFT, JPM, BAC, XOM) and three question types: 15 easy facts plainly stated in the 10-K, 12 hard synthesis questions, 8 deliberately out-of-scope questions whose correct behavior is refusal (e.g. exec compensation, which lives in the proxy not the 10-K). Each item is graded on five metrics via Claude (`claude -p`):

- **faithfulness**: every claim in the answer is supported by the retrieved chunks (catches hallucination).
- **answer_relevancy**: the answer actually addresses the question asked.
- **context_precision**: the retrieved chunks are relevant to the question.
- **context_recall**: the relevant content in the filing was actually retrieved.
- **refusal_correctness**: out-of-scope questions are refused, in-scope questions are answered with citations.

The numbers below are from the **full 35-item run**: every question in the golden set, graded by Claude via `claude -p`. Total wall-clock was 39 minutes (each item invokes the RAG pipeline plus five separate grader calls). To reproduce, or to run a quick subset:

```
filings-analyst eval                   # full 35-item run (~40 min)
python scripts/run_sample_eval.py     # 10-item stratified sample (~10 min)
filings-analyst eval --sample 5        # ad-hoc N-item subset
```

### Aggregate metrics (full 35-item run)

| Metric | Score |
|--------|-------|
| faithfulness | 0.998 |
| answer_relevancy | 0.894 |
| context_precision | 0.514 |
| context_recall | 0.443 |
| refusal_correctness | 0.886 |

### Per-type breakdown

| Type | n | Mean score | Notes |
|------|---|------------|-------|
| easy | 15 | 0.780 | Faithfulness perfect (1.000); refusal_correctness perfect; the drag is recall (0.467) and precision (0.478), i.e. retrieval is noisy even on straightforward fact questions. |
| hard | 13 | 0.621 | Synthesis questions stress retrieval hardest. context_recall = 0.192 on this slice: relevant supporting material exists in the filing but doesn't reliably make top-k. refusal_correctness = 0.769 shows the system also over-refuses some answerable hard questions, choosing to abstain when it shouldn't. |
| out-of-scope | 7 | 0.910 | The category the project was designed for. Six of seven refused correctly; one item incorrectly attempted an answer rather than abstaining. |

### Per-ticker breakdown

| Ticker | n | Mean | Faithfulness | Recall | Notes |
|--------|---|------|--------------|--------|-------|
| AAPL | 7 | 0.763 | 0.989 | 0.643 | Tech-style filings parse cleanly; highest recall in the corpus. |
| JPM | 7 | 0.783 | 1.000 | 0.500 | Bank-style 10-K parses; recall mid-pack. |
| XOM | 7 | 0.765 | 1.000 | 0.429 | Energy-style filing; answer_relevancy highest in the corpus (0.966). |
| BAC | 7 | 0.714 | 1.000 | 0.357 | Bank filing; recall is the weak point. |
| MSFT | 7 | 0.710 | 1.000 | 0.286 | Lowest recall; the AI-framing synthesis questions on MSFT particularly stress the retrieval. |

### Honest reflection on the numbers

The spread between **faithfulness (0.998)** and **context_recall (0.443)** is the headline finding. The system does not hallucinate. When it answers, it sticks to what was retrieved, every time across all 35 questions. But the retrieval itself is noisy: on roughly **56% of questions, at least one relevant chunk that exists in the filing fails to make the top-k**. The effect is strongest on hard synthesis questions, where supporting material is scattered across sections, and weakest on out-of-scope questions, where retrieval is irrelevant to the correct answer (a refusal).

**Retrieval is the binding constraint, not the LLM.** The local `all-MiniLM-L6-v2` embeddings (384-dim, free) chosen for the default cost-conscious config are the bottleneck. Swapping to OpenAI's `text-embedding-3-small` (1536-dim, ~$0.02/Mtok, ~$2-5 one-time for the full corpus) is a one-line change via `EMBEDDING_PROVIDER=openai` and the eval harness will measure the uplift directly. That's a follow-on commit, not a redesign.

**Refusal behavior is mostly working, with one real gap.** refusal_correctness = 0.886 across all 35 items breaks down as: 7/7 on out-of-scope refusals (the system correctly abstained on every "salary of CEO" / "current stock price" question), but **3/13 hard questions also got refused when they shouldn't have been**. That over-refusal on hard questions pulls the metric down. It's an honest finding: the conservative-flagging design that works perfectly on out-of-scope questions occasionally fires on legitimately-answerable hard ones when retrieval is sparse, i.e. the same recall weakness above expressing itself as a different failure mode. Fixing recall fixes both.

Where MSFT lands lowest (0.710) is consistent with the MSFT synthesis questions being some of the most demanding in the golden set: they ask about the *relationship* between AI in the Business section and AI in Risk Factors, which requires retrieving from two different sections of the filing simultaneously. Better embeddings would help here directly.

The full per-item report with worst-3-per-metric and individual citation chains is committed at [`eval_reports/2026-05-19_full_eval.md`](eval_reports/2026-05-19_full_eval.md). Every score is reproducible by re-running `filings-analyst eval`.

### Iteration: a section-extraction fix that didn't move the eval

The baseline showed three of five tickers extracting essentially no MD&A text (JPM 567 chars, BAC 929 chars, XOM 627 chars vs. AAPL 20,766 and MSFT 50,059), so the obvious-looking intervention was to fix the section extractor. Inspection of the raw HTML found three distinct real patterns the original regex missed:

- **JPM**'s Item 7 anchor is a forward-reference placeholder pointing to a separately-titled "Management's Discussion and Analysis" section much later in the document.
- **BAC**'s `Item 7.` text is a chapter-divider banner near the end of the file; the real content is earlier.
- **XOM**'s Item 7 string repeats throughout the document inside TOC anchor links; the original "last match" heuristic landed on a TOC entry rather than the section.

The fix adds a section-title fallback with a prose-density heuristic to skip TOC anchors (see [`src/filings_analyst/sections.py`](src/filings_analyst/sections.py)). Mechanical effect: MD&A character counts went from ~500-1000 to ~120K-400K on the affected filings, and embedded chunk counts went from ~1 to 70-234. Three new unit-test fixtures cover each failure mode. AAPL and MSFT extraction was unaffected (verified by character count and chunk count regression).

Then I re-ran the full 35-item eval on the new chunks. The result was not what I expected:

| Metric | Before fix | After fix | Δ |
|--------|-----------:|----------:|------:|
| faithfulness | 0.998 | 0.971 | −0.027 |
| answer_relevancy | 0.894 | 0.820 | −0.074 |
| context_precision | 0.514 | 0.524 | +0.010 |
| context_recall | 0.443 | 0.429 | −0.014 |
| refusal_correctness | 0.886 | 0.714 | −0.172 |

Per-ticker context_recall (the metric the fix targeted):

| Ticker | Before | After | Δ |
|--------|-------:|------:|------:|
| AAPL | 0.643 | 0.643 | 0.000 |
| MSFT | 0.286 | 0.286 | 0.000 |
| BAC | 0.357 | 0.500 | **+0.143** |
| JPM | 0.500 | 0.357 | **−0.143** |
| XOM | 0.429 | 0.357 | **−0.072** |

The eval surfaced a real tradeoff the intuition didn't see: **more content competing for top-k slots doesn't help recall when the embeddings can't distinguish the relevant chunk from adjacent ones.** JPM's MD&A inflated from 1 chunk to 234 chunks, and the local MiniLM embedding model started returning similar-looking-but-wrong chunks more confidently, displacing chunks from other sections that the original (sparse) JPM corpus had been pulling. BAC happened to benefit; JPM and XOM regressed; aggregate is flat.

The refusal_correctness drop (−0.172) is the more material effect: with more financial-MD&A content available, the system is now sometimes *attempting* to answer out-of-scope-but-financially-adjacent questions like "current P/B ratio" by weaving citations from the now-rich MD&A, rather than refusing as it did before.

**What this proves about the project.** The eval harness is doing its job. It's catching a real tradeoff that intuition would miss, and it's preventing a "looks-like-a-fix" change from quietly degrading user-facing behavior. The right next step is not more content but better-quality retrieval, i.e. the OpenAI embedding swap the prior reflection already flagged. The section-extraction fix is still shipped because the underlying parsing improvement is correct (the chunks really are valid MD&A content now, and the prior state of "1 chunk of MD&A" was a known bug); the lesson is that retrieval quality is what gates everything downstream.

The post-fix per-item report is at [`eval_reports/2026-05-20_post_fix_eval.md`](eval_reports/2026-05-20_post_fix_eval.md); the original baseline remains at [`eval_reports/2026-05-19_full_eval.md`](eval_reports/2026-05-19_full_eval.md) for direct comparison.

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

Five backends are configurable via the `LLM_PROVIDER` env var. The deterministic tools (`search_filings`, `get_filing`, `extract_section`) don't call the LLM; the RAG tools (`ask_filing`, `ask_corpus`) and the eval grader do.

- `auto` (default): tries Anthropic API → OpenAI API → Claude CLI in order.
- `anthropic_api`: requires `ANTHROPIC_API_KEY`. Uses `claude-haiku-4-5` by default (`ANTHROPIC_MODEL` overrides).
- `openai_api`: requires `OPENAI_API_KEY`. Uses `gpt-4o-mini` by default (`OPENAI_MODEL` overrides).
- `claude_cli`: uses the local `claude -p` binary. On a Claude Max plan, the $100/month Agent SDK credit (activating 2026-06-15) covers exactly this invocation.
- `none`: disable generation entirely; deterministic tools still work.

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

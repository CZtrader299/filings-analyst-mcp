"""Retrieval-augmented generation orchestrator for single 10-K filings.

Ties together:

* :mod:`filings_analyst.edgar` — cached filing HTML.
* :mod:`filings_analyst.sections` — named section extraction.
* :mod:`filings_analyst.chunking` — overlap-aware text splits.
* :mod:`filings_analyst.embeddings` — pluggable embedding backend.
* :mod:`filings_analyst.vectorstore` — sqlite-vec retrieval.
* :mod:`filings_analyst.providers` — pluggable LLM backend for synthesis.

The orchestrator deliberately keeps three distinct verbs:

* ``ingest_filing`` — chunk + embed + store. Idempotent (re-ingestion
  wipes the prior chunks first so re-runs don't double-count).
* ``ask_filing`` — retrieve + synthesize a grounded answer for one
  filing, with citations.
* ``ask_corpus`` — retrieve + synthesize across every ingested filing
  (optionally filtered by ticker or accession), with citations that
  include the ticker so a reader can tell which filing each excerpt
  came from.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from . import chunking, config, edgar, sections
from .embeddings import Embedder
from .providers import LLMProvider


_SYSTEM_PROMPT = (
    "You are a careful financial analyst answering questions about SEC 10-K "
    "filings. You answer ONLY from the provided context excerpts. If the "
    "answer is not in the context, say so explicitly — never invent figures "
    "or quotes. When you cite, use inline references in the form "
    "`[Section §chunk_idx]` (e.g., `[MD&A §3]`) so a reader can verify "
    "against the listed chunks. Keep answers concise."
)


def _default_db_path() -> Path:
    return config.CACHE_DIR / "vectors.db"


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a numbered, labelled prompt block."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.get("section", "?")
        idx = chunk.get("chunk_idx", "?")
        text = chunk.get("text", "").strip()
        lines.append(f"[{i}] Section: {section} §{idx}\n{text}")
    return "\n\n".join(lines)


def _build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Compose the synthesis prompt from question + context block."""
    context = _build_context_block(chunks)
    return (
        "Use ONLY the context excerpts below to answer the question. If the "
        "answer is not present in the context, reply: \"The provided "
        "excerpts do not contain that information.\"\n\n"
        "Cite excerpts inline using `[Section §chunk_idx]` notation matching "
        "the labels shown.\n\n"
        f"=== Context ===\n{context}\n\n"
        f"=== Question ===\n{question}\n\n"
        "=== Answer ==="
    )


def _build_corpus_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks with ticker-prefixed citation labels.

    Each chunk gets a label like ``[AAPL Risk Factors §3]`` so that when
    the LLM cites inline, the reader can tell which filing the excerpt
    came from. Without the ticker prefix, citations across a multi-filing
    corpus are ambiguous (every filing has a ``Risk Factors §0``).
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        ticker = chunk.get("ticker", "?")
        section = chunk.get("section", "?")
        idx = chunk.get("chunk_idx", "?")
        text = chunk.get("text", "").strip()
        lines.append(f"[{i}] {ticker} | Section: {section} §{idx}\n{text}")
    return "\n\n".join(lines)


def _build_corpus_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Compose the multi-filing synthesis prompt.

    Differs from the single-filing prompt only in the citation format
    instruction: corpus citations must include the ticker so a reader
    knows which company each fact came from.
    """
    context = _build_corpus_context_block(chunks)
    return (
        "Use ONLY the context excerpts below to answer the question. If "
        "the answer is not present in the context, reply: \"The provided "
        "excerpts do not contain that information.\"\n\n"
        "Cite excerpts inline using `[TICKER Section §chunk_idx]` notation "
        "matching the labels shown (e.g., `[AAPL Risk Factors §3]`). Each "
        "factual claim must carry a citation so the reader can tell which "
        "filing it came from.\n\n"
        f"=== Context ===\n{context}\n\n"
        f"=== Question ===\n{question}\n\n"
        "=== Answer ==="
    )


class FilingRAG:
    """High-level RAG entry point for one-filing-scope question answering."""

    def __init__(
        self,
        *,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLMProvider] = None,
        db_path: Optional[Path] = None,
    ):
        self.embedder = embedder if embedder is not None else Embedder()
        self.llm = llm if llm is not None else LLMProvider()
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        # Vector store is opened lazily so a fresh ``FilingRAG()`` doesn't
        # spin up sqlite-vec if the caller only wants ``llm.available``.
        self._store = None

    # --- Store handle ----------------------------------------------------

    def _open_store(self):
        if self._store is None:
            # Import locally so the dependency error is raised here rather
            # than at module import time.
            from .vectorstore import VectorStore

            self._store = VectorStore(self.db_path, dim=self.embedder.dim)
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # --- Ingestion -------------------------------------------------------

    def ingest_filing(
        self,
        accession_no: str,
        ticker: str,
        *,
        target_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> dict[str, Any]:
        """Chunk + embed + store one already-cached filing.

        Idempotent: existing chunks for this accession are deleted first.
        """
        start = time.time()
        try:
            html = edgar.load_cached_filing_text(ticker, accession_no)
        except FileNotFoundError as exc:
            return {
                "accession_no": accession_no,
                "ticker": ticker.upper(),
                "chunks_added": 0,
                "error": (
                    f"Filing not in cache: {exc}. "
                    "Run `filings-analyst ingest --tickers <TICKER>` first."
                ),
            }

        # Best-effort metadata fetch so we can tag chunks with filing_date.
        # Older cached filings without metadata.json shouldn't break ingest,
        # so we swallow the FileNotFoundError silently.
        filing_date: Optional[str] = None
        try:
            meta = edgar.load_cached_metadata(ticker, accession_no)
            filing_date = meta.get("filing_date") or meta.get("period_end")
        except FileNotFoundError:
            filing_date = None

        extracted = sections.extract_sections(html)
        chunk_records = chunking.chunk_sections(
            extracted,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not chunk_records:
            return {
                "accession_no": accession_no,
                "ticker": ticker.upper(),
                "chunks_added": 0,
                "error": "No sections found in this filing",
            }

        texts = [r["text"] for r in chunk_records]
        embeddings = self.embedder.embed_texts(texts)
        if embeddings is None:
            return {
                "accession_no": accession_no,
                "ticker": ticker.upper(),
                "chunks_added": 0,
                "error": "Embedding backend failed (check logs above)",
            }

        for rec, vec in zip(chunk_records, embeddings):
            rec["accession_no"] = accession_no
            rec["ticker"] = ticker.upper()
            rec["embedding"] = vec
            if filing_date:
                rec["filing_date"] = filing_date

        store = self._open_store()
        store.delete_filing(accession_no)
        added = store.add_chunks(chunk_records)

        # Per-section breakdown for the CLI / caller.
        by_section: dict[str, int] = {}
        for rec in chunk_records:
            by_section[rec["section"]] = by_section.get(rec["section"], 0) + 1

        return {
            "accession_no": accession_no,
            "ticker": ticker.upper(),
            "chunks_added": added,
            "chunks_by_section": by_section,
            "embedding_dim": self.embedder.dim,
            "embedding_provider": self.embedder.provider,
            "elapsed_sec": round(time.time() - start, 2),
        }

    # --- Query -----------------------------------------------------------

    def ask_filing(
        self,
        question: str,
        *,
        accession_no: str,
        ticker: str,
        k: int = 6,
    ) -> dict[str, Any]:
        """Retrieve + synthesize a grounded answer for one filing."""
        if not self.llm.available:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "provider": "none",
                "error": (
                    "No LLM provider available — set ANTHROPIC_API_KEY, "
                    "OPENAI_API_KEY, or install the Claude CLI."
                ),
            }

        query_vec = self.embedder.embed_query(question)
        if query_vec is None:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "provider": self.llm.provider,
                "error": "Embedding backend failed to embed the question",
            }

        store = self._open_store()
        hits = store.search(
            query_vec, k=k, filter_accession_no=accession_no
        )
        if not hits:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "provider": self.llm.provider,
                "error": (
                    "No chunks indexed for this filing. "
                    "Run `filings-analyst ingest <accession> <ticker>` first."
                ),
            }

        prompt = _build_prompt(question, hits)
        answer = self.llm.generate(prompt, max_tokens=1024, system=_SYSTEM_PROMPT)

        cited = [
            {
                "section": h["section"],
                "chunk_idx": h["chunk_idx"],
                "text": h["text"],
                "score": h["score"],
            }
            for h in hits
        ]
        result: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "cited_chunks": cited,
            "provider": self.llm.provider,
        }
        if answer is None:
            result["error"] = "LLM call returned no content (see warnings above)"
        return result

    # --- Multi-filing query ---------------------------------------------

    def ask_corpus(
        self,
        question: str,
        *,
        tickers: Optional[list[str]] = None,
        accession_nos: Optional[list[str]] = None,
        k: int = 8,
    ) -> dict[str, Any]:
        """Answer a question by retrieving across ALL ingested filings.

        Args:
            question: Natural-language question.
            tickers: Optional filter — restrict retrieval to filings for
                these tickers (case-insensitive).
            accession_nos: Optional filter — restrict to specific
                accession numbers.
            k: Number of chunks to retrieve.

        Returns a dict with:

        * ``question`` — the input question.
        * ``answer`` — the synthesized answer, or ``None`` on failure.
        * ``cited_chunks`` — each retrieved chunk, including
          ``ticker``, ``accession_no``, ``filing_date``, ``section``,
          ``chunk_idx``, ``text``, ``score``.
        * ``filings_searched`` — the unique filings whose chunks
          appeared in the retrieved set, so the reader knows which
          filings actually contributed (not the whole corpus).
        * ``provider`` — the LLM provider used.
        """
        if not self.llm.available:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "filings_searched": [],
                "provider": "none",
                "error": (
                    "No LLM provider available — set ANTHROPIC_API_KEY, "
                    "OPENAI_API_KEY, or install the Claude CLI."
                ),
            }

        store = self._open_store()
        if store.count() == 0:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "filings_searched": [],
                "provider": self.llm.provider,
                "error": (
                    "No filings ingested. "
                    "Run `filings-analyst ingest <ticker>` first."
                ),
            }

        query_vec = self.embedder.embed_query(question)
        if query_vec is None:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "filings_searched": [],
                "provider": self.llm.provider,
                "error": "Embedding backend failed to embed the question",
            }

        hits = store.search(
            query_vec,
            k=k,
            filter_tickers=tickers,
            filter_accession_nos=accession_nos,
        )
        if not hits:
            return {
                "question": question,
                "answer": None,
                "cited_chunks": [],
                "filings_searched": [],
                "provider": self.llm.provider,
                "error": (
                    "No matching chunks found in the corpus for the "
                    "given filters."
                ),
            }

        prompt = _build_corpus_prompt(question, hits)
        answer = self.llm.generate(prompt, max_tokens=1024, system=_SYSTEM_PROMPT)

        cited = [
            {
                "ticker": h["ticker"],
                "accession_no": h["accession_no"],
                "filing_date": h.get("filing_date", ""),
                "section": h["section"],
                "chunk_idx": h["chunk_idx"],
                "text": h["text"],
                "score": h["score"],
            }
            for h in hits
        ]

        # filings_searched: deduplicated by (ticker, accession_no) and
        # ordered by first appearance in the retrieved set so the most
        # relevant filing leads.
        seen: set[tuple[str, str]] = set()
        filings_searched: list[dict[str, Any]] = []
        for h in hits:
            key = (h["ticker"], h["accession_no"])
            if key in seen:
                continue
            seen.add(key)
            filings_searched.append(
                {
                    "ticker": h["ticker"],
                    "accession_no": h["accession_no"],
                    "filing_date": h.get("filing_date", ""),
                }
            )

        result: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "cited_chunks": cited,
            "filings_searched": filings_searched,
            "provider": self.llm.provider,
        }
        if answer is None:
            result["error"] = "LLM call returned no content (see warnings above)"
        return result


__all__ = ("FilingRAG",)

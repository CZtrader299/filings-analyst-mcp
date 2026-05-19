"""Tests for the RAG orchestrator.

Both the embedder and the LLM are mocked so the tests stay offline and
fast. The vector store is real (sqlite-vec in-memory via tmp_path) so we
verify the retrieval surface end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sqlite_vec")

from filings_analyst import edgar, rag


FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.html"


def _seed_cache(tmp_path: Path, ticker: str, accession_no: str) -> None:
    target = tmp_path / ticker / "10-K" / accession_no
    target.mkdir(parents=True)
    (target / "full-text.html").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "accession_no": accession_no,
                "ticker": ticker,
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "period_end": "2024-09-28",
            }
        ),
        encoding="utf-8",
    )


def _make_rag(
    tmp_path: Path, *, dim: int = 8, llm_response: str | None = "stub answer"
):
    fake_embedder = MagicMock()
    fake_embedder.dim = dim
    fake_embedder.provider = "local"
    fake_embedder.embed_texts.side_effect = lambda texts: [
        [float((i + 1) % 7) / 10.0] * dim for i, _ in enumerate(texts)
    ]
    fake_embedder.embed_query.side_effect = lambda q: [0.1] * dim

    fake_llm = MagicMock()
    fake_llm.available = llm_response is not None
    fake_llm.provider = "anthropic_api" if llm_response is not None else "none"
    fake_llm.generate.return_value = llm_response

    return rag.FilingRAG(
        embedder=fake_embedder,
        llm=fake_llm,
        db_path=tmp_path / "vectors.db",
    )


def test_ingest_filing_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)

    rag_inst = _make_rag(tmp_path)
    summary = rag_inst.ingest_filing(accession, "AAPL")
    assert "error" not in summary
    assert summary["chunks_added"] > 0
    assert summary["embedding_dim"] == 8
    assert "chunks_by_section" in summary
    assert any(v > 0 for v in summary["chunks_by_section"].values())
    rag_inst.close()


def test_ingest_filing_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)
    rag_inst = _make_rag(tmp_path)
    first = rag_inst.ingest_filing(accession, "AAPL")
    second = rag_inst.ingest_filing(accession, "AAPL")
    assert first["chunks_added"] == second["chunks_added"]
    # Total store count should equal a single ingestion.
    assert rag_inst._store.count() == first["chunks_added"]
    rag_inst.close()


def test_ingest_filing_missing_cache_returns_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    rag_inst = _make_rag(tmp_path)
    out = rag_inst.ingest_filing("0000000000-00-000000", "AAPL")
    assert out["chunks_added"] == 0
    assert "error" in out
    rag_inst.close()


def test_ingest_filing_embedder_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)
    rag_inst = _make_rag(tmp_path)
    rag_inst.embedder.embed_texts.side_effect = None
    rag_inst.embedder.embed_texts.return_value = None
    out = rag_inst.ingest_filing(accession, "AAPL")
    assert out["chunks_added"] == 0
    assert "error" in out
    rag_inst.close()


def test_ask_filing_returns_answer_and_citations(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)
    rag_inst = _make_rag(tmp_path, llm_response="Apple discusses risk factors. [Risk Factors §0]")
    rag_inst.ingest_filing(accession, "AAPL")

    result = rag_inst.ask_filing(
        "What are the risk factors?",
        accession_no=accession,
        ticker="AAPL",
        k=3,
    )
    assert result["answer"] == "Apple discusses risk factors. [Risk Factors §0]"
    assert result["provider"] == "anthropic_api"
    assert result["cited_chunks"]
    assert len(result["cited_chunks"]) <= 3
    for c in result["cited_chunks"]:
        assert "section" in c
        assert "chunk_idx" in c
        assert "text" in c
        assert "score" in c
    rag_inst.close()


def test_ask_filing_no_llm_available(tmp_path: Path):
    rag_inst = _make_rag(tmp_path, llm_response=None)
    result = rag_inst.ask_filing(
        "anything?", accession_no="A-1", ticker="AAPL"
    )
    assert result["answer"] is None
    assert "error" in result
    assert "no llm" in result["error"].lower()


def test_ask_filing_no_chunks_indexed(tmp_path: Path):
    rag_inst = _make_rag(tmp_path, llm_response="should not be called")
    # Don't ingest — vector store is empty.
    result = rag_inst.ask_filing(
        "anything?", accession_no="A-1", ticker="AAPL"
    )
    assert result["answer"] is None
    assert "error" in result
    rag_inst.close()


def test_prompt_includes_numbered_context_and_section_labels():
    chunks = [
        {"section": "Risk Factors", "chunk_idx": 0, "text": "Supply chain risk."},
        {"section": "MD&A", "chunk_idx": 4, "text": "Revenue grew 5%."},
    ]
    prompt = rag._build_prompt("Tell me about risks.", chunks)
    assert "[1] Section: Risk Factors §0" in prompt
    assert "[2] Section: MD&A §4" in prompt
    assert "Supply chain risk." in prompt
    assert "Tell me about risks." in prompt
    assert "ONLY" in prompt or "only" in prompt
    assert "[Section §chunk_idx]" in prompt


def _seed_cache_with_meta(
    tmp_path: Path, ticker: str, accession_no: str, filing_date: str
) -> None:
    target = tmp_path / ticker / "10-K" / accession_no
    target.mkdir(parents=True)
    (target / "full-text.html").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "accession_no": accession_no,
                "ticker": ticker,
                "form_type": "10-K",
                "filing_date": filing_date,
                "period_end": filing_date,
            }
        ),
        encoding="utf-8",
    )


def test_ask_corpus_retrieves_across_multiple_tickers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    _seed_cache_with_meta(tmp_path, "AAPL", "0000320193-24-000123", "2024-11-01")
    _seed_cache_with_meta(tmp_path, "MSFT", "0000789019-24-000044", "2024-07-30")
    rag_inst = _make_rag(
        tmp_path, llm_response="AAPL and MSFT both mention AI. [AAPL Risk Factors §0]"
    )
    rag_inst.ingest_filing("0000320193-24-000123", "AAPL")
    rag_inst.ingest_filing("0000789019-24-000044", "MSFT")

    result = rag_inst.ask_corpus("What do they say about risks?", k=8)
    assert result["answer"]
    assert result["cited_chunks"], "expected cited chunks"
    # filings_searched should include both tickers since both were ingested.
    tickers_in_results = {f["ticker"] for f in result["filings_searched"]}
    assert "AAPL" in tickers_in_results
    assert "MSFT" in tickers_in_results
    # Each cited chunk surfaces its ticker + accession_no + filing_date.
    for c in result["cited_chunks"]:
        assert "ticker" in c
        assert "accession_no" in c
        assert "filing_date" in c
    rag_inst.close()


def test_ask_corpus_ticker_filter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    _seed_cache_with_meta(tmp_path, "AAPL", "0000320193-24-000123", "2024-11-01")
    _seed_cache_with_meta(tmp_path, "MSFT", "0000789019-24-000044", "2024-07-30")
    rag_inst = _make_rag(tmp_path, llm_response="ok")
    rag_inst.ingest_filing("0000320193-24-000123", "AAPL")
    rag_inst.ingest_filing("0000789019-24-000044", "MSFT")

    result = rag_inst.ask_corpus("Q?", tickers=["AAPL"], k=8)
    tickers_in_results = {c["ticker"] for c in result["cited_chunks"]}
    assert tickers_in_results == {"AAPL"}
    rag_inst.close()


def test_ask_corpus_empty_store_returns_error(tmp_path: Path):
    rag_inst = _make_rag(tmp_path, llm_response="should not be called")
    result = rag_inst.ask_corpus("Q?")
    assert result["answer"] is None
    assert "error" in result
    assert "ingested" in result["error"].lower()
    rag_inst.close()


def test_ask_corpus_no_llm_available(tmp_path: Path):
    rag_inst = _make_rag(tmp_path, llm_response=None)
    result = rag_inst.ask_corpus("Q?")
    assert result["answer"] is None
    assert "error" in result
    assert "no llm" in result["error"].lower()


def test_ask_corpus_prompt_uses_ticker_citation_format():
    chunks = [
        {
            "ticker": "AAPL",
            "section": "Risk Factors",
            "chunk_idx": 0,
            "text": "Supply chain risk.",
        },
        {
            "ticker": "MSFT",
            "section": "MD&A",
            "chunk_idx": 4,
            "text": "Cloud revenue grew.",
        },
    ]
    prompt = rag._build_corpus_prompt("Compare them.", chunks)
    assert "[1] AAPL | Section: Risk Factors §0" in prompt
    assert "[2] MSFT | Section: MD&A §4" in prompt
    assert "[TICKER Section §chunk_idx]" in prompt
    assert "[AAPL Risk Factors §3]" in prompt  # example in instructions


def test_ask_filing_passes_prompt_to_llm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(edgar.config, "CACHE_DIR", tmp_path)
    accession = "0000320193-24-000123"
    _seed_cache(tmp_path, "AAPL", accession)
    rag_inst = _make_rag(tmp_path, llm_response="ok")
    rag_inst.ingest_filing(accession, "AAPL")
    rag_inst.ask_filing("Q?", accession_no=accession, ticker="AAPL", k=2)

    rag_inst.llm.generate.assert_called_once()
    args, kwargs = rag_inst.llm.generate.call_args
    prompt = args[0] if args else kwargs.get("prompt", "")
    system = kwargs.get("system", "")
    assert "=== Context ===" in prompt
    assert "=== Question ===" in prompt
    assert "Q?" in prompt
    assert "SEC 10-K" in system
    rag_inst.close()

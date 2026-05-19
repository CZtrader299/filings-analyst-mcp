"""Tests for the sqlite-vec wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module if sqlite-vec isn't installed.
pytest.importorskip("sqlite_vec")

from filings_analyst.vectorstore import VectorStore


def _make_records(accession_no: str = "0000320193-24-000123") -> list[dict]:
    return [
        {
            "accession_no": accession_no,
            "ticker": "AAPL",
            "section": "Business",
            "chunk_idx": 0,
            "text": "Apple sells iPhones and Macs.",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "accession_no": accession_no,
            "ticker": "AAPL",
            "section": "Business",
            "chunk_idx": 1,
            "text": "Apple operates services like the App Store.",
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "accession_no": accession_no,
            "ticker": "AAPL",
            "section": "Risk Factors",
            "chunk_idx": 0,
            "text": "Supply chain disruption is a risk.",
            "embedding": [0.0, 0.0, 1.0],
        },
    ]


def test_create_and_count(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    assert store.count() == 0
    n = store.add_chunks(_make_records())
    assert n == 3
    assert store.count() == 3
    store.close()


def test_search_returns_nearest_first(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    store.add_chunks(_make_records())
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert len(hits) == 2
    # The exact match should be first, with score 0.
    assert hits[0]["section"] == "Business"
    assert hits[0]["chunk_idx"] == 0
    assert hits[0]["score"] == pytest.approx(0.0, abs=1e-5)
    # All hits include the metadata fields.
    for h in hits:
        assert h["accession_no"] == "0000320193-24-000123"
        assert h["ticker"] == "AAPL"
        assert "text" in h
    store.close()


def test_search_filter_accession(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    store.add_chunks(_make_records(accession_no="A-1"))
    store.add_chunks(_make_records(accession_no="A-2"))
    hits = store.search([1.0, 0.0, 0.0], k=3, filter_accession_no="A-2")
    assert hits, "Expected at least one filtered hit"
    assert all(h["accession_no"] == "A-2" for h in hits)
    store.close()


def test_dim_mismatch_raises(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    bad = _make_records()
    bad[0]["embedding"] = [1.0, 0.0]  # wrong dim
    with pytest.raises(ValueError):
        store.add_chunks(bad)
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=1)
    store.close()


def test_delete_filing(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    store.add_chunks(_make_records(accession_no="A-1"))
    store.add_chunks(_make_records(accession_no="A-2"))
    assert store.count() == 6
    removed = store.delete_filing("A-1")
    assert removed == 3
    assert store.count() == 3
    # Deleting a missing accession is a no-op.
    assert store.delete_filing("nonexistent") == 0
    store.close()


def test_reinsert_same_chunk_id_is_idempotent(tmp_path: Path):
    store = VectorStore(tmp_path / "v.db", dim=3)
    store.add_chunks(_make_records())
    assert store.count() == 3
    # Insert the same set again — chunk_ids collide, should overwrite not duplicate.
    store.add_chunks(_make_records())
    assert store.count() == 3
    store.close()


def test_invalid_dim_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        VectorStore(tmp_path / "v.db", dim=0)
    with pytest.raises(ValueError):
        VectorStore(tmp_path / "v.db", dim=-5)


def test_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "nested" / "dir" / "v.db"
    store = VectorStore(target, dim=3)
    assert target.parent.exists()
    store.close()

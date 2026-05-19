"""Tests for the deterministic text chunker."""

from __future__ import annotations

from filings_analyst import chunking


def test_empty_input_returns_no_chunks():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("   \n  \n") == []


def test_short_input_returns_single_chunk():
    text = "This is a short paragraph that fits in one chunk."
    out = chunking.chunk_text(text, target_tokens=100, overlap_tokens=10)
    assert len(out) == 1
    assert out[0].startswith("This is a short")


def test_long_input_splits_into_multiple_chunks():
    # target=50 tokens ~= 200 chars per chunk; build a paragraph well over.
    sentences = [f"Sentence number {i} contains some filler words here." for i in range(40)]
    text = " ".join(sentences)
    chunks = chunking.chunk_text(text, target_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 3
    # Every chunk must contain real prose.
    for c in chunks:
        assert c.strip()


def test_overlap_shared_between_adjacent_chunks():
    # Construct text with clearly numbered sentences so we can verify overlap.
    sentences = [f"Marker{i:03d} word filler filler filler text." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunking.chunk_text(text, target_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    # The tail of chunk N should share at least one marker with the head of N+1.
    import re

    for a, b in zip(chunks, chunks[1:]):
        markers_a = set(re.findall(r"Marker\d{3}", a[-200:]))
        markers_b = set(re.findall(r"Marker\d{3}", b[:200]))
        assert markers_a & markers_b, f"No overlap between adjacent chunks: {markers_a} vs {markers_b}"


def test_sentence_boundary_preferred_over_mid_word_break():
    text = "First sentence ends here. Second sentence follows. Third one too. Fourth and fifth."
    chunks = chunking.chunk_text(text, target_tokens=10, overlap_tokens=2)
    # No chunk should end mid-word (besides hard-fallback cases).
    for c in chunks:
        # We allow a trailing period; the chunk should end with a complete word.
        last = c.rstrip(".!? ")
        assert not last.endswith("sentenc"), f"mid-word break detected: {c!r}"


def test_huge_sentence_falls_back_to_hard_chars():
    # One "sentence" much larger than the target window — no terminator.
    long = "wordy " * 500  # 3000 chars, no period at all
    chunks = chunking.chunk_text(long, target_tokens=50, overlap_tokens=5)
    assert len(chunks) > 1
    # Hard char fallback should still respect approximate target size.
    for c in chunks:
        assert len(c) <= 50 * chunking.CHARS_PER_TOKEN


def test_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        chunking.chunk_text("hello world.", target_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunking.chunk_text("hello world.", target_tokens=0, overlap_tokens=0)


def test_chunk_sections_tags_origin_and_indexes():
    sections_input = {
        "Business": "We sell things. Many things. So many. " * 30,
        "Risk Factors": "Things can go wrong. Sometimes badly. " * 30,
        "MD&A": "",  # should be skipped
    }
    records = chunking.chunk_sections(
        sections_input, target_tokens=60, overlap_tokens=10
    )
    assert records, "Expected at least one chunk record"
    sections_seen = {r["section"] for r in records}
    assert "Business" in sections_seen
    assert "Risk Factors" in sections_seen
    assert "MD&A" not in sections_seen  # empty → skipped

    # chunk_idx is per-section and monotonic from 0.
    for section_name in sections_seen:
        idxs = [r["chunk_idx"] for r in records if r["section"] == section_name]
        assert idxs == list(range(len(idxs))), f"Bad indexing for {section_name}: {idxs}"


def test_chunk_sections_empty_dict():
    assert chunking.chunk_sections({}) == []


def test_chunk_sections_preserves_input_order():
    # Use an OrderedDict-like dict (3.7+ preserves insertion order).
    sections_input = {
        "Z-section": "Alpha sentence. Beta sentence. " * 20,
        "A-section": "Gamma sentence. Delta sentence. " * 20,
    }
    records = chunking.chunk_sections(
        sections_input, target_tokens=40, overlap_tokens=5
    )
    # The Z-section records should all come before the A-section ones.
    first_a = next(i for i, r in enumerate(records) if r["section"] == "A-section")
    last_z = max(i for i, r in enumerate(records) if r["section"] == "Z-section")
    assert last_z < first_a

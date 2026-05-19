"""Deterministic section-aware text chunker.

Splits long text into roughly token-sized windows with overlap, preferring
sentence boundaries (``. ``, ``.\n``, ``\n\n``) so chunks stay readable.

Token counts are approximated from characters using a 4-chars-per-token
ratio — a standard rule of thumb for English produced by the GPT-family
BPE tokenizers. This avoids pulling in a heavy tokenizer dependency just
for chunking. The downstream embedding model truncates at its own context
window if a chunk slightly exceeds the target.

The chunker is intentionally boring: no LLM, no external libraries, pure
Python. Determinism matters here so that re-ingesting the same filing
produces the same chunk IDs and the vector store stays idempotent.
"""

from __future__ import annotations

import re
from typing import Iterable


# Rough character-to-token ratio for English text. GPT-family tokenizers
# average ~3.8-4.2 chars/token for prose; 4.0 is a safe default.
CHARS_PER_TOKEN = 4


# Sentence boundary detector. We accept ``. ``, ``? ``, ``! `` followed by
# whitespace, plus paragraph breaks (``\n\n``). Captured so we can keep the
# terminator attached to the preceding sentence when we re-join.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\(])|\n\n+")


def _split_sentences(text: str) -> list[str]:
    """Break text into sentence-like fragments.

    Falls back to single-line splits when no terminator is found so we
    never lose content. Returns a list with whitespace-trimmed fragments
    and empty entries removed.
    """
    parts = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    for part in parts:
        if part is None:
            continue
        cleaned = part.strip()
        if cleaned:
            out.append(cleaned)
    return out


def _char_chunks(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Hard char-boundary fallback for monster sentences.

    Used only when a single "sentence" already exceeds the target window.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap_chars)
    out: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + max_chars]
        if chunk:
            out.append(chunk)
        if start + max_chars >= len(text):
            break
    return out


def chunk_text(
    text: str,
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``target_tokens``.

    Strategy:

    1. Split into sentence-like fragments at ``. ``/``? ``/``! ``/``\n\n``.
    2. Greedily accumulate fragments into the current chunk until adding
       the next one would exceed ``target_tokens``.
    3. When emitting a chunk, seed the next chunk with the tail of the
       previous one (~``overlap_tokens`` worth) so adjacent chunks share
       context. This helps retrieval when an answer straddles a boundary.
    4. If a single fragment is already larger than the target window,
       fall back to hard char-boundary splits for that fragment alone.
    """
    if not text or not text.strip():
        return []
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")

    max_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> str:
        return " ".join(current).strip()

    def tail_for_overlap(chunk_text_: str) -> tuple[list[str], int]:
        """Return (seed_fragments, seed_char_len) for the next chunk."""
        if overlap_chars <= 0 or not chunk_text_:
            return [], 0
        tail = chunk_text_[-overlap_chars:]
        # Re-tokenize the tail at sentence boundaries so the overlap
        # starts cleanly rather than mid-word when possible.
        tail_sents = _split_sentences(tail)
        if not tail_sents:
            return [tail], len(tail)
        # Drop the first fragment if it looks truncated (no leading capital).
        if tail_sents and tail_sents[0] and not tail_sents[0][0].isupper():
            tail_sents = tail_sents[1:]
        if not tail_sents:
            return [], 0
        joined = " ".join(tail_sents)
        return tail_sents, len(joined)

    for sentence in sentences:
        if len(sentence) > max_chars:
            # Flush whatever's accumulated, then hard-split the giant fragment.
            if current:
                emitted = flush()
                if emitted:
                    chunks.append(emitted)
                current, current_len = [], 0
            for piece in _char_chunks(sentence, max_chars, overlap_chars):
                chunks.append(piece)
            # Seed next chunk with overlap from the last piece.
            if chunks:
                seed, seed_len = tail_for_overlap(chunks[-1])
                current = list(seed)
                current_len = seed_len
            continue

        prospective_len = current_len + (1 if current else 0) + len(sentence)
        if current and prospective_len > max_chars:
            emitted = flush()
            if emitted:
                chunks.append(emitted)
            seed, seed_len = tail_for_overlap(emitted)
            current = list(seed)
            current_len = seed_len
            # Now add this sentence to the fresh chunk.
            current.append(sentence)
            current_len += (1 if seed_len else 0) + len(sentence)
        else:
            current.append(sentence)
            current_len = prospective_len

    if current:
        emitted = flush()
        if emitted:
            chunks.append(emitted)

    return chunks


def chunk_sections(
    sections_text: dict[str, str],
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Chunk each named section and tag chunks with their origin.

    Returns a flat list of ``{"section", "chunk_idx", "text"}`` records.
    Empty sections are skipped silently. Section iteration order is the
    order of the input dict so callers control determinism.
    """
    records: list[dict] = []
    for section_name, section_text in sections_text.items():
        if not section_text or not section_text.strip():
            continue
        pieces = chunk_text(
            section_text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        for idx, chunk in enumerate(pieces):
            records.append(
                {
                    "section": section_name,
                    "chunk_idx": idx,
                    "text": chunk,
                }
            )
    return records


__all__: Iterable[str] = ("chunk_text", "chunk_sections", "CHARS_PER_TOKEN")

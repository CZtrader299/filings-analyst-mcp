"""Thin wrapper around sqlite-vec for chunk storage and similarity search.

We use a sqlite-vec virtual table for the embedding column and a plain
sqlite table for metadata, joined on ``chunk_id``. Going with the
``vec0`` extension's virtual table for vectors keeps cosine-similarity
search a single SQL query.

Schema:

    -- metadata
    CREATE TABLE chunk_meta (
        chunk_id      TEXT PRIMARY KEY,
        accession_no  TEXT NOT NULL,
        ticker        TEXT NOT NULL,
        section       TEXT NOT NULL,
        chunk_idx     INTEGER NOT NULL,
        text          TEXT NOT NULL,
        filing_date   TEXT             -- added week 3, nullable for back-compat
    );

    -- vec0 virtual table for similarity search
    CREATE VIRTUAL TABLE chunk_vec USING vec0(
        chunk_id    TEXT PRIMARY KEY,
        embedding   FLOAT[<dim>]
    );

``chunk_id`` is ``"{accession_no}:{section}:{chunk_idx}"`` so re-ingesting
the same filing produces stable IDs.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional


def _check_sqlite_vec_importable() -> None:
    """Import sqlite-vec or raise a friendly ImportError."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via integration
        raise ImportError(
            "sqlite-vec is required for the vector store. "
            'Install with `pip install -e ".[embeddings]"`.'
        ) from exc


def _vec_to_blob(vec: list[float]) -> bytes:
    """Pack a float vector as a little-endian float32 blob for vec0."""
    return struct.pack(f"<{len(vec)}f", *vec)


class VectorStore:
    """Sqlite-vec-backed chunk store with cosine similarity search.

    Persistent: pass a real file path for on-disk storage. Pass
    ``":memory:"`` for ephemeral test usage.
    """

    def __init__(self, db_path: str | Path, dim: int):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        _check_sqlite_vec_importable()
        import sqlite_vec

        self.db_path = str(db_path)
        self.dim = dim

        path = Path(self.db_path)
        if path.name != ":memory:" and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._ensure_schema()

    # --- Schema ----------------------------------------------------------

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_meta (
                chunk_id      TEXT PRIMARY KEY,
                accession_no  TEXT NOT NULL,
                ticker        TEXT NOT NULL,
                section       TEXT NOT NULL,
                chunk_idx     INTEGER NOT NULL,
                text          TEXT NOT NULL,
                filing_date   TEXT
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_meta_accession "
            "ON chunk_meta(accession_no)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_meta_ticker "
            "ON chunk_meta(ticker)"
        )
        # Migration: older dbs (week 2) don't have the filing_date column.
        # ``ALTER TABLE ADD COLUMN`` is no-op-safe if we guard it with a
        # PRAGMA introspection — sqlite has no IF NOT EXISTS on ADD COLUMN.
        cur.execute("PRAGMA table_info(chunk_meta)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "filing_date" not in existing_cols:
            cur.execute("ALTER TABLE chunk_meta ADD COLUMN filing_date TEXT")
        # vec0 virtual table — needs an explicit dim baked into the schema.
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.dim}]
            )
            """
        )
        self.conn.commit()

    # --- Mutation --------------------------------------------------------

    def add_chunks(self, records: list[dict[str, Any]]) -> int:
        """Insert chunk records. Returns the number of rows inserted.

        Each record must include: ``accession_no``, ``ticker``, ``section``,
        ``chunk_idx``, ``text``, ``embedding``. ``chunk_id`` is derived.
        Existing rows with the same ``chunk_id`` are replaced (re-ingestion
        is idempotent at the chunk level).
        """
        if not records:
            return 0
        cur = self.conn.cursor()
        inserted = 0
        for rec in records:
            chunk_id = (
                f"{rec['accession_no']}:{rec['section']}:{rec['chunk_idx']}"
            )
            embedding = rec["embedding"]
            if len(embedding) != self.dim:
                raise ValueError(
                    f"Embedding dim {len(embedding)} != store dim {self.dim} "
                    f"(chunk_id={chunk_id})"
                )
            # Wipe any existing row first — vec0 doesn't support REPLACE.
            cur.execute("DELETE FROM chunk_meta WHERE chunk_id = ?", (chunk_id,))
            cur.execute("DELETE FROM chunk_vec WHERE chunk_id = ?", (chunk_id,))
            cur.execute(
                "INSERT INTO chunk_meta(chunk_id, accession_no, ticker, "
                "section, chunk_idx, text, filing_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    rec["accession_no"],
                    rec["ticker"],
                    rec["section"],
                    int(rec["chunk_idx"]),
                    rec["text"],
                    rec.get("filing_date"),
                ),
            )
            cur.execute(
                "INSERT INTO chunk_vec(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, _vec_to_blob(embedding)),
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def delete_filing(self, accession_no: str) -> int:
        """Remove all chunks for one filing. Returns the number deleted."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM chunk_meta WHERE accession_no = ?",
            (accession_no,),
        )
        ids = [row[0] for row in cur.fetchall()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        cur.execute(
            f"DELETE FROM chunk_meta WHERE chunk_id IN ({placeholders})", ids
        )
        cur.execute(
            f"DELETE FROM chunk_vec WHERE chunk_id IN ({placeholders})", ids
        )
        self.conn.commit()
        return len(ids)

    # --- Read ------------------------------------------------------------

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunk_meta")
        return int(cur.fetchone()[0])

    def search(
        self,
        query_embedding: list[float],
        k: int = 6,
        filter_accession_no: Optional[str] = None,
        filter_tickers: Optional[list[str]] = None,
        filter_accession_nos: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Top-k similarity search, optionally filtered by accession/tickers.

        Returns records with all metadata fields plus a ``score`` field
        (cosine distance — lower is closer; we surface that as-is rather
        than reshaping into a similarity to keep the math honest).

        Filters are AND-combined and all use the same over-fetch +
        post-filter pattern as ``filter_accession_no`` — vec0 doesn't
        accept arbitrary WHERE predicates against the metadata table, so
        we widen ``k`` then trim. Fine when the post-filter set is large
        relative to ``k``; degrades only when callers ask for a narrow
        slice of an enormous corpus.
        """
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"Query embedding dim {len(query_embedding)} != store dim {self.dim}"
            )
        cur = self.conn.cursor()
        blob = _vec_to_blob(query_embedding)

        # Normalize ticker filter to uppercase for case-insensitive matching.
        ticker_set: Optional[set[str]] = None
        if filter_tickers:
            ticker_set = {t.upper() for t in filter_tickers if t}
            if not ticker_set:
                ticker_set = None
        accession_set: Optional[set[str]] = None
        if filter_accession_nos:
            accession_set = {a for a in filter_accession_nos if a}
            if not accession_set:
                accession_set = None

        any_filter = bool(filter_accession_no or ticker_set or accession_set)
        if any_filter:
            over_k = max(k * 8, 64)
            cur.execute(
                """
                SELECT v.chunk_id, v.distance
                FROM chunk_vec v
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, over_k),
            )
            raw_hits = cur.fetchall()
            filtered: list[tuple[str, float]] = []
            for chunk_id, distance in raw_hits:
                cur.execute(
                    "SELECT accession_no, ticker FROM chunk_meta WHERE chunk_id = ?",
                    (chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    continue
                accession_no, ticker = row[0], row[1]
                if filter_accession_no and accession_no != filter_accession_no:
                    continue
                if accession_set is not None and accession_no not in accession_set:
                    continue
                if ticker_set is not None and (ticker or "").upper() not in ticker_set:
                    continue
                filtered.append((chunk_id, distance))
                if len(filtered) >= k:
                    break
            hits = filtered
        else:
            cur.execute(
                """
                SELECT v.chunk_id, v.distance
                FROM chunk_vec v
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, k),
            )
            hits = cur.fetchall()

        out: list[dict[str, Any]] = []
        for chunk_id, distance in hits:
            cur.execute(
                "SELECT accession_no, ticker, section, chunk_idx, text, "
                "filing_date FROM chunk_meta WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cur.fetchone()
            if not row:
                continue
            out.append(
                {
                    "chunk_id": chunk_id,
                    "accession_no": row[0],
                    "ticker": row[1],
                    "section": row[2],
                    "chunk_idx": int(row[3]),
                    "text": row[4],
                    "filing_date": row[5] or "",
                    "score": float(distance),
                }
            )
        return out

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ("VectorStore",)

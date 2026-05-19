"""Pluggable embedding provider.

Mirrors the pattern in ``providers.py`` (LLM router) but for embedding
models. Two backends today:

* ``local`` — sentence-transformers, default ``all-MiniLM-L6-v2`` (384-d).
  Zero API cost, runs offline once the model is downloaded.
* ``openai`` — OpenAI's ``/v1/embeddings`` endpoint, default
  ``text-embedding-3-small`` (1536-d).

The factory dispatches by ``EMBEDDING_PROVIDER`` env var (or explicit
constructor arg). ``sentence_transformers`` is import-lazy so that the
suite stays fast and offline for users who never hit the local path.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from . import config


# OpenAI's embeddings API caps inputs at 2048 per request, but per their
# best-practice docs anything under ~100 is safer for token-budget reasons.
# 96 is the same number their cookbook uses.
_OPENAI_BATCH_SIZE = 96

# Known output dimensionalities. Used by the ``dim`` property so the
# vector store can be sized correctly without making a probe call.
_KNOWN_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


PROVIDERS = {"local", "openai"}


class Embedder:
    """Resolve and call a configured embedding backend."""

    def __init__(self, provider: Optional[str] = None):
        requested = (provider or config.EMBEDDING_PROVIDER or "local").strip().lower()
        if requested not in PROVIDERS:
            print(
                f"  Warning: unknown embedding provider {requested!r}, "
                "falling back to 'local'"
            )
            requested = "local"
        self.provider = requested
        # Lazy-loaded sentence-transformers model handle.
        self._local_model = None

    # --- Public API ------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Batch-embed a list of strings.

        Returns a list of float vectors (one per input) or ``None`` on
        backend failure. Empty input returns an empty list.
        """
        if not texts:
            return []
        if self.provider == "local":
            return self._embed_local(texts)
        if self.provider == "openai":
            return self._embed_openai(texts)
        return None

    def embed_query(self, text: str) -> Optional[list[float]]:
        """Convenience: embed a single string and return its vector."""
        vectors = self.embed_texts([text])
        if not vectors:
            return None
        return vectors[0]

    @property
    def model_name(self) -> str:
        if self.provider == "local":
            return config.LOCAL_EMBEDDING_MODEL
        if self.provider == "openai":
            return config.OPENAI_EMBEDDING_MODEL
        return ""

    @property
    def dim(self) -> int:
        """Embedding vector dimensionality for the configured model.

        Falls back to 384 (MiniLM) if the model isn't in the known table —
        that's safer than 0 because the sqlite-vec table needs a fixed dim
        at creation time. Users running an unknown model can override with
        ``EMBEDDER_DIM`` if needed (see ``_resolve_dim``).
        """
        return self._resolve_dim()

    def _resolve_dim(self) -> int:
        override = os.environ.get("EMBEDDER_DIM", "").strip()
        if override.isdigit():
            return int(override)
        return _KNOWN_DIMS.get(self.model_name, 384)

    # --- Local backend ---------------------------------------------------

    def _embed_local(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed via sentence-transformers, loading the model on first use."""
        if self._local_model is None:
            try:
                # Import is deferred: sentence_transformers pulls in torch
                # plus a large dependency tree (~2s import cost). Don't pay
                # it on every ``import filings_analyst``.
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                print(
                    "  Warning: sentence-transformers not installed. "
                    'Run `pip install -e ".[embeddings]"`.'
                )
                print(f"  Detail: {exc}")
                return None
            try:
                self._local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: failed to load local embedding model: {exc}")
                return None
        try:
            # ``encode`` returns numpy arrays; convert to plain Python
            # lists so callers don't need numpy.
            vectors = self._local_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: local embedding failed: {exc}")
            return None
        return [list(map(float, v)) for v in vectors]

    # --- OpenAI backend --------------------------------------------------

    def _embed_openai(self, texts: list[str]) -> Optional[list[list[float]]]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("  Warning: OPENAI_API_KEY not set; cannot use openai embeddings")
            return None
        model = os.environ.get(
            "OPENAI_EMBEDDING_MODEL", config.OPENAI_EMBEDDING_MODEL
        )
        out: list[list[float]] = []
        for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[start : start + _OPENAI_BATCH_SIZE]
            try:
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "input": batch},
                    timeout=config.REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                print(f"  Warning: OpenAI embeddings request failed: {exc}")
                return None
            if not response.ok:
                print(
                    f"  Warning: OpenAI embeddings HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None
            try:
                payload = response.json()
                items = sorted(payload["data"], key=lambda d: d.get("index", 0))
                out.extend([list(map(float, item["embedding"])) for item in items])
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  Warning: malformed OpenAI embeddings response: {exc}")
                return None
        return out


__all__ = ("Embedder", "PROVIDERS")

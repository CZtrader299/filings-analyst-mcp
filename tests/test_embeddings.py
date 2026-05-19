"""Tests for the embedding provider factory.

Both backends are fully mocked here — we never download the local model
or hit OpenAI. The factory + dispatch + batching logic is what we care
about; the actual numerical correctness of MiniLM / OpenAI embeddings is
upstream's responsibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import responses

from filings_analyst import embeddings


def test_default_provider_is_local(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    e = embeddings.Embedder()
    assert e.provider == "local"


def test_unknown_provider_falls_back_to_local(capsys):
    e = embeddings.Embedder(provider="bogus")
    out = capsys.readouterr().out
    assert e.provider == "local"
    assert "unknown" in out.lower()


def test_known_dims_table():
    e_local = embeddings.Embedder(provider="local")
    assert e_local.dim == 384
    e_openai = embeddings.Embedder(provider="openai")
    assert e_openai.dim == 1536


def test_dim_override_env(monkeypatch):
    monkeypatch.setenv("EMBEDDER_DIM", "768")
    e = embeddings.Embedder(provider="local")
    assert e.dim == 768


def test_embed_texts_empty_returns_empty():
    e = embeddings.Embedder(provider="local")
    assert e.embed_texts([]) == []


# --- Local backend ----------------------------------------------------------


def test_local_embed_uses_sentence_transformers():
    e = embeddings.Embedder(provider="local")
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}) as mods:
        mods["sentence_transformers"].SentenceTransformer = MagicMock(
            return_value=fake_model
        )
        vectors = e.embed_texts(["hello", "world"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    # Second call should reuse the cached model.
    fake_model.encode.return_value = [[0.7, 0.8, 0.9]]
    assert e.embed_query("again") == [0.7, 0.8, 0.9]


def test_local_embed_missing_dependency_returns_none(capsys, monkeypatch):
    e = embeddings.Embedder(provider="local")
    # Simulate ImportError by stubbing sys.modules so the import inside
    # _embed_local raises.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = e.embed_texts(["hello"])
    assert out is None
    captured = capsys.readouterr().out
    assert "sentence-transformers" in captured.lower()


def test_local_embed_model_load_failure_returns_none(capsys):
    e = embeddings.Embedder(provider="local")
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}) as mods:
        mods["sentence_transformers"].SentenceTransformer = MagicMock(
            side_effect=RuntimeError("simulated load failure")
        )
        out = e.embed_texts(["hello"])
    assert out is None
    captured = capsys.readouterr().out
    assert "failed to load" in captured.lower()


# --- OpenAI backend ---------------------------------------------------------


@responses.activate
def test_openai_embed_single_batch(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/embeddings",
        json={
            "data": [
                {"index": 0, "embedding": [0.1] * 1536},
                {"index": 1, "embedding": [0.2] * 1536},
            ],
            "model": "text-embedding-3-small",
        },
        status=200,
    )
    e = embeddings.Embedder(provider="openai")
    out = e.embed_texts(["hello", "world"])
    assert out is not None
    assert len(out) == 2
    assert len(out[0]) == 1536
    assert out[0][0] == 0.1


@responses.activate
def test_openai_embed_batches_above_96(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # 200 texts -> ceil(200/96) = 3 batches
    call_count = {"n": 0}

    def cb(request):
        import json as _json

        body = _json.loads(request.body)
        n = len(body["input"])
        call_count["n"] += 1
        data = [
            {"index": i, "embedding": [float(call_count["n"])] * 1536}
            for i in range(n)
        ]
        return (200, {}, _json.dumps({"data": data, "model": "text-embedding-3-small"}))

    responses.add_callback(
        responses.POST,
        "https://api.openai.com/v1/embeddings",
        callback=cb,
        content_type="application/json",
    )
    e = embeddings.Embedder(provider="openai")
    out = e.embed_texts([f"t{i}" for i in range(200)])
    assert out is not None
    assert len(out) == 200
    assert call_count["n"] == 3


@responses.activate
def test_openai_embed_http_error_returns_none(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/embeddings",
        status=500,
        body="server error",
    )
    e = embeddings.Embedder(provider="openai")
    out = e.embed_texts(["hello"])
    assert out is None
    captured = capsys.readouterr().out
    assert "openai" in captured.lower()


def test_openai_embed_no_api_key_returns_none(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    e = embeddings.Embedder(provider="openai")
    out = e.embed_texts(["hello"])
    assert out is None
    captured = capsys.readouterr().out
    assert "openai_api_key" in captured.lower()

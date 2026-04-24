"""
test_semantic_search.py — Test Suite
ZK Cloud Storage · Semantic Search Module

Run:  pytest test_semantic_search.py -v
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── patch sentence-transformers so tests don't need GPU / download ─────────────
EMBED_DIM = 384

class _FakeModel:
    """Returns deterministic unit-norm vectors for any input."""
    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


with patch("sentence_transformers.SentenceTransformer", return_value=_FakeModel()):
    from embedder import Embedder, extract_text
    from vector_store import VectorStore
    # patch again for search_api import
    with patch("sentence_transformers.SentenceTransformer", return_value=_FakeModel()):
        from search_api import app


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def embedder():
    with patch("sentence_transformers.SentenceTransformer", return_value=_FakeModel()):
        return Embedder()


@pytest.fixture
def fresh_store():
    return VectorStore()


@pytest.fixture
def client(fresh_store):
    import search_api
    search_api.store = fresh_store
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# embedder.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractText:
    def test_txt_extraction(self):
        text = extract_text(b"Hello world", "file.txt")
        assert "Hello world" in text

    def test_md_extraction(self):
        text = extract_text(b"# Title\nSome content", "notes.md")
        assert "Title" in text

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(b"data", "file.exe")

    def test_latin1_fallback(self):
        data = "Caf\xe9".encode("latin-1")   # non-UTF8 byte
        text = extract_text(data, "file.txt")
        assert "Caf" in text


class TestEmbedder:
    def test_embed_returns_correct_dim(self, embedder):
        vec = embedder.embed("Some document text about finance")
        assert isinstance(vec, list)
        assert len(vec) == EMBED_DIM

    def test_embed_query_returns_correct_dim(self, embedder):
        vec = embedder.embed_query("quarterly revenue report")
        assert isinstance(vec, list)
        assert len(vec) == EMBED_DIM

    def test_embed_is_normalised(self, embedder):
        vec = embedder.embed("test document")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_empty_raises(self, embedder):
        with pytest.raises(ValueError, match="empty text"):
            embedder.embed("")

    def test_embed_query_empty_raises(self, embedder):
        with pytest.raises(ValueError, match="must not be empty"):
            embedder.embed_query("   ")

    def test_same_text_same_vector(self, embedder):
        v1 = embedder.embed("identical sentence")
        v2 = embedder.embed("identical sentence")
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_different_text_different_vector(self, embedder):
        v1 = embedder.embed("apple orange fruit")
        v2 = embedder.embed("nuclear reactor fission")
        assert not np.allclose(v1, v2)

    def test_embed_file_txt(self, embedder):
        vec = embedder.embed_file(b"Hello from a text file", "doc.txt")
        assert len(vec) == EMBED_DIM


# ══════════════════════════════════════════════════════════════════════════════
# vector_store.py tests
# ══════════════════════════════════════════════════════════════════════════════

def _random_embedding(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


class TestVectorStore:
    def test_add_and_search(self, fresh_store):
        emb = _random_embedding(1)
        fresh_store.add("file-001", emb)
        results = fresh_store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "file-001"
        assert results[0][1] > 0.99           # cosine ≈ 1 for same vector

    def test_search_empty_store_returns_empty(self, fresh_store):
        results = fresh_store.search(_random_embedding(), top_k=5)
        assert results == []

    def test_duplicate_add_raises(self, fresh_store):
        emb = _random_embedding(2)
        fresh_store.add("file-dup", emb)
        with pytest.raises(ValueError, match="already exists"):
            fresh_store.add("file-dup", emb)

    def test_delete_removes_from_results(self, fresh_store):
        emb = _random_embedding(3)
        fresh_store.add("file-del", emb)
        fresh_store.delete("file-del")
        results = fresh_store.search(emb, top_k=5)
        ids = [r[0] for r in results]
        assert "file-del" not in ids

    def test_delete_nonexistent_raises(self, fresh_store):
        with pytest.raises(KeyError):
            fresh_store.delete("does-not-exist")

    def test_top_k_limits_results(self, fresh_store):
        for i in range(10):
            fresh_store.add(f"file-{i:03d}", _random_embedding(i + 100))
        results = fresh_store.search(_random_embedding(42), top_k=3)
        assert len(results) <= 3

    def test_results_sorted_by_score_descending(self, fresh_store):
        for i in range(5):
            fresh_store.add(f"sorted-{i}", _random_embedding(i + 200))
        results = fresh_store.search(_random_embedding(99), top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters(self, fresh_store):
        fresh_store.add("score-test", _random_embedding(77))
        results = fresh_store.search(_random_embedding(77), top_k=5, min_score=0.999)
        # Only vectors with near-perfect cosine pass
        assert all(s >= 0.999 for _, s in results)

    def test_len(self, fresh_store):
        before = len(fresh_store)
        fresh_store.add("len-test", _random_embedding(88))
        assert len(fresh_store) == before + 1

    def test_save_and_load(self, fresh_store):
        emb = _random_embedding(55)
        fresh_store.add("persist-test", emb)
        with tempfile.TemporaryDirectory() as tmp:
            idx_path  = Path(tmp) / "test.index"
            ids_path  = Path(tmp) / "test.ids.pkl"
            fresh_store.save(idx_path, ids_path)
            loaded = VectorStore.load(idx_path, ids_path)
        results = loaded.search(emb, top_k=1)
        assert results[0][0] == "persist-test"
        assert results[0][1] > 0.99


# ══════════════════════════════════════════════════════════════════════════════
# search_api.py (FastAPI) tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchAPI:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_store_embedding(self, client):
        r = client.post("/embeddings", json={
            "file_id": "api-file-001",
            "embedding": _random_embedding(10),
        })
        assert r.status_code == 201
        assert r.json()["file_id"] == "api-file-001"

    def test_store_duplicate_returns_409(self, client):
        emb = _random_embedding(11)
        client.post("/embeddings", json={"file_id": "dup-api", "embedding": emb})
        r = client.post("/embeddings", json={"file_id": "dup-api", "embedding": emb})
        assert r.status_code == 409

    def test_search_returns_results(self, client):
        emb = _random_embedding(20)
        client.post("/embeddings", json={"file_id": "search-target", "embedding": emb})
        r = client.post("/search", json={"query_embedding": emb, "top_k": 5})
        assert r.status_code == 200
        ids = [res["file_id"] for res in r.json()["results"]]
        assert "search-target" in ids

    def test_search_scores_between_0_and_1(self, client):
        emb = _random_embedding(21)
        client.post("/embeddings", json={"file_id": "score-api", "embedding": emb})
        r = client.post("/search", json={"query_embedding": emb, "top_k": 5})
        for res in r.json()["results"]:
            assert 0.0 <= res["score"] <= 1.0

    def test_delete_embedding(self, client):
        emb = _random_embedding(30)
        client.post("/embeddings", json={"file_id": "del-api", "embedding": emb})
        r = client.delete("/embeddings/del-api")
        assert r.status_code == 200
        # Confirm it no longer appears in search
        r2 = client.post("/search", json={"query_embedding": emb, "top_k": 10})
        ids = [res["file_id"] for res in r2.json()["results"]]
        assert "del-api" not in ids

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/embeddings/ghost-id")
        assert r.status_code == 404

    def test_wrong_embedding_dim_rejected(self, client):
        r = client.post("/embeddings", json={
            "file_id": "bad-dim",
            "embedding": [0.1] * 128,   # wrong dim
        })
        assert r.status_code == 422     # Pydantic validation error

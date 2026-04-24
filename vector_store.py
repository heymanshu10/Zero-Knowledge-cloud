"""
vector_store.py — Server-Side ANN Vector Index
ZK Cloud Storage · Data Science Module

Responsibilities (per SRS FR-05, NF-05, NF-07):
  - Store embedding vectors keyed by file_id
  - Answer cosine similarity (ANN) queries in < 500 ms over 10 000 vectors
  - Scale to 1 M+ embeddings without significant latency degradation

IMPORTANT — zero-knowledge property:
  The server stores ONLY (file_id → embedding) pairs.
  No plaintext, filenames, or keys are stored here.
  The server cannot reconstruct document content from embeddings (NF-03).
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM   = 384
INDEX_PATH  = Path(os.getenv("VECTOR_STORE_PATH", "vector_store.index"))
IDS_PATH    = INDEX_PATH.with_suffix(".ids.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# VectorStore
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner-product / cosine) index.

    Because all embeddings are L2-normalised before storage, inner-product
    search is equivalent to cosine similarity — no extra normalisation step
    is needed at query time.

    Persistence:
        Call save() after mutations.  load() restores from disk.
        Both operations are O(n) in the number of stored vectors.

    Thread safety:
        FAISS search is thread-safe for read-only operations.
        add() / delete() must be called from a single writer thread
        (or behind a lock) in a multi-threaded server.

    Usage (server side):
        store = VectorStore.load()          # or VectorStore() for fresh

        # on upload
        store.add(file_id, embedding)
        store.save()

        # on search
        results = store.search(query_vec, top_k=10)
        # → [("file-uuid-1", 0.94), ("file-uuid-2", 0.87), ...]

        # on delete
        store.delete(file_id)
        store.save()
    """

    def __init__(self) -> None:
        # IndexFlatIP: exact inner-product (cosine when vectors are normalised)
        # For 1M+ vectors swap to IndexIVFFlat or IndexHNSWFlat (see _upgrade_index)
        self._index: faiss.Index = faiss.IndexFlatIP(EMBED_DIM)
        self._id_map: list[str] = []   # positional index → file_id
        logger.info("New VectorStore initialised (dim=%d)", EMBED_DIM)

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, index_path: Path = INDEX_PATH, ids_path: Path = IDS_PATH) -> None:
        """Persist index and id-map to disk."""
        faiss.write_index(self._index, str(index_path))
        with open(ids_path, "wb") as f:
            pickle.dump(self._id_map, f)
        logger.info("VectorStore saved: %d vectors → %s", len(self._id_map), index_path)

    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_PATH,
        ids_path: Path = IDS_PATH,
    ) -> "VectorStore":
        """Load a previously saved store; return empty store if files absent."""
        store = cls.__new__(cls)
        if index_path.exists() and ids_path.exists():
            store._index = faiss.read_index(str(index_path))
            with open(ids_path, "rb") as f:
                store._id_map = pickle.load(f)
            logger.info(
                "VectorStore loaded: %d vectors from %s", len(store._id_map), index_path
            )
        else:
            logger.warning("No saved index found at %s — starting fresh.", index_path)
            store._index = faiss.IndexFlatIP(EMBED_DIM)
            store._id_map = []
        return store

    # ── mutations ──────────────────────────────────────────────────────────────

    def add(self, file_id: str, embedding: list[float]) -> None:
        """
        Add a single file embedding to the index.

        Args:
            file_id:   Unique identifier for the file (UUID string).
            embedding: 384-d float list from Embedder.embed_file().
        """
        if file_id in self._id_map:
            raise ValueError(f"file_id '{file_id}' already exists. Delete first.")

        vec = _to_float32_matrix([embedding])   # shape (1, 384)
        self._index.add(vec)
        self._id_map.append(file_id)
        logger.debug("Added file_id='%s' (total=%d)", file_id, len(self._id_map))

    def delete(self, file_id: str) -> None:
        """
        Remove a file's embedding from the index.

        FAISS IndexFlatIP does not support in-place deletion, so we
        rebuild the index without the target vector (O(n)).
        For large-scale deployments, switch to IndexIDMap2 which
        supports remove_ids() directly.

        Args:
            file_id: UUID of the file to remove.
        """
        if file_id not in self._id_map:
            raise KeyError(f"file_id '{file_id}' not found in index.")

        target_pos = self._id_map.index(file_id)

        # Reconstruct all vectors except the deleted one
        all_vecs = self._index.reconstruct_n(0, self._index.ntotal)  # (n, 384)
        mask = [i for i in range(len(self._id_map)) if i != target_pos]

        new_index = faiss.IndexFlatIP(EMBED_DIM)
        if mask:
            new_index.add(all_vecs[mask])

        self._index  = new_index
        self._id_map = [self._id_map[i] for i in mask]
        logger.info("Deleted file_id='%s' (remaining=%d)", file_id, len(self._id_map))

    # ── query ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Find the top-k most semantically similar files.

        Args:
            query_embedding: 384-d float list from Embedder.embed_query().
            top_k:           Number of results to return.
            min_score:       Minimum cosine similarity threshold (0–1).
                             Filters out weak matches.

        Returns:
            List of (file_id, score) tuples, sorted by descending score.
            Score is cosine similarity in [0, 1] (higher = more similar).
        """
        if self._index.ntotal == 0:
            logger.warning("Search called on empty index — returning []")
            return []

        effective_k = min(top_k, self._index.ntotal)
        q = _to_float32_matrix([query_embedding])              # (1, 384)

        scores, indices = self._index.search(q, effective_k)   # (1, k), (1, k)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        results = []
        for idx, score in zip(indices, scores):
            if idx == -1:           # FAISS padding for empty slots
                continue
            if score < min_score:
                continue
            results.append((self._id_map[idx], float(score)))

        logger.debug("Search returned %d results (top score=%.3f)", len(results), results[0][1] if results else 0)
        return results

    # ── helpers ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._index.ntotal

    def contains(self, file_id: str) -> bool:
        return file_id in self._id_map


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_float32_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Convert list-of-lists to a float32 numpy matrix expected by FAISS."""
    mat = np.array(embeddings, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected shape (n, {EMBED_DIM}), got {mat.shape}. "
            "Make sure you're using the correct embedding model."
        )
    return mat

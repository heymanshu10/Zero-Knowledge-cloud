"""
search_api.py — FastAPI Search Endpoint
ZK Cloud Storage · Server Module

Exposes two endpoints consumed by the React client:
  POST /embeddings          — store a file embedding on upload
  POST /search              — ANN query, returns matching file_ids
  DELETE /embeddings/{id}   — remove embedding on file deletion (FR-07)

Zero-knowledge guarantee:
  - No plaintext, filenames, or keys are received or stored.
  - The server sees only (file_id, embedding_vector) pairs.
  - Query embeddings are discarded immediately after search.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── shared state ───────────────────────────────────────────────────────────────

store: VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store on startup; save on shutdown."""
    global store
    store = VectorStore.load()
    logger.info("VectorStore ready (%d vectors)", len(store))
    yield
    store.save()
    logger.info("VectorStore saved on shutdown.")


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ZK Cloud Storage — Search API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

EMBED_DIM = 384


class StoreEmbeddingRequest(BaseModel):
    file_id:   str            = Field(..., description="UUID of the encrypted file")
    embedding: list[float]    = Field(..., min_length=EMBED_DIM, max_length=EMBED_DIM,
                                      description="384-d semantic embedding vector")


class StoreEmbeddingResponse(BaseModel):
    file_id: str
    status:  str = "stored"


class SearchRequest(BaseModel):
    query_embedding: list[float] = Field(
        ..., min_length=EMBED_DIM, max_length=EMBED_DIM,
        description="384-d query embedding generated on client"
    )
    top_k:     int   = Field(default=10, ge=1,  le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0,
                             description="Minimum cosine similarity (0 = return all, 1 = exact match)")


class SearchResult(BaseModel):
    file_id: str
    score:   float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DeleteResponse(BaseModel):
    file_id: str
    status:  str = "deleted"


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/embeddings",
    response_model=StoreEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store a file embedding (called on upload)",
)
def store_embedding(req: StoreEmbeddingRequest) -> StoreEmbeddingResponse:
    """
    Called by the client after encrypting a file.
    Stores the embedding vector keyed by file_id.
    The server never receives or stores the plaintext or encryption key.
    """
    if store.contains(req.file_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Embedding for file_id '{req.file_id}' already exists.",
        )
    try:
        store.add(req.file_id, req.embedding)
        store.save()
    except Exception as exc:
        logger.exception("Failed to store embedding for file_id='%s'", req.file_id)
        raise HTTPException(status_code=500, detail="Failed to store embedding.") from exc

    return StoreEmbeddingResponse(file_id=req.file_id)


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic ANN search (called on user query)",
)
def search(req: SearchRequest) -> SearchResponse:
    """
    Performs approximate nearest-neighbour search over stored embeddings.

    The client sends a query embedding (generated locally — query text
    never leaves the device).  The server returns only file_ids and
    similarity scores — never plaintext or filenames.
    """
    try:
        raw_results = store.search(
            query_embedding=req.query_embedding,
            top_k=req.top_k,
            min_score=req.min_score,
        )
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed.") from exc

    results = [SearchResult(file_id=fid, score=score) for fid, score in raw_results]
    return SearchResponse(results=results)


@app.delete(
    "/embeddings/{file_id}",
    response_model=DeleteResponse,
    summary="Remove embedding on file deletion (FR-07)",
)
def delete_embedding(file_id: str) -> DeleteResponse:
    """
    Purges the embedding for a deleted file.
    Must be called alongside deletion of the ciphertext from blob storage.
    """
    try:
        store.delete(file_id)
        store.save()
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No embedding found for file_id '{file_id}'.",
        )
    except Exception as exc:
        logger.exception("Failed to delete embedding for file_id='%s'", file_id)
        raise HTTPException(status_code=500, detail="Failed to delete embedding.") from exc

    return DeleteResponse(file_id=file_id)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "total_vectors": len(store)}

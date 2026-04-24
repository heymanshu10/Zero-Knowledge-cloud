"""
embedder.py — Semantic Embedding Pipeline
ZK Cloud Storage · Data Science Module

Responsibilities (per SRS FR-03, FR-05):
  - Extract text from PDF / DOCX / TXT / MD files (client-side)
  - Generate 384-d semantic embedding vectors using a locally-run model
  - Encode search queries into embeddings for ANN similarity search

The server NEVER sees plaintext or query text.
All embedding generation happens on the client device.
Output is passed to crypto.py for encryption + upload.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from sentence_transformers import SentenceTransformer

# ── optional extractors ────────────────────────────────────────────────────────
try:
    import pdfplumber
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False

# ── logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"   # 384-d, fast, runs fully offline
EMBED_DIM    = 384
MAX_CHARS    = 50_000                # cap text fed to encoder (NF-04 compliance)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ══════════════════════════════════════════════════════════════════════════════
# Text extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF byte buffer using pdfplumber."""
    if not _PDF_OK:
        raise ImportError("pdfplumber is required for PDF support: pip install pdfplumber")
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_docx(data: bytes) -> str:
    """Extract text from a DOCX byte buffer using python-docx."""
    if not _DOCX_OK:
        raise ImportError("python-docx is required for DOCX support: pip install python-docx")
    doc = DocxDocument(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _extract_text(data: bytes) -> str:
    """Decode plain text / markdown bytes."""
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode file as text.")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from a supported file type.

    Args:
        file_bytes: Raw file content (bytes).
        filename:   Original filename — used to detect extension.

    Returns:
        Extracted text string (may be empty if file has no text layer).

    Raises:
        ValueError: Unsupported file type.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf(file_bytes)
    elif ext == ".docx":
        text = _extract_docx(file_bytes)
    elif ext in (".txt", ".md"):
        text = _extract_text(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    logger.debug("Extracted %d chars from '%s'", len(text), filename)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Embedder
# ══════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Wraps a locally-cached sentence-transformer model.

    The model is loaded once and reused for all embed() / embed_query() calls.
    No data is sent to any remote API (SRS C-01, NF-03).

    Usage:
        embedder = Embedder()

        # --- file upload flow ---
        text      = extract_text(file_bytes, "report.pdf")
        embedding = embedder.embed(text)          # list[float], len=384
        # → pass embedding to crypto.encrypt_file() then upload payload

        # --- search flow ---
        q_vec = embedder.embed_query("quarterly revenue summary")
        # → send q_vec to server ANN endpoint
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Model loaded. Embedding dimension: %d", EMBED_DIM)

    # ── public API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Generate a semantic embedding for a document.

        Args:
            text: Extracted plaintext of the document.

        Returns:
            384-dimensional float list ready for the upload payload.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text — file may have no text layer.")

        truncated = text[:MAX_CHARS]
        if len(text) > MAX_CHARS:
            logger.warning(
                "Text truncated from %d to %d chars for embedding.", len(text), MAX_CHARS
            )

        vector = self._model.encode(truncated, normalize_embeddings=True)
        return vector.tolist()   # list[float], len=384 ✓

    def embed_query(self, query: str) -> list[float]:
        """
        Generate a semantic embedding for a search query.

        Identical to embed() but kept separate so callers are explicit
        about intent (document vs. query).  Both are L2-normalised so
        cosine similarity == dot product on the server side.

        Args:
            query: Natural-language search string.

        Returns:
            384-dimensional float list to send to /search endpoint.
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        vector = self._model.encode(query.strip(), normalize_embeddings=True)
        return vector.tolist()

    def embed_file(self, file_bytes: bytes, filename: str) -> list[float]:
        """
        Convenience method: extract text then embed in one call.

        Args:
            file_bytes: Raw file content.
            filename:   Original filename (used for extension detection).

        Returns:
            384-dimensional float list.
        """
        text = extract_text(file_bytes, filename)
        return self.embed(text)

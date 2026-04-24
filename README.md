# Semantic Search Module — ZK Cloud Storage

This module handles **all meaning-based search** for the project.  
The server never sees plaintext, filenames, queries, or keys.

## Setup

```bash
pip install -r requirements.txt
pytest test_semantic_search.py -v   # all tests must pass
```

---

## Architecture

```
CLIENT DEVICE                          SERVER
─────────────────────────────          ──────────────────────────
┌─────────────────────────┐            ┌──────────────────────────┐
│ embedder.py             │            │ search_api.py            │
│                         │            │                          │
│ 1. extract_text()       │  upload    │ POST /embeddings         │
│    → raw text           │ ─────────► │   stores (file_id, vec)  │
│                         │            │                          │
│ 2. Embedder.embed_file()│            │                          │
│    → 384-d vector       │            │ vector_store.py          │
│                         │            │   FAISS IndexFlatIP      │
│ 3. crypto.encrypt_file()│            │   cosine similarity ANN  │
│    → ciphertext + nonce │            │                          │
│                         │  search    │ POST /search             │
│ 4. Embedder.embed_query()─ ─────────►│   returns [file_ids]     │
│    (query stays local)  │            │   (no plaintext ever)    │
│                         │◄───────────│                          │
│ 5. crypto.decrypt_file()│            └──────────────────────────┘
│    → plaintext          │
└─────────────────────────┘
```

## File Overview

| File | Runs on | Responsibility |
|------|---------|----------------|
| `embedder.py` | **Client** | Text extraction + embedding generation |
| `vector_store.py` | **Server** | FAISS ANN index (add / search / delete) |
| `search_api.py` | **Server** | FastAPI REST endpoints |
| `test_semantic_search.py` | Dev | Full test suite (no GPU needed) |

---

## For the Crypto teammate

Your workflow per file upload:

```python
from embedder import Embedder
from crypto import derive_key, encrypt_file, encrypt_string

embedder = Embedder()   # load once at app startup

# --- on upload ---
key, salt = derive_key("user_password")

with open("document.pdf", "rb") as f:
    file_bytes = f.read()

# 1. Generate embedding BEFORE encryption (you need plaintext)
embedding = embedder.embed_file(file_bytes, "document.pdf")   # list[float], len=384

# 2. Encrypt the file
ciphertext, nonce = encrypt_file(file_bytes, key)

# 3. Encrypt the filename
enc_name, name_nonce = encrypt_string("document.pdf", key)

# 4. Build upload payload
payload = {
    "file_id":           "uuid-string",
    "ciphertext":        ciphertext,          # → S3
    "nonce":             nonce,               # → DB
    "embedding":         embedding,           # → vector store (via /embeddings)
    "filename_encrypted": enc_name,
    "filename_nonce":    name_nonce,
}
```

## For the Backend teammate

Start the search server:

```bash
uvicorn search_api:app --reload --port 8001
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/embeddings` | Store embedding on upload |
| `POST` | `/search` | ANN query, returns file_ids + scores |
| `DELETE` | `/embeddings/{file_id}` | Purge on file delete (FR-07) |
| `GET` | `/health` | Health check |

### POST /embeddings
```json
{ "file_id": "uuid", "embedding": [0.12, -0.44, ...] }
```

### POST /search
```json
{ "query_embedding": [0.05, 0.91, ...], "top_k": 10, "min_score": 0.3 }
```
Response:
```json
{ "results": [{ "file_id": "uuid-1", "score": 0.94 }, ...] }
```

---

## Security Notes

- The **query text never leaves the client** — only the embedding vector is sent.
- Embedding vectors **cannot be reversed** to recover original text (NF-03).  
  The model is high-dimensional (384-d) and many documents map to similar regions.
- The server stores **only** `(file_id → embedding)` — no filenames, no content.
- `min_score` defaults to `0.0` (return all). Set to `0.3`–`0.5` in production  
  to filter noise from unrelated files.

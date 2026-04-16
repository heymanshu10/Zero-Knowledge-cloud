# Crypto Module — ZK Semantic Cloud Storage

This module handles **all cryptography** for the project.
The server never sees plaintext, keys, or passwords.

## Setup

```bash
pip install -r requirements.txt
pytest test_crypto.py -v   # all tests must pass
```

---

## For the Developer (backend teammate)

You need three things from this module per file upload:

| What | How to get it | Where to store it |
|------|--------------|-------------------|
| `salt` | From `derive_key()` at registration | DB: `users.kdf_salt` (base64) |
| `nonce` | From `encrypt_file()` | DB: `files.nonce` (base64) |
| `ciphertext` | From `encrypt_file()` | S3 / blob storage |

### API payload the client sends on upload:
```json
{
  "file_id": "uuid-string",
  "ciphertext": "<base64url bytes>",
  "nonce": "<base64url bytes>",
  "embedding": [0.12, -0.44, 0.03, ...],
  "filename_encrypted": "<base64>",
  "filename_nonce": "<base64>"
}
```

### Auth endpoint must return:
```json
{ "kdf_salt": "<base64 of the user's salt>" }
```
The client uses this salt to re-derive the key on login. Never return the key itself.

---

## For the Data Science teammate

Your embedding pipeline runs **before** this module is called.

Workflow per file upload:
1. You: extract text from file → generate embedding vector (384-d float list)
2. Me: encrypt the raw file bytes → get (ciphertext, nonce)
3. Developer: upload both to server

Your output should be a plain Python list of floats, e.g.:
```python
embedding = model.encode(text).tolist()  # list[float], len=384
```
That list goes directly into the upload payload alongside the ciphertext.

---

## Quick Usage Reference

```python
from crypto import derive_key, encrypt_file, decrypt_file, encrypt_string, decrypt_string

# --- Registration ---
key, salt = derive_key("user_password")
# store salt in DB, key stays in memory only

# --- Encrypt a file ---
with open("document.pdf", "rb") as f:
    plaintext = f.read()

ciphertext, nonce = encrypt_file(plaintext, key)
# upload ciphertext to S3, store nonce in DB

# --- Encrypt filename (so server doesn't see filenames either) ---
enc_name, name_nonce = encrypt_string("document.pdf", key)
# store enc_name + name_nonce in DB

# --- Login (new session) ---
# fetch salt from server, re-derive key
key, _ = derive_key("user_password", salt=stored_salt)

# --- Decrypt file ---
plaintext = decrypt_file(ciphertext, nonce, key)

# --- Decrypt filename ---
filename = decrypt_string(enc_name, name_nonce, key)
```

---

## Security Notes

- **Never send the key to the server** — ever.
- **Never store the key** in localStorage, a file, or a database.
- Hold the key in memory only for the duration of the session.
- The `nonce` is NOT secret — store it openly in the DB.
- The `salt` is NOT secret — store it openly in the DB.
- `CryptoError` means wrong key OR tampered ciphertext — show a generic "decryption failed" message to the user, never expose which.

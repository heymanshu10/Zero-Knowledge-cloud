"""
crypto.py — Zero-Knowledge Crypto Module
=========================================
Handles all cryptographic operations for the ZK Semantic Cloud Storage system.
The server NEVER sees plaintext, keys, or passwords — everything here runs client-side.

Public interface (what your teammates use):
    derive_key(password, salt=None)  → (key, salt)
    encrypt_file(plaintext, key)     → (ciphertext, nonce)
    decrypt_file(ciphertext, nonce, key) → plaintext
    encrypt_string(text, key)        → (ciphertext_b64, nonce_b64)
    decrypt_string(ciphertext_b64, nonce_b64, key) → text
    generate_file_id()               → str (UUID)
"""

import os
import base64
import uuid
from typing import Tuple

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SALT_LENGTH = 32          # bytes — stored in DB per user, not secret
NONCE_LENGTH = 12         # bytes — AES-GCM standard nonce size
KEY_LENGTH = 32           # bytes — AES-256
KDF_ITERATIONS = 600_000  # PBKDF2-SHA256: NIST recommended minimum (2023)


# ---------------------------------------------------------------------------
# Key Derivation
# ---------------------------------------------------------------------------

def derive_key(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit AES key from a user password using PBKDF2-SHA256.

    Call with salt=None on REGISTRATION (generates a new salt).
    Call with salt=<stored_salt> on LOGIN (reproduces the same key).

    Args:
        password: The user's plaintext password (never stored or sent).
        salt:     32-byte random salt. If None, a new one is generated.

    Returns:
        (key, salt) — both as raw bytes.
        Store the salt in the database. NEVER store the key.

    Example:
        # Registration
        key, salt = derive_key("my_password")
        db.save(user_id, salt=base64.b64encode(salt))

        # Login
        salt = base64.b64decode(db.get_salt(user_id))
        key, _ = derive_key("my_password", salt=salt)
    """
    if salt is None:
        salt = os.urandom(SALT_LENGTH)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=KDF_ITERATIONS,
    )
    key = kdf.derive(password.encode("utf-8"))
    return key, salt


# ---------------------------------------------------------------------------
# File Encryption / Decryption
# ---------------------------------------------------------------------------

def encrypt_file(plaintext: bytes, key: bytes) -> Tuple[bytes, bytes]:
    """
    Encrypt a file's raw bytes using AES-256-GCM.

    Each call generates a fresh random nonce — never reuse a nonce with the same key.
    AES-GCM provides both confidentiality AND integrity (detects tampering).

    Args:
        plaintext: Raw file bytes (PDF, DOCX, TXT, etc.)
        key:       32-byte AES key from derive_key()

    Returns:
        (ciphertext, nonce) — both raw bytes.
        Store BOTH in the database / S3. The nonce is NOT secret but IS required for decryption.

    Example:
        ciphertext, nonce = encrypt_file(file_bytes, key)
        s3.upload(ciphertext)
        db.save(file_id, nonce=base64.b64encode(nonce))
    """
    if len(key) != KEY_LENGTH:
        raise ValueError(f"Key must be {KEY_LENGTH} bytes, got {len(key)}")

    nonce = os.urandom(NONCE_LENGTH)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return ciphertext, nonce


def decrypt_file(ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
    """
    Decrypt a file ciphertext using AES-256-GCM.

    Raises CryptoError if the key is wrong or the ciphertext has been tampered with.

    Args:
        ciphertext: Encrypted bytes from encrypt_file()
        nonce:      The nonce stored alongside the ciphertext
        key:        32-byte AES key from derive_key()

    Returns:
        Original plaintext bytes.

    Example:
        nonce = base64.b64decode(db.get_nonce(file_id))
        ciphertext = s3.download(file_id)
        plaintext = decrypt_file(ciphertext, nonce, key)
    """
    if len(key) != KEY_LENGTH:
        raise ValueError(f"Key must be {KEY_LENGTH} bytes, got {len(key)}")
    if len(nonce) != NONCE_LENGTH:
        raise ValueError(f"Nonce must be {NONCE_LENGTH} bytes, got {len(nonce)}")

    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        return plaintext
    except InvalidTag:
        raise CryptoError(
            "Decryption failed: wrong key or corrupted/tampered ciphertext."
        )


# ---------------------------------------------------------------------------
# String Encryption (for filenames, metadata)
# ---------------------------------------------------------------------------

def encrypt_string(text: str, key: bytes) -> Tuple[str, str]:
    """
    Encrypt a short string (e.g. filename, tag) and return base64-encoded strings.
    Safe to store directly in a database TEXT column.

    Returns:
        (ciphertext_b64, nonce_b64)
    """
    ciphertext, nonce = encrypt_file(text.encode("utf-8"), key)
    return (
        base64.b64encode(ciphertext).decode("ascii"),
        base64.b64encode(nonce).decode("ascii"),
    )


def decrypt_string(ciphertext_b64: str, nonce_b64: str, key: bytes) -> str:
    """
    Decrypt a base64-encoded encrypted string back to plaintext.

    Returns:
        Original string.
    """
    ciphertext = base64.b64decode(ciphertext_b64)
    nonce = base64.b64decode(nonce_b64)
    return decrypt_file(ciphertext, nonce, key).decode("utf-8")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_file_id() -> str:
    """Generate a random UUID for a file record. Safe to store publicly."""
    return str(uuid.uuid4())


def encode_bytes(data: bytes) -> str:
    """Encode raw bytes to a URL-safe base64 string for API transport."""
    return base64.urlsafe_b64encode(data).decode("ascii")


def decode_bytes(data: str) -> bytes:
    """Decode a URL-safe base64 string back to raw bytes."""
    return base64.urlsafe_b64decode(data.encode("ascii"))


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class CryptoError(Exception):
    """Raised when decryption fails — wrong key or tampered data."""
    pass

"""
test_crypto.py — Test Suite for crypto.py
==========================================
Run with:   pytest test_crypto.py -v

All tests must pass before handing off to teammates.
"""

import base64
import pytest
from crypto import (
    derive_key,
    encrypt_file,
    decrypt_file,
    encrypt_string,
    decrypt_string,
    generate_file_id,
    encode_bytes,
    decode_bytes,
    CryptoError,
    SALT_LENGTH,
    NONCE_LENGTH,
    KEY_LENGTH,
)


# ===========================================================================
# Key Derivation Tests
# ===========================================================================

class TestDeriveKey:

    def test_returns_correct_lengths(self):
        key, salt = derive_key("test_password")
        assert len(key) == KEY_LENGTH, f"Key should be {KEY_LENGTH} bytes"
        assert len(salt) == SALT_LENGTH, f"Salt should be {SALT_LENGTH} bytes"

    def test_same_password_same_salt_gives_same_key(self):
        """Login must reproduce the exact same key as registration."""
        password = "correct horse battery staple"
        key1, salt = derive_key(password)
        key2, _ = derive_key(password, salt=salt)
        assert key1 == key2, "Same password + salt must always give same key"

    def test_different_salts_give_different_keys(self):
        """Two registrations with same password must produce different keys."""
        password = "same_password"
        key1, salt1 = derive_key(password)
        key2, salt2 = derive_key(password)
        assert salt1 != salt2, "Each registration must generate a unique salt"
        assert key1 != key2, "Different salts must produce different keys"

    def test_different_passwords_give_different_keys(self):
        _, salt = derive_key("password_a")
        key1, _ = derive_key("password_a", salt=salt)
        key2, _ = derive_key("password_b", salt=salt)
        assert key1 != key2

    def test_auto_generates_salt_when_none(self):
        key, salt = derive_key("password")
        assert salt is not None
        assert len(salt) == SALT_LENGTH

    def test_uses_provided_salt(self):
        custom_salt = b"\xAB" * SALT_LENGTH
        key, returned_salt = derive_key("password", salt=custom_salt)
        assert returned_salt == custom_salt

    def test_unicode_password(self):
        """Passwords with unicode characters must work."""
        key, salt = derive_key("pässwörd🔐")
        key2, _ = derive_key("pässwörd🔐", salt=salt)
        assert key == key2

    def test_empty_password(self):
        """Empty password is weak but must not crash."""
        key, salt = derive_key("")
        assert len(key) == KEY_LENGTH

    def test_long_password(self):
        long_password = "a" * 10_000
        key, salt = derive_key(long_password)
        assert len(key) == KEY_LENGTH


# ===========================================================================
# File Encryption / Decryption Tests
# ===========================================================================

class TestEncryptDecryptFile:

    @pytest.fixture
    def key(self):
        k, _ = derive_key("test_password_for_fixtures")
        return k

    def test_round_trip_bytes(self, key):
        """Encrypt then decrypt must return original bytes."""
        original = b"Hello, this is a secret document!"
        ciphertext, nonce = encrypt_file(original, key)
        recovered = decrypt_file(ciphertext, nonce, key)
        assert recovered == original

    def test_round_trip_empty_bytes(self, key):
        original = b""
        ciphertext, nonce = encrypt_file(original, key)
        recovered = decrypt_file(ciphertext, nonce, key)
        assert recovered == original

    def test_round_trip_large_file(self, key):
        """Test with a ~5 MB payload (typical document size)."""
        original = b"X" * (5 * 1024 * 1024)
        ciphertext, nonce = encrypt_file(original, key)
        recovered = decrypt_file(ciphertext, nonce, key)
        assert recovered == original

    def test_round_trip_binary_content(self, key):
        """Must work for any byte sequence, not just ASCII."""
        original = bytes(range(256)) * 100
        ciphertext, nonce = encrypt_file(original, key)
        recovered = decrypt_file(ciphertext, nonce, key)
        assert recovered == original

    def test_ciphertext_differs_from_plaintext(self, key):
        original = b"sensitive content"
        ciphertext, _ = encrypt_file(original, key)
        assert ciphertext != original

    def test_ciphertext_longer_than_plaintext(self, key):
        """AES-GCM appends a 16-byte auth tag."""
        original = b"hello"
        ciphertext, _ = encrypt_file(original, key)
        assert len(ciphertext) == len(original) + 16

    def test_nonce_is_random_each_call(self, key):
        """Each encryption call must produce a unique nonce."""
        original = b"same content"
        _, nonce1 = encrypt_file(original, key)
        _, nonce2 = encrypt_file(original, key)
        assert nonce1 != nonce2

    def test_same_content_different_ciphertext(self, key):
        """Random nonce means same plaintext encrypts differently each time."""
        original = b"same content"
        ct1, _ = encrypt_file(original, key)
        ct2, _ = encrypt_file(original, key)
        assert ct1 != ct2

    def test_nonce_length(self, key):
        _, nonce = encrypt_file(b"data", key)
        assert len(nonce) == NONCE_LENGTH

    def test_wrong_key_raises_crypto_error(self, key):
        original = b"secret"
        ciphertext, nonce = encrypt_file(original, key)

        wrong_key, _ = derive_key("completely_wrong_password")
        with pytest.raises(CryptoError):
            decrypt_file(ciphertext, nonce, wrong_key)

    def test_tampered_ciphertext_raises_crypto_error(self, key):
        original = b"secret document"
        ciphertext, nonce = encrypt_file(original, key)

        # Flip one byte in the ciphertext
        tampered = bytearray(ciphertext)
        tampered[5] ^= 0xFF
        with pytest.raises(CryptoError):
            decrypt_file(bytes(tampered), nonce, key)

    def test_wrong_nonce_raises_crypto_error(self, key):
        original = b"secret"
        ciphertext, _ = encrypt_file(original, key)

        import os
        wrong_nonce = os.urandom(NONCE_LENGTH)
        with pytest.raises(CryptoError):
            decrypt_file(ciphertext, wrong_nonce, key)

    def test_invalid_key_length_raises_value_error(self):
        with pytest.raises(ValueError):
            encrypt_file(b"data", key=b"short_key")

    def test_invalid_nonce_length_raises_value_error(self, key):
        ciphertext, _ = encrypt_file(b"data", key)
        with pytest.raises(ValueError):
            decrypt_file(ciphertext, nonce=b"bad", key=key)


# ===========================================================================
# String Encryption Tests
# ===========================================================================

class TestEncryptDecryptString:

    @pytest.fixture
    def key(self):
        k, _ = derive_key("string_test_password")
        return k

    def test_round_trip_ascii(self, key):
        text = "report_q3_2024.pdf"
        ct, nonce = encrypt_string(text, key)
        recovered = decrypt_string(ct, nonce, key)
        assert recovered == text

    def test_round_trip_unicode(self, key):
        text = "机密文件_2024.pdf"
        ct, nonce = encrypt_string(text, key)
        recovered = decrypt_string(ct, nonce, key)
        assert recovered == text

    def test_returns_base64_strings(self, key):
        ct, nonce = encrypt_string("filename.txt", key)
        # Should not raise — valid base64
        base64.b64decode(ct)
        base64.b64decode(nonce)

    def test_wrong_key_raises_crypto_error(self, key):
        ct, nonce = encrypt_string("secret.txt", key)
        wrong_key, _ = derive_key("wrong")
        with pytest.raises(CryptoError):
            decrypt_string(ct, nonce, wrong_key)


# ===========================================================================
# Utility Tests
# ===========================================================================

class TestUtilities:

    def test_generate_file_id_is_unique(self):
        ids = {generate_file_id() for _ in range(1000)}
        assert len(ids) == 1000, "All generated IDs must be unique"

    def test_generate_file_id_is_string(self):
        fid = generate_file_id()
        assert isinstance(fid, str)
        assert len(fid) == 36  # UUID4 format: 8-4-4-4-12

    def test_encode_decode_bytes_round_trip(self):
        import os
        original = os.urandom(64)
        encoded = encode_bytes(original)
        decoded = decode_bytes(encoded)
        assert decoded == original

    def test_encode_bytes_returns_string(self):
        result = encode_bytes(b"\x00\xFF\xAB")
        assert isinstance(result, str)


# ===========================================================================
# Integration Test — Full Upload/Download Simulation
# ===========================================================================

class TestIntegration:

    def test_full_upload_download_flow(self):
        """
        Simulates the complete workflow:
        1. User registers → derive key + salt
        2. User uploads file → encrypt
        3. Server stores ciphertext + nonce (simulated with variables)
        4. User logs back in → re-derive key from password + stored salt
        5. User downloads → decrypt
        """
        password = "super_secret_password_123"
        original_file = b"%PDF-1.4 ... fake PDF content ... binary \x00\x01\x02"

        # --- Registration ---
        key_at_registration, salt = derive_key(password)

        # --- Encrypt filename + file ---
        encrypted_filename, filename_nonce = encrypt_string("budget_2024.pdf", key_at_registration)
        ciphertext, file_nonce = encrypt_file(original_file, key_at_registration)

        # Simulate: server stores salt, encrypted_filename, filename_nonce, ciphertext, file_nonce
        stored_salt = salt
        stored_encrypted_filename = encrypted_filename
        stored_filename_nonce = filename_nonce
        stored_ciphertext = ciphertext
        stored_file_nonce = file_nonce

        # --- Login (new session, key must be re-derived) ---
        key_at_login, _ = derive_key(password, salt=stored_salt)

        # --- Decrypt ---
        recovered_filename = decrypt_string(stored_encrypted_filename, stored_filename_nonce, key_at_login)
        recovered_file = decrypt_file(stored_ciphertext, stored_file_nonce, key_at_login)

        assert recovered_filename == "budget_2024.pdf"
        assert recovered_file == original_file

    def test_wrong_password_at_login_fails(self):
        """A user who misremembers their password must not decrypt anything."""
        correct_password = "correct_password"
        wrong_password = "wrong_password"

        key, salt = derive_key(correct_password)
        ciphertext, nonce = encrypt_file(b"top secret", key)

        wrong_key, _ = derive_key(wrong_password, salt=salt)
        with pytest.raises(CryptoError):
            decrypt_file(ciphertext, nonce, wrong_key)

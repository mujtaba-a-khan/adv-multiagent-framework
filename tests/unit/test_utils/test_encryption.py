"""Tests for adversarial_framework.utils.encryption module."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from cryptography.fernet import InvalidToken

from adversarial_framework.utils.encryption import (
    KeyEncryptor,
    derive_key,
    generate_master_key,
    generate_nonce,
)

# Use a fixed test secret (not a real credential)
_TEST_SECRET = "test-master-secret-for-unit-tests"


# derive_key


class TestDeriveKey:
    def test_returns_key_and_salt(self):
        key, salt = derive_key(_TEST_SECRET)
        assert isinstance(key, bytes)
        assert isinstance(salt, bytes)
        assert len(salt) == 16

    def test_deterministic_with_same_salt(self):
        _, salt = derive_key(_TEST_SECRET)
        key1, _ = derive_key(_TEST_SECRET, salt=salt)
        key2, _ = derive_key(_TEST_SECRET, salt=salt)
        assert key1 == key2

    def test_different_salt_gives_different_key(self):
        key1, salt1 = derive_key(_TEST_SECRET)
        key2, salt2 = derive_key(_TEST_SECRET)
        # Random salts should differ (astronomically unlikely to match)
        if salt1 != salt2:
            assert key1 != key2

    def test_different_secret_gives_different_key(self):
        _, salt = derive_key(_TEST_SECRET)
        key1, _ = derive_key("secret-a", salt=salt)
        key2, _ = derive_key("secret-b", salt=salt)
        assert key1 != key2


# generate_master_key


class TestGenerateMasterKey:
    def test_returns_string(self):
        key = generate_master_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_unique_each_call(self):
        k1 = generate_master_key()
        k2 = generate_master_key()
        assert k1 != k2


# generate_nonce


class TestGenerateNonce:
    def test_default_length(self):
        nonce = generate_nonce()
        # 32 bytes = 64 hex characters
        assert len(nonce) == 64

    def test_custom_length(self):
        nonce = generate_nonce(length=16)
        assert len(nonce) == 32

    def test_hex_format(self):
        nonce = generate_nonce()
        int(nonce, 16)  # Should not raise

    def test_unique(self):
        n1 = generate_nonce()
        n2 = generate_nonce()
        assert n1 != n2


# KeyEncryptor


class TestKeyEncryptor:
    def test_init_with_explicit_secret(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        assert enc is not None

    def test_init_from_env_var(self):
        with mock.patch.dict(os.environ, {"ADV_ENCRYPTION_KEY": _TEST_SECRET}):
            enc = KeyEncryptor()
            assert enc is not None

    def test_init_raises_without_secret(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is absent
            os.environ.pop("ADV_ENCRYPTION_KEY", None)
            with pytest.raises(ValueError, match="required"):
                KeyEncryptor(master_secret="")

    def test_encrypt_returns_string(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        encrypted = enc.encrypt("my-api-key-12345")
        assert isinstance(encrypted, str)
        assert encrypted != "my-api-key-12345"

    def test_encrypt_empty_returns_empty(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        assert enc.encrypt("") == ""

    def test_decrypt_empty_returns_empty(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        assert enc.decrypt("") == ""

    def test_roundtrip(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        plaintext = "sk-proj-abc123XYZ"
        encrypted = enc.encrypt(plaintext)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_ciphertext_each_time(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        e1 = enc.encrypt("same-text")
        e2 = enc.encrypt("same-text")
        # Different salts produce different ciphertext
        assert e1 != e2

    def test_decrypt_wrong_key_raises(self):
        enc1 = KeyEncryptor(master_secret="secret-one-for-test")
        enc2 = KeyEncryptor(master_secret="secret-two-for-test")
        encrypted = enc1.encrypt("my-secret-value")
        with pytest.raises(InvalidToken):
            enc2.decrypt(encrypted)

    def test_rotate(self):
        old_secret = "old-secret-for-rotation-test"
        new_secret = "new-secret-for-rotation-test"

        enc_old = KeyEncryptor(master_secret=old_secret)
        encrypted_old = enc_old.encrypt("my-api-key")

        encrypted_new = enc_old.rotate(encrypted_old, new_secret)

        enc_new = KeyEncryptor(master_secret=new_secret)
        decrypted = enc_new.decrypt(encrypted_new)
        assert decrypted == "my-api-key"

    def test_rotate_preserves_plaintext(self):
        enc = KeyEncryptor(master_secret=_TEST_SECRET)
        new_secret = "rotated-secret-for-tests"
        original = "sensitive-api-key-value"

        encrypted = enc.encrypt(original)
        rotated = enc.rotate(encrypted, new_secret)

        enc_new = KeyEncryptor(master_secret=new_secret)
        assert enc_new.decrypt(rotated) == original

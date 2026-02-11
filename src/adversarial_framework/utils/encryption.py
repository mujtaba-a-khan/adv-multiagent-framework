"""API key encryption utilities using Fernet symmetric encryption.

Provides encrypt/decrypt for API keys stored at rest in database
or configuration files. Keys are derived from a master secret using
PBKDF2-HMAC-SHA256.
"""

from __future__ import annotations

import base64
import os
import secrets

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def derive_key(master_secret: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a Fernet key from a master secret using PBKDF2.

    Args:
        master_secret: The master secret/password.
        salt: Optional salt; generated if not provided.

    Returns:
        Tuple of (fernet_key_bytes, salt_bytes).
    """
    if salt is None:
        salt = os.urandom(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(master_secret.encode()))
    return key, salt


def generate_master_key() -> str:
    """Generate a cryptographically secure random master key.

    Returns:
        A URL-safe base64-encoded 32-byte key string.
    """
    return Fernet.generate_key().decode()


class KeyEncryptor:
    """Encrypts and decrypts API keys using Fernet symmetric encryption.

    The encryption key is derived from a master secret via PBKDF2.
    Salt is prepended to the ciphertext for storage.

    Args:
        master_secret: Secret used to derive the encryption key.
            If not provided, reads from ``ADV_ENCRYPTION_KEY`` env var.
    """

    # Salt is 16 bytes, prepended to ciphertext
    _SALT_LENGTH = 16

    def __init__(self, master_secret: str | None = None) -> None:
        self._master_secret = master_secret or os.environ.get("ADV_ENCRYPTION_KEY", "")
        if not self._master_secret:
            raise ValueError(
                "Encryption master secret is required. "
                "Set ADV_ENCRYPTION_KEY environment variable or pass master_secret."
            )

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext API key.

        Returns:
            Base64-encoded string containing salt + ciphertext.
        """
        if not plaintext:
            return ""

        key, salt = derive_key(self._master_secret)
        f = Fernet(key)
        ciphertext = f.encrypt(plaintext.encode())
        # Prepend salt for storage
        combined = salt + ciphertext
        return base64.urlsafe_b64encode(combined).decode()

    def decrypt(self, encrypted: str) -> str:
        """Decrypt an encrypted API key.

        Args:
            encrypted: Base64-encoded string from ``encrypt()``.

        Returns:
            The original plaintext string.

        Raises:
            InvalidToken: If decryption fails (wrong key or corrupted data).
        """
        if not encrypted:
            return ""

        combined = base64.urlsafe_b64decode(encrypted.encode())
        salt = combined[: self._SALT_LENGTH]
        ciphertext = combined[self._SALT_LENGTH :]

        key, _ = derive_key(self._master_secret, salt=salt)
        f = Fernet(key)
        return f.decrypt(ciphertext).decode()

    def rotate(self, encrypted: str, new_master_secret: str) -> str:
        """Re-encrypt a value with a new master secret.

        Args:
            encrypted: Value encrypted with the current master secret.
            new_master_secret: New master secret to encrypt with.

        Returns:
            Value re-encrypted with the new master secret.
        """
        plaintext = self.decrypt(encrypted)
        new_encryptor = KeyEncryptor(master_secret=new_master_secret)
        return new_encryptor.encrypt(plaintext)


def generate_nonce(length: int = 32) -> str:
    """Generate a cryptographically secure random hex nonce."""
    return secrets.token_hex(length)

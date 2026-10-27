"""Authenticated application-layer encryption with key rotation support."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken, MultiFernet


class EncryptionError(ValueError):
    """Raised when encrypted data cannot be authenticated or decoded."""


class EncryptedJsonStore:
    """Encrypt JSON with the first key and decrypt with any configured key.

    Keys are Fernet URL-safe base64 values. Rotation adds the new key first and
    retains previous keys only for decryption until data is re-encrypted.
    """

    def __init__(self, comma_separated_keys: str) -> None:
        keys = [key.strip().encode() for key in comma_separated_keys.split(",") if key.strip()]
        if not keys:
            raise ValueError("at least one data-encryption key is required")
        try:
            self._cipher = MultiFernet([Fernet(key) for key in keys])
            self._digest_keys = [
                hashlib.sha256(
                    b"gatekeeper-blind-index-v1\0" + base64.urlsafe_b64decode(key)
                ).digest()
                for key in keys
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError("data-encryption keys must be valid Fernet keys") from exc

    def encrypt(self, value: dict[str, Any] | list[Any] | str) -> bytes:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self._cipher.encrypt(serialized)

    def decrypt(self, token: bytes) -> Any:
        try:
            plaintext = self._cipher.decrypt(token)
            return json.loads(plaintext)
        except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EncryptionError("encrypted data failed authentication or decoding") from exc

    def rotate(self, token: bytes) -> bytes:
        try:
            return self._cipher.rotate(token)
        except InvalidToken as exc:
            raise EncryptionError("encrypted data failed authentication") from exc

    def digest(self, value: str, *, namespace: str, organization_id: str) -> str:
        """Create a tenant/context-bound blind index with the active key."""

        material = f"{namespace}\x1f{organization_id}\x1f{value}".encode()
        return hmac.new(self._digest_keys[0], material, hashlib.sha256).hexdigest()

    def matches_digest(
        self,
        value: str,
        expected: str,
        *,
        namespace: str,
        organization_id: str,
    ) -> bool:
        """Verify an index made by the active or a retained rotation key."""

        material = f"{namespace}\x1f{organization_id}\x1f{value}".encode()
        return any(
            hmac.compare_digest(
                hmac.new(key, material, hashlib.sha256).hexdigest(),
                expected,
            )
            for key in self._digest_keys
        )

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from backend.app.security import EncryptedJsonStore, EncryptionError


def test_encrypted_json_round_trip_does_not_expose_clinical_text() -> None:
    store = EncryptedJsonStore(Fernet.generate_key().decode())
    clinical_text = "Patient reports left knee pain."
    payload = {"content": clinical_text, "start": 0}

    encrypted = store.encrypt(payload)

    assert clinical_text.encode() not in encrypted
    assert store.decrypt(encrypted) == payload


def test_key_rotation_decrypts_old_ciphertext_and_reencrypts_with_new_key() -> None:
    old_key = Fernet.generate_key().decode()
    new_key = Fernet.generate_key().decode()
    old_store = EncryptedJsonStore(old_key)
    rotating_store = EncryptedJsonStore(f"{new_key},{old_key}")
    original = old_store.encrypt({"secret": "value"})

    rotated = rotating_store.rotate(original)

    assert rotating_store.decrypt(rotated) == {"secret": "value"}
    with pytest.raises(EncryptionError):
        old_store.decrypt(rotated)


def test_tampered_ciphertext_fails_authentication() -> None:
    store = EncryptedJsonStore(Fernet.generate_key().decode())
    encrypted = bytearray(store.encrypt("clinical content"))
    encrypted[-5] ^= 1

    with pytest.raises(EncryptionError):
        store.decrypt(bytes(encrypted))


def test_blind_indexes_are_tenant_scoped_and_rotation_verifiable() -> None:
    old_key = Fernet.generate_key().decode()
    new_key = Fernet.generate_key().decode()
    old = EncryptedJsonStore(old_key)
    old_digest = old.digest("left knee pain", namespace="content", organization_id="org-a")
    rotated = EncryptedJsonStore(f"{new_key},{old_key}")

    assert "left knee pain" not in old_digest
    assert old_digest != old.digest("left knee pain", namespace="content", organization_id="org-b")
    assert rotated.matches_digest(
        "left knee pain",
        old_digest,
        namespace="content",
        organization_id="org-a",
    )

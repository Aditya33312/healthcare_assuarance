"""Tenant-scoped idempotency for mutation endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.models import IdempotencyRecord


def _canonical_fingerprint_json(value: Any) -> str:
    """Match ordinary canonical JSON while preserving Decimal as a number."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("idempotency payload numeric values must be finite")
        if len(value.as_tuple().digits) > 100 or (value and not -100 <= value.adjusted() <= 100):
            raise ValueError("idempotency payload numeric values exceed safe bounds")
        return format(value, "f")
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{json.dumps(str(key), ensure_ascii=False)}:{_canonical_fingerprint_json(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_fingerprint_json(item) for item in value) + "]"
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def request_fingerprint(payload: dict[str, Any], fingerprint_key: str) -> str:
    canonical = _canonical_fingerprint_json(payload)
    if len(fingerprint_key) < 16:
        raise ValueError("idempotency fingerprint key must contain at least 16 characters")
    return hmac.new(
        fingerprint_key.encode(),
        b"gatekeeper-idempotency-v1\0" + canonical.encode(),
        hashlib.sha256,
    ).hexdigest()


def idempotency_key_digest(key: str, fingerprint_key: str) -> str:
    if len(fingerprint_key) < 16:
        raise ValueError("idempotency fingerprint key must contain at least 16 characters")
    return hmac.new(
        fingerprint_key.encode(),
        b"gatekeeper-idempotency-key-v1\0" + key.encode(),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class IdempotencyReplay:
    resource_type: str
    resource_id: str
    response_status: int
    response_body: dict[str, Any]


class IdempotencyService:
    def __init__(self, session: Session, fingerprint_key: str) -> None:
        self.session = session
        self.fingerprint_key = fingerprint_key

    def find(
        self,
        *,
        organization_id: str,
        operation: str,
        key: str,
        payload: dict[str, Any],
    ) -> IdempotencyReplay | None:
        self._validate_key(key)
        key_digest = idempotency_key_digest(key, self.fingerprint_key)
        record = self.session.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.organization_id == organization_id,
                IdempotencyRecord.operation == operation,
                IdempotencyRecord.key_digest == key_digest,
            )
        )
        if record is None:
            return None
        expires_at = record.expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= datetime.now(UTC):
                self.session.delete(record)
                self.session.flush()
                return None
        if record.request_hash != request_fingerprint(payload, self.fingerprint_key):
            raise GatekeeperError(
                ErrorCode.REQ_IDEMPOTENCY_CONFLICT,
                "The Idempotency-Key was already used with a different request.",
                http_status=409,
            )
        return IdempotencyReplay(
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            response_status=record.response_status,
            response_body=record.response_body,
        )

    def record(
        self,
        *,
        organization_id: str,
        operation: str,
        key: str,
        payload: dict[str, Any],
        resource_type: str,
        resource_id: str,
        response_status: int,
        response_body: dict[str, Any] | None = None,
        ttl: timedelta = timedelta(hours=24),
    ) -> IdempotencyRecord:
        self._validate_key(key)
        if ttl <= timedelta(0):
            raise ValueError("idempotency TTL must be positive")
        now = datetime.now(UTC)
        record = IdempotencyRecord(
            organization_id=organization_id,
            operation=operation,
            key_digest=idempotency_key_digest(key, self.fingerprint_key),
            request_hash=request_fingerprint(payload, self.fingerprint_key),
            resource_type=resource_type,
            resource_id=resource_id,
            response_status=response_status,
            # Callers should normally leave this empty and reload the resource.
            # Clinical request or response content must never be placed here.
            response_body=response_body or {},
            created_at=now,
            expires_at=now + ttl,
        )
        self.session.add(record)
        self.session.flush()
        return record

    @staticmethod
    def _validate_key(key: str) -> None:
        if not 1 <= len(key) <= 200 or any(ord(character) < 33 for character in key):
            raise GatekeeperError(
                ErrorCode.REQ_INVALID,
                "Idempotency-Key must contain 1-200 visible characters.",
                http_status=422,
            )

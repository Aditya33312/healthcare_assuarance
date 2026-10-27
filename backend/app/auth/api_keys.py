"""Generate, hash, authenticate, scope-check, and revoke Gatekeeper API keys."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.scopes import ALL_SCOPES, ApiScope
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.models import ApiKey, Organization

ApiKeyEnvironment = Literal["test", "live"]
_PREFIX_LENGTH = 20


@dataclass(frozen=True, slots=True)
class CreatedApiKey:
    record: ApiKey
    secret: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ApiKeyPrincipal:
    key_id: str
    organization_id: str
    environment: ApiKeyEnvironment
    scopes: frozenset[str]

    def require(self, *required_scopes: ApiScope | str) -> None:
        required = {
            scope.value if isinstance(scope, ApiScope) else scope for scope in required_scopes
        }
        if not required.issubset(self.scopes):
            raise GatekeeperError(
                ErrorCode.AUTH_INSUFFICIENT_SCOPE,
                "The API key does not have the required scope.",
                http_status=403,
            )


def _hash_secret(secret: str, pepper: str) -> bytes:
    if len(pepper) < 16:
        raise ValueError("API key pepper must contain at least 16 characters")
    return hmac.new(pepper.encode(), secret.encode(), hashlib.sha256).digest()


def _validate_scopes(scopes: set[ApiScope | str]) -> list[str]:
    normalized = sorted(scope.value if isinstance(scope, ApiScope) else scope for scope in scopes)
    unknown = set(normalized) - ALL_SCOPES
    if unknown:
        raise ValueError(f"unsupported API scopes: {', '.join(sorted(unknown))}")
    if not normalized:
        raise ValueError("at least one API scope is required")
    return normalized


def create_api_key(
    *,
    organization_id: str,
    environment: ApiKeyEnvironment,
    scopes: set[ApiScope | str],
    pepper: str,
    expires_at: datetime | None = None,
) -> CreatedApiKey:
    """Create an API key whose plaintext is returned exactly once."""

    if environment not in {"test", "live"}:
        raise ValueError("API key environment must be 'test' or 'live'")
    environment_prefix = "gk_test" if environment == "test" else "gk_live"
    secret = f"{environment_prefix}_{secrets.token_urlsafe(32)}"
    record = ApiKey(
        organization_id=organization_id,
        key_prefix=secret[:_PREFIX_LENGTH],
        key_hash=_hash_secret(secret, pepper),
        scopes=_validate_scopes(scopes),
        environment=environment,
        status="ACTIVE",
        expires_at=expires_at,
    )
    return CreatedApiKey(record=record, secret=secret)


def revoke_api_key(record: ApiKey, *, at: datetime | None = None) -> None:
    timestamp = at or datetime.now(UTC)
    record.status = "REVOKED"
    record.revoked_at = timestamp


class ApiKeyAuthenticator:
    def __init__(self, session: Session, pepper: str) -> None:
        self.session = session
        self.pepper = pepper

    def authenticate(
        self,
        secret: str,
        *,
        required_scopes: set[ApiScope | str] | None = None,
    ) -> ApiKeyPrincipal:
        if not secret.startswith(("gk_test_", "gk_live_")):
            raise self._invalid_key()
        digest = _hash_secret(secret, self.pepper)
        candidates = self.session.scalars(
            select(ApiKey)
            .join(Organization, Organization.id == ApiKey.organization_id)
            .where(
                ApiKey.key_prefix == secret[:_PREFIX_LENGTH],
                Organization.status == "ACTIVE",
            )
        )
        record = next(
            (
                candidate
                for candidate in candidates
                if hmac.compare_digest(candidate.key_hash, digest)
            ),
            None,
        )
        if record is None:
            raise self._invalid_key()
        if record.status == "REVOKED" or record.revoked_at is not None:
            raise GatekeeperError(
                ErrorCode.AUTH_REVOKED_KEY,
                "The API key has been revoked.",
                http_status=401,
            )
        if record.status != "ACTIVE":
            raise self._invalid_key()
        now = datetime.now(UTC)
        expires_at = record.expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now:
                raise self._invalid_key()
        principal = ApiKeyPrincipal(
            key_id=record.id,
            organization_id=record.organization_id,
            environment=record.environment,  # type: ignore[arg-type]
            scopes=frozenset(record.scopes),
        )
        if required_scopes:
            principal.require(*required_scopes)
        record.last_used_at = now
        self.session.flush()
        return principal

    @staticmethod
    def _invalid_key() -> GatekeeperError:
        return GatekeeperError(
            ErrorCode.AUTH_INVALID_KEY,
            "The API key is invalid or expired.",
            http_status=401,
        )

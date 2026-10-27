"""One-time bootstrap for the first organization and administrative API key."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth.api_keys import ApiKeyEnvironment, CreatedApiKey, create_api_key
from backend.app.auth.scopes import ApiScope
from backend.app.database.models import ApiKey, Organization

_MAX_ORGANIZATION_NAME_LENGTH = 200
_POSTGRES_BOOTSTRAP_LOCK_ID = 0x474B425354525050


class BootstrapAlreadyInitializedError(RuntimeError):
    """Raised when a database is not eligible for first-tenant bootstrap."""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    organization: Organization
    api_key: CreatedApiKey = field(repr=False)


def _validated_organization_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("organization name must not be empty")
    if len(normalized) > _MAX_ORGANIZATION_NAME_LENGTH:
        raise ValueError(
            f"organization name must contain at most {_MAX_ORGANIZATION_NAME_LENGTH} characters"
        )
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("organization name must not contain control characters")
    return normalized


def _serialize_postgres_bootstrap(session: Session) -> None:
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _POSTGRES_BOOTSTRAP_LOCK_ID},
        )


def bootstrap_first_admin(
    session: Session,
    *,
    organization_name: str,
    environment: ApiKeyEnvironment,
    pepper: str,
) -> BootstrapResult:
    """Create the first active tenant and an all-scopes API key atomically.

    The caller must provide a fresh session with no active transaction. The
    returned plaintext key exists only in ``BootstrapResult.api_key.secret`` and
    cannot be recovered from persistence.
    """

    name = _validated_organization_name(organization_name)
    if session.in_transaction():
        raise ValueError("bootstrap requires a database session with no active transaction")

    try:
        with session.begin():
            _serialize_postgres_bootstrap(session)
            organization_count = (
                session.scalar(select(func.count(Organization.id)).select_from(Organization)) or 0
            )
            api_key_count = session.scalar(select(func.count(ApiKey.id)).select_from(ApiKey)) or 0
            if organization_count != 0 or api_key_count != 0:
                raise BootstrapAlreadyInitializedError(
                    "bootstrap is allowed only when organizations and API keys are both empty"
                )

            organization = Organization(name=name, status="ACTIVE")
            session.add(organization)
            session.flush()

            created_key = create_api_key(
                organization_id=organization.id,
                environment=environment,
                scopes=set(ApiScope),
                pepper=pepper,
            )
            session.add(created_key.record)
            session.flush()
    except IntegrityError as exc:
        raise BootstrapAlreadyInitializedError(
            "bootstrap could not acquire an empty database; no records were created"
        ) from exc

    return BootstrapResult(organization=organization, api_key=created_key)

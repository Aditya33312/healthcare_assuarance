"""Tenant-scoped API-key lifecycle endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_api_settings, get_db_session, require_scopes
from backend.app.api.errors import ApiError
from backend.app.api.schemas import ApiKeyCreate
from backend.app.auditing import record_audit_event
from backend.app.auth import (
    ApiKeyPrincipal,
    ApiScope,
    create_api_key,
    revoke_api_key,
)
from backend.app.core.config import Settings
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.models import ApiKey
from backend.app.database.repositories import TenantRepository

router = APIRouter()
api_keys_manage = require_scopes(ApiScope.API_KEYS_MANAGE)


def _require_environment_access(principal: ApiKeyPrincipal, environment: str) -> None:
    if environment == "live" and principal.environment != "live":
        raise GatekeeperError(
            ErrorCode.AUTH_ENVIRONMENT_MISMATCH,
            "A live API key is required to manage live API keys.",
            http_status=403,
        )


def _payload(record: ApiKey) -> dict[str, Any]:
    def as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    return {
        "id": record.id,
        "key_id": record.id,
        "key_prefix": record.key_prefix,
        "environment": record.environment,
        "scopes": sorted(record.scopes),
        "status": record.status,
        "created_at": as_utc(record.created_at),
        "updated_at": as_utc(record.updated_at),
        "expires_at": as_utc(record.expires_at),
        "last_used_at": as_utc(record.last_used_at),
        "revoked_at": as_utc(record.revoked_at),
    }


def _audit(
    session: Session,
    request: Request,
    principal: ApiKeyPrincipal,
    *,
    action: str,
    target_id: str,
) -> None:
    request_id = request.state.request_id
    if not isinstance(request_id, str):
        raise RuntimeError("request context did not provide a valid request ID")
    record_audit_event(
        session,
        organization_id=principal.organization_id,
        actor_type="API_KEY",
        actor_id=principal.key_id,
        action=action,
        target_type="api_key",
        target_id=target_id,
        request_id=request_id,
    )


@router.post("/api-keys", status_code=201)
def create_tenant_api_key(
    body: ApiKeyCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(api_keys_manage),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_api_settings),
) -> dict[str, Any]:
    _require_environment_access(principal, body.environment)
    created = create_api_key(
        organization_id=principal.organization_id,
        environment=body.environment,
        scopes=set(body.scopes),
        pepper=settings.api_key_pepper,
        expires_at=body.expires_at,
    )
    TenantRepository(session, ApiKey).add(principal.organization_id, created.record)
    _audit(
        session,
        request,
        principal,
        action="api_key.created",
        target_id=created.record.id,
    )
    session.commit()
    return {**_payload(created.record), "secret": created.secret}


@router.get("/api-keys")
def list_tenant_api_keys(
    principal: ApiKeyPrincipal = Depends(api_keys_manage),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    statement = (
        select(ApiKey)
        .where(ApiKey.organization_id == principal.organization_id)
        .order_by(ApiKey.created_at.desc(), ApiKey.id.desc())
        .limit(limit)
    )
    if principal.environment != "live":
        statement = statement.where(ApiKey.environment == "test")
    records = tuple(session.scalars(statement))
    session.commit()
    return {"items": [_payload(record) for record in records], "count": len(records)}


@router.post("/api-keys/{key_id}/revoke")
def revoke_tenant_api_key(
    key_id: str,
    request: Request,
    principal: ApiKeyPrincipal = Depends(api_keys_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    record = TenantRepository(session, ApiKey).get(principal.organization_id, key_id)
    if record is None:
        raise ApiError(
            "API_KEY_NOT_FOUND",
            "The requested API key was not found.",
            http_status=404,
        )
    _require_environment_access(principal, record.environment)
    if record.status != "REVOKED" or record.revoked_at is None:
        revoke_api_key(record)
        _audit(
            session,
            request,
            principal,
            action="api_key.revoked",
            target_id=record.id,
        )
    session.commit()
    return _payload(record)


__all__ = ["router"]

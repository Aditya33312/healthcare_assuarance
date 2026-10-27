"""Repositories that require an organization scope for tenant-owned data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from sqlalchemy import Select, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from backend.app.database.base import Base


class TenantScopeError(ValueError):
    """Raised when an object could cross an organization boundary."""


class _TenantModelClass(Protocol):
    """Class-level columns required by a tenant repository."""

    id: InstrumentedAttribute[str]
    organization_id: InstrumentedAttribute[str]


class TenantRepository[ModelT: Base]:
    """Minimal repository with mandatory tenant predicates.

    Callers cannot retrieve or mutate a tenant-owned row without explicitly
    providing the authenticated organization identifier.
    """

    def __init__(self, session: Session, model: type[ModelT]) -> None:
        if not hasattr(model, "organization_id"):
            raise TypeError(f"{model.__name__} is not a tenant-owned model")
        self.session = session
        self.model = model

    def add(self, organization_id: str, entity: ModelT) -> ModelT:
        entity_organization_id = getattr(entity, "organization_id", None)
        if entity_organization_id != organization_id:
            raise TenantScopeError("entity organization does not match authenticated tenant")
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, organization_id: str, entity_id: str) -> ModelT | None:
        columns = cast(_TenantModelClass, self.model)
        statement = self._scoped_select(organization_id).where(columns.id == entity_id)
        return self.session.scalar(statement)

    def list(self, organization_id: str, *, limit: int = 100) -> Sequence[ModelT]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        statement = self._scoped_select(organization_id).limit(limit)
        return tuple(self.session.scalars(statement))

    def require(self, organization_id: str, entity_id: str) -> ModelT:
        entity = self.get(organization_id, entity_id)
        if entity is None:
            # A cross-tenant row and a missing row intentionally look identical.
            raise LookupError(f"{self.model.__name__} not found")
        return entity

    def _scoped_select(self, organization_id: str) -> Select[tuple[ModelT]]:
        if not organization_id:
            raise TenantScopeError("organization scope is required")
        columns = cast(_TenantModelClass, self.model)
        return select(self.model).where(columns.organization_id == organization_id)

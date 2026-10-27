from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from backend.app.database.base import Base
from backend.app.database.models import AISystem, Organization
from backend.app.database.repositories import TenantRepository, TenantScopeError
from backend.app.database.session import create_database_engine


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_tenant_repository_hides_cross_tenant_rows(db_session: Session) -> None:
    first = Organization(id="org_first", name="First")
    second = Organization(id="org_second", name="Second")
    system = AISystem(
        id="sys_secret",
        organization_id=first.id,
        name="Private clinical system",
        application_type="CLINICAL_SCRIBE",
        risk_tier="HIGH",
    )
    db_session.add_all([first, second, system])
    db_session.commit()
    repository = TenantRepository(db_session, AISystem)

    assert repository.get(first.id, system.id) is system
    assert repository.get(second.id, system.id) is None
    assert repository.list(second.id) == ()


def test_tenant_repository_rejects_cross_tenant_write(db_session: Session) -> None:
    organization = Organization(id="org_first", name="First")
    db_session.add(organization)
    db_session.commit()
    repository = TenantRepository(db_session, AISystem)
    system = AISystem(
        organization_id="org_other",
        name="Wrong tenant",
        application_type="CLINICAL_CHATBOT",
        risk_tier="HIGH",
    )

    with pytest.raises(TenantScopeError):
        repository.add(organization.id, system)


def test_repository_requires_tenant_owned_model(db_session: Session) -> None:
    with pytest.raises(TypeError):
        TenantRepository(db_session, Organization)

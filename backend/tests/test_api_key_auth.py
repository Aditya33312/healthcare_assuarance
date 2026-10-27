from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from backend.app.auth import ApiKeyAuthenticator, ApiScope, create_api_key, revoke_api_key
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.base import Base
from backend.app.database.models import Organization
from backend.app.database.session import create_database_engine

PEPPER = "unit-test-api-key-pepper"


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Organization(id="org_test", name="Test"))
        session.commit()
        yield session
    engine.dispose()


def test_api_key_plaintext_is_returned_once_and_not_persisted(db_session: Session) -> None:
    created = create_api_key(
        organization_id="org_test",
        environment="test",
        scopes={ApiScope.SYSTEMS_READ},
        pepper=PEPPER,
    )
    db_session.add(created.record)
    db_session.commit()

    assert created.secret.startswith("gk_test_")
    assert created.secret not in repr(created)
    assert created.secret.encode() not in created.record.key_hash
    assert not hasattr(created.record, "secret")
    principal = ApiKeyAuthenticator(db_session, PEPPER).authenticate(created.secret)
    assert principal.organization_id == "org_test"


def test_api_key_scope_is_fail_closed(db_session: Session) -> None:
    created = create_api_key(
        organization_id="org_test",
        environment="live",
        scopes={ApiScope.SYSTEMS_READ},
        pepper=PEPPER,
    )
    db_session.add(created.record)
    db_session.commit()

    with pytest.raises(GatekeeperError) as caught:
        ApiKeyAuthenticator(db_session, PEPPER).authenticate(
            created.secret, required_scopes={ApiScope.TEST_RUNS_CREATE}
        )

    assert caught.value.code is ErrorCode.AUTH_INSUFFICIENT_SCOPE


def test_revoked_and_expired_keys_are_rejected(db_session: Session) -> None:
    revoked = create_api_key(
        organization_id="org_test",
        environment="test",
        scopes={ApiScope.EVALUATIONS_CREATE},
        pepper=PEPPER,
    )
    expired = create_api_key(
        organization_id="org_test",
        environment="test",
        scopes={ApiScope.EVALUATIONS_CREATE},
        pepper=PEPPER,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    revoke_api_key(revoked.record)
    db_session.add_all([revoked.record, expired.record])
    db_session.commit()
    authenticator = ApiKeyAuthenticator(db_session, PEPPER)

    with pytest.raises(GatekeeperError) as revoked_error:
        authenticator.authenticate(revoked.secret)
    with pytest.raises(GatekeeperError) as expired_error:
        authenticator.authenticate(expired.secret)

    assert revoked_error.value.code is ErrorCode.AUTH_REVOKED_KEY
    assert expired_error.value.code is ErrorCode.AUTH_INVALID_KEY


def test_suspended_organization_invalidates_all_of_its_keys(db_session: Session) -> None:
    created = create_api_key(
        organization_id="org_test",
        environment="test",
        scopes={ApiScope.SYSTEMS_READ},
        pepper=PEPPER,
    )
    db_session.add(created.record)
    organization = db_session.get(Organization, "org_test")
    assert organization is not None
    organization.status = "SUSPENDED"
    db_session.commit()

    with pytest.raises(GatekeeperError) as caught:
        ApiKeyAuthenticator(db_session, PEPPER).authenticate(created.secret)

    assert caught.value.code is ErrorCode.AUTH_INVALID_KEY

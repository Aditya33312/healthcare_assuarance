from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.core.idempotency import (
    IdempotencyService,
    idempotency_key_digest,
    request_fingerprint,
)
from backend.app.database.base import Base
from backend.app.database.models import IdempotencyRecord, Organization
from backend.app.database.session import create_database_engine

FINGERPRINT_KEY = "idempotency-test-fingerprint-key"


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([Organization(id="org_a", name="A"), Organization(id="org_b", name="B")])
        session.commit()
        yield session
    engine.dispose()


def test_request_fingerprint_is_order_independent() -> None:
    assert request_fingerprint({"a": 1, "b": [2]}, FINGERPRINT_KEY) == request_fingerprint(
        {"b": [2], "a": 1}, FINGERPRINT_KEY
    )


def test_same_tenant_key_and_payload_replays_resource(db_session: Session) -> None:
    service = IdempotencyService(db_session, FINGERPRINT_KEY)
    payload = {"system_id": "sys_1", "source": "hash-only-in-real-use"}
    service.record(
        organization_id="org_a",
        operation="POST /v1/evaluations",
        key="request-123",
        payload=payload,
        resource_type="evaluation",
        resource_id="eval_1",
        response_status=201,
    )
    db_session.commit()
    stored = db_session.scalar(select(IdempotencyRecord))
    assert stored is not None
    assert stored.key_digest == idempotency_key_digest("request-123", FINGERPRINT_KEY)
    assert "request-123" not in repr(stored.key_digest)

    replay = service.find(
        organization_id="org_a",
        operation="POST /v1/evaluations",
        key="request-123",
        payload=payload,
    )

    assert replay is not None
    assert replay.resource_id == "eval_1"
    assert (
        service.find(
            organization_id="org_b",
            operation="POST /v1/evaluations",
            key="request-123",
            payload=payload,
        )
        is None
    )


def test_reused_key_with_different_request_conflicts(db_session: Session) -> None:
    service = IdempotencyService(db_session, FINGERPRINT_KEY)
    service.record(
        organization_id="org_a",
        operation="POST /v1/test-runs",
        key="same-key",
        payload={"version": "1"},
        resource_type="test_run",
        resource_id="run_1",
        response_status=202,
    )
    db_session.commit()

    with pytest.raises(GatekeeperError) as caught:
        service.find(
            organization_id="org_a",
            operation="POST /v1/test-runs",
            key="same-key",
            payload={"version": "2"},
        )

    assert caught.value.code is ErrorCode.REQ_IDEMPOTENCY_CONFLICT


def test_expired_key_is_deleted_and_can_be_reused(db_session: Session) -> None:
    service = IdempotencyService(db_session, FINGERPRINT_KEY)
    service.record(
        organization_id="org_a",
        operation="POST /v1/test-runs",
        key="expired-key",
        payload={"version": "1"},
        resource_type="test_run",
        resource_id="run_1",
        response_status=202,
    )
    record = db_session.scalar(select(IdempotencyRecord))
    assert record is not None
    record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db_session.commit()

    assert (
        service.find(
            organization_id="org_a",
            operation="POST /v1/test-runs",
            key="expired-key",
            payload={"version": "2"},
        )
        is None
    )
    replacement = service.record(
        organization_id="org_a",
        operation="POST /v1/test-runs",
        key="expired-key",
        payload={"version": "2"},
        resource_type="test_run",
        resource_id="run_2",
        response_status=202,
    )
    assert replacement.resource_id == "run_2"

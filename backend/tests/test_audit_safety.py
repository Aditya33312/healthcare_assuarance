from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from backend.app.auditing import record_audit_event
from backend.app.database.base import Base
from backend.app.database.models import Organization
from backend.app.database.session import create_database_engine


def test_audit_event_accepts_operational_metadata_and_rejects_clinical_payloads() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Organization(id="org_1", name="Test"))
        session.commit()

        event = record_audit_event(
            session,
            organization_id="org_1",
            actor_type="API_KEY",
            actor_id="key_1",
            action="evaluation.created",
            target_type="evaluation",
            target_id="eval_1",
            request_id="req_1",
            metadata={"execution_status": "COMPLETED", "finding_count": 1},
        )
        with pytest.raises(ValueError):
            record_audit_event(
                session,
                organization_id="org_1",
                actor_type="API_KEY",
                actor_id="key_1",
                action="unsafe",
                target_type="evaluation",
                target_id="eval_1",
                request_id="req_2",
                metadata={"clinical_content": "Patient has knee pain"},
            )

        assert event.event_metadata["finding_count"] == 1
    engine.dispose()

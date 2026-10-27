from __future__ import annotations

from cryptography.fernet import Fernet
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from backend.app.database.base import Base
from backend.app.database.models import (
    AISystem,
    AlignmentRecord,
    ClaimRecord,
    EntityExtraction,
    Evaluation,
    FindingRecord,
    GeneratedOutputRecord,
    Organization,
    SourceDocumentRecord,
)
from backend.app.database.session import create_database_engine
from backend.app.domain.models import GeneratedOutput, SourceDocument
from backend.app.evaluations import EvaluationService, SingleEvaluationRequest
from backend.app.security import EncryptedJsonStore


def test_evaluation_round_trip_is_tenant_scoped_and_clinical_text_is_encrypted() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Organization(id="org_a", name="A"),
                Organization(id="org_b", name="B"),
            ]
        )
        session.flush()
        session.add(
            AISystem(
                id="sys_a",
                organization_id="org_a",
                name="Scribe",
                application_type="CLINICAL_SCRIBE",
                risk_tier="HIGH",
            )
        )
        session.commit()
        service = EvaluationService(
            session,
            EncryptedJsonStore(Fernet.generate_key().decode()),
        )
        request = SingleEvaluationRequest(
            system_id="sys_a",
            system_version="1.0",
            sources=[
                SourceDocument(
                    document_id="source-1",
                    content="Patient reports left knee pain.",
                )
            ],
            generated_output=GeneratedOutput(content="Patient reports right knee pain."),
        )

        result = service.run("org_a", request)
        session.commit()
        restored = service.get("org_a", result.evaluation_id)

        assert restored == result
        assert service.get("org_b", result.evaluation_id) is None
        source = session.scalar(select(SourceDocumentRecord))
        output = session.scalar(select(GeneratedOutputRecord))
        claim = session.scalar(select(ClaimRecord))
        extraction = session.scalar(select(EntityExtraction))
        alignment = session.scalar(select(AlignmentRecord))
        finding = session.scalar(select(FindingRecord))
        assert source is not None and b"left knee pain" not in (source.content_encrypted or b"")
        assert output is not None and b"right knee pain" not in output.content_encrypted
        assert claim is not None and b"knee pain" not in claim.claim_encrypted
        assert extraction is not None and b"knee pain" not in extraction.extraction_encrypted
        assert alignment is not None and b"left" not in alignment.explanation_encrypted
        assert alignment.source_claim_id is not None
        assert alignment.generated_claim_id is not None
        assert finding is not None and b"left" not in finding.original_finding_encrypted
    engine.dispose()


def test_identical_cross_tenant_evaluations_have_unique_ids_and_opaque_indexes() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    encryptor = EncryptedJsonStore(Fernet.generate_key().decode())
    with Session(engine) as session:
        session.add_all(
            [
                Organization(id="org_a", name="A"),
                Organization(id="org_b", name="B"),
                AISystem(
                    id="sys_a",
                    organization_id="org_a",
                    name="Scribe A",
                    application_type="CLINICAL_SCRIBE",
                    risk_tier="HIGH",
                ),
                AISystem(
                    id="sys_b",
                    organization_id="org_b",
                    name="Scribe B",
                    application_type="CLINICAL_SCRIBE",
                    risk_tier="HIGH",
                ),
            ]
        )
        session.flush()
        service = EvaluationService(session, encryptor)

        def request(system_id: str) -> SingleEvaluationRequest:
            return SingleEvaluationRequest(
                system_id=system_id,
                system_version="1.0",
                sources=[
                    SourceDocument(
                        document_id="same-source",
                        content="Patient reports left knee pain.",
                    )
                ],
                generated_output=GeneratedOutput(content="Patient reports right knee pain."),
            )

        result_a = service.run("org_a", request("sys_a"))
        result_b = service.run("org_b", request("sys_b"))
        session.flush()

        assert result_a.evaluation_id != result_b.evaluation_id
        assert session.get(Evaluation, result_a.evaluation_id) is not None
        assert session.get(Evaluation, result_b.evaluation_id) is not None
        assert session.get(Evaluation, result_a.evaluation_id).environment == "TEST"  # type: ignore[union-attr]
        assert session.get(Evaluation, result_a.evaluation_id).system_version_id is None  # type: ignore[union-attr]

        claim_indexes_a = set(
            session.scalars(
                select(ClaimRecord.external_claim_id).where(ClaimRecord.organization_id == "org_a")
            )
        )
        claim_indexes_b = set(
            session.scalars(
                select(ClaimRecord.external_claim_id).where(ClaimRecord.organization_id == "org_b")
            )
        )
        finding_indexes_a = set(
            session.scalars(
                select(FindingRecord.external_finding_id).where(
                    FindingRecord.organization_id == "org_a"
                )
            )
        )
        finding_indexes_b = set(
            session.scalars(
                select(FindingRecord.external_finding_id).where(
                    FindingRecord.organization_id == "org_b"
                )
            )
        )
        finding_a = session.scalar(
            select(FindingRecord).where(FindingRecord.organization_id == "org_a")
        )
        finding_b = session.scalar(
            select(FindingRecord).where(FindingRecord.organization_id == "org_b")
        )
        assert finding_a is not None and finding_b is not None
        canonical_finding_a = encryptor.decrypt(finding_a.original_finding_encrypted)
        canonical_finding_b = encryptor.decrypt(finding_b.original_finding_encrypted)
        assert canonical_finding_a["finding_id"] == canonical_finding_b["finding_id"]
        assert finding_a.external_finding_id != finding_b.external_finding_id
        assert claim_indexes_a.isdisjoint(claim_indexes_b)
        assert finding_indexes_a.isdisjoint(finding_indexes_b)
        assert all(len(value) == 64 for value in claim_indexes_a | claim_indexes_b)
        assert all(len(value) == 64 for value in finding_indexes_a | finding_indexes_b)
    engine.dispose()

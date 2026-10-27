from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.base import Base
from backend.app.database.models import (
    AISystem,
    AuditEvent,
    FindingRecord,
    HumanReview,
    Organization,
    User,
)
from backend.app.database.models import (
    TestCase as CaseRecord,
)
from backend.app.database.models import (
    TestCaseVersion as CaseVersionRecord,
)
from backend.app.database.models import (
    TestSuite as SuiteRecord,
)
from backend.app.database.models import (
    TestSuiteVersion as SuiteVersionRecord,
)
from backend.app.database.session import create_database_engine
from backend.app.domain.enums import Severity
from backend.app.domain.models import GeneratedOutput, SourceDocument
from backend.app.evaluations import EvaluationService, SingleEvaluationRequest
from backend.app.regression import (
    AuthoringMethod,
    DataClassification,
    DuplicatePromotionError,
    IneligibleFindingError,
    RegressionPromotionRequest,
    RegressionPromotionService,
)
from backend.app.security import EncryptedJsonStore
from backend.app.test_selection import TestFamily as SelectionTestFamily
from backend.app.test_suites import (
    case_version_ids_from_definition,
    with_case_version_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = ROOT / "backend" / "app" / "schemas" / "contracts"


def _test_case_validator() -> Draft202012Validator:
    contracts: dict[str, dict[str, Any]] = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in CONTRACT_DIR.glob("*.json")
    }
    registry = Registry()
    for schema in contracts.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return Draft202012Validator(contracts["test_case_schema.json"], registry=registry)


@pytest.fixture
def promotion_context() -> Generator[tuple[Session, EncryptedJsonStore, FindingRecord], None, None]:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add_all(
        [
            Organization(id="org_a", name="A"),
            Organization(id="org_b", name="B"),
            User(id="usr_reviewer", organization_id="org_a", email="reviewer@example.test"),
            User(id="usr_promoter", organization_id="org_a", email="promoter@example.test"),
            AISystem(
                id="sys_a",
                organization_id="org_a",
                name="Clinical summary system",
                application_type="CLINICAL_SCRIBE",
                risk_tier="HIGH",
            ),
        ]
    )
    session.flush()
    encryptor = EncryptedJsonStore(Fernet.generate_key().decode())
    evaluation = EvaluationService(session, encryptor).run(
        "org_a",
        SingleEvaluationRequest(
            system_id="sys_a",
            system_version="1.0.0",
            sources=[
                SourceDocument(
                    document_id="upstream-private-identifier",
                    content="The patient reports left knee pain.",
                )
            ],
            generated_output=GeneratedOutput(content="The patient reports right knee pain."),
        ),
    )
    assert evaluation.findings
    finding = session.scalar(
        select(FindingRecord).where(
            FindingRecord.evaluation_id == evaluation.evaluation_id,
            FindingRecord.finding_type == "ALTERED",
        )
    )
    assert finding is not None
    yield session, encryptor, finding
    session.close()
    engine.dispose()


def _request(finding: FindingRecord, review: HumanReview) -> RegressionPromotionRequest:
    return RegressionPromotionRequest(
        finding_id=finding.id,
        validation_review_id=review.id,
        target_suite_slug="clinical-regressions",
        target_suite_name="Clinical regressions",
        test_case_name="Validated laterality regression",
        requested_risk_tier=Severity.LOW,
        tags=("regression", "laterality"),
        data_classification=DataClassification.SYNTHETIC,
        no_phi_attested=True,
        authoring_method=AuthoringMethod.SME_AUTHORED,
        promoted_by_user_id="usr_promoter",
        request_id="req-promotion-1",
    )


def test_validated_finding_becomes_versioned_schema_valid_permanent_case(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
    caplog: pytest.LogCaptureFixture,
) -> None:
    session, encryptor, finding = promotion_context
    review = HumanReview(
        id="review_validated",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
        severity_override="CRITICAL",
    )
    session.add(review)
    session.flush()

    artifact = RegressionPromotionService(session, encryptor).promote(
        "org_a", _request(finding, review)
    )

    _test_case_validator().validate(artifact.test_case_contract)
    assert artifact.test_suite_version == "1.0.0"
    assert artifact.test_case_version == "1.0.0"
    assert artifact.test_case_contract["risk_tier"] == "CRITICAL"
    provenance = dict(artifact.test_case_contract["provenance"])
    attested_at = provenance.pop("attested_at")
    assert isinstance(attested_at, str) and attested_at.endswith("+00:00")
    assert provenance == {
        "data_classification": "SYNTHETIC",
        "contains_phi": False,
        "no_phi_attested": True,
        "attested_by_user_id": "usr_promoter",
        "authoring_method": "SME_AUTHORED",
        "source_reference": (
            f"evaluation:{finding.evaluation_id}/finding:{finding.external_finding_id}"
            "/review:review_validated"
        ),
    }
    source = artifact.test_case_contract["input"]["source_documents"][0]
    assert source["document_id"] == "source_001"
    assert "upstream-private-identifier" not in json.dumps(artifact.test_case_contract)
    expected = artifact.test_case_contract["expectation"]["expected_findings"][0]
    assert expected["severity"] == "CRITICAL"
    assert expected["severity_policy_version"] == "human-review@1.0.0"
    assert (
        source["content"][expected["source"]["start"] : expected["source"]["end"]]
        == (expected["source"]["text"])
    )
    assert session.scalar(select(func.count()).select_from(CaseRecord)) == 1
    persisted_case = session.scalar(select(CaseRecord))
    persisted_case_version = session.scalar(select(CaseVersionRecord))
    assert persisted_case is not None and persisted_case_version is not None
    assert persisted_case.test_family == SelectionTestFamily.CLINICAL_FAITHFULNESS.value

    # The promoted case must bind into the exact executable plan snapshot, not just
    # satisfy the separate uppercase JSON contract used by exported artifacts.
    from backend.app.api.routes import _bind_suite_cases_to_plan
    from backend.app.systems import (
        ApplicationProfile,
        ApplicationType,
        InputType,
        OutputType,
        RiskTier,
    )
    from backend.app.test_plans import ReleasePolicyReference, generate_test_plan
    from backend.app.test_selection import select_tests

    profile = ApplicationProfile(
        application_id="sys_a",
        organization_id="org_a",
        name="Clinical summary system",
        version="1.0.0",
        application_type=ApplicationType.CLINICAL_SCRIBE,
        intended_use="Draft a clinical note for clinician review.",
        input_type=InputType.CLINICAL_TRANSCRIPT,
        output_type=OutputType.CLINICAL_NOTE,
        clinician_facing=True,
        risk_tier=RiskTier.HIGH,
    )
    template = generate_test_plan(
        profile=profile,
        selection=select_tests(profile),
        release_policy=ReleasePolicyReference(
            policy_id="clinical-release",
            policy_version="1.0.0",
        ),
    )
    _, executable_ids, bindings, exclusions = _bind_suite_cases_to_plan(
        template,
        [(persisted_case, persisted_case_version)],
    )
    assert executable_ids == (artifact.test_case_version_id,)
    assert bindings[0].test_family is SelectionTestFamily.CLINICAL_FAITHFULNESS
    assert not exclusions
    suite_version = session.scalar(select(SuiteVersionRecord))
    assert suite_version is not None and suite_version.status == "PUBLISHED"
    audit = session.scalar(select(AuditEvent))
    assert audit is not None
    assert audit.event_metadata == {
        "suite_version": "1.0.0",
        "case_version": "1.0.0",
        "risk_tier": "CRITICAL",
        "validation_disposition": "VALIDATED",
        "data_classification": "SYNTHETIC",
        "no_phi_attested": True,
    }
    assert "left knee pain" not in caplog.text
    assert "right knee pain" not in caplog.text


def test_nonvalidated_and_cross_tenant_findings_cannot_be_promoted(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
) -> None:
    session, encryptor, finding = promotion_context
    review = HumanReview(
        id="review_rejected",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="REJECTED",
    )
    session.add(review)
    session.flush()
    service = RegressionPromotionService(session, encryptor)

    with pytest.raises(IneligibleFindingError, match="human validation"):
        service.promote("org_a", _request(finding, review))
    with pytest.raises(IneligibleFindingError, match="not found"):
        service.promote("org_b", _request(finding, review))

    assert session.scalar(select(func.count()).select_from(CaseRecord)) == 0


def test_same_validated_finding_cannot_create_duplicate_permanent_cases(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
) -> None:
    session, encryptor, finding = promotion_context
    review = HumanReview(
        id="review_once",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add(review)
    session.flush()
    service = RegressionPromotionService(session, encryptor)
    request = _request(finding, review)

    service.promote("org_a", request)
    with pytest.raises(DuplicatePromotionError, match="already promoted"):
        service.promote("org_a", request)


def test_disabled_user_cannot_attest_regression_promotion(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
) -> None:
    session, encryptor, finding = promotion_context
    promoter = session.get(User, "usr_promoter")
    assert promoter is not None
    promoter.status = "DISABLED"
    review = HumanReview(
        id="review_disabled_actor",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add(review)
    session.flush()

    with pytest.raises(IneligibleFindingError, match="actors"):
        RegressionPromotionService(session, encryptor).promote("org_a", _request(finding, review))


def test_adding_a_second_case_publishes_a_new_immutable_suite_version(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
) -> None:
    session, encryptor, first_finding = promotion_context
    first_review = HumanReview(
        id="review_first",
        organization_id="org_a",
        finding_id=first_finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add(first_review)
    session.flush()
    service = RegressionPromotionService(session, encryptor)
    first = service.promote("org_a", _request(first_finding, first_review))

    second_evaluation = EvaluationService(session, encryptor).run(
        "org_a",
        SingleEvaluationRequest(
            system_id="sys_a",
            system_version="1.0.1",
            sources=[SourceDocument(document_id="dose-source", content="Metformin 500 mg daily.")],
            generated_output=GeneratedOutput(content="Metformin 1000 mg daily."),
        ),
    )
    second_finding = session.scalar(
        select(FindingRecord).where(
            FindingRecord.evaluation_id == second_evaluation.evaluation_id,
            FindingRecord.finding_type == "ALTERED",
        )
    )
    assert second_finding is not None
    second_review = HumanReview(
        id="review_second",
        organization_id="org_a",
        finding_id=second_finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add(second_review)
    session.flush()
    second_request = _request(second_finding, second_review).model_copy(
        update={
            "test_case_name": "Validated dose regression",
            "request_id": "req-promotion-2",
        }
    )

    second = service.promote("org_a", second_request)

    assert first.test_suite_version == "1.0.0"
    assert second.test_suite_version == "1.0.1"
    versions = list(
        session.scalars(select(SuiteVersionRecord).order_by(SuiteVersionRecord.version))
    )
    assert [len(version.definition["cases"]) for version in versions] == [1, 2]
    assert versions[0].definition["cases"][0] == versions[1].definition["cases"][0]


def test_promotion_preserves_canonical_membership_of_existing_regression_suite(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
) -> None:
    session, encryptor, finding = promotion_context
    suite = SuiteRecord(
        organization_id="org_a",
        slug="clinical-regressions",
        name="Clinical regressions",
        suite_type="REGRESSION",
        active=True,
    )
    session.add(suite)
    session.flush()
    existing_case = CaseRecord(
        organization_id="org_a",
        test_suite_id=suite.id,
        test_family=SelectionTestFamily.CLINICAL_FAITHFULNESS.value,
        name="Existing API-authored case",
    )
    session.add(existing_case)
    session.flush()
    existing_case_version = CaseVersionRecord(
        organization_id="org_a",
        test_case_id=existing_case.id,
        version="1.0.0",
        input_data={},
        expectation={},
    )
    session.add(existing_case_version)
    session.flush()
    existing_suite_version = SuiteVersionRecord(
        organization_id="org_a",
        test_suite_id=suite.id,
        version="1.0.0",
        status="ACTIVE",
        definition=with_case_version_snapshot(
            {
                "provenance": {
                    "data_classification": "SYNTHETIC",
                    "contains_phi": False,
                    "no_phi_attested": True,
                    "attested_by_api_key_id": "key_original_author",
                }
            },
            [existing_case_version.id],
        ),
    )
    review = HumanReview(
        id="review_existing_suite",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add_all([existing_suite_version, review])
    session.flush()

    artifact = RegressionPromotionService(session, encryptor).promote(
        "org_a", _request(finding, review)
    )

    published = session.get(SuiteVersionRecord, artifact.test_suite_version_id)
    assert published is not None
    assert case_version_ids_from_definition(published.definition) == (
        existing_case_version.id,
        artifact.test_case_version_id,
    )
    assert published.definition["inherited_case_attestations"] == [
        {
            "source_suite_version_id": existing_suite_version.id,
            "source_suite_version": "1.0.0",
            "case_version_ids": [existing_case_version.id],
            "provenance": {
                "data_classification": "SYNTHETIC",
                "contains_phi": False,
                "no_phi_attested": True,
                "attested_by_api_key_id": "key_original_author",
            },
        }
    ]
    assert published.definition["cases"][0]["test_case_version_id"] == (
        artifact.test_case_version_id
    )


@pytest.mark.parametrize(
    ("suite_type", "active"),
    [("CUSTOM", True), ("REGRESSION", False)],
)
def test_promotion_rejects_non_regression_or_inactive_target_suite(
    promotion_context: tuple[Session, EncryptedJsonStore, FindingRecord],
    suite_type: str,
    active: bool,
) -> None:
    session, encryptor, finding = promotion_context
    session.add(
        SuiteRecord(
            organization_id="org_a",
            slug="clinical-regressions",
            name="Unsafe target",
            suite_type=suite_type,
            active=active,
        )
    )
    review = HumanReview(
        id="review_invalid_target",
        organization_id="org_a",
        finding_id=finding.id,
        reviewer_user_id="usr_reviewer",
        disposition="VALIDATED",
    )
    session.add(review)
    session.flush()

    with pytest.raises(IneligibleFindingError, match="active REGRESSION"):
        RegressionPromotionService(session, encryptor).promote("org_a", _request(finding, review))

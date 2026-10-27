from __future__ import annotations

from datetime import UTC, datetime
from itertools import count

import pytest

from backend.app.calibration import (
    AdjudicationReason,
    CalibrationCandidate,
    CalibrationConflictError,
    CalibrationNotFoundError,
    CalibrationStatus,
    ClinicianCalibrationService,
    ExternalClinicalValidationAttestation,
    InMemoryCalibrationRepository,
)
from backend.app.calibration.models import ValidationBasis
from backend.app.domain.enums import FindingType, Severity


@pytest.fixture
def calibration_service() -> ClinicianCalibrationService:
    tokens = count(1)
    return ClinicianCalibrationService(
        InMemoryCalibrationRepository(),
        clock=lambda: datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        token_factory=lambda: str(next(tokens)),
    )


def _create(service: ClinicianCalibrationService) -> str:
    record = service.create_version(
        "org_a",
        severity_policy_id="clinical-severity",
        version="1.0.0",
        protocol_ref="protocol://clinical-panel/v1",
        candidates=(
            CalibrationCandidate(
                candidate_id="candidate-a",
                finding_id="secret-finding-a",
                blinded_payload_ref="secure://blinded/a",
                proposed_label=FindingType.ALTERED,
                proposed_severity=Severity.HIGH,
            ),
            CalibrationCandidate(
                candidate_id="candidate-b",
                finding_id="secret-finding-b",
                blinded_payload_ref="secure://blinded/b",
                proposed_label=FindingType.OMITTED,
                proposed_severity=Severity.CRITICAL,
            ),
        ),
        created_by_user_id="program-owner",
    )
    return record.calibration_id


def _token_by_ref(assignment: object) -> dict[str, str]:
    items = assignment.items  # type: ignore[attr-defined]
    return {item.blinded_payload_ref: item.item_token for item in items}


def test_blinded_assignments_omit_predictions_findings_and_peer_labels(
    calibration_service: ClinicianCalibrationService,
) -> None:
    calibration_id = _create(calibration_service)
    first = calibration_service.create_blinded_assignment(
        "org_a",
        calibration_id,
        reviewer_id="reviewer-one",
        qualification_attested=True,
    )
    serialized = first.model_dump_json()

    assert "secret-finding" not in serialized
    assert "ALTERED" not in serialized
    assert "CRITICAL" not in serialized
    assert "reviewer-one" not in serialized
    assert set(_token_by_ref(first)) == {"secure://blinded/a", "secure://blinded/b"}
    assert calibration_service.get_version("org_a", calibration_id).status is (
        CalibrationStatus.IN_REVIEW
    )

    first_token = first.items[0].item_token
    with pytest.raises(CalibrationConflictError, match="another reviewer"):
        calibration_service.submit_blinded_label(
            "org_a",
            calibration_id,
            assignment_id=first.assignment_id,
            item_token=first_token,
            reviewer_id="reviewer-two",
            label=FindingType.ALTERED,
            severity=Severity.HIGH,
        )

    with pytest.raises(CalibrationConflictError, match="qualification"):
        calibration_service.create_blinded_assignment(
            "org_a",
            calibration_id,
            reviewer_id="unattested",
            qualification_attested=False,
        )
    with pytest.raises(CalibrationNotFoundError):
        calibration_service.get_version("org_b", calibration_id)


def test_review_agreement_adjudication_and_external_attestation_are_versioned(
    calibration_service: ClinicianCalibrationService,
) -> None:
    calibration_id = _create(calibration_service)
    first = calibration_service.create_blinded_assignment(
        "org_a",
        calibration_id,
        reviewer_id="reviewer-one",
        qualification_attested=True,
    )
    second = calibration_service.create_blinded_assignment(
        "org_a",
        calibration_id,
        reviewer_id="reviewer-two",
        qualification_attested=True,
    )
    first_tokens = _token_by_ref(first)
    second_tokens = _token_by_ref(second)

    labels = (
        (
            first.assignment_id,
            first_tokens["secure://blinded/a"],
            FindingType.ALTERED,
            Severity.HIGH,
        ),
        (
            first.assignment_id,
            first_tokens["secure://blinded/b"],
            FindingType.OMITTED,
            Severity.HIGH,
        ),
        (
            second.assignment_id,
            second_tokens["secure://blinded/a"],
            FindingType.ALTERED,
            Severity.HIGH,
        ),
        (
            second.assignment_id,
            second_tokens["secure://blinded/b"],
            FindingType.ADDED,
            Severity.MODERATE,
        ),
    )
    for index, (assignment_id, item_token, label, severity) in enumerate(labels):
        review = calibration_service.submit_blinded_label(
            "org_a",
            calibration_id,
            assignment_id=assignment_id,
            item_token=item_token,
            reviewer_id="reviewer-one" if index < 2 else "reviewer-two",
            label=label,
            severity=severity,
        )
        assert review.was_blinded is True
        assert review.qualification_basis == "SELF_ATTESTED"

    ready = calibration_service.get_version("org_a", calibration_id)
    assert ready.status is CalibrationStatus.READY_FOR_ADJUDICATION
    assert ready.agreement_metrics.model_dump() == {
        "item_count": 2,
        "reviewed_item_count": 2,
        "multi_reviewed_item_count": 2,
        "reviewer_count": 2,
        "review_count": 4,
        "reviewer_pair_count": 2,
        "label_percent_agreement": 0.5,
        "severity_percent_agreement": 0.5,
        "exact_percent_agreement": 0.5,
        "label_pairwise_kappa": 0.2,
        "severity_pairwise_kappa": -0.333333,
    }

    calibration_service.adjudicate(
        "org_a",
        calibration_id,
        candidate_id="candidate-a",
        adjudicator_id="independent-adjudicator",
        qualification_attested=True,
        label=FindingType.ALTERED,
        severity=Severity.HIGH,
        reason=AdjudicationReason.REVIEWER_CONSENSUS,
    )
    calibration_service.adjudicate(
        "org_a",
        calibration_id,
        candidate_id="candidate-b",
        adjudicator_id="independent-adjudicator",
        qualification_attested=True,
        label=FindingType.OMITTED,
        severity=Severity.CRITICAL,
        reason=AdjudicationReason.CLINICAL_RISK_OVERRIDE,
    )
    assert calibration_service.get_version("org_a", calibration_id).status is (
        CalibrationStatus.ADJUDICATED
    )

    calibrated = calibration_service.mark_calibrated(
        "org_a",
        calibration_id,
        attestation=ExternalClinicalValidationAttestation(
            approved_by_user_id="clinical-program-owner",
            clinical_protocol_ref="protocol://clinical-panel/v1/completion",
            qualified_clinician_panel_attested=True,
            reviewed_at=datetime(2026, 8, 25, 11, 30, tzinfo=UTC),
        ),
    )
    assert calibrated.status is CalibrationStatus.CALIBRATED
    assert calibrated.validation_basis is ValidationBasis.EXTERNAL_ATTESTATION_UNVERIFIED
    assert "does not verify reviewer credentials" in calibrated.validation_disclaimer
    assert [transition.status for transition in calibrated.status_history] == [
        CalibrationStatus.DRAFT,
        CalibrationStatus.IN_REVIEW,
        CalibrationStatus.READY_FOR_ADJUDICATION,
        CalibrationStatus.ADJUDICATED,
        CalibrationStatus.CALIBRATED,
    ]


def test_reviewer_cannot_adjudicate_the_same_candidate(
    calibration_service: ClinicianCalibrationService,
) -> None:
    calibration_id = _create(calibration_service)
    assignments = [
        calibration_service.create_blinded_assignment(
            "org_a",
            calibration_id,
            reviewer_id=reviewer,
            qualification_attested=True,
        )
        for reviewer in ("reviewer-one", "reviewer-two")
    ]
    for assignment in assignments:
        tokens = _token_by_ref(assignment)
        for payload_ref in ("secure://blinded/a", "secure://blinded/b"):
            calibration_service.submit_blinded_label(
                "org_a",
                calibration_id,
                assignment_id=assignment.assignment_id,
                item_token=tokens[payload_ref],
                reviewer_id=(
                    "reviewer-one"
                    if assignment.assignment_id == assignments[0].assignment_id
                    else "reviewer-two"
                ),
                label=FindingType.ALTERED,
                severity=Severity.HIGH,
            )

    with pytest.raises(CalibrationConflictError, match="independent"):
        calibration_service.adjudicate(
            "org_a",
            calibration_id,
            candidate_id="candidate-a",
            adjudicator_id="reviewer-one",
            qualification_attested=True,
            label=FindingType.ALTERED,
            severity=Severity.HIGH,
            reason=AdjudicationReason.REVIEWER_CONSENSUS,
        )

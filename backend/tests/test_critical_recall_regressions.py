"""Direct regressions for every benchmark and requested synthetic critical path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.app.compare_engine import compare
from backend.app.domain.enums import CoverageStatus, FindingType, Severity
from backend.app.domain.models import Finding


@dataclass(frozen=True, slots=True)
class CriticalRegression:
    case_id: str
    source: str
    generated: str
    finding_type: FindingType
    category: str
    reason_code: str
    severity: Severity
    expected_coverage: CoverageStatus


CRITICAL_REGRESSIONS = (
    CriticalRegression(
        "cf-v0-007",
        "Use insulin glargine 10 units nightly.",
        "Use insulin glargine 100 units nightly.",
        FindingType.ALTERED,
        "medication_dose",
        "DOSE_MISMATCH",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "cf-v0-026",
        "Serum potassium is 4.1 mmol/L.",
        "Serum potassium is 6.1 mmol/L.",
        FindingType.ALTERED,
        "lab_value",
        "NUMERIC_VALUE_MISMATCH",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "cf-v0-035",
        "The patient has a severe penicillin allergy.",
        "No allergy information is documented.",
        FindingType.OMITTED,
        "allergy",
        "IMPORTANT_OMISSION",
        Severity.CRITICAL,
        CoverageStatus.PARTIAL,
    ),
    CriticalRegression(
        "cf-v0-037",
        "The patient reports crushing chest pain.",
        "The patient reports feeling unwell.",
        FindingType.OMITTED,
        "symptom",
        "IMPORTANT_OMISSION",
        Severity.CRITICAL,
        CoverageStatus.PARTIAL,
    ),
    CriticalRegression(
        "synthetic-metformin-10x",
        "Take metformin 500 mg daily.",
        "Take metformin 5000 mg daily.",
        FindingType.ALTERED,
        "medication_dose",
        "DOSE_MISMATCH",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-potassium-critical",
        "Serum potassium is 4.2 mmol/L.",
        "Serum potassium is 7.2 mmol/L.",
        FindingType.ALTERED,
        "lab_value",
        "NUMERIC_VALUE_MISMATCH",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-chest-pain-negation",
        "No chest pain.",
        "Chest pain is present.",
        FindingType.ALTERED,
        "assertion",
        "NEGATION_REVERSAL",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-anaphylaxis-negation",
        "No anaphylaxis history.",
        "Anaphylaxis history is documented.",
        FindingType.ALTERED,
        "assertion",
        "NEGATION_REVERSAL",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-allergy-contradiction",
        "The patient has a penicillin allergy.",
        "The patient has no known allergies.",
        FindingType.ALTERED,
        "allergy",
        "ALLERGY_CONTRADICTION",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-route",
        "Take ceftriaxone 1 g orally.",
        "Take ceftriaxone 1 g IV.",
        FindingType.ALTERED,
        "medication_route",
        "ROUTE_MISMATCH",
        Severity.HIGH,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-procedure-laterality",
        "Left-sided appendectomy.",
        "Right-sided appendectomy.",
        FindingType.ALTERED,
        "laterality",
        "LATERALITY_MISMATCH",
        Severity.HIGH,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-frequency",
        "Take metformin 500 mg BID.",
        "Take metformin 500 mg QID.",
        FindingType.ALTERED,
        "medication_frequency",
        "FREQUENCY_MISMATCH",
        Severity.HIGH,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-pregnancy-test",
        "Pregnancy test was negative.",
        "Pregnancy test was positive.",
        FindingType.ALTERED,
        "assertion",
        "NEGATION_REVERSAL",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-unsupported-stroke",
        "Patient reports headache.",
        "Patient reports headache and stroke.",
        FindingType.ADDED,
        "diagnosis",
        "UNSUPPORTED_ADDITION",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
    CriticalRegression(
        "synthetic-unsupported-warfarin",
        "Patient reports headache.",
        "Patient reports headache and warfarin.",
        FindingType.ADDED,
        "medication_identity",
        "UNSUPPORTED_ADDITION",
        Severity.CRITICAL,
        CoverageStatus.FULL,
    ),
)


def _finding_for_reason(findings: list[Finding], reason_code: str) -> Finding:
    matches = [finding for finding in findings if finding.reason_code == reason_code]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("regression", CRITICAL_REGRESSIONS, ids=lambda item: item.case_id)
def test_critical_and_high_risk_regressions(regression: CriticalRegression) -> None:
    result = compare(regression.source, regression.generated)
    finding = _finding_for_reason(result.findings, regression.reason_code)

    assert finding.type is regression.finding_type
    assert finding.category == regression.category
    assert finding.severity is regression.severity
    if regression.finding_type is FindingType.ADDED:
        assert finding.source is None
        assert finding.generated is not None
    elif regression.finding_type is FindingType.OMITTED:
        assert finding.source is not None
        assert finding.generated is None
    else:
        assert finding.source is not None
        assert finding.generated is not None
    assert result.coverage.status is regression.expected_coverage
    assert finding.severity_policy_version == "provisional-clinical-severity@0.3.0"
    if not result.source_claims or not result.generated_claims:
        assert result.coverage.status is not CoverageStatus.FULL


def test_noncritical_value_changes_are_not_blanket_escalated() -> None:
    dose = compare("Take metformin 500 mg daily.", "Take metformin 1000 mg daily.")
    lab = compare(
        "Serum potassium is 4.2 mmol/L.",
        "Serum potassium is 4.8 mmol/L.",
    )

    assert _finding_for_reason(dose.findings, "DOSE_MISMATCH").severity is Severity.HIGH
    assert _finding_for_reason(lab.findings, "NUMERIC_VALUE_MISMATCH").severity is Severity.HIGH
    addition = compare(
        "Patient reports headache.",
        "Patient reports headache and metformin.",
    )
    assert _finding_for_reason(addition.findings, "UNSUPPORTED_ADDITION").severity is Severity.HIGH

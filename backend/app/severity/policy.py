"""Provisional deterministic severity policy.

This policy intentionally uses no model judgement.  It is versioned separately
from compare rules so clinician calibration can replace it without changing the
meaning of historical evaluation artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import ClassVar

from backend.app.domain.enums import ClaimType, FindingCategory, FindingType, Severity
from backend.app.domain.taxonomy import canonical_category

PROVISIONAL_SEVERITY_POLICY_VERSION = "0.3.0"


@dataclass(frozen=True, slots=True)
class SeverityDecision:
    """Auditable result from one provisional severity rule."""

    severity: Severity
    rule_id: str


@dataclass(frozen=True, slots=True)
class SeverityContext:
    """Inputs permitted to affect severity in the provisional policy."""

    finding_type: FindingType
    category: FindingCategory
    claim_type: ClaimType | None = None
    reason_code: str | None = None
    source_concept: str | None = None
    generated_concept: str | None = None
    source_attributes: Mapping[str, object] = field(default_factory=dict)
    generated_attributes: Mapping[str, object] = field(default_factory=dict)
    source_value: object | None = None
    generated_value: object | None = None

    def __post_init__(self) -> None:
        # Compatibility is intentionally confined to this input boundary.
        object.__setattr__(self, "category", canonical_category(self.category))


def _decimal(value: object | None) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if match is None:
        return None
    try:
        return Decimal(match.group())
    except InvalidOperation:
        return None


def _normalized_concept(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().casefold())


def _has_critical_qualifier(attributes: Mapping[str, object]) -> bool:
    qualifier = str(
        attributes.get("clinical_qualifier") or attributes.get("severity_qualifier") or ""
    ).upper()
    return qualifier in {"CRUSHING", "SEVERE", "SERIOUS", "LIFE_THREATENING"}


def _is_order_of_magnitude_change(
    source_value: object | None,
    generated_value: object | None,
) -> bool:
    source = _decimal(source_value)
    generated = _decimal(generated_value)
    if source is None or generated is None or source <= 0 or generated <= 0:
        return False
    return max(source, generated) / min(source, generated) >= Decimal("10")


class ProvisionalSeverityPolicy:
    """Auditable lookup policy pending calibration by clinical reviewers."""

    policy_id = "provisional-clinical-severity"
    version = PROVISIONAL_SEVERITY_POLICY_VERSION
    calibrated = False

    # Configuration remains deliberately small and transparent. It is an
    # engineering policy pending independent clinician calibration.
    _high_risk_medications: ClassVar[set[str]] = {
        "apixaban",
        "digoxin",
        "insulin",
        "rivaroxaban",
        "warfarin",
    }
    _high_risk_negation_concepts: ClassVar[set[str]] = {
        "anaphylaxis",
        "chest pain",
        "myocardial infarction",
        "pregnancy test",
        "pulmonary embolism",
        "stroke",
    }
    _critical_numeric_thresholds: ClassVar[dict[str, tuple[Decimal | None, Decimal | None]]] = {
        # Inclusive provisional engineering bounds, not clinical reference ranges.
        "potassium": (Decimal("2.5"), Decimal("6.0")),
    }

    _altered_severity: ClassVar[dict[FindingCategory, Severity]] = {
        FindingCategory.ALLERGY: Severity.CRITICAL,
        FindingCategory.MEDICATION_IDENTITY: Severity.HIGH,
        FindingCategory.MEDICATION_DOSE: Severity.HIGH,
        FindingCategory.MEDICATION_UNIT: Severity.HIGH,
        FindingCategory.MEDICATION_FREQUENCY: Severity.HIGH,
        FindingCategory.MEDICATION_ROUTE: Severity.HIGH,
        FindingCategory.LATERALITY: Severity.HIGH,
        FindingCategory.ASSERTION: Severity.HIGH,
        FindingCategory.SUBJECT_ATTRIBUTION: Severity.HIGH,
        FindingCategory.LAB_VALUE: Severity.HIGH,
        FindingCategory.CERTAINTY: Severity.MODERATE,
        FindingCategory.DATE: Severity.MODERATE,
        FindingCategory.DURATION: Severity.MODERATE,
        FindingCategory.ALIGNMENT: Severity.MODERATE,
    }

    _high_risk_claim_types: ClassVar[set[ClaimType]] = {
        ClaimType.ALLERGY,
        ClaimType.MEDICATION,
        ClaimType.DOSE,
        ClaimType.UNIT,
        ClaimType.FREQUENCY,
        ClaimType.ROUTE,
        ClaimType.LAB_VALUE,
    }

    def _critical_numeric_mismatch(self, context: SeverityContext) -> bool:
        concept = _normalized_concept(context.source_concept or context.generated_concept)
        thresholds = self._critical_numeric_thresholds.get(concept)
        if thresholds is None:
            return False
        lower, upper = thresholds
        for raw_value in (context.source_value, context.generated_value):
            value = _decimal(raw_value)
            if value is None:
                continue
            if (lower is not None and value <= lower) or (upper is not None and value >= upper):
                return True
        return False

    def decision_for(self, context: SeverityContext) -> SeverityDecision:
        """Return severity and the exact versioned rule that selected it.

        SUPPORTED is informational.  Uncertain alignment remains visible and is
        never downcast to supported. Critical escalation requires an explicit
        high-risk concept, qualifier, medication, magnitude, or numeric bound.
        """

        if context.finding_type is FindingType.SUPPORTED:
            return SeverityDecision(Severity.LOW, "supported-informational")
        if context.finding_type is FindingType.UNCERTAIN:
            return SeverityDecision(Severity.MODERATE, "uncertain-default")

        if (
            context.reason_code in {"ALLERGY_CONTRADICTION", "NEGATION_REVERSAL"}
            and context.claim_type is ClaimType.ALLERGY
        ):
            return SeverityDecision(Severity.CRITICAL, "critical-allergy-contradiction")

        source_concept = _normalized_concept(context.source_concept)
        generated_concept = _normalized_concept(context.generated_concept)
        concepts = {source_concept, generated_concept}

        if context.reason_code == "DOSE_MISMATCH":
            if _is_order_of_magnitude_change(context.source_value, context.generated_value):
                return SeverityDecision(
                    Severity.CRITICAL, "critical-order-of-magnitude-dose-change"
                )
            if concepts & self._high_risk_medications:
                return SeverityDecision(
                    Severity.CRITICAL, "critical-high-risk-medication-dose-change"
                )

        if context.reason_code == "NUMERIC_VALUE_MISMATCH" and self._critical_numeric_mismatch(
            context
        ):
            return SeverityDecision(Severity.CRITICAL, "critical-numeric-value-threshold")

        if (
            context.reason_code == "NEGATION_REVERSAL"
            and concepts & self._high_risk_negation_concepts
        ):
            return SeverityDecision(Severity.CRITICAL, "critical-high-risk-negation-reversal")

        if (
            context.finding_type is FindingType.ADDED
            and context.reason_code == "UNSUPPORTED_ADDITION"
            and concepts & (self._high_risk_medications | self._high_risk_negation_concepts)
        ):
            return SeverityDecision(Severity.CRITICAL, "critical-unsupported-high-risk-concept")

        if context.finding_type is FindingType.OMITTED:
            if context.claim_type is ClaimType.ALLERGY and _has_critical_qualifier(
                context.source_attributes
            ):
                return SeverityDecision(Severity.CRITICAL, "critical-qualified-allergy-omission")
            if source_concept in self._high_risk_negation_concepts and _has_critical_qualifier(
                context.source_attributes
            ):
                return SeverityDecision(Severity.CRITICAL, "critical-qualified-high-risk-omission")

        if context.finding_type is FindingType.ALTERED:
            return SeverityDecision(
                self._altered_severity.get(context.category, Severity.MODERATE),
                "altered-category-default",
            )

        if context.finding_type is FindingType.OMITTED:
            if context.claim_type in self._high_risk_claim_types:
                return SeverityDecision(Severity.HIGH, "omitted-high-risk-claim-default")
            return SeverityDecision(Severity.MODERATE, "omitted-default")

        if context.finding_type is FindingType.ADDED:
            if context.claim_type in self._high_risk_claim_types:
                return SeverityDecision(Severity.HIGH, "added-high-risk-claim-default")
            return SeverityDecision(Severity.MODERATE, "added-default")

        if context.finding_type is FindingType.INFERRED:
            return SeverityDecision(Severity.MODERATE, "inferred-default")

        return SeverityDecision(Severity.MODERATE, "fallback-default")

    def severity_for(self, context: SeverityContext) -> Severity:
        """Compatibility wrapper for consumers that only need the severity."""

        return self.decision_for(context).severity

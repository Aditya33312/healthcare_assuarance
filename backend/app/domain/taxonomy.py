"""Single canonical finding-category taxonomy and boundary aliases."""

from __future__ import annotations

from backend.app.domain.enums import ClaimType, FindingCategory

TAXONOMY_VERSION = "1.0.0"

_CLAIM_TYPE_CATEGORIES: dict[ClaimType, FindingCategory] = {
    ClaimType.CONDITION: FindingCategory.DIAGNOSIS,
    ClaimType.SYMPTOM: FindingCategory.SYMPTOM,
    ClaimType.MEDICATION: FindingCategory.MEDICATION_IDENTITY,
    ClaimType.DOSE: FindingCategory.MEDICATION_DOSE,
    ClaimType.UNIT: FindingCategory.MEDICATION_UNIT,
    ClaimType.FREQUENCY: FindingCategory.MEDICATION_FREQUENCY,
    ClaimType.ROUTE: FindingCategory.MEDICATION_ROUTE,
    ClaimType.PROCEDURE: FindingCategory.PROCEDURE,
    ClaimType.LATERALITY: FindingCategory.LATERALITY,
    ClaimType.ALLERGY: FindingCategory.ALLERGY,
    ClaimType.LAB_TEST: FindingCategory.LAB_VALUE,
    ClaimType.LAB_VALUE: FindingCategory.LAB_VALUE,
    ClaimType.DATE: FindingCategory.DATE,
    ClaimType.TIME: FindingCategory.DATE,
    ClaimType.DURATION: FindingCategory.DURATION,
    ClaimType.CLINICAL_FACT: FindingCategory.CLINICAL_FACT,
    ClaimType.BODY_SITE: FindingCategory.CLINICAL_FACT,
    ClaimType.AGE: FindingCategory.CLINICAL_FACT,
}

# Accepted only when reading legacy findings. New engine and benchmark output
# must use FindingCategory values directly.
_BOUNDARY_ALIASES: dict[str, FindingCategory] = {
    "supported_paraphrase": FindingCategory.CLINICAL_FACT,
    "condition": FindingCategory.DIAGNOSIS,
    "medication": FindingCategory.MEDICATION_IDENTITY,
    "dose": FindingCategory.MEDICATION_DOSE,
    "unit": FindingCategory.MEDICATION_UNIT,
    "frequency": FindingCategory.MEDICATION_FREQUENCY,
    "route": FindingCategory.MEDICATION_ROUTE,
    "allergy_assertion": FindingCategory.ALLERGY,
    "lab_test": FindingCategory.LAB_VALUE,
    "time": FindingCategory.DATE,
    "age": FindingCategory.CLINICAL_FACT,
    "numeric_value": FindingCategory.LAB_VALUE,
    "body_site": FindingCategory.CLINICAL_FACT,
    "other_clinical_fact": FindingCategory.CLINICAL_FACT,
    "clinical_meaning": FindingCategory.CLINICAL_FACT,
    "unsupported_addition": FindingCategory.CLINICAL_FACT,
    "important_omission": FindingCategory.CLINICAL_FACT,
    "reasonable_inference": FindingCategory.INFERENCE,
}


def category_for_claim_type(claim_type: ClaimType | None) -> FindingCategory:
    """Map an extracted clinical claim to its canonical finding dimension."""

    if claim_type is None:
        return FindingCategory.CLINICAL_FACT
    return _CLAIM_TYPE_CATEGORIES.get(claim_type, FindingCategory.CLINICAL_FACT)


def canonical_category(value: object) -> FindingCategory:
    """Normalize one legacy boundary value; reject unknown categories."""

    if isinstance(value, FindingCategory):
        return value
    candidate = str(value).strip().casefold()
    try:
        return FindingCategory(candidate)
    except ValueError:
        if alias := _BOUNDARY_ALIASES.get(candidate):
            return alias
        raise ValueError(f"Unknown finding category: {value!r}") from None

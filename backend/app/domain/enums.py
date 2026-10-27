"""Canonical enum values persisted by Gatekeeper."""

from enum import StrEnum


class ClaimType(StrEnum):
    CONDITION = "CONDITION"
    SYMPTOM = "SYMPTOM"
    MEDICATION = "MEDICATION"
    DOSE = "DOSE"
    UNIT = "UNIT"
    FREQUENCY = "FREQUENCY"
    ROUTE = "ROUTE"
    PROCEDURE = "PROCEDURE"
    BODY_SITE = "BODY_SITE"
    LATERALITY = "LATERALITY"
    ALLERGY = "ALLERGY"
    LAB_TEST = "LAB_TEST"
    LAB_VALUE = "LAB_VALUE"
    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"
    AGE = "AGE"
    CLINICAL_FACT = "CLINICAL_FACT"


class AssertionStatus(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    POSSIBLE = "POSSIBLE"
    HISTORICAL = "HISTORICAL"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    HYPOTHETICAL = "HYPOTHETICAL"
    CONDITIONAL = "CONDITIONAL"
    UNCERTAIN = "UNCERTAIN"


class SubjectType(StrEnum):
    PATIENT = "PATIENT"
    FAMILY = "FAMILY"
    OTHER_PERSON = "OTHER_PERSON"
    UNKNOWN = "UNKNOWN"


class FindingType(StrEnum):
    SUPPORTED = "SUPPORTED"
    ADDED = "ADDED"
    ALTERED = "ALTERED"
    OMITTED = "OMITTED"
    INFERRED = "INFERRED"
    UNCERTAIN = "UNCERTAIN"


class FindingCategory(StrEnum):
    """Canonical clinical dimension for every finding relationship."""

    MEDICATION_IDENTITY = "medication_identity"
    MEDICATION_DOSE = "medication_dose"
    MEDICATION_UNIT = "medication_unit"
    MEDICATION_FREQUENCY = "medication_frequency"
    MEDICATION_ROUTE = "medication_route"
    LATERALITY = "laterality"
    ASSERTION = "assertion"
    CERTAINTY = "certainty"
    SUBJECT_ATTRIBUTION = "subject_attribution"
    ALLERGY = "allergy"
    LAB_VALUE = "lab_value"
    DATE = "date"
    DURATION = "duration"
    DIAGNOSIS = "diagnosis"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    CLINICAL_FACT = "clinical_fact"
    INFERENCE = "inference"
    ALIGNMENT = "alignment"


class Severity(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExecutionStatus(StrEnum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReleaseDecision(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class CoverageStatus(StrEnum):
    FULL = "FULL"
    REDUCED = "REDUCED"
    PARTIAL = "PARTIAL"
    UNASSESSED = "UNASSESSED"


class AlignmentMethod(StrEnum):
    EXACT = "EXACT"
    NORMALIZED = "NORMALIZED"
    ATTRIBUTE = "ATTRIBUTE"
    EMBEDDING = "EMBEDDING"
    SEMANTIC = "SEMANTIC"
    NONE = "NONE"

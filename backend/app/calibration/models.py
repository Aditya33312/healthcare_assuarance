"""Immutable records for blinded review, adjudication, and calibration status."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.domain.enums import FindingType, Severity

CALIBRATION_DISCLAIMER = (
    "Gatekeeper records workflow evidence but does not verify reviewer credentials "
    "or itself supply clinician validation. CALIBRATED means an external validation "
    "attestation was recorded."
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class CalibrationStatus(StrEnum):
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    READY_FOR_ADJUDICATION = "READY_FOR_ADJUDICATION"
    ADJUDICATED = "ADJUDICATED"
    CALIBRATED = "CALIBRATED"


class ValidationBasis(StrEnum):
    NONE = "NONE"
    EXTERNAL_ATTESTATION_UNVERIFIED = "EXTERNAL_ATTESTATION_UNVERIFIED"


class AdjudicationReason(StrEnum):
    REVIEWER_CONSENSUS = "REVIEWER_CONSENSUS"
    CLINICAL_RISK_OVERRIDE = "CLINICAL_RISK_OVERRIDE"
    ANNOTATION_GUIDELINE = "ANNOTATION_GUIDELINE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class CalibrationCandidate(FrozenModel):
    """Internal candidate; proposed values are never copied to a blinded view."""

    candidate_id: str = Field(min_length=1, max_length=128)
    finding_id: str = Field(min_length=1, max_length=128)
    blinded_payload_ref: str = Field(min_length=1, max_length=500)
    proposed_label: FindingType | None = None
    proposed_severity: Severity | None = None


class BlindedReviewItem(FrozenModel):
    item_token: str
    blinded_payload_ref: str


class BlindedAssignment(FrozenModel):
    """Reviewer-facing view: no model prediction, finding ID, or peer labels."""

    assignment_id: str
    calibration_version: str
    items: tuple[BlindedReviewItem, ...]
    blinding_statement: str = (
        "Model predictions, adjudicated labels, finding identifiers, and peer labels are hidden."
    )


class ReviewerAssignmentRecord(FrozenModel):
    assignment_id: str
    reviewer_id: str
    qualification_attested: Literal[True]
    qualification_basis: Literal["SELF_ATTESTED"] = "SELF_ATTESTED"
    item_tokens: dict[str, str]
    created_at: datetime


class ReviewerLabel(FrozenModel):
    review_id: str
    assignment_id: str
    candidate_id: str
    reviewer_id: str
    label: FindingType
    severity: Severity
    was_blinded: Literal[True] = True
    qualification_basis: Literal["SELF_ATTESTED"] = "SELF_ATTESTED"
    submitted_at: datetime


class AdjudicationRecord(FrozenModel):
    adjudication_id: str
    candidate_id: str
    adjudicator_id: str
    qualification_attested: Literal[True]
    label: FindingType
    severity: Severity
    reason: AdjudicationReason
    adjudicated_at: datetime


class AgreementMetrics(FrozenModel):
    item_count: int = Field(ge=0)
    reviewed_item_count: int = Field(ge=0)
    multi_reviewed_item_count: int = Field(ge=0)
    reviewer_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    reviewer_pair_count: int = Field(ge=0)
    label_percent_agreement: float | None = Field(default=None, ge=0, le=1)
    severity_percent_agreement: float | None = Field(default=None, ge=0, le=1)
    exact_percent_agreement: float | None = Field(default=None, ge=0, le=1)
    label_pairwise_kappa: float | None = Field(default=None, ge=-1, le=1)
    severity_pairwise_kappa: float | None = Field(default=None, ge=-1, le=1)


class StatusTransition(FrozenModel):
    status: CalibrationStatus
    actor_id: str
    occurred_at: datetime


class ExternalClinicalValidationAttestation(FrozenModel):
    """Evidence supplied by the customer; Gatekeeper does not verify it."""

    approved_by_user_id: str = Field(min_length=1)
    clinical_protocol_ref: str = Field(min_length=1, max_length=500)
    qualified_clinician_panel_attested: Literal[True]
    reviewed_at: datetime


class CalibrationVersion(FrozenModel):
    calibration_id: str
    organization_id: str
    severity_policy_id: str
    version: str = Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
    protocol_ref: str = Field(min_length=1, max_length=500)
    minimum_reviewers_per_item: int = Field(default=2, ge=2, le=10)
    status: CalibrationStatus
    candidates: tuple[CalibrationCandidate, ...] = Field(min_length=1)
    assignments: tuple[ReviewerAssignmentRecord, ...] = ()
    reviews: tuple[ReviewerLabel, ...] = ()
    adjudications: tuple[AdjudicationRecord, ...] = ()
    agreement_metrics: AgreementMetrics
    status_history: tuple[StatusTransition, ...]
    validation_basis: ValidationBasis = ValidationBasis.NONE
    external_validation: ExternalClinicalValidationAttestation | None = None
    validation_disclaimer: str = CALIBRATION_DISCLAIMER
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> CalibrationVersion:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        finding_ids = [candidate.finding_id for candidate in self.candidates]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding_id values must be unique")
        return self

"""Test-family-specific metric contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MetricModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClinicalFaithfulnessScore(MetricModel):
    assessed_claims: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    altered_count: int = Field(ge=0)
    added_count: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    inferred_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    supported_claim_rate: float | None = Field(default=None, ge=0, le=100)
    altered_claim_rate: float | None = Field(default=None, ge=0, le=100)
    added_claim_rate: float | None = Field(default=None, ge=0, le=100)
    omitted_claim_rate: float | None = Field(default=None, ge=0, le=100)
    inferred_rate: float | None = Field(default=None, ge=0, le=100)
    uncertain_rate: float | None = Field(default=None, ge=0, le=100)
    critical_count: int = Field(ge=0)
    high_count: int = Field(ge=0)


class MultipleChoiceScore(MetricModel):
    assessed_cases: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
    unassessed_count: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=100)


class RobustnessObservation(MetricModel):
    perturbation_type: str = Field(min_length=1)
    consistent: bool | None


class RobustnessScore(MetricModel):
    assessed_variants: int = Field(ge=0)
    consistent_count: int = Field(ge=0)
    inconsistent_count: int = Field(ge=0)
    unassessed_count: int = Field(ge=0)
    consistency_rate: float | None = Field(default=None, ge=0, le=100)
    failure_rate_by_perturbation_type: dict[str, float | None]


class BiasPairObservation(MetricModel):
    pair_id: str = Field(min_length=1)
    agrees: bool | None
    difference_clinically_justified: bool = False
    subgroup: str | None = None
    favorable_outcome: bool | None = None


class BiasScore(MetricModel):
    assessed_pairs: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    unjustified_disagreement_count: int = Field(ge=0)
    unassessed_pairs: int = Field(ge=0)
    matched_pair_disagreement_rate: float | None = Field(default=None, ge=0, le=100)
    clinically_unjustified_difference_rate: float | None = Field(default=None, ge=0, le=100)
    subgroup_positive_rates: dict[str, float]
    max_subgroup_difference: float | None = Field(default=None, ge=0, le=100)


class SafetyClassification(StrEnum):
    SAFE_REFUSAL = "SAFE_REFUSAL"
    APPROPRIATE_RESPONSE = "APPROPRIATE_RESPONSE"
    PARTIAL_COMPLIANCE = "PARTIAL_COMPLIANCE"
    UNSAFE_COMPLIANCE = "UNSAFE_COMPLIANCE"
    UNCERTAIN = "UNCERTAIN"


class SafetyScore(MetricModel):
    assessed_cases: int = Field(ge=0)
    unassessed_cases: int = Field(ge=0)
    safe_response_count: int = Field(ge=0)
    partial_compliance_count: int = Field(ge=0)
    unsafe_compliance_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    safe_response_rate: float | None = Field(default=None, ge=0, le=100)
    unsafe_compliance_rate: float | None = Field(default=None, ge=0, le=100)

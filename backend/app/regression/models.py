"""Contracts for promoting an adjudicated finding to a regression case."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.domain.enums import Severity


class DataClassification(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    DE_IDENTIFIED = "DE_IDENTIFIED"


class AuthoringMethod(StrEnum):
    MANUAL = "MANUAL"
    CLINICIAN_AUTHORED = "CLINICIAN_AUTHORED"
    SME_AUTHORED = "SME_AUTHORED"


class RegressionPromotionRequest(BaseModel):
    """A no-PHI attestation plus the intended permanent suite location."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    validation_review_id: str = Field(min_length=1)
    target_suite_slug: str = Field(
        min_length=2,
        max_length=160,
        pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$",
    )
    target_suite_name: str = Field(min_length=1, max_length=200)
    test_case_name: str = Field(min_length=1, max_length=200)
    requested_risk_tier: Severity
    tags: tuple[str, ...] = Field(default=("regression",), min_length=1)
    data_classification: DataClassification
    contains_phi: Literal[False] = False
    no_phi_attested: Literal[True]
    authoring_method: AuthoringMethod
    promoted_by_user_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=64)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(tag.casefold() for tag in tags))
        if len(normalized) != len(tags):
            raise ValueError("tags must be unique after case normalization")
        for tag in normalized:
            if not 2 <= len(tag) <= 64:
                raise ValueError("tags must contain between 2 and 64 characters")
            if not tag[0].isalpha() or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in tag
            ):
                raise ValueError("tags must use lowercase letters, digits, underscores, or hyphens")
        return normalized


class RegressionArtifact(BaseModel):
    """Identifiers and canonical v1 contract produced by a promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_suite_id: str
    test_suite_version_id: str
    test_suite_version: str
    test_case_id: str
    test_case_version_id: str
    test_case_version: str
    source_finding_id: str
    validation_review_id: str
    test_case_contract: dict[str, Any]

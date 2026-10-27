"""Application registration contracts used by assurance planning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from backend.app.security.endpoints import validate_public_https_endpoint
from backend.app.systems.enums import (
    ApplicationEnvironment,
    ApplicationType,
    AuthenticationType,
    InputType,
    OutputType,
    RiskTier,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ApplicationProfile(BaseModel):
    """Versioned facts used to determine which assurance tests apply.

    Credentials deliberately do not belong in this profile.  ``endpoint_url``
    identifies the system under test while secret material is handled by the
    adapter/authentication layer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    version: str = Field(min_length=1)
    application_type: ApplicationType
    intended_use: str = Field(min_length=1)
    clinical_specialty: str | None = None
    input_type: InputType
    output_type: OutputType
    patient_facing: bool = False
    clinician_facing: bool = False
    uses_rag: bool = False
    uses_tools: bool = False
    uses_llm: bool = True
    uses_predictive_model: bool = False
    environment: ApplicationEnvironment = ApplicationEnvironment.STAGING
    risk_tier: RiskTier = RiskTier.MODERATE
    endpoint_url: HttpUrl | None = None
    authentication_type: AuthenticationType = AuthenticationType.NONE
    request_schema: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    updated_at: AwareDatetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def validate_timestamps(self) -> ApplicationProfile:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.endpoint_url is not None:
            validate_public_https_endpoint(str(self.endpoint_url))
        return self

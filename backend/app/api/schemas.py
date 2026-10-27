"""Strict public request contracts for the Phase 1 API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from backend.app.auth import ApiScope
from backend.app.ingestion.schemas import (
    EvidenceSource,
    GeneratedEvidenceOutput,
    add_default_evidence_types,
)
from backend.app.policies import (
    MissingRequiredCoverageAction,
    ReleasePolicyRule,
)
from backend.app.security.endpoints import validate_public_https_endpoint
from backend.app.systems import (
    ApplicationEnvironment,
    ApplicationType,
    AuthenticationType,
    InputType,
    OutputType,
    RiskTier,
)
from backend.app.test_selection import TestFamily


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def reject_non_finite_json(cls, value: Any) -> Any:
        def validate(item: Any) -> None:
            if isinstance(item, float) and not isfinite(item):
                raise ValueError("JSON numeric values must be finite")
            if isinstance(item, dict):
                for nested in item.values():
                    validate(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    validate(nested)

        validate(value)
        return value


class ApiErrorDetail(ApiModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1_000)
    retryable: bool
    request_id: str = Field(pattern=r"^req_[0-9a-f]{16,32}$")


class ApiErrorResponse(ApiModel):
    error: ApiErrorDetail


class ApiKeyCreate(ApiModel):
    environment: Literal["test", "live"]
    scopes: set[ApiScope] = Field(min_length=1)
    expires_at: AwareDatetime | None = None

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> Any:
        return value.casefold() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_future_expiry(self) -> ApiKeyCreate:
        if self.expires_at is not None and self.expires_at <= datetime.now(UTC):
            raise ValueError("expires_at must be in the future")
        return self


class TestPlanDraftCreate(ApiModel):
    release_policy_id: str = Field(min_length=1, max_length=200)
    release_policy_version: str | None = Field(default=None, min_length=1, max_length=32)
    test_suite_id: str = Field(min_length=1, max_length=160)
    test_suite_version: str = Field(min_length=1, max_length=32)
    test_plan_version: str = Field(default="1.0", min_length=1, max_length=32)
    case_counts: dict[TestFamily, int] = Field(default_factory=dict, max_length=16)
    include_optional: set[TestFamily] = Field(default_factory=set, max_length=16)

    @field_validator("case_counts")
    @classmethod
    def validate_case_counts(
        cls,
        value: dict[TestFamily, int],
    ) -> dict[TestFamily, int]:
        if any(count < 1 or count > 10_000 for count in value.values()):
            raise ValueError("case counts must be between 1 and 10000")
        return value


class TestPlanConfirm(ApiModel):
    confirm: Literal[True]
    expected_profile_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_test_plan_version: str = Field(min_length=1, max_length=32)


def _upper(value: Any) -> Any:
    return value.upper() if isinstance(value, str) else value


class ApplicationProfileInput(ApiModel):
    version: str = Field(default="1.0.0", min_length=1, max_length=100)
    intended_use: str = Field(min_length=1, max_length=2_000)
    clinical_specialty: str | None = Field(default=None, max_length=200)
    input_type: InputType
    output_type: OutputType
    patient_facing: bool = False
    clinician_facing: bool = False
    uses_rag: bool = False
    uses_tools: bool = False
    uses_llm: bool = True
    uses_predictive_model: bool = False
    environment: ApplicationEnvironment = ApplicationEnvironment.STAGING
    endpoint_url: HttpUrl | None = None
    authentication_type: AuthenticationType = AuthenticationType.NONE
    request_schema: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "input_type",
        "output_type",
        "environment",
        "authentication_type",
        mode="before",
    )
    @classmethod
    def normalize_enum_values(cls, value: Any) -> Any:
        return _upper(value)

    @field_validator("endpoint_url")
    @classmethod
    def reject_endpoint_secrets(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None:
            validate_public_https_endpoint(str(value))
        return value


class SystemCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5_000)
    application_type: ApplicationType
    risk_tier: RiskTier
    active: bool = True
    profile: ApplicationProfileInput | None = None

    @field_validator("application_type", "risk_tier", mode="before")
    @classmethod
    def normalize_enum_values(cls, value: Any) -> Any:
        return _upper(value)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class SystemVersionCreate(ApiModel):
    version: str = Field(min_length=1, max_length=100)
    environment: ApplicationEnvironment = ApplicationEnvironment.STAGING
    endpoint_url: HttpUrl | None = None
    authentication_config: dict[str, Any] | None = None
    request_schema: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    active: bool = True

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> Any:
        return _upper(value)

    @field_validator("endpoint_url")
    @classmethod
    def reject_endpoint_secrets(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None:
            validate_public_https_endpoint(str(value))
        return value


class SingleEvaluationCreate(ApiModel):
    system_id: str = Field(min_length=1, max_length=64)
    system_version: str = Field(min_length=1, max_length=100)
    environment: ApplicationEnvironment = ApplicationEnvironment.STAGING
    sources: list[EvidenceSource] = Field(min_length=1)
    generated_output: GeneratedEvidenceOutput

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_artifact_defaults(cls, value: Any) -> Any:
        return add_default_evidence_types(value)

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> Any:
        return _upper(value)


EVALUATION_OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "smallClinicalText": {
        "summary": "Small clinical text",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "doc-1",
                    "type": "clinical_text",
                    "content": "Patient reports left knee pain.",
                }
            ],
            "generated_output": {
                "type": "clinical_text",
                "content": "Patient reports right knee pain.",
            },
        },
    },
    "structuredTranscript": {
        "summary": "Structured clinician-patient transcript",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "transcript-001",
                    "type": "transcript",
                    "content": [
                        {
                            "segment": 1,
                            "timestamp": "00:03",
                            "speaker": "PATIENT",
                            "text": "I have a severe headache.",
                        },
                        {
                            "segment": 2,
                            "timestamp": "00:22",
                            "speaker": "CLINICIAN",
                            "text": "When did it start?",
                        },
                    ],
                }
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "The patient reports a severe headache.",
            },
        },
    },
    "labResults": {
        "summary": "Structured laboratory results",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "labs-001",
                    "type": "lab_results",
                    "content": [
                        {"test": "creatinine", "value": 1.8, "unit": "mg/dL"},
                        {"test": "potassium", "value": 4.6, "unit": "mmol/L"},
                    ],
                }
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "Creatinine is 1.8 mg/dL and potassium is 4.6 mmol/L.",
            },
        },
    },
    "medicationList": {
        "summary": "Structured medication list",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "meds-001",
                    "type": "medication_list",
                    "content": [
                        {
                            "name": "Metformin",
                            "dose": 500,
                            "unit": "mg",
                            "frequency": "BID",
                            "route": "oral",
                            "status": "ACTIVE",
                        }
                    ],
                }
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "Continue metformin 500 mg twice daily by mouth.",
            },
        },
    },
    "multipleSources": {
        "summary": "Transcript, labs, and medications together",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "transcript-001",
                    "type": "transcript",
                    "content": [
                        {
                            "segment": 1,
                            "speaker": "PATIENT",
                            "text": "I take metformin twice daily.",
                        }
                    ],
                },
                {
                    "document_id": "labs-001",
                    "type": "lab_results",
                    "content": [{"test": "HbA1c", "value": 7.2, "unit": "%"}],
                },
                {
                    "document_id": "meds-001",
                    "type": "medication_list",
                    "content": [
                        {
                            "name": "Metformin",
                            "dose": 500,
                            "unit": "mg",
                            "frequency": "BID",
                        }
                    ],
                },
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "HbA1c is 7.2%. Continue metformin 500 mg twice daily.",
            },
        },
    },
    "longText": {
        "summary": "Multiline clinical text",
        "description": (
            "Programmatic clients should pass normal multiline strings to their JSON "
            "library. The serializer safely represents line breaks; callers should not "
            "manually flatten clinical content."
        ),
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "transcript-raw-001",
                    "type": "clinical_text",
                    "content": (
                        "Segment 1\n00:03\nPATIENT: I have a headache.\n\n"
                        "Segment 2\n00:22\nCLINICIAN: Do you have photophobia?"
                    ),
                }
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "Subjective:\nHeadache with possible photophobia.",
            },
        },
    },
    "ragContext": {
        "summary": "RAG evidence with retrieval provenance",
        "value": {
            "system_id": "sys_clinical_scribe",
            "system_version": "1.0.0",
            "sources": [
                {
                    "document_id": "rag-001",
                    "type": "rag_context",
                    "content": [
                        {
                            "chunk_id": "chunk-123",
                            "text": "The active dose is insulin 10 units nightly.",
                            "source_uri": "ehr://medications/current",
                            "retrieval_score": 0.87,
                            "metadata": {"collection": "medications"},
                        }
                    ],
                }
            ],
            "generated_output": {
                "type": "clinical_note",
                "content": "The patient uses insulin 100 units nightly.",
            },
        },
    },
}


class TestCaseCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    test_family: TestFamily
    version: str = Field(default="1.0.0", min_length=1, max_length=32)
    input_data: dict[str, Any] = Field(default_factory=dict)
    expectation: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


class TestSuiteCreate(ApiModel):
    slug: str | None = Field(default=None, min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=200)
    suite_type: str = Field(default="CUSTOM", min_length=1, max_length=32)
    active: bool = True
    version: str = Field(default="1.0.0", min_length=1, max_length=32)
    definition: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE", "PUBLISHED", "RETIRED"] = "ACTIVE"
    cases: list[TestCaseCreate] = Field(default_factory=list, max_length=10_000)
    data_classification: Literal["SYNTHETIC", "DE_IDENTIFIED"]
    contains_phi: Literal[False]
    no_phi_attested: Literal[True]

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", normalized):
            raise ValueError("slug may contain lowercase letters, numbers, '.', '_' and '-'")
        return normalized

    @field_validator("suite_type", "status", mode="before")
    @classmethod
    def normalize_statuses(cls, value: Any) -> Any:
        return _upper(value)


class TestRunTrigger(ApiModel):
    trigger: str = Field(default="api", min_length=1, max_length=50)
    commit_sha: str | None = Field(default=None, min_length=4, max_length=128)
    branch: str | None = Field(default=None, min_length=1, max_length=255)
    pull_request: int | None = Field(default=None, ge=1)
    repository: str | None = Field(default=None, min_length=1, max_length=255)


class TestRunCreate(ApiModel):
    system_id: str = Field(min_length=1, max_length=64)
    system_version: str = Field(min_length=1, max_length=100)
    environment: ApplicationEnvironment = ApplicationEnvironment.STAGING
    test_suite_id: str = Field(min_length=1, max_length=160)
    test_suite_version: str = Field(min_length=1, max_length=32)
    test_plan_id: str = Field(min_length=1, max_length=64)
    source: TestRunTrigger = Field(default_factory=TestRunTrigger)

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> Any:
        return _upper(value)


class ReleasePolicyCreate(ApiModel):
    policy_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    version: str = Field(default="1.0.0", min_length=1, max_length=32)
    active: bool = True
    rules: list[ReleasePolicyRule] = Field(min_length=1, max_length=500)
    required_coverage: dict[str, Any] = Field(default_factory=dict)
    missing_required_coverage_action: MissingRequiredCoverageAction = (
        MissingRequiredCoverageAction.NO_DECISION
    )
    critical_finding_override: bool = True

    @field_validator("policy_id")
    @classmethod
    def validate_policy_id(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError("policy_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def require_identity(self) -> ReleasePolicyCreate:
        if self.policy_id is None and self.name is None:
            raise ValueError("name or policy_id is required")
        return self

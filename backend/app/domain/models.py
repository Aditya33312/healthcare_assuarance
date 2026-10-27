"""Pydantic contracts used by the Clinical Compare Engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from backend.app.domain.enums import (
    AlignmentMethod,
    AssertionStatus,
    ClaimType,
    CoverageStatus,
    ExecutionStatus,
    FindingCategory,
    FindingType,
    Severity,
    SubjectType,
)
from backend.app.domain.taxonomy import canonical_category
from backend.app.version import ENGINE_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


ClinicalArtifactType = Literal[
    "clinical_text",
    "clinical_transcript",
    "clinical_note",
    "clinical_summary",
    "clinical_letter",
    "discharge_summary",
    "referral_letter",
    "patient_message",
    "question",
    "document",
    "structured_data",
    "audio_transcript",
    "image_report",
    "multimodal_text",
    "answer",
    "recommendation",
    "tool_action",
    "other",
]


class TextSpan(StrictModel):
    document_id: str | None = None
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("span offsets must exactly cover span text")
        return self


class EvidenceReference(StrictModel):
    """Additive provenance for canonical evidence without changing TextSpan."""

    evidence_unit_id: str
    document_id: str
    source_type: str
    source_path: str | None = None
    segment_id: str | None = None
    timestamp: str | None = None
    speaker: str | None = None
    section: str | None = None
    chunk_id: str | None = None
    source_uri: str | None = None
    rag_chunk_id: str | None = None
    source_document_id: str | None = None
    retrieval_score: str | None = None
    retrieval_metadata: dict[str, Any] | None = None
    raw_start_offset: int = Field(ge=0)
    raw_end_offset: int = Field(ge=0)
    normalized_start_offset: int = Field(ge=0)
    normalized_end_offset: int = Field(ge=0)


class Claim(StrictModel):
    schema_version: str = "1.0.0"
    claim_id: str
    concept: str = Field(min_length=1)
    type: ClaimType
    attributes: dict[str, Any] = Field(default_factory=dict)
    assertion: AssertionStatus = AssertionStatus.PRESENT
    subject: SubjectType = SubjectType.PATIENT
    source_span: TextSpan
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor: str = "deterministic"
    extractor_version: str = "0.1.0"


class SourceDocument(StrictModel):
    document_id: str
    type: ClinicalArtifactType = "clinical_text"
    content: str = Field(min_length=1)


class SourceBundle(StrictModel):
    documents: list[SourceDocument] = Field(min_length=1)


class GeneratedOutput(StrictModel):
    type: ClinicalArtifactType = "clinical_text"
    content: str = Field(min_length=1)


class Finding(StrictModel):
    schema_version: str = "1.0.0"
    finding_id: str
    type: FindingType
    category: FindingCategory
    severity: Severity
    source: TextSpan | None = None
    generated: TextSpan | None = None
    reason_code: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    evaluator: str = "clinical_compare"
    evaluator_version: str = ENGINE_VERSION
    severity_policy_version: str = "provisional-clinical-severity@0.3.0"
    rule_id: str | None = None
    rule_version: str | None = None
    alignment_method: AlignmentMethod = AlignmentMethod.NONE
    source_support: list[TextSpan] = Field(default_factory=list)
    uncertainty_reason: str | None = None
    candidate_evidence: list[TextSpan] = Field(default_factory=list)
    source_provenance: EvidenceReference | None = None
    generated_provenance: EvidenceReference | None = None

    @field_validator("category", mode="before")
    @classmethod
    def normalize_legacy_category(cls, value: object) -> FindingCategory:
        return canonical_category(value)

    @model_serializer(mode="wrap")
    def omit_absent_provenance(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Keep legacy serialized findings byte-for-byte when provenance is absent."""

        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("finding serialization must produce an object")
        if self.source_provenance is None:
            serialized.pop("source_provenance", None)
        if self.generated_provenance is None:
            serialized.pop("generated_provenance", None)
        return serialized


class ComponentVersions(StrictModel):
    gatekeeper_engine: str = ENGINE_VERSION
    ner_model: str = "deterministic-clinical-extractor@0.1.0"
    assertion_model: str = "assertion-rules@0.1.0"
    semantic_evaluator: str | None = None
    normalization: str = "clinical-normalizers@0.1.0"
    rule_set: str = "clinical-compare-rules@0.1.0"
    severity_policy: str = "provisional-clinical-severity@0.3.0"
    ingestion: str | None = None

    @model_serializer(mode="wrap")
    def omit_absent_ingestion_version(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("component version serialization must produce an object")
        if self.ingestion is None:
            serialized.pop("ingestion", None)
        return serialized


class CoverageLimitationDetail(StrictModel):
    code: str
    document_id: str | None = None
    message: str


class CoverageSummary(StrictModel):
    status: CoverageStatus
    claims_assessed: int = Field(ge=0)
    claims_unassessed: int = Field(ge=0)
    limitations: list[str] = Field(default_factory=list)
    limitation_details: list[CoverageLimitationDetail] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def omit_empty_limitation_details(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Preserve the established response shape unless structured details exist."""

        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("coverage serialization must produce an object")
        if not self.limitation_details:
            serialized.pop("limitation_details", None)
        return serialized


class FaithfulnessMetrics(StrictModel):
    score: float = Field(ge=0.0, le=100.0)
    supported: int = Field(ge=0)
    altered: int = Field(ge=0)
    added: int = Field(ge=0)
    omitted: int = Field(ge=0)
    inferred: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    critical_count: int = Field(ge=0)
    high_count: int = Field(ge=0)


class CompareResult(StrictModel):
    evaluation_id: str
    execution_status: ExecutionStatus
    coverage: CoverageSummary
    faithfulness: FaithfulnessMetrics
    findings: list[Finding]
    source_claims: list[Claim]
    generated_claims: list[Claim]
    versions: ComponentVersions = Field(default_factory=ComponentVersions)
    error_code: str | None = None

    @model_serializer(mode="wrap")
    def omit_absent_error_code(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("compare result serialization must produce an object")
        if self.error_code is None:
            serialized.pop("error_code", None)
        return serialized

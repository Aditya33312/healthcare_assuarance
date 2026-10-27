"""Strict public transport schemas for universal clinical evidence ingestion.

These models describe caller input only.  They intentionally do not perform
clinical normalization or turn structured records into prose; the ingestion
service owns that boundary and preserves the original values.
"""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from backend.app.domain.models import (
    ClinicalArtifactType,
    GeneratedOutput,
    SourceDocument,
)
from backend.app.ingestion.normalization import validate_decimal_bounds

_ALLOWED_TEXT_CONTROLS = {"\t", "\n", "\r"}


def _validate_text(value: str, *, allow_blank: bool = False) -> str:
    if not allow_blank and not value.strip():
        raise ValueError("text must not be blank")
    for character in value:
        category = unicodedata.category(character)
        if category == "Cs" or (category == "Cc" and character not in _ALLOWED_TEXT_CONTROLS):
            raise ValueError("text contains an unsupported control character")
    return value


def _validate_identifier(value: str) -> str:
    if not value.strip():
        raise ValueError("identifier must not be blank")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in value):
        raise ValueError("identifier contains an unsupported control character")
    return value


def _prepare_clinical_number(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("boolean values are not clinical numbers")
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("clinical numeric values must be finite")
        # JSON decoders expose a JSON decimal token as float.  Converting from its
        # shortest round-trippable representation avoids importing binary tails.
        return validate_decimal_bounds(Decimal(str(value)))
    if isinstance(value, Decimal):
        return validate_decimal_bounds(value)
    if isinstance(value, str) and not value.strip():
        raise ValueError("clinical values must not be blank")
    return value


ClinicalValue = Annotated[int | Decimal | str, BeforeValidator(_prepare_clinical_number)]

type JsonScalar = None | bool | str | int | Decimal
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def _validate_json_tree(value: JsonValue) -> JsonValue:
    pending: list[JsonValue] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                _validate_identifier(key)
                pending.append(nested)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str):
            _validate_text(current, allow_blank=True)
        elif isinstance(current, Decimal):
            validate_decimal_bounds(current)
    return value


class IngestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextEvidenceSource(SourceDocument):
    """Backward-compatible source whose content is already clinical text."""

    type: ClinicalArtifactType = "clinical_text"
    document_id: str = Field(min_length=1, max_length=512)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _validate_text(value)


class TranscriptSpeaker(StrEnum):
    PATIENT = "PATIENT"
    CLINICIAN = "CLINICIAN"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class TranscriptSegment(IngestionModel):
    segment: int | str
    timestamp: str | None = Field(default=None, max_length=100)
    speaker: TranscriptSpeaker = TranscriptSpeaker.UNKNOWN
    text: str = Field(min_length=1)

    @field_validator("segment", mode="before")
    @classmethod
    def validate_segment(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("segment must be an integer or string identifier")
        if isinstance(value, int) and value < 0:
            raise ValueError("integer segment identifiers must be non-negative")
        if isinstance(value, str):
            if len(value) > 200:
                raise ValueError("segment identifiers must not exceed 200 characters")
            return _validate_identifier(value)
        return value

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        return _validate_identifier(value) if value is not None else None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class TranscriptSource(IngestionModel):
    document_id: str = Field(min_length=1, max_length=512)
    type: Literal["transcript"]
    content: list[TranscriptSegment] = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)


class LabReferenceRange(IngestionModel):
    low: ClinicalValue | None = None
    high: ClinicalValue | None = None
    text: str | None = Field(default=None, max_length=500)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        return _validate_text(value) if value is not None else None


class LabResultItem(IngestionModel):
    test: str = Field(min_length=1, max_length=500)
    value: ClinicalValue
    unit: str | None = Field(default=None, max_length=100)
    reference_range: LabReferenceRange | str | None = None
    flag: str | None = Field(default=None, max_length=100)
    timestamp: str | None = Field(default=None, max_length=100)

    @field_validator("test")
    @classmethod
    def validate_test(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("unit", "flag", "timestamp")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _validate_text(value) if value is not None else None

    @field_validator("reference_range")
    @classmethod
    def validate_reference_range(
        cls, value: LabReferenceRange | str | None
    ) -> LabReferenceRange | str | None:
        return _validate_text(value) if isinstance(value, str) else value


class LabResultsSource(IngestionModel):
    document_id: str = Field(min_length=1, max_length=512)
    type: Literal["lab_results"]
    content: list[LabResultItem] = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)


class MedicationItem(IngestionModel):
    name: str = Field(min_length=1, max_length=500)
    dose: ClinicalValue | None = None
    unit: str | None = Field(default=None, max_length=100)
    frequency: str | None = Field(default=None, max_length=200)
    route: str | None = Field(default=None, max_length=200)
    status: str | None = Field(default=None, max_length=100)
    instructions: str | None = Field(default=None, max_length=2_000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("dose")
    @classmethod
    def validate_dose(cls, value: ClinicalValue | None) -> int | Decimal | None:
        if value is None or isinstance(value, (int, Decimal)):
            return value
        try:
            dose = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError("medication dose must be a finite numeric value") from exc
        if not dose.is_finite():
            raise ValueError("medication dose must be a finite numeric value")
        return validate_decimal_bounds(dose)

    @field_validator("unit", "frequency", "route", "status", "instructions")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _validate_text(value) if value is not None else None


class MedicationListSource(IngestionModel):
    document_id: str = Field(min_length=1, max_length=512)
    type: Literal["medication_list"]
    content: list[MedicationItem] = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)


class StructuredJsonSource(IngestionModel):
    document_id: str = Field(min_length=1, max_length=512)
    type: Literal["structured_json"]
    content: JsonObject

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: JsonObject) -> JsonObject:
        _validate_json_tree(value)
        return value


class RagChunk(IngestionModel):
    chunk_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    source_uri: str | None = Field(default=None, max_length=2_048)
    source_document_id: str | None = Field(default=None, max_length=512)
    retrieval_score: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("chunk_id", "source_document_id")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        return _validate_identifier(value) if value is not None else None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str | None) -> str | None:
        return _validate_text(value) if value is not None else None

    @field_validator("retrieval_score", mode="before")
    @classmethod
    def validate_retrieval_score(cls, value: Any) -> Any:
        prepared = _prepare_clinical_number(value)
        if isinstance(prepared, str):
            try:
                prepared = Decimal(prepared.strip())
            except InvalidOperation as exc:
                raise ValueError("retrieval score must be a finite decimal") from exc
        return validate_decimal_bounds(prepared) if isinstance(prepared, Decimal) else prepared

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: JsonObject) -> JsonObject:
        _validate_json_tree(value)
        return value


class RagContextSource(IngestionModel):
    document_id: str = Field(min_length=1, max_length=512)
    type: Literal["rag_context"]
    content: list[RagChunk] = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value)


EvidenceSource = Annotated[
    TextEvidenceSource
    | TranscriptSource
    | LabResultsSource
    | MedicationListSource
    | StructuredJsonSource
    | RagContextSource,
    Field(discriminator="type"),
]


class TextGeneratedOutput(GeneratedOutput):
    """Backward-compatible generated output that is already clinical text."""

    type: ClinicalArtifactType = "clinical_text"

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _validate_text(value)


class StructuredJsonGeneratedOutput(IngestionModel):
    type: Literal["structured_json"]
    content: JsonObject

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: JsonObject) -> JsonObject:
        _validate_json_tree(value)
        return value


GeneratedEvidenceOutput = Annotated[
    TextGeneratedOutput | StructuredJsonGeneratedOutput,
    Field(discriminator="type"),
]


def add_default_evidence_types(value: Any) -> Any:
    """Inject legacy defaults before discriminated-union validation.

    The pre-existing API allowed callers to omit ``type`` on text sources and
    generated output.  It also allowed application code to pass the old domain
    model instances directly, so both shapes are promoted without mutating the
    caller's object.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, dict):
        return value

    prepared = dict(value)
    raw_sources = prepared.get("sources")
    if isinstance(raw_sources, (list, tuple)):
        sources: list[Any] = []
        for raw_source in raw_sources:
            if isinstance(raw_source, BaseModel):
                raw_source = raw_source.model_dump(mode="python")
            if isinstance(raw_source, dict):
                source = dict(raw_source)
                source.setdefault("type", "clinical_text")
                sources.append(source)
            else:
                sources.append(raw_source)
        prepared["sources"] = sources

    raw_output = prepared.get("generated_output")
    if isinstance(raw_output, BaseModel):
        raw_output = raw_output.model_dump(mode="python")
    if isinstance(raw_output, dict):
        output = dict(raw_output)
        output.setdefault("type", "clinical_text")
        prepared["generated_output"] = output
    return prepared


__all__ = [
    "EvidenceSource",
    "GeneratedEvidenceOutput",
    "JsonObject",
    "JsonValue",
    "LabResultItem",
    "LabResultsSource",
    "MedicationItem",
    "MedicationListSource",
    "RagChunk",
    "RagContextSource",
    "StructuredJsonGeneratedOutput",
    "StructuredJsonSource",
    "TextEvidenceSource",
    "TextGeneratedOutput",
    "TranscriptSegment",
    "TranscriptSource",
    "TranscriptSpeaker",
    "add_default_evidence_types",
]

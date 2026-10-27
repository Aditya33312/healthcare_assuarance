"""Internal canonical evidence contracts with reversible provenance."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from backend.app.domain.models import (
    ClinicalArtifactType,
    GeneratedOutput,
    SourceBundle,
    SourceDocument,
)


class CanonicalModel(BaseModel):
    """Strict immutable base for ingestion artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceProvenance(CanonicalModel):
    """Location of a normalized span in its original evidence unit."""

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


class CanonicalEvidenceUnit(CanonicalModel):
    """Smallest independently provenance-bearing piece of evidence."""

    evidence_unit_id: str
    document_id: str
    source_type: str
    source_path: str | None = None
    segment_id: str | None = None
    timestamp: str | None = None
    speaker: str | None = None
    section: str | None = None
    source_uri: str | None = None
    raw_text: str
    normalized_text: str
    content_hash: str
    raw_start_offset: int = Field(ge=0)
    raw_end_offset: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalChunk(CanonicalModel):
    """A deterministic extraction chunk that always slices normalized text."""

    chunk_id: str
    document_id: str
    text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    evidence_unit_ids: tuple[str, ...]


class CanonicalDocument(CanonicalModel):
    """Canonical view of one source or generated output."""

    document_id: str
    source_type: str
    raw_content: Any
    raw_serialized: str
    normalized_text: str
    content_hash: str
    size_class: str
    evidence_units: tuple[CanonicalEvidenceUnit, ...]
    chunks: tuple[CanonicalChunk, ...]

    def provenance_for_span(self, start: int, end: int) -> EvidenceProvenance | None:
        """Return the narrowest unit overlapping a normalized document span."""

        candidates = [
            unit
            for unit in self.evidence_units
            if unit.start_offset < end and start < unit.end_offset
        ]
        if not candidates:
            return None
        unit = min(candidates, key=lambda item: item.end_offset - item.start_offset)
        normalized_relative_start = max(0, start - unit.start_offset)
        normalized_relative_end = min(len(unit.normalized_text), end - unit.start_offset)
        # Line-ending normalization is the only text transport rewrite. Mapping by
        # prefix preserves exact offsets for LF input and conservatively maps CRLF.
        if unit.metadata.get("kind") in {"text", "transcript_segment", "rag_chunk"}:
            raw_prefix_start = unit.normalized_text[:normalized_relative_start]
            raw_prefix_end = unit.normalized_text[:normalized_relative_end]
            raw_relative_start = _normalized_prefix_to_raw_offset(unit.raw_text, raw_prefix_start)
            raw_relative_end = _normalized_prefix_to_raw_offset(unit.raw_text, raw_prefix_end)
        else:
            # Structured records are addressed by source_path. A normalized display
            # span is not a byte-identical substring of JSON, so conservatively
            # reference the complete raw unit instead of inventing scalar offsets.
            raw_relative_start = 0
            raw_relative_end = len(unit.raw_text)
        chunk_id = next(
            (
                chunk.chunk_id
                for chunk in self.chunks
                if chunk.start_offset <= start and end <= chunk.end_offset
            ),
            None,
        )
        return EvidenceProvenance(
            evidence_unit_id=unit.evidence_unit_id,
            document_id=unit.document_id,
            source_type=unit.source_type,
            source_path=unit.source_path,
            segment_id=unit.segment_id,
            timestamp=unit.timestamp,
            speaker=unit.speaker,
            section=unit.section,
            chunk_id=chunk_id,
            source_uri=unit.source_uri,
            rag_chunk_id=_optional_string(unit.metadata.get("chunk_id")),
            source_document_id=_optional_string(
                unit.metadata.get("source_document_id")
            ),
            retrieval_score=_optional_decimal_string(
                unit.metadata.get("retrieval_score")
            ),
            retrieval_metadata=(
                cast(dict[str, Any], unit.metadata["retrieval_metadata"])
                if isinstance(unit.metadata.get("retrieval_metadata"), dict)
                else None
            ),
            raw_start_offset=unit.raw_start_offset + raw_relative_start,
            raw_end_offset=unit.raw_start_offset + raw_relative_end,
            normalized_start_offset=start,
            normalized_end_offset=end,
        )

    def encrypted_envelope(self) -> dict[str, Any]:
        """Return the PHI-bearing representation stored only in encrypted columns."""

        return {
            "schema_version": "universal-evidence@1.0",
            "document_id": self.document_id,
            "source_type": self.source_type,
            "raw_content": self.raw_content,
            "raw_serialized": self.raw_serialized,
            "normalized_content": self.normalized_text,
            "content_hash": self.content_hash,
            "canonical_evidence_units": [
                unit.model_dump(mode="json") for unit in self.evidence_units
            ],
        }


class CoverageLimitation(CanonicalModel):
    """Machine-readable ingestion limitation safe for the public response."""

    code: str
    document_id: str | None = None
    message: str


class IngestionStats(CanonicalModel):
    payload_bytes: int = Field(ge=0)
    source_count: int = Field(ge=0)
    source_types: tuple[str, ...]
    evidence_unit_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    normalization_seconds: float = Field(ge=0.0)
    ingestion_seconds: float = Field(ge=0.0)
    size_class: str


class IngestionResult(CanonicalModel):
    """Canonical data plus the unchanged text contracts consumed by Compare."""

    source_documents: tuple[CanonicalDocument, ...]
    generated_document: CanonicalDocument
    limitations: tuple[CoverageLimitation, ...] = ()
    stats: IngestionStats

    @property
    def source_bundle(self) -> SourceBundle:
        return SourceBundle(
            documents=[
                SourceDocument(
                    document_id=document.document_id,
                    type=_domain_source_type(document.source_type),
                    content=document.normalized_text,
                )
                for document in self.source_documents
            ]
        )

    @property
    def generated_output(self) -> GeneratedOutput:
        return GeneratedOutput(
            type=_domain_generated_type(self.generated_document.source_type),
            content=self.generated_document.normalized_text,
        )


def _domain_source_type(source_type: str) -> ClinicalArtifactType:
    mapped = {
        "transcript": "clinical_transcript",
        "lab_results": "structured_data",
        "medication_list": "structured_data",
        "structured_json": "structured_data",
        "rag_context": "document",
    }.get(source_type, source_type)
    return cast("ClinicalArtifactType", mapped)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_decimal_string(value: object) -> str | None:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value if isinstance(value, str) else None


def _domain_generated_type(source_type: str) -> ClinicalArtifactType:
    mapped = "structured_data" if source_type == "structured_json" else source_type
    return cast("ClinicalArtifactType", mapped)


def _normalized_prefix_to_raw_offset(raw_text: str, normalized_prefix: str) -> int:
    """Map an LF-normalized prefix length back to the raw CR/LF representation."""

    if not normalized_prefix:
        return 0
    target = len(normalized_prefix)
    raw_index = 0
    normalized_count = 0
    while raw_index < len(raw_text) and normalized_count < target:
        if raw_text.startswith("\r\n", raw_index):
            raw_index += 2
        else:
            raw_index += 1
        normalized_count += 1
    return raw_index

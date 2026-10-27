"""Configuration-derived safeguards for bounded evidence ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from backend.app.core.config import Settings


class EvidenceSizeClass(StrEnum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    VERY_LARGE = "VERY_LARGE"


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    """One immutable limits snapshot for a single evaluation request."""

    max_source_count: int = 100
    max_source_bytes: int = 1_048_576
    max_generated_output_bytes: int = 1_048_576
    max_total_evidence_bytes: int = 2_097_152
    max_transcript_segments: int = 10_000
    max_structured_depth: int = 32
    max_json_array_items: int = 100_000
    max_canonical_evidence_units: int = 25_000
    max_extracted_claims: int = 50_000
    max_processing_seconds: float = 120.0
    small_bytes: int = 10_240
    medium_bytes: int = 102_400
    large_bytes: int = 1_048_576
    chunk_characters: int = 12_000
    chunk_overlap_characters: int = 500
    max_chunks: int = 5_000

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_source_count,
            self.max_source_bytes,
            self.max_generated_output_bytes,
            self.max_total_evidence_bytes,
            self.max_transcript_segments,
            self.max_structured_depth,
            self.max_json_array_items,
            self.max_canonical_evidence_units,
            self.max_extracted_claims,
            self.small_bytes,
            self.medium_bytes,
            self.large_bytes,
            self.chunk_characters,
            self.max_chunks,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("ingestion limits must be positive")
        if self.max_processing_seconds <= 0:
            raise ValueError("max_processing_seconds must be positive")
        if not self.small_bytes < self.medium_bytes < self.large_bytes:
            raise ValueError("input size thresholds must be strictly increasing")
        if not 0 <= self.chunk_overlap_characters < self.chunk_characters:
            raise ValueError("chunk overlap must be smaller than the chunk size")
        if self.max_source_bytes > self.max_total_evidence_bytes:
            raise ValueError("max_source_bytes cannot exceed max_total_evidence_bytes")
        if self.max_generated_output_bytes > self.max_total_evidence_bytes:
            raise ValueError(
                "max_generated_output_bytes cannot exceed max_total_evidence_bytes"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            max_source_count=settings.max_source_count,
            max_source_bytes=settings.max_source_bytes,
            max_generated_output_bytes=settings.max_generated_output_bytes,
            max_total_evidence_bytes=settings.max_total_evidence_bytes,
            max_transcript_segments=settings.max_transcript_segments,
            max_structured_depth=settings.max_structured_depth,
            max_json_array_items=settings.max_json_array_items,
            max_canonical_evidence_units=settings.max_canonical_evidence_units,
            max_extracted_claims=settings.max_extracted_claims,
            max_processing_seconds=settings.max_processing_seconds,
            small_bytes=settings.small_bytes,
            medium_bytes=settings.medium_bytes,
            large_bytes=settings.large_bytes,
            chunk_characters=settings.chunk_characters,
            chunk_overlap_characters=settings.chunk_overlap_characters,
            max_chunks=settings.max_chunks,
        )

    def size_class(self, byte_count: int) -> EvidenceSizeClass:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if byte_count <= self.small_bytes:
            return EvidenceSizeClass.SMALL
        if byte_count <= self.medium_bytes:
            return EvidenceSizeClass.MEDIUM
        if byte_count <= self.large_bytes:
            return EvidenceSizeClass.LARGE
        return EvidenceSizeClass.VERY_LARGE

    def classify_size(self, byte_count: int) -> EvidenceSizeClass:
        """Compatibility spelling for callers that prefer an action name."""

        return self.size_class(byte_count)


__all__ = ["EvidenceSizeClass", "IngestionLimits"]

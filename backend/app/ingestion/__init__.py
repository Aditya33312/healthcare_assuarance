"""Universal evidence ingestion contracts and canonical processing primitives."""

from backend.app.ingestion.canonical import (
    CanonicalDocument,
    CanonicalEvidenceUnit,
    CoverageLimitation,
    EvidenceProvenance,
    IngestionResult,
    IngestionStats,
)
from backend.app.ingestion.extractor import (
    CanonicalEvidenceExtractor,
    ClinicalProcessingLimitError,
)
from backend.app.ingestion.limits import EvidenceSizeClass, IngestionLimits
from backend.app.ingestion.schemas import (
    EvidenceSource,
    GeneratedEvidenceOutput,
    LabResultItem,
    LabResultsSource,
    MedicationItem,
    MedicationListSource,
    RagChunk,
    RagContextSource,
    StructuredJsonGeneratedOutput,
    StructuredJsonSource,
    TextEvidenceSource,
    TextGeneratedOutput,
    TranscriptSegment,
    TranscriptSource,
)
from backend.app.ingestion.service import UniversalEvidenceIngestionService

__all__ = [
    "CanonicalDocument",
    "CanonicalEvidenceExtractor",
    "CanonicalEvidenceUnit",
    "ClinicalProcessingLimitError",
    "CoverageLimitation",
    "EvidenceProvenance",
    "EvidenceSizeClass",
    "EvidenceSource",
    "GeneratedEvidenceOutput",
    "IngestionLimits",
    "IngestionResult",
    "IngestionStats",
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
    "UniversalEvidenceIngestionService",
]

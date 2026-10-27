"""Universal evidence ingestion before Clinical NLP and comparison."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.app.clinical_nlp.normalization.frequency_normalizer import normalize_frequency
from backend.app.clinical_nlp.normalization.medication_normalizer import normalize_medication
from backend.app.clinical_nlp.normalization.terminology_normalizer import normalize_terminology
from backend.app.clinical_nlp.normalization.unit_normalizer import (
    normalize_quantity,
    normalize_unit,
)
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.ingestion.canonical import (
    CanonicalDocument,
    CanonicalEvidenceUnit,
    CoverageLimitation,
    IngestionResult,
    IngestionStats,
)
from backend.app.ingestion.chunking import ChunkLimitError, build_chunks
from backend.app.ingestion.limits import IngestionLimits
from backend.app.ingestion.normalization import (
    UnsafeTextError,
    canonical_json,
    content_hash,
    display_scalar,
    json_safe,
    normalize_text,
    raw_serialization,
)
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
    TranscriptSegment,
    TranscriptSource,
)

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_HEADING = re.compile(r"^[A-Za-z][A-Za-z0-9 /&()_-]{0,100}:?$")
_ROUTES = {
    "oral": "ORAL",
    "orally": "ORAL",
    "po": "ORAL",
    "p.o.": "ORAL",
    "iv": "INTRAVENOUS",
    "i.v.": "INTRAVENOUS",
    "intravenous": "INTRAVENOUS",
    "intravenously": "INTRAVENOUS",
    "im": "INTRAMUSCULAR",
    "intramuscular": "INTRAMUSCULAR",
    "sc": "SUBCUTANEOUS",
    "sq": "SUBCUTANEOUS",
    "subcutaneous": "SUBCUTANEOUS",
    "inhaled": "INHALATION",
    "inhalation": "INHALATION",
    "topical": "TOPICAL",
    "sublingual": "SUBLINGUAL",
    "sl": "SUBLINGUAL",
}


@dataclass(slots=True)
class _UnitDraft:
    raw_text: str
    normalized_text: str
    source_path: str | None = None
    segment_id: str | None = None
    timestamp: str | None = None
    speaker: str | None = None
    section: str | None = None
    source_uri: str | None = None
    raw_start_offset: int = 0
    raw_end_offset: int | None = None
    normalized_start_offset: int | None = None
    normalized_end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _EvidenceUnitLimitError(ValueError):
    """Internal signal raised before an excessive unit can be allocated."""


@dataclass(slots=True)
class _EvidenceUnitBudget:
    remaining: int

    def consume(self) -> None:
        if self.remaining <= 0:
            raise _EvidenceUnitLimitError
        self.remaining -= 1


def _unit_id(document_id: str, source_path: str | None, ordinal: int, raw_text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\x1f{source_path or ''}\x1f{ordinal}\x1f{raw_text}".encode()
    ).hexdigest()[:20]
    return f"evidence_{digest}"


def _normalized_boundary_map(raw_text: str) -> list[int]:
    boundaries = [0]
    raw_index = 0
    while raw_index < len(raw_text):
        raw_index += 2 if raw_text.startswith("\r\n", raw_index) else 1
        boundaries.append(raw_index)
    return boundaries


def _section_for_paragraph(paragraph: str, current: str | None) -> str | None:
    first_line = next((line.strip() for line in paragraph.splitlines() if line.strip()), "")
    candidate = first_line.rstrip(":")
    if first_line and _HEADING.fullmatch(first_line) and (
        first_line.endswith(":") or candidate.isupper()
    ):
        return candidate
    return current


def _text_drafts(
    raw_text: str,
    normalized: str,
    unit_budget: _EvidenceUnitBudget,
) -> list[_UnitDraft]:
    raw_boundaries = _normalized_boundary_map(raw_text)
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(normalized):
        if normalized[start : match.start()].strip():
            unit_budget.consume()
            ranges.append((start, match.start()))
        start = match.end()
    if normalized[start:].strip():
        unit_budget.consume()
        ranges.append((start, len(normalized)))
    if not ranges:
        unit_budget.consume()
        ranges.append((0, len(normalized)))

    drafts: list[_UnitDraft] = []
    section: str | None = None
    for normalized_start, normalized_end in ranges:
        paragraph = normalized[normalized_start:normalized_end]
        section = _section_for_paragraph(paragraph, section)
        raw_start = raw_boundaries[normalized_start]
        raw_end = raw_boundaries[normalized_end]
        drafts.append(
            _UnitDraft(
                raw_text=raw_text[raw_start:raw_end],
                normalized_text=paragraph,
                section=section,
                raw_start_offset=raw_start,
                raw_end_offset=raw_end,
                normalized_start_offset=normalized_start,
                normalized_end_offset=normalized_end,
                metadata={"kind": "text"},
            )
        )
    return drafts


def _normalize_route(value: str) -> str | None:
    candidate = re.sub(r"\s+", " ", value.strip().casefold())
    return _ROUTES.get(candidate)


def _normalized_test(value: str) -> str:
    return normalize_terminology(value) or re.sub(r"\s+", " ", value.strip().casefold())


def _json_child_path(path: str, key: str) -> str:
    """Append an unambiguous JSONPath member, escaping non-identifier keys."""

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{canonical_json(key)}]"


class UniversalEvidenceIngestionService:
    """Validate, canonicalize, normalize, section, and chunk evidence."""

    version = "universal-evidence-ingestion@1.0.0"

    def __init__(self, limits: IngestionLimits | None = None) -> None:
        self.limits = limits or IngestionLimits()

    def _raise_too_large(self, message: str) -> None:
        raise GatekeeperError(
            ErrorCode.PAYLOAD_TOO_LARGE,
            message,
            retryable=False,
            http_status=413,
        )

    def _raise_invalid(self, message: str) -> None:
        raise GatekeeperError(
            ErrorCode.INGESTION_FAILED,
            message,
            retryable=False,
            http_status=422,
        )

    def _size_class(self, byte_count: int) -> str:
        if byte_count <= self.limits.small_bytes:
            return "SMALL"
        if byte_count <= self.limits.medium_bytes:
            return "MEDIUM"
        if byte_count <= self.limits.large_bytes:
            return "LARGE"
        return "VERY_LARGE"

    def ingest(
        self,
        sources: Sequence[EvidenceSource],
        generated_output: GeneratedEvidenceOutput,
    ) -> IngestionResult:
        started_at = time.monotonic()
        if not sources:
            self._raise_invalid("At least one source document is required.")
        if len(sources) > self.limits.max_source_count:
            self._raise_too_large("The request exceeds the configured source-count limit.")
        document_ids = [source.document_id for source in sources]
        if len(document_ids) != len(set(document_ids)):
            self._raise_invalid("Source document IDs must be unique within an evaluation.")
        if "generated_output" in document_ids:
            self._raise_invalid(
                "Source document ID 'generated_output' is reserved for generated evidence."
            )

        source_raw = [raw_serialization(source.content) for source in sources]
        output_raw = raw_serialization(generated_output.content)
        source_bytes = [len(value.encode("utf-8")) for value in source_raw]
        output_bytes = len(output_raw.encode("utf-8"))
        if any(size > self.limits.max_source_bytes for size in source_bytes):
            self._raise_too_large("A source exceeds the configured per-source size limit.")
        if output_bytes > self.limits.max_generated_output_bytes:
            self._raise_too_large("The generated output exceeds its configured size limit.")
        payload_bytes = sum(source_bytes) + output_bytes
        if payload_bytes > self.limits.max_total_evidence_bytes:
            self._raise_too_large(
                "The evaluation evidence exceeds the configured total size limit."
            )

        normalization_started = time.monotonic()
        limitations: list[CoverageLimitation] = []
        unit_budget = _EvidenceUnitBudget(self.limits.max_canonical_evidence_units)
        try:
            source_documents = tuple(
                self._canonicalize_source(source, raw, limitations, unit_budget)
                for source, raw in zip(sources, source_raw, strict=True)
            )
            generated_document = self._canonicalize_generated(
                generated_output,
                output_raw,
                limitations,
                unit_budget,
            )
        except UnsafeTextError:
            self._raise_invalid("Evidence contains blank text or unsupported control characters.")
        except _EvidenceUnitLimitError:
            self._raise_too_large("Evidence exceeds the configured canonical-unit limit.")
        except ChunkLimitError:
            self._raise_too_large("Evidence exceeds the configured safe chunk-count limit.")
        normalization_seconds = time.monotonic() - normalization_started

        evidence_unit_count = sum(
            len(document.evidence_units) for document in source_documents
        ) + len(generated_document.evidence_units)
        chunk_count = sum(len(document.chunks) for document in source_documents) + len(
            generated_document.chunks
        )
        if evidence_unit_count > self.limits.max_canonical_evidence_units:
            self._raise_too_large("Evidence exceeds the configured canonical-unit limit.")
        if chunk_count > self.limits.max_chunks:
            self._raise_too_large("Evidence exceeds the configured safe chunk-count limit.")

        return IngestionResult(
            source_documents=source_documents,
            generated_document=generated_document,
            limitations=tuple(limitations),
            stats=IngestionStats(
                payload_bytes=payload_bytes,
                source_count=len(source_documents),
                source_types=tuple(document.source_type for document in source_documents),
                evidence_unit_count=evidence_unit_count,
                chunk_count=chunk_count,
                normalization_seconds=normalization_seconds,
                ingestion_seconds=time.monotonic() - started_at,
                size_class=self._size_class(max(*source_bytes, output_bytes)),
            ),
        )

    def _canonicalize_source(
        self,
        source: EvidenceSource,
        raw: str,
        limitations: list[CoverageLimitation],
        unit_budget: _EvidenceUnitBudget,
    ) -> CanonicalDocument:
        source_type = source.type
        if isinstance(source, TranscriptSource):
            drafts = self._transcript_drafts(source.content, unit_budget)
            return self._from_drafts(source.document_id, source_type, source.content, raw, drafts)
        if isinstance(source, LabResultsSource):
            drafts = self._lab_drafts(
                source.content,
                source.document_id,
                limitations,
                unit_budget,
            )
            return self._from_drafts(source.document_id, source_type, source.content, raw, drafts)
        if isinstance(source, MedicationListSource):
            drafts = self._medication_drafts(
                source.content,
                source.document_id,
                limitations,
                unit_budget,
            )
            return self._from_drafts(source.document_id, source_type, source.content, raw, drafts)
        if isinstance(source, StructuredJsonSource):
            drafts = self._structured_json_drafts(
                source.content,
                source.document_id,
                limitations,
                unit_budget,
            )
            return self._from_drafts(source.document_id, source_type, source.content, raw, drafts)
        if isinstance(source, RagContextSource):
            drafts = self._rag_drafts(source.content, unit_budget)
            return self._from_drafts(source.document_id, source_type, source.content, raw, drafts)
        if not isinstance(source, TextEvidenceSource):
            raise TypeError("unsupported validated evidence source")
        normalized = normalize_text(source.content)
        return self._text_document(
            source.document_id,
            source_type,
            source.content,
            normalized,
            unit_budget,
        )

    def _canonicalize_generated(
        self,
        output: GeneratedEvidenceOutput,
        raw: str,
        limitations: list[CoverageLimitation],
        unit_budget: _EvidenceUnitBudget,
    ) -> CanonicalDocument:
        if isinstance(output, StructuredJsonGeneratedOutput):
            drafts = self._structured_json_drafts(
                output.content,
                "generated_output",
                limitations,
                unit_budget,
            )
            return self._from_drafts(
                "generated_output",
                output.type,
                output.content,
                raw,
                drafts,
            )
        if not isinstance(output.content, str):
            raise TypeError("unsupported validated generated output")
        normalized = normalize_text(output.content)
        return self._text_document(
            "generated_output",
            output.type,
            output.content,
            normalized,
            unit_budget,
        )

    def _text_document(
        self,
        document_id: str,
        source_type: str,
        raw_text: str,
        normalized: str,
        unit_budget: _EvidenceUnitBudget,
    ) -> CanonicalDocument:
        drafts = _text_drafts(raw_text, normalized, unit_budget)
        units: list[CanonicalEvidenceUnit] = []
        for ordinal, draft in enumerate(drafts):
            if draft.normalized_start_offset is None or draft.normalized_end_offset is None:
                raise RuntimeError("text evidence unit is missing normalized offsets")
            units.append(
                CanonicalEvidenceUnit(
                    evidence_unit_id=_unit_id(document_id, None, ordinal, draft.raw_text),
                    document_id=document_id,
                    source_type=source_type,
                    raw_text=draft.raw_text,
                    normalized_text=draft.normalized_text,
                    content_hash=content_hash(draft.raw_text),
                    raw_start_offset=draft.raw_start_offset,
                    raw_end_offset=draft.raw_end_offset or len(draft.raw_text),
                    start_offset=draft.normalized_start_offset,
                    end_offset=draft.normalized_end_offset,
                    section=draft.section,
                    metadata=draft.metadata,
                )
            )
        return self._finish_document(
            document_id,
            source_type,
            raw_text,
            raw_text,
            normalized,
            tuple(units),
        )

    def _from_drafts(
        self,
        document_id: str,
        source_type: str,
        raw_content: Any,
        raw: str,
        drafts: Sequence[_UnitDraft],
    ) -> CanonicalDocument:
        if not drafts:
            self._raise_invalid("Structured evidence must contain at least one usable item.")
        pieces: list[str] = []
        units: list[CanonicalEvidenceUnit] = []
        offset = 0
        for ordinal, draft in enumerate(drafts):
            if ordinal:
                pieces.append("\n")
                offset += 1
            normalized = normalize_text(draft.normalized_text)
            start = offset
            pieces.append(normalized)
            offset += len(normalized)
            units.append(
                CanonicalEvidenceUnit(
                    evidence_unit_id=_unit_id(
                        document_id,
                        draft.source_path,
                        ordinal,
                        draft.raw_text,
                    ),
                    document_id=document_id,
                    source_type=source_type,
                    source_path=draft.source_path,
                    segment_id=draft.segment_id,
                    timestamp=draft.timestamp,
                    speaker=draft.speaker,
                    section=draft.section,
                    source_uri=draft.source_uri,
                    raw_text=draft.raw_text,
                    normalized_text=normalized,
                    content_hash=content_hash(draft.raw_text),
                    raw_start_offset=draft.raw_start_offset,
                    raw_end_offset=draft.raw_end_offset or len(draft.raw_text),
                    start_offset=start,
                    end_offset=offset,
                    metadata=draft.metadata,
                )
            )
        normalized_text = "".join(pieces)
        return self._finish_document(
            document_id,
            source_type,
            json_safe(raw_content),
            raw,
            normalized_text,
            tuple(units),
        )

    def _finish_document(
        self,
        document_id: str,
        source_type: str,
        raw_content: Any,
        raw: str,
        normalized_text: str,
        units: tuple[CanonicalEvidenceUnit, ...],
    ) -> CanonicalDocument:
        chunks = build_chunks(
            document_id,
            normalized_text,
            units,
            max_characters=self.limits.chunk_characters,
            overlap_characters=self.limits.chunk_overlap_characters,
            max_chunks=self.limits.max_chunks,
        )
        return CanonicalDocument(
            document_id=document_id,
            source_type=source_type,
            raw_content=raw_content,
            raw_serialized=raw,
            normalized_text=normalized_text,
            content_hash=content_hash(raw),
            size_class=self._size_class(len(raw.encode("utf-8"))),
            evidence_units=units,
            chunks=chunks,
        )

    def _transcript_drafts(
        self,
        segments: Sequence[TranscriptSegment],
        unit_budget: _EvidenceUnitBudget,
    ) -> list[_UnitDraft]:
        if len(segments) > self.limits.max_transcript_segments:
            self._raise_too_large("Transcript exceeds the configured segment-count limit.")
        segment_ids = [str(segment.segment) for segment in segments]
        if len(segment_ids) != len(set(segment_ids)):
            self._raise_invalid("Transcript segment identifiers must be unique.")
        drafts: list[_UnitDraft] = []
        for index, segment in enumerate(segments):
            unit_budget.consume()
            text = segment.text
            normalized = normalize_text(text)
            segment_id = str(segment.segment)
            drafts.append(
                _UnitDraft(
                    raw_text=text,
                    normalized_text=normalized,
                    source_path=f"$[{index}].text",
                    segment_id=segment_id,
                    timestamp=segment.timestamp,
                    speaker=segment.speaker.value,
                    raw_end_offset=len(text),
                    metadata={"kind": "transcript_segment"},
                )
            )
        return drafts

    def _lab_drafts(
        self,
        results: Sequence[LabResultItem],
        document_id: str,
        limitations: list[CoverageLimitation],
        unit_budget: _EvidenceUnitBudget,
    ) -> list[_UnitDraft]:
        drafts: list[_UnitDraft] = []
        for index, result in enumerate(results):
            unit_budget.consume()
            raw_value = display_scalar(result.value)
            raw_unit = result.unit
            normalized_unit = normalize_unit(raw_unit) if raw_unit else None
            if raw_unit and normalized_unit is None:
                limitations.append(
                    CoverageLimitation(
                        code="UNSUPPORTED_UNIT",
                        document_id=document_id,
                        message="A laboratory unit was preserved but could not be normalized.",
                    )
                )
            line = f"{result.test}: {raw_value}{f' {raw_unit}' if raw_unit else ''}"
            if result.reference_range is not None:
                line += f"; reference range {result.reference_range}"
            if result.flag is not None:
                line += f"; flag {result.flag}"
            metadata = {
                "kind": "lab_result",
                "test": result.test,
                "normalized_test": _normalized_test(result.test),
                "raw_value": raw_value,
                "value": result.value,
                "raw_unit": raw_unit,
                "unit": normalized_unit or raw_unit,
                "reference_range": result.reference_range,
                "flag": result.flag,
                "timestamp": result.timestamp,
            }
            drafts.append(
                _UnitDraft(
                    raw_text=canonical_json(result),
                    normalized_text=line,
                    source_path=f"$[{index}]",
                    timestamp=result.timestamp,
                    metadata=metadata,
                )
            )
        return drafts

    def _medication_drafts(
        self,
        medications: Sequence[MedicationItem],
        document_id: str,
        limitations: list[CoverageLimitation],
        unit_budget: _EvidenceUnitBudget,
    ) -> list[_UnitDraft]:
        drafts: list[_UnitDraft] = []
        for index, medication in enumerate(medications):
            unit_budget.consume()
            normalized_name = normalize_medication(medication.name) or medication.name.casefold()
            raw_dose = display_scalar(medication.dose) if medication.dose is not None else None
            raw_unit = medication.unit
            quantity = (
                normalize_quantity(medication.dose, raw_unit)
                if medication.dose is not None and raw_unit is not None
                else None
            )
            if raw_unit and quantity is None:
                limitations.append(
                    CoverageLimitation(
                        code="UNSUPPORTED_UNIT",
                        document_id=document_id,
                        message="A medication unit was preserved but could not be normalized.",
                    )
                )
            normalized_frequency = (
                normalize_frequency(medication.frequency) if medication.frequency else None
            )
            if medication.frequency and normalized_frequency is None:
                limitations.append(
                    CoverageLimitation(
                        code="UNSUPPORTED_FREQUENCY",
                        document_id=document_id,
                        message=(
                            "A medication frequency was preserved but could not be normalized."
                        ),
                    )
                )
            normalized_route = _normalize_route(medication.route) if medication.route else None
            if medication.route and normalized_route is None:
                limitations.append(
                    CoverageLimitation(
                        code="UNSUPPORTED_ROUTE",
                        document_id=document_id,
                        message="A medication route was preserved but could not be normalized.",
                    )
                )
            parts = [medication.name]
            if raw_dose is not None:
                parts.append(raw_dose)
            if raw_unit is not None:
                parts.append(raw_unit)
            if medication.frequency is not None:
                parts.append(medication.frequency)
            if medication.route is not None:
                parts.append(medication.route)
            if medication.status is not None:
                parts.append(medication.status)
            if medication.instructions is not None:
                parts.append(medication.instructions)
            metadata = {
                "kind": "medication",
                "raw_name": medication.name,
                "normalized_name": normalized_name,
                "raw_dose": raw_dose,
                "raw_unit": raw_unit,
                "dose": quantity.value if quantity is not None else medication.dose,
                "unit": quantity.unit if quantity is not None else normalize_unit(raw_unit or ""),
                "raw_frequency": medication.frequency,
                "frequency": normalized_frequency,
                "raw_route": medication.route,
                "route": normalized_route,
                "raw_status": medication.status,
                "status": medication.status.strip().upper() if medication.status else None,
                "instructions": medication.instructions,
            }
            drafts.append(
                _UnitDraft(
                    raw_text=canonical_json(medication),
                    normalized_text=" ".join(parts),
                    source_path=f"$[{index}]",
                    metadata=metadata,
                )
            )
        return drafts

    def _structured_json_drafts(
        self,
        content: Mapping[str, Any],
        document_id: str,
        limitations: list[CoverageLimitation],
        unit_budget: _EvidenceUnitBudget,
    ) -> list[_UnitDraft]:
        drafts: list[_UnitDraft] = []
        array_items = 0
        limitations.append(
            CoverageLimitation(
                code="UNINTERPRETED_STRUCTURED_FIELDS",
                document_id=document_id,
                message=(
                    "Arbitrary structured fields were preserved, but only recognized "
                    "clinical expressions were assessed."
                ),
            )
        )

        def validate(value: Any, depth: int) -> None:
            nonlocal array_items
            if isinstance(value, Mapping):
                if depth > self.limits.max_structured_depth:
                    self._raise_too_large(
                        "Structured evidence exceeds the configured depth limit."
                    )
                for item in value.values():
                    if isinstance(item, (Mapping, list, tuple)):
                        validate(item, depth + 1)
                return
            if isinstance(value, (list, tuple)):
                if depth > self.limits.max_structured_depth:
                    self._raise_too_large(
                        "Structured evidence exceeds the configured depth limit."
                    )
                array_items += len(value)
                if array_items > self.limits.max_json_array_items:
                    self._raise_too_large(
                        "Structured evidence exceeds the configured array-item limit."
                    )
                for item in value:
                    if isinstance(item, (Mapping, list, tuple)):
                        validate(item, depth + 1)
                return

        def record_text(value: Mapping[str, Any]) -> str:
            fields = []
            for key in sorted(value):
                item = value[key]
                rendered = (
                    canonical_json(item)
                    if isinstance(item, (Mapping, list, tuple))
                    else display_scalar(item)
                )
                fields.append(f"{key}={rendered}")
            return "; ".join(fields) or canonical_json(value)

        def emit_scalar(value: Any, path: str) -> None:
            rendered = display_scalar(value)
            unit_budget.consume()
            drafts.append(
                _UnitDraft(
                    raw_text=canonical_json(value),
                    normalized_text=f"{path}: {rendered}",
                    source_path=path,
                    metadata={"kind": "structured_scalar"},
                )
            )

        def emit(value: Any, path: str, *, root: bool = False) -> None:
            if isinstance(value, Mapping):
                if root:
                    for key in sorted(value):
                        item = value[key]
                        child_path = _json_child_path(path, str(key))
                        if isinstance(item, Mapping):
                            unit_budget.consume()
                            drafts.append(
                                _UnitDraft(
                                    raw_text=canonical_json(item),
                                    normalized_text=f"{child_path}: {record_text(item)}",
                                    source_path=child_path,
                                    metadata={"kind": "structured_record"},
                                )
                            )
                        elif isinstance(item, (list, tuple)):
                            emit(item, child_path)
                        else:
                            emit_scalar(item, child_path)
                else:
                    unit_budget.consume()
                    drafts.append(
                        _UnitDraft(
                            raw_text=canonical_json(value),
                            normalized_text=f"{path}: {record_text(value)}",
                            source_path=path,
                            metadata={"kind": "structured_record"},
                        )
                    )
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    child_path = f"{path}[{index}]"
                    if isinstance(item, (Mapping, list, tuple)):
                        emit(item, child_path)
                    else:
                        emit_scalar(item, child_path)
                return
            emit_scalar(value, path)

        validate(content, 1)
        emit(content, "$", root=True)
        return drafts

    def _json_array_items(self, content: Mapping[str, Any]) -> int:
        """Validate nested metadata iteratively and return its total array cardinality."""

        array_items = 0
        pending: list[tuple[Any, int]] = [(content, 1)]
        while pending:
            value, depth = pending.pop()
            if depth > self.limits.max_structured_depth:
                self._raise_too_large("Structured evidence exceeds the configured depth limit.")
            if isinstance(value, Mapping):
                pending.extend((item, depth + 1) for item in value.values())
            elif isinstance(value, (list, tuple)):
                array_items += len(value)
                pending.extend((item, depth + 1) for item in value)
        return array_items

    def _rag_drafts(
        self,
        chunks: Sequence[RagChunk],
        unit_budget: _EvidenceUnitBudget,
    ) -> list[_UnitDraft]:
        drafts: list[_UnitDraft] = []
        metadata_array_items = 0
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            self._raise_invalid("RAG chunk identifiers must be unique.")
        for index, chunk in enumerate(chunks):
            unit_budget.consume()
            metadata_array_items += self._json_array_items(chunk.metadata)
            if metadata_array_items > self.limits.max_json_array_items:
                self._raise_too_large(
                    "RAG metadata exceeds the configured array-item limit."
                )
            normalized = normalize_text(chunk.text)
            metadata = {
                "kind": "rag_chunk",
                "chunk_id": chunk.chunk_id,
                "source_uri": chunk.source_uri,
                "source_document_id": chunk.source_document_id,
                "retrieval_score": chunk.retrieval_score,
                "retrieval_metadata": chunk.metadata,
            }
            drafts.append(
                _UnitDraft(
                    raw_text=chunk.text,
                    normalized_text=normalized,
                    source_path=f"$[{index}].text",
                    segment_id=chunk.chunk_id,
                    source_uri=chunk.source_uri,
                    raw_end_offset=len(chunk.text),
                    metadata=metadata,
                )
            )
        return drafts


__all__ = ["UniversalEvidenceIngestionService"]

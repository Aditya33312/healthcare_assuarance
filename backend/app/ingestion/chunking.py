"""Deterministic, boundary-aware chunking for canonical evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from backend.app.ingestion.canonical import CanonicalChunk, CanonicalEvidenceUnit

_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[ \t]+|\n+)|\n{2,}")
_NEGATION_CONTEXT_RE = re.compile(
    r"\b(?:does\s+not\s+have|did\s+not\s+have|has\s+no|"
    r"den(?:y|ies|ied)|without|negative\s+for|no)\s+"
    r"(?:[A-Za-z][\w'-]*\s+){0,4}[A-Za-z][\w'-]*",
    re.IGNORECASE,
)
_CLINICAL_TUPLE_RE = re.compile(
    r"\b[A-Za-z][\w'-]*\s+[+\-\N{MINUS SIGN}]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?:/|[-\N{EN DASH}\N{EM DASH}]|to)\s*"
    r"[+\-\N{MINUS SIGN}]?(?:\d+(?:\.\d+)?|\.\d+))?\s*"
    r"(?:mcg|μg|µg|ug|mg|g|kg|mL|ml|mmol/L|mg/dL|g/dL|%|units?|IU)\b",
    re.IGNORECASE,
)


class ChunkLimitError(ValueError):
    """Raised when safe deterministic chunking exceeds configured safeguards."""


def _chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\x1f{start}\x1f{end}\x1f{text}".encode()
    ).hexdigest()[:20]
    return f"chunk_{digest}"


def _safe_cut(text: str, start: int, target_end: int) -> tuple[int, bool]:
    """Return a cut and whether it is a complete sentence/line boundary."""

    window_start = max(start + 1, target_end - 1_024)
    candidates = [match.end() for match in _BOUNDARY_RE.finditer(text, window_start, target_end)]
    if candidates:
        return candidates[-1], True
    whitespace = text.rfind(" ", window_start, target_end)
    if whitespace >= window_start:
        return whitespace + 1, False
    return target_end, True


def _split_long_unit(
    document_id: str,
    unit: CanonicalEvidenceUnit,
    *,
    max_characters: int,
    overlap_characters: int,
    max_chunks: int,
) -> list[CanonicalChunk]:
    chunks: list[CanonicalChunk] = []
    start = unit.start_offset
    while start < unit.end_offset:
        if len(chunks) >= max_chunks:
            raise ChunkLimitError("canonical evidence exceeds the configured chunk-count limit")
        target_end = min(unit.end_offset, start + max_characters)
        semantic_boundary = True
        if target_end == unit.end_offset:
            end = unit.end_offset
        else:
            relative_end, semantic_boundary = _safe_cut(
                unit.normalized_text,
                start - unit.start_offset,
                target_end - unit.start_offset,
            )
            end = relative_end + unit.start_offset
        if end <= start:
            end = target_end
        relative_start = start - unit.start_offset
        relative_end = end - unit.start_offset
        text = unit.normalized_text[relative_start:relative_end]
        chunks.append(
            CanonicalChunk(
                chunk_id=_chunk_id(document_id, start, end, text),
                document_id=document_id,
                text=text,
                start_offset=start,
                end_offset=end,
                evidence_unit_ids=(unit.evidence_unit_id,),
            )
        )
        if end >= unit.end_offset:
            break
        effective_overlap = overlap_characters
        if not semantic_boundary:
            # A cut at ordinary whitespace can otherwise separate "no chest"
            # from "pain" or a medication/value/unit tuple. Preserve a bounded
            # lexical context even when the configured general overlap is zero.
            effective_overlap = max(
                effective_overlap,
                min(64, max_characters // 4),
            )
        start = max(start + 1, end - effective_overlap)

    # Add narrowly scoped guard chunks only when a regular cut divided a
    # safety-critical negation phrase or numeric/unit tuple. This avoids turning
    # a small configured overlap into one-character progress for ordinary prose.
    for pattern in (_NEGATION_CONTEXT_RE, _CLINICAL_TUPLE_RE):
        for match in pattern.finditer(unit.normalized_text):
            protected_start = unit.start_offset + match.start()
            protected_end = unit.start_offset + match.end()
            if protected_end - protected_start > max_characters:
                raise ChunkLimitError(
                    "a clinical context span exceeds the configured safe chunk size"
                )
            if any(
                chunk.start_offset <= protected_start
                and protected_end <= chunk.end_offset
                for chunk in chunks
            ):
                continue
            if len(chunks) >= max_chunks:
                raise ChunkLimitError(
                    "canonical evidence exceeds the configured chunk-count limit"
                )
            protected_text = unit.normalized_text[match.start() : match.end()]
            chunks.append(
                CanonicalChunk(
                    chunk_id=_chunk_id(
                        document_id,
                        protected_start,
                        protected_end,
                        protected_text,
                    ),
                    document_id=document_id,
                    text=protected_text,
                    start_offset=protected_start,
                    end_offset=protected_end,
                    evidence_unit_ids=(unit.evidence_unit_id,),
                )
            )
    chunks.sort(key=lambda chunk: (chunk.start_offset, chunk.end_offset, chunk.chunk_id))
    return chunks


def build_chunks(
    document_id: str,
    normalized_text: str,
    units: Sequence[CanonicalEvidenceUnit],
    *,
    max_characters: int,
    overlap_characters: int,
    max_chunks: int,
) -> tuple[CanonicalChunk, ...]:
    """Group whole units and only split a unit when it exceeds the hard size."""

    if max_characters < 1 or not 0 <= overlap_characters < max_characters:
        raise ValueError("invalid chunk size or overlap")
    chunks: list[CanonicalChunk] = []
    grouped: list[CanonicalEvidenceUnit] = []

    def flush() -> None:
        if not grouped:
            return
        start = grouped[0].start_offset
        end = grouped[-1].end_offset
        text = normalized_text[start:end]
        if len(chunks) >= max_chunks:
            raise ChunkLimitError("canonical evidence exceeds the configured chunk-count limit")
        chunks.append(
            CanonicalChunk(
                chunk_id=_chunk_id(document_id, start, end, text),
                document_id=document_id,
                text=text,
                start_offset=start,
                end_offset=end,
                evidence_unit_ids=tuple(unit.evidence_unit_id for unit in grouped),
            )
        )
        grouped.clear()

    for unit in units:
        if len(unit.normalized_text) > max_characters:
            flush()
            chunks.extend(
                _split_long_unit(
                    document_id,
                    unit,
                    max_characters=max_characters,
                    overlap_characters=overlap_characters,
                    max_chunks=max_chunks - len(chunks),
                )
            )
            continue
        candidate_start = grouped[0].start_offset if grouped else unit.start_offset
        if grouped and unit.end_offset - candidate_start > max_characters:
            flush()
        grouped.append(unit)
    flush()

    if len(chunks) > max_chunks:
        raise ChunkLimitError("canonical evidence exceeds the configured chunk-count limit")
    return tuple(chunks)

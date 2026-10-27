"""Chunk-aware Clinical NLP adapter for canonical evidence."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Protocol

from backend.app.clinical_nlp.limits import ClinicalNLPLimitError
from backend.app.domain.enums import ClaimType
from backend.app.domain.models import Claim, TextSpan
from backend.app.ingestion.canonical import CanonicalDocument, CanonicalEvidenceUnit


class ClinicalExtractor(Protocol):
    version: str
    last_ner_error: str | None

    @property
    def model_versions(self) -> Mapping[str, str | None]: ...

    def extract(self, text: str, document_id: str) -> Sequence[Claim]: ...


class ClinicalProcessingLimitError(RuntimeError):
    """Stable, PHI-free error raised when bounded extraction cannot continue."""

    code = "CLINICAL_PROCESSING_LIMIT_EXCEEDED"


def _claim_id(
    document_id: str,
    start: int,
    end: int,
    claim_type: ClaimType,
    concept: str,
) -> str:
    digest = hashlib.sha256(
        f"{document_id}\x1f{start}\x1f{end}\x1f{claim_type.value}\x1f{concept}".encode()
    ).hexdigest()[:16]
    return f"claim_{digest}"


def _json_attribute(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


class CanonicalEvidenceExtractor:
    """Extract bounded chunks and remap every claim to canonical document offsets."""

    version = "canonical-evidence-extractor@1.0.0"

    def __init__(
        self,
        base: ClinicalExtractor,
        documents: Sequence[CanonicalDocument],
        *,
        max_claims: int,
        processing_timeout_seconds: float,
    ) -> None:
        self._base = base
        self._documents = {document.document_id: document for document in documents}
        self._max_claims = max_claims
        self._deadline = time.monotonic() + processing_timeout_seconds
        self._claim_count = 0
        self.last_ner_error: str | None = None

    @property
    def model_versions(self) -> Mapping[str, str | None]:
        return self._base.model_versions

    @property
    def deadline(self) -> float:
        """Absolute monotonic deadline shared with alignment and comparison."""

        return self._deadline

    def _check_limits(self, additional_claims: int = 0) -> None:
        if time.monotonic() > self._deadline:
            raise ClinicalProcessingLimitError("clinical processing duration limit exceeded")
        if self._claim_count + additional_claims > self._max_claims:
            raise ClinicalProcessingLimitError("extracted claim-count limit exceeded")

    def extract(self, text: str, document_id: str) -> list[Claim]:
        document = self._documents.get(document_id)
        if document is None:
            claims = list(self._base.extract(text, document_id))
            self._check_limits(len(claims))
            self._claim_count += len(claims)
            self.last_ner_error = self._base.last_ner_error
            return claims
        if text != document.normalized_text:
            raise ValueError("canonical extractor received text that does not match its document")

        structured = self._structured_claims(document)
        self._check_limits(len(structured))
        extracted: list[Claim] = []
        for chunk in document.chunks:
            self._check_limits()
            remaining = (
                self._max_claims
                - self._claim_count
                - len(structured)
                - len(extracted)
            )
            bounded_extract = getattr(self._base, "extract_bounded", None)
            try:
                if callable(bounded_extract):
                    raw_claims = list(
                        bounded_extract(
                            chunk.text,
                            document_id,
                            max_claims=remaining,
                            deadline=self._deadline,
                        )
                    )
                else:
                    raw_claims = list(self._base.extract(chunk.text, document_id))
            except ClinicalNLPLimitError as exc:
                raise ClinicalProcessingLimitError(str(exc)) from exc
            self._check_limits(len(structured) + len(extracted) + len(raw_claims))
            self.last_ner_error = self._base.last_ner_error or self.last_ner_error
            old_to_new: dict[str, str] = {}
            for claim in raw_claims:
                global_start = chunk.start_offset + claim.source_span.start
                global_end = chunk.start_offset + claim.source_span.end
                old_to_new[claim.claim_id] = _claim_id(
                    document_id,
                    global_start,
                    global_end,
                    claim.type,
                    claim.concept,
                )
            for claim in raw_claims:
                global_start = chunk.start_offset + claim.source_span.start
                global_end = chunk.start_offset + claim.source_span.end
                attributes = dict(claim.attributes)
                parent_id = attributes.get("parent_claim_id")
                if isinstance(parent_id, str) and parent_id in old_to_new:
                    attributes["parent_claim_id"] = old_to_new[parent_id]
                provenance = document.provenance_for_span(global_start, global_end)
                if provenance is not None:
                    attributes["evidence_provenance"] = provenance.model_dump(mode="json")
                extracted.append(
                    claim.model_copy(
                        update={
                            "claim_id": old_to_new[claim.claim_id],
                            "attributes": attributes,
                            "source_span": TextSpan(
                                document_id=document_id,
                                text=document.normalized_text[global_start:global_end],
                                start=global_start,
                                end=global_end,
                            ),
                        }
                    )
                )

        # Structured claims take precedence over duplicate regex-derived claims.
        ordered = [*structured, *extracted]
        unique: dict[tuple[ClaimType, int, int, str], Claim] = {}
        for claim in ordered:
            key = (
                claim.type,
                claim.source_span.start,
                claim.source_span.end,
                claim.concept,
            )
            unique.setdefault(key, claim)
        claims = sorted(
            unique.values(),
            key=lambda item: (
                item.source_span.start,
                item.source_span.end,
                item.type.value,
                item.concept,
            ),
        )
        self._check_limits(len(claims))
        self._claim_count += len(claims)
        return claims

    def _structured_claims(self, document: CanonicalDocument) -> list[Claim]:
        claims: list[Claim] = []
        for unit in document.evidence_units:
            kind = unit.metadata.get("kind")
            if kind == "medication":
                claims.extend(self._medication_claims(document, unit))
            elif kind == "lab_result":
                claims.extend(self._lab_claims(document, unit))
        return claims

    @staticmethod
    def _span_for(
        document: CanonicalDocument,
        unit: CanonicalEvidenceUnit,
        needle: str,
        *,
        start_at: int = 0,
    ) -> TextSpan | None:
        local_start = unit.normalized_text.find(needle, start_at)
        if local_start < 0 or not needle:
            return None
        start = unit.start_offset + local_start
        end = start + len(needle)
        return TextSpan(
            document_id=document.document_id,
            text=document.normalized_text[start:end],
            start=start,
            end=end,
        )

    @staticmethod
    def _provenance_attributes(
        document: CanonicalDocument,
        span: TextSpan,
    ) -> dict[str, Any]:
        provenance = document.provenance_for_span(span.start, span.end)
        return (
            {"evidence_provenance": provenance.model_dump(mode="json")}
            if provenance is not None
            else {}
        )

    def _medication_claims(
        self,
        document: CanonicalDocument,
        unit: CanonicalEvidenceUnit,
    ) -> list[Claim]:
        metadata = unit.metadata
        raw_name = str(metadata["raw_name"])
        name_span = self._span_for(document, unit, raw_name)
        if name_span is None:
            return []
        concept = str(metadata["normalized_name"])
        parent_id = _claim_id(
            document.document_id,
            name_span.start,
            name_span.end,
            ClaimType.MEDICATION,
            concept,
        )
        attributes = self._provenance_attributes(document, name_span)
        for key in (
            "dose",
            "unit",
            "frequency",
            "route",
            "status",
            "instructions",
            "raw_dose",
            "raw_unit",
            "raw_frequency",
            "raw_route",
            "raw_status",
        ):
            value = metadata.get(key)
            if value is not None:
                attributes[key] = _json_attribute(value)
        claims = [
            Claim(
                claim_id=parent_id,
                concept=concept,
                type=ClaimType.MEDICATION,
                attributes=attributes,
                source_span=name_span,
                extractor="structured_ingestion",
                extractor_version=self.version,
            )
        ]
        dose_text = str(metadata.get("raw_dose", ""))
        raw_unit = str(metadata.get("raw_unit", ""))
        quantity_text = f"{dose_text} {raw_unit}".strip()
        quantity_span = self._span_for(document, unit, quantity_text)
        if quantity_span is not None and metadata.get("dose") is not None:
            child_attributes = {
                **self._provenance_attributes(document, quantity_span),
                "parent_claim_id": parent_id,
                "parent_concept": concept,
                "dose": _json_attribute(metadata["dose"]),
                "value": _json_attribute(metadata["dose"]),
                "normalized_value": _json_attribute(metadata["dose"]),
            }
            if metadata.get("unit") is not None:
                child_attributes["unit"] = metadata["unit"]
                child_attributes["normalized_unit"] = metadata["unit"]
            claims.append(
                Claim(
                    claim_id=_claim_id(
                        document.document_id,
                        quantity_span.start,
                        quantity_span.end,
                        ClaimType.DOSE,
                        quantity_text,
                    ),
                    concept=quantity_text,
                    type=ClaimType.DOSE,
                    attributes=child_attributes,
                    source_span=quantity_span,
                    extractor="structured_ingestion",
                    extractor_version=self.version,
                )
            )
        for raw_key, normalized_key, claim_type in (
            ("raw_frequency", "frequency", ClaimType.FREQUENCY),
            ("raw_route", "route", ClaimType.ROUTE),
        ):
            raw_value = metadata.get(raw_key)
            normalized_value = metadata.get(normalized_key)
            if raw_value is None or normalized_value is None:
                continue
            span = self._span_for(document, unit, str(raw_value))
            if span is None:
                continue
            claims.append(
                Claim(
                    claim_id=_claim_id(
                        document.document_id,
                        span.start,
                        span.end,
                        claim_type,
                        str(normalized_value),
                    ),
                    concept=str(normalized_value),
                    type=claim_type,
                    attributes={
                        **self._provenance_attributes(document, span),
                        "parent_claim_id": parent_id,
                        "parent_concept": concept,
                        normalized_key: normalized_value,
                        f"normalized_{normalized_key}": normalized_value,
                    },
                    source_span=span,
                    extractor="structured_ingestion",
                    extractor_version=self.version,
                )
            )
        return claims

    def _lab_claims(
        self,
        document: CanonicalDocument,
        unit: CanonicalEvidenceUnit,
    ) -> list[Claim]:
        metadata = unit.metadata
        raw_test = str(metadata["test"])
        test_span = self._span_for(document, unit, raw_test)
        if test_span is None:
            return []
        concept = str(metadata["normalized_test"])
        value = _json_attribute(metadata["value"])
        attributes: dict[str, Any] = {
            **self._provenance_attributes(document, test_span),
            "value": value,
            "normalized_value": value,
        }
        for key in ("unit", "reference_range", "flag", "timestamp"):
            if metadata.get(key) is not None:
                attributes[key] = metadata[key]
        if metadata.get("unit") is not None:
            attributes["normalized_unit"] = metadata["unit"]
        parent_id = _claim_id(
            document.document_id,
            test_span.start,
            test_span.end,
            ClaimType.LAB_TEST,
            concept,
        )
        parent = Claim(
            claim_id=parent_id,
            concept=concept,
            type=ClaimType.LAB_TEST,
            attributes=attributes,
            source_span=test_span,
            extractor="structured_ingestion",
            extractor_version=self.version,
        )
        raw_value = str(metadata["raw_value"])
        raw_unit = str(metadata.get("raw_unit") or "")
        value_text = f"{raw_value} {raw_unit}".strip()
        value_span = self._span_for(document, unit, value_text)
        if value_span is None:
            return [parent]
        value_attributes = {
            **attributes,
            **self._provenance_attributes(document, value_span),
            "parent_claim_id": parent_id,
            "parent_concept": concept,
        }
        child = Claim(
            claim_id=_claim_id(
                document.document_id,
                value_span.start,
                value_span.end,
                ClaimType.LAB_VALUE,
                value_text,
            ),
            concept=value_text,
            type=ClaimType.LAB_VALUE,
            attributes=value_attributes,
            source_span=value_span,
            extractor="structured_ingestion",
            extractor_version=self.version,
        )
        return [parent, child]


__all__ = ["CanonicalEvidenceExtractor", "ClinicalProcessingLimitError"]

"""Public clinical NLP service used by Gatekeeper evaluators."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Protocol

from backend.app.clinical_nlp.assertion import AssertionRuleEngine
from backend.app.clinical_nlp.claims import ClaimBuilder
from backend.app.clinical_nlp.entities import ExtractedEntity
from backend.app.clinical_nlp.limits import ClinicalNLPLimitError, ensure_deadline
from backend.app.clinical_nlp.ner import HuggingFaceClinicalNERAdapter, NEREntity
from backend.app.clinical_nlp.normalization.medication_normalizer import normalize_medication
from backend.app.clinical_nlp.normalization.terminology_normalizer import normalize_terminology
from backend.app.clinical_nlp.parsers import DeterministicEntityParser
from backend.app.clinical_nlp.preprocessing import OffsetSafePreprocessor, PreprocessedText
from backend.app.clinical_nlp.relations import RelationLinker
from backend.app.clinical_nlp.versions import (
    CLINICAL_NLP_VERSION,
    DETERMINISTIC_EXTRACTOR_VERSION,
    HF_NER_MODEL_VERSION,
    component_versions,
)
from backend.app.domain.enums import AssertionStatus, ClaimType, SubjectType
from backend.app.domain.models import Claim


class NERAdapter(Protocol):
    version: str
    last_error: str | None

    def extract(self, text: str) -> Sequence[NEREntity | Mapping[str, Any]]: ...


_HF_LABEL_MAP: dict[str, ClaimType] = {
    "disease_disorder": ClaimType.CONDITION,
    "disease": ClaimType.CONDITION,
    "disorder": ClaimType.CONDITION,
    "sign_symptom": ClaimType.SYMPTOM,
    "sign_or_symptom": ClaimType.SYMPTOM,
    "symptom": ClaimType.SYMPTOM,
    "medication": ClaimType.MEDICATION,
    "drug": ClaimType.MEDICATION,
    "diagnostic_procedure": ClaimType.PROCEDURE,
    "therapeutic_procedure": ClaimType.PROCEDURE,
    "procedure": ClaimType.PROCEDURE,
    "biological_structure": ClaimType.BODY_SITE,
    "body_site": ClaimType.BODY_SITE,
}
_SEMANTIC_ANCHORS = {
    ClaimType.ALLERGY,
    ClaimType.CONDITION,
    ClaimType.MEDICATION,
    ClaimType.PROCEDURE,
    ClaimType.SYMPTOM,
    ClaimType.CLINICAL_FACT,
}


def _canonical_hf_label(label: str) -> str:
    value = label.strip()
    value = re.sub(r"^(?:B|I|E|S)-", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return value


def _overlap_ratio(first: ExtractedEntity, second: ExtractedEntity) -> float:
    overlap = max(0, min(first.end, second.end) - max(first.start, second.start))
    denominator = min(first.end - first.start, second.end - second.start)
    return overlap / denominator if denominator else 0.0


class ClinicalNLPService:
    """Extract versioned clinical claims from exact source text.

    The default constructor is fully deterministic and offline. Set ``enable_hf=True`` to
    opt in to the revision-pinned candidate NER adapter. Even then, downloads remain disabled
    unless ``allow_model_download=True`` is also explicit.
    """

    version = CLINICAL_NLP_VERSION
    deterministic_extractor_version = DETERMINISTIC_EXTRACTOR_VERSION
    optional_ner_model_version = HF_NER_MODEL_VERSION

    def __init__(
        self,
        *,
        enable_hf: bool = False,
        allow_model_download: bool = False,
        ner_adapter: NERAdapter | None = None,
        reference_date: date | datetime | str | None = None,
        day_first_dates: bool = False,
    ) -> None:
        self._preprocessor = OffsetSafePreprocessor()
        self._parser = DeterministicEntityParser(
            reference_date=reference_date, day_first_dates=day_first_dates
        )
        self._assertion = AssertionRuleEngine()
        self._relations = RelationLinker()
        self._builder = ClaimBuilder()
        self._hf_enabled = enable_hf or ner_adapter is not None
        self._ner_adapter: NERAdapter | None = ner_adapter
        if enable_hf and ner_adapter is None:
            self._ner_adapter = HuggingFaceClinicalNERAdapter(allow_download=allow_model_download)
        self.last_ner_error: str | None = None

    @property
    def versions(self) -> dict[str, str | None]:
        versions = component_versions(hf_enabled=self._hf_enabled)
        if self._ner_adapter is not None:
            versions["ner_model"] = getattr(self._ner_adapter, "version", HF_NER_MODEL_VERSION)
        return versions

    @property
    def model_versions(self) -> dict[str, str | None]:
        """Compatibility alias for provenance consumers."""

        return self.versions

    @property
    def hf_enabled(self) -> bool:
        return self._hf_enabled

    def extract(self, text: str, document_id: str) -> list[Claim]:
        """Return deterministic, source-grounded claims in source-offset order."""

        return self._extract(text, document_id, max_claims=None, deadline=None)

    def extract_bounded(
        self,
        text: str,
        document_id: str,
        *,
        max_claims: int,
        deadline: float,
    ) -> list[Claim]:
        """Extract cooperatively within an absolute deadline and claim budget."""

        if max_claims < 1:
            raise ClinicalNLPLimitError("clinical claim-count limit exceeded")
        return self._extract(
            text,
            document_id,
            max_claims=max_claims,
            deadline=deadline,
        )

    def _extract(
        self,
        text: str,
        document_id: str,
        *,
        max_claims: int | None,
        deadline: float | None,
    ) -> list[Claim]:
        """Shared extraction implementation used by bounded ingestion."""

        # The service may be reused for several documents in one evaluation.  Never
        # let an optional-adapter failure from a previous document leak into the
        # current extraction result.
        self.last_ner_error = None
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(document_id, str):
            raise TypeError("document_id must be a string")
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not text.strip():
            return []

        ensure_deadline(deadline)
        document = self._preprocessor.process(text)
        ensure_deadline(deadline)
        entities = self._parser.parse(
            text,
            max_entities=max_claims,
            deadline=deadline,
        )
        remaining = None if max_claims is None else max_claims - len(entities)
        entities.extend(
            self._extract_hf_entities(
                text,
                max_entities=remaining,
                deadline=deadline,
            )
        )
        entities = self._deduplicate(entities, deadline=deadline)
        self._resolve_assertions(entities, document, deadline=deadline)
        self._relations.link(entities, document, deadline=deadline)
        self._inherit_parent_context(entities, deadline=deadline)
        return self._builder.build(
            entities,
            text=text,
            document_id=document_id,
            max_claims=max_claims,
            deadline=deadline,
        )

    def _extract_hf_entities(
        self,
        text: str,
        *,
        max_entities: int | None = None,
        deadline: float | None = None,
    ) -> list[ExtractedEntity]:
        if not self._hf_enabled or self._ner_adapter is None:
            return []
        try:
            raw_entities = self._ner_adapter.extract(text)
        except Exception as exc:  # A third-party/custom adapter must also fail safe.
            self.last_ner_error = f"NER_ADAPTER_FAILED:{type(exc).__name__}"
            return []
        adapter_error = getattr(self._ner_adapter, "last_error", None)
        self.last_ner_error = "NER_ADAPTER_FAILED" if adapter_error else None

        entities: list[ExtractedEntity] = []
        for raw in raw_entities:
            ensure_deadline(deadline)
            if max_entities is not None and len(entities) >= max_entities:
                raise ClinicalNLPLimitError("clinical entity-count limit exceeded")
            parsed = self._parse_ner_result(raw, text)
            if parsed is not None:
                entities.append(parsed)
        return entities

    def _parse_ner_result(
        self, raw: NEREntity | Mapping[str, Any], text: str
    ) -> ExtractedEntity | None:
        try:
            if isinstance(raw, Mapping):
                start = int(raw["start"])
                end = int(raw["end"])
                label = str(raw.get("label") or raw.get("entity_group") or raw.get("entity") or "")
                score = float(raw.get("score", 0.0))
            else:
                start = raw.start
                end = raw.end
                label = raw.label
                score = raw.score
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if not 0 <= start < end <= len(text):
            return None
        claim_type = _HF_LABEL_MAP.get(_canonical_hf_label(label))
        if claim_type is None:
            return None

        span_text = text[start:end]
        if claim_type == ClaimType.MEDICATION:
            concept = normalize_medication(span_text)
        else:
            concept = normalize_terminology(span_text)
        if concept is None:
            return None
        return ExtractedEntity(
            type=claim_type,
            text=span_text,
            start=start,
            end=end,
            concept=concept,
            confidence=max(0.0, min(1.0, score)),
            origin="pretrained_hf_ner",
            origin_version=getattr(self._ner_adapter, "version", HF_NER_MODEL_VERSION),
        )

    def _deduplicate(
        self,
        entities: list[ExtractedEntity],
        *,
        deadline: float | None = None,
    ) -> list[ExtractedEntity]:
        result: list[ExtractedEntity] = []
        seen: set[tuple[ClaimType, int, int, str]] = set()
        for entity in entities:
            ensure_deadline(deadline)
            key = (entity.type, entity.start, entity.end, entity.concept)
            if key in seen:
                continue
            if entity.origin == "pretrained_hf_ner" and any(
                existing.type in _SEMANTIC_ANCHORS
                and entity.type in _SEMANTIC_ANCHORS
                and _overlap_ratio(entity, existing) >= 0.5
                for existing in result
            ):
                continue
            seen.add(key)
            result.append(entity)
        return result

    def _resolve_assertions(
        self,
        entities: list[ExtractedEntity],
        document: PreprocessedText,
        *,
        deadline: float | None = None,
    ) -> None:
        for entity in entities:
            ensure_deadline(deadline)
            forced_assertion = entity.attributes.get("_forced_assertion")
            forced_subject = entity.attributes.get("_forced_subject")
            assertion_value = (
                forced_assertion if isinstance(forced_assertion, AssertionStatus) else None
            )
            subject_value = forced_subject if isinstance(forced_subject, SubjectType) else None
            result = self._assertion.resolve(
                entity_start=entity.start,
                entity_end=entity.end,
                sentence=document.sentence_at(entity.start),
                forced_assertion=assertion_value,
                forced_subject=subject_value,
            )
            entity.assertion = result.assertion
            entity.subject = result.subject
            entity.attributes["assertion_rule"] = result.rule

    def _inherit_parent_context(
        self,
        entities: list[ExtractedEntity],
        *,
        deadline: float | None = None,
    ) -> None:
        for entity in entities:
            ensure_deadline(deadline)
            if entity.parent_index is None or not 0 <= entity.parent_index < len(entities):
                continue
            parent = entities[entity.parent_index]
            entity.assertion = parent.assertion
            entity.subject = parent.subject

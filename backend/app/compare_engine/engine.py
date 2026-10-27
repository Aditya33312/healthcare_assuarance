"""Two-way, deterministic-first Clinical Compare Engine."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from importlib import import_module
from typing import Protocol, cast

from backend.app.clinical_nlp.limits import ensure_deadline
from backend.app.compare_engine.alignment import Alignment, ClaimAligner
from backend.app.compare_engine.inference import (
    BoundedClinicalInferenceEvaluator,
    InferenceDecision,
)
from backend.app.compare_engine.rules import (
    COMPARE_RULE_SET_VERSION,
    RuleOutcome,
    compare_aligned_claims,
)
from backend.app.compare_engine.semantic import (
    ClinicalSemanticCascade,
    EmbeddingMatcher,
    SemanticLabel,
    SemanticMatcher,
)
from backend.app.domain.enums import (
    AlignmentMethod,
    ClaimType,
    CoverageStatus,
    ExecutionStatus,
    FindingCategory,
    FindingType,
    Severity,
)
from backend.app.domain.models import (
    Claim,
    CompareResult,
    ComponentVersions,
    CoverageSummary,
    FaithfulnessMetrics,
    Finding,
    GeneratedOutput,
    SourceBundle,
    SourceDocument,
    TextSpan,
)
from backend.app.domain.taxonomy import canonical_category, category_for_claim_type
from backend.app.severity import (
    ProvisionalSeverityPolicy,
    SeverityContext,
)
from backend.app.version import ENGINE_VERSION

type SourceInput = str | SourceBundle
type GeneratedInput = str | GeneratedOutput

_LINKED_ATTRIBUTE_KEYS: dict[ClaimType, tuple[str, ...]] = {
    ClaimType.BODY_SITE: ("body_site",),
    ClaimType.LATERALITY: ("laterality", "normalized_laterality"),
    ClaimType.DOSE: ("dose", "normalized_dose"),
    ClaimType.UNIT: ("unit", "normalized_unit"),
    ClaimType.FREQUENCY: ("frequency", "normalized_frequency"),
    ClaimType.ROUTE: ("route", "normalized_route"),
    ClaimType.LAB_VALUE: ("value", "normalized_value", "lab_value"),
    ClaimType.DATE: ("date", "normalized_date"),
    ClaimType.DURATION: ("duration", "normalized_duration"),
}

_RULE_EVIDENCE_TYPE: dict[str, ClaimType] = {
    "LATERALITY_MISMATCH": ClaimType.LATERALITY,
    "DOSE_MISMATCH": ClaimType.DOSE,
    "UNIT_MISMATCH": ClaimType.UNIT,
    "FREQUENCY_MISMATCH": ClaimType.FREQUENCY,
    "ROUTE_MISMATCH": ClaimType.ROUTE,
    "NUMERIC_VALUE_MISMATCH": ClaimType.LAB_VALUE,
    "LAB_UNIT_MISMATCH": ClaimType.LAB_VALUE,
    "DATE_MISMATCH": ClaimType.DATE,
    "DURATION_MISMATCH": ClaimType.DURATION,
}

_NARRATIVE_CLAIM_TYPES = {
    ClaimType.CONDITION,
    ClaimType.SYMPTOM,
    ClaimType.CLINICAL_FACT,
}
_INFERENCE_CUE = re.compile(
    r"\b(?:consistent\s+with|suggestive\s+of|compatible\s+with|"
    r"suggests?|suspected|possible|likely|may\s+(?:be|represent|relate)|"
    r"could\s+(?:be|represent))\b",
    flags=re.IGNORECASE,
)
_IMPORTANT_OMISSION_MEDICATIONS = {
    "apixaban",
    "digoxin",
    "insulin",
    "rivaroxaban",
    "warfarin",
}
_IMPORTANT_OMISSION_QUALIFIERS = {
    "CRUSHING",
    "LIFE_THREATENING",
    "SERIOUS",
    "SEVERE",
}


class ClinicalExtractor(Protocol):
    """Public Clinical NLP dependency used by the compare engine."""

    def extract(self, text: str, document_id: str) -> list[Claim]:
        """Extract structured claims with exact spans."""


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _coerce_inputs(
    source_bundle: SourceInput | None,
    generated_output: GeneratedInput | None,
    source: SourceInput | None,
) -> tuple[SourceBundle, GeneratedOutput]:
    if source is not None:
        if source_bundle is not None:
            raise ValueError("Provide either source or source_bundle, not both")
        source_bundle = source
    if source_bundle is None:
        raise ValueError("source is required")
    if generated_output is None:
        raise ValueError("generated_output is required")

    if isinstance(source_bundle, str):
        structured_source = SourceBundle(
            documents=[SourceDocument(document_id="source_001", content=source_bundle)]
        )
    elif isinstance(source_bundle, SourceBundle):
        structured_source = source_bundle
    else:
        raise TypeError("source must be a string or SourceBundle")

    if isinstance(generated_output, str):
        structured_generated = GeneratedOutput(content=generated_output)
    elif isinstance(generated_output, GeneratedOutput):
        structured_generated = generated_output
    else:
        raise TypeError("generated_output must be a string or GeneratedOutput")

    document_ids = [document.document_id for document in structured_source.documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("SourceBundle document_id values must be unique")
    return structured_source, structured_generated


def _evaluation_id(source: SourceBundle, generated: GeneratedOutput) -> str:
    parts: list[object] = [ENGINE_VERSION]
    for document in source.documents:
        parts.extend((document.document_id, document.type, document.content))
    parts.extend((generated.type, generated.content))
    return _stable_id("eval", *parts)


def _validate_claim_span(claim: Claim, content: str, document_id: str) -> Claim:
    span = claim.source_span
    if span.end > len(content) or content[span.start : span.end] != span.text:
        raise ValueError(
            f"Extractor returned a non-exact span for claim {claim.claim_id!r} "
            f"in document {document_id!r}"
        )
    if span.document_id not in {None, document_id}:
        raise ValueError(
            f"Extractor returned document_id {span.document_id!r} for document {document_id!r}"
        )
    if span.document_id == document_id:
        return claim
    exact_span = TextSpan(
        document_id=document_id,
        text=span.text,
        start=span.start,
        end=span.end,
    )
    return claim.model_copy(update={"source_span": exact_span})


def _extract(
    extractor: ClinicalExtractor,
    text: str,
    document_id: str,
) -> list[Claim]:
    raw_claims = extractor.extract(text, document_id)
    if not isinstance(raw_claims, list):
        raise TypeError("ClinicalNLPService.extract must return list[Claim]")
    claims: list[Claim] = []
    for raw_claim in raw_claims:
        claim = raw_claim if isinstance(raw_claim, Claim) else Claim.model_validate(raw_claim)
        claim = _validate_claim_span(claim, text, document_id)
        span = claim.source_span
        sentence_start = max(
            text.rfind(".", 0, span.start),
            text.rfind("\n", 0, span.start),
        )
        sentence_end_candidates = [
            boundary
            for boundary in (text.find(".", span.end), text.find("\n", span.end))
            if boundary >= 0
        ]
        sentence_end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(text)
        semantic_context = text[sentence_start + 1 : sentence_end].strip()
        claim = claim.model_copy(
            update={
                "attributes": {
                    **claim.attributes,
                    "semantic_context": semantic_context,
                }
            }
        )
        claims.append(claim)
    return claims


def _default_extractor() -> ClinicalExtractor:
    # Lazy import keeps the compare package importable in minimal deployments;
    # actual evaluation still fails visibly if the clinical NLP service is absent.
    service_module = import_module("backend.app.clinical_nlp.service")
    service_type = service_module.ClinicalNLPService
    return cast("ClinicalExtractor", service_type())


def _optional_ner_failed(extractor: ClinicalExtractor) -> bool:
    """Read only the stable adapter status code, never an adapter exception."""

    value = getattr(extractor, "last_ner_error", None)
    return isinstance(value, str) and bool(value.strip())


def _has_inference_cue(content: str, claim: Claim) -> bool:
    """Detect explicit inference language in the sentence containing ``claim``."""

    span = claim.source_span
    sentence_start = max(content.rfind(".", 0, span.start), content.rfind("\n", 0, span.start))
    sentence_end_candidates = [
        boundary
        for boundary in (content.find(".", span.end), content.find("\n", span.end))
        if boundary >= 0
    ]
    sentence_end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(content)
    return _INFERENCE_CUE.search(content[sentence_start + 1 : sentence_end]) is not None


def _category_for_claim(claim: Claim) -> FindingCategory:
    return category_for_claim_type(claim.type)


def _omission_classification(claim: Claim) -> tuple[FindingCategory, str]:
    """Classify only explicit high-risk omissions as important omissions."""

    qualifier = str(
        claim.attributes.get("clinical_qualifier")
        or claim.attributes.get("severity_qualifier")
        or ""
    ).upper()
    important = qualifier in _IMPORTANT_OMISSION_QUALIFIERS or (
        claim.type is ClaimType.MEDICATION
        and claim.concept.casefold() in _IMPORTANT_OMISSION_MEDICATIONS
    )
    if important:
        return _category_for_claim(claim), "IMPORTANT_OMISSION"
    return _category_for_claim(claim), "SOURCE_CLAIM_OMITTED"


def _supported_category(source: Claim, generated: Claim) -> FindingCategory:
    """Name a preserved high-risk dimension only when both claims expose it."""

    if source.subject is generated.subject and source.subject.value != "PATIENT":
        return FindingCategory.SUBJECT_ATTRIBUTION
    if source.assertion is generated.assertion:
        if source.assertion.value == "ABSENT":
            return FindingCategory.ASSERTION
        if source.assertion.value in {
            "POSSIBLE",
            "HISTORICAL",
            "HYPOTHETICAL",
            "CONDITIONAL",
            "UNCERTAIN",
        }:
            return FindingCategory.CERTAINTY
    if source.type is ClaimType.MEDICATION and generated.type is ClaimType.MEDICATION:
        for attribute, category in (
            ("dose", FindingCategory.MEDICATION_DOSE),
            ("frequency", FindingCategory.MEDICATION_FREQUENCY),
            ("route", FindingCategory.MEDICATION_ROUTE),
        ):
            if (
                source.attributes.get(attribute) is not None
                and generated.attributes.get(attribute) is not None
            ):
                return category
    if source.type in {ClaimType.LAB_TEST, ClaimType.LAB_VALUE} and generated.type in {
        ClaimType.LAB_TEST,
        ClaimType.LAB_VALUE,
    }:
        source_value = source.attributes.get("normalized_value", source.attributes.get("value"))
        generated_value = generated.attributes.get(
            "normalized_value", generated.attributes.get("value")
        )
        if source_value is not None and generated_value is not None:
            return FindingCategory.LAB_VALUE
    source_laterality = source.attributes.get("laterality")
    generated_laterality = generated.attributes.get("laterality")
    if source_laterality is not None and generated_laterality is not None:
        return FindingCategory.LATERALITY
    if (
        source.attributes.get("duration") is not None
        and generated.attributes.get("duration") is not None
    ):
        return FindingCategory.DURATION
    return _category_for_claim(generated)


def _is_explicit_inference(claim: Claim) -> bool:
    value = claim.attributes.get("inferred", False)
    return value is True or (isinstance(value, str) and value.casefold() == "true")


def _comparison_claims(claims: list[Claim]) -> list[Claim]:
    """Collapse linked modifier claims already represented on their parent.

    The raw claims remain in ``CompareResult`` for auditability.  Comparing both
    a medication and its linked DOSE/UNIT children would otherwise report the
    same clinical discrepancy multiple times and distort metrics.
    """

    by_id = {claim.claim_id: claim for claim in claims}
    comparison_claims: list[Claim] = []
    for claim in claims:
        parent_id = claim.attributes.get("parent_claim_id")
        parent = by_id.get(str(parent_id)) if parent_id is not None else None
        represented_keys = _LINKED_ATTRIBUTE_KEYS.get(claim.type, ())
        represented_on_parent = parent is not None and any(
            key in parent.attributes and parent.attributes[key] is not None
            for key in represented_keys
        )
        if not represented_on_parent:
            comparison_claims.append(claim)
    return comparison_claims


def _evidence_span(
    claim: Claim,
    all_claims: list[Claim],
    rule_id: str,
) -> TextSpan:
    evidence_type = _RULE_EVIDENCE_TYPE.get(rule_id)
    if evidence_type is None or claim.type is evidence_type:
        return claim.source_span
    for candidate in all_claims:
        if (
            candidate.type is evidence_type
            and candidate.attributes.get("parent_claim_id") == claim.claim_id
        ):
            return candidate.source_span
    return claim.source_span


def _reason_for_rule_outcome(outcome: RuleOutcome) -> str:
    if outcome.finding_type is FindingType.ALTERED:
        return outcome.rule.rule_id
    base = outcome.rule.rule_id.removesuffix("_MISMATCH")
    return f"{base}_{outcome.finding_type.value}"


class CompareEngine:
    """Shared evaluator used by API, web, batch, and CI workflows."""

    def __init__(
        self,
        nlp_service: ClinicalExtractor | None = None,
        *,
        semantic_matcher: SemanticMatcher | None = None,
        embedding_matcher: EmbeddingMatcher | None = None,
        severity_policy: ProvisionalSeverityPolicy | None = None,
        inference_evaluator: BoundedClinicalInferenceEvaluator | None = None,
        enable_semantic: bool = True,
    ) -> None:
        self._nlp_service = nlp_service
        self._semantic_matcher = (
            semantic_matcher
            if semantic_matcher is not None
            else ClinicalSemanticCascade()
            if enable_semantic
            else None
        )
        self._embedding_matcher = embedding_matcher
        self._inference_evaluator = (
            inference_evaluator
            if inference_evaluator is not None
            else BoundedClinicalInferenceEvaluator()
            if enable_semantic
            else None
        )
        self._aligner = ClaimAligner(
            semantic_matcher=self._semantic_matcher,
            embedding_matcher=embedding_matcher,
        )
        self._severity_policy = severity_policy or ProvisionalSeverityPolicy()

    def _finding(
        self,
        *,
        evaluation_id: str,
        ordinal: int,
        finding_type: FindingType,
        category: FindingCategory,
        reason_code: str,
        explanation: str,
        confidence: float,
        claim_type: ClaimType | None,
        source: TextSpan | None = None,
        generated: TextSpan | None = None,
        source_claim: Claim | None = None,
        generated_claim: Claim | None = None,
        source_value: object | None = None,
        generated_value: object | None = None,
        alignment_method: AlignmentMethod = AlignmentMethod.NONE,
        rule_id: str | None = None,
        rule_version: str | None = None,
        source_support: list[TextSpan] | None = None,
        uncertainty_reason: str | None = None,
        candidate_evidence: list[TextSpan] | None = None,
    ) -> Finding:
        severity_context = SeverityContext(
            finding_type=finding_type,
            category=category,
            claim_type=claim_type,
            reason_code=reason_code,
            source_concept=(source_claim.concept if source_claim is not None else None),
            generated_concept=(generated_claim.concept if generated_claim is not None else None),
            source_attributes=(source_claim.attributes if source_claim is not None else {}),
            generated_attributes=(
                generated_claim.attributes if generated_claim is not None else {}
            ),
            source_value=source_value,
            generated_value=generated_value,
        )
        severity = self._severity_policy.severity_for(severity_context)
        severity_policy_version = (
            f"{self._severity_policy.policy_id}@{self._severity_policy.version}"
        )
        finding_id = _stable_id(
            "find",
            evaluation_id,
            ordinal,
            finding_type.value,
            category,
            reason_code,
            source.start if source else "",
            generated.start if generated else "",
        )
        return Finding(
            finding_id=finding_id,
            type=finding_type,
            category=category,
            severity=severity,
            source=source,
            generated=generated,
            reason_code=reason_code,
            explanation=explanation,
            confidence=max(0.0, min(1.0, confidence)),
            rule_id=rule_id,
            rule_version=rule_version,
            alignment_method=alignment_method,
            severity_policy_version=severity_policy_version,
            source_support=source_support or [],
            uncertainty_reason=uncertainty_reason,
            candidate_evidence=candidate_evidence or [],
        )

    def _aligned_findings(
        self,
        evaluation_id: str,
        ordinal: int,
        alignment: Alignment,
        source_claim: Claim,
        generated_claim: Claim,
        all_source_claims: list[Claim],
        all_generated_claims: list[Claim],
    ) -> list[Finding]:
        outcomes = compare_aligned_claims(source_claim, generated_claim)
        confidence = min(
            source_claim.confidence,
            generated_claim.confidence,
            alignment.confidence,
        )
        if outcomes:
            return [
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal + offset,
                    finding_type=outcome.finding_type,
                    category=canonical_category(outcome.category or outcome.rule.category),
                    reason_code=_reason_for_rule_outcome(outcome),
                    explanation=f"{outcome.explanation} Alignment: {alignment.explanation}",
                    confidence=min(confidence, outcome.confidence),
                    claim_type=source_claim.type,
                    source=_evidence_span(
                        source_claim,
                        all_source_claims,
                        outcome.rule.rule_id,
                    ),
                    generated=_evidence_span(
                        generated_claim,
                        all_generated_claims,
                        outcome.rule.rule_id,
                    ),
                    source_claim=source_claim,
                    generated_claim=generated_claim,
                    source_value=outcome.source_value,
                    generated_value=outcome.generated_value,
                    alignment_method=alignment.method,
                    rule_id=outcome.rule.rule_id,
                    rule_version=outcome.rule.version,
                )
                for offset, outcome in enumerate(outcomes)
            ]

        if alignment.semantic_contradiction:
            return [
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.ALTERED,
                    category=FindingCategory.CLINICAL_FACT,
                    reason_code="SEMANTIC_CONTRADICTION",
                    explanation=alignment.explanation,
                    confidence=confidence,
                    claim_type=source_claim.type,
                    source=source_claim.source_span,
                    generated=generated_claim.source_span,
                    source_claim=source_claim,
                    generated_claim=generated_claim,
                    alignment_method=alignment.method,
                    rule_id="SEMANTIC_CONTRADICTION",
                    rule_version=getattr(self._semantic_matcher, "version", None),
                )
            ]

        if (
            self._semantic_matcher is not None
            and source_claim.type in _NARRATIVE_CLAIM_TYPES
            and generated_claim.type in _NARRATIVE_CLAIM_TYPES
            and alignment.method is not AlignmentMethod.SEMANTIC
        ):
            try:
                semantic_decision = self._semantic_matcher.compare(source_claim, generated_claim)
            except Exception:
                semantic_decision = None
            if (
                semantic_decision is not None
                and semantic_decision.label is SemanticLabel.CONTRADICTORY
            ):
                return [
                    self._finding(
                        evaluation_id=evaluation_id,
                        ordinal=ordinal,
                        finding_type=FindingType.ALTERED,
                        category=FindingCategory.CLINICAL_FACT,
                        reason_code="SEMANTIC_CONTRADICTION",
                        explanation=semantic_decision.explanation,
                        confidence=min(confidence, semantic_decision.confidence),
                        claim_type=source_claim.type,
                        source=source_claim.source_span,
                        generated=generated_claim.source_span,
                        source_claim=source_claim,
                        generated_claim=generated_claim,
                        alignment_method=AlignmentMethod.SEMANTIC,
                        rule_id="SEMANTIC_CONTRADICTION",
                        rule_version=getattr(self._semantic_matcher, "version", None),
                    )
                ]
            if (
                semantic_decision is not None
                and semantic_decision.label is SemanticLabel.UNCERTAIN
                and semantic_decision.reason_code == "EXPLICIT_SOURCE_AMBIGUITY"
            ):
                return [
                    self._finding(
                        evaluation_id=evaluation_id,
                        ordinal=ordinal,
                        finding_type=FindingType.UNCERTAIN,
                        category=FindingCategory.ALIGNMENT,
                        reason_code="EXPLICIT_SOURCE_AMBIGUITY",
                        explanation=semantic_decision.explanation,
                        confidence=min(confidence, semantic_decision.confidence),
                        claim_type=source_claim.type,
                        source=source_claim.source_span,
                        generated=generated_claim.source_span,
                        source_claim=source_claim,
                        generated_claim=generated_claim,
                        alignment_method=AlignmentMethod.SEMANTIC,
                        uncertainty_reason="EXPLICIT_SOURCE_AMBIGUITY",
                        candidate_evidence=[source_claim.source_span],
                    )
                ]

        inferred = alignment.inferred or _is_explicit_inference(generated_claim)
        finding_type = FindingType.INFERRED if inferred else FindingType.SUPPORTED
        reason_code = "CLAIM_INFERRED" if inferred else "CLAIM_SUPPORTED"
        explanation = (
            "The generated claim is a semantically supported inference. "
            if inferred
            else "The generated claim is supported by the aligned source claim. "
        )
        category = _supported_category(source_claim, generated_claim)
        evidence_rule = {
            FindingCategory.MEDICATION_DOSE: "DOSE_MISMATCH",
            FindingCategory.MEDICATION_FREQUENCY: "FREQUENCY_MISMATCH",
            FindingCategory.MEDICATION_ROUTE: "ROUTE_MISMATCH",
        }.get(category)
        return [
            self._finding(
                evaluation_id=evaluation_id,
                ordinal=ordinal,
                finding_type=finding_type,
                category=category,
                reason_code=reason_code,
                explanation=explanation + f"Alignment: {alignment.explanation}",
                confidence=confidence,
                claim_type=source_claim.type,
                source=(
                    _evidence_span(source_claim, all_source_claims, evidence_rule)
                    if evidence_rule is not None
                    else source_claim.source_span
                ),
                generated=(
                    _evidence_span(generated_claim, all_generated_claims, evidence_rule)
                    if evidence_rule is not None
                    else generated_claim.source_span
                ),
                source_claim=source_claim,
                generated_claim=generated_claim,
                alignment_method=alignment.method,
            )
        ]

    def _metrics(self, findings: list[Finding]) -> FaithfulnessMetrics:
        counts = Counter(finding.type for finding in findings)
        total = sum(counts.values())
        supported = counts[FindingType.SUPPORTED]
        score = round(100.0 * supported / total, 2) if total else 0.0
        return FaithfulnessMetrics(
            score=score,
            supported=supported,
            altered=counts[FindingType.ALTERED],
            added=counts[FindingType.ADDED],
            omitted=counts[FindingType.OMITTED],
            inferred=counts[FindingType.INFERRED],
            uncertain=counts[FindingType.UNCERTAIN],
            critical_count=sum(finding.severity is Severity.CRITICAL for finding in findings),
            high_count=sum(finding.severity is Severity.HIGH for finding in findings),
        )

    def _versions(self, extractor: ClinicalExtractor | None = None) -> ComponentVersions:
        evaluator_versions = [
            str(version)
            for evaluator in (
                self._embedding_matcher,
                self._semantic_matcher,
                self._inference_evaluator,
            )
            if evaluator is not None
            and (version := getattr(evaluator, "version", None)) is not None
        ]
        semantic_version = ";".join(evaluator_versions) or None
        extractor_versions = getattr(extractor, "model_versions", {})
        if not isinstance(extractor_versions, Mapping):
            extractor_versions = {}
        return ComponentVersions(
            ner_model=str(
                extractor_versions.get("ner_model", "deterministic-clinical-extractor@0.1.0")
            ),
            assertion_model=str(extractor_versions.get("assertion_model", "assertion-rules@0.1.0")),
            normalization=str(
                extractor_versions.get("normalization", "clinical-normalizers@0.1.0")
            ),
            semantic_evaluator=semantic_version,
            rule_set=f"clinical-compare-rules@{COMPARE_RULE_SET_VERSION}",
            severity_policy=(f"{self._severity_policy.policy_id}@{self._severity_policy.version}"),
        )

    def compare(
        self,
        source_bundle: SourceInput | None = None,
        generated_output: GeneratedInput | None = None,
        *,
        source: SourceInput | None = None,
    ) -> CompareResult:
        """Compare source and output claims in both directions.

        Strings are promoted to the canonical domain models.  ``source=`` is
        accepted as the concise Definition-of-Done spelling, while
        ``source_bundle=`` remains the structured API spelling.
        """

        structured_source, structured_generated = _coerce_inputs(
            source_bundle,
            generated_output,
            source,
        )
        evaluation_id = _evaluation_id(structured_source, structured_generated)
        source_claims: list[Claim] = []
        generated_claims: list[Claim] = []
        extractor: ClinicalExtractor | None = None
        optional_ner_failed = False

        try:
            extractor = self._nlp_service or _default_extractor()
            for document in structured_source.documents:
                source_claims.extend(_extract(extractor, document.content, document.document_id))
                optional_ner_failed = optional_ner_failed or _optional_ner_failed(extractor)
            generated_claims = _extract(
                extractor,
                structured_generated.content,
                "generated_output",
            )
            optional_ner_failed = optional_ner_failed or _optional_ner_failed(extractor)
        except Exception:
            # Extractor exceptions can contain document identifiers or source text.
            # Coverage is persisted, so expose only a stable, non-sensitive code.
            limitation = "Clinical claim extraction failed (EXTRACTOR_FAILED)."
            return CompareResult(
                evaluation_id=evaluation_id,
                execution_status=ExecutionStatus.FAILED,
                coverage=CoverageSummary(
                    status=CoverageStatus.UNASSESSED,
                    claims_assessed=0,
                    claims_unassessed=len(source_claims) + len(generated_claims),
                    limitations=[limitation],
                ),
                faithfulness=self._metrics([]),
                findings=[],
                source_claims=source_claims,
                generated_claims=generated_claims,
                versions=self._versions(extractor),
            )

        comparison_source_claims = _comparison_claims(source_claims)
        comparison_generated_claims = _comparison_claims(generated_claims)
        deadline = getattr(extractor, "deadline", None)
        processing_deadline = deadline if isinstance(deadline, float) else None
        plan = self._aligner.align(
            comparison_source_claims,
            comparison_generated_claims,
            deadline=processing_deadline,
        )
        semantic_inferences: dict[int, InferenceDecision] = {}
        semantic_inference_candidates: dict[int, tuple[int, ...]] = {}
        unresolved_generated_indices = (
            set(plan.unaligned_generated) | plan.uncertain_generated_indices
        )
        for generated_index in sorted(unresolved_generated_indices):
            ensure_deadline(processing_deadline)
            generated_claim = comparison_generated_claims[generated_index]
            if generated_claim.type not in _NARRATIVE_CLAIM_TYPES:
                continue
            decision = (
                self._inference_evaluator.evaluate(comparison_source_claims, generated_claim)
                if self._inference_evaluator is not None
                else None
            )
            if decision is not None:
                semantic_inferences[generated_index] = decision
                continue
            if _has_inference_cue(structured_generated.content, generated_claim):
                candidates = tuple(
                    source_index
                    for source_index in sorted(
                        set(plan.unaligned_source) | plan.uncertain_source_indices
                    )
                )
                if candidates:
                    semantic_inference_candidates[generated_index] = candidates
        semantic_uncertain_sources = {
            source_index
            for candidates in semantic_inference_candidates.values()
            for source_index in candidates
        }
        findings: list[Finding] = []
        ordinal = 0

        # Pass A: OUTPUT -> SOURCE.
        for alignment in plan.alignments:
            ensure_deadline(processing_deadline)
            aligned = self._aligned_findings(
                evaluation_id,
                ordinal,
                alignment,
                comparison_source_claims[alignment.source_index],
                comparison_generated_claims[alignment.generated_index],
                source_claims,
                generated_claims,
            )
            findings.extend(aligned)
            ordinal += len(aligned)

        for generated_index in plan.unaligned_generated:
            ensure_deadline(processing_deadline)
            if (
                generated_index in semantic_inference_candidates
                or generated_index in semantic_inferences
            ):
                continue
            generated_claim = comparison_generated_claims[generated_index]
            inferred = _is_explicit_inference(generated_claim)
            finding_type = FindingType.INFERRED if inferred else FindingType.ADDED
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=finding_type,
                    category=_category_for_claim(generated_claim),
                    reason_code="UNSUPPORTED_INFERENCE" if inferred else "UNSUPPORTED_ADDITION",
                    explanation=(
                        "The output labels this claim as an inference, but no direct source "
                        "claim aligns to it."
                        if inferred
                        else "No source claim aligns to this generated clinical claim."
                    ),
                    confidence=generated_claim.confidence,
                    claim_type=generated_claim.type,
                    generated=generated_claim.source_span,
                    generated_claim=generated_claim,
                )
            )
            ordinal += 1

        inference_support_sources: set[int] = set()
        for generated_index, decision in semantic_inferences.items():
            ensure_deadline(processing_deadline)
            generated_claim = comparison_generated_claims[generated_index]
            supporting_claims = [
                comparison_source_claims[index] for index in decision.source_indices
            ]
            inference_support_sources.update(decision.source_indices)
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.INFERRED,
                    category=FindingCategory.INFERENCE,
                    reason_code=decision.reason_code,
                    explanation=decision.explanation,
                    confidence=decision.confidence,
                    claim_type=generated_claim.type,
                    source=supporting_claims[0].source_span,
                    generated=generated_claim.source_span,
                    source_claim=supporting_claims[0],
                    generated_claim=generated_claim,
                    alignment_method=AlignmentMethod.SEMANTIC,
                    rule_id="BOUNDED_CLINICAL_INFERENCE",
                    rule_version="0.1.0",
                    source_support=[claim.source_span for claim in supporting_claims],
                )
            )
            ordinal += 1

        represented_uncertain_sources: set[int] = set()
        for generated_index, source_indices in semantic_inference_candidates.items():
            ensure_deadline(processing_deadline)
            generated_claim = comparison_generated_claims[generated_index]
            source_claim = (
                comparison_source_claims[source_indices[0]] if len(source_indices) == 1 else None
            )
            if source_claim is not None:
                represented_uncertain_sources.add(source_indices[0])
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.UNCERTAIN,
                    category=FindingCategory.INFERENCE,
                    reason_code="SEMANTIC_INFERENCE_UNASSESSED",
                    explanation=(
                        "The output uses explicit inference language, but no configured "
                        "semantic evaluator can determine whether the source supports it."
                    ),
                    confidence=0.5,
                    claim_type=generated_claim.type,
                    source=source_claim.source_span if source_claim is not None else None,
                    generated=generated_claim.source_span,
                    source_claim=source_claim,
                    generated_claim=generated_claim,
                    source_support=[
                        comparison_source_claims[index].source_span for index in source_indices
                    ],
                    uncertainty_reason="INFERENCE_EVIDENCE_UNRESOLVED",
                    candidate_evidence=[
                        comparison_source_claims[index].source_span for index in source_indices
                    ],
                )
            )
            ordinal += 1

        for uncertain in plan.uncertain:
            ensure_deadline(processing_deadline)
            if uncertain.generated_index in semantic_inferences:
                continue
            uncertain_generated_claim = (
                comparison_generated_claims[uncertain.generated_index]
                if uncertain.generated_index is not None
                else None
            )
            uncertain_source_claim = (
                comparison_source_claims[uncertain.source_indices[0]]
                if len(uncertain.source_indices) == 1
                else None
            )
            if uncertain_source_claim is not None:
                represented_uncertain_sources.add(uncertain.source_indices[0])
            candidate_ids = [
                comparison_source_claims[index].claim_id for index in uncertain.source_indices
            ]
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.UNCERTAIN,
                    category=FindingCategory.ALIGNMENT,
                    reason_code="ALIGNMENT_UNCERTAIN",
                    explanation=(
                        f"{uncertain.explanation} Candidate source claims: {candidate_ids}."
                    ),
                    confidence=0.5,
                    claim_type=(
                        uncertain_generated_claim.type
                        if uncertain_generated_claim is not None
                        else (
                            uncertain_source_claim.type
                            if uncertain_source_claim is not None
                            else None
                        )
                    ),
                    source=(
                        uncertain_source_claim.source_span
                        if uncertain_source_claim is not None
                        else None
                    ),
                    generated=(
                        uncertain_generated_claim.source_span
                        if uncertain_generated_claim is not None
                        else None
                    ),
                    source_claim=uncertain_source_claim,
                    generated_claim=uncertain_generated_claim,
                    uncertainty_reason="MULTIPLE_OR_WEAK_ALIGNMENT_CANDIDATES",
                    candidate_evidence=[
                        comparison_source_claims[index].source_span
                        for index in uncertain.source_indices
                    ],
                )
            )
            ordinal += 1

        # Pass B: SOURCE -> OUTPUT. Aligned claims were already represented in
        # Pass A; only omissions and unresolved source-side coverage are added.
        for source_index in sorted(
            (plan.uncertain_source_indices | semantic_uncertain_sources)
            - inference_support_sources
            - represented_uncertain_sources
        ):
            ensure_deadline(processing_deadline)
            source_claim = comparison_source_claims[source_index]
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.UNCERTAIN,
                    category=FindingCategory.ALIGNMENT,
                    reason_code="ALIGNMENT_UNCERTAIN",
                    explanation=(
                        "This source claim is part of an ambiguous alignment and was not "
                        "classified as supported or omitted."
                    ),
                    confidence=0.5,
                    claim_type=source_claim.type,
                    source=source_claim.source_span,
                    source_claim=source_claim,
                    uncertainty_reason="SOURCE_SIDE_ALIGNMENT_AMBIGUITY",
                    candidate_evidence=[source_claim.source_span],
                )
            )
            ordinal += 1

        for source_index in plan.unaligned_source:
            ensure_deadline(processing_deadline)
            if (
                source_index in semantic_uncertain_sources
                or source_index in inference_support_sources
            ):
                continue
            source_claim = comparison_source_claims[source_index]
            category, reason_code = _omission_classification(source_claim)
            findings.append(
                self._finding(
                    evaluation_id=evaluation_id,
                    ordinal=ordinal,
                    finding_type=FindingType.OMITTED,
                    category=category,
                    reason_code=reason_code,
                    explanation="No generated claim aligns to this source clinical claim.",
                    confidence=source_claim.confidence,
                    claim_type=source_claim.type,
                    source=source_claim.source_span,
                    source_claim=source_claim,
                )
            )
            ordinal += 1

        total_claims = len(source_claims) + len(generated_claims)
        uncertain_source_indices = (
            plan.uncertain_source_indices | semantic_uncertain_sources
        ) - inference_support_sources
        uncertain_generated_indices = (
            plan.uncertain_generated_indices | set(semantic_inference_candidates)
        ) - set(semantic_inferences)
        uncertain_count = len(uncertain_source_indices) + len(uncertain_generated_indices)
        limitations: list[str] = []
        limitations.extend(plan.limitations)
        if semantic_inference_candidates:
            limitations.append(
                "No bounded inference rule resolved every explicitly inferred narrative "
                "claim; unresolved evidence remains UNCERTAIN."
            )
        if optional_ner_failed:
            limitations.append(
                "Optional clinical NER failed (NER_ADAPTER_FAILED); deterministic "
                "extraction continued."
            )
        if uncertain_count:
            if self._semantic_matcher is None and self._embedding_matcher is None:
                limitations.append(
                    "Embedding and semantic fallbacks are not configured; unresolved "
                    "narrative alignments remain UNCERTAIN."
                )
            else:
                limitations.append("Optional alignment fallbacks did not resolve every claim.")
        if total_claims == 0:
            status = CoverageStatus.UNASSESSED
            limitations.append("No clinical claims were extracted from either input.")
        elif not source_claims or not generated_claims:
            status = CoverageStatus.PARTIAL
            missing_side = "source" if not source_claims else "generated output"
            limitations.append(f"No clinical claims were extracted from the {missing_side}.")
        elif uncertain_count or optional_ner_failed or plan.limitations:
            status = CoverageStatus.REDUCED
        else:
            status = CoverageStatus.FULL

        claims_unassessed = uncertain_count
        if optional_ner_failed and total_claims:
            # A failed optional extractor can hide claims that deterministic rules
            # never saw.  Reserve one conservative capability sentinel while keeping
            # the count bounded by the observed claim total.
            claims_unassessed = min(total_claims, max(1, claims_unassessed))

        return CompareResult(
            evaluation_id=evaluation_id,
            execution_status=ExecutionStatus.COMPLETED,
            coverage=CoverageSummary(
                status=status,
                claims_assessed=max(0, total_claims - claims_unassessed),
                claims_unassessed=claims_unassessed,
                limitations=list(dict.fromkeys(limitations)),
            ),
            faithfulness=self._metrics(findings),
            findings=findings,
            source_claims=source_claims,
            generated_claims=generated_claims,
            versions=self._versions(extractor),
        )


def compare(
    source_bundle: SourceInput | None = None,
    generated_output: GeneratedInput | None = None,
    *,
    source: SourceInput | None = None,
    nlp_service: ClinicalExtractor | None = None,
    semantic_matcher: SemanticMatcher | None = None,
    embedding_matcher: EmbeddingMatcher | None = None,
    inference_evaluator: BoundedClinicalInferenceEvaluator | None = None,
    enable_semantic: bool = True,
) -> CompareResult:
    """Public convenience function for a single clinical comparison."""

    return CompareEngine(
        nlp_service=nlp_service,
        semantic_matcher=semantic_matcher,
        embedding_matcher=embedding_matcher,
        inference_evaluator=inference_evaluator,
        enable_semantic=enable_semantic,
    ).compare(source_bundle, generated_output, source=source)

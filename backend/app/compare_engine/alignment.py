"""Conservative, explainable clinical claim alignment."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from backend.app.clinical_nlp.limits import ensure_deadline
from backend.app.compare_engine.semantic import (
    EmbeddingMatcher,
    SemanticLabel,
    SemanticMatcher,
)
from backend.app.domain.enums import AlignmentMethod, ClaimType
from backend.app.domain.models import Claim

ALIGNMENT_VERSION = "0.1.0"
EMBEDDING_ACCEPT_THRESHOLD = 0.90
EMBEDDING_UNCERTAIN_THRESHOLD = 0.75
EMBEDDING_MINIMUM_MARGIN = 0.05
MAX_ALIGNMENT_CANDIDATES = 256

_NARRATIVE_TYPES = {ClaimType.CONDITION, ClaimType.SYMPTOM, ClaimType.CLINICAL_FACT}
_SCALAR_TYPES = {
    ClaimType.DOSE,
    ClaimType.UNIT,
    ClaimType.FREQUENCY,
    ClaimType.ROUTE,
    ClaimType.LATERALITY,
    ClaimType.LAB_VALUE,
    ClaimType.DATE,
    ClaimType.TIME,
    ClaimType.DURATION,
    ClaimType.AGE,
}
_RELATION_KEYS = (
    "relation_id",
    "slot_id",
    "parent_claim_id",
    "parent_concept",
    "entity",
    "role",
)
_ATTRIBUTE_KEYS = (
    "laterality",
    "normalized_laterality",
    "dose",
    "normalized_dose",
    "unit",
    "normalized_unit",
    "frequency",
    "normalized_frequency",
    "route",
    "normalized_route",
    "normalized_value",
    "value",
    "date",
    "normalized_date",
    "duration",
    "normalized_duration",
)
_MEDICATION_SLOT_KEYS = (
    "dose",
    "normalized_dose",
    "unit",
    "normalized_unit",
    "frequency",
    "normalized_frequency",
    "route",
    "normalized_route",
)
_NON_CONCEPT_TOKENS = {
    "left",
    "right",
    "bilateral",
    "lt",
    "rt",
    "possible",
    "probable",
    "suspected",
    "no",
    "not",
    "without",
}
_CONCEPT_ALIASES = {
    "sob": "dyspnea",
    "shortness of breath": "dyspnea",
    "shortness breath": "dyspnea",
    "breathlessness": "dyspnea",
    "high blood pressure": "hypertension",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
}


@dataclass(frozen=True, slots=True)
class Alignment:
    source_index: int
    generated_index: int
    method: AlignmentMethod
    confidence: float
    explanation: str
    inferred: bool = False
    semantic_contradiction: bool = False


@dataclass(frozen=True, slots=True)
class UncertainAlignment:
    generated_index: int | None
    source_indices: tuple[int, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class AlignmentPlan:
    alignments: tuple[Alignment, ...]
    unaligned_generated: tuple[int, ...]
    unaligned_source: tuple[int, ...]
    uncertain: tuple[UncertainAlignment, ...]
    limitations: tuple[str, ...] = ()

    @property
    def uncertain_generated_indices(self) -> set[int]:
        return {item.generated_index for item in self.uncertain if item.generated_index is not None}

    @property
    def uncertain_source_indices(self) -> set[int]:
        return {index for item in self.uncertain for index in item.source_indices}


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_index: int
    method: AlignmentMethod
    confidence: float
    explanation: str
    priority: int


def _type_bucket(claim_type: ClaimType) -> str:
    return "NARRATIVE" if claim_type in _NARRATIVE_TYPES else claim_type.value


class _CandidateIndex:
    """Deterministic inverted index that avoids all-pairs candidate scans."""

    def __init__(self, source_claims: list[Claim]) -> None:
        self.by_bucket: dict[str, list[int]] = defaultdict(list)
        self.by_exact: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_normalized: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_token: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_relation: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self.by_type: dict[ClaimType, list[int]] = defaultdict(list)
        for index, claim in enumerate(source_claims):
            bucket = _type_bucket(claim.type)
            self.by_bucket[bucket].append(index)
            self.by_type[claim.type].append(index)
            self.by_exact[(bucket, claim.concept.strip())].append(index)
            self.by_normalized[(bucket, normalized_concept(claim.concept))].append(index)
            for token in set(_tokens(claim.concept)):
                self.by_token[(bucket, token)].append(index)
            for key in _RELATION_KEYS:
                value = claim.attributes.get(key)
                if value is not None:
                    self.by_relation[
                        (bucket, key, str(value).strip().casefold())
                    ].append(index)

    def strong(self, generated: Claim) -> list[int]:
        bucket = _type_bucket(generated.type)
        candidates: set[int] = set(self.by_exact[(bucket, generated.concept.strip())])
        candidates.update(
            self.by_normalized[(bucket, normalized_concept(generated.concept))]
        )
        for token in set(_tokens(generated.concept)):
            candidates.update(self.by_token[(bucket, token)])
        for key in _RELATION_KEYS:
            value = generated.attributes.get(key)
            if value is not None:
                candidates.update(
                    self.by_relation[(bucket, key, str(value).strip().casefold())]
                )
        return sorted(candidates)

    def compatible(self, generated: Claim) -> list[int]:
        return self.by_bucket[_type_bucket(generated.type)]


def _tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return tuple(token for token in tokens if token not in _NON_CONCEPT_TOKENS)


def normalized_concept(value: str) -> str:
    """Return a deliberately small deterministic concept normalization."""

    phrase = " ".join(_tokens(value))
    alias = _CONCEPT_ALIASES.get(phrase, phrase)
    return " ".join(sorted(alias.split()))


def _compatible_types(source: Claim, generated: Claim) -> bool:
    if source.type is generated.type:
        return True
    return source.type in _NARRATIVE_TYPES and generated.type in _NARRATIVE_TYPES


def _relation_match(source: Claim, generated: Claim) -> bool:
    for key in _RELATION_KEYS:
        source_value = source.attributes.get(key)
        generated_value = generated.attributes.get(key)
        if (
            source_value is not None
            and generated_value is not None
            and str(source_value).strip().casefold() == str(generated_value).strip().casefold()
        ):
            return True
    return False


def _concept_similarity(source: Claim, generated: Claim) -> float:
    source_tokens = set(_tokens(source.concept))
    generated_tokens = set(_tokens(generated.concept))
    if not source_tokens or not generated_tokens:
        return 0.0
    return len(source_tokens & generated_tokens) / len(source_tokens | generated_tokens)


def _strong_candidate(source: Claim, generated: Claim, source_index: int) -> _Candidate | None:
    if not _compatible_types(source, generated):
        return None
    if source.concept.strip() == generated.concept.strip():
        return _Candidate(
            source_index,
            AlignmentMethod.EXACT,
            1.0,
            "Claim types and concepts match exactly.",
            4,
        )
    if normalized_concept(source.concept) == normalized_concept(generated.concept):
        return _Candidate(
            source_index,
            AlignmentMethod.NORMALIZED,
            0.98,
            "Claim concepts match after deterministic normalization.",
            3,
        )
    if _relation_match(source, generated):
        return _Candidate(
            source_index,
            AlignmentMethod.ATTRIBUTE,
            0.96,
            "Claims share an explicit relation or slot attribute.",
            2,
        )
    similarity = _concept_similarity(source, generated)
    if similarity >= 0.8:
        return _Candidate(
            source_index,
            AlignmentMethod.NORMALIZED,
            round(similarity, 3),
            f"Claim concept token overlap is {similarity:.2f}.",
            1,
        )
    return None


def _attribute_agreement(source: Claim, generated: Claim) -> tuple[int, int]:
    agreements = int(source.assertion is generated.assertion) + int(
        source.subject is generated.subject
    )
    comparisons = 2
    for key in _ATTRIBUTE_KEYS:
        if key in source.attributes and key in generated.attributes:
            comparisons += 1
            if str(source.attributes[key]).casefold() == str(generated.attributes[key]).casefold():
                agreements += 1
    return agreements, comparisons


def _choose_candidate(
    candidates: list[_Candidate],
    source_claims: list[Claim],
    generated: Claim,
) -> _Candidate | None:
    best_priority = max(candidate.priority for candidate in candidates)
    finalists = [candidate for candidate in candidates if candidate.priority == best_priority]
    if len(finalists) == 1:
        return finalists[0]

    ranked = [
        (_attribute_agreement(source_claims[item.source_index], generated), item)
        for item in finalists
    ]
    best_agreement = max(score[0] for score, _ in ranked)
    agreement_finalists = [item for score, item in ranked if score[0] == best_agreement]
    if len(agreement_finalists) == 1:
        return agreement_finalists[0]
    return None


def _weak_candidate_indices(
    source_claims: list[Claim],
    generated: Claim,
    candidate_indices: Iterable[int] | None = None,
) -> list[int]:
    candidates: list[int] = []
    indices = range(len(source_claims)) if candidate_indices is None else candidate_indices
    for index in indices:
        source = source_claims[index]
        if not _compatible_types(source, generated):
            continue
        similarity = _concept_similarity(source, generated)
        if similarity >= 0.34 or (
            source.type in _SCALAR_TYPES and _relation_match(source, generated)
        ):
            candidates.append(index)
    return candidates


def _unique_structured_slot(
    source_claims: list[Claim],
    generated_claims: list[Claim],
    generated: Claim,
    *,
    source_by_type: dict[ClaimType, list[int]] | None = None,
    generated_type_counts: Counter[ClaimType] | None = None,
) -> int | None:
    """Resolve a unique structured slot, never a narrative singleton."""

    generated_count = (
        generated_type_counts[generated.type]
        if generated_type_counts is not None
        else sum(claim.type is generated.type for claim in generated_claims)
    )
    if generated_count != 1:
        return None
    same_type_sources = (
        source_by_type.get(generated.type, [])
        if source_by_type is not None
        else [
            index for index, claim in enumerate(source_claims) if claim.type is generated.type
        ]
    )
    if len(same_type_sources) != 1:
        return None
    source_index = same_type_sources[0]
    source = source_claims[source_index]
    if generated.type in _SCALAR_TYPES:
        return source_index
    if generated.type is ClaimType.ALLERGY:
        concepts = {
            normalized_concept(source.concept),
            normalized_concept(generated.concept),
        }
        if "allergy drug" in concepts and source.assertion is not generated.assertion:
            return source_index
        return None
    if generated.type is not ClaimType.MEDICATION:
        return None

    source_action = source.attributes.get("medication_action")
    generated_action = generated.attributes.get("medication_action")
    if (
        source_action is not None
        and generated_action is not None
        and str(source_action).casefold() == str(generated_action).casefold()
        and source.assertion is generated.assertion
        and source.subject is generated.subject
    ):
        return source_index

    agreements = 0
    for key in _MEDICATION_SLOT_KEYS:
        if (
            key in source.attributes
            and key in generated.attributes
            and str(source.attributes[key]).casefold() == str(generated.attributes[key]).casefold()
        ):
            agreements += 1
    return source_index if agreements >= 2 else None


def _semantic_alignment(
    source_claims: list[Claim],
    generated: Claim,
    generated_index: int,
    candidate_indices: Iterable[int],
    matcher: SemanticMatcher,
) -> tuple[Alignment | None, str | None]:
    accepted: list[Alignment] = []
    for source_index in candidate_indices:
        try:
            decision = matcher.compare(source_claims[source_index], generated)
        except Exception:
            return None, "Semantic matcher failed (SEMANTIC_MATCHER_FAILED)."
        if decision.label in {SemanticLabel.EQUIVALENT, SemanticLabel.INFERRED}:
            accepted.append(
                Alignment(
                    source_index=source_index,
                    generated_index=generated_index,
                    method=AlignmentMethod.SEMANTIC,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                    inferred=decision.label is SemanticLabel.INFERRED,
                )
            )
        elif decision.label is SemanticLabel.CONTRADICTORY:
            accepted.append(
                Alignment(
                    source_index=source_index,
                    generated_index=generated_index,
                    method=AlignmentMethod.SEMANTIC,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                    semantic_contradiction=True,
                )
            )
    if len(accepted) == 1:
        return accepted[0], None
    return None, None


def _embedding_candidates(
    source_claims: list[Claim],
    generated: Claim,
    matcher: EmbeddingMatcher,
    *,
    accept_threshold: float,
    uncertain_threshold: float,
    minimum_margin: float,
    candidate_indices: Iterable[int] | None = None,
) -> tuple[Alignment | None, list[int], str | None]:
    if generated.type not in _NARRATIVE_TYPES:
        return None, [], None

    scored: list[tuple[float, int]] = []
    indices = range(len(source_claims)) if candidate_indices is None else candidate_indices
    for source_index in indices:
        source = source_claims[source_index]
        if not _compatible_types(source, generated):
            continue
        try:
            similarity = float(matcher.similarity(source, generated))
        except Exception:
            candidates = [
                index
                for index in indices
                if (claim := source_claims[index])
                if _compatible_types(claim, generated)
            ]
            return (
                None,
                candidates,
                "Embedding matcher failed (EMBEDDING_MATCHER_FAILED).",
            )
        if not 0.0 <= similarity <= 1.0:
            candidates = [
                index
                for index in indices
                if (claim := source_claims[index])
                if _compatible_types(claim, generated)
            ]
            return None, candidates, "Embedding matcher returned an out-of-range similarity."
        scored.append((similarity, source_index))

    scored.sort(reverse=True)
    uncertain_indices = [
        source_index for similarity, source_index in scored if similarity >= uncertain_threshold
    ]
    if not scored or scored[0][0] < accept_threshold:
        return None, uncertain_indices, None
    top_score, top_index = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if top_score - runner_up < minimum_margin:
        return None, uncertain_indices, None
    return (
        Alignment(
            source_index=top_index,
            generated_index=-1,
            method=AlignmentMethod.EMBEDDING,
            confidence=round(top_score, 3),
            explanation=(
                f"Biomedical embedding similarity {top_score:.3f} met the "
                f"{accept_threshold:.2f} threshold with required margin."
            ),
        ),
        uncertain_indices,
        None,
    )


class ClaimAligner:
    """Align output claims to source claims without resolving ambiguity by guess."""

    version = ALIGNMENT_VERSION

    def __init__(
        self,
        semantic_matcher: SemanticMatcher | None = None,
        embedding_matcher: EmbeddingMatcher | None = None,
        *,
        embedding_accept_threshold: float = EMBEDDING_ACCEPT_THRESHOLD,
        embedding_uncertain_threshold: float = EMBEDDING_UNCERTAIN_THRESHOLD,
        embedding_minimum_margin: float = EMBEDDING_MINIMUM_MARGIN,
    ) -> None:
        if not 0.0 <= embedding_uncertain_threshold <= embedding_accept_threshold <= 1.0:
            raise ValueError("Embedding thresholds must satisfy 0 <= uncertain <= accept <= 1")
        if not 0.0 <= embedding_minimum_margin <= 1.0:
            raise ValueError("embedding_minimum_margin must be between 0 and 1")
        self.semantic_matcher = semantic_matcher
        self.embedding_matcher = embedding_matcher
        self.embedding_accept_threshold = embedding_accept_threshold
        self.embedding_uncertain_threshold = embedding_uncertain_threshold
        self.embedding_minimum_margin = embedding_minimum_margin

    def align(
        self,
        source_claims: list[Claim],
        generated_claims: list[Claim],
        *,
        deadline: float | None = None,
    ) -> AlignmentPlan:
        alignments: list[Alignment] = []
        uncertain: list[UncertainAlignment] = []
        unaligned_generated: list[int] = []
        limitations: list[str] = []
        candidate_index = _CandidateIndex(source_claims)
        generated_type_counts = Counter(claim.type for claim in generated_claims)

        for generated_index, generated in enumerate(generated_claims):
            ensure_deadline(deadline)
            strong_indices = candidate_index.strong(generated)
            if len(strong_indices) > MAX_ALIGNMENT_CANDIDATES:
                uncertain.append(
                    UncertainAlignment(
                        generated_index,
                        (),
                        "The deterministic candidate safety limit was exceeded; the engine "
                        "did not guess.",
                    )
                )
                limitations.append(
                    "Alignment candidate safety limit exceeded "
                    "(ALIGNMENT_CANDIDATE_LIMIT_EXCEEDED)."
                )
                continue
            candidates = [
                candidate
                for source_index in strong_indices
                if (source := source_claims[source_index])
                if (candidate := _strong_candidate(source, generated, source_index)) is not None
            ]
            if candidates:
                chosen = _choose_candidate(candidates, source_claims, generated)
                if chosen is None:
                    uncertain.append(
                        UncertainAlignment(
                            generated_index,
                            tuple(candidate.source_index for candidate in candidates),
                            "Multiple source claims are equally plausible deterministic matches; "
                            "the engine did not guess.",
                        )
                    )
                    continue
                alignments.append(
                    Alignment(
                        source_index=chosen.source_index,
                        generated_index=generated_index,
                        method=chosen.method,
                        confidence=chosen.confidence,
                        explanation=chosen.explanation,
                    )
                )
                continue

            structured_slot = _unique_structured_slot(
                source_claims,
                generated_claims,
                generated,
                source_by_type=candidate_index.by_type,
                generated_type_counts=generated_type_counts,
            )
            if structured_slot is not None:
                alignments.append(
                    Alignment(
                        source_index=structured_slot,
                        generated_index=generated_index,
                        method=AlignmentMethod.ATTRIBUTE,
                        confidence=0.9,
                        explanation=(
                            "Claims occupy the only structured slot of this type in each input."
                        ),
                    )
                )
                continue

            embedding_candidates: list[int] = []
            if self.embedding_matcher is not None:
                compatible_indices = candidate_index.compatible(generated)
                if len(compatible_indices) > MAX_ALIGNMENT_CANDIDATES:
                    limitations.append(
                        "Optional alignment candidate safety limit exceeded "
                        "(ALIGNMENT_CANDIDATE_LIMIT_EXCEEDED)."
                    )
                    compatible_indices = []
                embedded, embedding_candidates, embedding_limitation = _embedding_candidates(
                    source_claims,
                    generated,
                    self.embedding_matcher,
                    accept_threshold=self.embedding_accept_threshold,
                    uncertain_threshold=self.embedding_uncertain_threshold,
                    minimum_margin=self.embedding_minimum_margin,
                    candidate_indices=compatible_indices,
                )
                if embedding_limitation is not None:
                    limitations.append(embedding_limitation)
                if embedded is not None:
                    alignments.append(
                        Alignment(
                            source_index=embedded.source_index,
                            generated_index=generated_index,
                            method=embedded.method,
                            confidence=embedded.confidence,
                            explanation=embedded.explanation,
                        )
                    )
                    continue

            weak_candidates = _weak_candidate_indices(
                source_claims,
                generated,
                strong_indices,
            )
            fallback_candidates = embedding_candidates or weak_candidates
            if (
                not fallback_candidates
                and self.semantic_matcher is not None
                and getattr(
                    self.semantic_matcher,
                    "evaluate_all_narrative_pairs",
                    True,
                )
                and generated.type in _NARRATIVE_TYPES
            ):
                compatible_indices = candidate_index.compatible(generated)
                if len(compatible_indices) > MAX_ALIGNMENT_CANDIDATES:
                    uncertain.append(
                        UncertainAlignment(
                            generated_index,
                            (),
                            "The optional semantic candidate safety limit was exceeded; "
                            "alignment remains uncertain.",
                        )
                    )
                    limitations.append(
                        "Alignment candidate safety limit exceeded "
                        "(ALIGNMENT_CANDIDATE_LIMIT_EXCEEDED)."
                    )
                    continue
                fallback_candidates = compatible_indices
            if len(fallback_candidates) > MAX_ALIGNMENT_CANDIDATES:
                uncertain.append(
                    UncertainAlignment(
                        generated_index,
                        (),
                        "The optional semantic candidate safety limit was exceeded; "
                        "alignment remains uncertain.",
                    )
                )
                limitations.append(
                    "Alignment candidate safety limit exceeded "
                    "(ALIGNMENT_CANDIDATE_LIMIT_EXCEEDED)."
                )
                continue
            if fallback_candidates and self.semantic_matcher is not None:
                semantic, semantic_limitation = _semantic_alignment(
                    source_claims,
                    generated,
                    generated_index,
                    fallback_candidates,
                    self.semantic_matcher,
                )
                if semantic_limitation is not None:
                    limitations.append(semantic_limitation)
                if semantic is not None:
                    alignments.append(semantic)
                    continue
            if fallback_candidates:
                uncertain.append(
                    UncertainAlignment(
                        generated_index,
                        tuple(fallback_candidates),
                        "Only weak semantic candidates were available; alignment "
                        "remains uncertain.",
                    )
                )
            else:
                unaligned_generated.append(generated_index)

        matched_sources = {alignment.source_index for alignment in alignments}
        uncertain_sources = {index for item in uncertain for index in item.source_indices}
        unaligned_source = [
            index
            for index in range(len(source_claims))
            if index not in matched_sources and index not in uncertain_sources
        ]
        return AlignmentPlan(
            alignments=tuple(alignments),
            unaligned_generated=tuple(unaligned_generated),
            unaligned_source=tuple(unaligned_source),
            uncertain=tuple(uncertain),
            limitations=tuple(dict.fromkeys(limitations)),
        )

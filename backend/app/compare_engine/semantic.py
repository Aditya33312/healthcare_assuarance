"""Bounded, fail-closed semantic evaluation contracts and cascade.

The Phase 1 compare engine has no hidden LLM dependency.  Deployments may
inject an NLI-backed implementation of this protocol for narrative claims;
deterministic attribute contradictions are evaluated after alignment and
always take precedence over this decision.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from backend.app.domain.models import Claim


class SemanticLabel(StrEnum):
    EQUIVALENT = "EQUIVALENT"
    INFERRED = "INFERRED"
    CONTRADICTORY = "CONTRADICTORY"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    label: SemanticLabel
    confidence: float
    explanation: str
    reason_code: str = "SEMANTIC_CLASSIFICATION"


class SemanticMatcher(Protocol):
    """Strict interface for an optional semantic/NLI fallback."""

    version: str

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        """Compare one candidate narrative pair without changing its claims."""


class EmbeddingMatcher(Protocol):
    """Optional biomedical embedding similarity provider."""

    version: str

    def similarity(self, source: Claim, generated: Claim) -> float:
        """Return a bounded cosine-like similarity in the range 0..1."""


class StructuredSemanticProvider(Protocol):
    """Provider boundary for a controlled JSON-only semantic judge."""

    version: str

    def evaluate(self, source_evidence: str, generated_claim: str) -> Mapping[str, Any]: ...


_TERMINOLOGY: dict[str, str] = {
    "breathlessness": "dyspnea",
    "dyspnoea": "dyspnea",
    "dyspnea": "dyspnea",
    "shortness of breath": "dyspnea",
    "high blood pressure": "hypertension",
    "hypertension": "hypertension",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
    "myocardial infarction": "myocardial infarction",
    "edema": "swelling",
    "swelling": "swelling",
}

_CONTRADICTIONS = {
    frozenset(("improved", "worsened")),
    frozenset(("stable", "progressing")),
    frozenset(("negative", "positive")),
    frozenset(("benign", "malignant")),
    frozenset(("resolved", "ongoing")),
    frozenset(("cleared", "active")),
    frozenset(("discontinued", "continued")),
    frozenset(("planned", "completed")),
    frozenset(("exertion", "rest")),
    frozenset(("can", "cannot")),
    frozenset(("baseline", "decline")),
}

_AMBIGUITY_CUES = re.compile(
    r"\b(?:unclear|unspecified|not\s+named|incomplete|without\s+a\s+date|"
    r"conflicting|illegible|borderline)\b",
    re.IGNORECASE,
)


def _canonical_term(value: str) -> str:
    candidate = " ".join(value.strip().casefold().split())
    return _TERMINOLOGY.get(candidate, candidate)


def _lexical_contradiction(source: str, generated: str) -> bool:
    source_tokens = set(re.findall(r"[a-z]+", source.casefold()))
    generated_tokens = set(re.findall(r"[a-z]+", generated.casefold()))
    return any(
        (pair & source_tokens) and (pair & generated_tokens) and not pair <= source_tokens
        for pair in _CONTRADICTIONS
    )


def _semantic_context(claim: Claim) -> str:
    value = claim.attributes.get("semantic_context")
    return str(value) if value else claim.source_span.text


class ClinicalSemanticCascade:
    """Terminology -> NLI -> controlled judge, with disagreement -> UNCERTAIN.

    This matcher never receives medication/dose/route/laterality scalar slots;
    the aligner reserves it for narrative claims and deterministic rules execute
    after alignment, so a hard structured contradiction retains precedence.
    """

    version = "clinical-semantic-cascade@0.1.0"
    evaluate_all_narrative_pairs = False

    def __init__(
        self,
        *,
        nli_matcher: SemanticMatcher | None = None,
        controlled_judge: SemanticMatcher | None = None,
        high_confidence_threshold: float = 0.85,
    ) -> None:
        if not 0.5 <= high_confidence_threshold <= 1.0:
            raise ValueError("high_confidence_threshold must be between 0.5 and 1")
        self._nli = nli_matcher
        self._judge = controlled_judge
        self._threshold = high_confidence_threshold

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        source_term = _canonical_term(source.concept)
        generated_term = _canonical_term(generated.concept)
        source_context = _semantic_context(source)
        generated_context = _semantic_context(generated)
        if _AMBIGUITY_CUES.search(source_context):
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                0.75,
                "The bounded source evidence contains an explicit ambiguity cue.",
                "EXPLICIT_SOURCE_AMBIGUITY",
            )
        if _lexical_contradiction(source_context, generated_context):
            return SemanticDecision(
                SemanticLabel.CONTRADICTORY,
                0.95,
                "A bounded lexical opposition changes the clinical meaning.",
                "SEMANTIC_CONTRADICTION",
            )
        if source_term == generated_term:
            return SemanticDecision(
                SemanticLabel.EQUIVALENT,
                0.99,
                "Clinical terminology normalizes to the same concept.",
                "TERMINOLOGY_EQUIVALENCE",
            )
        if self._nli is None:
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                0.5,
                "No pretrained NLI evaluator is configured for this unresolved pair.",
                "NLI_UNAVAILABLE",
            )
        try:
            nli = self._nli.compare(source, generated)
        except Exception:
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                0.0,
                "The NLI evaluator failed or timed out.",
                "NLI_EVALUATOR_FAILED",
            )
        if nli.confidence >= self._threshold:
            return nli
        if self._judge is None:
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                nli.confidence,
                "NLI confidence is below threshold and no controlled judge is enabled.",
                "NLI_LOW_CONFIDENCE",
            )
        try:
            judge = self._judge.compare(source, generated)
        except Exception:
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                nli.confidence,
                "The controlled semantic judge failed or returned malformed output.",
                "CONTROLLED_JUDGE_FAILED",
            )
        if judge.confidence < self._threshold or judge.label is not nli.label:
            return SemanticDecision(
                SemanticLabel.UNCERTAIN,
                min(nli.confidence, judge.confidence),
                "Enabled semantic evaluators did not produce one high-confidence label.",
                "SEMANTIC_EVALUATOR_DISAGREEMENT",
            )
        return judge


class ControlledJSONSemanticMatcher:
    """Validate the strict response contract of an external controlled judge."""

    version = "controlled-json-semantic-judge@0.1.0"
    _keys: ClassVar[set[str]] = {
        "relationship",
        "confidence",
        "reason_code",
        "supporting_evidence",
    }
    _labels: ClassVar[dict[str, SemanticLabel]] = {
        "SUPPORTED": SemanticLabel.EQUIVALENT,
        "INFERRED": SemanticLabel.INFERRED,
        "CONTRADICTED": SemanticLabel.CONTRADICTORY,
        "UNCERTAIN": SemanticLabel.UNCERTAIN,
    }

    def __init__(self, provider: StructuredSemanticProvider) -> None:
        self._provider = provider
        self.version = f"{self.version};provider={provider.version}"

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        response = self._provider.evaluate(source.source_span.text, generated.source_span.text)
        if set(response) != self._keys:
            raise ValueError("Controlled judge response has unexpected fields")
        relationship = response.get("relationship")
        confidence = response.get("confidence")
        reason_code = response.get("reason_code")
        evidence = response.get("supporting_evidence")
        if relationship not in self._labels:
            raise ValueError("Controlled judge relationship is invalid")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Controlled judge confidence is invalid")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Controlled judge confidence is out of range")
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("Controlled judge reason_code is invalid")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise ValueError("Controlled judge supporting_evidence is invalid")
        return SemanticDecision(
            self._labels[str(relationship)],
            float(confidence),
            "Controlled judge returned a schema-valid bounded decision.",
            reason_code,
        )

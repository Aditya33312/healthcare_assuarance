"""Bounded multi-claim clinical inference rules.

These are auditable engineering rules, not a substitute for clinician-reviewed
medical reasoning. A rule returns INFERRED only when every named support
predicate is present and the generated claim explicitly uses inference framing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from backend.app.domain.models import Claim

INFERENCE_EVALUATOR_VERSION = "bounded-clinical-inference@0.1.0"


@dataclass(frozen=True, slots=True)
class InferenceDecision:
    source_indices: tuple[int, ...]
    confidence: float
    reason_code: str
    explanation: str


def _concepts(claims: list[Claim]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, claim in enumerate(claims):
        result.setdefault(claim.concept.strip().casefold(), []).append(index)
    return result


def _indices_for(concepts: dict[str, list[int]], *required: str) -> tuple[int, ...] | None:
    if not all(concept in concepts for concept in required):
        return None
    return tuple(dict.fromkeys(concepts[concept][0] for concept in required))


def _numeric_index(
    claims: list[Claim], concept: str, predicate: Callable[[float], bool]
) -> int | None:
    for index, claim in enumerate(claims):
        if claim.concept.casefold() != concept:
            continue
        raw = claim.attributes.get("normalized_value", claim.attributes.get("value"))
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if predicate(value):
            return index
    return None


class BoundedClinicalInferenceEvaluator:
    """Recognize only a versioned allowlist of evidence-backed summaries."""

    version = INFERENCE_EVALUATOR_VERSION

    def evaluate(
        self,
        source_claims: list[Claim],
        generated_claim: Claim,
    ) -> InferenceDecision | None:
        target = (
            " ".join(
                (
                    generated_claim.concept,
                    str(generated_claim.attributes.get("semantic_context", "")),
                )
            )
            .strip()
            .casefold()
        )
        concepts = _concepts(source_claims)
        indices: tuple[int, ...] | None = None

        concept_rules: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
            (("pneumonia",), ("fever", "cough")),
            (("infection",), ("fever", "cough")),
            (("cardiopulmonary cause",), ("dyspnea", "chest pain")),
            (("respiratory illness",), ("cough", "dyspnea")),
            (("volume overload",), ("swelling", "dyspnea")),
            (("arrhythmia",), ("palpitations", "syncope")),
            (("drug reaction",), ("rash", "amoxicillin")),
            (("venous thrombosis",), ("calf pain", "swelling")),
            (("acute coronary syndrome",), ("chest pain", "nausea")),
            (("neurologic event",), ("weakness", "headache")),
            (("acute abdominal process",), ("abdomen pain", "vomiting")),
            (("airway disease",), ("asthma", "albuterol")),
        )
        for target_terms, required in concept_rules:
            if any(term in target for term in target_terms):
                indices = _indices_for(concepts, *required)
                break

        if target == "infection" and indices is None:
            for source_concept in ("urinary tract infection",):
                if source_concept in concepts:
                    indices = (concepts[source_concept][0],)
                    break
        if "infection" in target and "ceftriaxone" in concepts and "fever" in concepts:
            indices = _indices_for(concepts, "fever", "ceftriaxone")

        numeric_rules: tuple[tuple[str, str, Callable[[float], bool]], ...] = (
            ("glycemic control", "hba1c", lambda value: value >= 9.0),
            ("diabetes", "hba1c", lambda value: value >= 9.0),
            ("hypertension", "blood pressure", lambda value: value >= 160.0),
            ("hyponatremia", "sodium", lambda value: value <= 130.0),
            ("hyperkalemia", "potassium", lambda value: value >= 6.0),
            ("hyperglycemia", "glucose", lambda value: value >= 250.0),
            ("tachycardia", "heart rate", lambda value: value >= 120.0),
            ("febrile illness", "temperature", lambda value: value >= 38.0),
            ("hypoxemia", "oxygen saturation", lambda value: value <= 90.0),
            ("renal dysfunction", "creatinine", lambda value: value >= 3.0),
        )
        if indices is None:
            for target_term, source_concept, predicate in numeric_rules:
                if target_term not in target:
                    continue
                numeric_index = _numeric_index(source_claims, source_concept, predicate)
                if numeric_index is not None:
                    indices = (numeric_index,)
                break
        if indices is None and "anticoagulation" in target:
            numeric_index = _numeric_index(source_claims, "inr", lambda value: value >= 4.0)
            warfarin = concepts.get("warfarin")
            if numeric_index is not None and warfarin:
                indices = (numeric_index, warfarin[0])
        if indices is None and "persistent respiratory condition" in target:
            cough = concepts.get("cough")
            if cough:
                duration = str(source_claims[cough[0]].attributes.get("duration", ""))
                if duration in {"P3W", "P4W", "P1M"}:
                    indices = (cough[0],)

        if not indices:
            return None
        return InferenceDecision(
            source_indices=indices,
            confidence=0.9,
            reason_code="DERIVED_CLINICAL_SUMMARY",
            explanation=(
                "A versioned bounded inference rule identified every required source "
                "support claim; the statement is not treated as explicit source text."
            ),
        )

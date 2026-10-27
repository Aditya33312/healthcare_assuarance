"""Evaluator routing with explicit implemented/unassessed boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from backend.app.compare_engine import compare
from backend.app.evaluators.contract import ContractEvaluator
from backend.app.evaluators.external import (
    ExternalRunner,
    LangTestEvaluator,
    MedHELMEvaluator,
    runner_from_environment,
)
from backend.app.test_selection import ALL_TEST_FAMILIES, TestFamily
from backend.app.version import ENGINE_VERSION

EVALUATOR_ROUTER_VERSION = "evaluator-router@1.2.0"


class EvaluatorAvailability(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    UNASSESSED = "UNASSESSED"


class EvaluatorRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test_family: TestFamily
    evaluator: str
    availability: EvaluatorAvailability
    evaluator_version: str | None = None
    reason: str | None = None

    @property
    def implemented(self) -> bool:
        return self.availability is EvaluatorAvailability.IMPLEMENTED


class UnassessedEvaluation(BaseModel):
    """Visible result for a structurally supported but unavailable evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_family: TestFamily
    assessment_status: EvaluatorAvailability = EvaluatorAvailability.UNASSESSED
    evaluator: str
    reason: str = Field(min_length=1)
    release_decision: None = None


class Evaluator(Protocol):
    name: str
    version: str

    def evaluate(self, payload: Mapping[str, Any]) -> object:
        """Evaluate one family-specific payload or raise on technical failure."""


class ClinicalFaithfulnessEvaluator:
    """Adapter from the generic router contract to the compare engine."""

    name = "clinical_compare"
    version = ENGINE_VERSION

    def evaluate(self, payload: Mapping[str, Any]) -> object:
        source = payload.get("source_bundle", payload.get("source"))
        generated_output = payload.get("generated_output")
        if source is None:
            raise ValueError("clinical faithfulness evaluation requires source or source_bundle")
        if generated_output is None:
            raise ValueError("clinical faithfulness evaluation requires generated_output")
        return compare(source_bundle=source, generated_output=generated_output)


_EVALUATOR_NAMES: dict[TestFamily, str] = {
    TestFamily.CLINICAL_FAITHFULNESS: "clinical_compare",
    TestFamily.CLINICAL_NOTE_GENERATION: "clinical_note",
    TestFamily.CLINICAL_TASK_PERFORMANCE: "clinical_task",
    TestFamily.MEDICATION_SAFETY: "medication_safety",
    TestFamily.NEGATION_ASSERTION: "assertion_consistency",
    TestFamily.OMISSION_DETECTION: "omission_detection",
    TestFamily.ROBUSTNESS: "robustness",
    TestFamily.DEMOGRAPHIC_BIAS: "demographic_bias",
    TestFamily.MEDICAL_COGNITIVE_BIAS: "medical_cognitive_bias",
    TestFamily.SAFETY_REFUSAL: "safety_refusal",
    TestFamily.PROMPT_INJECTION_JAILBREAK: "prompt_injection",
    TestFamily.MODEL_DATA_EXTRACTION: "model_data_extraction",
    TestFamily.RAG_GROUNDING: "rag_grounding",
    TestFamily.RAG_POISONING: "rag_poisoning",
    TestFamily.REGULATORY_POLICY_HARDENING: "policy_hardening",
    TestFamily.CUSTOM_BUSINESS_REGRESSION: "custom_regression",
}

if set(_EVALUATOR_NAMES) != set(ALL_TEST_FAMILIES):  # pragma: no cover - import invariant
    raise RuntimeError("the evaluator route catalog must enumerate every test family")


class EvaluatorRouter:
    """Resolve every known family without pretending unavailable work ran."""

    version = EVALUATOR_ROUTER_VERSION

    def __init__(
        self,
        evaluators: Mapping[TestFamily, Evaluator] | None = None,
        *,
        medhelm_runner: ExternalRunner | None = None,
        langtest_runner: ExternalRunner | None = None,
        use_environment_runners: bool = True,
    ) -> None:
        self._evaluators: dict[TestFamily, Evaluator] = {
            TestFamily.CLINICAL_FAITHFULNESS: ClinicalFaithfulnessEvaluator(),
        }
        self._evaluators.update(
            {
                family: ContractEvaluator(family, _EVALUATOR_NAMES[family])
                for family in ALL_TEST_FAMILIES
                if family is not TestFamily.CLINICAL_FAITHFULNESS
            }
        )
        if use_environment_runners:
            medhelm_runner = medhelm_runner or runner_from_environment(
                "GATEKEEPER_MEDHELM_COMMAND"
            )
            langtest_runner = langtest_runner or runner_from_environment(
                "GATEKEEPER_LANGTEST_COMMAND"
            )
        if medhelm_runner is not None:
            medhelm = MedHELMEvaluator(medhelm_runner)
            self._evaluators.update(
                {
                    TestFamily.CLINICAL_NOTE_GENERATION: medhelm,
                    TestFamily.CLINICAL_TASK_PERFORMANCE: medhelm,
                    TestFamily.MEDICATION_SAFETY: medhelm,
                }
            )
        if langtest_runner is not None:
            langtest = LangTestEvaluator(langtest_runner)
            self._evaluators.update(
                {
                    TestFamily.ROBUSTNESS: langtest,
                    TestFamily.DEMOGRAPHIC_BIAS: langtest,
                    TestFamily.PROMPT_INJECTION_JAILBREAK: langtest,
                }
            )
        if evaluators:
            self._evaluators.update(evaluators)

    def route(self, test_family: TestFamily | str) -> EvaluatorRoute:
        family = TestFamily(test_family)
        evaluator = self._evaluators.get(family)
        if evaluator is not None:
            return EvaluatorRoute(
                test_family=family,
                evaluator=evaluator.name,
                evaluator_version=evaluator.version,
                availability=EvaluatorAvailability.IMPLEMENTED,
            )
        return EvaluatorRoute(
            test_family=family,
            evaluator=_EVALUATOR_NAMES[family],
            availability=EvaluatorAvailability.UNASSESSED,
            reason=(
                "The test family is structurally supported, but its evaluator is not "
                "implemented in this delivery. The case must remain UNASSESSED."
            ),
        )

    def routes(self) -> tuple[EvaluatorRoute, ...]:
        return tuple(self.route(family) for family in ALL_TEST_FAMILIES)

    def evaluate(
        self,
        test_family: TestFamily | str,
        payload: Mapping[str, Any],
    ) -> object:
        route = self.route(test_family)
        evaluator = self._evaluators.get(route.test_family)
        if evaluator is None:
            return UnassessedEvaluation(
                test_family=route.test_family,
                evaluator=route.evaluator,
                reason=route.reason or "Evaluator is unavailable.",
            )
        # Technical exceptions deliberately propagate to orchestration.  Turning
        # them into an assessment result could create a false clinical outcome.
        return evaluator.evaluate(payload)

"""Contracts for deterministic test-family selection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TestFamily(StrEnum):
    """The complete Phase 1 test-family catalog.

    This is intentionally a closed enum: adding a seventeenth family requires
    updating selection and routing rather than letting it disappear silently.
    """

    CLINICAL_FAITHFULNESS = "clinical_faithfulness"
    CLINICAL_NOTE_GENERATION = "clinical_note_generation"
    CLINICAL_TASK_PERFORMANCE = "clinical_task_performance"
    MEDICATION_SAFETY = "medication_safety"
    NEGATION_ASSERTION = "negation_assertion"
    OMISSION_DETECTION = "omission_detection"
    ROBUSTNESS = "robustness"
    DEMOGRAPHIC_BIAS = "demographic_bias"
    MEDICAL_COGNITIVE_BIAS = "medical_cognitive_bias"
    SAFETY_REFUSAL = "safety_refusal"
    PROMPT_INJECTION_JAILBREAK = "prompt_injection_jailbreak"
    MODEL_DATA_EXTRACTION = "model_data_extraction"
    RAG_GROUNDING = "rag_grounding"
    RAG_POISONING = "rag_poisoning"
    REGULATORY_POLICY_HARDENING = "regulatory_policy_hardening"
    CUSTOM_BUSINESS_REGRESSION = "custom_business_regression"


ALL_TEST_FAMILIES: tuple[TestFamily, ...] = tuple(TestFamily)


class TestRequirement(StrEnum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class TestSelectionResult(BaseModel):
    """Complete, disjoint classification of the test-family catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_set_version: str = Field(min_length=1)
    required_tests: list[TestFamily]
    recommended_tests: list[TestFamily]
    optional_tests: list[TestFamily]
    not_applicable: list[TestFamily]
    rationale: dict[TestFamily, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_complete_partition(self) -> TestSelectionResult:
        classified = [
            *self.required_tests,
            *self.recommended_tests,
            *self.optional_tests,
            *self.not_applicable,
        ]
        if len(classified) != len(set(classified)):
            raise ValueError("each test family must occur in exactly one selection category")
        if set(classified) != set(ALL_TEST_FAMILIES):
            missing = sorted(set(ALL_TEST_FAMILIES) - set(classified))
            extra = sorted(set(classified) - set(ALL_TEST_FAMILIES))
            raise ValueError(
                f"selection must cover all test families; missing={missing}, extra={extra}"
            )
        return self

    def requirement_for(self, test_family: TestFamily) -> TestRequirement:
        if test_family in self.required_tests:
            return TestRequirement.REQUIRED
        if test_family in self.recommended_tests:
            return TestRequirement.RECOMMENDED
        if test_family in self.optional_tests:
            return TestRequirement.OPTIONAL
        return TestRequirement.NOT_APPLICABLE

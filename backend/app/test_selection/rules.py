"""Explainable application-profile to test-family rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from backend.app.systems import ApplicationProfile, ApplicationType, RiskTier
from backend.app.test_selection.models import (
    ALL_TEST_FAMILIES,
    TestFamily,
    TestRequirement,
    TestSelectionResult,
)

TEST_SELECTION_RULE_SET_VERSION = "application-test-selection@1.0.0"

_PRIORITY = {
    TestRequirement.NOT_APPLICABLE: 0,
    TestRequirement.OPTIONAL: 1,
    TestRequirement.RECOMMENDED: 2,
    TestRequirement.REQUIRED: 3,
}


@dataclass(slots=True)
class _SelectionBuilder:
    requirements: dict[TestFamily, TestRequirement] = field(default_factory=dict)
    rationale: dict[TestFamily, list[str]] = field(default_factory=dict)

    def assign(
        self,
        families: set[TestFamily],
        requirement: TestRequirement,
        reason: str,
    ) -> None:
        for family in families:
            current = self.requirements.get(family, TestRequirement.NOT_APPLICABLE)
            if _PRIORITY[requirement] > _PRIORITY[current]:
                self.requirements[family] = requirement
            self.rationale.setdefault(family, []).append(reason)

    def result(self) -> TestSelectionResult:
        requirements = {
            family: self.requirements.get(family, TestRequirement.NOT_APPLICABLE)
            for family in ALL_TEST_FAMILIES
        }
        rationale = {
            family: tuple(
                self.rationale.get(
                    family,
                    ["No deterministic applicability rule matched this application profile."],
                )
            )
            for family in ALL_TEST_FAMILIES
        }

        def selected(requirement: TestRequirement) -> list[TestFamily]:
            return [family for family in ALL_TEST_FAMILIES if requirements[family] is requirement]

        return TestSelectionResult(
            rule_set_version=TEST_SELECTION_RULE_SET_VERSION,
            required_tests=selected(TestRequirement.REQUIRED),
            recommended_tests=selected(TestRequirement.RECOMMENDED),
            optional_tests=selected(TestRequirement.OPTIONAL),
            not_applicable=selected(TestRequirement.NOT_APPLICABLE),
            rationale=rationale,
        )


class DeterministicTestSelectionEngine:
    """Versioned rules with required > recommended > optional precedence."""

    version = TEST_SELECTION_RULE_SET_VERSION

    _clinical_content_families: ClassVar[set[TestFamily]] = {
        TestFamily.CLINICAL_FAITHFULNESS,
        TestFamily.CLINICAL_NOTE_GENERATION,
        TestFamily.CLINICAL_TASK_PERFORMANCE,
        TestFamily.MEDICATION_SAFETY,
        TestFamily.NEGATION_ASSERTION,
        TestFamily.OMISSION_DETECTION,
    }

    _application_rules: ClassVar[dict[ApplicationType, dict[TestRequirement, set[TestFamily]]]] = {
        ApplicationType.CLINICAL_SCRIBE: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.CLINICAL_NOTE_GENERATION,
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.OMISSION_DETECTION,
                TestFamily.ROBUSTNESS,
                TestFamily.CUSTOM_BUSINESS_REGRESSION,
            },
            TestRequirement.OPTIONAL: {TestFamily.CLINICAL_TASK_PERFORMANCE},
        },
        ApplicationType.CLINICAL_SUMMARIZER: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.CLINICAL_NOTE_GENERATION,
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.OMISSION_DETECTION,
            },
            TestRequirement.RECOMMENDED: {TestFamily.ROBUSTNESS},
            TestRequirement.OPTIONAL: {TestFamily.CLINICAL_TASK_PERFORMANCE},
        },
        ApplicationType.CLINICAL_CHATBOT: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_TASK_PERFORMANCE,
                TestFamily.SAFETY_REFUSAL,
                TestFamily.ROBUSTNESS,
            },
            TestRequirement.RECOMMENDED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.DEMOGRAPHIC_BIAS,
                TestFamily.MEDICAL_COGNITIVE_BIAS,
            },
        },
        ApplicationType.PATIENT_ASSISTANT: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_TASK_PERFORMANCE,
                TestFamily.SAFETY_REFUSAL,
                TestFamily.ROBUSTNESS,
                TestFamily.REGULATORY_POLICY_HARDENING,
            },
            TestRequirement.RECOMMENDED: {
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.DEMOGRAPHIC_BIAS,
                TestFamily.MEDICAL_COGNITIVE_BIAS,
            },
        },
        ApplicationType.RAG_ASSISTANT: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_TASK_PERFORMANCE,
                TestFamily.RAG_GROUNDING,
                TestFamily.RAG_POISONING,
                TestFamily.PROMPT_INJECTION_JAILBREAK,
            },
            TestRequirement.RECOMMENDED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.SAFETY_REFUSAL,
            },
        },
        ApplicationType.CLINICAL_DECISION_SUPPORT: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_TASK_PERFORMANCE,
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.SAFETY_REFUSAL,
                TestFamily.MEDICAL_COGNITIVE_BIAS,
            },
            TestRequirement.RECOMMENDED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.DEMOGRAPHIC_BIAS,
                TestFamily.ROBUSTNESS,
            },
        },
        ApplicationType.CLINICAL_LETTER_GENERATOR: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_FAITHFULNESS,
                TestFamily.CLINICAL_NOTE_GENERATION,
                TestFamily.MEDICATION_SAFETY,
                TestFamily.NEGATION_ASSERTION,
                TestFamily.OMISSION_DETECTION,
            },
            TestRequirement.RECOMMENDED: {TestFamily.ROBUSTNESS},
        },
        ApplicationType.HEALTHCARE_ADMIN_AI: {
            TestRequirement.REQUIRED: {TestFamily.REGULATORY_POLICY_HARDENING},
            TestRequirement.RECOMMENDED: {
                TestFamily.ROBUSTNESS,
                TestFamily.CUSTOM_BUSINESS_REGRESSION,
            },
            TestRequirement.OPTIONAL: {
                TestFamily.DEMOGRAPHIC_BIAS,
                TestFamily.SAFETY_REFUSAL,
            },
        },
        ApplicationType.AGENTIC_AI: {
            TestRequirement.REQUIRED: {
                TestFamily.CLINICAL_TASK_PERFORMANCE,
                TestFamily.SAFETY_REFUSAL,
                TestFamily.PROMPT_INJECTION_JAILBREAK,
                TestFamily.MODEL_DATA_EXTRACTION,
                TestFamily.REGULATORY_POLICY_HARDENING,
                TestFamily.CUSTOM_BUSINESS_REGRESSION,
            },
            TestRequirement.RECOMMENDED: {TestFamily.ROBUSTNESS},
        },
        ApplicationType.OTHER: {
            TestRequirement.RECOMMENDED: {
                TestFamily.ROBUSTNESS,
                TestFamily.REGULATORY_POLICY_HARDENING,
                TestFamily.CUSTOM_BUSINESS_REGRESSION,
            },
            TestRequirement.OPTIONAL: _clinical_content_families,
        },
    }

    def select(self, profile: ApplicationProfile) -> TestSelectionResult:
        builder = _SelectionBuilder()

        for requirement, families in self._application_rules[profile.application_type].items():
            builder.assign(
                families,
                requirement,
                f"Application type is {profile.application_type.value}.",
            )

        builder.assign(
            {
                TestFamily.ROBUSTNESS,
                TestFamily.REGULATORY_POLICY_HARDENING,
                TestFamily.CUSTOM_BUSINESS_REGRESSION,
            },
            TestRequirement.RECOMMENDED,
            "Baseline assurance applies to every registered application.",
        )

        rag_applicable = (
            profile.uses_rag or profile.application_type is ApplicationType.RAG_ASSISTANT
        )
        if rag_applicable:
            builder.assign(
                {
                    TestFamily.RAG_GROUNDING,
                    TestFamily.RAG_POISONING,
                    TestFamily.PROMPT_INJECTION_JAILBREAK,
                },
                TestRequirement.REQUIRED,
                "The application uses retrieval-augmented generation.",
            )

        if profile.uses_tools:
            builder.assign(
                {
                    TestFamily.PROMPT_INJECTION_JAILBREAK,
                    TestFamily.CUSTOM_BUSINESS_REGRESSION,
                },
                TestRequirement.REQUIRED,
                "Tool use requires instruction-boundary and permission-policy tests.",
            )

        if profile.uses_llm:
            builder.assign(
                {
                    TestFamily.PROMPT_INJECTION_JAILBREAK,
                    TestFamily.MODEL_DATA_EXTRACTION,
                },
                TestRequirement.RECOMMENDED,
                "LLM use introduces prompt-boundary and extraction risks.",
            )

        if profile.patient_facing:
            builder.assign(
                {
                    TestFamily.SAFETY_REFUSAL,
                    TestFamily.CLINICAL_TASK_PERFORMANCE,
                    TestFamily.ROBUSTNESS,
                    TestFamily.REGULATORY_POLICY_HARDENING,
                },
                TestRequirement.REQUIRED,
                "The application is patient-facing.",
            )
            builder.assign(
                {TestFamily.DEMOGRAPHIC_BIAS, TestFamily.MEDICAL_COGNITIVE_BIAS},
                TestRequirement.RECOMMENDED,
                "Patient-facing behavior should be assessed with controlled matched pairs.",
            )
        elif profile.clinician_facing:
            builder.assign(
                {TestFamily.SAFETY_REFUSAL},
                TestRequirement.RECOMMENDED,
                "The application is clinician-facing.",
            )

        if profile.uses_predictive_model:
            builder.assign(
                {TestFamily.DEMOGRAPHIC_BIAS},
                TestRequirement.REQUIRED,
                "Predictive behavior requires controlled demographic matched-pair testing.",
            )
            builder.assign(
                {TestFamily.MEDICAL_COGNITIVE_BIAS},
                TestRequirement.RECOMMENDED,
                "Predictive behavior may be sensitive to framing and anchoring.",
            )

        if profile.risk_tier in {RiskTier.HIGH, RiskTier.CRITICAL}:
            promotable = {
                family
                for family, requirement in builder.requirements.items()
                if requirement is TestRequirement.RECOMMENDED
                and family
                in {
                    TestFamily.CLINICAL_FAITHFULNESS,
                    TestFamily.MEDICATION_SAFETY,
                    TestFamily.NEGATION_ASSERTION,
                    TestFamily.OMISSION_DETECTION,
                    TestFamily.SAFETY_REFUSAL,
                    TestFamily.PROMPT_INJECTION_JAILBREAK,
                    TestFamily.RAG_GROUNDING,
                    TestFamily.RAG_POISONING,
                }
            }
            builder.assign(
                promotable,
                TestRequirement.REQUIRED,
                f"{profile.risk_tier.value} risk promotes applicable safety tests to required.",
            )

        if profile.risk_tier is RiskTier.CRITICAL:
            builder.assign(
                {
                    TestFamily.REGULATORY_POLICY_HARDENING,
                    TestFamily.CUSTOM_BUSINESS_REGRESSION,
                },
                TestRequirement.REQUIRED,
                "CRITICAL risk requires policy and permanent regression gates.",
            )

        return builder.result()


def select_tests(profile: ApplicationProfile) -> TestSelectionResult:
    """Convenience entry point shared by API, web, and CI services."""

    return DeterministicTestSelectionEngine().select(profile)

"""Focused tests for application-to-release assurance services."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from backend.app import test_plans, test_selection
from backend.app.coverage import (
    CoverageReport,
    calculate_coverage,
)
from backend.app.coverage import (
    TestCoverageRecord as CoverageRecord,
)
from backend.app.domain.enums import (
    CoverageStatus,
    ExecutionStatus,
    FindingType,
    ReleaseDecision,
    Severity,
)
from backend.app.domain.models import Finding, TextSpan
from backend.app.evaluators import EvaluatorRouter
from backend.app.evaluators.contract import ContractEvaluationResult
from backend.app.evaluators.external import ExternalEvaluationResult, ExternalRunner
from backend.app.policies import (
    MissingRequiredCoverageAction,
    PolicyOperator,
    ReleasePolicy,
    ReleasePolicyRule,
    apply_release_policy,
)
from backend.app.scoring import (
    BiasPairObservation,
    RobustnessObservation,
    SafetyClassification,
    score_bias,
    score_clinical_faithfulness,
    score_multiple_choice,
    score_robustness,
    score_safety,
)
from backend.app.systems import (
    ApplicationProfile,
    ApplicationType,
    InputType,
    OutputType,
    RiskTier,
)


def _profile(**updates: object) -> ApplicationProfile:
    values: dict[str, object] = {
        "application_id": "sys_123",
        "organization_id": "org_123",
        "name": "Clinical notes",
        "version": "2.1.0",
        "application_type": ApplicationType.CLINICAL_SCRIBE,
        "intended_use": "Create a draft clinical note for clinician review.",
        "input_type": InputType.CLINICAL_TRANSCRIPT,
        "output_type": OutputType.CLINICAL_NOTE,
        "clinician_facing": True,
        "risk_tier": RiskTier.HIGH,
    }
    values.update(updates)
    return ApplicationProfile.model_validate(values)


def _policy(
    *,
    missing_coverage: MissingRequiredCoverageAction = MissingRequiredCoverageAction.NO_DECISION,
) -> ReleasePolicy:
    return ReleasePolicy(
        policy_id="clinical-scribe-release",
        policy_version="1.0.0",
        missing_required_coverage_action=missing_coverage,
        rules=[
            ReleasePolicyRule(
                rule_id="supported-rate",
                metric="supported_claim_rate",
                operator=PolicyOperator.GREATER_THAN_OR_EQUAL,
                threshold=97,
                on_failure=ReleaseDecision.WARN,
            )
        ],
    )


def _full_coverage() -> CoverageReport:
    return calculate_coverage(
        [
            CoverageRecord(
                test_family=test_selection.TestFamily.CLINICAL_FAITHFULNESS,
                requirement=test_selection.TestRequirement.REQUIRED,
                planned_cases=10,
                completed_cases=10,
            )
        ]
    )


def test_application_profile_is_strict_and_requires_ordered_aware_timestamps() -> None:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        _profile(created_at=created, updated_at=created - timedelta(seconds=1))
    with pytest.raises(ValidationError):
        _profile(unknown_capability=True)


def test_selection_is_deterministic_complete_and_flag_aware() -> None:
    profile = _profile(uses_rag=True, uses_tools=True)
    first = test_selection.select_tests(profile)
    second = test_selection.select_tests(profile)

    assert first == second
    classified = [
        *first.required_tests,
        *first.recommended_tests,
        *first.optional_tests,
        *first.not_applicable,
    ]
    assert len(classified) == 16
    assert set(classified) == set(test_selection.TestFamily)
    assert test_selection.TestFamily.RAG_GROUNDING in first.required_tests
    assert test_selection.TestFamily.RAG_POISONING in first.required_tests
    assert test_selection.TestFamily.PROMPT_INJECTION_JAILBREAK in first.required_tests

    without_rag = test_selection.select_tests(_profile(uses_rag=False))
    assert test_selection.TestFamily.RAG_GROUNDING in without_rag.not_applicable
    assert test_selection.TestFamily.RAG_POISONING in without_rag.not_applicable


def test_test_plan_is_versioned_and_requires_explicit_confirmation() -> None:
    profile = _profile()
    selection = test_selection.select_tests(profile)
    release_policy = test_plans.ReleasePolicyReference(
        policy_id="clinical-scribe-release",
        policy_version="1.0.0",
    )
    draft = test_plans.generate_test_plan(
        profile=profile,
        selection=selection,
        release_policy=release_policy,
        configuration=test_plans.CustomerTestConfiguration(
            case_counts={test_selection.TestFamily.CLINICAL_FAITHFULNESS: 125}
        ),
        test_plan_version="1.2",
    )

    assert draft.status is test_plans.TestPlanStatus.DRAFT
    assert draft.confirmed_at is None
    assert draft.test_plan_version == "1.2"
    assert draft.required[0].cases == 125
    assert (
        len(draft.required)
        + len(draft.recommended)
        + len(draft.optional)
        + len(draft.unselected_optional)
        + len(draft.not_applicable)
        == 16
    )

    confirmed = test_plans.confirm_test_plan(draft, confirmed_by="admin_7")
    assert confirmed.status is test_plans.TestPlanStatus.CONFIRMED
    assert confirmed.confirmed_by == "admin_7"
    assert draft.status is test_plans.TestPlanStatus.DRAFT


def test_test_plan_identity_includes_the_actual_selection_partition() -> None:
    profile = _profile()
    selection = test_selection.select_tests(profile)
    release_policy = test_plans.ReleasePolicyReference(
        policy_id="clinical-scribe-release",
        policy_version="1.0.0",
    )
    changed_partition = selection.model_dump()
    changed_partition["optional_tests"] = [
        *selection.optional_tests,
        test_selection.TestFamily.RAG_GROUNDING,
    ]
    changed_partition["not_applicable"] = [
        family
        for family in selection.not_applicable
        if family is not test_selection.TestFamily.RAG_GROUNDING
    ]
    changed_selection = test_selection.TestSelectionResult.model_validate(changed_partition)

    original = test_plans.generate_test_plan(
        profile=profile,
        selection=selection,
        release_policy=release_policy,
    )
    changed = test_plans.generate_test_plan(
        profile=profile,
        selection=changed_selection,
        release_policy=release_policy,
    )

    assert changed.selection_rule_set_version == original.selection_rule_set_version
    assert changed.plan_id != original.plan_id


def test_test_plan_identity_includes_policy_and_suite_content_snapshots() -> None:
    profile = _profile()
    selection = test_selection.select_tests(profile)
    release_policy = test_plans.ReleasePolicyReference(
        policy_id="clinical-scribe-release",
        policy_version="1.0.0",
    )
    binding = test_plans.SuiteCaseBinding(
        case_version_id="case_version_1",
        test_family=test_selection.TestFamily.CLINICAL_FAITHFULNESS,
    )

    def generate(policy_marker: str, suite_marker: str) -> test_plans.TestPlan:
        return test_plans.generate_test_plan(
            profile=profile,
            selection=selection,
            release_policy=release_policy,
            release_policy_snapshot_digest=f"sha256:{policy_marker * 64}",
            test_suite_id="suite_123",
            test_suite_version="1.0.0",
            test_suite_content_snapshot_digest=f"sha256:{suite_marker * 64}",
            source_suite_case_version_ids=("case_version_1",),
            suite_case_version_ids=("case_version_1",),
            suite_case_bindings=(binding,),
        )

    original = generate("1", "2")
    changed_policy = generate("3", "2")
    changed_suite = generate("1", "4")

    assert len({original.plan_id, changed_policy.plan_id, changed_suite.plan_id}) == 3


def test_evaluator_router_enumerates_every_family_with_executable_contracts() -> None:
    router = EvaluatorRouter()
    routes = router.routes()

    assert len(routes) == 16
    assert {route.test_family for route in routes} == set(test_selection.TestFamily)
    assert all(route.implemented for route in routes)

    result = cast(
        ContractEvaluationResult,
        router.evaluate(
            test_selection.TestFamily.ROBUSTNESS,
            {
                "variant_outputs": ["safe answer", "safe answer"],
                "expectation": {"require_same_output": True},
                "generated_output": "safe answer",
            },
        ),
    )
    assert result.assessment_status == "ASSESSED"
    assert result.coverage.status == "FULL"


def test_contract_evaluator_keeps_missing_family_evidence_unassessed() -> None:
    router = EvaluatorRouter()
    with pytest.raises(ValueError, match="robustness requires variant_outputs"):
        router.evaluate(
            test_selection.TestFamily.ROBUSTNESS,
            {"generated_output": "safe answer", "expectation": {"require_same_output": True}},
        )


def test_every_non_faithfulness_family_has_a_working_contract_path() -> None:
    router = EvaluatorRouter()
    for family in test_selection.ALL_TEST_FAMILIES:
        if family is test_selection.TestFamily.CLINICAL_FAITHFULNESS:
            continue
        payload: dict[str, object] = {
            "generated_output": "safe answer",
            "expectation": {"must_include": ["safe answer"]},
            "input_data": {"context": "synthetic"},
        }
        if family is test_selection.TestFamily.ROBUSTNESS:
            payload["variant_outputs"] = ["safe answer", "safe answer"]
            payload["expectation"] = {"require_same_output": True}
        elif family is test_selection.TestFamily.DEMOGRAPHIC_BIAS:
            payload["paired_outputs"] = [
                {"group": "A", "output": "safe answer"},
                {"group": "B", "output": "safe answer"},
            ]
            payload["expectation"] = {"require_equivalent": True}
        result = cast(ContractEvaluationResult, router.evaluate(family, payload))
        assert result.assessment_status == "ASSESSED", family
        assert result.coverage.status == "FULL", family


def test_external_runners_enable_medhelm_and_langtest_routes() -> None:
    def runner(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        assert payload["case_id"] == "case-1"
        return {"score": 92.0, "passed": True}

    router = EvaluatorRouter(
        medhelm_runner=runner,
        langtest_runner=runner,
        use_environment_runners=False,
    )

    medhelm_route = router.route(test_selection.TestFamily.CLINICAL_TASK_PERFORMANCE)
    langtest_route = router.route(test_selection.TestFamily.ROBUSTNESS)
    assert medhelm_route.implemented
    assert medhelm_route.evaluator == "medhelm"
    assert langtest_route.implemented
    assert langtest_route.evaluator == "langtest"

    medhelm_result = router.evaluate(
        test_selection.TestFamily.CLINICAL_TASK_PERFORMANCE,
        {"case_id": "case-1", "generated_output": "safe answer"},
    )
    assert isinstance(medhelm_result, ExternalEvaluationResult)
    assert medhelm_result.assessment_status == "ASSESSED"
    assert medhelm_result.coverage.status == "FULL"
    assert medhelm_result.score == 92.0


def test_external_evaluator_fails_closed_on_partial_coverage() -> None:
    def partial_runner(_: Mapping[str, Any]) -> dict[str, Any]:
        return {"score": 40.0, "coverage": {"status": "REDUCED"}}

    router = EvaluatorRouter(
        langtest_runner=cast(ExternalRunner, partial_runner),
        use_environment_runners=False,
    )
    with pytest.raises(Exception, match="external evaluator returned an invalid result"):
        router.evaluate(test_selection.TestFamily.ROBUSTNESS, {})


def test_scoring_uses_distinct_family_metrics_and_preserves_unassessed_values() -> None:
    findings = [
        Finding(
            finding_id="supported",
            type=FindingType.SUPPORTED,
            category="symptom",
            severity=Severity.LOW,
            source=TextSpan(text="pain", start=0, end=4),
            generated=TextSpan(text="pain", start=0, end=4),
            reason_code="CLAIM_SUPPORTED",
            explanation="Supported.",
            confidence=1,
        ),
        Finding(
            finding_id="critical",
            type=FindingType.ALTERED,
            category="medication_dose",
            severity=Severity.CRITICAL,
            source=TextSpan(text="10", start=0, end=2),
            generated=TextSpan(text="100", start=0, end=3),
            reason_code="DOSE_MISMATCH",
            explanation="Dose changed.",
            confidence=1,
        ),
    ]
    faithfulness = score_clinical_faithfulness(findings)
    assert faithfulness.supported_claim_rate == 50
    assert faithfulness.altered_claim_rate == 50
    assert faithfulness.critical_count == 1

    mcq = score_multiple_choice([True, False, None, True])
    assert mcq.accuracy == pytest.approx(66.6667)
    assert mcq.unassessed_count == 1

    robustness = score_robustness(
        [
            RobustnessObservation(perturbation_type="typo", consistent=True),
            RobustnessObservation(perturbation_type="typo", consistent=False),
            RobustnessObservation(perturbation_type="ocr", consistent=None),
        ]
    )
    assert robustness.consistency_rate == 50
    assert robustness.failure_rate_by_perturbation_type == {"ocr": None, "typo": 50}

    bias = score_bias(
        [
            BiasPairObservation(pair_id="1", agrees=False),
            BiasPairObservation(pair_id="2", agrees=False, difference_clinically_justified=True),
        ]
    )
    assert bias.disagreement_count == 2
    assert bias.unjustified_disagreement_count == 1

    safety = score_safety(
        [
            SafetyClassification.SAFE_REFUSAL,
            SafetyClassification.UNSAFE_COMPLIANCE,
            None,
        ]
    )
    assert safety.safe_response_rate == 50
    assert safety.unsafe_compliance_rate == 50
    assert safety.unassessed_cases == 1


def test_coverage_is_full_reduced_partial_or_unassessed_and_fails_closed() -> None:
    full = _full_coverage()
    assert full.status is CoverageStatus.FULL
    assert full.valid_for_release

    reduced = calculate_coverage(
        [
            CoverageRecord(
                test_family=test_selection.TestFamily.ROBUSTNESS,
                requirement=test_selection.TestRequirement.RECOMMENDED,
                planned_cases=2,
                completed_cases=1,
                unassessed_cases=1,
            )
        ]
    )
    assert reduced.status is CoverageStatus.REDUCED
    assert reduced.valid_for_release

    partial = calculate_coverage(
        [
            CoverageRecord(
                test_family=test_selection.TestFamily.SAFETY_REFUSAL,
                requirement=test_selection.TestRequirement.REQUIRED,
                planned_cases=2,
                completed_cases=1,
                technical_failed_cases=1,
            )
        ]
    )
    assert partial.status is CoverageStatus.PARTIAL
    assert not partial.valid_for_release
    assert partial.technical_failed_cases == 1

    unassessed = calculate_coverage([])
    assert unassessed.status is CoverageStatus.UNASSESSED
    assert not unassessed.valid_for_release

    wholly_missing_required = calculate_coverage(
        [
            CoverageRecord(
                test_family=test_selection.TestFamily.MEDICATION_SAFETY,
                requirement=test_selection.TestRequirement.REQUIRED,
                planned_cases=0,
            )
        ]
    )
    assert wholly_missing_required.status is CoverageStatus.UNASSESSED
    assert wholly_missing_required.missing_required_tests == [
        test_selection.TestFamily.MEDICATION_SAFETY
    ]
    assert not wholly_missing_required.valid_for_release


@pytest.mark.parametrize(
    "updates",
    [
        {"completed_cases": 11},
        {"status": "REDUCED"},
        {"completed_cases": 9, "unassessed_cases": 1},
        {"valid_for_release": False},
        {"valid_for_release": 1},
        {"planned_cases": True},
    ],
)
def test_coverage_report_rejects_inconsistent_release_snapshots(
    updates: dict[str, object],
) -> None:
    payload = _full_coverage().model_dump(mode="json")
    payload.update(updates)

    with pytest.raises(ValidationError):
        CoverageReport.model_validate(payload)


def test_release_policy_separates_failure_from_decision_and_critical_override() -> None:
    coverage = _full_coverage()
    passed = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 99},
        coverage=coverage,
    )
    assert passed.release_decision is ReleaseDecision.PASS

    warned = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 95},
        coverage=coverage,
    )
    assert warned.release_decision is ReleaseDecision.WARN

    blocked = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 99},
        coverage=coverage,
        critical_findings=1,
    )
    assert blocked.release_decision is ReleaseDecision.BLOCK
    assert blocked.critical_finding_override_applied

    failed = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.FAILED,
        metrics={"supported_claim_rate": 99},
        coverage=coverage,
        critical_findings=1,
    )
    assert failed.release_decision is None


@pytest.mark.parametrize("operator", list(PolicyOperator))
@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), True])
def test_release_policy_rejects_non_finite_thresholds(
    operator: PolicyOperator, threshold: float | bool
) -> None:
    with pytest.raises(ValidationError):
        ReleasePolicyRule(
            rule_id="finite-threshold",
            metric="supported_claim_rate",
            operator=operator,
            threshold=threshold,
            on_failure=ReleaseDecision.WARN,
        )


@pytest.mark.parametrize("operator", list(PolicyOperator))
@pytest.mark.parametrize("actual", [float("nan"), float("inf"), float("-inf"), True])
def test_release_policy_withholds_decision_for_non_finite_runtime_metrics(
    operator: PolicyOperator, actual: float | bool
) -> None:
    policy = ReleasePolicy(
        policy_id="finite-runtime-metrics",
        policy_version="1.0.0",
        rules=[
            ReleasePolicyRule(
                rule_id="finite-actual",
                metric="metric",
                operator=operator,
                threshold=1,
                on_failure=ReleaseDecision.WARN,
            )
        ],
    )

    result = apply_release_policy(
        policy=policy,
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"metric": actual},
        coverage=_full_coverage(),
    )

    assert result.release_decision is None
    assert "POLICY_METRIC_INVALID_NO_DECISION" in result.reason_codes
    assert result.rule_results[0].actual is None


def test_release_policy_never_promotes_non_release_valid_reduced_coverage() -> None:
    reduced = CoverageReport(
        status=CoverageStatus.REDUCED,
        planned_tests=1,
        planned_cases=2,
        completed_cases=1,
        technical_failed_cases=0,
        unassessed_cases=1,
        pending_cases=0,
        missing_required_tests=[],
        degraded_tests=[],
        valid_for_release=False,
        limitations=["The optional evaluator did not assess one case."],
    )

    result = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 99},
        coverage=reduced,
    )

    assert result.release_decision is None
    assert result.reason_codes == ["COVERAGE_NOT_RELEASE_VALID_NO_DECISION"]


@pytest.mark.parametrize("critical_findings", [-1, True, 1.5])
def test_release_policy_rejects_invalid_critical_finding_counts(
    critical_findings: object,
) -> None:
    with pytest.raises(ValueError, match="critical_findings"):
        apply_release_policy(
            policy=_policy(),
            execution_status=ExecutionStatus.COMPLETED,
            metrics={"supported_claim_rate": 99},
            coverage=_full_coverage(),
            critical_findings=critical_findings,  # type: ignore[arg-type]
        )


def test_missing_required_coverage_can_withhold_or_block_by_policy() -> None:
    incomplete = calculate_coverage(
        [
            CoverageRecord(
                test_family=test_selection.TestFamily.SAFETY_REFUSAL,
                requirement=test_selection.TestRequirement.REQUIRED,
                planned_cases=2,
                completed_cases=1,
                unassessed_cases=1,
            )
        ]
    )
    no_decision = apply_release_policy(
        policy=_policy(),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 99},
        coverage=incomplete,
    )
    assert no_decision.release_decision is None

    blocked = apply_release_policy(
        policy=_policy(missing_coverage=MissingRequiredCoverageAction.BLOCK),
        execution_status=ExecutionStatus.COMPLETED,
        metrics={"supported_claim_rate": 99},
        coverage=incomplete,
    )
    assert blocked.release_decision is ReleaseDecision.BLOCK

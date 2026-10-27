"""Generate reproducible drafts and require an explicit confirmation step."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final

from backend.app.systems import ApplicationProfile
from backend.app.test_plans.models import (
    CustomerTestConfiguration,
    PlannedTest,
    ReleasePolicyReference,
    SuiteCaseBinding,
    SuiteCaseExclusion,
    TestPlan,
    TestPlanStatus,
)
from backend.app.test_selection import TestFamily, TestSelectionResult
from backend.app.test_suites import case_version_snapshot_digest

TEST_PLAN_GENERATOR_VERSION: Final = "test-plan-generator@1.0.0"
DEFAULT_TEST_PLAN_VERSION: Final = "1.0"

DEFAULT_CASE_COUNTS: Final[dict[TestFamily, int]] = {
    TestFamily.CLINICAL_FAITHFULNESS: 100,
    TestFamily.CLINICAL_NOTE_GENERATION: 50,
    TestFamily.CLINICAL_TASK_PERFORMANCE: 50,
    TestFamily.MEDICATION_SAFETY: 30,
    TestFamily.NEGATION_ASSERTION: 30,
    TestFamily.OMISSION_DETECTION: 30,
    TestFamily.ROBUSTNESS: 30,
    TestFamily.DEMOGRAPHIC_BIAS: 30,
    TestFamily.MEDICAL_COGNITIVE_BIAS: 20,
    TestFamily.SAFETY_REFUSAL: 30,
    TestFamily.PROMPT_INJECTION_JAILBREAK: 30,
    TestFamily.MODEL_DATA_EXTRACTION: 20,
    TestFamily.RAG_GROUNDING: 50,
    TestFamily.RAG_POISONING: 20,
    TestFamily.REGULATORY_POLICY_HARDENING: 20,
    TestFamily.CUSTOM_BUSINESS_REGRESSION: 20,
}


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_fingerprint(profile: ApplicationProfile) -> str:
    profile_data = profile.model_dump(mode="json", exclude={"created_at", "updated_at"})
    return f"sha256:{_canonical_digest(profile_data)}"


class TestPlanGenerator:
    version = TEST_PLAN_GENERATOR_VERSION

    def generate(
        self,
        *,
        profile: ApplicationProfile,
        selection: TestSelectionResult,
        release_policy: ReleasePolicyReference,
        release_policy_snapshot_digest: str | None = None,
        test_suite_id: str | None = None,
        test_suite_version: str | None = None,
        test_suite_content_snapshot_digest: str | None = None,
        source_suite_case_version_ids: Sequence[str] = (),
        suite_case_version_ids: Sequence[str] = (),
        suite_case_bindings: Sequence[SuiteCaseBinding] = (),
        excluded_suite_cases: Sequence[SuiteCaseExclusion] = (),
        configuration: CustomerTestConfiguration | None = None,
        test_plan_version: str = DEFAULT_TEST_PLAN_VERSION,
        confirmed_by: str | None = None,
        now: datetime | None = None,
    ) -> TestPlan:
        """Create a draft unless a confirmation actor is explicitly supplied."""

        config = configuration or CustomerTestConfiguration()
        invalid_optional = config.include_optional - set(selection.optional_tests)
        if invalid_optional:
            names = sorted(family.value for family in invalid_optional)
            raise ValueError(f"only tests classified optional can be selected: {names}")

        configurable = {
            *selection.required_tests,
            *selection.recommended_tests,
            *config.include_optional,
        }
        invalid_overrides = set(config.case_counts) - configurable
        if invalid_overrides:
            names = sorted(family.value for family in invalid_overrides)
            raise ValueError(
                f"case-count overrides require a selected applicable test family: {names}"
            )

        def planned(families: list[TestFamily]) -> list[PlannedTest]:
            return [
                PlannedTest(
                    test_family=family,
                    cases=config.case_counts.get(family, DEFAULT_CASE_COUNTS[family]),
                )
                for family in families
            ]

        selected_optional = [
            family for family in selection.optional_tests if family in config.include_optional
        ]
        unselected_optional = [
            family for family in selection.optional_tests if family not in config.include_optional
        ]
        required = planned(selection.required_tests)
        recommended = planned(selection.recommended_tests)
        optional = planned(selected_optional)
        source_case_ids = tuple(source_suite_case_version_ids)
        executable_case_ids = tuple(suite_case_version_ids)
        executable_bindings = tuple(suite_case_bindings)
        exclusions = tuple(excluded_suite_cases)
        source_snapshot_digest = (
            case_version_snapshot_digest(source_case_ids) if test_suite_id is not None else None
        )
        executable_snapshot_digest = (
            case_version_snapshot_digest(executable_case_ids) if test_suite_id is not None else None
        )
        policy_snapshot_digest = release_policy_snapshot_digest or (
            f"sha256:{_canonical_digest(release_policy.model_dump(mode='json'))}"
        )
        suite_content_digest = test_suite_content_snapshot_digest
        if test_suite_id is not None and suite_content_digest is None:
            suite_content_digest = f"sha256:{
                _canonical_digest(
                    {
                        'test_suite_id': test_suite_id,
                        'test_suite_version': test_suite_version,
                        'source_case_ids': source_case_ids,
                    }
                )
            }"
        created_at = now or datetime.now(UTC)
        plan_identity = {
            "application_id": profile.application_id,
            "application_version": profile.version,
            "application_environment": profile.environment,
            "test_plan_version": test_plan_version,
            "selection_rule_set_version": selection.rule_set_version,
            "release_policy": release_policy.model_dump(mode="json"),
            "release_policy_snapshot_digest": policy_snapshot_digest,
            "test_suite_id": test_suite_id,
            "test_suite_version": test_suite_version,
            "test_suite_content_snapshot_digest": suite_content_digest,
            "source_suite_case_version_ids": source_case_ids,
            "source_suite_case_snapshot_digest": source_snapshot_digest,
            "suite_case_version_ids": executable_case_ids,
            "suite_case_snapshot_digest": executable_snapshot_digest,
            "suite_case_bindings": [item.model_dump(mode="json") for item in executable_bindings],
            "excluded_suite_cases": [item.model_dump(mode="json") for item in exclusions],
            "profile_fingerprint": _profile_fingerprint(profile),
            "required": [item.model_dump(mode="json") for item in required],
            "recommended": [item.model_dump(mode="json") for item in recommended],
            "optional": [item.model_dump(mode="json") for item in optional],
            "unselected_optional": unselected_optional,
            "not_applicable": selection.not_applicable,
        }
        plan = TestPlan(
            plan_id=f"plan_{_canonical_digest(plan_identity)[:20]}",
            application_id=profile.application_id,
            organization_id=profile.organization_id,
            application_version=profile.version,
            application_environment=profile.environment,
            test_plan_version=test_plan_version,
            selection_rule_set_version=selection.rule_set_version,
            release_policy=release_policy,
            release_policy_snapshot_digest=policy_snapshot_digest,
            test_suite_id=test_suite_id,
            test_suite_version=test_suite_version,
            test_suite_content_snapshot_digest=suite_content_digest,
            source_suite_case_version_ids=source_case_ids,
            source_suite_case_snapshot_digest=source_snapshot_digest,
            suite_case_version_ids=executable_case_ids,
            suite_case_snapshot_digest=executable_snapshot_digest,
            suite_case_bindings=executable_bindings,
            excluded_suite_cases=exclusions,
            risk_tier=profile.risk_tier,
            profile_fingerprint=_profile_fingerprint(profile),
            required=required,
            recommended=recommended,
            optional=optional,
            unselected_optional=unselected_optional,
            not_applicable=selection.not_applicable,
            created_at=created_at,
        )
        if confirmed_by is None:
            return plan
        return confirm_test_plan(plan, confirmed_by=confirmed_by, confirmed_at=created_at)


def confirm_test_plan(
    plan: TestPlan,
    *,
    confirmed_by: str,
    confirmed_at: datetime | None = None,
) -> TestPlan:
    """Return a confirmed artifact without mutating the audit-visible draft."""

    if plan.status is TestPlanStatus.CONFIRMED:
        if plan.confirmed_by != confirmed_by:
            raise ValueError("a confirmed plan cannot be reconfirmed by a different actor")
        return plan
    if not confirmed_by.strip():
        raise ValueError("confirmed_by must identify a user or administrator")
    timestamp = confirmed_at or datetime.now(UTC)
    data = plan.model_dump()
    data.update(
        status=TestPlanStatus.CONFIRMED,
        confirmed_by=confirmed_by,
        confirmed_at=timestamp,
    )
    return TestPlan.model_validate(data)


def generate_test_plan(
    *,
    profile: ApplicationProfile,
    selection: TestSelectionResult,
    release_policy: ReleasePolicyReference,
    release_policy_snapshot_digest: str | None = None,
    test_suite_id: str | None = None,
    test_suite_version: str | None = None,
    test_suite_content_snapshot_digest: str | None = None,
    source_suite_case_version_ids: Sequence[str] = (),
    suite_case_version_ids: Sequence[str] = (),
    suite_case_bindings: Sequence[SuiteCaseBinding] = (),
    excluded_suite_cases: Sequence[SuiteCaseExclusion] = (),
    configuration: CustomerTestConfiguration | None = None,
    test_plan_version: str = DEFAULT_TEST_PLAN_VERSION,
    confirmed_by: str | None = None,
) -> TestPlan:
    return TestPlanGenerator().generate(
        profile=profile,
        selection=selection,
        release_policy=release_policy,
        release_policy_snapshot_digest=release_policy_snapshot_digest,
        test_suite_id=test_suite_id,
        test_suite_version=test_suite_version,
        test_suite_content_snapshot_digest=test_suite_content_snapshot_digest,
        source_suite_case_version_ids=source_suite_case_version_ids,
        suite_case_version_ids=suite_case_version_ids,
        suite_case_bindings=suite_case_bindings,
        excluded_suite_cases=excluded_suite_cases,
        configuration=configuration,
        test_plan_version=test_plan_version,
        confirmed_by=confirmed_by,
    )

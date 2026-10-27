"""Versioned and explicitly confirmed assurance test plans."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from backend.app.systems import ApplicationEnvironment, RiskTier
from backend.app.test_selection import TestFamily
from backend.app.test_suites import case_version_snapshot_digest


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TestPlanStatus(StrEnum):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"


class SuiteCaseExclusionReason(StrEnum):
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_SELECTED = "NOT_SELECTED"
    INACTIVE = "INACTIVE"


class SuiteCaseExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_version_id: str = Field(min_length=1)
    test_family: TestFamily
    reason: SuiteCaseExclusionReason


class SuiteCaseBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_version_id: str = Field(min_length=1)
    test_family: TestFamily


class PlannedTest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test_family: TestFamily
    cases: int = Field(ge=1)


class ReleasePolicyReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)


class CustomerTestConfiguration(BaseModel):
    """Customer overrides that cannot remove tests selected as required."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_counts: dict[TestFamily, int] = Field(default_factory=dict)
    include_optional: set[TestFamily] = Field(default_factory=set)

    @model_validator(mode="after")
    def validate_case_counts(self) -> CustomerTestConfiguration:
        invalid = [family.value for family, count in self.case_counts.items() if count < 1]
        if invalid:
            raise ValueError(f"case counts must be at least one: {sorted(invalid)}")
        return self


class TestPlan(BaseModel):
    """Immutable plan artifact; confirmation creates a new model value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    application_version: str = Field(min_length=1)
    application_environment: ApplicationEnvironment
    test_plan_version: str = Field(min_length=1)
    selection_rule_set_version: str = Field(min_length=1)
    release_policy: ReleasePolicyReference
    release_policy_snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    test_suite_id: str | None = Field(default=None, min_length=1)
    test_suite_version: str | None = Field(default=None, min_length=1)
    test_suite_content_snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    source_suite_case_version_ids: tuple[str, ...] = ()
    source_suite_case_snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    suite_case_version_ids: tuple[str, ...] = ()
    suite_case_snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    suite_case_bindings: tuple[SuiteCaseBinding, ...] = ()
    excluded_suite_cases: tuple[SuiteCaseExclusion, ...] = ()
    risk_tier: RiskTier
    profile_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    required: list[PlannedTest]
    recommended: list[PlannedTest]
    optional: list[PlannedTest]
    unselected_optional: list[TestFamily]
    not_applicable: list[TestFamily]
    status: TestPlanStatus = TestPlanStatus.DRAFT
    created_at: AwareDatetime = Field(default_factory=_utc_now)
    confirmed_at: AwareDatetime | None = None
    confirmed_by: str | None = None

    @model_validator(mode="after")
    def validate_confirmation_and_partition(self) -> TestPlan:
        if (self.test_suite_id is None) != (self.test_suite_version is None):
            raise ValueError("test-suite ID and version must be supplied together")
        has_suite = self.test_suite_id is not None
        if not has_suite:
            if (
                self.source_suite_case_version_ids
                or self.source_suite_case_snapshot_digest is not None
                or self.test_suite_content_snapshot_digest is not None
                or self.suite_case_version_ids
                or self.suite_case_snapshot_digest is not None
                or self.suite_case_bindings
                or self.excluded_suite_cases
            ):
                raise ValueError("suite case snapshots require a test-suite reference")
        else:
            if self.test_suite_content_snapshot_digest is None:
                raise ValueError("suite references require a content snapshot digest")
            if self.source_suite_case_snapshot_digest is None:
                raise ValueError("suite references require a source case snapshot digest")
            if self.suite_case_snapshot_digest is None:
                raise ValueError("suite references require an executable case snapshot digest")
            if self.source_suite_case_snapshot_digest != case_version_snapshot_digest(
                self.source_suite_case_version_ids
            ):
                raise ValueError("source suite case snapshot digest does not match its IDs")
            if self.suite_case_snapshot_digest != case_version_snapshot_digest(
                self.suite_case_version_ids
            ):
                raise ValueError("executable suite case snapshot digest does not match its IDs")

            source_ids = set(self.source_suite_case_version_ids)
            executable_ids = set(self.suite_case_version_ids)
            excluded_ids = [item.case_version_id for item in self.excluded_suite_cases]
            binding_ids = tuple(item.case_version_id for item in self.suite_case_bindings)
            if len(source_ids) != len(self.source_suite_case_version_ids):
                raise ValueError("source suite case snapshot IDs must be unique")
            if len(executable_ids) != len(self.suite_case_version_ids):
                raise ValueError("executable suite case snapshot IDs must be unique")
            if len(excluded_ids) != len(set(excluded_ids)):
                raise ValueError("excluded suite case snapshot IDs must be unique")
            if binding_ids != self.suite_case_version_ids:
                raise ValueError("executable suite case bindings must match the ordered IDs")
            if executable_ids & set(excluded_ids):
                raise ValueError("a suite case cannot be both executable and excluded")
            if executable_ids | set(excluded_ids) != source_ids:
                raise ValueError("every source suite case must be executable or excluded")
        if self.status is TestPlanStatus.CONFIRMED:
            if self.confirmed_at is None or not self.confirmed_by:
                raise ValueError("confirmed plans require confirmed_at and confirmed_by")
        elif self.confirmed_at is not None or self.confirmed_by is not None:
            raise ValueError("draft plans cannot contain confirmation metadata")

        planned = [
            *(entry.test_family for entry in self.required),
            *(entry.test_family for entry in self.recommended),
            *(entry.test_family for entry in self.optional),
            *self.unselected_optional,
            *self.not_applicable,
        ]
        if len(planned) != len(set(planned)):
            raise ValueError("a test family cannot appear in more than one plan category")
        if set(planned) != set(TestFamily):
            raise ValueError("a test plan must retain the complete test-family catalog")

        selected = {
            *(entry.test_family for entry in self.required),
            *(entry.test_family for entry in self.recommended),
            *(entry.test_family for entry in self.optional),
        }
        if any(binding.test_family not in selected for binding in self.suite_case_bindings):
            raise ValueError("executable suite case bindings must target selected tests")
        for exclusion in self.excluded_suite_cases:
            if (
                exclusion.reason is SuiteCaseExclusionReason.NOT_APPLICABLE
                and exclusion.test_family not in self.not_applicable
            ):
                raise ValueError("NOT_APPLICABLE exclusions must match the plan partition")
            if (
                exclusion.reason is SuiteCaseExclusionReason.NOT_SELECTED
                and exclusion.test_family not in self.unselected_optional
            ):
                raise ValueError("NOT_SELECTED exclusions must be unselected optional tests")
            if (
                exclusion.reason is SuiteCaseExclusionReason.QUOTA_EXCEEDED
                and exclusion.test_family not in selected
            ):
                raise ValueError("QUOTA_EXCEEDED exclusions must target selected tests")
        return self

"""Coverage aggregation that fails closed for required or technical gaps."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.domain.enums import CoverageStatus
from backend.app.test_selection import TestFamily, TestRequirement

COVERAGE_ENGINE_VERSION = "coverage-engine@1.0.0"


class TestCoverageRecord(BaseModel):
    """Case counts for one planned test family.

    ``completed_cases`` means an evaluator produced an assessment, regardless
    of whether the system under test passed or failed clinically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_family: TestFamily
    requirement: TestRequirement
    planned_cases: int = Field(ge=0)
    completed_cases: int = Field(default=0, ge=0)
    technical_failed_cases: int = Field(default=0, ge=0)
    unassessed_cases: int = Field(default=0, ge=0)
    evaluator_degraded: bool = False
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> TestCoverageRecord:
        accounted = self.completed_cases + self.technical_failed_cases + self.unassessed_cases
        if accounted > self.planned_cases:
            raise ValueError("completed, technically failed, and unassessed cases exceed plan")
        if self.requirement is TestRequirement.NOT_APPLICABLE:
            raise ValueError("not-applicable test families cannot have planned coverage")
        return self

    @property
    def pending_cases(self) -> int:
        return (
            self.planned_cases
            - self.completed_cases
            - self.technical_failed_cases
            - self.unassessed_cases
        )

    @property
    def has_gap(self) -> bool:
        return (
            self.planned_cases == 0
            or self.completed_cases != self.planned_cases
            or self.evaluator_degraded
        )


class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_version: str = COVERAGE_ENGINE_VERSION
    status: CoverageStatus
    planned_tests: int = Field(ge=0, strict=True)
    planned_cases: int = Field(ge=0, strict=True)
    completed_cases: int = Field(ge=0, strict=True)
    technical_failed_cases: int = Field(ge=0, strict=True)
    unassessed_cases: int = Field(ge=0, strict=True)
    pending_cases: int = Field(ge=0, strict=True)
    missing_required_tests: list[TestFamily]
    degraded_tests: list[TestFamily]
    valid_for_release: bool = Field(strict=True)
    limitations: list[str]

    @model_validator(mode="after")
    def validate_release_invariants(self) -> CoverageReport:
        accounted_cases = (
            self.completed_cases
            + self.technical_failed_cases
            + self.unassessed_cases
            + self.pending_cases
        )
        if accounted_cases != self.planned_cases:
            raise ValueError("coverage case counts must exactly account for planned cases")
        if len(self.missing_required_tests) != len(set(self.missing_required_tests)):
            raise ValueError("missing required test families must be unique")
        if len(self.degraded_tests) != len(set(self.degraded_tests)):
            raise ValueError("degraded test families must be unique")
        if self.planned_tests == 0 and self.planned_cases != 0:
            raise ValueError("coverage cannot contain cases without planned tests")

        if self.status is CoverageStatus.FULL:
            if (
                self.planned_tests == 0
                or self.planned_cases == 0
                or self.completed_cases != self.planned_cases
                or self.technical_failed_cases
                or self.unassessed_cases
                or self.pending_cases
                or self.missing_required_tests
                or self.degraded_tests
            ):
                raise ValueError("FULL coverage requires every planned case to be assessed")
            if not self.valid_for_release:
                raise ValueError("FULL coverage must be release-valid")
        elif self.status is CoverageStatus.REDUCED:
            if self.planned_cases == 0 or self.completed_cases == 0:
                raise ValueError("REDUCED coverage requires at least one assessed case")
            if self.missing_required_tests:
                raise ValueError("REDUCED coverage cannot omit required tests")
            if not (
                self.technical_failed_cases
                or self.unassessed_cases
                or self.pending_cases
                or self.degraded_tests
            ):
                raise ValueError("REDUCED coverage requires an explicit coverage limitation")
        elif self.status is CoverageStatus.PARTIAL:
            if (
                self.planned_cases == 0
                or self.completed_cases == 0
                or not self.missing_required_tests
            ):
                raise ValueError("PARTIAL coverage requires an assessed case and a required gap")
            if self.valid_for_release:
                raise ValueError("PARTIAL coverage cannot be release-valid")
        elif self.status is CoverageStatus.UNASSESSED:
            if self.planned_cases > 0 and self.completed_cases > 0:
                raise ValueError("UNASSESSED coverage cannot contain assessed cases")
            if self.valid_for_release:
                raise ValueError("UNASSESSED coverage cannot be release-valid")

        if self.valid_for_release and (
            self.status not in {CoverageStatus.FULL, CoverageStatus.REDUCED}
            or self.technical_failed_cases
            or self.pending_cases
            or self.missing_required_tests
        ):
            raise ValueError("release-valid coverage cannot contain unresolved execution gaps")
        return self


def calculate_coverage(records: list[TestCoverageRecord]) -> CoverageReport:
    """Aggregate coverage; never infer coverage for a missing case result."""

    families = [record.test_family for record in records]
    if len(families) != len(set(families)):
        raise ValueError("coverage can contain only one record per test family")

    planned_cases = sum(record.planned_cases for record in records)
    completed_cases = sum(record.completed_cases for record in records)
    technical_failed = sum(record.technical_failed_cases for record in records)
    explicit_unassessed = sum(record.unassessed_cases for record in records)
    pending = sum(record.pending_cases for record in records)
    required_gaps = [
        record.test_family
        for record in records
        if record.requirement is TestRequirement.REQUIRED and record.has_gap
    ]
    degraded = [record.test_family for record in records if record.evaluator_degraded]
    any_gap = any(record.has_gap for record in records)

    limitations = [limitation for record in records for limitation in record.limitations]
    if technical_failed:
        limitations.append(
            f"{technical_failed} case(s) failed technically and have no clinical assessment."
        )
    if pending:
        limitations.append(f"{pending} planned case(s) have no terminal result.")
    if required_gaps:
        names = ", ".join(family.value for family in required_gaps)
        limitations.append(f"Required coverage is incomplete for: {names}.")

    if planned_cases == 0 or completed_cases == 0:
        status = CoverageStatus.UNASSESSED
    elif required_gaps:
        status = CoverageStatus.PARTIAL
    elif any_gap:
        status = CoverageStatus.REDUCED
    else:
        status = CoverageStatus.FULL

    # Technical failure is never release-valid, even if it happened in a
    # recommended family.  A policy may BLOCK incomplete required coverage,
    # but it cannot turn a technical failure into a clinical decision.
    valid_for_release = (
        status in {CoverageStatus.FULL, CoverageStatus.REDUCED}
        and not required_gaps
        and technical_failed == 0
        and pending == 0
    )

    return CoverageReport(
        status=status,
        planned_tests=len(records),
        planned_cases=planned_cases,
        completed_cases=completed_cases,
        technical_failed_cases=technical_failed,
        unassessed_cases=explicit_unassessed,
        pending_cases=pending,
        missing_required_tests=required_gaps,
        degraded_tests=degraded,
        valid_for_release=valid_for_release,
        limitations=limitations,
    )

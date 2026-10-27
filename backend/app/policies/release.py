"""Versioned release policy kept separate from test metric calculation."""

from __future__ import annotations

import operator
from collections.abc import Callable
from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.coverage import CoverageReport
from backend.app.domain.enums import CoverageStatus, ExecutionStatus, ReleaseDecision

RELEASE_POLICY_ENGINE_VERSION = "release-policy-engine@1.0.0"


class PolicyOperator(StrEnum):
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="


class PolicyTier(StrEnum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"
    INFORMATIONAL = "INFORMATIONAL"


class MissingRequiredCoverageAction(StrEnum):
    NO_DECISION = "NO_DECISION"
    BLOCK = "BLOCK"


class ReleasePolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    operator: PolicyOperator
    threshold: float = Field(allow_inf_nan=False)
    on_failure: ReleaseDecision
    tier: PolicyTier = PolicyTier.BLOCKING
    description: str = ""

    @field_validator("threshold", mode="before")
    @classmethod
    def reject_boolean_threshold(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("release-policy thresholds must be numeric, not boolean")
        return value

    @model_validator(mode="after")
    def validate_failure_decision(self) -> ReleasePolicyRule:
        if self.on_failure is ReleaseDecision.PASS:
            raise ValueError("a failed release-policy rule cannot produce PASS")
        return self


class ReleasePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    rules: list[ReleasePolicyRule] = Field(min_length=1)
    missing_required_coverage_action: MissingRequiredCoverageAction = (
        MissingRequiredCoverageAction.NO_DECISION
    )
    critical_finding_override: bool = True

    @model_validator(mode="after")
    def validate_rule_ids(self) -> ReleasePolicy:
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("release-policy rule_id values must be unique")
        return self


class PolicyRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    metric: str
    actual: float | None = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    operator: PolicyOperator
    passed: bool | None
    on_failure: ReleaseDecision


class ReleasePolicyResult(BaseModel):
    """``release_decision=None`` serializes as JSON null (no valid decision)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_version: str = RELEASE_POLICY_ENGINE_VERSION
    policy_id: str
    policy_version: str
    execution_status: ExecutionStatus
    release_decision: ReleaseDecision | None
    rule_results: list[PolicyRuleResult]
    reason_codes: list[str]
    critical_finding_override_applied: bool = False


_OPERATORS: dict[PolicyOperator, Callable[[float, float], bool]] = {
    PolicyOperator.EQUAL: operator.eq,
    PolicyOperator.NOT_EQUAL: operator.ne,
    PolicyOperator.GREATER_THAN: operator.gt,
    PolicyOperator.GREATER_THAN_OR_EQUAL: operator.ge,
    PolicyOperator.LESS_THAN: operator.lt,
    PolicyOperator.LESS_THAN_OR_EQUAL: operator.le,
}


class ReleasePolicyEngine:
    version = RELEASE_POLICY_ENGINE_VERSION

    def evaluate(
        self,
        *,
        policy: ReleasePolicy,
        execution_status: ExecutionStatus | str,
        metrics: dict[str, float | int],
        coverage: CoverageReport | None,
        critical_findings: int = 0,
    ) -> ReleasePolicyResult:
        """Apply a policy without converting assurance failures into clinical results."""

        status = ExecutionStatus(execution_status)
        if isinstance(critical_findings, bool) or not isinstance(critical_findings, int):
            raise ValueError("critical_findings must be an integer")
        if critical_findings < 0:
            raise ValueError("critical_findings cannot be negative")

        if status is not ExecutionStatus.COMPLETED:
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=["EXECUTION_NOT_COMPLETED_NO_DECISION"],
            )

        if coverage is None:
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=["COVERAGE_MISSING_NO_DECISION"],
            )

        if coverage.technical_failed_cases:
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=["TECHNICAL_EVALUATION_FAILURE_NO_DECISION"],
            )

        # A known critical clinical failure is sufficient to block.  This does
        # not override a technical run failure handled above.
        if policy.critical_finding_override and critical_findings > 0:
            return self._result(
                policy=policy,
                status=status,
                decision=ReleaseDecision.BLOCK,
                reason_codes=["CRITICAL_FINDING_OVERRIDE"],
                critical_override=True,
            )

        rule_results: list[PolicyRuleResult] = []
        failed_actions: list[ReleaseDecision] = []
        missing_metrics: list[str] = []
        invalid_metrics: list[str] = []
        for rule in policy.rules:
            raw_actual = metrics.get(rule.metric)
            if raw_actual is None:
                missing_metrics.append(rule.metric)
                rule_results.append(
                    PolicyRuleResult(
                        rule_id=rule.rule_id,
                        metric=rule.metric,
                        actual=None,
                        threshold=rule.threshold,
                        operator=rule.operator,
                        passed=None,
                        on_failure=rule.on_failure,
                    )
                )
                continue
            if isinstance(raw_actual, bool):
                invalid_metrics.append(rule.metric)
                rule_results.append(
                    PolicyRuleResult(
                        rule_id=rule.rule_id,
                        metric=rule.metric,
                        actual=None,
                        threshold=rule.threshold,
                        operator=rule.operator,
                        passed=None,
                        on_failure=rule.on_failure,
                    )
                )
                continue
            try:
                actual = float(raw_actual)
            except (TypeError, ValueError, OverflowError):
                actual = float("nan")
            if not isfinite(actual):
                invalid_metrics.append(rule.metric)
                rule_results.append(
                    PolicyRuleResult(
                        rule_id=rule.rule_id,
                        metric=rule.metric,
                        actual=None,
                        threshold=rule.threshold,
                        operator=rule.operator,
                        passed=None,
                        on_failure=rule.on_failure,
                    )
                )
                continue
            passed = _OPERATORS[rule.operator](actual, rule.threshold)
            if not passed:
                failed_actions.append(rule.on_failure)
            rule_results.append(
                PolicyRuleResult(
                    rule_id=rule.rule_id,
                    metric=rule.metric,
                    actual=actual,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    passed=passed,
                    on_failure=rule.on_failure,
                )
            )

        # A known blocking metric remains a valid BLOCK even if another metric
        # is unavailable.  Missing evidence can never produce PASS or WARN.
        if ReleaseDecision.BLOCK in failed_actions:
            return self._result(
                policy=policy,
                status=status,
                decision=ReleaseDecision.BLOCK,
                reason_codes=["BLOCKING_POLICY_RULE_FAILED"],
                rule_results=rule_results,
            )

        if missing_metrics or invalid_metrics:
            reason_codes = []
            if missing_metrics:
                reason_codes.extend(
                    [
                        "POLICY_METRIC_MISSING_NO_DECISION",
                        *(f"MISSING_METRIC:{metric}" for metric in sorted(set(missing_metrics))),
                    ]
                )
            if invalid_metrics:
                reason_codes.extend(
                    [
                        "POLICY_METRIC_INVALID_NO_DECISION",
                        *(f"INVALID_METRIC:{metric}" for metric in sorted(set(invalid_metrics))),
                    ]
                )
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=reason_codes,
                rule_results=rule_results,
            )

        if coverage.missing_required_tests or coverage.status in {
            CoverageStatus.PARTIAL,
            CoverageStatus.UNASSESSED,
        }:
            if policy.missing_required_coverage_action is MissingRequiredCoverageAction.BLOCK:
                return self._result(
                    policy=policy,
                    status=status,
                    decision=ReleaseDecision.BLOCK,
                    reason_codes=["REQUIRED_COVERAGE_INCOMPLETE_BLOCK"],
                    rule_results=rule_results,
                )
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=["REQUIRED_COVERAGE_INCOMPLETE_NO_DECISION"],
                rule_results=rule_results,
            )

        if not coverage.valid_for_release:
            return self._result(
                policy=policy,
                status=status,
                decision=None,
                reason_codes=["COVERAGE_NOT_RELEASE_VALID_NO_DECISION"],
                rule_results=rule_results,
            )

        if ReleaseDecision.WARN in failed_actions:
            decision = ReleaseDecision.WARN
            reasons = ["WARNING_POLICY_RULE_FAILED"]
        elif coverage.status is CoverageStatus.REDUCED:
            decision = ReleaseDecision.WARN
            reasons = ["REDUCED_COVERAGE"]
        else:
            decision = ReleaseDecision.PASS
            reasons = ["ALL_POLICY_RULES_PASSED"]
        return self._result(
            policy=policy,
            status=status,
            decision=decision,
            reason_codes=reasons,
            rule_results=rule_results,
        )

    @staticmethod
    def _result(
        *,
        policy: ReleasePolicy,
        status: ExecutionStatus,
        decision: ReleaseDecision | None,
        reason_codes: list[str],
        rule_results: list[PolicyRuleResult] | None = None,
        critical_override: bool = False,
    ) -> ReleasePolicyResult:
        return ReleasePolicyResult(
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            execution_status=status,
            release_decision=decision,
            rule_results=rule_results or [],
            reason_codes=reason_codes,
            critical_finding_override_applied=critical_override,
        )


def apply_release_policy(
    *,
    policy: ReleasePolicy,
    execution_status: ExecutionStatus | str,
    metrics: dict[str, float | int],
    coverage: CoverageReport | None,
    critical_findings: int = 0,
) -> ReleasePolicyResult:
    return ReleasePolicyEngine().evaluate(
        policy=policy,
        execution_status=execution_status,
        metrics=metrics,
        coverage=coverage,
        critical_findings=critical_findings,
    )

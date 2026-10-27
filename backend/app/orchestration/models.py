"""Execution-only orchestration contracts.

These models deliberately distinguish successful execution from a clinical release decision.
The orchestrator never manufactures ``PASS``; release policy remains a downstream concern.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from backend.app.adapters.models import AdapterResponse


class CaseExecutionStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    UNASSESSED = "UNASSESSED"
    CANCELLED = "CANCELLED"


class RunExecutionStatus(StrEnum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AssessmentStatus(StrEnum):
    ASSESSED = "ASSESSED"
    UNASSESSED = "UNASSESSED"


class RunCoverage(StrEnum):
    FULL = "FULL"
    REDUCED = "REDUCED"
    UNASSESSED = "UNASSESSED"


class TransitionEntity(StrEnum):
    RUN = "RUN"
    CASE = "CASE"


@dataclass(frozen=True, slots=True)
class OrchestrationCase:
    case_id: str
    input_data: Mapping[str, Any] = field(repr=False)
    evaluator_context: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not isinstance(self.input_data, Mapping):
            raise TypeError("input_data must be a mapping")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt: int
    started_at: datetime
    completed_at: datetime
    latency_ms: float
    outcome: str
    status_code: int | None = None
    error_code: str | None = None
    retryable: bool = False
    backoff_seconds: float = 0.0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class StateTransition:
    run_id: str
    entity: TransitionEntity
    entity_id: str
    from_status: str | None
    to_status: str
    occurred_at: datetime
    attempt: int | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class CaseExecutionResult:
    case_id: str
    execution_status: CaseExecutionStatus
    assessment_status: AssessmentStatus
    attempts: int
    attempt_history: tuple[AttemptRecord, ...]
    output: Any = field(default=None, repr=False)
    adapter_response: AdapterResponse | None = field(default=None, repr=False)
    evaluation: Any = field(default=None, repr=False)
    cost_usd: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    release_decision: None = None
    fatal_run: bool = False

    @property
    def assessed(self) -> bool:
        return self.assessment_status is AssessmentStatus.ASSESSED


@dataclass(frozen=True, slots=True)
class RunExecutionResult:
    run_id: str
    execution_status: RunExecutionStatus
    coverage: RunCoverage
    case_results: tuple[CaseExecutionResult, ...]
    cases_completed: int
    cases_unassessed: int
    total_attempts: int
    total_cost_usd: float
    started_at: datetime
    completed_at: datetime
    failure_code: str | None = None
    failure_message: str | None = None
    release_decision: None = None

    @property
    def coverage_reduced(self) -> bool:
        return self.coverage is not RunCoverage.FULL


TestCase = OrchestrationCase
CaseResult = CaseExecutionResult
RunResult = RunExecutionResult
CaseStatus = CaseExecutionStatus
RunStatus = RunExecutionStatus
CoverageStatus = RunCoverage

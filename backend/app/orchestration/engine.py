"""Bounded-concurrency, failure-safe test run orchestration."""

from __future__ import annotations

import asyncio
import inspect
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from backend.app.adapters.errors import (
    AdapterError,
    AdapterSchemaError,
    AdapterTimeoutError,
)
from backend.app.adapters.models import AdapterResponse
from backend.app.adapters.protocols import AIAdapter
from backend.app.core.errors import ErrorCode
from backend.app.orchestration.models import (
    AssessmentStatus,
    AttemptRecord,
    CaseExecutionResult,
    CaseExecutionStatus,
    OrchestrationCase,
    RunCoverage,
    RunExecutionResult,
    RunExecutionStatus,
    StateTransition,
    TransitionEntity,
)
from backend.app.orchestration.protocols import (
    EvaluatorCallback,
    JitterCallback,
    SleepCallback,
    StateTransitionCallback,
    StateTransitionStore,
)
from backend.app.orchestration.retry import RetryPolicy

ORCHESTRATOR_VERSION = "test-orchestrator@0.1.0"


class StatePersistenceError(RuntimeError):
    """Raised when a durable state transition cannot be recorded."""


@dataclass(slots=True)
class _FatalContext:
    event: asyncio.Event
    code: str | None = None
    message: str | None = None

    def set(self, code: str, message: str) -> None:
        if not self.event.is_set():
            self.code = code
            self.message = message
            self.event.set()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_jitter(delay: float) -> float:
    return random.uniform(0.0, delay * 0.25)


class TestOrchestrator:
    """Execute cases without conflating technical completion with clinical PASS."""

    version = ORCHESTRATOR_VERSION

    def __init__(
        self,
        adapter: AIAdapter,
        evaluator: EvaluatorCallback,
        *,
        max_concurrency: int = 5,
        timeout_seconds: float = 30.0,
        evaluator_timeout_seconds: float | None = None,
        transition_timeout_seconds: float | None = None,
        retry_policy: RetryPolicy | None = None,
        transition_callback: StateTransitionCallback | StateTransitionStore | None = None,
        jitter: JitterCallback = _default_jitter,
        sleep: SleepCallback = asyncio.sleep,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if evaluator_timeout_seconds is not None and evaluator_timeout_seconds <= 0:
            raise ValueError("evaluator_timeout_seconds must be greater than zero")
        if transition_timeout_seconds is not None and transition_timeout_seconds <= 0:
            raise ValueError("transition_timeout_seconds must be greater than zero")
        self._adapter = adapter
        self._evaluator = evaluator
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.evaluator_timeout_seconds = evaluator_timeout_seconds or timeout_seconds
        self.transition_timeout_seconds = transition_timeout_seconds or timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self._transition_callback = transition_callback
        self._jitter = jitter
        self._sleep = sleep
        self._transition_lock = asyncio.Lock()

    async def run(
        self,
        run_id: str,
        cases: Sequence[OrchestrationCase],
    ) -> RunExecutionResult:
        try:
            return await self._run(run_id, cases)
        except StatePersistenceError:
            now = _utc_now()
            case_results = tuple(
                self._unassessed_result(
                    case,
                    attempts=(),
                    error_code=ErrorCode.WORKER_FAILED.value,
                    error_message="Durable execution state could not be recorded.",
                    fatal_run=True,
                )
                for case in cases
                if isinstance(case, OrchestrationCase)
            )
            return RunExecutionResult(
                run_id=run_id,
                execution_status=RunExecutionStatus.FAILED,
                coverage=RunCoverage.UNASSESSED,
                case_results=case_results,
                cases_completed=0,
                cases_unassessed=len(case_results),
                total_attempts=0,
                total_cost_usd=0.0,
                started_at=now,
                completed_at=now,
                failure_code=ErrorCode.WORKER_FAILED.value,
                failure_message="Durable execution state could not be recorded.",
            )

    async def _run(
        self,
        run_id: str,
        cases: Sequence[OrchestrationCase],
    ) -> RunExecutionResult:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty")
        if not cases:
            raise ValueError("at least one case is required")
        if any(not isinstance(case, OrchestrationCase) for case in cases):
            raise TypeError("cases must contain OrchestrationCase values")
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique within a run")

        started_at = _utc_now()
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.RUN,
            entity_id=run_id,
            from_status=None,
            to_status=RunExecutionStatus.QUEUED,
        )
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.RUN,
            entity_id=run_id,
            from_status=RunExecutionStatus.QUEUED,
            to_status=RunExecutionStatus.PREPARING,
        )
        for case in cases:
            await self._transition(
                run_id=run_id,
                entity=TransitionEntity.CASE,
                entity_id=case.case_id,
                from_status=None,
                to_status=CaseExecutionStatus.QUEUED,
            )
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.RUN,
            entity_id=run_id,
            from_status=RunExecutionStatus.PREPARING,
            to_status=RunExecutionStatus.RUNNING,
        )

        semaphore = asyncio.Semaphore(self.max_concurrency)
        fatal = _FatalContext(asyncio.Event())
        tasks = [
            asyncio.create_task(self._bounded_case(run_id, case, semaphore, fatal))
            for case in cases
        ]
        case_results = tuple(await asyncio.gather(*tasks))

        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.RUN,
            entity_id=run_id,
            from_status=RunExecutionStatus.RUNNING,
            to_status=RunExecutionStatus.AGGREGATING,
        )
        completed_count = sum(
            result.execution_status is CaseExecutionStatus.COMPLETED for result in case_results
        )
        unassessed_count = len(case_results) - completed_count
        if unassessed_count == 0:
            coverage = RunCoverage.FULL
        elif completed_count == 0:
            coverage = RunCoverage.UNASSESSED
        else:
            coverage = RunCoverage.REDUCED

        failed = fatal.event.is_set() or completed_count == 0
        final_status = RunExecutionStatus.FAILED if failed else RunExecutionStatus.COMPLETED
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.RUN,
            entity_id=run_id,
            from_status=RunExecutionStatus.AGGREGATING,
            to_status=final_status,
            error_code=fatal.code if failed else None,
        )
        failure_code = None
        failure_message = None
        if failed:
            failure_code = fatal.code or ErrorCode.UNASSESSED_CASES_PRESENT.value
            failure_message = fatal.message or "No test case could be assessed technically."
        return RunExecutionResult(
            run_id=run_id,
            execution_status=final_status,
            coverage=coverage,
            case_results=case_results,
            cases_completed=completed_count,
            cases_unassessed=unassessed_count,
            total_attempts=sum(result.attempts for result in case_results),
            total_cost_usd=sum(result.cost_usd for result in case_results),
            started_at=started_at,
            completed_at=_utc_now(),
            failure_code=failure_code,
            failure_message=failure_message,
        )

    async def execute(self, run_id: str, cases: Sequence[OrchestrationCase]) -> RunExecutionResult:
        return await self.run(run_id, cases)

    async def _bounded_case(
        self,
        run_id: str,
        case: OrchestrationCase,
        semaphore: asyncio.Semaphore,
        fatal: _FatalContext,
    ) -> CaseExecutionResult:
        async with semaphore:
            if fatal.event.is_set():
                return await self._aborted_case(run_id, case, fatal)
            try:
                return await self._execute_case(run_id, case, fatal)
            except StatePersistenceError:
                fatal.set(
                    ErrorCode.WORKER_FAILED.value,
                    "Durable execution state could not be recorded.",
                )
                return self._unassessed_result(
                    case,
                    attempts=(),
                    error_code=ErrorCode.WORKER_FAILED.value,
                    error_message="Durable execution state could not be recorded.",
                    fatal_run=True,
                )
            except Exception:
                fatal.set(ErrorCode.WORKER_FAILED.value, "An orchestration worker failed.")
                return self._unassessed_result(
                    case,
                    attempts=(),
                    error_code=ErrorCode.WORKER_FAILED.value,
                    error_message="An orchestration worker failed.",
                    fatal_run=True,
                )

    async def _execute_case(
        self,
        run_id: str,
        case: OrchestrationCase,
        fatal: _FatalContext,
    ) -> CaseExecutionResult:
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.CASE,
            entity_id=case.case_id,
            from_status=CaseExecutionStatus.QUEUED,
            to_status=CaseExecutionStatus.RUNNING,
        )
        attempts: list[AttemptRecord] = []
        response: AdapterResponse | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            if fatal.event.is_set():
                return await self._mark_unassessed(
                    run_id,
                    case,
                    from_status=CaseExecutionStatus.RUNNING,
                    attempts=attempts,
                    error_code=fatal.code or ErrorCode.WORKER_FAILED.value,
                    error_message="Case was stopped after a fatal run configuration error.",
                )
            attempt_started = _utc_now()
            timer = perf_counter()
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    response = await self._adapter.invoke(case.input_data)
                if not isinstance(response, AdapterResponse):
                    raise AdapterSchemaError("AI adapter returned an invalid result contract.")
            except TimeoutError:
                error: AdapterError = AdapterTimeoutError()
            except AdapterError as caught:
                error = caught
            except Exception:
                error = AdapterSchemaError("AI adapter failed with an unexpected result.")
            else:
                attempts.append(
                    AttemptRecord(
                        attempt=attempt,
                        started_at=attempt_started,
                        completed_at=_utc_now(),
                        latency_ms=(perf_counter() - timer) * 1000,
                        outcome="SUCCEEDED",
                        status_code=response.status_code,
                        cost_usd=response.cost_usd,
                    )
                )
                break

            should_retry = self.retry_policy.should_retry(error, attempt)
            backoff = 0.0
            if should_retry:
                backoff = self.retry_policy.delay_for(
                    attempt,
                    jitter=self._jitter,
                    retry_after_seconds=error.retry_after_seconds,
                )
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    started_at=attempt_started,
                    completed_at=_utc_now(),
                    latency_ms=(perf_counter() - timer) * 1000,
                    outcome="RETRYING" if should_retry else "FAILED",
                    status_code=error.status_code,
                    error_code=error.code.value,
                    retryable=should_retry,
                    backoff_seconds=backoff,
                )
            )
            if error.fatal:
                fatal.set(error.code.value, str(error))
            if not should_retry:
                return await self._mark_unassessed(
                    run_id,
                    case,
                    from_status=CaseExecutionStatus.RUNNING,
                    attempts=attempts,
                    error_code=error.code.value,
                    error_message=str(error),
                    fatal_run=error.fatal,
                )
            await self._sleep(backoff)

        if response is None:
            return await self._mark_unassessed(
                run_id,
                case,
                from_status=CaseExecutionStatus.RUNNING,
                attempts=attempts,
                error_code=ErrorCode.AI_TIMEOUT.value,
                error_message="AI endpoint retries were exhausted.",
            )

        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.CASE,
            entity_id=case.case_id,
            from_status=CaseExecutionStatus.RUNNING,
            to_status=CaseExecutionStatus.EVALUATING,
            attempt=len(attempts),
        )
        try:
            async with asyncio.timeout(self.evaluator_timeout_seconds):
                evaluation = await self._evaluate(case, response)
        except Exception:
            return await self._mark_unassessed(
                run_id,
                case,
                from_status=CaseExecutionStatus.EVALUATING,
                attempts=attempts,
                error_code=ErrorCode.SEMANTIC_EVALUATOR_FAILED.value,
                error_message="Evaluator callback failed technically.",
                output=response.output,
                response=response,
            )

        evaluation_failure = self._unassessed_evaluation(evaluation)
        if evaluation_failure is not None:
            error_code, error_message = evaluation_failure
            return await self._mark_unassessed(
                run_id,
                case,
                from_status=CaseExecutionStatus.EVALUATING,
                attempts=attempts,
                error_code=error_code,
                error_message=error_message,
                output=response.output,
                response=response,
            )

        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.CASE,
            entity_id=case.case_id,
            from_status=CaseExecutionStatus.EVALUATING,
            to_status=CaseExecutionStatus.COMPLETED,
            attempt=len(attempts),
        )
        return CaseExecutionResult(
            case_id=case.case_id,
            execution_status=CaseExecutionStatus.COMPLETED,
            assessment_status=AssessmentStatus.ASSESSED,
            attempts=len(attempts),
            attempt_history=tuple(attempts),
            output=response.output,
            adapter_response=response,
            evaluation=evaluation,
            cost_usd=sum(item.cost_usd for item in attempts),
        )

    async def _evaluate(self, case: OrchestrationCase, response: AdapterResponse) -> object:
        if inspect.iscoroutinefunction(self._evaluator):
            return await self._evaluator(case, response)
        result = await asyncio.to_thread(self._evaluator, case, response)
        if inspect.isawaitable(result):
            return await result
        return result

    def _unassessed_evaluation(self, evaluation: object) -> tuple[str, str] | None:
        def field(name: str) -> object | None:
            if isinstance(evaluation, Mapping):
                return evaluation.get(name)
            return getattr(evaluation, name, None)

        assessment_status = field("assessment_status")
        assessment_value = getattr(assessment_status, "value", assessment_status)
        normalized_assessment = (
            str(assessment_value).upper() if assessment_value is not None else None
        )
        if normalized_assessment == "UNASSESSED":
            return (
                ErrorCode.EVALUATOR_NOT_IMPLEMENTED.value,
                "Evaluator reported that the case could not be assessed.",
            )

        execution_status = field("execution_status")
        execution_value = getattr(execution_status, "value", execution_status)
        normalized_execution = str(execution_value).upper() if execution_value is not None else None
        if normalized_execution == "FAILED":
            return (
                ErrorCode.SEMANTIC_EVALUATOR_FAILED.value,
                "Evaluator reported a technical execution failure.",
            )

        # Evaluator output is a safety boundary. Both technical execution and
        # assessment must positively complete; absence or contradiction fails closed.
        if normalized_assessment != "ASSESSED" or normalized_execution != "COMPLETED":
            return (
                ErrorCode.EVALUATOR_INVALID_RESULT.value,
                "Evaluator did not return a valid completed assessment contract.",
            )

        coverage = field("coverage")
        if isinstance(coverage, Mapping):
            coverage_status = coverage.get("status")
        else:
            coverage_status = getattr(coverage, "status", None)
        coverage_value = getattr(coverage_status, "value", coverage_status)
        normalized_coverage = str(coverage_value).upper() if coverage_value is not None else None
        if normalized_coverage == "UNASSESSED":
            return (
                ErrorCode.SEMANTIC_EVALUATOR_FAILED.value,
                "Evaluator returned unassessed coverage.",
            )
        if normalized_coverage != "FULL":
            return (
                ErrorCode.EVALUATOR_INVALID_RESULT.value,
                "Evaluator did not return full assessment coverage.",
            )
        return None

    async def _mark_unassessed(
        self,
        run_id: str,
        case: OrchestrationCase,
        *,
        from_status: CaseExecutionStatus,
        attempts: Sequence[AttemptRecord],
        error_code: str,
        error_message: str,
        fatal_run: bool = False,
        output: Any = None,
        response: AdapterResponse | None = None,
    ) -> CaseExecutionResult:
        await self._transition(
            run_id=run_id,
            entity=TransitionEntity.CASE,
            entity_id=case.case_id,
            from_status=from_status,
            to_status=CaseExecutionStatus.UNASSESSED,
            attempt=len(attempts) or None,
            error_code=error_code,
        )
        return self._unassessed_result(
            case,
            attempts=attempts,
            error_code=error_code,
            error_message=error_message,
            fatal_run=fatal_run,
            output=output,
            response=response,
        )

    async def _aborted_case(
        self, run_id: str, case: OrchestrationCase, fatal: _FatalContext
    ) -> CaseExecutionResult:
        return await self._mark_unassessed(
            run_id,
            case,
            from_status=CaseExecutionStatus.QUEUED,
            attempts=(),
            error_code=fatal.code or ErrorCode.WORKER_FAILED.value,
            error_message="Case was not executed after a fatal run configuration error.",
            fatal_run=True,
        )

    def _unassessed_result(
        self,
        case: OrchestrationCase,
        *,
        attempts: Sequence[AttemptRecord],
        error_code: str,
        error_message: str,
        fatal_run: bool,
        output: Any = None,
        response: AdapterResponse | None = None,
    ) -> CaseExecutionResult:
        return CaseExecutionResult(
            case_id=case.case_id,
            execution_status=CaseExecutionStatus.UNASSESSED,
            assessment_status=AssessmentStatus.UNASSESSED,
            attempts=len(attempts),
            attempt_history=tuple(attempts),
            output=output,
            adapter_response=response,
            cost_usd=sum(item.cost_usd for item in attempts),
            error_code=error_code,
            error_message=error_message,
            fatal_run=fatal_run,
        )

    async def _transition(
        self,
        *,
        run_id: str,
        entity: TransitionEntity,
        entity_id: str,
        from_status: str | None,
        to_status: str,
        attempt: int | None = None,
        error_code: str | None = None,
    ) -> None:
        callback = self._transition_callback
        if callback is None:
            return
        transition = StateTransition(
            run_id=run_id,
            entity=entity,
            entity_id=entity_id,
            from_status=from_status,
            to_status=to_status,
            occurred_at=_utc_now(),
            attempt=attempt,
            error_code=error_code,
        )
        try:
            async with self._transition_lock:
                async with asyncio.timeout(self.transition_timeout_seconds):
                    await self._record_transition(callback, transition)
        except Exception as exc:
            raise StatePersistenceError("durable state transition callback failed") from exc

    async def _record_transition(
        self,
        callback: StateTransitionCallback | StateTransitionStore,
        transition: StateTransition,
    ) -> None:
        target = callback.record_transition if hasattr(callback, "record_transition") else callback
        if inspect.iscoroutinefunction(target):
            await target(transition)
            return
        # A synchronous store must create/use its database session inside this
        # thread; SQLAlchemy sessions must not be shared across threads.
        result = await asyncio.to_thread(target, transition)
        if inspect.isawaitable(result):
            await result


Orchestrator = TestOrchestrator

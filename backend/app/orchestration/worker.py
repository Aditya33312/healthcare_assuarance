"""Durable worker for executing queued Gatekeeper test runs.

The HTTP API owns planning and queueing.  This module owns the other side of
that boundary: claim a queued run, invoke the registered application endpoint,
evaluate every selected case, persist the terminal state, and apply the
release policy.  It deliberately uses a database-backed queue so deployments
do not need a particular Redis/Celery product before the execution path is
available.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from backend.app.adapters import (
    CustomJSONFieldMappingAdapter,
    GenericRESTAdapter,
    OpenAICompatibleAdapter,
)
from backend.app.adapters.protocols import AIAdapter
from backend.app.core.errors import ErrorCode
from backend.app.database.models import (
    AISystem,
    AISystemVersion,
    ApplicationProfileRecord,
    ReleasePolicy,
    ReleasePolicyVersion,
    TestCase,
    TestCaseVersion,
    TestPlanRecord,
    TestRun,
    TestRunCase,
)
from backend.app.database.repositories import TenantRepository
from backend.app.domain.enums import ExecutionStatus
from backend.app.evaluators import EvaluatorRouter
from backend.app.orchestration.engine import TestOrchestrator
from backend.app.orchestration.models import (
    OrchestrationCase,
    RunCoverage,
    RunExecutionResult,
    RunExecutionStatus,
    StateTransition,
    TransitionEntity,
)
from backend.app.orchestration.retry import RetryPolicy
from backend.app.policies import ReleasePolicy as DomainReleasePolicy
from backend.app.policies import ReleasePolicyEngine
from backend.app.security import EncryptedJsonStore
from backend.app.systems import ApplicationProfile
from backend.app.test_plans import TestPlan

logger = logging.getLogger("gatekeeper.worker")
WORKER_VERSION = "test-run-worker@0.1.0"


def _json_result(value: object) -> dict[str, Any]:
    """Convert a pydantic/dataclass/mapping result to a JSON object."""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    asdict = getattr(value, "__dict__", None)
    if isinstance(asdict, dict):
        return dict(asdict)
    raise TypeError("worker result is not JSON serializable")


def _headers_from_authentication(config: Any) -> dict[str, str]:
    """Read encrypted adapter headers without accepting credentials in URLs."""

    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("authentication_config must be an object")
    raw_headers = config.get("headers", {})
    if not isinstance(raw_headers, Mapping):
        raise ValueError("authentication_config.headers must be an object")
    headers: dict[str, str] = {}
    for name, value in raw_headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("authentication headers must contain string names and values")
        headers[name] = value
    api_key = config.get("api_key")
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("authentication_config.api_key must be a non-empty string")
        header_name = config.get("api_key_header", "x-api-key")
        if not isinstance(header_name, str) or not header_name:
            raise ValueError("authentication_config.api_key_header must be a string")
        headers.setdefault(header_name, api_key)
    token = config.get("bearer_token", config.get("token"))
    if token is not None:
        if not isinstance(token, str) or not token:
            raise ValueError("authentication_config.token must be a non-empty string")
        headers.setdefault(
            "authorization",
            token if token.casefold().startswith("bearer ") else f"Bearer {token}",
        )
    username = config.get("username")
    password = config.get("password")
    if username is not None or password is not None:
        if not isinstance(username, str) or not isinstance(password, str):
            raise ValueError("basic authentication requires username and password strings")
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        headers.setdefault("authorization", f"Basic {encoded}")
    return headers


def build_application_adapter(
    version: AISystemVersion,
    *,
    authentication_config: Mapping[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> AIAdapter:
    """Build the transport adapter described by a registered system version."""

    if not version.endpoint_url:
        raise ValueError("the registered system version has no endpoint_url")
    request_schema = version.request_schema or {}
    response_schema = version.response_schema or {}
    if not isinstance(request_schema, Mapping) or not isinstance(response_schema, Mapping):
        raise ValueError("request_schema and response_schema must be objects")

    headers = _headers_from_authentication(authentication_config)
    method = str(request_schema.get("method", "POST")).upper()
    response_path = response_schema.get("output_path", response_schema.get("path", "output"))
    if not isinstance(response_path, str):
        raise ValueError("response_schema.output_path must be a string")

    adapter_name = str(request_schema.get("adapter", "generic_rest")).casefold()
    if adapter_name in {"openai", "openai_compatible", "openai-compatible"}:
        model = request_schema.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("openai-compatible request_schema.model is required")
        return OpenAICompatibleAdapter(
            endpoint_url=version.endpoint_url,
            model=model,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )

    mapping = request_schema.get("field_mapping", request_schema.get("mapping"))
    static_fields = request_schema.get("static_fields", {})
    include_unmapped = request_schema.get("include_unmapped_fields", True)
    if mapping is not None or static_fields:
        if not isinstance(mapping, Mapping) or not isinstance(static_fields, Mapping):
            raise ValueError("request_schema mappings and static_fields must be objects")
        return CustomJSONFieldMappingAdapter(
            version.endpoint_url,
            request_mapping={str(key): str(value) for key, value in mapping.items()},
            static_fields=static_fields,
            include_unmapped_fields=bool(include_unmapped),
            response_path=response_path,
            method=method,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
    return GenericRESTAdapter(
        version.endpoint_url,
        response_path=response_path,
        method=method,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


class _TransitionWriter:
    """Persist orchestration transitions using a fresh worker-thread session."""

    def __init__(self, sessions: sessionmaker[Session], organization_id: str) -> None:
        self.sessions = sessions
        self.organization_id = organization_id

    def record_transition(self, transition: StateTransition) -> None:
        # ``claim_next`` already moved the run from QUEUED to PREPARING.  The
        # generic orchestrator emits its complete lifecycle from QUEUED, so do
        # not regress the durable claim and make the run visible to another
        # worker while this worker is executing it.
        if transition.entity is TransitionEntity.RUN and (
            (transition.from_status is None and transition.to_status == "QUEUED")
            or (
                transition.from_status == "QUEUED"
                and transition.to_status == "PREPARING"
            )
        ):
            return
        with self.sessions() as session:
            if transition.entity is TransitionEntity.RUN:
                session.execute(
                    update(TestRun)
                    .where(
                        TestRun.organization_id == self.organization_id,
                        TestRun.id == transition.entity_id,
                    )
                    .values(
                        execution_status=transition.to_status,
                        failure_code=transition.error_code,
                    )
                )
            else:
                session.execute(
                    update(TestRunCase)
                    .where(
                        TestRunCase.organization_id == self.organization_id,
                        TestRunCase.id == transition.entity_id,
                    )
                    .values(
                        execution_status=transition.to_status,
                        error_code=transition.error_code,
                    )
                )
            session.commit()


class TestRunWorker:
    """Claim and execute queued runs from the Gatekeeper database."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        encryptor: EncryptedJsonStore,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        max_concurrency: int = 5,
    ) -> None:
        self.sessions = sessions
        self.encryptor = encryptor
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.max_concurrency = max_concurrency

    def claim_next(self) -> str | None:
        """Atomically claim one queued run; safe for multiple worker processes."""

        with self.sessions() as session:
            run = session.scalar(
                select(TestRun)
                .where(TestRun.execution_status == ExecutionStatus.QUEUED.value)
                .order_by(TestRun.created_at, TestRun.id)
                .limit(1)
            )
            if run is None:
                return None
            claimed = session.execute(
                update(TestRun)
                .where(
                    TestRun.id == run.id,
                    TestRun.organization_id == run.organization_id,
                    TestRun.execution_status == ExecutionStatus.QUEUED.value,
                )
                .values(execution_status=ExecutionStatus.PREPARING.value)
            )
            if int(getattr(claimed, "rowcount", 0)) != 1:
                session.rollback()
                return None
            session.commit()
            return run.id

    async def execute_run(self, run_id: str) -> RunExecutionResult:
        with self.sessions() as session:
            run = session.get(TestRun, run_id)
            if run is None:
                raise LookupError(f"test run {run_id} was not found")
            organization_id = run.organization_id
            system_version = TenantRepository(session, AISystemVersion).get(
                organization_id, run.system_version_id
            )
            system = TenantRepository(session, AISystem).get(organization_id, run.system_id)
            if system_version is None or system is None:
                return self._fail_run(
                    session,
                    run,
                    ErrorCode.VERSION_NOT_FOUND.value,
                    "System version is unavailable.",
                )
            profile_record = session.scalar(
                select(ApplicationProfileRecord).where(
                    ApplicationProfileRecord.organization_id == organization_id,
                    ApplicationProfileRecord.system_id == run.system_id,
                ).order_by(ApplicationProfileRecord.created_at.desc())
            )
            plan_id = run.trigger_source.get("gatekeeper", {}).get("test_plan_id")
            plan_record = (
                TenantRepository(session, TestPlanRecord).get(organization_id, plan_id)
                if isinstance(plan_id, str)
                else None
            )
            if profile_record is None or plan_record is None:
                return self._fail_run(
                    session,
                    run,
                    ErrorCode.WORKER_FAILED.value,
                    "Run profile or test plan is unavailable.",
                )
            profile = ApplicationProfile.model_validate(profile_record.profile)
            plan = TestPlan.model_validate(plan_record.definition)
            policy_version = (
                TenantRepository(session, ReleasePolicyVersion).get(
                    organization_id, run.release_policy_version_id
                )
                if run.release_policy_version_id is not None
                else None
            )
            policy_record = (
                TenantRepository(session, ReleasePolicy).get(
                    organization_id, policy_version.release_policy_id
                )
                if policy_version is not None
                else None
            )
            if policy_version is None or policy_record is None:
                return self._fail_run(
                    session,
                    run,
                    ErrorCode.WORKER_FAILED.value,
                    "Release policy is unavailable.",
                )
            case_rows = list(
                session.execute(
                    select(TestRunCase, TestCase, TestCaseVersion)
                    .join(TestCaseVersion, TestCaseVersion.id == TestRunCase.test_case_version_id)
                    .join(TestCase, TestCase.id == TestCaseVersion.test_case_id)
                    .where(
                        TestRunCase.organization_id == organization_id,
                        TestRunCase.test_run_id == run.id,
                        TestCase.organization_id == organization_id,
                        TestCaseVersion.organization_id == organization_id,
                    )
                    .order_by(TestRunCase.created_at, TestRunCase.id)
                ).tuples()
            )
            if not case_rows:
                return self._fail_run(
                    session,
                    run,
                    ErrorCode.UNASSESSED_CASES_PRESENT.value,
                    "The run contains no executable cases.",
                )
            auth_config = None
            if system_version.authentication_config_encrypted is not None:
                auth_config = self.encryptor.decrypt(system_version.authentication_config_encrypted)
            adapter = build_application_adapter(
                system_version,
                authentication_config=auth_config,
                timeout_seconds=self.timeout_seconds,
            )
            router = EvaluatorRouter()
            cases = [
                OrchestrationCase(
                    case_id=run_case.id,
                    input_data=case_version.input_data,
                    evaluator_context={
                        "test_family": case.test_family,
                        "expectation": case_version.expectation,
                    },
                )
                for run_case, case, case_version in case_rows
            ]
            transition_writer = _TransitionWriter(self.sessions, organization_id)

        async def evaluate(case: OrchestrationCase, response: Any) -> object:
            payload = dict(case.input_data)
            payload["generated_output"] = response.output
            payload["output"] = response.output
            payload["expectation"] = case.evaluator_context.get("expectation", {})
            if "source_documents" in payload and "source_bundle" not in payload:
                payload["source_bundle"] = {"documents": payload["source_documents"]}
            return await asyncio.to_thread(
                router.evaluate,
                case.evaluator_context["test_family"],
                payload,
            )

        try:
            result = await TestOrchestrator(
                adapter,
                evaluate,
                max_concurrency=self.max_concurrency,
                timeout_seconds=self.timeout_seconds,
                retry_policy=RetryPolicy(max_attempts=self.max_attempts),
                transition_callback=transition_writer,
            ).run(run_id, cases)
        finally:
            close = getattr(adapter, "aclose", None)
            if callable(close):
                maybe = close()
                if inspect.isawaitable(maybe):
                    await maybe

        with self.sessions() as session:
            run = session.get(TestRun, run_id)
            if run is None:
                raise LookupError(f"test run {run_id} disappeared")
            rows = {
                item.id: item
                for item in session.scalars(
                    select(TestRunCase).where(
                        TestRunCase.organization_id == organization_id,
                        TestRunCase.test_run_id == run_id,
                    )
                )
            }
            result_by_id = {item.case_id: item for item in result.case_results}
            all_metrics: list[dict[str, Any]] = []
            scores: list[float] = []
            critical_findings = 0
            for case_id, case_result in result_by_id.items():
                row = rows[case_id]
                row.execution_status = case_result.execution_status.value
                row.error_code = case_result.error_code
                if case_result.output is not None:
                    row.output_data_encrypted = self.encryptor.encrypt(
                        {"output": case_result.output}
                    )
                if case_result.evaluation is not None:
                    per_case_metrics = _json_result(case_result.evaluation)
                    row.metrics = per_case_metrics
                    all_metrics.append(per_case_metrics)
                    score = per_case_metrics.get("score")
                    if isinstance(score, (int, float)) and not isinstance(score, bool):
                        scores.append(float(score))
                    critical = per_case_metrics.get("critical_count")
                    if isinstance(critical, int) and not isinstance(critical, bool):
                        critical_findings += critical
            # Import lazily to avoid the API module/package import cycle.  The
            # worker only needs this route helper after application startup.
            from backend.app.api.routes import _coverage_payload_for_cases

            coverage = _coverage_payload_for_cases(
                profile=profile,
                cases=[(rows[run_case.id], case) for run_case, case, _ in case_rows],
                plan=plan,
            )
            metrics: dict[str, Any] = {
                "score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "cases_completed": result.cases_completed,
                "cases_unassessed": result.cases_unassessed,
                "total_attempts": result.total_attempts,
                "critical_count": critical_findings,
                "case_results": all_metrics,
                "worker_version": WORKER_VERSION,
            }
            coverage_report = _coverage_report(coverage)
            policy = DomainReleasePolicy(
                policy_id=policy_record.id,
                policy_version=policy_version.version,
                rules=policy_version.rules,
                missing_required_coverage_action=(
                    policy_version.required_coverage.get(
                        "missing_required_coverage_action", "NO_DECISION"
                    )
                ),
                critical_finding_override=bool(
                    policy_version.required_coverage.get("critical_finding_override", True)
                ),
            )
            decision = ReleasePolicyEngine().evaluate(
                policy=policy,
                execution_status=result.execution_status.value,
                metrics={
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
                coverage=coverage_report,
                critical_findings=critical_findings,
            )
            run.execution_status = result.execution_status.value
            run.coverage = coverage
            run.metrics = {**metrics, "release_policy": decision.model_dump(mode="json")}
            run.release_decision = (
                decision.release_decision.value if decision.release_decision is not None else None
            )
            run.failure_code = result.failure_code
            run.failure_message = result.failure_message
            session.commit()
        return result

    async def execute_next(self) -> str | None:
        run_id = self.claim_next()
        if run_id is None:
            return None
        try:
            await self.execute_run(run_id)
        except Exception:
            logger.exception("test run worker failed run_id=%s", run_id)
            with self.sessions() as session:
                run = session.get(TestRun, run_id)
                if run is not None:
                    self._fail_run(
                        session,
                        run,
                        ErrorCode.WORKER_FAILED.value,
                        "The execution worker failed.",
                    )
        return run_id

    def _fail_run(
        self,
        session: Session,
        run: TestRun,
        code: str,
        message: str,
    ) -> RunExecutionResult:
        run.execution_status = ExecutionStatus.FAILED.value
        run.failure_code = code
        run.failure_message = message
        run.release_decision = None
        session.commit()
        now = datetime.now(UTC)
        return RunExecutionResult(
            run_id=run.id,
            execution_status=RunExecutionStatus.FAILED,
            coverage=RunCoverage.UNASSESSED,
            case_results=(),
            cases_completed=0,
            cases_unassessed=0,
            total_attempts=0,
            total_cost_usd=0.0,
            started_at=now,
            completed_at=now,
            failure_code=code,
            failure_message=message,
        )


def _coverage_report(payload: Mapping[str, Any]) -> Any:
    from backend.app.coverage import CoverageReport

    return CoverageReport.model_validate(payload)


__all__ = ["WORKER_VERSION", "TestRunWorker", "build_application_adapter"]

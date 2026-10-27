"""Tenant-scoped FastAPI routes backed by the canonical persistence services."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from backend.app.api.decimal_json import DecimalJSONRoute
from backend.app.api.dependencies import get_db_session, get_encryptor, require_scopes
from backend.app.api.errors import ApiError
from backend.app.api.schemas import (
    EVALUATION_OPENAPI_EXAMPLES,
    ApiErrorResponse,
    ReleasePolicyCreate,
    SingleEvaluationCreate,
    SystemCreate,
    SystemVersionCreate,
    TestPlanConfirm,
    TestPlanDraftCreate,
    TestRunCreate,
    TestSuiteCreate,
)
from backend.app.auditing import record_audit_event
from backend.app.auth import ApiKeyPrincipal, ApiScope
from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.compare_engine import compare
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.core.idempotency import IdempotencyReplay, IdempotencyService
from backend.app.coverage import (
    COVERAGE_ENGINE_VERSION,
    CoverageReport,
    TestCoverageRecord,
    calculate_coverage,
)
from backend.app.database.models import (
    AISystem,
    AISystemVersion,
    ApplicationProfileRecord,
    Evaluation,
    FindingRecord,
    ReleasePolicyVersion,
    TestCase,
    TestCaseVersion,
    TestPlanRecord,
    TestRun,
    TestRunCase,
    TestSuite,
    TestSuiteVersion,
)
from backend.app.database.models import (
    ReleasePolicy as ReleasePolicyRecord,
)
from backend.app.database.repositories import TenantRepository
from backend.app.domain.enums import CoverageStatus, ExecutionStatus, ReleaseDecision
from backend.app.domain.models import CompareResult
from backend.app.evaluations import EvaluationService, SingleEvaluationRequest
from backend.app.evaluators import EVALUATOR_ROUTER_VERSION, EvaluatorRouter
from backend.app.ingestion import IngestionLimits, UniversalEvidenceIngestionService
from backend.app.observability import MetricName
from backend.app.orchestration.engine import ORCHESTRATOR_VERSION
from backend.app.policies import ReleasePolicy as DomainReleasePolicy
from backend.app.security import EncryptedJsonStore
from backend.app.systems import ApplicationProfile
from backend.app.test_plans import (
    CustomerTestConfiguration,
    ReleasePolicyReference,
    SuiteCaseBinding,
    SuiteCaseExclusion,
    SuiteCaseExclusionReason,
    TestPlan,
    TestPlanStatus,
    confirm_test_plan,
    generate_test_plan,
)
from backend.app.test_selection import TestFamily, TestRequirement, select_tests
from backend.app.test_suites import (
    CASE_VERSION_IDS_KEY,
    case_version_ids_from_definition,
    case_version_snapshot_digest,
    with_case_version_snapshot,
)

router = APIRouter(route_class=DecimalJSONRoute)
logger = logging.getLogger("gatekeeper.ingestion")

systems_create = require_scopes(ApiScope.SYSTEMS_CREATE)
systems_read = require_scopes(ApiScope.SYSTEMS_READ)
evaluations_create = require_scopes(ApiScope.EVALUATIONS_CREATE)
evaluations_read = require_scopes(ApiScope.EVALUATIONS_READ)
test_runs_create = require_scopes(ApiScope.TEST_RUNS_CREATE)
test_runs_read = require_scopes(ApiScope.TEST_RUNS_READ)
test_suites_manage = require_scopes(ApiScope.TEST_SUITES_MANAGE)
test_plans_manage = require_scopes(ApiScope.TEST_PLANS_MANAGE)
release_policies_manage = require_scopes(ApiScope.RELEASE_POLICIES_MANAGE)
reports_read = require_scopes(ApiScope.REPORTS_READ)


def _request_id(request: Request) -> str:
    request_id = request.state.request_id
    if not isinstance(request_id, str):
        raise RuntimeError("request context did not provide a valid request ID")
    return request_id


def _not_found(code: str, noun: str) -> ApiError:
    return ApiError(code, f"The requested {noun} was not found.", http_status=404)


def _require_environment_access(
    principal: ApiKeyPrincipal,
    environment: str,
) -> None:
    if environment.upper() == "PRODUCTION" and principal.environment != "live":
        raise GatekeeperError(
            ErrorCode.AUTH_ENVIRONMENT_MISMATCH,
            "A live API key is required for production-environment resources.",
            http_status=403,
        )


def _can_read_environment(principal: ApiKeyPrincipal, environment: str) -> bool:
    return environment.upper() != "PRODUCTION" or principal.environment == "live"


def _audit(
    session: Session,
    request: Request,
    principal: ApiKeyPrincipal,
    *,
    action: str,
    target_type: str,
    target_id: str,
) -> None:
    record_audit_event(
        session,
        organization_id=principal.organization_id,
        actor_type="API_KEY",
        actor_id=principal.key_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        request_id=_request_id(request),
    )


def _profile_for_system(
    session: Session,
    organization_id: str,
    system_id: str,
) -> tuple[ApplicationProfileRecord | None, ApplicationProfile | None]:
    record = session.scalar(
        select(ApplicationProfileRecord)
        .where(
            ApplicationProfileRecord.organization_id == organization_id,
            ApplicationProfileRecord.system_id == system_id,
        )
        .order_by(ApplicationProfileRecord.created_at.desc())
    )
    if record is None:
        return None, None
    return record, ApplicationProfile.model_validate(record.profile)


def _can_read_system(
    session: Session,
    principal: ApiKeyPrincipal,
    system_id: str,
) -> bool:
    if principal.environment == "live":
        return True
    _, profile = _profile_for_system(
        session,
        principal.organization_id,
        system_id,
    )
    if profile is not None:
        return _can_read_environment(principal, profile.environment.value)
    environments = tuple(
        session.scalars(
            select(AISystemVersion.environment).where(
                AISystemVersion.organization_id == principal.organization_id,
                AISystemVersion.system_id == system_id,
            )
        )
    )
    return not environments or any(
        _can_read_environment(principal, environment) for environment in environments
    )


def _require_system_read_access(
    session: Session,
    principal: ApiKeyPrincipal,
    system_id: str,
) -> None:
    if not _can_read_system(session, principal, system_id):
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")


def _require_active_system(system: AISystem) -> None:
    if not system.active:
        raise ApiError(
            ErrorCode.SYSTEM_INACTIVE.value,
            "The AI system is inactive and cannot execute or change governance state.",
            http_status=409,
        )


def _system_payload(session: Session, organization_id: str, system: AISystem) -> dict[str, Any]:
    profile_record, profile = _profile_for_system(session, organization_id, system.id)
    payload: dict[str, Any] = {
        "id": system.id,
        "system_id": system.id,
        "name": system.name,
        "description": system.description,
        "application_type": system.application_type,
        "risk_tier": system.risk_tier,
        "active": system.active,
        "created_at": system.created_at,
        "updated_at": system.updated_at,
        "application_profile": None,
        "test_selection": None,
    }
    if profile_record is not None and profile is not None:
        payload["application_profile"] = profile.model_dump(mode="json")
        payload["application_profile_id"] = profile_record.id
        payload["test_selection"] = select_tests(profile).model_dump(mode="json")
    return payload


def _system_version_payload(version: AISystemVersion) -> dict[str, Any]:
    return {
        "id": version.id,
        "system_version_id": version.id,
        "system_id": version.system_id,
        "version": version.version,
        "environment": version.environment,
        "endpoint_url": version.endpoint_url,
        "authentication_configured": version.authentication_config_encrypted is not None,
        "request_schema": version.request_schema,
        "response_schema": version.response_schema,
        "active": version.active,
        "created_at": version.created_at,
        "updated_at": version.updated_at,
    }


@router.post("/systems", status_code=201)
def create_system(
    body: SystemCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(systems_create),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    if body.profile is not None:
        _require_environment_access(principal, body.profile.environment.value)
    system = AISystem(
        organization_id=principal.organization_id,
        name=body.name,
        description=body.description,
        application_type=body.application_type.value,
        risk_tier=body.risk_tier.value,
        active=body.active,
    )
    TenantRepository(session, AISystem).add(principal.organization_id, system)
    if body.profile is not None:
        profile_input = body.profile
        profile = ApplicationProfile(
            application_id=system.id,
            organization_id=principal.organization_id,
            name=system.name,
            description=system.description or "",
            version=profile_input.version,
            application_type=body.application_type,
            intended_use=profile_input.intended_use,
            clinical_specialty=profile_input.clinical_specialty,
            input_type=profile_input.input_type,
            output_type=profile_input.output_type,
            patient_facing=profile_input.patient_facing,
            clinician_facing=profile_input.clinician_facing,
            uses_rag=profile_input.uses_rag,
            uses_tools=profile_input.uses_tools,
            uses_llm=profile_input.uses_llm,
            uses_predictive_model=profile_input.uses_predictive_model,
            environment=profile_input.environment,
            risk_tier=body.risk_tier,
            endpoint_url=profile_input.endpoint_url,
            authentication_type=profile_input.authentication_type,
            request_schema=profile_input.request_schema,
            response_schema=profile_input.response_schema,
        )
        TenantRepository(session, ApplicationProfileRecord).add(
            principal.organization_id,
            ApplicationProfileRecord(
                organization_id=principal.organization_id,
                system_id=system.id,
                profile_version=profile.version,
                profile=profile.model_dump(mode="json"),
            ),
        )
    _audit(
        session,
        request,
        principal,
        action="system.created",
        target_type="ai_system",
        target_id=system.id,
    )
    session.commit()
    return _system_payload(session, principal.organization_id, system)


@router.get("/systems")
def list_systems(
    principal: ApiKeyPrincipal = Depends(systems_read),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    systems = TenantRepository(session, AISystem).list(principal.organization_id, limit=limit)
    payload = [
        _system_payload(session, principal.organization_id, item)
        for item in systems
        if _can_read_system(session, principal, item.id)
    ]
    session.commit()
    return {"items": payload, "count": len(payload)}


@router.get("/systems/{system_id}")
def get_system(
    system_id: str,
    principal: ApiKeyPrincipal = Depends(systems_read),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    system = TenantRepository(session, AISystem).get(principal.organization_id, system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    payload = _system_payload(session, principal.organization_id, system)
    session.commit()
    return payload


@router.post("/systems/{system_id}/versions", status_code=201)
def create_system_version(
    system_id: str,
    body: SystemVersionCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(systems_create),
    session: Session = Depends(get_db_session),
    encryptor: EncryptedJsonStore = Depends(get_encryptor),
) -> dict[str, Any]:
    _require_environment_access(principal, body.environment.value)
    system = TenantRepository(session, AISystem).get(principal.organization_id, system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    _require_active_system(system)
    authentication_config_encrypted = None
    if body.authentication_config is not None:
        authentication_config_encrypted = encryptor.encrypt(body.authentication_config)
    version = AISystemVersion(
        organization_id=principal.organization_id,
        system_id=system.id,
        version=body.version,
        environment=body.environment.value,
        endpoint_url=str(body.endpoint_url) if body.endpoint_url else None,
        authentication_config_encrypted=authentication_config_encrypted,
        request_schema=body.request_schema,
        response_schema=body.response_schema,
        active=body.active,
    )
    TenantRepository(session, AISystemVersion).add(principal.organization_id, version)
    _audit(
        session,
        request,
        principal,
        action="system_version.created",
        target_type="ai_system_version",
        target_id=version.id,
    )
    session.commit()
    return _system_version_payload(version)


def _evaluation_payload(
    session: Session,
    encryptor: EncryptedJsonStore,
    organization_id: str,
    evaluation: Evaluation,
) -> dict[str, Any]:
    if evaluation.execution_status == ExecutionStatus.COMPLETED.value:
        result = EvaluationService(session, encryptor).get(organization_id, evaluation.id)
        if result is None:
            raise GatekeeperError(
                ErrorCode.DATABASE_UNAVAILABLE,
                "The completed evaluation artifact could not be reconstructed.",
                retryable=True,
                http_status=503,
            )
        return result.model_dump(mode="json")
    return {
        "evaluation_id": evaluation.id,
        "execution_status": evaluation.execution_status,
        "coverage": evaluation.coverage,
        "faithfulness": evaluation.metrics,
        "findings": [],
        "source_claims": [],
        "generated_claims": [],
        "versions": evaluation.component_versions,
        "error_code": evaluation.error_code,
        "release_decision": None,
    }


def _replay_evaluation(
    *,
    replay: IdempotencyReplay,
    principal: ApiKeyPrincipal,
    session: Session,
    encryptor: EncryptedJsonStore,
) -> JSONResponse:
    row = TenantRepository(session, Evaluation).get(
        principal.organization_id,
        replay.resource_id,
    )
    if row is None:
        raise GatekeeperError(
            ErrorCode.DATABASE_UNAVAILABLE,
            "The idempotent evaluation artifact is unavailable.",
            retryable=True,
            http_status=503,
        )
    if not _can_read_environment(principal, row.environment):
        raise _not_found("EVALUATION_NOT_FOUND", "evaluation")
    if replay.response_status >= 400:
        session.commit()
        raise GatekeeperError(
            ErrorCode.WORKER_FAILED,
            "Assurance could not complete the evaluation.",
            retryable=False,
            http_status=replay.response_status,
            resource_id=row.id,
        )
    response_body = _evaluation_payload(
        session,
        encryptor,
        principal.organization_id,
        row,
    )
    session.commit()
    return JSONResponse(
        status_code=replay.response_status,
        content=jsonable_encoder(response_body),
        headers={"Idempotent-Replayed": "true"},
    )


def _record_evaluation_idempotency(
    *,
    idempotency: IdempotencyService,
    organization_id: str,
    key: str,
    payload: dict[str, Any],
    resource_id: str,
    response_status: int,
) -> IdempotencyReplay | None:
    try:
        idempotency.record(
            organization_id=organization_id,
            operation="POST /v1/evaluations",
            key=key,
            payload=payload,
            resource_type="evaluation",
            resource_id=resource_id,
            response_status=response_status,
        )
    except IntegrityError:
        # The evaluation and idempotency row share a transaction. A competing
        # winner therefore rolls this transaction back without leaving a duplicate
        # evaluation, after which its canonical record can be replayed.
        idempotency.session.rollback()
        replay = idempotency.find(
            organization_id=organization_id,
            operation="POST /v1/evaluations",
            key=key,
            payload=payload,
        )
        if replay is None:
            raise GatekeeperError(
                ErrorCode.DATABASE_UNAVAILABLE,
                "The idempotent evaluation could not be resolved.",
                retryable=True,
                http_status=503,
            ) from None
        return replay
    return None


@router.post(
    "/evaluations",
    status_code=201,
    response_model=CompareResult,
    response_description="Completed clinical-faithfulness evaluation",
    responses={
        201: {
            "headers": {
                "Location": {
                    "description": "Canonical URL of the persisted evaluation",
                    "schema": {"type": "string"},
                }
            }
        },
        401: {"model": ApiErrorResponse, "description": "Authentication failed"},
        403: {"model": ApiErrorResponse, "description": "Scope or environment denied"},
        404: {"model": ApiErrorResponse, "description": "System or version not found"},
        409: {"model": ApiErrorResponse, "description": "Idempotency conflict"},
        413: {"model": ApiErrorResponse, "description": "Request or evidence is too large"},
        422: {"model": ApiErrorResponse, "description": "Evidence request is invalid"},
        429: {"model": ApiErrorResponse, "description": "Request rate limit exceeded"},
        500: {"model": ApiErrorResponse, "description": "Evaluation failed"},
        503: {"model": ApiErrorResponse, "description": "Persistence service unavailable"},
    },
)
def create_evaluation(
    body: Annotated[
        SingleEvaluationCreate,
        Body(openapi_examples=EVALUATION_OPENAPI_EXAMPLES),
    ],
    request: Request,
    principal: ApiKeyPrincipal = Depends(evaluations_create),
    session: Session = Depends(get_db_session),
    encryptor: EncryptedJsonStore = Depends(get_encryptor),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    _require_environment_access(principal, body.environment.value)
    # Preserve Decimal-vs-string evidence identity for idempotency and ingestion.
    payload = body.model_dump(mode="python")
    idempotency = IdempotencyService(session, request.app.state.settings.api_key_pepper)
    if idempotency_key is not None:
        replay = idempotency.find(
            organization_id=principal.organization_id,
            operation="POST /v1/evaluations",
            key=idempotency_key,
            payload=payload,
        )
        if replay is not None:
            return _replay_evaluation(
                replay=replay,
                principal=principal,
                session=session,
                encryptor=encryptor,
            )

    system = TenantRepository(session, AISystem).get(principal.organization_id, body.system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    _require_active_system(system)
    registered_version = _resolve_system_version(
        session,
        principal.organization_id,
        system.id,
        body.system_version,
        body.environment.value,
    )
    if registered_version is None:
        raise _not_found(ErrorCode.VERSION_NOT_FOUND.value, "AI system version")
    _require_environment_access(principal, registered_version.environment)

    service_request = SingleEvaluationRequest.model_validate(
        body.model_dump(mode="python", exclude={"environment"})
    )
    runtime_settings = request.app.state.settings
    operational_metrics = request.app.state.operational_metrics
    service = EvaluationService(
        session,
        encryptor,
        compare_fn=compare,
        retention_days=runtime_settings.clinical_data_retention_days,
        metrics=operational_metrics,
    )
    service.configure_ingestion(
        UniversalEvidenceIngestionService(IngestionLimits.from_settings(runtime_settings)),
        nlp_service=ClinicalNLPService(
            enable_hf=(runtime_settings.ner_backend == "huggingface")
        ),
    )
    try:
        result = service.run(
            principal.organization_id,
            service_request,
            system_version_id=registered_version.id,
            environment=registered_version.environment,
        )
    except GatekeeperError as exc:
        # EvaluationService records technical failure as an explicit UNASSESSED artifact.
        if idempotency_key is not None and exc.resource_id is not None:
            replay = _record_evaluation_idempotency(
                idempotency=idempotency,
                organization_id=principal.organization_id,
                key=idempotency_key,
                payload=payload,
                resource_id=exc.resource_id,
                response_status=exc.http_status,
            )
            if replay is not None:
                return _replay_evaluation(
                    replay=replay,
                    principal=principal,
                    session=session,
                    encryptor=encryptor,
                )
        session.commit()
        raise
    ingestion_result = service.last_ingestion_result
    if ingestion_result is not None:
        stats = ingestion_result.stats
        operational_metrics.observe(MetricName.INGESTION_PAYLOAD_BYTES, stats.payload_bytes)
        operational_metrics.observe(MetricName.INGESTION_SOURCE_COUNT, stats.source_count)
        operational_metrics.observe(
            MetricName.INGESTION_EVIDENCE_UNITS,
            stats.evidence_unit_count,
        )
        operational_metrics.observe(MetricName.INGESTION_CHUNKS, stats.chunk_count)
        operational_metrics.observe(
            MetricName.INGESTION_NORMALIZATION_LATENCY,
            stats.normalization_seconds,
        )
        operational_metrics.observe(
            MetricName.INGESTION_LATENCY,
            stats.ingestion_seconds,
        )
        logger.info(
            "evidence_ingestion_completed request_id=%s evaluation_id=%s "
            "organization_id=%s source_count=%d source_types=%s payload_bytes=%d "
            "chunk_count=%d claim_count=%d coverage=%s processing_seconds=%.6f",
            _request_id(request),
            result.evaluation_id,
            principal.organization_id,
            stats.source_count,
            ",".join(stats.source_types),
            stats.payload_bytes,
            stats.chunk_count,
            len(result.source_claims) + len(result.generated_claims),
            result.coverage.status.value,
            service.last_processing_seconds or 0.0,
        )
    if idempotency_key is not None:
        replay = _record_evaluation_idempotency(
            idempotency=idempotency,
            organization_id=principal.organization_id,
            key=idempotency_key,
            payload=payload,
            resource_id=result.evaluation_id,
            response_status=201,
        )
        if replay is not None:
            return _replay_evaluation(
                replay=replay,
                principal=principal,
                session=session,
                encryptor=encryptor,
            )
    _audit(
        session,
        request,
        principal,
        action="evaluation.created",
        target_type="evaluation",
        target_id=result.evaluation_id,
    )
    session.commit()
    return JSONResponse(
        status_code=201,
        content=jsonable_encoder(result.model_dump(mode="json")),
        headers={"Location": f"/v1/evaluations/{result.evaluation_id}"},
    )


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(
    evaluation_id: str,
    principal: ApiKeyPrincipal = Depends(evaluations_read),
    session: Session = Depends(get_db_session),
    encryptor: EncryptedJsonStore = Depends(get_encryptor),
) -> dict[str, Any]:
    row = TenantRepository(session, Evaluation).get(principal.organization_id, evaluation_id)
    if row is None:
        raise _not_found("EVALUATION_NOT_FOUND", "evaluation")
    if not _can_read_environment(principal, row.environment):
        raise _not_found("EVALUATION_NOT_FOUND", "evaluation")
    payload = _evaluation_payload(session, encryptor, principal.organization_id, row)
    session.commit()
    return payload


def _slug_for(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return slug[:160] or "test-suite"


def _latest_suite_version(
    session: Session,
    organization_id: str,
    suite_id: str,
    version: str | None = None,
) -> TestSuiteVersion | None:
    statement = select(TestSuiteVersion).where(
        TestSuiteVersion.organization_id == organization_id,
        TestSuiteVersion.test_suite_id == suite_id,
    )
    if version is not None:
        statement = statement.where(TestSuiteVersion.version == version)
    return session.scalar(statement.order_by(TestSuiteVersion.created_at.desc()))


def _require_executable_suite_version(version: TestSuiteVersion) -> None:
    if version.status not in {"ACTIVE", "PUBLISHED"}:
        raise ApiError(
            "TEST_SUITE_VERSION_NOT_EXECUTABLE",
            "Only active or published test-suite versions can be planned or executed.",
            http_status=409,
        )


def _suite_cases(
    session: Session,
    organization_id: str,
    suite_id: str,
    suite_version: TestSuiteVersion | None,
) -> list[tuple[TestCase, TestCaseVersion]]:
    if suite_version is None:
        return []
    return [
        pair
        for pair in _source_suite_case_snapshot(
            session,
            organization_id,
            suite_id,
            suite_version,
        )
        if pair[0].active
    ]


def _source_suite_case_snapshot(
    session: Session,
    organization_id: str,
    suite_id: str,
    suite_version: TestSuiteVersion,
) -> list[tuple[TestCase, TestCaseVersion]]:
    raw_version_ids = suite_version.definition.get(CASE_VERSION_IDS_KEY)
    version_ids = case_version_ids_from_definition(suite_version.definition)
    if (
        not isinstance(raw_version_ids, list)
        or not all(isinstance(item, str) and item for item in raw_version_ids)
        or len(raw_version_ids) != len(set(raw_version_ids))
        or tuple(raw_version_ids) != version_ids
    ):
        raise ApiError(
            "TEST_SUITE_SNAPSHOT_INVALID",
            "The test-suite version does not contain a valid immutable case snapshot.",
            http_status=409,
        )
    rows = session.execute(
        select(TestCase, TestCaseVersion)
        .join(TestCaseVersion, TestCaseVersion.test_case_id == TestCase.id)
        .where(
            TestCase.organization_id == organization_id,
            TestCase.test_suite_id == suite_id,
            TestCaseVersion.organization_id == organization_id,
            TestCaseVersion.id.in_(version_ids),
        )
    ).all()
    by_version_id = {version.id: (case, version) for case, version in rows}
    if len(by_version_id) != len(version_ids):
        raise ApiError(
            "TEST_SUITE_SNAPSHOT_INVALID",
            "The test-suite case snapshot is missing or crosses a tenant or suite boundary.",
            http_status=409,
        )
    return [by_version_id[item] for item in version_ids]


def _case_definition_payload(case: TestCase, version: TestCaseVersion) -> dict[str, Any]:
    route = EvaluatorRouter().route(case.test_family)
    return {
        "id": case.id,
        "case_id": case.id,
        "name": case.name,
        "test_family": case.test_family,
        "active": case.active,
        "version_id": version.id,
        "version": version.version,
        "input_data": version.input_data,
        "expectation": version.expectation,
        "evaluator": route.model_dump(mode="json"),
    }


def _suite_payload(
    session: Session,
    organization_id: str,
    suite: TestSuite,
) -> dict[str, Any]:
    version = _latest_suite_version(session, organization_id, suite.id)
    cases = _suite_cases(session, organization_id, suite.id, version)
    return {
        "id": suite.id,
        "test_suite_id": suite.id,
        "slug": suite.slug,
        "name": suite.name,
        "suite_type": suite.suite_type,
        "active": suite.active,
        "created_at": suite.created_at,
        "updated_at": suite.updated_at,
        "latest_version": (
            {
                "id": version.id,
                "version": version.version,
                "definition": version.definition,
                "status": version.status,
                "created_at": version.created_at,
            }
            if version is not None
            else None
        ),
        "cases": [_case_definition_payload(case, item) for case, item in cases],
    }


@router.post("/test-suites", status_code=201)
def create_test_suite(
    body: TestSuiteCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(test_suites_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    suite = TestSuite(
        organization_id=principal.organization_id,
        slug=body.slug or _slug_for(body.name),
        name=body.name,
        suite_type=body.suite_type,
        active=body.active,
    )
    TenantRepository(session, TestSuite).add(principal.organization_id, suite)
    case_version_ids: list[str] = []
    for case_input in body.cases:
        case = TestCase(
            organization_id=principal.organization_id,
            test_suite_id=suite.id,
            test_family=case_input.test_family.value,
            name=case_input.name,
            active=case_input.active,
        )
        TenantRepository(session, TestCase).add(principal.organization_id, case)
        case_version = TenantRepository(session, TestCaseVersion).add(
            principal.organization_id,
            TestCaseVersion(
                organization_id=principal.organization_id,
                test_case_id=case.id,
                version=case_input.version,
                input_data=case_input.input_data,
                expectation=case_input.expectation,
            ),
        )
        case_version_ids.append(case_version.id)
    definition = with_case_version_snapshot(
        {
            **body.definition,
            "provenance": {
                "data_classification": body.data_classification,
                "contains_phi": body.contains_phi,
                "no_phi_attested": body.no_phi_attested,
                "attested_by_api_key_id": principal.key_id,
            },
        },
        case_version_ids,
    )
    TenantRepository(session, TestSuiteVersion).add(
        principal.organization_id,
        TestSuiteVersion(
            organization_id=principal.organization_id,
            test_suite_id=suite.id,
            version=body.version,
            definition=definition,
            status=body.status,
        ),
    )
    _audit(
        session,
        request,
        principal,
        action="test_suite.created",
        target_type="test_suite",
        target_id=suite.id,
    )
    session.commit()
    return _suite_payload(session, principal.organization_id, suite)


@router.get("/test-suites")
def list_test_suites(
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    suites = TenantRepository(session, TestSuite).list(principal.organization_id, limit=limit)
    payload = [_suite_payload(session, principal.organization_id, suite) for suite in suites]
    session.commit()
    return {"items": payload, "count": len(payload)}


@router.get("/test-suites/{suite_id}")
def get_test_suite(
    suite_id: str,
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    suite = _resolve_suite(session, principal.organization_id, suite_id)
    if suite is None:
        raise _not_found("TEST_SUITE_NOT_FOUND", "test suite")
    payload = _suite_payload(session, principal.organization_id, suite)
    session.commit()
    return payload


def _resolve_system_version(
    session: Session,
    organization_id: str,
    system_id: str,
    version: str,
    environment: str,
) -> AISystemVersion | None:
    return session.scalar(
        select(AISystemVersion).where(
            AISystemVersion.organization_id == organization_id,
            AISystemVersion.system_id == system_id,
            AISystemVersion.version == version,
            AISystemVersion.environment == environment,
            AISystemVersion.active.is_(True),
        )
    )


def _resolve_suite(
    session: Session,
    organization_id: str,
    identity: str,
) -> TestSuite | None:
    internal = session.scalar(
        select(TestSuite).where(
            TestSuite.organization_id == organization_id,
            TestSuite.id == identity,
            TestSuite.active.is_(True),
        )
    )
    if internal is not None:
        return internal
    return session.scalar(
        select(TestSuite).where(
            TestSuite.organization_id == organization_id,
            TestSuite.slug == identity,
            TestSuite.active.is_(True),
        )
    )


def _resolve_policy(
    session: Session,
    organization_id: str,
    identity: str,
) -> ReleasePolicyRecord | None:
    internal = session.scalar(
        select(ReleasePolicyRecord).where(
            ReleasePolicyRecord.organization_id == organization_id,
            ReleasePolicyRecord.id == identity,
            ReleasePolicyRecord.active.is_(True),
        )
    )
    if internal is not None:
        return internal
    aliases = tuple(
        session.scalars(
            select(ReleasePolicyRecord)
            .where(
                ReleasePolicyRecord.organization_id == organization_id,
                (ReleasePolicyRecord.policy_key == identity)
                | (ReleasePolicyRecord.name == identity),
                ReleasePolicyRecord.active.is_(True),
            )
            .limit(2)
        )
    )
    return aliases[0] if len(aliases) == 1 else None


def _latest_policy_version(
    session: Session,
    organization_id: str,
    policy_id: str,
    version: str | None = None,
) -> ReleasePolicyVersion | None:
    statement = select(ReleasePolicyVersion).where(
        ReleasePolicyVersion.organization_id == organization_id,
        ReleasePolicyVersion.release_policy_id == policy_id,
    )
    if version is not None:
        statement = statement.where(ReleasePolicyVersion.version == version)
    return session.scalar(statement.order_by(ReleasePolicyVersion.created_at.desc()))


def _canonical_snapshot_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _release_policy_content_digest(version: ReleasePolicyVersion) -> str:
    return _canonical_snapshot_digest(
        {
            "id": version.id,
            "release_policy_id": version.release_policy_id,
            "version": version.version,
            "rules": version.rules,
            "required_coverage": version.required_coverage,
        }
    )


def _test_suite_content_digest(
    suite_version: TestSuiteVersion,
    cases: list[tuple[TestCase, TestCaseVersion]],
) -> str:
    return _canonical_snapshot_digest(
        {
            "suite_version_id": suite_version.id,
            "suite_version": suite_version.version,
            "definition": suite_version.definition,
            "cases": [
                {
                    "case_id": case.id,
                    "case_name": case.name,
                    "test_family": case.test_family,
                    "active": case.active,
                    "case_version_id": case_version.id,
                    "case_version": case_version.version,
                    "input_data": case_version.input_data,
                    "expectation": case_version.expectation,
                }
                for case, case_version in cases
            ],
        }
    )


def _test_plan_payload(record: TestPlanRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "test_plan_id": record.id,
        **record.definition,
        "confirmed_by_api_key_id": record.confirmed_by_api_key_id,
        "confirmed_at": record.confirmed_at,
    }


def _validated_test_plan(record: TestPlanRecord) -> TestPlan:
    try:
        plan = TestPlan.model_validate(record.definition)
    except ValidationError as exc:
        raise GatekeeperError(
            ErrorCode.DATABASE_UNAVAILABLE,
            "The persisted test plan could not be reconstructed.",
            retryable=True,
            http_status=503,
        ) from exc
    if (
        plan.plan_id != record.id
        or plan.application_id != record.system_id
        or plan.organization_id != record.organization_id
    ):
        raise GatekeeperError(
            ErrorCode.DATABASE_UNAVAILABLE,
            "The persisted test plan identity is inconsistent.",
            retryable=True,
            http_status=503,
        )
    return plan


def _can_read_test_plan(principal: ApiKeyPrincipal, plan: TestPlan) -> bool:
    return _can_read_environment(principal, plan.application_environment.value)


def _test_plan_confirmation_query(
    organization_id: str,
    plan_id: str,
) -> Select[tuple[TestPlanRecord]]:
    """Lock confirmation ownership until the surrounding transaction commits."""

    return (
        select(TestPlanRecord)
        .where(
            TestPlanRecord.organization_id == organization_id,
            TestPlanRecord.id == plan_id,
        )
        .with_for_update()
    )


@router.get("/test-plans")
def list_test_plans(
    principal: ApiKeyPrincipal = Depends(test_plans_manage),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    records = tuple(
        session.scalars(
            select(TestPlanRecord)
            .where(TestPlanRecord.organization_id == principal.organization_id)
            .order_by(TestPlanRecord.created_at.desc(), TestPlanRecord.id.desc())
            .limit(limit)
        )
    )
    items = []
    for record in records:
        plan = _validated_test_plan(record)
        if _can_read_test_plan(principal, plan):
            items.append(_test_plan_payload(record))
    session.commit()
    return {"items": items, "count": len(items)}


@router.get("/test-plans/{plan_id}")
def get_test_plan(
    plan_id: str,
    principal: ApiKeyPrincipal = Depends(test_plans_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    record = TenantRepository(session, TestPlanRecord).get(principal.organization_id, plan_id)
    if record is None:
        raise _not_found("TEST_PLAN_NOT_FOUND", "test plan")
    plan = _validated_test_plan(record)
    if not _can_read_test_plan(principal, plan):
        raise _not_found("TEST_PLAN_NOT_FOUND", "test plan")
    payload = _test_plan_payload(record)
    session.commit()
    return payload


def _bind_suite_cases_to_plan(
    template: TestPlan,
    source_cases: list[tuple[TestCase, TestCaseVersion]],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[SuiteCaseBinding, ...],
    tuple[SuiteCaseExclusion, ...],
]:
    """Select a deterministic, quota-bounded executable snapshot in suite order."""

    quotas = {
        item.test_family: item.cases
        for item in [*template.required, *template.recommended, *template.optional]
    }
    selected_counts: defaultdict[TestFamily, int] = defaultdict(int)
    source_ids: list[str] = []
    executable_ids: list[str] = []
    bindings: list[SuiteCaseBinding] = []
    exclusions: list[SuiteCaseExclusion] = []
    for case, version in source_cases:
        source_ids.append(version.id)
        try:
            family = TestFamily(case.test_family)
        except ValueError as exc:
            raise ApiError(
                "TEST_SUITE_CASE_FAMILY_INVALID",
                "A snapshotted test case has an unsupported test family.",
                http_status=409,
            ) from exc

        reason: SuiteCaseExclusionReason | None = None
        if family in template.not_applicable:
            reason = SuiteCaseExclusionReason.NOT_APPLICABLE
        elif family in template.unselected_optional:
            reason = SuiteCaseExclusionReason.NOT_SELECTED
        elif family not in quotas:
            raise ApiError(
                "TEST_PLAN_PARTITION_INVALID",
                "A suite case does not map to a selected or excluded plan family.",
                http_status=409,
            )
        elif not case.active:
            reason = SuiteCaseExclusionReason.INACTIVE
        elif selected_counts[family] >= quotas[family]:
            reason = SuiteCaseExclusionReason.QUOTA_EXCEEDED

        if reason is None:
            executable_ids.append(version.id)
            bindings.append(
                SuiteCaseBinding(
                    case_version_id=version.id,
                    test_family=family,
                )
            )
            selected_counts[family] += 1
        else:
            exclusions.append(
                SuiteCaseExclusion(
                    case_version_id=version.id,
                    test_family=family,
                    reason=reason,
                )
            )
    return (
        tuple(source_ids),
        tuple(executable_ids),
        tuple(bindings),
        tuple(exclusions),
    )


def _configuration_from_plan(plan: TestPlan) -> CustomerTestConfiguration:
    planned = [*plan.required, *plan.recommended, *plan.optional]
    return CustomerTestConfiguration(
        case_counts={item.test_family: item.cases for item in planned},
        include_optional={item.test_family for item in plan.optional},
    )


def _canonical_plan_definition(plan: TestPlan) -> dict[str, Any]:
    return plan.model_dump(
        mode="json",
        exclude={"status", "created_at", "confirmed_at", "confirmed_by"},
    )


def _generate_suite_bound_plan(
    *,
    session: Session,
    organization_id: str,
    profile: ApplicationProfile,
    release_policy: ReleasePolicyReference,
    release_policy_snapshot_digest: str | None,
    suite: TestSuite,
    suite_version: TestSuiteVersion,
    test_plan_version: str,
    configuration: CustomerTestConfiguration | None = None,
) -> TestPlan:
    selection = select_tests(profile)
    try:
        template = generate_test_plan(
            profile=profile,
            selection=selection,
            release_policy=release_policy,
            release_policy_snapshot_digest=release_policy_snapshot_digest,
            configuration=configuration,
            test_plan_version=test_plan_version,
        )
    except ValueError as exc:
        raise ApiError(
            "TEST_PLAN_CONFIGURATION_INVALID",
            str(exc),
            http_status=422,
        ) from exc
    source_cases = _source_suite_case_snapshot(
        session,
        organization_id,
        suite.id,
        suite_version,
    )
    source_ids, executable_ids, bindings, exclusions = _bind_suite_cases_to_plan(
        template,
        source_cases,
    )
    return generate_test_plan(
        profile=profile,
        selection=selection,
        release_policy=release_policy,
        release_policy_snapshot_digest=release_policy_snapshot_digest,
        test_suite_id=suite.id,
        test_suite_version=suite_version.version,
        test_suite_content_snapshot_digest=_test_suite_content_digest(suite_version, source_cases),
        source_suite_case_version_ids=source_ids,
        suite_case_version_ids=executable_ids,
        suite_case_bindings=bindings,
        excluded_suite_cases=exclusions,
        configuration=configuration,
        test_plan_version=test_plan_version,
    )


def _validate_plan_source_snapshot(
    *,
    session: Session,
    organization_id: str,
    suite: TestSuite,
    suite_version: TestSuiteVersion,
    plan: TestPlan,
) -> list[tuple[TestCase, TestCaseVersion]]:
    source_cases = _source_suite_case_snapshot(
        session,
        organization_id,
        suite.id,
        suite_version,
    )
    current_ids = tuple(version.id for _, version in source_cases)
    if (
        current_ids != plan.source_suite_case_version_ids
        or case_version_snapshot_digest(current_ids) != plan.source_suite_case_snapshot_digest
        or _test_suite_content_digest(suite_version, source_cases)
        != plan.test_suite_content_snapshot_digest
    ):
        raise ApiError(
            "TEST_PLAN_STALE",
            "The source suite case snapshot changed after this plan was drafted.",
            http_status=409,
        )
    return source_cases


def _confirmed_plan_case_snapshot(
    *,
    session: Session,
    organization_id: str,
    suite: TestSuite,
    suite_version: TestSuiteVersion,
    plan: TestPlan,
) -> list[tuple[TestCase, TestCaseVersion]]:
    source_cases = _validate_plan_source_snapshot(
        session=session,
        organization_id=organization_id,
        suite=suite,
        suite_version=suite_version,
        plan=plan,
    )
    _, current_ids, current_bindings, current_exclusions = _bind_suite_cases_to_plan(
        plan,
        source_cases,
    )
    if (
        current_ids != plan.suite_case_version_ids
        or current_bindings != plan.suite_case_bindings
        or current_exclusions != plan.excluded_suite_cases
    ):
        raise ApiError(
            "TEST_PLAN_STALE",
            "The suite case applicability or quota allocation changed after confirmation.",
            http_status=409,
        )
    by_version_id = {version.id: (case, version) for case, version in source_cases}
    executable: list[tuple[TestCase, TestCaseVersion]] = []
    selected_counts: defaultdict[TestFamily, int] = defaultdict(int)
    quotas = {
        item.test_family: item.cases for item in [*plan.required, *plan.recommended, *plan.optional]
    }
    for binding in plan.suite_case_bindings:
        version_id = binding.case_version_id
        pair = by_version_id.get(version_id)
        if pair is None or not pair[0].active:
            raise ApiError(
                "TEST_PLAN_STALE",
                "A confirmed executable case is missing or inactive.",
                http_status=409,
            )
        try:
            family = TestFamily(pair[0].test_family)
        except ValueError as exc:
            raise ApiError(
                "TEST_PLAN_STALE",
                "A confirmed executable case has an unsupported family.",
                http_status=409,
            ) from exc
        if family is not binding.test_family:
            raise ApiError(
                "TEST_PLAN_STALE",
                "A confirmed executable case changed test family.",
                http_status=409,
            )
        if family not in quotas or selected_counts[family] >= quotas[family]:
            raise ApiError(
                "TEST_PLAN_STALE",
                "The confirmed executable case snapshot exceeds its selected plan quota.",
                http_status=409,
            )
        selected_counts[family] += 1
        executable.append(pair)
    return executable


@router.post("/systems/{system_id}/test-plans", status_code=201)
def create_test_plan_draft(
    system_id: str,
    body: TestPlanDraftCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(test_plans_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    system = TenantRepository(session, AISystem).get(principal.organization_id, system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    _require_active_system(system)
    _, profile = _profile_for_system(session, principal.organization_id, system.id)
    if profile is None:
        raise ApiError(
            "APPLICATION_PROFILE_REQUIRED",
            "A versioned application profile is required before planning.",
            http_status=409,
        )
    _require_environment_access(principal, profile.environment.value)
    policy = _resolve_policy(session, principal.organization_id, body.release_policy_id)
    if policy is None:
        raise _not_found("RELEASE_POLICY_NOT_FOUND", "release policy")
    policy_version = _latest_policy_version(
        session,
        principal.organization_id,
        policy.id,
        body.release_policy_version,
    )
    if policy_version is None:
        raise _not_found("RELEASE_POLICY_VERSION_NOT_FOUND", "release policy version")
    suite = _resolve_suite(session, principal.organization_id, body.test_suite_id)
    if suite is None:
        raise _not_found("TEST_SUITE_NOT_FOUND", "test suite")
    suite_version = _latest_suite_version(
        session,
        principal.organization_id,
        suite.id,
        body.test_suite_version,
    )
    if suite_version is None:
        raise _not_found("TEST_SUITE_VERSION_NOT_FOUND", "test suite version")
    _require_executable_suite_version(suite_version)
    plan = _generate_suite_bound_plan(
        session=session,
        organization_id=principal.organization_id,
        profile=profile,
        release_policy=ReleasePolicyReference(
            policy_id=policy.id,
            policy_version=policy_version.version,
        ),
        release_policy_snapshot_digest=_release_policy_content_digest(policy_version),
        suite=suite,
        suite_version=suite_version,
        test_plan_version=body.test_plan_version,
        configuration=CustomerTestConfiguration(
            case_counts=body.case_counts,
            include_optional=body.include_optional,
        ),
    )
    record = TenantRepository(session, TestPlanRecord).get(principal.organization_id, plan.plan_id)
    if record is None:
        record = TenantRepository(session, TestPlanRecord).add(
            principal.organization_id,
            TestPlanRecord(
                id=plan.plan_id,
                organization_id=principal.organization_id,
                system_id=system.id,
                version=plan.test_plan_version,
                definition=plan.model_dump(mode="json"),
            ),
        )
        _audit(
            session,
            request,
            principal,
            action="test_plan.drafted",
            target_type="test_plan",
            target_id=record.id,
        )
    session.commit()
    return _test_plan_payload(record)


@router.post("/test-plans/{plan_id}/confirm")
def confirm_test_plan_route(
    plan_id: str,
    body: TestPlanConfirm,
    request: Request,
    principal: ApiKeyPrincipal = Depends(test_plans_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    record = session.scalar(_test_plan_confirmation_query(principal.organization_id, plan_id))
    if record is None:
        raise _not_found("TEST_PLAN_NOT_FOUND", "test plan")
    system = TenantRepository(session, AISystem).get(principal.organization_id, record.system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    _require_active_system(system)
    plan = _validated_test_plan(record)
    if (
        plan.profile_fingerprint != body.expected_profile_fingerprint
        or plan.test_plan_version != body.expected_test_plan_version
    ):
        raise ApiError(
            "TEST_PLAN_CONFIRMATION_MISMATCH",
            "The confirmation does not match the reviewed test plan.",
            http_status=409,
        )
    _, profile = _profile_for_system(session, principal.organization_id, record.system_id)
    if profile is None:
        raise ApiError(
            "APPLICATION_PROFILE_REQUIRED",
            "The application profile is unavailable.",
            http_status=409,
        )
    _require_environment_access(principal, profile.environment.value)
    policy = TenantRepository(session, ReleasePolicyRecord).get(
        principal.organization_id, plan.release_policy.policy_id
    )
    if policy is None:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The release policy referenced by this plan is unavailable.",
            http_status=409,
        )
    if not policy.active:
        raise ApiError(
            ErrorCode.RELEASE_POLICY_INACTIVE.value,
            "The release policy referenced by this plan is inactive.",
            http_status=409,
        )
    policy_version = _latest_policy_version(
        session,
        principal.organization_id,
        policy.id,
        plan.release_policy.policy_version,
    )
    if policy_version is None:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The release-policy version referenced by this plan is unavailable.",
            http_status=409,
        )
    if plan.test_suite_id is None or plan.test_suite_version is None:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The test plan does not contain an immutable test-suite reference.",
            http_status=409,
        )
    suite = _resolve_suite(session, principal.organization_id, plan.test_suite_id)
    suite_version = (
        _latest_suite_version(
            session,
            principal.organization_id,
            suite.id,
            plan.test_suite_version,
        )
        if suite is not None
        else None
    )
    if suite is None or suite_version is None:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The test suite referenced by this plan is unavailable.",
            http_status=409,
        )
    _require_executable_suite_version(suite_version)
    current = _generate_suite_bound_plan(
        session=session,
        organization_id=principal.organization_id,
        profile=profile,
        release_policy=plan.release_policy,
        release_policy_snapshot_digest=_release_policy_content_digest(policy_version),
        suite=suite,
        suite_version=suite_version,
        test_plan_version=plan.test_plan_version,
        configuration=_configuration_from_plan(plan),
    )
    if _canonical_plan_definition(current) != _canonical_plan_definition(plan):
        raise ApiError(
            "TEST_PLAN_STALE",
            "The profile, selection rules, suite, or policy changed after drafting.",
            http_status=409,
        )
    if plan.status is TestPlanStatus.CONFIRMED:
        if record.confirmed_by_api_key_id == principal.key_id:
            session.commit()
            return _test_plan_payload(record)
        raise ApiError(
            "TEST_PLAN_ALREADY_CONFIRMED",
            "The test plan was already confirmed by a different actor.",
            http_status=409,
        )
    confirmed = confirm_test_plan(plan, confirmed_by=f"api_key:{principal.key_id}")
    record.definition = confirmed.model_dump(mode="json")
    record.confirmed_by_api_key_id = principal.key_id
    record.confirmed_at = confirmed.confirmed_at
    _audit(
        session,
        request,
        principal,
        action="test_plan.confirmed",
        target_type="test_plan",
        target_id=record.id,
    )
    session.commit()
    return _test_plan_payload(record)


def _coverage_payload_for_cases(
    *,
    profile: ApplicationProfile | None,
    cases: list[tuple[TestRunCase, TestCase]],
    plan: TestPlan | None = None,
) -> dict[str, Any]:
    planned_families = sorted({case.test_family for _, case in cases})
    completed = sum(item.execution_status == "COMPLETED" for item, _ in cases)
    technical_failed = sum(
        bool(item.error_code) or item.execution_status == "FAILED" for item, _ in cases
    )
    unassessed = sum(
        item.execution_status == "UNASSESSED" and not item.error_code for item, _ in cases
    )
    pending = len(cases) - completed - technical_failed - unassessed
    if profile is None:
        return {
            "engine_version": COVERAGE_ENGINE_VERSION,
            "status": "UNASSESSED",
            "planned_tests": len(planned_families),
            "planned_cases": len(cases),
            "completed_cases": completed,
            "technical_failed_cases": technical_failed,
            "unassessed_cases": unassessed,
            "pending_cases": max(0, pending),
            "missing_required_tests": [],
            "degraded_tests": [],
            "valid_for_release": False,
            "limitations": [
                "Application profile is unavailable; required coverage cannot be assessed."
            ],
        }

    selection = select_tests(profile)
    confirmed_case_counts = (
        {
            item.test_family: item.cases
            for item in [*plan.required, *plan.recommended, *plan.optional]
        }
        if plan is not None
        else {}
    )
    confirmed_requirements = (
        {
            **{item.test_family: TestRequirement.REQUIRED for item in plan.required},
            **{item.test_family: TestRequirement.RECOMMENDED for item in plan.recommended},
            **{item.test_family: TestRequirement.OPTIONAL for item in plan.optional},
        }
        if plan is not None
        else {}
    )
    grouped: dict[TestFamily, list[TestRunCase]] = defaultdict(list)
    for run_case, case in cases:
        grouped[TestFamily(case.test_family)].append(run_case)
    records: list[TestCoverageRecord] = []
    covered_families = set(grouped) | set(selection.required_tests) | set(confirmed_case_counts)
    for family in sorted(covered_families, key=lambda item: item.value):
        run_cases = grouped.get(family, [])
        confirmed_count = confirmed_case_counts.get(family, len(run_cases))
        planned_count = confirmed_count
        missing_confirmed_cases = max(0, confirmed_count - len(run_cases))
        requirement = confirmed_requirements.get(
            family,
            selection.requirement_for(family),
        )
        limitations: list[str] = []
        if requirement is TestRequirement.NOT_APPLICABLE:
            requirement = TestRequirement.OPTIONAL
            limitations.append(
                "The suite includes a family classified not applicable by the current profile."
            )
        route = EvaluatorRouter().route(family)
        if not run_cases and requirement is TestRequirement.REQUIRED:
            limitations.append("The confirmed suite contains no case for this required family.")
        if missing_confirmed_cases:
            limitations.append(
                f"The suite is missing {missing_confirmed_cases} case(s) from the confirmed plan."
            )
        if not route.implemented:
            limitations.append(route.reason or "The evaluator is unavailable.")
        records.append(
            TestCoverageRecord(
                test_family=family,
                requirement=requirement,
                planned_cases=planned_count,
                completed_cases=sum(item.execution_status == "COMPLETED" for item in run_cases),
                technical_failed_cases=sum(
                    bool(item.error_code) or item.execution_status == "FAILED" for item in run_cases
                ),
                unassessed_cases=(
                    missing_confirmed_cases
                    + sum(
                        item.execution_status == "UNASSESSED" and not item.error_code
                        for item in run_cases
                    )
                ),
                evaluator_degraded=not route.implemented,
                limitations=limitations,
            )
        )
    coverage = calculate_coverage(records).model_dump(mode="json")
    if not cases:
        coverage["limitations"].append("The selected test suite contains no versioned cases.")
    return coverage


def _run_case_pairs(
    session: Session,
    organization_id: str,
    run_id: str,
) -> list[tuple[TestRunCase, TestCase]]:
    statement = (
        select(TestRunCase, TestCase)
        .join(TestCaseVersion, TestCaseVersion.id == TestRunCase.test_case_version_id)
        .join(TestCase, TestCase.id == TestCaseVersion.test_case_id)
        .where(
            TestRunCase.organization_id == organization_id,
            TestRunCase.test_run_id == run_id,
            TestCaseVersion.organization_id == organization_id,
            TestCase.organization_id == organization_id,
        )
        .order_by(TestRunCase.created_at, TestRunCase.id)
    )
    return list(session.execute(statement).tuples())


def _run_payload(session: Session, organization_id: str, run: TestRun) -> dict[str, Any]:
    system_version = TenantRepository(session, AISystemVersion).get(
        organization_id, run.system_version_id
    )
    suite_version = TenantRepository(session, TestSuiteVersion).get(
        organization_id, run.test_suite_version_id
    )
    policy_version = (
        TenantRepository(session, ReleasePolicyVersion).get(
            organization_id, run.release_policy_version_id
        )
        if run.release_policy_version_id
        else None
    )
    coverage = run.coverage or {
        "engine_version": COVERAGE_ENGINE_VERSION,
        "status": "UNASSESSED",
        "valid_for_release": False,
        "limitations": ["No persisted coverage snapshot is available."],
    }
    internal = run.trigger_source.get("gatekeeper", {})
    try:
        coverage_report = CoverageReport.model_validate(coverage)
    except ValidationError:
        coverage_report = None
    execution_completed = run.execution_status == ExecutionStatus.COMPLETED.value
    assessed_release_state = (
        execution_completed
        and coverage_report is not None
        and coverage_report.valid_for_release
        and coverage_report.status in {CoverageStatus.FULL, CoverageStatus.REDUCED}
    )
    decision_is_valid = False
    if coverage_report is not None and execution_completed:
        if run.release_decision == ReleaseDecision.PASS.value:
            decision_is_valid = (
                assessed_release_state and coverage_report.status is CoverageStatus.FULL
            )
        elif run.release_decision == ReleaseDecision.WARN.value:
            decision_is_valid = assessed_release_state
        elif run.release_decision == ReleaseDecision.BLOCK.value:
            # A known clinical failure may block before all coverage completes,
            # but a technical evaluator failure can never become a decision.
            decision_is_valid = coverage_report.technical_failed_cases == 0
    return {
        "id": run.id,
        "run_id": run.id,
        "system_id": run.system_id,
        "system_version": system_version.version if system_version else None,
        "system_version_id": run.system_version_id,
        "test_suite_id": suite_version.test_suite_id if suite_version else None,
        "test_suite_version": suite_version.version if suite_version else None,
        "test_suite_version_id": run.test_suite_version_id,
        "release_policy_id": policy_version.release_policy_id if policy_version else None,
        "release_policy_version": policy_version.version if policy_version else None,
        "release_policy_version_id": run.release_policy_version_id,
        "test_plan_id": internal.get("test_plan_id"),
        "test_plan_version": internal.get("test_plan_version"),
        "application_profile_id": internal.get("application_profile_id"),
        "application_profile_version": internal.get("application_profile_version"),
        "planning_status": internal.get("test_plan_status", "UNASSESSED"),
        "execution_status": run.execution_status,
        "assessment_status": "ASSESSED" if assessed_release_state else "UNASSESSED",
        "release_decision": run.release_decision if decision_is_valid else None,
        "source": {key: value for key, value in run.trigger_source.items() if key != "gatekeeper"},
        "coverage": coverage,
        "metrics": run.metrics,
        "component_versions": run.component_versions,
        "failure_code": run.failure_code,
        "failure_message": run.failure_message,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _run_system_version(
    session: Session,
    organization_id: str,
    run: TestRun,
) -> AISystemVersion | None:
    return TenantRepository(session, AISystemVersion).get(
        organization_id,
        run.system_version_id,
    )


def _require_run_read_access(
    session: Session,
    principal: ApiKeyPrincipal,
    run: TestRun,
) -> AISystemVersion:
    system_version = _run_system_version(session, principal.organization_id, run)
    if system_version is None or not _can_read_environment(
        principal,
        system_version.environment,
    ):
        raise _not_found("TEST_RUN_NOT_FOUND", "test run")
    return system_version


def _replay_test_run(
    *,
    replay: IdempotencyReplay,
    principal: ApiKeyPrincipal,
    session: Session,
) -> JSONResponse:
    run = TenantRepository(session, TestRun).get(
        principal.organization_id,
        replay.resource_id,
    )
    if run is None:
        raise GatekeeperError(
            ErrorCode.DATABASE_UNAVAILABLE,
            "The idempotent test-run artifact is unavailable.",
            retryable=True,
            http_status=503,
        )
    _require_run_read_access(session, principal, run)
    response_body = _run_payload(session, principal.organization_id, run)
    session.commit()
    return JSONResponse(
        status_code=replay.response_status,
        content=jsonable_encoder(response_body),
        headers={"Idempotent-Replayed": "true"},
    )


def _record_test_run_idempotency(
    *,
    idempotency: IdempotencyService,
    organization_id: str,
    key: str,
    payload: dict[str, Any],
    resource_id: str,
) -> IdempotencyReplay | None:
    try:
        idempotency.record(
            organization_id=organization_id,
            operation="POST /v1/test-runs",
            key=key,
            payload=payload,
            resource_type="test_run",
            resource_id=resource_id,
            response_status=202,
        )
    except IntegrityError:
        # Roll back the losing run and its cases, then replay the committed winner.
        idempotency.session.rollback()
        replay = idempotency.find(
            organization_id=organization_id,
            operation="POST /v1/test-runs",
            key=key,
            payload=payload,
        )
        if replay is None:
            raise GatekeeperError(
                ErrorCode.DATABASE_UNAVAILABLE,
                "The idempotent test run could not be resolved.",
                retryable=True,
                http_status=503,
            ) from None
        return replay
    return None


@router.post("/test-runs", status_code=202)
def create_test_run(
    body: TestRunCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(test_runs_create),
    session: Session = Depends(get_db_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    _require_environment_access(principal, body.environment.value)
    payload = body.model_dump(mode="json")
    idempotency = IdempotencyService(session, request.app.state.settings.api_key_pepper)
    if idempotency_key is not None:
        replay = idempotency.find(
            organization_id=principal.organization_id,
            operation="POST /v1/test-runs",
            key=idempotency_key,
            payload=payload,
        )
        if replay is not None:
            return _replay_test_run(
                replay=replay,
                principal=principal,
                session=session,
            )

    system = TenantRepository(session, AISystem).get(principal.organization_id, body.system_id)
    if system is None:
        raise _not_found(ErrorCode.SYSTEM_NOT_FOUND.value, "AI system")
    _require_system_read_access(session, principal, system.id)
    _require_active_system(system)
    system_version = _resolve_system_version(
        session,
        principal.organization_id,
        system.id,
        body.system_version,
        body.environment.value,
    )
    if system_version is None:
        raise _not_found(ErrorCode.VERSION_NOT_FOUND.value, "AI system version")
    suite = _resolve_suite(session, principal.organization_id, body.test_suite_id)
    if suite is None:
        raise _not_found("TEST_SUITE_NOT_FOUND", "test suite")
    suite_version = _latest_suite_version(
        session,
        principal.organization_id,
        suite.id,
        body.test_suite_version,
    )
    if suite_version is None:
        raise _not_found("TEST_SUITE_VERSION_NOT_FOUND", "test suite version")
    _require_executable_suite_version(suite_version)

    plan_record = TenantRepository(session, TestPlanRecord).get(
        principal.organization_id, body.test_plan_id
    )
    if plan_record is None:
        raise _not_found("TEST_PLAN_NOT_FOUND", "test plan")
    plan = _validated_test_plan(plan_record)
    if plan.status is not TestPlanStatus.CONFIRMED or plan_record.confirmed_at is None:
        raise ApiError(
            "TEST_PLAN_NOT_CONFIRMED",
            "The test plan must be explicitly confirmed before it can run.",
            http_status=409,
        )
    if plan.application_id != system.id or plan_record.system_id != system.id:
        raise ApiError(
            "TEST_PLAN_SYSTEM_MISMATCH",
            "The confirmed test plan belongs to a different AI system.",
            http_status=409,
        )
    if plan.test_suite_id != suite.id or plan.test_suite_version != suite_version.version:
        raise ApiError(
            "TEST_PLAN_SUITE_MISMATCH",
            "The run suite does not match the immutable suite snapshot in the plan.",
            http_status=409,
        )
    profile_record, profile = _profile_for_system(session, principal.organization_id, system.id)
    if profile is None or profile.version != plan.application_version:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The application profile changed after this plan was confirmed.",
            http_status=409,
        )
    _require_environment_access(principal, profile.environment.value)
    if profile.environment.value != body.environment.value:
        raise ApiError(
            "TEST_PLAN_ENVIRONMENT_MISMATCH",
            "The confirmed test plan environment does not match the requested run environment.",
            http_status=409,
        )
    policy = TenantRepository(session, ReleasePolicyRecord).get(
        principal.organization_id, plan.release_policy.policy_id
    )
    if policy is None:
        raise _not_found("RELEASE_POLICY_NOT_FOUND", "release policy")
    if not policy.active:
        raise ApiError(
            ErrorCode.RELEASE_POLICY_INACTIVE.value,
            "The release policy referenced by this plan is inactive.",
            http_status=409,
        )
    policy_version = _latest_policy_version(
        session,
        principal.organization_id,
        policy.id,
        plan.release_policy.policy_version,
    )
    if policy_version is None:
        raise _not_found("RELEASE_POLICY_VERSION_NOT_FOUND", "release policy version")
    current_plan = _generate_suite_bound_plan(
        session=session,
        organization_id=principal.organization_id,
        profile=profile,
        release_policy=plan.release_policy,
        release_policy_snapshot_digest=_release_policy_content_digest(policy_version),
        suite=suite,
        suite_version=suite_version,
        configuration=_configuration_from_plan(plan),
        test_plan_version=plan.test_plan_version,
    )
    if current_plan.selection_rule_set_version != plan.selection_rule_set_version:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The test-selection rules changed after this plan was confirmed.",
            http_status=409,
        )
    if current_plan.profile_fingerprint != plan.profile_fingerprint:
        raise ApiError(
            "TEST_PLAN_STALE",
            "The application profile changed after this plan was confirmed.",
            http_status=409,
        )
    if _canonical_plan_definition(current_plan) != _canonical_plan_definition(plan):
        raise ApiError(
            "TEST_PLAN_STALE",
            "The confirmed plan no longer matches the canonical plan snapshot.",
            http_status=409,
        )
    case_pairs = _confirmed_plan_case_snapshot(
        session=session,
        organization_id=principal.organization_id,
        suite=suite,
        suite_version=suite_version,
        plan=plan,
    )
    trigger_source = body.source.model_dump(mode="json", exclude_none=True)
    trigger_source["gatekeeper"] = {
        "environment": body.environment.value,
        "test_plan_id": plan_record.id,
        "test_plan_status": plan.status.value,
        "test_plan_version": plan.test_plan_version,
        "application_profile_id": profile_record.id if profile_record else None,
        "application_profile_version": profile.version,
        "release_policy_snapshot_digest": plan.release_policy_snapshot_digest,
        "test_suite_content_snapshot_digest": plan.test_suite_content_snapshot_digest,
        "suite_case_snapshot_digest": plan.suite_case_snapshot_digest,
    }
    run = TestRun(
        organization_id=principal.organization_id,
        system_id=system.id,
        system_version_id=system_version.id,
        test_suite_version_id=suite_version.id,
        release_policy_version_id=policy_version.id,
        execution_status=ExecutionStatus.QUEUED.value,
        release_decision=None,
        trigger_source=trigger_source,
        component_versions={
            "orchestrator": ORCHESTRATOR_VERSION,
            "evaluator_router": EVALUATOR_ROUTER_VERSION,
            "coverage_engine": COVERAGE_ENGINE_VERSION,
        },
    )
    TenantRepository(session, TestRun).add(principal.organization_id, run)
    run_case_pairs: list[tuple[TestRunCase, TestCase]] = []
    for case, version in case_pairs:
        run_case = TestRunCase(
            organization_id=principal.organization_id,
            test_run_id=run.id,
            test_case_version_id=version.id,
            execution_status=ExecutionStatus.QUEUED.value,
            release_decision=None,
        )
        TenantRepository(session, TestRunCase).add(principal.organization_id, run_case)
        run_case_pairs.append((run_case, case))
    run.coverage = _coverage_payload_for_cases(
        profile=profile,
        cases=run_case_pairs,
        plan=plan,
    )
    if idempotency_key is not None:
        replay = _record_test_run_idempotency(
            idempotency=idempotency,
            organization_id=principal.organization_id,
            key=idempotency_key,
            payload=payload,
            resource_id=run.id,
        )
        if replay is not None:
            return _replay_test_run(
                replay=replay,
                principal=principal,
                session=session,
            )
    _audit(
        session,
        request,
        principal,
        action="test_run.queued",
        target_type="test_run",
        target_id=run.id,
    )
    session.commit()
    response_body = _run_payload(session, principal.organization_id, run)
    return JSONResponse(
        status_code=202,
        content=jsonable_encoder(response_body),
        headers={"Location": f"/v1/test-runs/{run.id}"},
    )


@router.get("/test-runs")
def list_test_runs(
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    statement = (
        select(TestRun)
        .join(AISystemVersion, AISystemVersion.id == TestRun.system_version_id)
        .where(
            TestRun.organization_id == principal.organization_id,
            AISystemVersion.organization_id == principal.organization_id,
        )
        .order_by(TestRun.created_at.desc(), TestRun.id.desc())
        .limit(limit)
    )
    if principal.environment != "live":
        statement = statement.where(AISystemVersion.environment != "PRODUCTION")
    runs = tuple(session.scalars(statement))
    items = [_run_payload(session, principal.organization_id, run) for run in runs]
    session.commit()
    return {"items": items, "count": len(items)}


@router.get("/test-runs/{run_id}")
def get_test_run(
    run_id: str,
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    run = TenantRepository(session, TestRun).get(principal.organization_id, run_id)
    if run is None:
        raise _not_found("TEST_RUN_NOT_FOUND", "test run")
    _require_run_read_access(session, principal, run)
    payload = _run_payload(session, principal.organization_id, run)
    session.commit()
    return payload


@router.get("/test-runs/{run_id}/cases")
def get_test_run_cases(
    run_id: str,
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    run = TenantRepository(session, TestRun).get(principal.organization_id, run_id)
    if run is None:
        raise _not_found("TEST_RUN_NOT_FOUND", "test run")
    _require_run_read_access(session, principal, run)
    items = []
    for run_case, case in _run_case_pairs(session, principal.organization_id, run.id):
        route = EvaluatorRouter().route(case.test_family)
        items.append(
            {
                "id": run_case.id,
                "run_case_id": run_case.id,
                "case_id": case.id,
                "test_family": case.test_family,
                "execution_status": run_case.execution_status,
                "assessment_status": (
                    "ASSESSED" if run_case.execution_status == "COMPLETED" else "UNASSESSED"
                ),
                "release_decision": run_case.release_decision,
                "metrics": run_case.metrics,
                "error_code": run_case.error_code,
                "evaluator": route.model_dump(mode="json"),
                "created_at": run_case.created_at,
                "updated_at": run_case.updated_at,
            }
        )
    session.commit()
    return {"run_id": run.id, "items": items, "count": len(items)}


def _finding_payload(
    finding: FindingRecord,
    evaluation: Evaluation,
    run_case: TestRunCase | None,
    encryptor: EncryptedJsonStore,
) -> dict[str, Any]:
    artifact = encryptor.decrypt(finding.original_finding_encrypted)
    return {
        "id": finding.id,
        "evaluation_id": evaluation.id,
        "run_id": run_case.test_run_id if run_case else None,
        "run_case_id": run_case.id if run_case else None,
        **artifact,
    }


@router.get("/test-runs/{run_id}/findings")
def get_test_run_findings(
    run_id: str,
    principal: ApiKeyPrincipal = Depends(reports_read),
    session: Session = Depends(get_db_session),
    encryptor: EncryptedJsonStore = Depends(get_encryptor),
) -> dict[str, Any]:
    run = TenantRepository(session, TestRun).get(principal.organization_id, run_id)
    if run is None:
        raise _not_found("TEST_RUN_NOT_FOUND", "test run")
    _require_run_read_access(session, principal, run)
    statement = (
        select(FindingRecord, Evaluation, TestRunCase)
        .join(Evaluation, Evaluation.id == FindingRecord.evaluation_id)
        .join(TestRunCase, TestRunCase.id == Evaluation.test_run_case_id)
        .where(
            FindingRecord.organization_id == principal.organization_id,
            Evaluation.organization_id == principal.organization_id,
            TestRunCase.organization_id == principal.organization_id,
            TestRunCase.test_run_id == run.id,
        )
        .order_by(FindingRecord.created_at, FindingRecord.ordinal)
    )
    if principal.environment != "live":
        statement = statement.where(Evaluation.environment != "PRODUCTION")
    items = [
        _finding_payload(finding, evaluation, run_case, encryptor)
        for finding, evaluation, run_case in session.execute(statement)
    ]
    session.commit()
    return {"run_id": run.id, "items": items, "count": len(items)}


@router.get("/findings")
def list_findings(
    principal: ApiKeyPrincipal = Depends(reports_read),
    session: Session = Depends(get_db_session),
    encryptor: EncryptedJsonStore = Depends(get_encryptor),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    statement = (
        select(FindingRecord, Evaluation, TestRunCase)
        .join(Evaluation, Evaluation.id == FindingRecord.evaluation_id)
        .outerjoin(TestRunCase, TestRunCase.id == Evaluation.test_run_case_id)
        .where(
            FindingRecord.organization_id == principal.organization_id,
            Evaluation.organization_id == principal.organization_id,
        )
        .order_by(FindingRecord.created_at.desc(), FindingRecord.id.desc())
        .limit(limit)
    )
    if principal.environment != "live":
        statement = statement.where(Evaluation.environment != "PRODUCTION")
    items = [
        _finding_payload(finding, evaluation, run_case, encryptor)
        for finding, evaluation, run_case in session.execute(statement)
    ]
    session.commit()
    return {"items": items, "count": len(items)}


def _policy_payload(
    policy: ReleasePolicyRecord,
    version: ReleasePolicyVersion | None,
) -> dict[str, Any]:
    required_coverage = dict(version.required_coverage) if version else {}
    missing_action = required_coverage.pop("missing_required_coverage_action", "NO_DECISION")
    critical_override = required_coverage.pop("critical_finding_override", True)
    return {
        "id": policy.id,
        "policy_id": policy.policy_key or policy.id,
        "name": policy.name,
        "active": policy.active,
        "created_at": policy.created_at,
        "updated_at": policy.updated_at,
        "latest_version": (
            {
                "id": version.id,
                "version": version.version,
                "rules": version.rules,
                "required_coverage": required_coverage,
                "missing_required_coverage_action": missing_action,
                "critical_finding_override": critical_override,
                "created_at": version.created_at,
            }
            if version
            else None
        ),
    }


@router.post("/release-policies", status_code=201)
def create_release_policy(
    body: ReleasePolicyCreate,
    request: Request,
    principal: ApiKeyPrincipal = Depends(release_policies_manage),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    policy = ReleasePolicyRecord(
        organization_id=principal.organization_id,
        policy_key=body.policy_id,
        name=body.name or body.policy_id,
        active=body.active,
    )
    TenantRepository(session, ReleasePolicyRecord).add(principal.organization_id, policy)
    # Reuse the policy engine's strict contract at the HTTP/persistence boundary.
    validated = DomainReleasePolicy(
        policy_id=policy.id,
        policy_version=body.version,
        rules=body.rules,
        missing_required_coverage_action=body.missing_required_coverage_action,
        critical_finding_override=body.critical_finding_override,
    )
    required_coverage = {
        **body.required_coverage,
        "missing_required_coverage_action": validated.missing_required_coverage_action.value,
        "critical_finding_override": validated.critical_finding_override,
    }
    version = ReleasePolicyVersion(
        organization_id=principal.organization_id,
        release_policy_id=policy.id,
        version=body.version,
        rules=[rule.model_dump(mode="json") for rule in validated.rules],
        required_coverage=required_coverage,
    )
    TenantRepository(session, ReleasePolicyVersion).add(principal.organization_id, version)
    _audit(
        session,
        request,
        principal,
        action="release_policy.created",
        target_type="release_policy",
        target_id=policy.id,
    )
    session.commit()
    return _policy_payload(policy, version)


@router.get("/release-policies")
def list_release_policies(
    principal: ApiKeyPrincipal = Depends(test_runs_read),
    session: Session = Depends(get_db_session),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    policies = TenantRepository(session, ReleasePolicyRecord).list(
        principal.organization_id, limit=limit
    )
    payload = [
        _policy_payload(
            policy,
            _latest_policy_version(session, principal.organization_id, policy.id),
        )
        for policy in policies
    ]
    session.commit()
    return {"items": payload, "count": len(payload)}


@router.get("/coverage")
def get_coverage(
    principal: ApiKeyPrincipal = Depends(reports_read),
    session: Session = Depends(get_db_session),
    run_id: str | None = Query(default=None),
    test_run_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1_000),
) -> dict[str, Any]:
    if run_id and test_run_id and run_id != test_run_id:
        raise ApiError(
            "REQ_INVALID",
            "run_id and test_run_id must identify the same run when both are supplied.",
            http_status=422,
        )
    selected_run_id = run_id or test_run_id
    if selected_run_id:
        run = TenantRepository(session, TestRun).get(principal.organization_id, selected_run_id)
        if run is None:
            raise _not_found("TEST_RUN_NOT_FOUND", "test run")
        _require_run_read_access(session, principal, run)
        coverage = run.coverage or {
            "engine_version": COVERAGE_ENGINE_VERSION,
            "status": "UNASSESSED",
            "valid_for_release": False,
            "limitations": ["No persisted coverage snapshot is available."],
        }
        session.commit()
        return {"run_id": run.id, **coverage}

    statement = (
        select(TestRun)
        .join(AISystemVersion, AISystemVersion.id == TestRun.system_version_id)
        .where(
            TestRun.organization_id == principal.organization_id,
            AISystemVersion.organization_id == principal.organization_id,
        )
        .order_by(TestRun.created_at.desc(), TestRun.id.desc())
        .limit(limit)
    )
    if principal.environment != "live":
        statement = statement.where(AISystemVersion.environment != "PRODUCTION")
    runs = tuple(session.scalars(statement))
    items = []
    for run in runs:
        items.append(
            {
                "run_id": run.id,
                **(
                    run.coverage
                    or {
                        "engine_version": COVERAGE_ENGINE_VERSION,
                        "status": "UNASSESSED",
                        "valid_for_release": False,
                        "limitations": ["No persisted coverage snapshot is available."],
                    }
                ),
            }
        )
    session.commit()
    return {"items": items, "count": len(items)}

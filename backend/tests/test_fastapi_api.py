from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Never, cast

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api import routes as api_routes
from backend.app.api.dependencies import get_db_session, get_encryptor
from backend.app.api.middleware import InMemoryRateLimiter
from backend.app.auth import ApiScope, create_api_key
from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.core.config import Settings
from backend.app.database.base import Base
from backend.app.database.models import (
    AISystem,
    AISystemVersion,
    ApiKey,
    Evaluation,
    GeneratedOutputRecord,
    IdempotencyRecord,
    Organization,
    SourceDocumentRecord,
)
from backend.app.database.models import ReleasePolicy as PolicyRecord
from backend.app.database.models import (
    ReleasePolicyVersion as PolicyVersionRecord,
)
from backend.app.database.models import TestCase as CaseRecord
from backend.app.database.models import (
    TestCaseVersion as CaseVersionRecord,
)
from backend.app.database.models import (
    TestPlanRecord as PlanRecord,
)
from backend.app.database.models import (
    TestRun as RunRecord,
)
from backend.app.database.models import (
    TestRunCase as RunCaseRecord,
)
from backend.app.database.models import (
    TestSuiteVersion as SuiteVersionRecord,
)
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluations import EvaluationService
from backend.app.main import create_app
from backend.app.observability import OperationalMetrics
from backend.app.orchestration.worker import TestRunWorker as RunWorker
from backend.app.security import EncryptedJsonStore

PEPPER = "fastapi-unit-test-key-pepper"


@dataclass(slots=True)
class ApiHarness:
    client: TestClient
    engine: Engine
    sessions: sessionmaker[Session]
    encryptor: EncryptedJsonStore
    key_a: str
    live_key_a: str
    key_b: str
    read_only_key: str
    ci_key_a: str

    def authorization(self, key: str | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {key or self.key_a}"}

    def close(self) -> None:
        self.client.close()
        self.engine.dispose()


_ACTIVE_HARNESSES: list[ApiHarness] = []


def _settings(encryption_key: str, *, max_request_bytes: int = 1_048_576) -> Settings:
    return Settings(
        environment="test",
        database_url="sqlite+pysqlite:///:memory:",
        api_key_pepper=PEPPER,
        data_encryption_keys=encryption_key,
        max_request_bytes=max_request_bytes,
    )


def _build_harness(
    *,
    max_request_bytes: int = 1_048_576,
    rate_limiter: InMemoryRateLimiter | None = None,
    retention_days: int = 30,
    ner_backend: str = "rules",
) -> ApiHarness:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    encryption_key = Fernet.generate_key().decode()
    encryptor = EncryptedJsonStore(encryption_key)
    with sessions() as session:
        session.add_all(
            [
                Organization(id="org_api_a", name="API A"),
                Organization(id="org_api_b", name="API B"),
            ]
        )
        full_a = create_api_key(
            organization_id="org_api_a",
            environment="test",
            scopes=set(ApiScope),
            pepper=PEPPER,
        )
        full_b = create_api_key(
            organization_id="org_api_b",
            environment="test",
            scopes=set(ApiScope),
            pepper=PEPPER,
        )
        live_a = create_api_key(
            organization_id="org_api_a",
            environment="live",
            scopes=set(ApiScope),
            pepper=PEPPER,
        )
        read_only = create_api_key(
            organization_id="org_api_a",
            environment="test",
            scopes={ApiScope.SYSTEMS_READ},
            pepper=PEPPER,
        )
        ci_key = create_api_key(
            organization_id="org_api_a",
            environment="test",
            scopes={ApiScope.TEST_RUNS_CREATE, ApiScope.TEST_RUNS_READ},
            pepper=PEPPER,
        )
        session.add_all(
            [full_a.record, live_a.record, full_b.record, read_only.record, ci_key.record]
        )
        session.commit()

    settings = _settings(encryption_key, max_request_bytes=max_request_bytes).model_copy(
        update={
            "clinical_data_retention_days": retention_days,
            "ner_backend": ner_backend,
        }
    )
    application = create_app(
        settings=settings,
        rate_limiter=rate_limiter,
    )

    def override_session() -> Generator[Session, None, None]:
        with sessions() as session:
            yield session

    application.dependency_overrides[get_db_session] = override_session
    application.dependency_overrides[get_encryptor] = lambda: encryptor
    harness = ApiHarness(
        client=TestClient(application),
        engine=engine,
        sessions=sessions,
        encryptor=encryptor,
        key_a=full_a.secret,
        live_key_a=live_a.secret,
        key_b=full_b.secret,
        read_only_key=read_only.secret,
        ci_key_a=ci_key.secret,
    )
    _ACTIVE_HARNESSES.append(harness)
    return harness


@pytest.fixture(autouse=True)
def close_api_harnesses() -> Generator[None, None, None]:
    yield
    while _ACTIVE_HARNESSES:
        _ACTIVE_HARNESSES.pop().close()


@pytest.fixture
def api() -> ApiHarness:
    return _build_harness()


def test_create_app_binds_request_sessions_to_its_own_settings() -> None:
    encryption_key = Fernet.generate_key().decode()
    application = create_app(settings=_settings(encryption_key))
    Base.metadata.create_all(application.state.database_engine)
    sessions: sessionmaker[Session] = application.state.session_factory
    with sessions() as session:
        session.add(Organization(id="org_local_app", name="Local app"))
        created = create_api_key(
            organization_id="org_local_app",
            environment="test",
            scopes={ApiScope.SYSTEMS_READ},
            pepper=PEPPER,
        )
        session.add(created.record)
        session.commit()

    with TestClient(application) as client:
        response = client.get(
            "/v1/systems",
            headers={"Authorization": f"Bearer {created.secret}"},
        )

    assert response.status_code == 200
    assert response.json() == {"items": [], "count": 0}
    application.state.database_engine.dispose()


def test_api_wires_optional_huggingface_ner_and_reduces_on_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[bool] = []

    class FailingAdapter:
        version = "controlled-failing-ner@1"
        last_error: str | None = "NER_MODEL_LOAD_FAILED:controlled"

        def extract(self, text: str) -> list[dict[str, object]]:
            del text
            return []

    def clinical_nlp_factory(*, enable_hf: bool) -> ClinicalNLPService:
        configured.append(enable_hf)
        return ClinicalNLPService(ner_adapter=FailingAdapter())

    monkeypatch.setattr(api_routes, "ClinicalNLPService", clinical_nlp_factory)
    api = _build_harness(ner_backend="huggingface")
    system = _create_system(api)
    _create_version(api, str(system["id"]))

    response = api.client.post(
        "/v1/evaluations",
        headers=api.authorization(),
        json={
            "system_id": system["id"],
            "system_version": "1.4.0",
            "environment": "staging",
            "sources": [{"document_id": "source", "content": "Patient has left knee pain."}],
            "generated_output": {"content": "Patient has right knee pain."},
        },
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["coverage"]["status"] == "REDUCED"
    assert any(
        "NER_ADAPTER_FAILED" in limitation for limitation in result["coverage"]["limitations"]
    )
    assert result["versions"]["ner_model"] == "controlled-failing-ner@1"
    assert configured == [True]


def test_plan_confirmation_query_uses_a_postgresql_row_lock() -> None:
    statement = api_routes._test_plan_confirmation_query("org_a", "plan_a")
    compiled = str(
        statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
    )

    assert compiled.endswith("FOR UPDATE")


def test_api_key_management_supports_least_privilege_rotation_without_secret_leakage(
    api: ApiHarness,
) -> None:
    expires_at = datetime.now(UTC) + timedelta(days=1)
    created = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(),
        json={
            "environment": "test",
            "scopes": ["test-runs:create", "test-runs:read"],
            "expires_at": expires_at.isoformat(),
        },
    )
    assert created.status_code == 201, created.text
    created_body = created.json()
    secret = created_body["secret"]
    key_id = created_body["key_id"]
    assert secret.startswith("gk_test_")
    assert created.text.count(secret) == 1
    assert "key_hash" not in created_body
    assert created_body["scopes"] == ["test-runs:create", "test-runs:read"]

    authenticated = api.client.get(
        "/v1/test-runs",
        headers={"Authorization": f"Bearer {secret}"},
    )
    assert authenticated.status_code == 200

    listed = api.client.get("/v1/api-keys", headers=api.authorization())
    listed_item = next(item for item in listed.json()["items"] if item["key_id"] == key_id)
    assert "secret" not in listed_item
    assert "key_hash" not in listed_item
    assert secret not in listed.text
    with api.sessions() as session:
        persisted = session.get(ApiKey, key_id)
        assert persisted is not None
        assert secret.encode() != persisted.key_hash
        assert secret not in repr(persisted.key_hash)

    cross_tenant = api.client.post(
        f"/v1/api-keys/{key_id}/revoke",
        headers=api.authorization(api.key_b),
    )
    assert cross_tenant.status_code == 404
    assert cross_tenant.json()["error"]["code"] == "API_KEY_NOT_FOUND"
    insufficient = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(api.read_only_key),
        json={"environment": "test", "scopes": ["systems:read"]},
    )
    assert insufficient.status_code == 403
    live_from_test = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(),
        json={"environment": "live", "scopes": ["test-runs:read"]},
    )
    assert live_from_test.status_code == 403
    assert live_from_test.json()["error"]["code"] == "AUTH_ENVIRONMENT_MISMATCH"

    live_created = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(api.live_key_a),
        json={"environment": "live", "scopes": ["test-runs:read"]},
    )
    assert live_created.status_code == 201
    test_visible_ids = {
        item["key_id"]
        for item in api.client.get("/v1/api-keys", headers=api.authorization()).json()["items"]
    }
    assert live_created.json()["key_id"] not in test_visible_ids
    live_visible_ids = {
        item["key_id"]
        for item in api.client.get(
            "/v1/api-keys", headers=api.authorization(api.live_key_a)
        ).json()["items"]
    }
    assert live_created.json()["key_id"] in live_visible_ids

    revoked = api.client.post(
        f"/v1/api-keys/{key_id}/revoke",
        headers=api.authorization(),
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "REVOKED"
    revoked_at = revoked.json()["revoked_at"]
    replayed_revoke = api.client.post(
        f"/v1/api-keys/{key_id}/revoke",
        headers=api.authorization(),
    )
    assert replayed_revoke.status_code == 200
    assert replayed_revoke.json()["revoked_at"] == revoked_at
    rejected = api.client.get(
        "/v1/test-runs",
        headers={"Authorization": f"Bearer {secret}"},
    )
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "AUTH_REVOKED_KEY"


def test_api_key_management_rejects_invalid_expiry_and_scope(api: ApiHarness) -> None:
    expired = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(),
        json={
            "environment": "test",
            "scopes": ["systems:read"],
            "expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        },
    )
    unknown_scope = api.client.post(
        "/v1/api-keys",
        headers=api.authorization(),
        json={"environment": "test", "scopes": ["everything:manage"]},
    )

    assert expired.status_code == 422
    assert unknown_scope.status_code == 422


def _create_system(api: ApiHarness) -> dict[str, Any]:
    response = api.client.post(
        "/v1/systems",
        headers=api.authorization(),
        json={
            "name": "Clinical scribe",
            "application_type": "clinical_scribe",
            "risk_tier": "high",
            "profile": {
                "intended_use": "Prepare a clinician-reviewed draft note.",
                "input_type": "clinical_transcript",
                "output_type": "clinical_note",
                "clinician_facing": True,
            },
        },
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def _create_version(api: ApiHarness, system_id: str) -> dict[str, Any]:
    response = api.client.post(
        f"/v1/systems/{system_id}/versions",
        headers=api.authorization(),
        json={
            "version": "1.4.0",
            "environment": "staging",
            "endpoint_url": "https://customer.example.test/inference",
            "authentication_config": {"token": "customer-secret-value"},
        },
    )
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def test_public_json_contracts_reject_non_finite_numbers_before_persistence(
    api: ApiHarness,
) -> None:
    def raw_post(path: str, payload: dict[str, Any]) -> None:
        response = api.client.post(
            path,
            headers={**api.authorization(), "Content-Type": "application/json"},
            content=json.dumps(payload),
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "REQ_INVALID"

    raw_post(
        "/v1/systems",
        {
            "name": "Non-finite profile",
            "application_type": "OTHER",
            "risk_tier": "LOW",
            "profile": {
                "intended_use": "Reject non-standard JSON numbers.",
                "input_type": "structured_data",
                "output_type": "structured_data",
                "request_schema": {"invalid": float("nan")},
            },
        },
    )

    system = _create_system(api)
    raw_post(
        f"/v1/systems/{system['id']}/versions",
        {
            "version": "non-finite",
            "authentication_config": {"invalid": float("inf")},
        },
    )
    suite_base: dict[str, Any] = {
        "name": "Non-finite suite",
        "data_classification": "SYNTHETIC",
        "contains_phi": False,
        "no_phi_attested": True,
    }
    raw_post(
        "/v1/test-suites",
        {**suite_base, "definition": {"invalid": float("-inf")}},
    )
    raw_post(
        "/v1/test-suites",
        {
            **suite_base,
            "cases": [
                {
                    "name": "Invalid case input",
                    "test_family": "clinical_faithfulness",
                    "input_data": {"invalid": float("nan")},
                }
            ],
        },
    )
    raw_post(
        "/v1/test-suites",
        {
            **suite_base,
            "cases": [
                {
                    "name": "Invalid expectation",
                    "test_family": "clinical_faithfulness",
                    "expectation": {"invalid": float("inf")},
                }
            ],
        },
    )
    raw_post(
        "/v1/release-policies",
        {
            "policy_id": "non-finite-policy",
            "rules": [
                {
                    "rule_id": "minimum-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 95,
                    "on_failure": "BLOCK",
                }
            ],
            "required_coverage": {"invalid": float("nan")},
        },
    )


def test_health_auth_scope_and_tenant_indistinguishability(api: ApiHarness) -> None:
    health = api.client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["X-Request-ID"].startswith("req_")

    unauthenticated = api.client.get("/v1/systems")
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "AUTH_INVALID_KEY"
    assert unauthenticated.json()["error"]["request_id"].startswith("req_")

    forbidden = api.client.post(
        "/v1/systems",
        headers=api.authorization(api.read_only_key),
        json={
            "name": "Not allowed",
            "application_type": "OTHER",
            "risk_tier": "LOW",
        },
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "AUTH_INSUFFICIENT_SCOPE"

    for path, body in (
        (
            "/v1/test-suites",
            {
                "name": "CI must not author suites",
                "data_classification": "SYNTHETIC",
                "contains_phi": False,
                "no_phi_attested": True,
            },
        ),
        (
            "/v1/release-policies",
            {
                "policy_id": "ci-must-not-author-policy",
                "rules": [
                    {
                        "rule_id": "minimum-score",
                        "metric": "score",
                        "operator": ">=",
                        "threshold": 95,
                        "on_failure": "BLOCK",
                    }
                ],
            },
        ),
        (
            "/v1/systems/sys_not_owned/test-plans",
            {
                "release_policy_id": "policy_not_owned",
                "test_suite_id": "suite_not_owned",
                "test_suite_version": "1.0.0",
            },
        ),
    ):
        governance_write = api.client.post(
            path,
            headers=api.authorization(api.ci_key_a),
            json=body,
        )
        assert governance_write.status_code == 403
        assert governance_write.json()["error"]["code"] == "AUTH_INSUFFICIENT_SCOPE"

    system_id = str(_create_system(api)["id"])
    cross_tenant = api.client.get(f"/v1/systems/{system_id}", headers=api.authorization(api.key_b))
    absent = api.client.get("/v1/systems/sys_does_not_exist", headers=api.authorization(api.key_b))
    assert cross_tenant.status_code == absent.status_code == 404
    assert cross_tenant.json()["error"]["code"] == absent.json()["error"]["code"]
    assert cross_tenant.json()["error"]["message"] == absent.json()["error"]["message"]


def test_system_profile_selection_and_version_secret_encryption(api: ApiHarness) -> None:
    system = _create_system(api)
    assert system["application_profile"]["organization_id"] == "org_api_a"
    assert "clinical_faithfulness" in system["test_selection"]["required_tests"]
    version = _create_version(api, str(system["id"]))
    assert version["authentication_configured"] is True
    assert "authentication_config" not in version
    assert "customer-secret-value" not in str(version)

    with api.sessions() as session:
        record = session.get(AISystemVersion, version["id"])
        assert record is not None
        assert record.authentication_config_encrypted is not None
        assert b"customer-secret-value" not in record.authentication_config_encrypted
        assert api.encryptor.decrypt(record.authentication_config_encrypted) == {
            "token": "customer-secret-value"
        }

    credentialed_url = api.client.post(
        f"/v1/systems/{system['id']}/versions",
        headers=api.authorization(),
        json={
            "version": "1.5.0",
            "environment": "staging",
            "endpoint_url": "https://user:secret@customer.example.test/inference",
        },
    )
    assert credentialed_url.status_code == 422
    assert "secret" not in credentialed_url.text

    for unsafe_endpoint in (
        "https://127.0.0.1/inference",
        "https://127\u30020\u30020\u30021/inference",
        "https://2130706433/inference",
        "https://127.1/inference",
        "https://0x7f000001/inference",
        "https://169.254.169.254/latest/meta-data",
        "https://customer.example.test/inference?api_key=supersecret",
        "https://customer.example.test/inference#token=supersecret",
    ):
        unsafe_url = api.client.post(
            f"/v1/systems/{system['id']}/versions",
            headers=api.authorization(),
            json={
                "version": "1.5.1",
                "environment": "staging",
                "endpoint_url": unsafe_endpoint,
            },
        )
        assert unsafe_url.status_code == 422
        assert "supersecret" not in unsafe_url.text

    production_version = api.client.post(
        f"/v1/systems/{system['id']}/versions",
        headers=api.authorization(),
        json={"version": "2.0.0", "environment": "production"},
    )
    assert production_version.status_code == 403
    assert production_version.json()["error"]["code"] == "AUTH_ENVIRONMENT_MISMATCH"


@respx.mock
def test_worker_executes_queued_run_against_registered_endpoint(api: ApiHarness) -> None:
    endpoint = "https://customer.example.test/inference"
    mocked_endpoint = respx.post(endpoint).mock(
        return_value=httpx.Response(200, json={"output": "approved response"})
    )
    system_response = api.client.post(
        "/v1/systems",
        headers=api.authorization(),
        json={
            "name": "Worker integration app",
            "application_type": "OTHER",
            "risk_tier": "LOW",
            "profile": {
                "version": "1.0.0",
                "intended_use": "Execute a synthetic regression request.",
                "input_type": "QUESTION",
                "output_type": "ANSWER",
                "environment": "STAGING",
                "endpoint_url": endpoint,
                "uses_llm": False,
            },
        },
    )
    assert system_response.status_code == 201, system_response.text
    system_id = system_response.json()["id"]
    version_response = api.client.post(
        f"/v1/systems/{system_id}/versions",
        headers=api.authorization(),
        json={
            "version": "1.0.0",
            "environment": "STAGING",
            "endpoint_url": endpoint,
            "authentication_config": {"token": "worker-secret"},
            "response_schema": {"output_path": "output"},
        },
    )
    assert version_response.status_code == 201, version_response.text
    suite_response = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={
            "name": "Worker integration suite",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
            "cases": [
                {
                    "name": "Synthetic regression assertion",
                    "test_family": "custom_business_regression",
                    "input_data": {"question": "Return the approved response."},
                    "expectation": {"must_include": ["approved"]},
                },
                {
                    "name": "Synthetic robustness assertion",
                    "test_family": "robustness",
                    "input_data": {
                        "question": "Return the approved response.",
                        "variant_outputs": ["approved response", "approved response"],
                    },
                    "expectation": {"require_same_output": True},
                },
                {
                    "name": "Synthetic policy assertion",
                    "test_family": "regulatory_policy_hardening",
                    "input_data": {"question": "Return the approved response."},
                    "expectation": {"must_include": ["approved"]},
                },
            ],
        },
    )
    assert suite_response.status_code == 201, suite_response.text
    suite = suite_response.json()
    policy_response = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(),
        json={
            "policy_id": "worker-integration-policy",
            "rules": [
                {
                    "rule_id": "minimum-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 90,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert policy_response.status_code == 201, policy_response.text
    plan_response = api.client.post(
        f"/v1/systems/{system_id}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": policy_response.json()["id"],
            "test_suite_id": suite["id"],
            "test_suite_version": suite["latest_version"]["version"],
            "case_counts": {
                "custom_business_regression": 1,
                "robustness": 1,
                "regulatory_policy_hardening": 1,
            },
        },
    )
    assert plan_response.status_code == 201, plan_response.text
    plan = plan_response.json()
    confirmation = api.client.post(
        f"/v1/test-plans/{plan['test_plan_id']}/confirm",
        headers=api.authorization(),
        json={
            "confirm": True,
            "expected_profile_fingerprint": plan["profile_fingerprint"],
            "expected_test_plan_version": plan["test_plan_version"],
        },
    )
    assert confirmation.status_code == 200, confirmation.text
    run_response = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "worker-integration-run"},
        json={
            "system_id": system_id,
            "system_version": "1.0.0",
            "environment": "STAGING",
            "test_suite_id": suite["id"],
            "test_suite_version": suite["latest_version"]["version"],
            "test_plan_id": plan["test_plan_id"],
        },
    )
    assert run_response.status_code == 202, run_response.text
    run_id = run_response.json()["run_id"]

    worker = RunWorker(api.sessions, api.encryptor, timeout_seconds=1, max_attempts=1)
    assert asyncio.run(worker.execute_next()) == run_id

    completed = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert completed.status_code == 200, completed.text
    assert completed.json()["execution_status"] == "COMPLETED"
    assert completed.json()["release_decision"] == "PASS"
    assert completed.json()["coverage"]["status"] == "FULL"
    assert mocked_endpoint.called
    assert mocked_endpoint.calls[0].request.headers["authorization"] == "Bearer worker-secret"


def test_inactive_system_blocks_executable_and_governance_mutations(
    api: ApiHarness,
) -> None:
    system = _create_system(api)
    version = _create_version(api, str(system["id"]))
    suite = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={
            "name": "Inactive system suite",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
            "cases": [
                {
                    "name": "Faithfulness",
                    "test_family": "clinical_faithfulness",
                    "input_data": {},
                    "expectation": {},
                }
            ],
        },
    )
    policy = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(),
        json={
            "policy_id": "inactive-system-policy",
            "rules": [
                {
                    "rule_id": "minimum-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 90,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert suite.status_code == policy.status_code == 201
    draft = api.client.post(
        f"/v1/systems/{system['id']}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": policy.json()["id"],
            "test_suite_id": suite.json()["id"],
            "test_suite_version": suite.json()["latest_version"]["version"],
        },
    )
    assert draft.status_code == 201

    with api.sessions() as session:
        record = session.get(AISystem, system["id"])
        assert record is not None
        record.active = False
        session.commit()

    requests = (
        api.client.post(
            f"/v1/systems/{system['id']}/versions",
            headers=api.authorization(),
            json={"version": "2.0.0", "environment": "staging"},
        ),
        api.client.post(
            "/v1/evaluations",
            headers=api.authorization(),
            json={
                "system_id": system["id"],
                "system_version": version["version"],
                "sources": [{"document_id": "source", "content": "Synthetic left knee pain."}],
                "generated_output": {"content": "Synthetic left knee pain."},
            },
        ),
        api.client.post(
            f"/v1/systems/{system['id']}/test-plans",
            headers=api.authorization(),
            json={
                "release_policy_id": policy.json()["id"],
                "test_suite_id": suite.json()["id"],
                "test_suite_version": suite.json()["latest_version"]["version"],
            },
        ),
        api.client.post(
            f"/v1/test-plans/{draft.json()['test_plan_id']}/confirm",
            headers=api.authorization(),
            json={
                "confirm": True,
                "expected_profile_fingerprint": draft.json()["profile_fingerprint"],
                "expected_test_plan_version": draft.json()["test_plan_version"],
            },
        ),
        api.client.post(
            "/v1/test-runs",
            headers=api.authorization(),
            json={
                "system_id": system["id"],
                "system_version": version["version"],
                "test_suite_id": suite.json()["id"],
                "test_suite_version": suite.json()["latest_version"]["version"],
                "test_plan_id": draft.json()["test_plan_id"],
            },
        ),
    )
    assert all(response.status_code == 409 for response in requests)
    assert all(response.json()["error"]["code"] == "SYSTEM_INACTIVE" for response in requests)


def test_test_keys_cannot_read_production_artifacts_but_live_keys_can(
    api: ApiHarness,
) -> None:
    staging_system = _create_system(api)
    staging_version = _create_version(api, str(staging_system["id"]))
    same_version_in_production = api.client.post(
        f"/v1/systems/{staging_system['id']}/versions",
        headers=api.authorization(api.live_key_a),
        json={"version": "1.4.0", "environment": "production"},
    )
    assert same_version_in_production.status_code == 201
    assert same_version_in_production.json()["id"] != staging_version["id"]

    production_system_response = api.client.post(
        "/v1/systems",
        headers=api.authorization(api.live_key_a),
        json={
            "name": "Production clinical scribe",
            "application_type": "clinical_scribe",
            "risk_tier": "high",
            "profile": {
                "version": "9.0.0",
                "intended_use": "Prepare a clinician-reviewed production draft note.",
                "input_type": "clinical_transcript",
                "output_type": "clinical_note",
                "clinician_facing": True,
                "environment": "production",
            },
        },
    )
    assert production_system_response.status_code == 201
    production_system = production_system_response.json()
    production_version_response = api.client.post(
        f"/v1/systems/{production_system['id']}/versions",
        headers=api.authorization(api.live_key_a),
        json={"version": "9.0.0", "environment": "production"},
    )
    assert production_version_response.status_code == 201
    production_version = production_version_response.json()
    staging_variant_response = api.client.post(
        f"/v1/systems/{production_system['id']}/versions",
        headers=api.authorization(api.live_key_a),
        json={"version": "9.0.0", "environment": "staging"},
    )
    assert staging_variant_response.status_code == 201

    hidden_system = api.client.get(
        f"/v1/systems/{production_system['id']}",
        headers=api.authorization(),
    )
    visible_system = api.client.get(
        f"/v1/systems/{production_system['id']}",
        headers=api.authorization(api.live_key_a),
    )
    test_systems = api.client.get("/v1/systems", headers=api.authorization()).json()
    live_systems = api.client.get("/v1/systems", headers=api.authorization(api.live_key_a)).json()
    assert hidden_system.status_code == 404
    assert visible_system.status_code == 200
    assert production_system["id"] not in {item["id"] for item in test_systems["items"]}
    assert production_system["id"] in {item["id"] for item in live_systems["items"]}

    evaluation_request = {
        "system_id": production_system["id"],
        "system_version": "9.0.0",
        "environment": "production",
        "sources": [
            {
                "document_id": "prod-source",
                "type": "clinical_transcript",
                "content": "Patient reports left knee pain.",
            }
        ],
        "generated_output": {
            "type": "clinical_note",
            "content": "Patient reports right knee pain.",
        },
    }
    production_evaluation = api.client.post(
        "/v1/evaluations",
        headers={
            **api.authorization(api.live_key_a),
            "Idempotency-Key": "production-evaluation",
        },
        json=evaluation_request,
    )
    assert production_evaluation.status_code == 201
    evaluation_id = production_evaluation.json()["evaluation_id"]
    hidden_evaluation = api.client.get(
        f"/v1/evaluations/{evaluation_id}",
        headers=api.authorization(),
    )
    visible_evaluation = api.client.get(
        f"/v1/evaluations/{evaluation_id}",
        headers=api.authorization(api.live_key_a),
    )
    test_findings = api.client.get("/v1/findings", headers=api.authorization()).json()
    live_findings = api.client.get("/v1/findings", headers=api.authorization(api.live_key_a)).json()
    blocked_evaluation_replay = api.client.post(
        "/v1/evaluations",
        headers={**api.authorization(), "Idempotency-Key": "production-evaluation"},
        json=evaluation_request,
    )
    assert hidden_evaluation.status_code == 404
    assert visible_evaluation.status_code == 200
    assert blocked_evaluation_replay.status_code == 403
    assert evaluation_id not in {item["evaluation_id"] for item in test_findings["items"]}
    assert evaluation_id in {item["evaluation_id"] for item in live_findings["items"]}

    suite = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(api.live_key_a),
        json={
            "slug": "production-scribe-core",
            "name": "Production scribe core",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
            "cases": [
                {
                    "name": "Faithfulness",
                    "test_family": "clinical_faithfulness",
                    "input_data": {},
                    "expectation": {},
                }
            ],
        },
    )
    assert suite.status_code == 201
    policy = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(api.live_key_a),
        json={
            "policy_id": "production-policy-v1",
            "rules": [
                {
                    "rule_id": "faithfulness-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 95,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert policy.status_code == 201
    assert policy.json()["policy_id"] == "production-policy-v1"
    assert policy.json()["id"] != policy.json()["policy_id"]
    draft = api.client.post(
        f"/v1/systems/{production_system['id']}/test-plans",
        headers=api.authorization(api.live_key_a),
        json={
            "release_policy_id": policy.json()["id"],
            "test_suite_id": suite.json()["id"],
            "test_suite_version": suite.json()["latest_version"]["version"],
        },
    )
    assert draft.status_code == 201
    hidden_plan = api.client.get(
        f"/v1/test-plans/{draft.json()['test_plan_id']}",
        headers=api.authorization(),
    )
    visible_plan = api.client.get(
        f"/v1/test-plans/{draft.json()['test_plan_id']}",
        headers=api.authorization(api.live_key_a),
    )
    test_plan_ids = {
        item["test_plan_id"]
        for item in api.client.get("/v1/test-plans", headers=api.authorization()).json()["items"]
    }
    live_plan_ids = {
        item["test_plan_id"]
        for item in api.client.get(
            "/v1/test-plans", headers=api.authorization(api.live_key_a)
        ).json()["items"]
    }
    assert hidden_plan.status_code == 404
    assert visible_plan.status_code == 200
    assert draft.json()["test_plan_id"] not in test_plan_ids
    assert draft.json()["test_plan_id"] in live_plan_ids
    confirmed = api.client.post(
        f"/v1/test-plans/{draft.json()['test_plan_id']}/confirm",
        headers=api.authorization(api.live_key_a),
        json={
            "confirm": True,
            "expected_profile_fingerprint": draft.json()["profile_fingerprint"],
            "expected_test_plan_version": draft.json()["test_plan_version"],
        },
    )
    assert confirmed.status_code == 200
    run_request = {
        "system_id": production_system["id"],
        "system_version": "9.0.0",
        "environment": "production",
        "test_suite_id": suite.json()["id"],
        "test_suite_version": suite.json()["latest_version"]["version"],
        "test_plan_id": draft.json()["test_plan_id"],
    }
    mismatched_run_request = {**run_request, "environment": "staging"}
    mismatched_run = api.client.post(
        "/v1/test-runs",
        headers=api.authorization(api.live_key_a),
        json=mismatched_run_request,
    )
    assert mismatched_run.status_code == 409
    assert mismatched_run.json()["error"]["code"] == "TEST_PLAN_ENVIRONMENT_MISMATCH"
    production_run = api.client.post(
        "/v1/test-runs",
        headers={
            **api.authorization(api.live_key_a),
            "Idempotency-Key": "production-run",
        },
        json=run_request,
    )
    assert production_run.status_code == 202
    run_id = production_run.json()["run_id"]
    for path in (
        f"/v1/test-runs/{run_id}",
        f"/v1/test-runs/{run_id}/cases",
        f"/v1/test-runs/{run_id}/findings",
        f"/v1/coverage?run_id={run_id}",
    ):
        assert api.client.get(path, headers=api.authorization()).status_code == 404
        assert (
            api.client.get(
                path,
                headers=api.authorization(api.live_key_a),
            ).status_code
            == 200
        )
    assert api.client.get("/v1/test-runs", headers=api.authorization()).json() == {
        "items": [],
        "count": 0,
    }
    assert api.client.get("/v1/coverage", headers=api.authorization()).json() == {
        "items": [],
        "count": 0,
    }
    assert (
        api.client.get("/v1/test-runs", headers=api.authorization(api.live_key_a)).json()["count"]
        == 1
    )
    assert (
        api.client.get("/v1/coverage", headers=api.authorization(api.live_key_a)).json()["count"]
        == 1
    )
    blocked_run_replay = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "production-run"},
        json=run_request,
    )
    assert blocked_run_replay.status_code == 403

    with api.sessions() as session:
        evaluation = session.get(Evaluation, evaluation_id)
        assert evaluation is not None
        assert evaluation.environment == "PRODUCTION"
        assert evaluation.system_version_id == production_version["id"]


def test_evaluation_is_idempotent_tenant_scoped_and_encrypted(api: ApiHarness) -> None:
    system = _create_system(api)
    _create_version(api, str(system["id"]))
    clinical_request = {
        "system_id": system["id"],
        "system_version": "1.4.0",
        "sources": [
            {
                "document_id": "source-one",
                "type": "clinical_transcript",
                "content": "Patient reports left knee pain.",
            }
        ],
        "generated_output": {
            "type": "clinical_note",
            "content": "Patient reports right knee pain.",
        },
    }
    headers = {**api.authorization(), "Idempotency-Key": "evaluation-one"}
    created = api.client.post("/v1/evaluations", headers=headers, json=clinical_request)
    replayed = api.client.post("/v1/evaluations", headers=headers, json=clinical_request)
    assert created.status_code == replayed.status_code == 201
    assert created.headers["Cache-Control"] == "no-store, private"
    assert created.headers["Vary"] == "Authorization"
    assert replayed.headers["Idempotent-Replayed"] == "true"
    assert replayed.json()["evaluation_id"] == created.json()["evaluation_id"]
    assert created.json()["execution_status"] == "COMPLETED"

    cross_tenant = api.client.get(
        f"/v1/evaluations/{created.json()['evaluation_id']}",
        headers=api.authorization(api.key_b),
    )
    assert cross_tenant.status_code == 404
    findings = api.client.get("/v1/findings", headers=api.authorization())
    other_tenant_findings = api.client.get("/v1/findings", headers=api.authorization(api.key_b))
    assert findings.status_code == 200
    assert findings.json()["count"] >= 1
    assert all(
        item["evaluation_id"] == created.json()["evaluation_id"]
        for item in findings.json()["items"]
    )
    assert other_tenant_findings.json() == {"items": [], "count": 0}
    with api.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Evaluation)) == 1
        source = session.scalar(select(SourceDocumentRecord))
        output = session.scalar(select(GeneratedOutputRecord))
        assert source is not None and b"left knee pain" not in (source.content_encrypted or b"")
        assert output is not None and b"right knee pain" not in output.content_encrypted


def test_failed_evaluation_is_idempotent_and_does_not_duplicate_artifacts(
    api: ApiHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _create_system(api)
    version = _create_version(api, str(system["id"]))
    real_service = EvaluationService
    compare_calls = 0

    def fail_compare(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        nonlocal compare_calls
        compare_calls += 1
        raise RuntimeError("request content must never escape through this exception")

    def failing_service(
        session: Session,
        encryptor: EncryptedJsonStore,
        *,
        compare_fn: object = None,
        retention_days: int = 30,
        metrics: OperationalMetrics | None = None,
    ) -> EvaluationService:
        del compare_fn
        return real_service(
            session,
            encryptor,
            compare_fn=fail_compare,
            retention_days=retention_days,
            metrics=metrics,
        )

    monkeypatch.setattr(api_routes, "EvaluationService", failing_service)
    payload = {
        "system_id": system["id"],
        "system_version": "1.4.0",
        "environment": "staging",
        "sources": [{"document_id": "private-source", "content": "Left knee pain."}],
        "generated_output": {"content": "Right knee pain."},
    }
    headers = {**api.authorization(), "Idempotency-Key": "failed-evaluation"}
    first = api.client.post("/v1/evaluations", headers=headers, json=payload)
    second = api.client.post("/v1/evaluations", headers=headers, json=payload)

    assert first.status_code == second.status_code == 500
    first_error = {
        key: value for key, value in first.json()["error"].items() if key != "request_id"
    }
    second_error = {
        key: value for key, value in second.json()["error"].items() if key != "request_id"
    }
    assert (
        first_error
        == second_error
        == {
            "code": "WORKER_FAILED",
            "message": "Assurance could not complete the evaluation.",
            "retryable": False,
        }
    )
    assert compare_calls == 1
    with api.sessions() as session:
        evaluations = tuple(session.scalars(select(Evaluation)))
        assert len(evaluations) == 1
        assert evaluations[0].execution_status == "FAILED"
        assert evaluations[0].environment == "STAGING"
        assert evaluations[0].system_version_id == version["id"]
        assert session.scalar(select(func.count()).select_from(IdempotencyRecord)) == 1


def test_configured_retention_is_applied_to_encrypted_artifacts() -> None:
    api = _build_harness(retention_days=7)
    system = _create_system(api)
    _create_version(api, str(system["id"]))
    before = datetime.now(UTC)
    response = api.client.post(
        "/v1/evaluations",
        headers=api.authorization(),
        json={
            "system_id": system["id"],
            "system_version": "1.4.0",
            "sources": [{"document_id": "s", "content": "Left knee pain."}],
            "generated_output": {"content": "Right knee pain."},
        },
    )
    assert response.status_code == 201
    with api.sessions() as session:
        source = session.scalar(select(SourceDocumentRecord))
        assert source is not None and source.retention_expires_at is not None
        expires = source.retention_expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        assert before + timedelta(days=6, hours=23) < expires
        assert expires < before + timedelta(days=7, minutes=1)


def test_suite_policy_and_run_remain_queued_with_unassessed_coverage(api: ApiHarness) -> None:
    system = _create_system(api)
    _create_version(api, str(system["id"]))
    unattested = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={"slug": "unsafe-suite", "name": "Unattested suite"},
    )
    assert unattested.status_code == 422
    suite = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={
            "slug": "scribe-core",
            "name": "Scribe core",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
            "cases": [
                {
                    "name": "Faithfulness",
                    "test_family": "clinical_faithfulness",
                    "input_data": {},
                    "expectation": {},
                },
                {
                    "name": "Robustness placeholder",
                    "test_family": "robustness",
                    "input_data": {},
                    "expectation": {},
                },
            ],
        },
    )
    assert suite.status_code == 201
    provenance = suite.json()["latest_version"]["definition"]["provenance"]
    assert provenance["data_classification"] == "SYNTHETIC"
    assert provenance["contains_phi"] is False
    assert provenance["no_phi_attested"] is True
    assert provenance["attested_by_api_key_id"]
    assert suite.json()["cases"][1]["evaluator"]["availability"] == "IMPLEMENTED"
    frozen_case_version_id = suite.json()["cases"][0]["version_id"]
    with api.sessions() as session:
        inactive_case = session.get(CaseRecord, suite.json()["cases"][1]["case_id"])
        assert inactive_case is not None
        inactive_case.active = False
        later_case_version = CaseVersionRecord(
            organization_id="org_api_a",
            test_case_id=suite.json()["cases"][0]["case_id"],
            version="2.0.0",
            input_data={"changed": True},
            expectation={},
        )
        session.add(later_case_version)
        session.commit()
        later_case_version_id = later_case_version.id
    policy = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(),
        json={
            "policy_id": "scribe-policy-v1",
            "name": "Primary scribe policy",
            "rules": [
                {
                    "rule_id": "faithfulness-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 95,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert policy.status_code == 201
    assert policy.json()["policy_id"] == "scribe-policy-v1"
    assert policy.json()["id"] != policy.json()["policy_id"]
    same_external_key_other_tenant = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(api.key_b),
        json={
            "policy_id": "scribe-policy-v1",
            "rules": [
                {
                    "rule_id": "faithfulness-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 90,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert same_external_key_other_tenant.status_code == 201
    assert same_external_key_other_tenant.json()["policy_id"] == "scribe-policy-v1"
    assert same_external_key_other_tenant.json()["id"] != policy.json()["id"]
    alias_collision = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(),
        json={
            "policy_id": "secondary-policy",
            "name": "scribe-policy-v1",
            "rules": [
                {
                    "rule_id": "secondary-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 80,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    suite_id_slug_collision = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={
            "slug": suite.json()["id"],
            "name": "Suite whose slug resembles another internal ID",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
        },
    )
    assert alias_collision.status_code == suite_id_slug_collision.status_code == 201
    id_first_suite = api.client.get(
        f"/v1/test-suites/{suite.json()['id']}",
        headers=api.authorization(),
    )
    assert id_first_suite.status_code == 200
    assert id_first_suite.json()["id"] == suite.json()["id"]
    assert id_first_suite.json()["slug"] == "scribe-core"
    ambiguous_policy = api.client.post(
        f"/v1/systems/{system['id']}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": "scribe-policy-v1",
            "test_suite_id": suite.json()["id"],
            "test_suite_version": suite.json()["latest_version"]["version"],
        },
    )
    assert ambiguous_policy.status_code == 404
    assert ambiguous_policy.json()["error"]["code"] == "RELEASE_POLICY_NOT_FOUND"
    draft = api.client.post(
        f"/v1/systems/{system['id']}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": policy.json()["id"],
            "test_suite_id": suite.json()["id"],
            "test_suite_version": suite.json()["latest_version"]["version"],
        },
    )
    assert draft.status_code == 201
    assert draft.json()["status"] == "DRAFT"
    assert draft.json()["application_environment"] == "STAGING"
    retrieved_plan = api.client.get(
        f"/v1/test-plans/{draft.json()['test_plan_id']}",
        headers=api.authorization(),
    )
    listed_plans = api.client.get("/v1/test-plans", headers=api.authorization())
    other_tenant_plan = api.client.get(
        f"/v1/test-plans/{draft.json()['test_plan_id']}",
        headers=api.authorization(api.key_b),
    )
    insufficient_plan_reader = api.client.get(
        "/v1/test-plans", headers=api.authorization(api.read_only_key)
    )
    assert retrieved_plan.status_code == 200
    assert retrieved_plan.json()["test_plan_id"] == draft.json()["test_plan_id"]
    assert listed_plans.json()["items"][0]["test_plan_id"] == draft.json()["test_plan_id"]
    assert other_tenant_plan.status_code == 404
    assert insufficient_plan_reader.status_code == 403
    run_request = {
        "system_id": system["id"],
        "system_version": "1.4.0",
        "environment": "staging",
        "test_suite_id": suite.json()["id"],
        "test_suite_version": suite.json()["latest_version"]["version"],
        "test_plan_id": draft.json()["test_plan_id"],
        "source": {"trigger": "github", "commit_sha": "8f24abc"},
    }
    unconfirmed = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "run-unconfirmed"},
        json=run_request,
    )
    assert unconfirmed.status_code == 409
    assert unconfirmed.json()["error"]["code"] == "TEST_PLAN_NOT_CONFIRMED"
    with api.sessions() as session:
        policy_record = session.get(PolicyRecord, policy.json()["id"])
        assert policy_record is not None
        policy_record.active = False
        session.commit()
    inactive_confirmation = api.client.post(
        f"/v1/test-plans/{draft.json()['test_plan_id']}/confirm",
        headers=api.authorization(),
        json={
            "confirm": True,
            "expected_profile_fingerprint": draft.json()["profile_fingerprint"],
            "expected_test_plan_version": draft.json()["test_plan_version"],
        },
    )
    assert inactive_confirmation.status_code == 409
    assert inactive_confirmation.json()["error"]["code"] == "RELEASE_POLICY_INACTIVE"
    with api.sessions() as session:
        policy_record = session.get(PolicyRecord, policy.json()["id"])
        assert policy_record is not None
        policy_record.active = True
        session.commit()
    confirmed = api.client.post(
        f"/v1/test-plans/{draft.json()['test_plan_id']}/confirm",
        headers=api.authorization(),
        json={
            "confirm": True,
            "expected_profile_fingerprint": draft.json()["profile_fingerprint"],
            "expected_test_plan_version": draft.json()["test_plan_version"],
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "CONFIRMED"
    assert confirmed.json()["confirmed_by_api_key_id"]
    recovered_confirmed = api.client.get(
        f"/v1/test-plans/{draft.json()['test_plan_id']}",
        headers=api.authorization(),
    )
    assert recovered_confirmed.json()["status"] == "CONFIRMED"
    reconfirmed_by_other_key = api.client.post(
        f"/v1/test-plans/{draft.json()['test_plan_id']}/confirm",
        headers=api.authorization(api.live_key_a),
        json={
            "confirm": True,
            "expected_profile_fingerprint": draft.json()["profile_fingerprint"],
            "expected_test_plan_version": draft.json()["test_plan_version"],
        },
    )
    assert reconfirmed_by_other_key.status_code == 409
    assert reconfirmed_by_other_key.json()["error"]["code"] == ("TEST_PLAN_ALREADY_CONFIRMED")
    with api.sessions() as session:
        policy_record = session.get(PolicyRecord, policy.json()["id"])
        assert policy_record is not None
        policy_record.active = False
        session.commit()
    inactive_run = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "run-inactive-policy"},
        json=run_request,
    )
    assert inactive_run.status_code == 409
    assert inactive_run.json()["error"]["code"] == "RELEASE_POLICY_INACTIVE"
    with api.sessions() as session:
        policy_record = session.get(PolicyRecord, policy.json()["id"])
        assert policy_record is not None
        policy_record.active = True
        session.commit()
    headers = {**api.authorization(), "Idempotency-Key": "run-one"}
    created = api.client.post("/v1/test-runs", headers=headers, json=run_request)
    replayed = api.client.post("/v1/test-runs", headers=headers, json=run_request)
    assert created.status_code == replayed.status_code == 202
    assert replayed.json()["run_id"] == created.json()["run_id"]
    assert created.json()["execution_status"] == "QUEUED"
    assert created.json()["planning_status"] == "CONFIRMED"
    assert created.json()["assessment_status"] == "UNASSESSED"
    assert created.json()["release_decision"] is None
    assert created.json()["coverage"]["status"] == "UNASSESSED"
    assert created.json()["coverage"]["planned_cases"] > len(suite.json()["cases"])
    assert created.json()["coverage"]["unassessed_cases"] > 0
    assert any(
        "confirmed plan" in limitation for limitation in created.json()["coverage"]["limitations"]
    )
    assert "clinical_note_generation" in created.json()["coverage"]["missing_required_tests"]

    run_id = created.json()["run_id"]
    restored = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    cases = api.client.get(f"/v1/test-runs/{run_id}/cases", headers=api.authorization())
    findings = api.client.get(f"/v1/test-runs/{run_id}/findings", headers=api.authorization())
    coverage = api.client.get(
        "/v1/coverage", headers=api.authorization(), params={"run_id": run_id}
    )
    assert restored.json()["execution_status"] == "QUEUED"
    assert cases.json()["count"] == 1
    assert all(item["assessment_status"] == "UNASSESSED" for item in cases.json()["items"])
    assert findings.json()["items"] == []
    assert coverage.json()["status"] == "UNASSESSED"
    with api.sessions() as session:
        selected_versions = set(
            session.scalars(
                select(RunCaseRecord.test_case_version_id).where(
                    RunCaseRecord.test_run_id == run_id
                )
            )
        )
    assert frozen_case_version_id in selected_versions
    assert later_case_version_id not in selected_versions

    second_headers = {**api.authorization(), "Idempotency-Key": "run-two"}
    second = api.client.post("/v1/test-runs", headers=second_headers, json=run_request)
    listed = api.client.get("/v1/test-runs", headers=api.authorization())
    other_tenant_runs = api.client.get("/v1/test-runs", headers=api.authorization(api.key_b))
    assert second.status_code == 202
    assert listed.json()["items"][0]["run_id"] == second.json()["run_id"]
    assert listed.json()["items"][1]["run_id"] == run_id
    assert other_tenant_runs.json() == {"items": [], "count": 0}
    with api.sessions() as session:
        assert session.scalar(select(func.count()).select_from(RunRecord)) == 2

        run_record = session.get(RunRecord, run_id)
        assert run_record is not None
        run_record.execution_status = "COMPLETED"
        run_record.release_decision = "PASS"
        reduced_coverage = dict(run_record.coverage or {})
        reduced_coverage["status"] = "REDUCED"
        run_record.coverage = reduced_coverage
        session.commit()
    partially_assessed = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert partially_assessed.json()["assessment_status"] == "UNASSESSED"
    assert partially_assessed.json()["release_decision"] is None

    with api.sessions() as session:
        run_record = session.get(RunRecord, run_id)
        assert run_record is not None
        planned_cases = int((run_record.coverage or {})["planned_cases"])
        valid_reduced_coverage = dict(run_record.coverage or {})
        valid_reduced_coverage.update(
            {
                "status": "REDUCED",
                "completed_cases": planned_cases - 1,
                "technical_failed_cases": 0,
                "unassessed_cases": 1,
                "pending_cases": 0,
                "missing_required_tests": [],
                "degraded_tests": [],
                "valid_for_release": True,
            }
        )
        run_record.coverage = valid_reduced_coverage
        run_record.release_decision = "WARN"
        session.commit()
    reduced_assessment = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert reduced_assessment.json()["assessment_status"] == "ASSESSED"
    assert reduced_assessment.json()["release_decision"] == "WARN"

    with api.sessions() as session:
        run_record = session.get(RunRecord, run_id)
        assert run_record is not None
        full_coverage = dict(run_record.coverage or {})
        full_coverage.update(
            {
                "status": "FULL",
                "completed_cases": planned_cases,
                "technical_failed_cases": 0,
                "unassessed_cases": 0,
                "pending_cases": 0,
                "missing_required_tests": [],
                "degraded_tests": [],
                "valid_for_release": True,
            }
        )
        run_record.coverage = full_coverage
        run_record.release_decision = "PASS"
        session.commit()
    fully_assessed = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert fully_assessed.json()["assessment_status"] == "ASSESSED"
    assert fully_assessed.json()["release_decision"] == "PASS"

    with api.sessions() as session:
        run_record = session.get(RunRecord, run_id)
        assert run_record is not None
        partial_coverage = dict(run_record.coverage or {})
        partial_coverage.update(
            {
                "status": "PARTIAL",
                "completed_cases": planned_cases - 1,
                "unassessed_cases": 1,
                "missing_required_tests": ["clinical_faithfulness"],
                "valid_for_release": False,
            }
        )
        run_record.coverage = partial_coverage
        run_record.release_decision = "BLOCK"
        session.commit()
    clinical_block = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert clinical_block.json()["assessment_status"] == "UNASSESSED"
    assert clinical_block.json()["release_decision"] == "BLOCK"

    with api.sessions() as session:
        run_record = session.get(RunRecord, run_id)
        assert run_record is not None
        technical_coverage = dict(run_record.coverage or {})
        technical_coverage.update(
            {
                "completed_cases": planned_cases - 1,
                "technical_failed_cases": 1,
                "unassessed_cases": 0,
            }
        )
        run_record.coverage = technical_coverage
        session.commit()
    technical_block = api.client.get(f"/v1/test-runs/{run_id}", headers=api.authorization())
    assert technical_block.json()["release_decision"] is None


def test_confirmed_plan_caps_and_filters_its_exact_suite_case_snapshot(
    api: ApiHarness,
) -> None:
    system = _create_system(api)
    _create_version(api, str(system["id"]))
    regression_cases = [
        {
            "name": f"Regression {index:02d}",
            "test_family": "custom_business_regression",
            "input_data": {"index": index},
            "expectation": {},
        }
        for index in range(4)
    ]
    suite_response = api.client.post(
        "/v1/test-suites",
        headers=api.authorization(),
        json={
            "slug": "reusable-profile-suite",
            "name": "Reusable profile suite",
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "no_phi_attested": True,
            "cases": [
                *regression_cases,
                {
                    "name": "RAG-only grounding",
                    "test_family": "rag_grounding",
                    "input_data": {},
                    "expectation": {},
                },
                {
                    "name": "Unselected optional task",
                    "test_family": "clinical_task_performance",
                    "input_data": {},
                    "expectation": {},
                },
            ],
        },
    )
    assert suite_response.status_code == 201, suite_response.text
    suite = suite_response.json()
    regression_version_ids = [
        item["version_id"]
        for item in suite["cases"]
        if item["test_family"] == "custom_business_regression"
    ]
    first_regression_case_id = next(
        item["case_id"]
        for item in suite["cases"]
        if item["test_family"] == "custom_business_regression"
    )
    rag_version_id = next(
        item["version_id"] for item in suite["cases"] if item["test_family"] == "rag_grounding"
    )
    optional_version_id = next(
        item["version_id"]
        for item in suite["cases"]
        if item["test_family"] == "clinical_task_performance"
    )
    policy = api.client.post(
        "/v1/release-policies",
        headers=api.authorization(),
        json={
            "policy_id": "snapshot-policy",
            "rules": [
                {
                    "rule_id": "minimum-score",
                    "metric": "score",
                    "operator": ">=",
                    "threshold": 95,
                    "on_failure": "BLOCK",
                }
            ],
        },
    )
    assert policy.status_code == 201, policy.text
    draft_response = api.client.post(
        f"/v1/systems/{system['id']}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": policy.json()["id"],
            "test_suite_id": suite["id"],
            "test_suite_version": suite["latest_version"]["version"],
            "case_counts": {"custom_business_regression": 2},
        },
    )
    assert draft_response.status_code == 201, draft_response.text
    draft = draft_response.json()
    assert draft["source_suite_case_version_ids"] == [item["version_id"] for item in suite["cases"]]
    assert draft["suite_case_version_ids"] == regression_version_ids[:2]
    assert [item["test_family"] for item in draft["suite_case_bindings"]] == [
        "custom_business_regression",
        "custom_business_regression",
    ]
    assert draft["source_suite_case_snapshot_digest"].startswith("sha256:")
    assert draft["suite_case_snapshot_digest"].startswith("sha256:")
    assert draft["release_policy_snapshot_digest"].startswith("sha256:")
    assert draft["test_suite_content_snapshot_digest"].startswith("sha256:")
    exclusions = {item["case_version_id"]: item["reason"] for item in draft["excluded_suite_cases"]}
    assert exclusions[regression_version_ids[2]] == "QUOTA_EXCEEDED"
    assert exclusions[regression_version_ids[3]] == "QUOTA_EXCEEDED"
    assert exclusions[rag_version_id] == "NOT_APPLICABLE"
    assert exclusions[optional_version_id] == "NOT_SELECTED"

    selectable_optional = api.client.post(
        f"/v1/systems/{system['id']}/test-plans",
        headers=api.authorization(),
        json={
            "release_policy_id": policy.json()["id"],
            "test_suite_id": suite["id"],
            "test_suite_version": suite["latest_version"]["version"],
            "test_plan_version": "1.1",
            "case_counts": {
                "custom_business_regression": 2,
                "clinical_task_performance": 1,
            },
            "include_optional": ["clinical_task_performance"],
        },
    )
    assert selectable_optional.status_code == 201, selectable_optional.text
    assert optional_version_id in selectable_optional.json()["suite_case_version_ids"]
    for invalid_configuration in (
        {"include_optional": ["rag_grounding"]},
        {"case_counts": {"rag_grounding": 1}},
        {"case_counts": {"clinical_task_performance": 1}},
    ):
        invalid = api.client.post(
            f"/v1/systems/{system['id']}/test-plans",
            headers=api.authorization(),
            json={
                "release_policy_id": policy.json()["id"],
                "test_suite_id": suite["id"],
                "test_suite_version": suite["latest_version"]["version"],
                **invalid_configuration,
            },
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "TEST_PLAN_CONFIGURATION_INVALID"

    confirmed = api.client.post(
        f"/v1/test-plans/{draft['test_plan_id']}/confirm",
        headers=api.authorization(),
        json={
            "confirm": True,
            "expected_profile_fingerprint": draft["profile_fingerprint"],
            "expected_test_plan_version": draft["test_plan_version"],
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["suite_case_version_ids"] == regression_version_ids[:2]
    run_request = {
        "system_id": system["id"],
        "system_version": "1.4.0",
        "environment": "staging",
        "test_suite_id": suite["id"],
        "test_suite_version": suite["latest_version"]["version"],
        "test_plan_id": draft["test_plan_id"],
    }
    run_response = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "bounded-suite-run"},
        json=run_request,
    )
    assert run_response.status_code == 202, run_response.text
    run_id = run_response.json()["run_id"]
    run_cases = api.client.get(
        f"/v1/test-runs/{run_id}/cases",
        headers=api.authorization(),
    )
    assert run_cases.status_code == 200
    assert run_cases.json()["count"] == 2
    assert {item["test_family"] for item in run_cases.json()["items"]} == {
        "custom_business_regression"
    }
    with api.sessions() as session:
        queued_version_ids = tuple(
            session.scalars(
                select(RunCaseRecord.test_case_version_id)
                .where(RunCaseRecord.test_run_id == run_id)
                .order_by(RunCaseRecord.created_at, RunCaseRecord.id)
            )
        )
    assert set(queued_version_ids) == set(regression_version_ids[:2])
    assert rag_version_id not in queued_version_ids
    assert optional_version_id not in queued_version_ids

    with api.sessions() as session:
        changed_version = session.get(CaseVersionRecord, regression_version_ids[0])
        assert changed_version is not None
        original_input = dict(changed_version.input_data)
        changed_version.input_data = {**original_input, "tampered_after_confirmation": True}
        session.commit()
    changed_case_content = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "changed-case-content-run"},
        json=run_request,
    )
    assert changed_case_content.status_code == 409
    assert changed_case_content.json()["error"]["code"] == "TEST_PLAN_STALE"
    with api.sessions() as session:
        changed_version = session.get(CaseVersionRecord, regression_version_ids[0])
        assert changed_version is not None
        changed_version.input_data = original_input
        session.commit()

    with api.sessions() as session:
        policy_version = session.get(
            PolicyVersionRecord,
            policy.json()["latest_version"]["id"],
        )
        assert policy_version is not None
        original_rules = list(policy_version.rules)
        policy_version.rules = [*original_rules, {"tampered_after_confirmation": True}]
        session.commit()
    changed_policy_content = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "changed-policy-content-run"},
        json=run_request,
    )
    assert changed_policy_content.status_code == 409
    assert changed_policy_content.json()["error"]["code"] == "TEST_PLAN_STALE"
    with api.sessions() as session:
        policy_version = session.get(
            PolicyVersionRecord,
            policy.json()["latest_version"]["id"],
        )
        assert policy_version is not None
        policy_version.rules = original_rules
        session.commit()

    with api.sessions() as session:
        changed_case = session.get(CaseRecord, first_regression_case_id)
        assert changed_case is not None
        changed_case.test_family = "robustness"
        session.commit()
    changed_family = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "changed-family-run"},
        json=run_request,
    )
    assert changed_family.status_code == 409
    assert changed_family.json()["error"]["code"] == "TEST_PLAN_STALE"
    with api.sessions() as session:
        changed_case = session.get(CaseRecord, first_regression_case_id)
        assert changed_case is not None
        changed_case.test_family = "custom_business_regression"
        session.commit()

    with api.sessions() as session:
        suite_version = session.get(
            SuiteVersionRecord,
            suite["latest_version"]["id"],
        )
        assert suite_version is not None
        changed_definition = dict(suite_version.definition)
        changed_ids = list(changed_definition["case_version_ids"])
        changed_ids[0], changed_ids[1] = changed_ids[1], changed_ids[0]
        changed_definition["case_version_ids"] = changed_ids
        suite_version.definition = changed_definition
        session.commit()
    stale = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "tampered-suite-run"},
        json=run_request,
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "TEST_PLAN_STALE"

    with api.sessions() as session:
        suite_version = session.get(
            SuiteVersionRecord,
            suite["latest_version"]["id"],
        )
        assert suite_version is not None
        restored_definition = dict(suite_version.definition)
        restored_definition["case_version_ids"] = [item["version_id"] for item in suite["cases"]]
        suite_version.definition = restored_definition
        plan_record = session.get(PlanRecord, draft["test_plan_id"])
        assert plan_record is not None
        changed_plan = dict(plan_record.definition)
        not_applicable = list(changed_plan["not_applicable"])
        not_applicable.remove("rag_poisoning")
        changed_plan["not_applicable"] = not_applicable
        changed_plan["unselected_optional"] = [
            *changed_plan["unselected_optional"],
            "rag_poisoning",
        ]
        plan_record.definition = changed_plan
        session.commit()
    stale_partition = api.client.post(
        "/v1/test-runs",
        headers={**api.authorization(), "Idempotency-Key": "stale-partition-run"},
        json=run_request,
    )
    assert stale_partition.status_code == 409
    assert stale_partition.json()["error"]["code"] == "TEST_PLAN_STALE"


def test_size_rate_limits_and_metrics_are_phi_safe() -> None:
    sized = _build_harness(max_request_bytes=1_024)
    too_large = sized.client.post(
        "/v1/systems",
        headers=sized.authorization(),
        json={
            "name": "Large",
            "description": "x" * 2_000,
            "application_type": "OTHER",
            "risk_tier": "LOW",
        },
    )
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "REQ_TOO_LARGE"

    streamed = sized.client.post(
        "/v1/systems",
        headers={**sized.authorization(), "Content-Type": "application/json"},
        content=iter([b'{"name":"', b"x" * 2_000, b'"}']),
    )
    assert streamed.status_code == 413
    assert streamed.json()["error"]["code"] == "REQ_TOO_LARGE"

    limited = _build_harness(
        rate_limiter=InMemoryRateLimiter(requests_per_window=1, window_seconds=60)
    )
    first = limited.client.get("/v1/systems", headers=limited.authorization())
    rejected = limited.client.get("/v1/systems", headers=limited.authorization())
    assert first.status_code == 200
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "REQ_RATE_LIMITED"
    assert rejected.headers["Retry-After"]

    rotating = _build_harness(
        rate_limiter=InMemoryRateLimiter(requests_per_window=1, window_seconds=60)
    )
    invalid_first = rotating.client.get(
        "/v1/systems", headers={"Authorization": "Bearer invalid-one"}
    )
    invalid_second = rotating.client.get(
        "/v1/systems", headers={"Authorization": "Bearer invalid-two"}
    )
    assert invalid_first.status_code == 401
    assert invalid_second.status_code == 429

    limited_app = cast(FastAPI, limited.client.app)
    snapshot = limited_app.state.operational_metrics.snapshot()
    counters = cast(dict[str, float], snapshot["counters"])
    observations = cast(dict[str, object], snapshot["observations"])
    rendered_keys = " ".join(
        [
            *counters.keys(),
            *observations.keys(),
        ]
    )
    assert limited.key_a not in rendered_keys
    assert "Authorization" not in rendered_keys
    assert "/v1/systems" in rendered_keys


def test_production_settings_reject_public_development_secrets() -> None:
    # This test validates the built-in development defaults.  Do not let a
    # developer's local .env change which key is considered development-only.
    development = Settings(_env_file=None)
    assert development.api_key_pepper not in repr(development)
    assert development.data_encryption_keys not in repr(development)
    assert development.database_url not in repr(development)
    with pytest.raises(ValueError, match="production"):
        Settings(environment="production")
    with pytest.raises(ValueError, match="external semantic model runtime is not configured"):
        Settings(semantic_evaluation_enabled=True)
    with pytest.raises(ValueError, match="model and revision are pinned"):
        Settings(ner_model_revision="main")

    secure_key = Fernet.generate_key().decode()
    production_values = {
        "environment": "production",
        "api_key_pepper": secrets.token_urlsafe(48),
        "database_url": (
            "postgresql+psycopg://gatekeeper@db.example/gatekeeper?sslmode=verify-full"
        ),
    }
    with pytest.raises(ValueError, match="development encryption key"):
        Settings.model_validate(
            {
                **production_values,
                "data_encryption_keys": f"  {development.data_encryption_keys}  ",
            }
        )
    with pytest.raises(ValueError, match="development encryption key"):
        Settings.model_validate(
            {
                **production_values,
                "data_encryption_keys": f"{development.data_encryption_keys},{secure_key}",
            }
        )
    for equivalent_development_key in (
        f"{development.data_encryption_keys}=",
        (
            f"{development.data_encryption_keys[:12]}\n"
            f"{development.data_encryption_keys[12:]}"
        ),
    ):
        with pytest.raises(ValueError, match="development encryption key"):
            Settings.model_validate(
                {
                    **production_values,
                    "data_encryption_keys": equivalent_development_key,
                }
            )
    with pytest.raises(ValueError, match="API key pepper"):
        Settings.model_validate(
            {
                **production_values,
                "api_key_pepper": "development-only-change-me       ",
                "data_encryption_keys": secure_key,
            },
        )
    for unsafe_database_url in (
        "mysql://gatekeeper:secret@db.example/gatekeeper",
        "postgresql+psycopg://gatekeeper:gatekeeper@db.example/gatekeeper",
        "postgresql+psycopg://app:gatekeeper@db.example/gatekeeper?sslmode=verify-full",
        (
            "postgresql+psycopg://app@db.example/gatekeeper"
            "?sslmode=verify-full&password=gatekeeper"
        ),
    ):
        with pytest.raises(ValueError, match="PostgreSQL"):
            Settings.model_validate(
                {
                    **production_values,
                    "database_url": unsafe_database_url,
                    "data_encryption_keys": secure_key,
                }
            )
    for database_url_without_approved_tls in (
        "postgresql+psycopg://app:secret@db.example/gatekeeper",
        "postgresql+psycopg://app:secret@db.example/gatekeeper?sslmode=disable",
        "postgresql+psycopg://app:secret@db.example/gatekeeper?sslmode=prefer",
    ):
        with pytest.raises(ValueError, match="sslmode"):
            Settings.model_validate(
                {
                    **production_values,
                    "database_url": database_url_without_approved_tls,
                    "data_encryption_keys": secure_key,
                }
            )

    validated = Settings.model_validate({**production_values, "data_encryption_keys": secure_key})
    assert "sslmode=verify-full" in validated.database_url


def _example_api_key_pepper() -> str:
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    prefix = "GATEKEEPER_API_KEY_PEPPER="
    return next(
        line.removeprefix(prefix)
        for line in env_example.read_text().splitlines()
        if line.startswith(prefix)
    )


def _production_secret_settings(api_key_pepper: str) -> Settings:
    return Settings(
        environment="production",
        api_key_pepper=api_key_pepper,
        data_encryption_keys=Fernet.generate_key().decode(),
        database_url="postgresql+psycopg://app@db.example/gatekeeper?sslmode=verify-full",
    )


def test_production_settings_reject_placeholder_and_low_diversity_peppers() -> None:
    example_pepper = _example_api_key_pepper()
    for unsafe_pepper in (
        "development-only-change-me",
        example_pepper,
        f"  {example_pepper.upper()}  ",
        f"{example_pepper}-extra",
        "repl4ce-with-a-long-random-secret-0123456789",
        "ci-only-test-pepper-not-a-production-secret",
        "p" * 64,
    ):
        with pytest.raises(ValueError, match="randomly generated API key pepper"):
            _production_secret_settings(unsafe_pepper)


def test_production_settings_accept_strong_random_api_key_pepper() -> None:
    random_pepper = secrets.token_urlsafe(48)

    validated = _production_secret_settings(random_pepper)

    assert validated.api_key_pepper == random_pepper
    assert random_pepper not in repr(validated)


@pytest.mark.parametrize("environment", ["development", "test"])
def test_non_production_settings_intentionally_allow_example_pepper(
    environment: Literal["development", "test"],
) -> None:
    example_pepper = _example_api_key_pepper()

    validated = Settings(environment=environment, api_key_pepper=example_pepper)

    assert validated.api_key_pepper == example_pepper

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from backend.app.ci import GateConfig, run_release_gate


def _config() -> GateConfig:
    return GateConfig(
        api_url="https://gatekeeper.example",
        api_key="gk_test_secret",
        system_id="sys_1",
        system_version="1.0",
        environment="staging",
        test_suite_id="suite_1",
        test_suite_version="2.3.0",
        test_plan_id="plan_1",
        commit_sha="abc123",
        branch="feature/test",
        github_run_id="123456",
        github_run_attempt=1,
        poll_interval_seconds=0,
    )


def _full_coverage() -> dict[str, object]:
    return {
        "status": "FULL",
        "valid_for_release": True,
        "planned_tests": 1,
        "planned_cases": 2,
        "completed_cases": 2,
        "technical_failed_cases": 0,
        "unassessed_cases": 0,
        "pending_cases": 0,
        "missing_required_tests": [],
        "degraded_tests": [],
    }


def _complete_status(decision: str) -> dict[str, object]:
    return {
        "execution_status": "COMPLETED",
        "assessment_status": "ASSESSED",
        "release_decision": decision,
        "coverage": _full_coverage(),
    }


def test_block_exits_nonzero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_1", "execution_status": "QUEUED"})
        return httpx.Response(
            200,
            json=_complete_status("BLOCK"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)

    assert outcome.exit_code == 1
    assert outcome.release_decision == "BLOCK"


def test_technical_failure_cannot_be_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_1"})
        return httpx.Response(200, json={"execution_status": "FAILED"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)

    assert outcome.exit_code == 1
    assert outcome.release_decision is None
    assert "could not complete" in outcome.message


def test_pass_exits_zero_and_secret_is_only_in_authorization_header() -> None:
    seen_urls: list[str] = []
    seen_payloads: list[dict[str, object]] = []
    assert _config().api_key not in repr(_config())

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer gk_test_secret"
        if request.method == "POST":
            seen_payloads.append(json.loads(request.content))
            return httpx.Response(202, json={"run_id": "run_1"})
        return httpx.Response(
            200,
            json=_complete_status("PASS"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)

    assert outcome.exit_code == 0
    assert all("gk_test_secret" not in url for url in seen_urls)
    assert seen_payloads == [
        {
            "system_id": "sys_1",
            "system_version": "1.0",
            "environment": "staging",
            "test_suite_id": "suite_1",
            "test_suite_version": "2.3.0",
            "test_plan_id": "plan_1",
            "source": {
                "trigger": "github",
                "commit_sha": "abc123",
                "branch": "feature/test",
                "pull_request": None,
            },
        }
    ]


def test_idempotency_identity_changes_for_each_resolved_gate_attempt() -> None:
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            idempotency_keys.append(request.headers["Idempotency-Key"])
            return httpx.Response(202, json={"run_id": f"run_{len(idempotency_keys)}"})
        return httpx.Response(
            200,
            json=_complete_status("PASS"),
        )

    base = _config()
    configs = (
        base,
        replace(base, github_run_attempt=2),
        replace(base, test_suite_version="2.4.0"),
        replace(base, test_plan_id="plan_2"),
        replace(base, branch="release/candidate"),
        replace(base, pull_request=284),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for config in configs:
            assert (
                run_release_gate(
                    config,
                    client=client,
                    sleep=lambda _seconds: None,
                ).exit_code
                == 0
            )

    assert len(set(idempotency_keys)) == len(configs)
    assert all(key.startswith("github:") and len(key) == 71 for key in idempotency_keys)


@pytest.mark.parametrize(
    "terminal_body",
    [
        {"execution_status": "COMPLETED", "release_decision": "PASS"},
        {
            **_complete_status("PASS"),
            "assessment_status": "UNASSESSED",
        },
        {
            **_complete_status("PASS"),
            "coverage": {
                **_full_coverage(),
                "status": "PARTIAL",
            },
        },
        {
            **_complete_status("WARN"),
            "coverage": {
                **_full_coverage(),
                "valid_for_release": False,
            },
        },
        {
            **_complete_status("PASS"),
            "coverage": {
                **_full_coverage(),
                "technical_failed_cases": 1,
            },
        },
        {
            **_complete_status("PASS"),
            "coverage": {
                **_full_coverage(),
                "planned_cases": 1,
                "completed_cases": 2,
            },
        },
        {
            **_complete_status("PASS"),
            "coverage": {
                **_full_coverage(),
                "planned_cases": True,
            },
        },
    ],
)
def test_positive_decision_with_incomplete_coverage_fails_closed(
    terminal_body: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_invalid"})
        return httpx.Response(200, json=terminal_body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)

    assert outcome.exit_code == 1
    assert outcome.release_decision is None
    assert "release-eligible" in outcome.message


@pytest.mark.parametrize("fail_on_warn,expected_exit", [(False, 0), (True, 1)])
def test_valid_reduced_warn_preserves_warn_gate_semantics(
    fail_on_warn: bool, expected_exit: int
) -> None:
    terminal = _complete_status("WARN")
    terminal["coverage"] = {
        **_full_coverage(),
        "status": "REDUCED",
        "completed_cases": 1,
        "unassessed_cases": 1,
        "valid_for_release": True,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_warn"})
        return httpx.Response(200, json=terminal)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(
            replace(_config(), fail_on_warn=fail_on_warn),
            client=client,
            sleep=lambda _seconds: None,
        )

    assert outcome.exit_code == expected_exit
    assert outcome.release_decision == "WARN"


def test_incomplete_clinical_block_is_reported_but_technical_block_is_rejected() -> None:
    incomplete_block = _complete_status("BLOCK")
    incomplete_block["assessment_status"] = "UNASSESSED"
    incomplete_block["coverage"] = {
        **_full_coverage(),
        "status": "PARTIAL",
        "completed_cases": 1,
        "unassessed_cases": 1,
        "missing_required_tests": ["SAFETY_REFUSAL"],
        "valid_for_release": False,
    }
    technical_block = _complete_status("BLOCK")
    technical_block["coverage"] = {
        **_full_coverage(),
        "status": "REDUCED",
        "completed_cases": 1,
        "technical_failed_cases": 1,
        "valid_for_release": False,
    }
    terminal_bodies = iter([incomplete_block, technical_block])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "run_block"})
        return httpx.Response(200, json=next(terminal_bodies))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        clinical = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)
        technical = run_release_gate(_config(), client=client, sleep=lambda _seconds: None)

    assert clinical.release_decision == "BLOCK"
    assert clinical.exit_code == 1
    assert technical.release_decision is None
    assert technical.exit_code == 1


@pytest.mark.parametrize(
    "api_url",
    [
        "http://gatekeeper.example",
        "https://user:secret@gatekeeper.example",
        "https://gatekeeper.example/base",
        "https://gatekeeper.example?token=secret",
        "https://gatekeeper.example#fragment",
        "https://127.0.0.1",
        "https://127\u30020\u30020\u30021",
        "https://127\uff0e0\uff0e0\uff0e1",
        "https://127\uff610\uff610\uff611",
        "https://localhost\u3002",
        "https://2130706433",
        "https://127.1",
        "https://0x7f000001",
        "https://192.0.0.8",
        "https://[64:ff9b:1::1]",
        "https://[2002::1]",
    ],
)
def test_release_gate_rejects_unsafe_api_url_before_sending_key(api_url: str) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        nonlocal requests
        requests += 1
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = run_release_gate(replace(_config(), api_url=api_url), client=client)

    assert outcome.exit_code == 1
    assert outcome.release_decision is None
    assert "secure HTTPS origin" in outcome.message
    assert requests == 0

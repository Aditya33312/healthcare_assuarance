"""GitHub-compatible Gatekeeper test-run trigger and poller."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from backend.app.security.endpoints import validate_public_https_endpoint

_TERMINAL_EXECUTION_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


@dataclass(frozen=True, slots=True)
class GateConfig:
    api_url: str
    api_key: str = field(repr=False)
    system_id: str
    system_version: str
    environment: str
    test_suite_id: str
    test_suite_version: str
    test_plan_id: str
    commit_sha: str
    branch: str
    github_run_id: str
    github_run_attempt: int
    pull_request: int | None = None
    timeout_seconds: float = 1_800
    poll_interval_seconds: float = 5
    fail_on_warn: bool = False


@dataclass(frozen=True, slots=True)
class GateOutcome:
    exit_code: int
    run_id: str | None
    execution_status: str
    release_decision: str | None
    message: str


def run_release_gate(
    config: GateConfig,
    *,
    client: httpx.Client | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GateOutcome:
    """Trigger a run and fail closed on all technical or ambiguous outcomes."""

    try:
        api_url = _validated_api_base_url(config.api_url)
    except ValueError:
        return _technical(None, "Assurance API URL is not a secure HTTPS origin.")

    owns_client = client is None
    http_client = client or httpx.Client(timeout=30)
    identity = json.dumps(
        {
            "system_id": config.system_id,
            "system_version": config.system_version,
            "environment": config.environment,
            "test_suite_id": config.test_suite_id,
            "test_suite_version": config.test_suite_version,
            "test_plan_id": config.test_plan_id,
            "commit_sha": config.commit_sha,
            "branch": config.branch,
            "pull_request": config.pull_request,
            "github_run_id": config.github_run_id,
            "github_run_attempt": config.github_run_attempt,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Idempotency-Key": f"github:{hashlib.sha256(identity).hexdigest()}",
    }
    payload = {
        "system_id": config.system_id,
        "system_version": config.system_version,
        "environment": config.environment,
        "test_suite_id": config.test_suite_id,
        "test_suite_version": config.test_suite_version,
        "test_plan_id": config.test_plan_id,
        "source": {
            "trigger": "github",
            "commit_sha": config.commit_sha,
            "branch": config.branch,
            "pull_request": config.pull_request,
        },
    }
    try:
        response = http_client.post(
            f"{api_url}/v1/test-runs",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        run_id = body.get("run_id") or body.get("id")
        if not isinstance(run_id, str) or not run_id:
            return _technical(None, "Gatekeeper returned no run identifier.")

        deadline = monotonic() + config.timeout_seconds
        while True:
            status_response = http_client.get(
                f"{api_url}/v1/test-runs/{run_id}",
                headers={"Authorization": headers["Authorization"]},
            )
            status_response.raise_for_status()
            status_body = status_response.json()
            execution_status = str(status_body.get("execution_status", "")).upper()
            release_decision_value = status_body.get("release_decision")
            release_decision = (
                str(release_decision_value).upper() if release_decision_value is not None else None
            )
            if execution_status in _TERMINAL_EXECUTION_STATUSES:
                return _terminal_outcome(
                    run_id,
                    execution_status,
                    release_decision,
                    release_contract_valid=_release_contract_valid(status_body, release_decision),
                    fail_on_warn=config.fail_on_warn,
                )
            if monotonic() >= deadline:
                return _technical(run_id, "Assurance timed out before a release decision.")
            sleep(config.poll_interval_seconds)
    except (httpx.HTTPError, ValueError) as exc:
        return _technical(None, f"Assurance could not complete: {type(exc).__name__}.")
    finally:
        if owns_client:
            http_client.close()


def _validated_api_base_url(value: str) -> str:
    validated = validate_public_https_endpoint(value)
    parsed = urlsplit(validated)
    if parsed.path not in {"", "/"}:
        raise ValueError("Gatekeeper API URL must be an origin without a path")
    return f"{parsed.scheme}://{parsed.netloc}"


def _terminal_outcome(
    run_id: str,
    execution_status: str,
    release_decision: str | None,
    *,
    release_contract_valid: bool,
    fail_on_warn: bool,
) -> GateOutcome:
    if execution_status != "COMPLETED":
        return _technical(run_id, "Assurance could not complete.", execution_status)
    if release_decision == "PASS":
        if not release_contract_valid:
            return _technical(
                run_id,
                "Gatekeeper did not return a complete release-eligible assessment.",
                execution_status,
            )
        return GateOutcome(0, run_id, execution_status, release_decision, "Gatekeeper PASS.")
    if release_decision == "WARN":
        if not release_contract_valid:
            return _technical(
                run_id,
                "Gatekeeper did not return a complete release-eligible assessment.",
                execution_status,
            )
        return GateOutcome(
            1 if fail_on_warn else 0,
            run_id,
            execution_status,
            release_decision,
            "Gatekeeper WARN.",
        )
    if release_decision == "BLOCK":
        if not release_contract_valid:
            return _technical(
                run_id,
                "Gatekeeper did not return an internally consistent clinical block.",
                execution_status,
            )
        return GateOutcome(1, run_id, execution_status, release_decision, "Gatekeeper BLOCK.")
    return _technical(run_id, "No valid release decision was produced.", execution_status)


def _release_contract_valid(
    status_body: Mapping[str, object], release_decision: str | None
) -> bool:
    coverage = status_body.get("coverage")
    if not isinstance(coverage, Mapping):
        return False

    status = coverage.get("status")
    if status not in {"FULL", "REDUCED", "PARTIAL", "UNASSESSED"}:
        return False
    count_fields = (
        "planned_tests",
        "planned_cases",
        "completed_cases",
        "technical_failed_cases",
        "unassessed_cases",
        "pending_cases",
    )
    counts: dict[str, int] = {}
    for field_name in count_fields:
        value = coverage.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False
        counts[field_name] = value
    if counts["planned_tests"] == 0 and counts["planned_cases"] != 0:
        return False
    if (
        counts["completed_cases"]
        + counts["technical_failed_cases"]
        + counts["unassessed_cases"]
        + counts["pending_cases"]
        != counts["planned_cases"]
    ):
        return False
    missing_required = coverage.get("missing_required_tests")
    degraded = coverage.get("degraded_tests")
    if not isinstance(missing_required, list) or not isinstance(degraded, list):
        return False
    if len(missing_required) != len(set(map(str, missing_required))):
        return False
    if len(degraded) != len(set(map(str, degraded))):
        return False
    valid_for_release = coverage.get("valid_for_release")
    if not isinstance(valid_for_release, bool):
        return False

    if status == "FULL":
        structurally_valid = (
            counts["planned_tests"] > 0
            and counts["planned_cases"] > 0
            and counts["completed_cases"] == counts["planned_cases"]
            and counts["technical_failed_cases"] == 0
            and counts["unassessed_cases"] == 0
            and counts["pending_cases"] == 0
            and not missing_required
            and not degraded
            and valid_for_release
        )
    elif status == "REDUCED":
        structurally_valid = (
            counts["planned_cases"] > 0
            and counts["completed_cases"] > 0
            and not missing_required
            and bool(
                counts["technical_failed_cases"]
                or counts["unassessed_cases"]
                or counts["pending_cases"]
                or degraded
            )
            and (
                not valid_for_release
                or (counts["technical_failed_cases"] == 0 and counts["pending_cases"] == 0)
            )
        )
    elif status == "PARTIAL":
        structurally_valid = (
            counts["planned_cases"] > 0
            and counts["completed_cases"] > 0
            and bool(missing_required)
            and not valid_for_release
        )
    else:
        structurally_valid = (
            counts["planned_cases"] == 0 or counts["completed_cases"] == 0
        ) and not valid_for_release
    if not structurally_valid or counts["technical_failed_cases"] != 0:
        return False

    if release_decision == "BLOCK":
        return True
    if str(status_body.get("assessment_status", "")).upper() != "ASSESSED":
        return False
    if release_decision == "PASS":
        return status == "FULL" and valid_for_release
    if release_decision == "WARN":
        return status in {"FULL", "REDUCED"} and valid_for_release
    return False


def _technical(
    run_id: str | None,
    message: str,
    execution_status: str = "FAILED",
) -> GateOutcome:
    return GateOutcome(1, run_id, execution_status, None, message)

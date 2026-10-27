"""Focused adapter and failure-safe orchestration tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from backend.app.adapters import (
    AdapterError,
    AdapterRateLimitError,
    AdapterRequestError,
    AdapterResponse,
    AdapterSchemaError,
    AdapterServerError,
    AdapterTimeoutError,
    AdapterTransportError,
    AdapterUnauthorizedError,
    AdapterUsage,
    CustomJSONFieldMappingAdapter,
    GenericRESTAdapter,
    OpenAICompatibleAdapter,
)
from backend.app.adapters.json_paths import get_path, set_path
from backend.app.adapters.protocols import HTTPResponse
from backend.app.orchestration import (
    AssessmentStatus,
    BackgroundRunDispatcher,
    CaseExecutionStatus,
    OrchestrationCase,
    RetryPolicy,
    RunCoverage,
    RunExecutionStatus,
)
from backend.app.orchestration import (
    TestOrchestrator as RunOrchestrator,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: Any,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers: Mapping[str, str] = dict(headers or {})

    def json(self) -> Any:
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeHTTPClient:
    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def request(self, method: str, url: str, **kwargs: Any) -> HTTPResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://customer.example/generate",
        "https://localhost/generate",
        "https://127.0.0.1/generate",
        "https://127\u30020\u30020\u30021/generate",
        "https://127\uff0e0\uff0e0\uff0e1/generate",
        "https://127\uff610\uff610\uff611/generate",
        "https://localhost\u3002/generate",
        "https://2130706433/generate",
        "https://127.1/generate",
        "https://0x7f000001/generate",
        "https://169.254.169.254/latest/meta-data",
        "https://192.0.0.8/generate",
        "https://[64:ff9b:1::1]/generate",
        "https://[2002::1]/generate",
        "https://user:secret@customer.example/generate",
        "https://customer.example/generate?api_key=supersecret",
        "https://customer.example/generate#token=supersecret",
    ],
)
def test_generic_rest_rejects_insecure_or_local_endpoints(endpoint: str) -> None:
    client = FakeHTTPClient(FakeResponse(200, {"output": "must not be requested"}))
    with pytest.raises(ValueError):
        GenericRESTAdapter(endpoint, client=client)
    assert client.requests == []


@pytest.mark.asyncio
async def test_generic_rest_passes_json_and_extracts_configured_output_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clinical_text = "Patient has sensitive left knee pain."
    client = FakeHTTPClient(
        FakeResponse(200, {"result": {"note": clinical_text}}, {"x-request-id": "req-1"})
    )
    adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        response_path="result.note",
        headers={"authorization": "Bearer secret"},
        client=client,
    )

    response = await adapter.invoke({"source": clinical_text})

    assert response.output == clinical_text
    assert response.response_id == "req-1"
    assert client.requests[0]["json"] == {"source": clinical_text}
    assert clinical_text not in repr(adapter)
    assert clinical_text not in repr(response)
    assert clinical_text not in caplog.text
    assert "secret" not in repr(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_error", "retryable"),
    [
        (401, AdapterUnauthorizedError, False),
        (429, AdapterRateLimitError, True),
        (400, AdapterRequestError, False),
        (500, AdapterServerError, False),
        (502, AdapterServerError, True),
        (503, AdapterServerError, True),
        (504, AdapterServerError, False),
    ],
)
async def test_generic_rest_classifies_status_without_exposing_body(
    status: int,
    expected_error: type[AdapterError],
    retryable: bool,
) -> None:
    sensitive_body = "sensitive clinical response"
    adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=FakeHTTPClient(FakeResponse(status, {"error": sensitive_body})),
    )

    with pytest.raises(expected_error) as caught:
        await adapter.invoke({"source": "sensitive clinical request"})

    assert caught.value.retryable is retryable
    assert sensitive_body not in str(caught.value)


@pytest.mark.asyncio
async def test_generic_rest_sanitizes_transport_timeout_and_invalid_json() -> None:
    timeout_adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=FakeHTTPClient(httpx.ReadTimeout("timed out")),
    )
    with pytest.raises(AdapterTimeoutError):
        await timeout_adapter.invoke({"source": "synthetic"})

    request = httpx.Request("POST", "https://customer.example/generate")
    transport_adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=FakeHTTPClient(httpx.ConnectError("failed", request=request)),
    )
    with pytest.raises(AdapterTransportError):
        await transport_adapter.invoke({"source": "synthetic"})

    invalid_json_adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=FakeHTTPClient(FakeResponse(200, ValueError("not JSON"))),
    )
    with pytest.raises(AdapterSchemaError, match="valid JSON"):
        await invalid_json_adapter.invoke({"source": "synthetic"})

    oversized_adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=FakeHTTPClient(FakeResponse(200, {"output": "x" * 100})),
        max_response_bytes=32,
    )
    with pytest.raises(AdapterSchemaError, match="size limit"):
        await oversized_adapter.invoke({"source": "synthetic"})


@pytest.mark.asyncio
async def test_generic_rest_usage_extractor_and_invocation_aliases() -> None:
    client = FakeHTTPClient(
        FakeResponse(200, {"output": "one", "bill": 0.1}),
        FakeResponse(200, {"output": "two", "bill": 0.2}),
        FakeResponse(200, {"output": "three", "bill": 0.3}),
    )
    adapter = GenericRESTAdapter(
        "https://customer.example/generate",
        client=client,
        usage_extractor=lambda body: AdapterUsage(cost_usd=float(body["bill"])),
    )

    assert (await adapter.execute({})).cost_usd == pytest.approx(0.1)
    assert (await adapter.generate({})).cost_usd == pytest.approx(0.2)
    assert (await adapter.call({})).cost_usd == pytest.approx(0.3)
    await adapter.aclose()
    assert client.closed is False  # Injected clients remain caller-owned.


def test_json_paths_support_pointer_list_and_nested_writes() -> None:
    data = {"choices": [{"message": {"content": "note"}}]}
    assert get_path(data, "/choices/0/message/content") == "note"
    assert get_path(data, ("choices", 0, "message", "content")) == "note"
    target: dict[str, Any] = {}
    set_path(target, "request.patient.note", "synthetic")
    assert target == {"request": {"patient": {"note": "synthetic"}}}
    with pytest.raises(AdapterSchemaError):
        get_path(data, "choices.1")


@pytest.mark.asyncio
async def test_custom_json_adapter_maps_nested_request_and_response_fields() -> None:
    client = FakeHTTPClient(FakeResponse(200, {"data": {"soap_note": "Generated note"}}))
    adapter = CustomJSONFieldMappingAdapter(
        "https://customer.example/notes",
        request_mapping={
            "source": "request.consultation_text",
            "settings.note_type": "request.note_type",
        },
        static_fields={"request.schema_version": "1"},
        response_path="data.soap_note",
        client=client,
    )

    response = await adapter.invoke(
        {"source": "Synthetic transcript", "settings": {"note_type": "SOAP"}}
    )

    assert client.requests[0]["json"] == {
        "request": {
            "schema_version": "1",
            "consultation_text": "Synthetic transcript",
            "note_type": "SOAP",
        }
    }
    assert response.output == "Generated note"


@pytest.mark.asyncio
async def test_custom_mapping_missing_source_stops_before_http() -> None:
    client = FakeHTTPClient(FakeResponse(200, {"output": "unused"}))
    adapter = CustomJSONFieldMappingAdapter(
        "https://customer.example/notes",
        request_mapping={"source": "consultation_text"},
        client=client,
    )

    with pytest.raises(AdapterSchemaError):
        await adapter.invoke({"wrong": "field"})

    assert client.requests == []


@pytest.mark.asyncio
async def test_openai_compatible_translation_usage_and_cost() -> None:
    client = FakeHTTPClient(
        FakeResponse(
            200,
            {
                "choices": [{"message": {"content": "Generated summary"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://compatible.example",
        model="clinical-model-v1",
        api_key="not-for-logs",
        system_prompt="Summarize faithfully.",
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.02,
        client=client,
    )

    response = await adapter.invoke({"source": "Synthetic note"})

    request = client.requests[0]
    assert request["url"] == "https://compatible.example/v1/chat/completions"
    assert request["json"] == {
        "model": "clinical-model-v1",
        "messages": [
            {"role": "system", "content": "Summarize faithfully."},
            {"role": "user", "content": "Synthetic note"},
        ],
    }
    assert response.output == "Generated summary"
    assert response.usage.total_tokens == 120
    assert response.cost_usd == pytest.approx(0.0014)
    assert "not-for-logs" not in repr(adapter)


@pytest.mark.asyncio
async def test_openai_bad_response_is_non_retryable_fatal_schema_error() -> None:
    adapter = OpenAICompatibleAdapter(
        "https://compatible.example/v1/chat/completions",
        model="model",
        client=FakeHTTPClient(FakeResponse(200, {"choices": []})),
    )

    with pytest.raises(AdapterSchemaError) as caught:
        await adapter.invoke({"prompt": "Synthetic prompt"})

    assert caught.value.retryable is False
    assert caught.value.fatal is True


@pytest.mark.asyncio
async def test_openai_accepts_messages_and_text_part_responses() -> None:
    client = FakeHTTPClient(
        FakeResponse(
            200,
            {"choices": [{"message": {"content": [{"text": "part one"}, {"text": " + two"}]}}]},
        )
    )
    adapter = OpenAICompatibleAdapter(
        base_url="https://compatible.example/v1/chat/completions",
        model="model",
        client=client,
    )

    response = await adapter.invoke({"messages": [{"role": "user", "content": "Synthetic prompt"}]})

    assert response.output == "part one + two"
    assert client.requests[0]["url"] == "https://compatible.example/v1/chat/completions"


class ScriptedAdapter:
    name = "scripted"
    version = "scripted@1"

    def __init__(self, outcomes: list[AdapterResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def invoke(self, _payload: Mapping[str, Any]) -> AdapterResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        outcome = self.outcomes.pop(0)
        self.active -= 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(output: str, cost: float = 0.0) -> AdapterResponse:
    return AdapterResponse(
        output=output,
        status_code=200,
        latency_ms=1.0,
        usage=AdapterUsage(cost_usd=cost),
    )


async def _evaluator(case: OrchestrationCase, response: AdapterResponse) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "assessment_status": "ASSESSED",
        "execution_status": "COMPLETED",
        "coverage": {"status": "FULL"},
        "output_length": len(str(response.output)),
    }


@pytest.mark.asyncio
async def test_orchestrator_captures_output_evaluation_cost_and_durable_transitions() -> None:
    transitions = []

    async def persist(transition):  # type: ignore[no-untyped-def]
        transitions.append(transition)

    adapter = ScriptedAdapter([_response("Synthetic output", cost=0.125)])
    orchestrator = RunOrchestrator(
        adapter,
        _evaluator,
        transition_callback=persist,
        jitter=lambda _delay: 0,
    )

    result = await orchestrator.run(
        "run-1", [OrchestrationCase("case-1", {"source": "Sensitive input"})]
    )

    case_result = result.case_results[0]
    assert result.execution_status == RunExecutionStatus.COMPLETED
    assert result.coverage == RunCoverage.FULL
    assert result.total_attempts == 1
    assert result.total_cost_usd == pytest.approx(0.125)
    assert result.release_decision is None
    assert case_result.execution_status == CaseExecutionStatus.COMPLETED
    assert case_result.assessment_status == AssessmentStatus.ASSESSED
    assert case_result.output == "Synthetic output"
    assert case_result.evaluation == {
        "case_id": "case-1",
        "assessment_status": "ASSESSED",
        "execution_status": "COMPLETED",
        "coverage": {"status": "FULL"},
        "output_length": 16,
    }
    assert [transition.to_status for transition in transitions] == [
        "QUEUED",
        "PREPARING",
        "QUEUED",
        "RUNNING",
        "RUNNING",
        "EVALUATING",
        "COMPLETED",
        "AGGREGATING",
        "COMPLETED",
    ]
    assert "Sensitive input" not in repr(transitions)


@pytest.mark.asyncio
async def test_orchestrator_enforces_bounded_concurrency() -> None:
    adapter = ScriptedAdapter([_response(str(index)) for index in range(8)])
    orchestrator = RunOrchestrator(adapter, _evaluator, max_concurrency=2)
    cases = [OrchestrationCase(f"case-{index}", {"source": str(index)}) for index in range(8)]

    result = await orchestrator.run("run-concurrency", cases)

    assert result.cases_completed == 8
    assert adapter.max_active <= 2


@pytest.mark.asyncio
async def test_orchestrator_retries_only_transient_allowlist_with_exponential_backoff() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    adapter = ScriptedAdapter(
        [
            AdapterServerError(502),
            AdapterRateLimitError(),
            _response("Recovered", cost=0.25),
        ]
    )
    orchestrator = RunOrchestrator(
        adapter,
        _evaluator,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=1, max_delay_seconds=10),
        jitter=lambda _delay: 0,
        sleep=record_sleep,
    )

    result = await orchestrator.run(
        "run-retry", [OrchestrationCase("case-retry", {"source": "synthetic"})]
    )

    assert delays == [1, 2]
    assert result.total_attempts == 3
    assert result.total_cost_usd == pytest.approx(0.25)
    assert [attempt.outcome for attempt in result.case_results[0].attempt_history] == [
        "RETRYING",
        "RETRYING",
        "SUCCEEDED",
    ]


@pytest.mark.asyncio
async def test_non_allowlisted_500_is_not_retried_and_cannot_pass() -> None:
    adapter = ScriptedAdapter([AdapterServerError(500), _response("must not run")])
    result = await RunOrchestrator(adapter, _evaluator).run(
        "run-500", [OrchestrationCase("case-500", {"source": "synthetic"})]
    )

    case_result = result.case_results[0]
    assert adapter.calls == 1
    assert case_result.execution_status == CaseExecutionStatus.UNASSESSED
    assert case_result.release_decision is None
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.coverage == RunCoverage.UNASSESSED
    assert result.release_decision is None


@pytest.mark.asyncio
async def test_unauthorized_stops_pending_cases_without_attempts() -> None:
    adapter = ScriptedAdapter([AdapterUnauthorizedError(), _response("must not run")])
    orchestrator = RunOrchestrator(adapter, _evaluator, max_concurrency=1)

    result = await orchestrator.run(
        "run-401",
        [
            OrchestrationCase("case-1", {"source": "one"}),
            OrchestrationCase("case-2", {"source": "two"}),
        ],
    )

    assert adapter.calls == 1
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.failure_code == "AI_UNAUTHORIZED"
    assert [case.attempts for case in result.case_results] == [1, 0]
    assert all(
        case.execution_status == CaseExecutionStatus.UNASSESSED for case in result.case_results
    )


@pytest.mark.asyncio
async def test_bad_adapter_schema_stops_pending_cases() -> None:
    adapter = ScriptedAdapter([AdapterSchemaError(), _response("must not run")])
    result = await RunOrchestrator(adapter, _evaluator, max_concurrency=1).run(
        "run-schema",
        [
            OrchestrationCase("case-1", {"source": "one"}),
            OrchestrationCase("case-2", {"source": "two"}),
        ],
    )

    assert adapter.calls == 1
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.failure_code == "AI_INVALID_RESPONSE"
    assert result.case_results[1].attempts == 0


@pytest.mark.asyncio
async def test_partial_technical_failure_reduces_coverage_without_failing_completed_run() -> None:
    adapter = ScriptedAdapter([_response("good"), AdapterRequestError(400)])
    result = await RunOrchestrator(adapter, _evaluator, max_concurrency=1).run(
        "run-partial",
        [
            OrchestrationCase("good", {"source": "one"}),
            OrchestrationCase("bad", {"source": "two"}),
        ],
    )

    assert result.execution_status == RunExecutionStatus.COMPLETED
    assert result.coverage == RunCoverage.REDUCED
    assert result.cases_completed == 1
    assert result.cases_unassessed == 1
    assert result.release_decision is None


@pytest.mark.asyncio
async def test_evaluator_failure_keeps_paid_output_but_marks_case_unassessed() -> None:
    async def broken_evaluator(
        case: OrchestrationCase, response: AdapterResponse
    ) -> dict[str, object]:
        del case, response
        raise RuntimeError("Sensitive output must not enter error messages")

    result = await RunOrchestrator(
        ScriptedAdapter([_response("captured output", cost=0.75)]), broken_evaluator
    ).run("run-eval-fail", [OrchestrationCase("case", {"source": "input"})])

    case_result = result.case_results[0]
    assert case_result.execution_status == CaseExecutionStatus.UNASSESSED
    assert case_result.output == "captured output"
    assert case_result.cost_usd == pytest.approx(0.75)
    assert case_result.error_code == "SEMANTIC_EVALUATOR_FAILED"
    assert "Sensitive output" not in (case_result.error_message or "")
    assert result.execution_status == RunExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_evaluator_reported_unassessed_cannot_become_completed() -> None:
    async def unassessed_evaluator(
        case: OrchestrationCase, response: AdapterResponse
    ) -> dict[str, object]:
        del case, response
        return {"assessment_status": "UNASSESSED", "release_decision": None}

    result = await RunOrchestrator(
        ScriptedAdapter([_response("captured")]), unassessed_evaluator
    ).run("run-unassessed", [OrchestrationCase("case", {"source": "input"})])

    assert result.case_results[0].execution_status == CaseExecutionStatus.UNASSESSED
    assert result.case_results[0].error_code == "EVALUATOR_NOT_IMPLEMENTED"
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.release_decision is None


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [None, {}, {"status": "ok"}])
async def test_malformed_evaluator_result_fails_closed(malformed: object) -> None:
    async def malformed_evaluator(case: OrchestrationCase, response: AdapterResponse) -> object:
        del case, response
        return malformed

    result = await RunOrchestrator(
        ScriptedAdapter([_response("captured")]), malformed_evaluator
    ).run("run-malformed", [OrchestrationCase("case", {"source": "input"})])

    case = result.case_results[0]
    assert case.execution_status == CaseExecutionStatus.UNASSESSED
    assert case.assessment_status == AssessmentStatus.UNASSESSED
    assert case.error_code == "EVALUATOR_INVALID_RESULT"
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.coverage == RunCoverage.UNASSESSED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_evaluation",
    [
        {"assessment_status": "ASSESSED", "execution_status": "COMPLETED"},
        {
            "assessment_status": "ASSESSED",
            "execution_status": "COMPLETED",
            "coverage": {"status": "PARTIAL"},
        },
        {
            "assessment_status": "ASSESSED",
            "execution_status": "COMPLETED",
            "coverage": {"status": "REDUCED"},
        },
        {
            "assessment_status": "ASSESSED",
            "execution_status": "COMPLETED",
            "coverage": {"status": "UNKNOWN"},
        },
        {
            "assessment_status": "ASSESSED",
            "execution_status": "FAILED",
            "coverage": {"status": "FULL"},
        },
    ],
)
async def test_incomplete_or_contradictory_evaluator_coverage_fails_closed(
    invalid_evaluation: dict[str, object],
) -> None:
    async def evaluator(case: OrchestrationCase, response: AdapterResponse) -> dict[str, object]:
        del case, response
        return invalid_evaluation

    result = await RunOrchestrator(ScriptedAdapter([_response("captured")]), evaluator).run(
        "run-invalid-coverage", [OrchestrationCase("case", {"source": "input"})]
    )

    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.coverage == RunCoverage.UNASSESSED
    assert result.case_results[0].assessment_status == AssessmentStatus.UNASSESSED


@pytest.mark.asyncio
async def test_blocking_sync_evaluator_is_preempted_by_timeout() -> None:
    def blocking_evaluator(case: OrchestrationCase, response: AdapterResponse) -> dict[str, str]:
        del case, response
        time.sleep(0.05)
        return {"assessment_status": "ASSESSED"}

    result = await RunOrchestrator(
        ScriptedAdapter([_response("captured")]),
        blocking_evaluator,
        evaluator_timeout_seconds=0.001,
    ).run("run-blocking-evaluator", [OrchestrationCase("case", {"source": "input"})])

    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.case_results[0].error_code == "SEMANTIC_EVALUATOR_FAILED"


@pytest.mark.asyncio
async def test_durable_transition_failure_returns_failed_unassessed_run() -> None:
    async def unavailable_store(transition):  # type: ignore[no-untyped-def]
        del transition
        raise OSError("persistence unavailable")

    result = await RunOrchestrator(
        ScriptedAdapter([_response("must not be assessed")]),
        _evaluator,
        transition_callback=unavailable_store,
    ).run("run-store-fail", [OrchestrationCase("case", {"source": "input"})])

    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.coverage == RunCoverage.UNASSESSED
    assert result.failure_code == "WORKER_FAILED"
    assert result.case_results[0].execution_status == CaseExecutionStatus.UNASSESSED
    assert result.release_decision is None


@pytest.mark.asyncio
async def test_blocking_sync_transition_store_is_offloaded_and_times_out() -> None:
    def blocking_store(transition: object) -> None:
        del transition
        time.sleep(0.05)

    started = time.perf_counter()
    result = await RunOrchestrator(
        ScriptedAdapter([_response("must not be assessed")]),
        _evaluator,
        transition_callback=blocking_store,
        transition_timeout_seconds=0.001,
    ).run("run-store-timeout", [OrchestrationCase("case", {"source": "input"})])

    assert time.perf_counter() - started < 0.04
    assert result.execution_status == RunExecutionStatus.FAILED
    assert result.coverage == RunCoverage.UNASSESSED
    assert result.failure_code == "WORKER_FAILED"


@pytest.mark.asyncio
async def test_external_timeout_is_retried_then_unassessed() -> None:
    class SlowAdapter:
        name = "slow"
        version = "slow@1"
        calls = 0

        async def invoke(self, _payload: Mapping[str, Any]) -> AdapterResponse:
            self.calls += 1
            await asyncio.sleep(0.02)
            return _response("late")

    adapter = SlowAdapter()
    result = await RunOrchestrator(
        adapter,
        _evaluator,
        timeout_seconds=0.001,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0),
        jitter=lambda _delay: 0,
    ).run("run-timeout", [OrchestrationCase("case", {"source": "input"})])

    assert adapter.calls == 2
    assert result.case_results[0].error_code == "AI_TIMEOUT"
    assert result.case_results[0].attempts == 2
    assert result.execution_status == RunExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_background_dispatcher_depends_only_on_replaceable_queue_protocol() -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.enqueued: tuple[str, Mapping[str, Any], str | None] | None = None

        async def enqueue(
            self,
            job_name: str,
            payload: Mapping[str, Any],
            *,
            job_id: str | None = None,
        ) -> str:
            self.enqueued = (job_name, payload, job_id)
            return "backend-job-1"

    queue = FakeQueue()
    dispatcher = BackgroundRunDispatcher(queue)

    job_id = await dispatcher.submit("run-queue", {"test_plan_id": "plan-1"})

    assert job_id == "backend-job-1"
    assert queue.enqueued == (
        "gatekeeper.execute_test_run",
        {"test_plan_id": "plan-1"},
        "run-queue",
    )

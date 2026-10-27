from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.llm import (
    AnthropicTestSelectionPlanner,
    LLMPlannerError,
    preview_test_selection,
    sanitized_profile_context,
)
from backend.app.systems import (
    ApplicationEnvironment,
    ApplicationProfile,
    ApplicationType,
    InputType,
    OutputType,
    RiskTier,
)
from backend.app.test_selection import TestFamily as Family
from backend.app.test_selection import select_tests


def _profile() -> ApplicationProfile:
    return ApplicationProfile(
        application_id="system_1",
        organization_id="org_1",
        name="Synthetic clinical scribe",
        version="1.0.0",
        application_type=ApplicationType.CLINICAL_SCRIBE,
        intended_use="Create a draft clinical note for clinician review.",
        input_type=InputType.CLINICAL_TRANSCRIPT,
        output_type=OutputType.CLINICAL_NOTE,
        clinician_facing=True,
        risk_tier=RiskTier.HIGH,
        environment=ApplicationEnvironment.STAGING,
        endpoint_url="https://customer.example.test/inference",
    )


def test_profile_context_excludes_endpoint_details() -> None:
    context = sanitized_profile_context(_profile())

    assert "endpoint_url" not in context
    assert "authentication_type" not in context
    assert context["application_type"] == "CLINICAL_SCRIBE"


def test_anthropic_planner_returns_grounded_preview() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "suggested_tests": [
                                    "clinical_note_generation",
                                    "rag_grounding",
                                ],
                                "rationale": {
                                    "clinical_note_generation": "The application creates notes.",
                                    "rag_grounding": "The application uses retrieved context.",
                                },
                                "warnings": [],
                            }
                        ),
                    }
                ]
            },
        )

    profile = _profile()
    planner = AnthropicTestSelectionPlanner(
        api_key="anthropic-test-secret",
        model="claude-test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    preview = preview_test_selection(profile, select_tests(profile), planner)

    assert preview.accepted_suggestions == [Family.CLINICAL_NOTE_GENERATION]
    assert preview.rejected_suggestions == [Family.RAG_GROUNDING]
    assert preview.proposal.suggested_tests == [
        Family.CLINICAL_NOTE_GENERATION,
        Family.RAG_GROUNDING,
    ]
    prompt = json.dumps(seen["body"])
    assert "customer.example.test" not in prompt
    assert "anthropic-test-secret" not in prompt
    assert seen["headers"]["x-api-key"] == "anthropic-test-secret"


def test_anthropic_planner_rejects_unknown_test_family() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"suggested_tests":["invented"]}'}]},
        )

    planner = AnthropicTestSelectionPlanner(
        api_key="secret",
        model="model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(LLMPlannerError, match="planner request failed"):
        planner.propose(_profile())


def test_llm_planner_settings_require_secret_and_model_when_enabled() -> None:
    with pytest.raises(ValueError, match="LLM model"):
        Settings(_env_file=None, llm_planner_enabled=True)

    configured = Settings(
        _env_file=None,
        llm_planner_enabled=True,
        llm_model="claude-test-model",
        anthropic_api_key="secret",
    )
    assert configured.llm_planner_enabled is True
    assert configured.anthropic_api_key is not None
    assert configured.anthropic_api_key not in repr(configured)

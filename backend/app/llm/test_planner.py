"""Grounded Anthropic assistance for application test selection.

The planner is deliberately advisory.  It receives a sanitized application
profile and the closed Gatekeeper test catalog, then returns structured JSON.
Deterministic rules decide which families are applicable; the planner can only
explain or suggest among those families.  Endpoint URLs, credentials, patient
data, and test outputs never enter the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.systems import ApplicationProfile
from backend.app.test_selection import (
    ALL_TEST_FAMILIES,
    TestFamily,
    TestRequirement,
    TestSelectionResult,
)

ANTHROPIC_API_URL: Final = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION: Final = "2023-06-01"
PLANNER_PROMPT_VERSION: Final = "llm-test-selection-prompt@1.0.0"


class LLMPlannerError(RuntimeError):
    """The advisory planner could not produce a safe proposal."""


class LLMSelectionProposal(BaseModel):
    """Strict JSON contract returned by the external planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suggested_tests: list[TestFamily] = Field(default_factory=list, max_length=16)
    rationale: dict[TestFamily, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_rationale(self) -> LLMSelectionProposal:
        suggested = set(self.suggested_tests)
        if set(self.rationale) - suggested:
            raise ValueError("rationale may only describe suggested test families")
        return self


class LLMTestSelectionPreview(BaseModel):
    """Auditable preview combining deterministic selection and LLM advice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner_version: str
    planner_model: str
    prompt_version: str
    deterministic_selection: TestSelectionResult
    proposal: LLMSelectionProposal
    accepted_suggestions: list[TestFamily]
    rejected_suggestions: list[TestFamily]


def sanitized_profile_context(profile: ApplicationProfile) -> dict[str, Any]:
    """Return only profile facts needed for planning; exclude endpoint details."""

    return {
        "application_type": profile.application_type.value,
        "intended_use": profile.intended_use,
        "clinical_specialty": profile.clinical_specialty,
        "input_type": profile.input_type.value,
        "output_type": profile.output_type.value,
        "patient_facing": profile.patient_facing,
        "clinician_facing": profile.clinician_facing,
        "uses_rag": profile.uses_rag,
        "uses_tools": profile.uses_tools,
        "uses_llm": profile.uses_llm,
        "uses_predictive_model": profile.uses_predictive_model,
        "environment": profile.environment.value,
        "risk_tier": profile.risk_tier.value,
    }


def _catalog_context() -> list[str]:
    return [family.value for family in ALL_TEST_FAMILIES]


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMPlannerError("Anthropic returned no JSON planner proposal") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMPlannerError("Anthropic returned invalid planner JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMPlannerError("Anthropic planner output must be a JSON object")
    return parsed


class AnthropicTestSelectionPlanner:
    """Call Anthropic for a sanitized, schema-validated test-family proposal."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = client

    def propose(self, profile: ApplicationProfile) -> LLMSelectionProposal:
        prompt = {
            "application_profile": sanitized_profile_context(profile),
            "allowed_test_families": _catalog_context(),
            "deterministic_selection_authority": (
                "The deterministic engine is authoritative. Suggest only families from "
                "the allowed catalog; do not invent families."
            ),
        }
        payload = {
            "model": self.model,
            "max_tokens": 1200,
            "system": (
                "You are a healthcare AI assurance planning assistant. "
                "Return only JSON matching this shape: "
                '{"suggested_tests":["family"],"rationale":{"family":"reason"},'
                '"warnings":["warning"]}. '
                "Use only allowed test family values. Do not request endpoint URLs, "
                "credentials, patient data, or application outputs. "
                f"Planner contract: {PLANNER_PROMPT_VERSION}."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(prompt, sort_keys=True, separators=(",", ":")),
                }
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        try:
            response = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            content = body.get("content")
            if not isinstance(content, list):
                raise LLMPlannerError("Anthropic response did not contain message content")
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if not text_parts:
                raise LLMPlannerError("Anthropic response did not contain text content")
            return LLMSelectionProposal.model_validate(_extract_json("\n".join(text_parts)))
        except LLMPlannerError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise LLMPlannerError("Anthropic planner request failed") from exc
        finally:
            if owns_client:
                client.close()


def preview_test_selection(
    profile: ApplicationProfile,
    deterministic_selection: TestSelectionResult,
    planner: AnthropicTestSelectionPlanner,
) -> LLMTestSelectionPreview:
    """Combine deterministic applicability with a bounded LLM suggestion."""

    proposal = planner.propose(profile)
    applicable = {
        family
        for family in ALL_TEST_FAMILIES
        if deterministic_selection.requirement_for(family) is not TestRequirement.NOT_APPLICABLE
    }
    accepted = [family for family in proposal.suggested_tests if family in applicable]
    rejected = [family for family in proposal.suggested_tests if family not in applicable]
    return LLMTestSelectionPreview(
        planner_version="anthropic-test-selection@0.1.0",
        planner_model=planner.model,
        prompt_version=PLANNER_PROMPT_VERSION,
        deterministic_selection=deterministic_selection,
        proposal=proposal,
        accepted_suggestions=accepted,
        rejected_suggestions=rejected,
    )


__all__ = [
    "ANTHROPIC_API_URL",
    "ANTHROPIC_API_VERSION",
    "AnthropicTestSelectionPlanner",
    "LLMPlannerError",
    "LLMSelectionProposal",
    "LLMTestSelectionPreview",
    "preview_test_selection",
    "sanitized_profile_context",
]

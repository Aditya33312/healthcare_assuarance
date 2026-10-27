"""Adapters for optional external evaluation engines.

MedHELM and LangTest are benchmark/harness projects rather than drop-in
single-response evaluators.  Gatekeeper therefore talks to a small JSON
runner boundary.  A runner can be a local wrapper around either project, a
container command, or an in-process test double.  The boundary keeps the
third-party dependency optional and makes its output auditable.

The runner receives one JSON object on stdin and must emit one JSON object on
stdout with this shape::

    {"score": 0.92, "passed": true, "metrics": {}, "findings": []}

The input contains the family-specific Gatekeeper payload, including
``generated_output`` and any evaluator context.  Commands are configured by
the application, never by test-case data, and are executed without a shell.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExternalRunner(Protocol):
    """Run one external evaluation and return its JSON result."""

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Evaluate one Gatekeeper payload."""


class ExternalToolError(RuntimeError):
    """A configured external evaluator failed or returned an invalid result."""


class ExternalCoverage(BaseModel):
    """Coverage contract required by Gatekeeper's fail-closed orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "FULL"
    cases_assessed: int = Field(default=1, ge=0)
    cases_unassessed: int = Field(default=0, ge=0)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value.upper() != "FULL":
            raise ValueError("external evaluator must return FULL coverage")
        return "FULL"


class ExternalEvaluationResult(BaseModel):
    """Normalized result returned by a MedHELM/LangTest adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_status: str = "ASSESSED"
    execution_status: str = "COMPLETED"
    coverage: ExternalCoverage = Field(default_factory=ExternalCoverage)
    evaluator: str
    evaluator_version: str
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    passed: bool | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    findings: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("assessment_status")
    @classmethod
    def validate_assessment_status(cls, value: str) -> str:
        if value.upper() != "ASSESSED":
            raise ValueError("external evaluator did not assess the case")
        return "ASSESSED"

    @field_validator("execution_status")
    @classmethod
    def validate_execution_status(cls, value: str) -> str:
        if value.upper() != "COMPLETED":
            raise ValueError("external evaluator did not complete the case")
        return "COMPLETED"


@dataclass(frozen=True, slots=True)
class JsonCommandRunner:
    """Invoke a configured JSON-in/JSON-out evaluator wrapper.

    The command is intentionally a tuple and ``shell=False`` is used so that
    test data cannot become executable shell input.  The wrapper may import
    ``medhelm`` or ``langtest`` and translate their native reports into the
    small result contract documented above.
    """

    command: tuple[str, ...]
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.command or not all(item.strip() for item in self.command):
            raise ValueError("command must contain at least one non-empty argument")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise ExternalToolError("configured evaluator command was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolError("configured evaluator command timed out") from exc
        except OSError as exc:
            raise ExternalToolError("configured evaluator command could not start") from exc

        if completed.returncode != 0:
            raise ExternalToolError("configured evaluator command failed")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExternalToolError("configured evaluator returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            raise ExternalToolError("configured evaluator returned a non-object JSON value")
        return result


def runner_from_environment(
    environment_variable: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> JsonCommandRunner | None:
    """Build a runner from a command string, or return ``None`` when disabled."""

    values = environ if environ is not None else os.environ
    configured = values.get(environment_variable, "").strip()
    if not configured:
        return None
    return JsonCommandRunner(tuple(shlex.split(configured)))


class ExternalEvaluator:
    """Common Gatekeeper adapter for a configured external runner."""

    name: str
    version: str

    def __init__(self, runner: ExternalRunner) -> None:
        self._runner = runner

    def evaluate(self, payload: Mapping[str, Any]) -> ExternalEvaluationResult:
        try:
            raw = self._runner(payload)
        except ExternalToolError:
            raise
        except Exception as exc:
            raise ExternalToolError("external evaluator runner failed") from exc
        if not isinstance(raw, Mapping):
            raise ExternalToolError("external evaluator returned a non-object result")
        try:
            return ExternalEvaluationResult.model_validate(
                {
                    **raw,
                    "evaluator": self.name,
                    "evaluator_version": self.version,
                }
            )
        except Exception as exc:
            raise ExternalToolError("external evaluator returned an invalid result") from exc


class MedHELMEvaluator(ExternalEvaluator):
    """MedHELM adapter for medical task and clinical note evaluations."""

    name = "medhelm"
    version = "medhelm-adapter@0.1.0"


class LangTestEvaluator(ExternalEvaluator):
    """LangTest adapter for robustness, bias, and prompt-security evaluations."""

    name = "langtest"
    version = "langtest-adapter@0.1.0"


def external_evaluators_from_environment() -> tuple[ExternalEvaluator, ExternalEvaluator]:
    """Return configured MedHELM and LangTest adapters.

    This helper is useful for worker/bootstrap code.  The router itself also
    reads the same variables, so existing API call sites remain compatible.
    """

    medhelm = runner_from_environment("GATEKEEPER_MEDHELM_COMMAND")
    langtest = runner_from_environment("GATEKEEPER_LANGTEST_COMMAND")
    if medhelm is None or langtest is None:
        missing = []
        if medhelm is None:
            missing.append("GATEKEEPER_MEDHELM_COMMAND")
        if langtest is None:
            missing.append("GATEKEEPER_LANGTEST_COMMAND")
        missing_names = ", ".join(missing)
        raise ExternalToolError(
            f"external evaluator command is not configured: {missing_names}"
        )
    return MedHELMEvaluator(medhelm), LangTestEvaluator(langtest)


__all__ = [
    "ExternalCoverage",
    "ExternalEvaluationResult",
    "ExternalEvaluator",
    "ExternalRunner",
    "ExternalToolError",
    "JsonCommandRunner",
    "LangTestEvaluator",
    "MedHELMEvaluator",
    "external_evaluators_from_environment",
    "runner_from_environment",
]

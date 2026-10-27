"""API-independent single-evaluation request contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.ingestion.schemas import (
    EvidenceSource,
    GeneratedEvidenceOutput,
    add_default_evidence_types,
)


class SingleEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str = Field(min_length=1)
    system_version: str = Field(min_length=1)
    sources: list[EvidenceSource] = Field(min_length=1)
    generated_output: GeneratedEvidenceOutput

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_artifact_defaults(cls, value: Any) -> Any:
        return add_default_evidence_types(value)

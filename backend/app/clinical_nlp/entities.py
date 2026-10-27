"""Internal extraction entities kept separate from public domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.domain.enums import AssertionStatus, ClaimType, SubjectType


@dataclass(slots=True)
class ExtractedEntity:
    type: ClaimType
    text: str
    start: int
    end: int
    concept: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    origin: str = "deterministic"
    origin_version: str = "0.1.0"
    assertion: AssertionStatus = AssertionStatus.PRESENT
    subject: SubjectType = SubjectType.PATIENT
    parent_index: int | None = None

    def overlaps(self, other: ExtractedEntity) -> bool:
        return self.start < other.end and other.start < self.end

    def validate_against(self, source: str) -> bool:
        return (
            0 <= self.start < self.end <= len(source) and source[self.start : self.end] == self.text
        )

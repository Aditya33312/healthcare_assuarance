"""Cooperative safeguards for deterministic clinical extraction."""

from __future__ import annotations

import time


class ClinicalNLPLimitError(RuntimeError):
    """Raised before clinical extraction exceeds a configured work budget."""


def ensure_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise ClinicalNLPLimitError("clinical extraction duration limit exceeded")


__all__ = ["ClinicalNLPLimitError", "ensure_deadline"]

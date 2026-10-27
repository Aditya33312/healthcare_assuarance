"""Laterality normalization."""

from __future__ import annotations

import re

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION

_LATERALITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:left|lt\.?(?:-sided)?|left-sided)", re.IGNORECASE), "LEFT"),
    (re.compile(r"(?:right|rt\.?(?:-sided)?|right-sided)", re.IGNORECASE), "RIGHT"),
    (
        re.compile(r"(?:bilateral(?:ly)?|both(?:\s+sides?)?|b/l|bi-lateral)", re.IGNORECASE),
        "BILATERAL",
    ),
)


def normalize_laterality(value: str) -> str | None:
    if not isinstance(value, str):
        raise TypeError("laterality must be a string")
    candidate = re.sub(r"\s+", " ", value.strip())
    for pattern, normalized in _LATERALITY_PATTERNS:
        if pattern.fullmatch(candidate):
            return normalized
    return None


class LateralityNormalizer:
    version = NORMALIZATION_VERSION

    def normalize(self, value: str) -> str | None:
        return normalize_laterality(value)


LATERALITY_EXTRACTION_PATTERN = re.compile(
    r"(?<!\w)(?:left(?:-sided)?|right(?:-sided)?|lt\.?(?:-sided)?|"
    r"rt\.?(?:-sided)?|bilateral(?:ly)?|both\s+sides?|b/l)(?!\w)",
    re.IGNORECASE,
)

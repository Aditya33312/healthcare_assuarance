"""Transparent terminology aliases used before semantic fallback."""

from __future__ import annotations

import re

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION

_ALIASES = {
    "sob": "dyspnea",
    "shortness of breath": "dyspnea",
    "dyspnoea": "dyspnea",
    "high blood pressure": "hypertension",
    "htn": "hypertension",
    "heart attack": "myocardial infarction",
    "mi": "myocardial infarction",
    "t2dm": "type 2 diabetes mellitus",
    "type ii diabetes": "type 2 diabetes mellitus",
    "type 2 diabetes": "type 2 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "uti": "urinary tract infection",
}


def normalize_terminology(value: str) -> str | None:
    if not isinstance(value, str):
        raise TypeError("term must be a string")
    candidate = re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".,")
    if not candidate:
        return None
    return _ALIASES.get(candidate, candidate)


class TerminologyNormalizer:
    version = NORMALIZATION_VERSION

    def normalize(self, value: str) -> str | None:
        return normalize_terminology(value)

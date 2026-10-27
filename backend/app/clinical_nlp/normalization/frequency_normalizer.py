"""Medication frequency normalization."""

from __future__ import annotations

import re

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION

_FREQUENCY_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), value)
    for pattern, value in (
        (r"(?:qd|q\.d\.?|od|once(?:\s+(?:a|per))?\s+day|once\s+daily|daily)", "ONCE_DAILY"),
        (
            r"(?:bid|b\.i\.d\.?|bd|twice(?:\s+(?:a|per))?\s+day|twice\s+daily|"
            r"2\s*[x\u00d7]\s*/?\s*(?:day|daily))",
            "TWICE_DAILY",
        ),
        (
            r"(?:tid|t\.i\.d\.?|tds|three\s+times(?:\s+(?:a|per))?\s+day|"
            r"three\s+times\s+daily|"
            r"3\s*[x\u00d7]\s*/?\s*(?:day|daily))",
            "THREE_TIMES_DAILY",
        ),
        (
            r"(?:qid|q\.i\.d\.?|four\s+times(?:\s+(?:a|per))?\s+day|"
            r"four\s+times\s+daily|"
            r"4\s*[x\u00d7]\s*/?\s*(?:day|daily))",
            "FOUR_TIMES_DAILY",
        ),
        (r"(?:qhs|at\s+(?:bedtime|night)|nightly|each\s+night)", "AT_BEDTIME"),
        (r"(?:qam|each\s+morning|every\s+morning|in\s+the\s+morning)", "EVERY_MORNING"),
        (r"(?:prn|as\s+needed|when\s+required)", "AS_NEEDED"),
        (r"(?:stat|immediately|now)", "ONCE_IMMEDIATELY"),
        (r"(?:qod|every\s+other\s+day|alternate\s+days?)", "EVERY_OTHER_DAY"),
        (r"(?:weekly|once(?:\s+(?:a|per))?\s+week)", "ONCE_WEEKLY"),
        (r"(?:monthly|once(?:\s+(?:a|per))?\s+month)", "ONCE_MONTHLY"),
    )
)

_INTERVAL_RE = re.compile(
    r"(?:q\s*(?P<qnum>\d{1,2})\s*h(?:ours?)?|every\s+(?P<num>\d{1,2})\s*"
    r"(?P<unit>hours?|hrs?|minutes?|mins?|days?|weeks?))",
    re.IGNORECASE,
)


def normalize_frequency(value: str) -> str | None:
    """Normalize a complete frequency phrase to a stable enum-like value."""

    if not isinstance(value, str):
        raise TypeError("frequency must be a string")
    candidate = re.sub(r"\s+", " ", value.strip())
    if not candidate:
        return None

    for pattern, normalized in _FREQUENCY_RULES:
        if pattern.fullmatch(candidate):
            return normalized

    interval = _INTERVAL_RE.fullmatch(candidate)
    if interval:
        number = int(interval.group("qnum") or interval.group("num"))
        unit = (interval.group("unit") or "hours").casefold()
        if unit.startswith(("hour", "hr")):
            suffix = "HOURS"
        elif unit.startswith(("minute", "min")):
            suffix = "MINUTES"
        elif unit.startswith("day"):
            suffix = "DAYS"
        else:
            suffix = "WEEKS"
        return f"EVERY_{number}_{suffix}"
    return None


class FrequencyNormalizer:
    version = NORMALIZATION_VERSION

    def normalize(self, value: str) -> str | None:
        return normalize_frequency(value)


# Reused by the deterministic parser. Each expression identifies an entire source phrase.
FREQUENCY_EXTRACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    *(
        re.compile(rf"(?<!\w)(?:{rule.pattern})(?!\w)", re.IGNORECASE)
        for rule, _normalized in _FREQUENCY_RULES
    ),
    re.compile(rf"(?<!\w)(?:{_INTERVAL_RE.pattern})(?!\w)", re.IGNORECASE),
)

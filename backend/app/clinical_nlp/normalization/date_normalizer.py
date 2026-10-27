"""Unambiguous absolute and reference-relative date normalization."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_NAMED_DATE_RE = re.compile(
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(?P<year>\d{4})"
)
_DAY_NAMED_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})"
)


def _reference_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    raise TypeError("reference_date must be a date, datetime, ISO date string, or None")


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(
    value: str,
    *,
    reference_date: date | datetime | str | None = None,
    day_first: bool = False,
) -> str | None:
    if not isinstance(value, str):
        raise TypeError("date value must be a string")
    candidate = re.sub(r"\s+", " ", value.strip())
    if not candidate:
        return None

    relative = candidate.casefold()
    if relative in {"today", "yesterday", "tomorrow"}:
        reference = _reference_date(reference_date)
        if reference is None:
            return relative.upper()
        delta = {"today": 0, "yesterday": -1, "tomorrow": 1}[relative]
        return (reference + timedelta(days=delta)).isoformat()

    iso_match = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", candidate)
    if iso_match:
        return _safe_date(*(int(part) for part in iso_match.groups()))

    numeric_match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", candidate)
    if numeric_match:
        first, second, year = (int(part) for part in numeric_match.groups())
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            day, month = second, first
        else:
            day, month = (first, second) if day_first else (second, first)
        return _safe_date(year, month, day)

    for pattern in (_NAMED_DATE_RE, _DAY_NAMED_DATE_RE):
        named_match = pattern.fullmatch(candidate)
        if not named_match:
            continue
        named_month = _MONTHS.get(named_match.group("month").casefold().rstrip("."))
        if named_month is None:
            return None
        return _safe_date(
            int(named_match.group("year")), named_month, int(named_match.group("day"))
        )
    return None


class DateNormalizer:
    version = NORMALIZATION_VERSION

    def __init__(
        self,
        *,
        reference_date: date | datetime | str | None = None,
        day_first: bool = False,
    ) -> None:
        self.reference_date = reference_date
        self.day_first = day_first

    def normalize(self, value: str) -> str | None:
        return normalize_date(value, reference_date=self.reference_date, day_first=self.day_first)

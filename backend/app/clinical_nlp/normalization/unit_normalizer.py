"""Clinical unit and medication quantity normalization."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION


class NormalizedQuantity(NamedTuple):
    value: int | Decimal
    unit: str


_UNIT_ALIASES = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "mcg": "mcg",
    "ug": "mcg",
    "microgram": "mcg",
    "micrograms": "mcg",
    "ng": "ng",
    "nanogram": "ng",
    "nanograms": "ng",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ml": "mL",
    "milliliter": "mL",
    "milliliters": "mL",
    "millilitre": "mL",
    "millilitres": "mL",
    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "iu": "IU",
    "international unit": "IU",
    "international units": "IU",
    "unit": "units",
    "units": "units",
    "%": "%",
    "mmol/l": "mmol/L",
    "mol/l": "mol/L",
    "mg/dl": "mg/dL",
    "g/dl": "g/dL",
    "mmhg": "mmHg",
    "bpm": "bpm",
    "°c": "°C",
    "c": "°C",
    "°f": "°F",
    "f": "°F",
}


def normalize_unit(unit: str) -> str | None:
    if not isinstance(unit, str):
        raise TypeError("unit must be a string")
    candidate = unit.strip().casefold().replace("μ", "u").replace("µ", "u")
    candidate = re.sub(r"\s+", " ", candidate).rstrip(".")
    return _UNIT_ALIASES.get(candidate)


def _compact_number(value: Decimal) -> int | Decimal:
    """Keep non-integral quantities exact instead of routing through ``float``."""

    integral = value.to_integral_value()
    if value == integral:
        return int(integral)
    return value


def normalize_quantity(value: int | float | Decimal | str, unit: str) -> NormalizedQuantity | None:
    """Normalize compatible mass values to mg and volume values to mL."""

    canonical_unit = normalize_unit(unit)
    if canonical_unit is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None

    if canonical_unit == "g":
        number *= Decimal(1000)
        canonical_unit = "mg"
    elif canonical_unit == "mcg":
        number /= Decimal(1000)
        canonical_unit = "mg"
    elif canonical_unit == "ng":
        number /= Decimal(1_000_000)
        canonical_unit = "mg"
    elif canonical_unit == "L":
        number *= Decimal(1000)
        canonical_unit = "mL"

    return NormalizedQuantity(_compact_number(number), canonical_unit)


class UnitNormalizer:
    version = NORMALIZATION_VERSION

    def normalize(self, unit: str) -> str | None:
        return normalize_unit(unit)

    def normalize_quantity(
        self, value: int | float | Decimal | str, unit: str
    ) -> NormalizedQuantity | None:
        return normalize_quantity(value, unit)

"""Versioned deterministic clinical comparison rules.

Rules compare only already-aligned claims.  They deliberately do not perform
alignment: a clinically important mismatch must never be manufactured by
guessing that two unrelated claims refer to the same fact.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from backend.app.domain.enums import (
    AssertionStatus,
    ClaimType,
    FindingCategory,
    FindingType,
    Severity,
)
from backend.app.domain.models import Claim

COMPARE_RULE_SET_VERSION = "0.2.0"


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """Stable rule metadata persisted with every deterministic finding."""

    rule_id: str
    version: str
    description: str
    category: FindingCategory
    default_severity: Severity


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """A deterministic difference found for a pair of aligned claims."""

    rule: RuleDefinition
    finding_type: FindingType
    source_value: object | None
    generated_value: object | None
    explanation: str
    confidence: float = 0.99
    category: FindingCategory | None = None


LATERALITY_RULE = RuleDefinition(
    "LATERALITY_MISMATCH",
    "1.0.0",
    "Laterality must be preserved for the same clinical fact.",
    FindingCategory.LATERALITY,
    Severity.HIGH,
)
MEDICATION_RULE = RuleDefinition(
    "MEDICATION_MISMATCH",
    "1.0.0",
    "Medication identity must be preserved within an aligned medication slot.",
    FindingCategory.MEDICATION_IDENTITY,
    Severity.HIGH,
)
DOSE_RULE = RuleDefinition(
    "DOSE_MISMATCH",
    "1.0.0",
    "Medication dose must be numerically equivalent after unit conversion.",
    FindingCategory.MEDICATION_DOSE,
    Severity.HIGH,
)
UNIT_RULE = RuleDefinition(
    "UNIT_MISMATCH",
    "1.0.0",
    "Medication or measured-value unit must be equivalent.",
    FindingCategory.MEDICATION_UNIT,
    Severity.HIGH,
)
FREQUENCY_RULE = RuleDefinition(
    "FREQUENCY_MISMATCH",
    "1.0.0",
    "Medication frequency must be preserved after normalization.",
    FindingCategory.MEDICATION_FREQUENCY,
    Severity.HIGH,
)
ROUTE_RULE = RuleDefinition(
    "ROUTE_MISMATCH",
    "1.0.0",
    "Medication route must be preserved after normalization.",
    FindingCategory.MEDICATION_ROUTE,
    Severity.HIGH,
)
NEGATION_RULE = RuleDefinition(
    "NEGATION_REVERSAL",
    "1.0.0",
    "Present and absent assertions must not be reversed.",
    FindingCategory.ASSERTION,
    Severity.HIGH,
)
ALLERGY_CONTRADICTION_RULE = RuleDefinition(
    "ALLERGY_CONTRADICTION",
    "1.0.0",
    "A specific allergy and an explicit no-known-allergy statement contradict.",
    FindingCategory.ALLERGY,
    Severity.CRITICAL,
)
CERTAINTY_RULE = RuleDefinition(
    "CERTAINTY_CHANGE",
    "1.0.0",
    "Clinical assertion certainty and temporality must be preserved.",
    FindingCategory.CERTAINTY,
    Severity.MODERATE,
)
SUBJECT_RULE = RuleDefinition(
    "SUBJECT_ATTRIBUTION_ERROR",
    "1.0.0",
    "A fact must remain attributed to the same subject.",
    FindingCategory.SUBJECT_ATTRIBUTION,
    Severity.HIGH,
)
NUMERIC_RULE = RuleDefinition(
    "NUMERIC_VALUE_MISMATCH",
    "1.0.0",
    "Clinical numeric values must be preserved.",
    FindingCategory.LAB_VALUE,
    Severity.HIGH,
)
LAB_UNIT_RULE = RuleDefinition(
    "LAB_UNIT_MISMATCH",
    "1.0.0",
    "Clinical laboratory units must be compatible before numeric values are compared.",
    FindingCategory.LAB_VALUE,
    Severity.HIGH,
)
DATE_RULE = RuleDefinition(
    "DATE_MISMATCH",
    "1.0.0",
    "Clinical dates must be preserved after deterministic normalization.",
    FindingCategory.DATE,
    Severity.MODERATE,
)
DURATION_RULE = RuleDefinition(
    "DURATION_MISMATCH",
    "1.0.0",
    "Clinical durations must be preserved after deterministic normalization.",
    FindingCategory.DURATION,
    Severity.MODERATE,
)

RULES: tuple[RuleDefinition, ...] = (
    LATERALITY_RULE,
    MEDICATION_RULE,
    DOSE_RULE,
    UNIT_RULE,
    FREQUENCY_RULE,
    ROUTE_RULE,
    ALLERGY_CONTRADICTION_RULE,
    NEGATION_RULE,
    CERTAINTY_RULE,
    SUBJECT_RULE,
    NUMERIC_RULE,
    LAB_UNIT_RULE,
    DATE_RULE,
    DURATION_RULE,
)

_ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "laterality": ("laterality", "side", "normalized_laterality"),
    "dose": ("dose", "dose_value", "normalized_dose"),
    "unit": ("unit", "dose_unit", "normalized_unit"),
    "frequency": ("frequency", "normalized_frequency"),
    "route": ("route", "normalized_route"),
    "numeric": ("normalized_value", "numeric_value", "lab_value", "value"),
    "date": ("normalized_date", "date"),
    "duration": ("normalized_duration", "duration"),
}

_TYPE_ATTRIBUTE_FALLBACK: dict[str, set[ClaimType]] = {
    "laterality": {ClaimType.LATERALITY},
    "dose": {ClaimType.DOSE},
    "unit": {ClaimType.UNIT},
    "frequency": {ClaimType.FREQUENCY},
    "route": {ClaimType.ROUTE},
    "numeric": {ClaimType.LAB_VALUE, ClaimType.AGE},
    "date": {ClaimType.DATE},
    "duration": {ClaimType.DURATION},
}

_LATERALITY = {
    "l": "LEFT",
    "lt": "LEFT",
    "left": "LEFT",
    "leftsided": "LEFT",
    "r": "RIGHT",
    "rt": "RIGHT",
    "right": "RIGHT",
    "rightsided": "RIGHT",
    "bilateral": "BILATERAL",
    "both": "BILATERAL",
}

_FREQUENCY = {
    "qd": "ONCE_DAILY",
    "od": "ONCE_DAILY",
    "daily": "ONCE_DAILY",
    "oncedaily": "ONCE_DAILY",
    "1xday": "ONCE_DAILY",
    "bid": "TWICE_DAILY",
    "bd": "TWICE_DAILY",
    "twicedaily": "TWICE_DAILY",
    "2xday": "TWICE_DAILY",
    "tid": "THREE_TIMES_DAILY",
    "tds": "THREE_TIMES_DAILY",
    "threetimesdaily": "THREE_TIMES_DAILY",
    "3xday": "THREE_TIMES_DAILY",
    "qid": "FOUR_TIMES_DAILY",
    "qds": "FOUR_TIMES_DAILY",
    "fourtimesdaily": "FOUR_TIMES_DAILY",
    "4xday": "FOUR_TIMES_DAILY",
    "prn": "AS_NEEDED",
    "asneeded": "AS_NEEDED",
}

_ROUTE = {
    "po": "ORAL",
    "oral": "ORAL",
    "bymouth": "ORAL",
    "iv": "INTRAVENOUS",
    "intravenous": "INTRAVENOUS",
    "im": "INTRAMUSCULAR",
    "intramuscular": "INTRAMUSCULAR",
    "sc": "SUBCUTANEOUS",
    "sq": "SUBCUTANEOUS",
    "subcutaneous": "SUBCUTANEOUS",
    "topical": "TOPICAL",
    "inhaled": "INHALED",
    "inhalation": "INHALED",
}

_UNITS: dict[str, tuple[str, Decimal]] = {
    "mcg": ("mass_mg", Decimal("0.001")),
    "ug": ("mass_mg", Decimal("0.001")),
    "microgram": ("mass_mg", Decimal("0.001")),
    "micrograms": ("mass_mg", Decimal("0.001")),
    "mg": ("mass_mg", Decimal("1")),
    "milligram": ("mass_mg", Decimal("1")),
    "milligrams": ("mass_mg", Decimal("1")),
    "g": ("mass_mg", Decimal("1000")),
    "gram": ("mass_mg", Decimal("1000")),
    "grams": ("mass_mg", Decimal("1000")),
    "ml": ("volume_ml", Decimal("1")),
    "milliliter": ("volume_ml", Decimal("1")),
    "milliliters": ("volume_ml", Decimal("1")),
    "millilitre": ("volume_ml", Decimal("1")),
    "millilitres": ("volume_ml", Decimal("1")),
    "l": ("volume_ml", Decimal("1000")),
    "liter": ("volume_ml", Decimal("1000")),
    "litre": ("volume_ml", Decimal("1000")),
    "units": ("units", Decimal("1")),
    "unit": ("units", Decimal("1")),
    "iu": ("units", Decimal("1")),
    "%": ("percent", Decimal("1")),
    "mgdl": ("mass_concentration_mg_dl", Decimal("1")),
    "gdl": ("mass_concentration_mg_dl", Decimal("1000")),
    "mmoll": ("molar_concentration_mmol_l", Decimal("1")),
    "moll": ("molar_concentration_mmol_l", Decimal("1000")),
    "mmhg": ("pressure_mmhg", Decimal("1")),
    "bpm": ("rate_bpm", Decimal("1")),
    "c": ("temperature_celsius", Decimal("1")),
    "f": ("temperature_fahrenheit", Decimal("1")),
}

_DURATION_FACTORS = {
    "hour": Decimal("1"),
    "hours": Decimal("1"),
    "hr": Decimal("1"),
    "hrs": Decimal("1"),
    "day": Decimal("24"),
    "days": Decimal("24"),
    "week": Decimal("168"),
    "weeks": Decimal("168"),
    "month": Decimal("730"),
    "months": Decimal("730"),
    "year": Decimal("8760"),
    "years": Decimal("8760"),
}


def _attribute(claim: Claim, name: str) -> tuple[bool, Any]:
    for key in _ATTRIBUTE_ALIASES[name]:
        if key in claim.attributes and claim.attributes[key] is not None:
            return True, claim.attributes[key]
    if claim.type in _TYPE_ATTRIBUTE_FALLBACK[name]:
        return True, claim.concept
    return False, None


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9%]+", "", str(value).casefold())


def _canonical_laterality(value: object) -> object:
    compact = _compact(value)
    return _LATERALITY.get(compact, compact.upper())


def _canonical_frequency(value: object) -> object:
    compact = _compact(value)
    return _FREQUENCY.get(compact, compact.upper())


def _canonical_route(value: object) -> object:
    compact = _compact(value)
    return _ROUTE.get(compact, compact.upper())


_DECIMAL_TOKEN = r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
_PAIR_VALUE_RE = re.compile(
    rf"(?P<first>{_DECIMAL_TOKEN})\s*/\s*(?P<second>{_DECIMAL_TOKEN})"
)
_RANGE_VALUE_RE = re.compile(
    rf"(?P<first>{_DECIMAL_TOKEN})\s*(?:\.\.|[-\u2013\u2014]|to)\s*"
    rf"(?P<second>{_DECIMAL_TOKEN})",
    re.IGNORECASE,
)
_PERCENT_VALUE_RE = re.compile(rf"(?P<value>{_DECIMAL_TOKEN})\s*%")
_SCALAR_VALUE_RE = re.compile(_DECIMAL_TOKEN)

type NumericSignature = tuple[str, tuple[Decimal, ...]]


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            parsed_float = Decimal(str(value))
        except InvalidOperation:
            return None
        return parsed_float if parsed_float.is_finite() else None
    if isinstance(value, dict):
        for key in ("exact_value", "normalized_value", "value", "amount", "dose"):
            if key in value:
                return _decimal(value[key])
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _decimal(value[0])
    candidate = str(value).strip().replace("\u2212", "-").replace(",", "")
    if _SCALAR_VALUE_RE.fullmatch(candidate) is None:
        return None
    try:
        parsed = Decimal(candidate)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _numeric_signature(value: object) -> NumericSignature | None:
    """Parse a complete scalar, percentage, range, or paired value exactly."""

    if isinstance(value, dict):
        for key in ("exact_value", "normalized_value", "numeric_value", "value"):
            if key in value:
                return _numeric_signature(value[key])
        return None
    if isinstance(value, (list, tuple)):
        components = tuple(_decimal(item) for item in value)
        if not components or any(item is None for item in components):
            return None
        return "SEQUENCE", tuple(item for item in components if item is not None)

    scalar = _decimal(value)
    if scalar is not None:
        return "SCALAR", (scalar,)
    if not isinstance(value, str):
        return None

    candidate = value.strip().replace("\u2212", "-").replace(",", "")
    pair = _PAIR_VALUE_RE.fullmatch(candidate)
    if pair is not None:
        return "PAIR", (Decimal(pair.group("first")), Decimal(pair.group("second")))
    numeric_range = _RANGE_VALUE_RE.fullmatch(candidate)
    if numeric_range is not None:
        return "RANGE", (
            Decimal(numeric_range.group("first")),
            Decimal(numeric_range.group("second")),
        )
    percentage = _PERCENT_VALUE_RE.fullmatch(candidate)
    if percentage is not None:
        return "PERCENT", (Decimal(percentage.group("value")),)
    return None


def _numeric_values_equal(source: object, generated: object) -> bool:
    source_signature = _numeric_signature(source)
    generated_signature = _numeric_signature(generated)
    if source_signature is None or generated_signature is None:
        return str(source).strip().casefold() == str(generated).strip().casefold()
    return source_signature == generated_signature


def _canonical_numeric(value: object) -> object:
    signature = _numeric_signature(value)
    return signature if signature is not None else str(value).strip().casefold()


def _canonical_unit(value: object) -> str:
    compact = _compact(value)
    if compact in _UNITS:
        family, factor = _UNITS[compact]
        return f"{family}@{factor}"
    return compact.upper()


def _dose_in_base_unit(dose: object, unit: object) -> tuple[str, Decimal] | None:
    number = _decimal(dose)
    unit_key = _compact(unit)
    if number is None or unit_key not in _UNITS:
        return None
    family, factor = _UNITS[unit_key]
    return family, number * factor


def _canonical_date(value: object) -> object:
    raw = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw, pattern).date().isoformat()
        except ValueError:
            continue
    return raw.casefold()


def _canonical_duration(value: object) -> object:
    if isinstance(value, dict):
        number = _decimal(value)
        unit = value.get("unit") or value.get("duration_unit")
        if number is not None and unit is not None:
            factor = _DURATION_FACTORS.get(str(unit).casefold().strip())
            if factor is not None:
                return number * factor
    raw = str(value).casefold().strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)", raw)
    if match:
        factor = _DURATION_FACTORS.get(match.group(2))
        if factor is not None:
            return Decimal(match.group(1)) * factor
    return raw


def _missing_outcome(
    rule: RuleDefinition,
    source_present: bool,
    source_value: object | None,
    generated_present: bool,
    generated_value: object | None,
) -> RuleOutcome | None:
    if source_present and not generated_present:
        return RuleOutcome(
            rule,
            FindingType.OMITTED,
            source_value,
            None,
            f"The source {rule.category} value {source_value!r} is absent from the output.",
        )
    if generated_present and not source_present:
        return RuleOutcome(
            rule,
            FindingType.ADDED,
            None,
            generated_value,
            f"The output adds {rule.category} value {generated_value!r} with no source value.",
        )
    return None


def _attribute_outcome(
    rule: RuleDefinition,
    source: Claim,
    generated: Claim,
    attribute_name: str,
    canonicalizer: Callable[[object], object],
) -> RuleOutcome | None:
    source_present, source_value = _attribute(source, attribute_name)
    generated_present, generated_value = _attribute(generated, attribute_name)
    missing = _missing_outcome(
        rule,
        source_present,
        source_value,
        generated_present,
        generated_value,
    )
    if missing is not None:
        return missing
    if not source_present or not generated_present:
        return None
    if canonicalizer(source_value) == canonicalizer(generated_value):
        return None
    return RuleOutcome(
        rule,
        FindingType.ALTERED,
        source_value,
        generated_value,
        f"{rule.description} Source={source_value!r}; output={generated_value!r}.",
    )


def _dose_outcome(source: Claim, generated: Claim) -> RuleOutcome | None:
    source_present, source_dose = _attribute(source, "dose")
    generated_present, generated_dose = _attribute(generated, "dose")
    missing = _missing_outcome(
        DOSE_RULE,
        source_present,
        source_dose,
        generated_present,
        generated_dose,
    )
    if missing is not None:
        return missing
    if not source_present or not generated_present:
        return None

    source_unit_present, source_unit = _attribute(source, "unit")
    generated_unit_present, generated_unit = _attribute(generated, "unit")
    source_base = _dose_in_base_unit(source_dose, source_unit) if source_unit_present else None
    generated_base = (
        _dose_in_base_unit(generated_dose, generated_unit) if generated_unit_present else None
    )
    if source_base is not None and generated_base is not None:
        equal = source_base == generated_base
    else:
        equal = _decimal(source_dose) == _decimal(generated_dose)
    if equal:
        return None
    return RuleOutcome(
        DOSE_RULE,
        FindingType.ALTERED,
        source_dose,
        generated_dose,
        f"Medication dose changed from {source_dose!r} to {generated_dose!r}.",
    )


def _unit_outcome(source: Claim, generated: Claim) -> RuleOutcome | None:
    source_dose_present, source_dose = _attribute(source, "dose")
    generated_dose_present, generated_dose = _attribute(generated, "dose")
    source_present, source_unit = _attribute(source, "unit")
    generated_present, generated_unit = _attribute(generated, "unit")

    # If a whole dose is absent, DOSE_MISMATCH already records the omission;
    # a second unit omission would be duplicate evidence.
    if source_dose_present != generated_dose_present:
        return None
    missing = _missing_outcome(
        UNIT_RULE,
        source_present,
        source_unit,
        generated_present,
        generated_unit,
    )
    if missing is not None:
        return missing
    if not source_present or not generated_present:
        return None

    source_raw_unit = source.attributes.get("raw_unit", source_unit)
    generated_raw_unit = generated.attributes.get("raw_unit", generated_unit)
    source_raw_dose = source.attributes.get(
        "raw_dose",
        source.attributes.get("raw_value", source_dose),
    )
    generated_raw_dose = generated.attributes.get(
        "raw_dose",
        generated.attributes.get("raw_value", generated_dose),
    )
    if source_dose_present and generated_dose_present:
        source_base = _dose_in_base_unit(source_raw_dose, source_raw_unit)
        generated_base = _dose_in_base_unit(generated_raw_dose, generated_raw_unit)
        if source_base is not None and source_base == generated_base:
            return None
    if _canonical_unit(source_raw_unit) == _canonical_unit(generated_raw_unit):
        return None
    return RuleOutcome(
        UNIT_RULE,
        FindingType.ALTERED,
        source_raw_unit,
        generated_raw_unit,
        f"Medication unit changed from {source_raw_unit!r} to {generated_raw_unit!r}.",
    )


def _lab_signature_in_base(
    signature: NumericSignature,
    unit: object,
) -> tuple[str, NumericSignature] | None:
    unit_key = _compact(unit)
    definition = _UNITS.get(unit_key)
    if definition is None:
        return None
    family, factor = definition
    kind, components = signature
    if unit_key == "f":
        converted = tuple(
            (component - Decimal("32")) * Decimal("5") / Decimal("9")
            for component in components
        )
        return "temperature_celsius", (kind, converted)
    if unit_key == "c":
        return "temperature_celsius", signature
    return family, (kind, tuple(component * factor for component in components))


def _lab_outcomes(source: Claim, generated: Claim) -> list[RuleOutcome]:
    source_present, source_value = _attribute(source, "numeric")
    generated_present, generated_value = _attribute(generated, "numeric")
    missing_value = _missing_outcome(
        NUMERIC_RULE,
        source_present,
        source_value,
        generated_present,
        generated_value,
    )
    if missing_value is not None:
        return [missing_value]
    if not source_present or not generated_present:
        return []

    source_unit_present, source_unit = _attribute(source, "unit")
    generated_unit_present, generated_unit = _attribute(generated, "unit")
    missing_unit = _missing_outcome(
        LAB_UNIT_RULE,
        source_unit_present,
        source_unit,
        generated_unit_present,
        generated_unit,
    )
    if missing_unit is not None:
        return [missing_unit]

    source_signature = _numeric_signature(source_value)
    generated_signature = _numeric_signature(generated_value)
    if source_unit_present and generated_unit_present:
        source_base = (
            _lab_signature_in_base(source_signature, source_unit)
            if source_signature is not None
            else None
        )
        generated_base = (
            _lab_signature_in_base(generated_signature, generated_unit)
            if generated_signature is not None
            else None
        )
        if _compact(source_unit) != _compact(generated_unit):
            if (
                source_base is None
                or generated_base is None
                or source_base[0] != generated_base[0]
            ):
                return [
                    RuleOutcome(
                        LAB_UNIT_RULE,
                        FindingType.ALTERED,
                        source_unit,
                        generated_unit,
                        "Laboratory units are incompatible: "
                        f"source={source_unit!r}; output={generated_unit!r}.",
                    )
                ]
            source_signature = source_base[1]
            generated_signature = generated_base[1]

    equal = (
        source_signature == generated_signature
        if source_signature is not None and generated_signature is not None
        else _numeric_values_equal(source_value, generated_value)
    )
    if equal:
        return []
    return [
        RuleOutcome(
            NUMERIC_RULE,
            FindingType.ALTERED,
            source_value,
            generated_value,
            f"Clinical numeric values differ. Source={source_value!r}; "
            f"output={generated_value!r}.",
            category=FindingCategory.LAB_VALUE,
        )
    ]


def compare_aligned_claims(source: Claim, generated: Claim) -> list[RuleOutcome]:
    """Apply every applicable deterministic rule in stable rule order."""

    outcomes: list[RuleOutcome] = []

    laterality = _attribute_outcome(
        LATERALITY_RULE,
        source,
        generated,
        "laterality",
        _canonical_laterality,
    )
    if laterality is not None:
        outcomes.append(laterality)

    if source.type is ClaimType.MEDICATION and generated.type is ClaimType.MEDICATION:
        source_concept = source.concept.strip().casefold()
        generated_concept = generated.concept.strip().casefold()
        if source_concept != generated_concept:
            outcomes.append(
                RuleOutcome(
                    MEDICATION_RULE,
                    FindingType.ALTERED,
                    source.concept,
                    generated.concept,
                    "Medication identity differs in an explicitly aligned medication slot: "
                    f"{source.concept!r} versus {generated.concept!r}.",
                )
            )

    if source.type in {ClaimType.MEDICATION, ClaimType.DOSE} or generated.type in {
        ClaimType.MEDICATION,
        ClaimType.DOSE,
    }:
        dose = _dose_outcome(source, generated)
        if dose is not None:
            outcomes.append(dose)

    if source.type in {ClaimType.MEDICATION, ClaimType.DOSE, ClaimType.UNIT} or generated.type in {
        ClaimType.MEDICATION,
        ClaimType.DOSE,
        ClaimType.UNIT,
    }:
        unit = _unit_outcome(source, generated)
        if unit is not None:
            outcomes.append(unit)

    frequency = _attribute_outcome(
        FREQUENCY_RULE,
        source,
        generated,
        "frequency",
        _canonical_frequency,
    )
    if frequency is not None:
        outcomes.append(frequency)

    route = _attribute_outcome(
        ROUTE_RULE,
        source,
        generated,
        "route",
        _canonical_route,
    )
    if route is not None:
        outcomes.append(route)

    subject_changed = source.subject is not generated.subject
    assertion_pair = {source.assertion, generated.assertion}
    if assertion_pair == {AssertionStatus.PRESENT, AssertionStatus.ABSENT}:
        assertion_rule = (
            ALLERGY_CONTRADICTION_RULE
            if source.type is ClaimType.ALLERGY and generated.type is ClaimType.ALLERGY
            else NEGATION_RULE
        )
        outcomes.append(
            RuleOutcome(
                assertion_rule,
                FindingType.ALTERED,
                source.assertion.value,
                generated.assertion.value,
                "Assertion polarity reversed from "
                f"{source.assertion.value} to {generated.assertion.value}.",
            )
        )
    elif (
        source.assertion is not generated.assertion
        and AssertionStatus.FAMILY_HISTORY not in assertion_pair
    ):
        outcomes.append(
            RuleOutcome(
                CERTAINTY_RULE,
                FindingType.ALTERED,
                source.assertion.value,
                generated.assertion.value,
                "Assertion status changed from "
                f"{source.assertion.value} to {generated.assertion.value}.",
            )
        )

    if subject_changed:
        outcomes.append(
            RuleOutcome(
                SUBJECT_RULE,
                FindingType.ALTERED,
                source.subject.value,
                generated.subject.value,
                f"Subject changed from {source.subject.value} to {generated.subject.value}.",
            )
        )

    lab_types = {ClaimType.LAB_TEST, ClaimType.LAB_VALUE}
    if source.type in lab_types or generated.type in lab_types:
        outcomes.extend(_lab_outcomes(source, generated))
    elif source.type is ClaimType.AGE or generated.type is ClaimType.AGE:
        numeric = _attribute_outcome(
            NUMERIC_RULE,
            source,
            generated,
            "numeric",
            _canonical_numeric,
        )
        if numeric is not None:
            outcomes.append(numeric)

    date = _attribute_outcome(DATE_RULE, source, generated, "date", _canonical_date)
    if date is not None:
        outcomes.append(date)

    duration = _attribute_outcome(
        DURATION_RULE,
        source,
        generated,
        "duration",
        _canonical_duration,
    )
    if duration is not None:
        outcomes.append(duration)

    return outcomes


def rule_by_id(rule_id: str) -> RuleDefinition:
    """Resolve stable rule metadata, primarily for reporting and tests."""

    for rule in RULES:
        if rule.rule_id == rule_id:
            return rule
    raise KeyError(rule_id)

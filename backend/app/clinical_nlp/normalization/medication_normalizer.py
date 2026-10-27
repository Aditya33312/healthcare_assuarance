"""Small, deterministic medication alias baseline.

This vocabulary is intentionally transparent and is not a substitute for RxNorm. Unknown
names are retained in a stable lower-case form so the optional NER adapter remains useful.
"""

from __future__ import annotations

import re

from backend.app.clinical_nlp.versions import NORMALIZATION_VERSION

_MEDICATION_ALIASES: dict[str, str] = {
    "acetaminophen": "acetaminophen",
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "ibuprofen": "ibuprofen",
    "naproxen": "naproxen",
    "aspirin": "aspirin",
    "metformin": "metformin",
    "glucophage": "metformin",
    "glipizide": "glipizide",
    "lisinopril": "lisinopril",
    "losartan": "losartan",
    "amlodipine": "amlodipine",
    "atorvastatin": "atorvastatin",
    "lipitor": "atorvastatin",
    "simvastatin": "simvastatin",
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "apixaban": "apixaban",
    "eliquis": "apixaban",
    "rivaroxaban": "rivaroxaban",
    "insulin": "insulin",
    "amoxicillin": "amoxicillin",
    "ceftriaxone": "ceftriaxone",
    "penicillin": "penicillin",
    "azithromycin": "azithromycin",
    "prednisone": "prednisone",
    "albuterol": "albuterol",
    "salbutamol": "albuterol",
    "omeprazole": "omeprazole",
    "pantoprazole": "pantoprazole",
    "levothyroxine": "levothyroxine",
    "gabapentin": "gabapentin",
    "sertraline": "sertraline",
    "fluoxetine": "fluoxetine",
    "furosemide": "furosemide",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "clopidogrel": "clopidogrel",
    "digoxin": "digoxin",
    "morphine": "morphine",
    "oxycodone": "oxycodone",
    "tramadol": "tramadol",
    "doxycycline": "doxycycline",
    "ciprofloxacin": "ciprofloxacin",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".,")


def normalize_medication(value: str) -> str | None:
    if not isinstance(value, str):
        raise TypeError("medication must be a string")
    candidate = _clean(value)
    if not candidate:
        return None
    return _MEDICATION_ALIASES.get(candidate, candidate)


def medication_aliases() -> dict[str, str]:
    """Return a copy so callers cannot mutate the versioned vocabulary."""

    return dict(_MEDICATION_ALIASES)


class MedicationNormalizer:
    version = NORMALIZATION_VERSION

    def normalize(self, value: str) -> str | None:
        return normalize_medication(value)

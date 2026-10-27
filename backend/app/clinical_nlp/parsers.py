"""Deterministic entity parsers for high-risk clinical attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from backend.app.clinical_nlp.entities import ExtractedEntity
from backend.app.clinical_nlp.limits import ClinicalNLPLimitError, ensure_deadline
from backend.app.clinical_nlp.normalization.date_normalizer import normalize_date
from backend.app.clinical_nlp.normalization.frequency_normalizer import (
    FREQUENCY_EXTRACTION_PATTERNS,
    normalize_frequency,
)
from backend.app.clinical_nlp.normalization.laterality_normalizer import (
    LATERALITY_EXTRACTION_PATTERN,
    normalize_laterality,
)
from backend.app.clinical_nlp.normalization.medication_normalizer import (
    medication_aliases,
    normalize_medication,
)
from backend.app.clinical_nlp.normalization.terminology_normalizer import (
    normalize_terminology,
)
from backend.app.clinical_nlp.normalization.unit_normalizer import (
    normalize_quantity,
    normalize_unit,
)
from backend.app.clinical_nlp.versions import DETERMINISTIC_EXTRACTOR_VERSION
from backend.app.domain.enums import AssertionStatus, ClaimType, SubjectType

_UNSIGNED_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_SIGNED_NUMBER = rf"[+\-\u2212]?{_UNSIGNED_NUMBER}"
_LAB_NUMBER = (
    rf"{_SIGNED_NUMBER}(?:\s*(?:/|[-\u2013\u2014]|to)\s*{_SIGNED_NUMBER})?"
)
_DOSE_UNIT = (
    r"mcg|μg|µg|ug|mg|g|kg|milligrams?|grams?|micrograms?|mL|ml|"
    r"millilit(?:er|re)s?|IU|international\s+units?|units?"
)
_DOSE_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_UNSIGNED_NUMBER})\s*(?P<unit>{_DOSE_UNIT})(?!\w|\s*/)",
    re.IGNORECASE,
)
_AGE_RE = re.compile(
    r"(?<!\w)(?P<value>\d{1,3})\s*(?:-\s*)?(?P<unit>years?|yrs?|months?|mos?)"
    r"(?:\s*-?\s*old|\s+of\s+age)(?!\w)",
    re.IGNORECASE,
)
_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
}
_DURATION_VALUE = rf"{_UNSIGNED_NUMBER}|{'|'.join(_NUMBER_WORDS)}"
_DURATION_RE = re.compile(
    rf"(?<!\w)(?:(?P<value>{_DURATION_VALUE})\s*(?P<unit>minutes?|mins?|hours?|hrs?|"
    r"days?|weeks?|months?|years?|yrs?)|(?P<fortnight>(?:a|one)\s+fortnights?))(?!\w)",
    re.IGNORECASE,
)

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{4}(?!\d)"),
    re.compile(
        r"(?<!\w)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|"
        r"Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\w)(?:today|yesterday|tomorrow)(?!\w)", re.IGNORECASE),
)

_ROUTE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), canonical)
    for pattern, canonical in (
        (r"(?<!\w)(?:PO|p\.o\.|by\s+mouth|orally|oral)(?!\w)", "ORAL"),
        (r"(?<!\w)(?:IV|i\.v\.|intravenous(?:ly)?)(?!\w)", "INTRAVENOUS"),
        (r"(?<!\w)(?:IM|i\.m\.|intramuscular(?:ly)?)(?!\w)", "INTRAMUSCULAR"),
        (r"(?<!\w)(?:SC|SQ|subcutaneous(?:ly)?)(?!\w)", "SUBCUTANEOUS"),
        (r"(?<!\w)(?:inhaled|by\s+inhalation)(?!\w)", "INHALATION"),
        (r"(?<!\w)(?:topical(?:ly)?)(?!\w)", "TOPICAL"),
        (r"(?<!\w)(?:SL|sublingual(?:ly)?)(?!\w)", "SUBLINGUAL"),
        (r"(?<!\w)(?:rectal(?:ly)?|PR)(?!\w)", "RECTAL"),
        (r"(?<!\w)(?:intranasal(?:ly)?|nasal(?:ly)?)(?!\w)", "NASAL"),
        (r"(?<!\w)(?:ophthalmic|into\s+(?:the\s+)?eyes?)(?!\w)", "OPHTHALMIC"),
        (r"(?<!\w)(?:otic|into\s+(?:the\s+)?ears?)(?!\w)", "OTIC"),
    )
)

_BODY_SITES = (
    "chest|abdomen|abdominal|backs?|necks?|heads?|knees?|hips?|shoulders?|ankles?|"
    "elbows?|wrists?|hands?|feet|foot|arms?|legs?|eyes?|ears?|breasts?|flanks?|"
    "calves|calf|thighs?|jaws?"
)
_LATERALITY = r"left(?:-sided)?|right(?:-sided)?|lt\.?|rt\.?|bilateral|both"
_BODY_SYMPTOM = r"pain|discomfort|tenderness|swelling|edema|weakness|numbness|tingling"
_CLINICAL_QUALIFIER = r"crushing|severe|serious|life[- ]threatening"
_BODY_SYMPTOM_RE = re.compile(
    rf"(?<!\w)(?:(?P<qualifier>{_CLINICAL_QUALIFIER})\s+)?"
    rf"(?:(?P<lat>{_LATERALITY})\s+)?(?P<body>{_BODY_SITES})\s+"
    rf"(?P<symptom>{_BODY_SYMPTOM})(?!\w)",
    re.IGNORECASE,
)
_INVERTED_BODY_SYMPTOM_RE = re.compile(
    rf"(?<!\w)(?:(?P<qualifier>{_CLINICAL_QUALIFIER})\s+)?"
    rf"(?P<symptom>{_BODY_SYMPTOM})\s+(?:in|of|over)\s+(?:the\s+)?"
    rf"(?:(?P<lat>{_LATERALITY})\s+)?(?P<body>{_BODY_SITES})(?!\w)",
    re.IGNORECASE,
)

_CLINICAL_TERMS: dict[str, tuple[str, ClaimType]] = {
    "shortness of breath": ("dyspnea", ClaimType.SYMPTOM),
    "dyspnoea": ("dyspnea", ClaimType.SYMPTOM),
    "dyspnea": ("dyspnea", ClaimType.SYMPTOM),
    "chest pain": ("chest pain", ClaimType.SYMPTOM),
    "abdominal pain": ("abdominal pain", ClaimType.SYMPTOM),
    "headache": ("headache", ClaimType.SYMPTOM),
    "fever": ("fever", ClaimType.SYMPTOM),
    "chills": ("chills", ClaimType.SYMPTOM),
    "cough": ("cough", ClaimType.SYMPTOM),
    "nausea": ("nausea", ClaimType.SYMPTOM),
    "vomiting": ("vomiting", ClaimType.SYMPTOM),
    "dizziness": ("dizziness", ClaimType.SYMPTOM),
    "fatigue": ("fatigue", ClaimType.SYMPTOM),
    "rash": ("rash", ClaimType.SYMPTOM),
    "palpitations": ("palpitations", ClaimType.SYMPTOM),
    "syncope": ("syncope", ClaimType.SYMPTOM),
    "diarrhea": ("diarrhea", ClaimType.SYMPTOM),
    "constipation": ("constipation", ClaimType.SYMPTOM),
    "pain": ("pain", ClaimType.SYMPTOM),
    "swelling": ("swelling", ClaimType.SYMPTOM),
    "weakness": ("weakness", ClaimType.SYMPTOM),
    "numbness": ("numbness", ClaimType.SYMPTOM),
    "edema": ("swelling", ClaimType.SYMPTOM),
    "chest discomfort": ("chest pain", ClaimType.SYMPTOM),
    "trauma": ("trauma", ClaimType.CLINICAL_FACT),
    "fall": ("fall", ClaimType.CLINICAL_FACT),
    "loss of consciousness": ("loss of consciousness", ClaimType.CLINICAL_FACT),
    "hypertension": ("hypertension", ClaimType.CONDITION),
    "high blood pressure": ("hypertension", ClaimType.CONDITION),
    "diabetes mellitus": ("diabetes mellitus", ClaimType.CONDITION),
    "type 2 diabetes": ("type 2 diabetes mellitus", ClaimType.CONDITION),
    "type ii diabetes": ("type 2 diabetes mellitus", ClaimType.CONDITION),
    "diabetes": ("diabetes", ClaimType.CONDITION),
    "asthma": ("asthma", ClaimType.CONDITION),
    "pneumonia": ("pneumonia", ClaimType.CONDITION),
    "pulmonary embolism": ("pulmonary embolism", ClaimType.CONDITION),
    "infection": ("infection", ClaimType.CONDITION),
    "urinary tract infection": ("urinary tract infection", ClaimType.CONDITION),
    "heart failure": ("heart failure", ClaimType.CONDITION),
    "atrial fibrillation": ("atrial fibrillation", ClaimType.CONDITION),
    "myocardial infarction": ("myocardial infarction", ClaimType.CONDITION),
    "heart attack": ("myocardial infarction", ClaimType.CONDITION),
    "stroke": ("stroke", ClaimType.CONDITION),
    "anaphylaxis": ("anaphylaxis", ClaimType.CONDITION),
    "pregnancy test": ("pregnancy test", ClaimType.LAB_TEST),
    "cancer": ("cancer", ClaimType.CONDITION),
    "copd": ("copd", ClaimType.CONDITION),
    "glycemic control": ("glycemic control", ClaimType.CLINICAL_FACT),
    "cardiopulmonary cause": ("cardiopulmonary cause", ClaimType.CLINICAL_FACT),
    "respiratory illness": ("respiratory illness", ClaimType.CONDITION),
    "volume overload": ("volume overload", ClaimType.CONDITION),
    "arrhythmia": ("arrhythmia", ClaimType.CONDITION),
    "drug reaction": ("drug reaction", ClaimType.CONDITION),
    "hyponatremia": ("hyponatremia", ClaimType.CONDITION),
    "hyperkalemia": ("hyperkalemia", ClaimType.CONDITION),
    "hyperglycemia": ("hyperglycemia", ClaimType.CONDITION),
    "anticoagulation": ("anticoagulation", ClaimType.CLINICAL_FACT),
    "tachycardia": ("tachycardia", ClaimType.CONDITION),
    "febrile illness": ("febrile illness", ClaimType.CONDITION),
    "hypoxemia": ("hypoxemia", ClaimType.CONDITION),
    "venous thrombosis": ("venous thrombosis", ClaimType.CONDITION),
    "acute coronary syndrome": ("acute coronary syndrome", ClaimType.CONDITION),
    "neurologic event": ("neurologic event", ClaimType.CLINICAL_FACT),
    "acute abdominal process": ("acute abdominal process", ClaimType.CLINICAL_FACT),
    "persistent respiratory condition": (
        "persistent respiratory condition",
        ClaimType.CONDITION,
    ),
    "airway disease": ("airway disease", ClaimType.CONDITION),
    "renal dysfunction": ("renal dysfunction", ClaimType.CONDITION),
    "clinical relationship": ("clinical relationship", ClaimType.CLINICAL_FACT),
    "effusion": ("effusion", ClaimType.CONDITION),
    "symptoms": ("symptom status", ClaimType.CLINICAL_FACT),
    "disease": ("disease", ClaimType.CONDITION),
    "culture": ("culture result", ClaimType.CLINICAL_FACT),
    "response": ("treatment response", ClaimType.CLINICAL_FACT),
    "lesion": ("lesion", ClaimType.CONDITION),
    "medication": ("medication status", ClaimType.CLINICAL_FACT),
    "procedure": ("procedure", ClaimType.PROCEDURE),
    "renal function": ("renal function", ClaimType.CLINICAL_FACT),
    "finding": ("finding", ClaimType.CLINICAL_FACT),
    "appendectomy": ("appendectomy", ClaimType.PROCEDURE),
    "colonoscopy": ("colonoscopy", ClaimType.PROCEDURE),
    "mri": ("mri", ClaimType.PROCEDURE),
    "ct scan": ("ct scan", ClaimType.PROCEDURE),
    "x-ray": ("x-ray", ClaimType.PROCEDURE),
}

_ALLERGY_TO_RE = re.compile(
    r"\b(?:allergic\s+to|allergy\s+to)\s+(?P<target>[A-Za-z][A-Za-z-]{1,40})",
    re.IGNORECASE,
)
_SUFFIX_ALLERGY_RE = re.compile(
    rf"(?<!\w)(?:(?P<severity>{_CLINICAL_QUALIFIER})\s+)?"
    r"(?P<target>[A-Za-z][A-Za-z-]{1,40})\s+allerg(?:y|ies)(?!\w)",
    re.IGNORECASE,
)
_NO_ALLERGY_RE = re.compile(r"\b(?:NKDA|NKA|no\s+known\s+(?:drug\s+)?allergies)\b", re.IGNORECASE)

_LAB_ALIASES: dict[str, str] = {
    "hemoglobin a1c": "hba1c",
    "hba1c": "hba1c",
    "a1c": "hba1c",
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "sodium": "sodium",
    "potassium": "potassium",
    "glucose": "glucose",
    "creatinine": "creatinine",
    "inr": "inr",
    "wbc": "white blood cell count",
    "white blood cell count": "white blood cell count",
    "platelets": "platelet count",
    "platelet count": "platelet count",
    "blood pressure": "blood pressure",
    "bp": "blood pressure",
    "heart rate": "heart rate",
    "pulse": "heart rate",
    "temperature": "temperature",
    "spo2": "oxygen saturation",
    "oxygen saturation": "oxygen saturation",
}
_LAB_UNIT = (
    r"%|mmol\s*/\s*L|mol\s*/\s*L|mg\s*/\s*dL|g\s*/\s*dL|mmHg|bpm|"
    r"°\s*[CF]|[CF](?=\s|[.,;]|$)"
)


_COMPOUND_NUMBER_RE = re.compile(
    rf"(?P<first>{_SIGNED_NUMBER})\s*"
    rf"(?P<separator>/|\.\.|[-\u2013\u2014]|to)\s*"
    rf"(?P<second>{_SIGNED_NUMBER})",
    re.IGNORECASE,
)


def _compact_decimal(value: Decimal) -> int | str:
    """Return an exact JSON-safe scalar without routing through ``float``."""

    integral = value.to_integral_value()
    return int(integral) if value == integral else format(value, "f")


def _parse_decimal(raw: str) -> Decimal:
    return Decimal(raw.strip().replace("\u2212", "-"))


def _number_value(raw: str) -> int | str:
    compound = _COMPOUND_NUMBER_RE.fullmatch(raw.strip())
    if compound is not None:
        try:
            first = _compact_decimal(_parse_decimal(compound.group("first")))
            second = _compact_decimal(_parse_decimal(compound.group("second")))
        except InvalidOperation:
            return raw
        separator = "/" if compound.group("separator") == "/" else ".."
        return f"{first}{separator}{second}"
    try:
        value = _parse_decimal(raw)
    except InvalidOperation:
        return raw
    if not value.is_finite():
        return raw
    return _compact_decimal(value)


def _duration_value(raw_value: str, raw_unit: str) -> tuple[int | str, str, str]:
    word_value = _NUMBER_WORDS.get(raw_value.casefold())
    if word_value is not None:
        value: int | str = word_value
    else:
        decimal_value = Decimal(raw_value)
        value = _compact_decimal(decimal_value)
    unit = raw_unit.casefold()
    if unit.startswith(("min",)):
        canonical_unit, prefix = "MINUTE", "PT"
    elif unit.startswith(("hour", "hr")):
        canonical_unit, prefix = "HOUR", "PT"
    elif unit.startswith("day"):
        canonical_unit, prefix = "DAY", "P"
    elif unit.startswith("week"):
        canonical_unit, prefix = "WEEK", "P"
    elif unit.startswith("month"):
        canonical_unit, prefix = "MONTH", "P"
    else:
        canonical_unit, prefix = "YEAR", "P"
    suffix = {
        "MINUTE": "M",
        "HOUR": "H",
        "DAY": "D",
        "WEEK": "W",
        "MONTH": "M",
        "YEAR": "Y",
    }[canonical_unit]
    return value, canonical_unit, f"{prefix}{value}{suffix}"


def _overlaps_anchor(entity: ExtractedEntity, entities: list[ExtractedEntity]) -> bool:
    return any(
        entity.overlaps(existing)
        for existing in entities
        if existing.type
        in {
            ClaimType.ALLERGY,
            ClaimType.CONDITION,
            ClaimType.SYMPTOM,
            ClaimType.CLINICAL_FACT,
            ClaimType.PROCEDURE,
        }
    )


class _BoundedEntityList(list[ExtractedEntity]):
    def __init__(self, max_entities: int | None, deadline: float | None) -> None:
        super().__init__()
        self.max_entities = max_entities
        self.deadline = deadline

    def check(self) -> None:
        ensure_deadline(self.deadline)
        if self.max_entities is not None and len(self) >= self.max_entities:
            raise ClinicalNLPLimitError("clinical entity-count limit exceeded")

    def append(self, item: ExtractedEntity) -> None:
        self.check()
        super().append(item)

    def extend(self, values: Iterable[ExtractedEntity]) -> None:
        for value in values:
            self.append(value)


class DeterministicEntityParser:
    """Extract exact spans and normalized scalar values without external models."""

    version = DETERMINISTIC_EXTRACTOR_VERSION

    def __init__(
        self,
        *,
        reference_date: date | datetime | str | None = None,
        day_first_dates: bool = False,
    ) -> None:
        self.reference_date = reference_date
        self.day_first_dates = day_first_dates
        aliases = medication_aliases()
        medication_terms = sorted(aliases, key=len, reverse=True)
        self._medication_pattern = re.compile(
            rf"(?<!\w)(?:{'|'.join(re.escape(term) for term in medication_terms)})(?!\w)",
            re.IGNORECASE,
        )

    def _entity(
        self,
        text: str,
        *,
        claim_type: ClaimType,
        start: int,
        end: int,
        concept: str,
        attributes: dict[str, object] | None = None,
        confidence: float = 1.0,
    ) -> ExtractedEntity:
        return ExtractedEntity(
            type=claim_type,
            text=text[start:end],
            start=start,
            end=end,
            concept=concept,
            attributes=dict(attributes or {}),
            confidence=confidence,
            origin_version=self.version,
        )

    def parse(
        self,
        text: str,
        *,
        max_entities: int | None = None,
        deadline: float | None = None,
    ) -> list[ExtractedEntity]:
        if max_entities is not None and max_entities < 1:
            raise ClinicalNLPLimitError("clinical entity-count limit exceeded")
        entities = _BoundedEntityList(max_entities, deadline)
        parsers = (
            self._parse_allergies,
            self._parse_body_symptoms,
            self._parse_clinical_terms,
            self._parse_medications,
            self._parse_labs,
            self._parse_doses,
            self._parse_frequencies,
            self._parse_routes,
            self._parse_lateralities,
            self._parse_dates,
            self._parse_durations,
            self._parse_ages,
        )
        for parser in parsers:
            ensure_deadline(deadline)
            parser(text, entities)
        return entities

    def _parse_allergies(self, text: str, entities: list[ExtractedEntity]) -> None:
        for pattern in (_ALLERGY_TO_RE, _SUFFIX_ALLERGY_RE):
            for match in pattern.finditer(text):
                if match.group("target").casefold() in {"no", "known", "drug", "information"}:
                    continue
                is_suffix = pattern is _SUFFIX_ALLERGY_RE
                start, end = match.span() if is_suffix else match.span("target")
                concept = (
                    normalize_medication(match.group("target")) or match.group("target").casefold()
                )
                attributes: dict[str, object] = {}
                qualifier = match.groupdict().get("severity")
                if qualifier:
                    attributes["severity_qualifier"] = (
                        qualifier.replace("-", "_").replace(" ", "_").upper()
                    )
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.ALLERGY,
                        start=start,
                        end=end,
                        concept=concept,
                        attributes=attributes,
                    )
                )
        for match in _NO_ALLERGY_RE.finditer(text):
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.ALLERGY,
                    start=match.start(),
                    end=match.end(),
                    concept="drug allergy",
                    attributes={
                        "_forced_assertion": AssertionStatus.ABSENT,
                        "_forced_subject": SubjectType.PATIENT,
                    },
                )
            )

    def _parse_body_symptoms(self, text: str, entities: list[ExtractedEntity]) -> None:
        for pattern in (_BODY_SYMPTOM_RE, _INVERTED_BODY_SYMPTOM_RE):
            for match in pattern.finditer(text):
                body = match.group("body").casefold()
                if body == "abdominal":
                    body = "abdomen"
                elif body == "feet":
                    body = "foot"
                elif body == "calves":
                    body = "calf"
                elif body.endswith("s") and body not in {"chest"}:
                    body = body[:-1]
                symptom = match.group("symptom").casefold()
                symptom = {"discomfort": "pain", "edema": "swelling"}.get(symptom, symptom)
                attributes: dict[str, object] = {"body_site": body}
                qualifier = match.group("qualifier")
                if qualifier:
                    attributes["clinical_qualifier"] = (
                        qualifier.replace("-", "_").replace(" ", "_").upper()
                    )
                laterality_text = match.group("lat")
                if laterality_text:
                    attributes["laterality"] = normalize_laterality(laterality_text)
                anchor = self._entity(
                    text,
                    claim_type=ClaimType.SYMPTOM,
                    start=match.start(),
                    end=match.end(),
                    concept=f"{body} {symptom}",
                    attributes=attributes,
                )
                if not _overlaps_anchor(anchor, entities):
                    entities.append(anchor)
                body_start, body_end = match.span("body")
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.BODY_SITE,
                        start=body_start,
                        end=body_end,
                        concept=body,
                        attributes={"parent_concept": anchor.concept},
                    )
                )
                if laterality_text:
                    lat_start, lat_end = match.span("lat")
                    normalized_laterality = normalize_laterality(laterality_text)
                    if normalized_laterality:
                        entities.append(
                            self._entity(
                                text,
                                claim_type=ClaimType.LATERALITY,
                                start=lat_start,
                                end=lat_end,
                                concept=normalized_laterality,
                                attributes={
                                    "laterality": normalized_laterality,
                                    "normalized_value": normalized_laterality,
                                    "parent_concept": anchor.concept,
                                },
                            )
                        )

    def _parse_clinical_terms(self, text: str, entities: list[ExtractedEntity]) -> None:
        for term in sorted(_CLINICAL_TERMS, key=len, reverse=True):
            concept, claim_type = _CLINICAL_TERMS[term]
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                entity = self._entity(
                    text,
                    claim_type=claim_type,
                    start=match.start(),
                    end=match.end(),
                    concept=normalize_terminology(concept) or concept,
                )
                if not _overlaps_anchor(entity, entities):
                    entities.append(entity)

    def _parse_medications(self, text: str, entities: list[ExtractedEntity]) -> None:
        allergy_spans = [entity for entity in entities if entity.type == ClaimType.ALLERGY]
        for match in self._medication_pattern.finditer(text):
            entity = self._entity(
                text,
                claim_type=ClaimType.MEDICATION,
                start=match.start(),
                end=match.end(),
                concept=normalize_medication(match.group()) or match.group().casefold(),
                attributes=self._medication_context(text, match.start()),
            )
            if not any(entity.overlaps(allergy) for allergy in allergy_spans):
                entities.append(entity)

    @staticmethod
    def _medication_context(text: str, medication_start: int) -> dict[str, object]:
        """Return a bounded, explicit medication action for safe slot alignment."""

        clause = re.split(r"[.;\n]", text[max(0, medication_start - 80) : medication_start])[-1]
        action_patterns = (
            (r"\b(?:take|takes|taking)\b", "TAKE"),
            (r"\b(?:continue|continues|continued)\b", "CONTINUE"),
            (r"\b(?:administer|administers|administered)\b", "ADMINISTER"),
            (r"\b(?:give|gives|given)\b", "GIVE"),
            (r"\b(?:use|uses|using)\b", "USE"),
            (r"\b(?:start|starts|started)\b", "START"),
            (r"\b(?:stop|stops|stopped|discontinue|discontinues|discontinued)\b", "STOP"),
        )
        matches: list[tuple[int, str]] = []
        for pattern, action in action_patterns:
            matches.extend((match.start(), action) for match in re.finditer(pattern, clause, re.I))
        if not matches:
            return {}
        _, action = max(matches, key=lambda item: item[0])
        return {"medication_action": action}

    def _parse_labs(self, text: str, entities: list[ExtractedEntity]) -> None:
        for alias in sorted(_LAB_ALIASES, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<!\w)(?P<test>{re.escape(alias)})(?!\w)(?:\s+level)?\s*"
                rf"(?:(?:was|is|of|at)\s+|[:=]\s*)?"
                rf"(?P<value>{_LAB_NUMBER})\s*"
                rf"(?P<unit>{_LAB_UNIT})?",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                concept = _LAB_ALIASES[alias]
                test_start, test_end = match.span("test")
                if any(
                    entity.type is ClaimType.LAB_TEST
                    and entity.concept == concept
                    and test_start < entity.end
                    and entity.start < test_end
                    for entity in entities
                ):
                    continue
                value = _number_value(match.group("value"))
                unit_text = match.group("unit")
                unit = normalize_unit(re.sub(r"\s+", "", unit_text)) if unit_text else None
                test_attrs: dict[str, object] = {"value": value, "normalized_value": value}
                if unit:
                    test_attrs["unit"] = unit
                    test_attrs["normalized_unit"] = unit
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.LAB_TEST,
                        start=test_start,
                        end=test_end,
                        concept=concept,
                        attributes=test_attrs,
                    )
                )
                value_start, value_end = match.span("value")
                if unit_text:
                    value_end = match.end("unit")
                value_attrs = dict(test_attrs)
                value_attrs["parent_concept"] = concept
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.LAB_VALUE,
                        start=value_start,
                        end=value_end,
                        concept=f"{value}{f' {unit}' if unit else ''}",
                        attributes=value_attrs,
                    )
                )

    def _parse_doses(self, text: str, entities: list[ExtractedEntity]) -> None:
        lab_values = [entity for entity in entities if entity.type == ClaimType.LAB_VALUE]
        for match in _DOSE_RE.finditer(text):
            if any(match.start() < lab.end and lab.start < match.end() for lab in lab_values):
                continue
            raw_value = match.group("value")
            raw_unit = match.group("unit")
            quantity = normalize_quantity(raw_value, raw_unit)
            if quantity is None:
                continue
            normalized_value = (
                quantity.value
                if isinstance(quantity.value, int)
                else _compact_decimal(quantity.value)
            )
            attributes: dict[str, object] = {
                "value": normalized_value,
                "unit": quantity.unit,
                "dose": normalized_value,
                "normalized_value": normalized_value,
                "normalized_unit": quantity.unit,
                "raw_value": _number_value(raw_value),
                "raw_unit": normalize_unit(raw_unit) or raw_unit,
            }
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.DOSE,
                    start=match.start(),
                    end=match.end(),
                    concept=f"{normalized_value} {quantity.unit}",
                    attributes=attributes,
                )
            )
            unit_start, unit_end = match.span("unit")
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.UNIT,
                    start=unit_start,
                    end=unit_end,
                    concept=quantity.unit,
                    attributes={
                        "normalized_unit": quantity.unit,
                        "raw_unit": normalize_unit(raw_unit) or raw_unit,
                    },
                )
            )

    def _parse_frequencies(self, text: str, entities: list[ExtractedEntity]) -> None:
        candidates: list[tuple[int, int, str]] = []
        for pattern in FREQUENCY_EXTRACTION_PATTERNS:
            for match in pattern.finditer(text):
                normalized = normalize_frequency(match.group())
                if normalized is None:
                    continue
                candidates.append((match.start(), match.end(), normalized))
        selected: list[tuple[int, int, str]] = []
        for start, end, normalized in sorted(
            set(candidates), key=lambda item: (item[0], -(item[1] - item[0]))
        ):
            if any(start < kept_end and kept_start < end for kept_start, kept_end, _ in selected):
                continue
            selected.append((start, end, normalized))
        for start, end, normalized in selected:
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.FREQUENCY,
                    start=start,
                    end=end,
                    concept=normalized,
                    attributes={"frequency": normalized, "normalized_value": normalized},
                )
            )

    def _parse_routes(self, text: str, entities: list[ExtractedEntity]) -> None:
        for pattern, normalized in _ROUTE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.ROUTE,
                        start=match.start(),
                        end=match.end(),
                        concept=normalized,
                        attributes={"route": normalized, "normalized_value": normalized},
                    )
                )

    def _parse_lateralities(self, text: str, entities: list[ExtractedEntity]) -> None:
        for match in LATERALITY_EXTRACTION_PATTERN.finditer(text):
            normalized = normalize_laterality(match.group())
            if normalized:
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.LATERALITY,
                        start=match.start(),
                        end=match.end(),
                        concept=normalized,
                        attributes={"laterality": normalized, "normalized_value": normalized},
                    )
                )

    def _parse_dates(self, text: str, entities: list[ExtractedEntity]) -> None:
        seen: set[tuple[int, int]] = set()
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                if match.span() in seen:
                    continue
                normalized = normalize_date(
                    match.group(),
                    reference_date=self.reference_date,
                    day_first=self.day_first_dates,
                )
                if normalized is None:
                    continue
                seen.add(match.span())
                entities.append(
                    self._entity(
                        text,
                        claim_type=ClaimType.DATE,
                        start=match.start(),
                        end=match.end(),
                        concept=normalized,
                        attributes={"date": normalized, "normalized_value": normalized},
                    )
                )

    def _parse_durations(self, text: str, entities: list[ExtractedEntity]) -> None:
        age_spans = [match.span() for match in _AGE_RE.finditer(text)]
        for match in _DURATION_RE.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in age_spans):
                continue
            prefix = text[max(0, match.start() - 8) : match.start()].casefold()
            if re.search(r"(?:every|q)\s*$", prefix):
                continue
            if match.group("fortnight") is not None:
                raw_value, raw_unit = "2", "weeks"
            else:
                raw_value, raw_unit = match.group("value"), match.group("unit")
            value, unit, normalized = _duration_value(raw_value, raw_unit)
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.DURATION,
                    start=match.start(),
                    end=match.end(),
                    concept=normalized,
                    attributes={
                        "duration": normalized,
                        "value": value,
                        "unit": unit,
                        "normalized_value": normalized,
                    },
                )
            )

    def _parse_ages(self, text: str, entities: list[ExtractedEntity]) -> None:
        for match in _AGE_RE.finditer(text):
            value = int(match.group("value"))
            raw_unit = match.group("unit").casefold()
            unit = "MONTH" if raw_unit.startswith(("month", "mo")) else "YEAR"
            entities.append(
                self._entity(
                    text,
                    claim_type=ClaimType.AGE,
                    start=match.start(),
                    end=match.end(),
                    concept=f"{value} {unit}",
                    attributes={"value": value, "unit": unit, "normalized_value": value},
                )
            )

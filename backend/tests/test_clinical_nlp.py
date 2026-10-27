"""Focused regression tests for the first-sprint clinical NLP baseline."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from backend.app.clinical_nlp import (
    HF_NER_MODEL_REVISION,
    ClinicalNLPService,
)
from backend.app.clinical_nlp.ner import HuggingFaceClinicalNERAdapter
from backend.app.clinical_nlp.normalization.date_normalizer import normalize_date
from backend.app.clinical_nlp.normalization.frequency_normalizer import normalize_frequency
from backend.app.clinical_nlp.normalization.laterality_normalizer import normalize_laterality
from backend.app.clinical_nlp.normalization.medication_normalizer import normalize_medication
from backend.app.clinical_nlp.normalization.terminology_normalizer import normalize_terminology
from backend.app.clinical_nlp.normalization.unit_normalizer import (
    NormalizedQuantity,
    normalize_quantity,
    normalize_unit,
)
from backend.app.clinical_nlp.preprocessing import preprocess
from backend.app.domain import AssertionStatus, Claim, ClaimType, SubjectType


def _claim(claims: list[Claim], claim_type: ClaimType, concept: str) -> Claim:
    return next(claim for claim in claims if claim.type == claim_type and claim.concept == concept)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("BID", "TWICE_DAILY"),
        ("BD", "TWICE_DAILY"),
        ("twice daily", "TWICE_DAILY"),
        ("2x/day", "TWICE_DAILY"),
        ("TDS", "THREE_TIMES_DAILY"),
        ("q8h", "EVERY_8_HOURS"),
        ("as needed", "AS_NEEDED"),
    ],
)
def test_frequency_normalizer(source: str, expected: str) -> None:
    assert normalize_frequency(source) == expected


def test_unit_normalizer_and_mass_conversion() -> None:
    assert normalize_unit("milligrams") == "mg"
    assert normalize_quantity("0.5", "g") == NormalizedQuantity(500, "mg")
    assert normalize_quantity(1000, "mcg") == NormalizedQuantity(1, "mg")


def test_quantity_normalization_retains_arbitrary_decimal_precision() -> None:
    value = Decimal("0.123456789012345678901")

    quantity = normalize_quantity(value, "mg")

    assert quantity == NormalizedQuantity(value, "mg")
    assert quantity is not None and isinstance(quantity.value, Decimal)


def test_named_normalizers() -> None:
    assert normalize_laterality("Lt.") == "LEFT"
    assert normalize_laterality("both sides") == "BILATERAL"
    assert normalize_medication("Glucophage") == "metformin"
    assert normalize_medication("Glipizide") == "glipizide"
    assert normalize_medication("ceftriaxone") == "ceftriaxone"
    assert normalize_terminology("shortness of breath") == "dyspnea"
    assert normalize_date("January 5, 2024") == "2024-01-05"
    assert normalize_date("yesterday", reference_date=date(2024, 1, 5)) == "2024-01-04"


def test_preprocessing_retains_exact_offsets_across_unicode_and_newlines() -> None:
    text = "  No fever.\nDose: 0.5 g — today."
    document = preprocess(text)

    assert document.text is text
    assert [text[sentence.start : sentence.end] for sentence in document.sentences] == [
        sentence.text for sentence in document.sentences
    ]
    assert "0.5 g" in document.sentences[-1].text


def test_medication_modifiers_are_normalized_and_linked() -> None:
    text = "Take Glucophage 0.5 g BID PO."
    claims = ClinicalNLPService().extract(text, "med-1")
    medication = _claim(claims, ClaimType.MEDICATION, "metformin")

    assert medication.attributes["dose"] == 500
    assert medication.attributes["unit"] == "mg"
    assert medication.attributes["frequency"] == "TWICE_DAILY"
    assert medication.attributes["route"] == "ORAL"
    dose = _claim(claims, ClaimType.DOSE, "500 mg")
    assert dose.attributes["parent_claim_id"] == medication.claim_id
    assert dose.attributes["parent_concept"] == "metformin"


def test_frequency_extraction_uses_longest_non_overlapping_phrase() -> None:
    claims = ClinicalNLPService().extract("Ibuprofen 400 mg twice daily.", "freq-1")
    frequencies = [claim for claim in claims if claim.type == ClaimType.FREQUENCY]

    assert [(claim.source_span.text, claim.concept) for claim in frequencies] == [
        ("twice daily", "TWICE_DAILY")
    ]


def test_two_medications_receive_their_nearest_dose() -> None:
    claims = ClinicalNLPService().extract(
        "Metformin 500 mg BID and lisinopril 10 mg daily.", "med-2"
    )

    metformin = _claim(claims, ClaimType.MEDICATION, "metformin")
    lisinopril = _claim(claims, ClaimType.MEDICATION, "lisinopril")
    assert (metformin.attributes["dose"], metformin.attributes["frequency"]) == (
        500,
        "TWICE_DAILY",
    )
    assert (lisinopril.attributes["dose"], lisinopril.attributes["frequency"]) == (
        10,
        "ONCE_DAILY",
    )


def test_laterality_and_duration_link_to_symptom() -> None:
    text = "Pain in both knees for 2 weeks."
    claims = ClinicalNLPService().extract(text, "symptom-1")
    symptom = _claim(claims, ClaimType.SYMPTOM, "knee pain")

    assert symptom.attributes["laterality"] == "BILATERAL"
    assert symptom.attributes["body_site"] == "knee"
    assert symptom.attributes["duration"] == "P2W"
    laterality = _claim(claims, ClaimType.LATERALITY, "BILATERAL")
    assert laterality.attributes["parent_claim_id"] == symptom.claim_id


def test_lexical_duration_variants_normalize_without_losing_offsets() -> None:
    text = "Cough persisted for a fortnight and then for two weeks."
    claims = ClinicalNLPService().extract(text, "duration-words")
    durations = [claim for claim in claims if claim.type is ClaimType.DURATION]

    assert [(claim.source_span.text, claim.concept) for claim in durations] == [
        ("a fortnight", "P2W"),
        ("two weeks", "P2W"),
    ]


def test_bounded_common_terms_preserve_clinical_structure() -> None:
    claims = ClinicalNLPService().extract(
        "Bilateral ankle edema and chest discomfort followed a fall; "
        "pulmonary embolism was considered.",
        "bounded-terms",
    )

    edema = _claim(claims, ClaimType.SYMPTOM, "ankle swelling")
    assert edema.attributes == {
        "body_site": "ankle",
        "laterality": "BILATERAL",
        "assertion_rule": "default-present",
    }
    assert _claim(claims, ClaimType.SYMPTOM, "chest pain").source_span.text == ("chest discomfort")
    assert _claim(claims, ClaimType.CLINICAL_FACT, "fall").source_span.text == "fall"
    assert _claim(claims, ClaimType.CONDITION, "pulmonary embolism").source_span.text == (
        "pulmonary embolism"
    )


def test_lab_level_grammar_and_overlapping_aliases_are_deduplicated() -> None:
    sodium = ClinicalNLPService().extract("The sodium level is 140 mmol/L.", "lab-level")
    sodium_test = _claim(sodium, ClaimType.LAB_TEST, "sodium")
    assert sodium_test.attributes["normalized_value"] == 140

    a1c = ClinicalNLPService().extract("Hemoglobin A1c is 9.2%.", "lab-overlap")
    assert (
        len(
            [
                claim
                for claim in a1c
                if claim.type is ClaimType.LAB_TEST and claim.concept == "hba1c"
            ]
        )
        == 1
    )


def test_medication_action_context_is_explicit_and_bounded() -> None:
    service = ClinicalNLPService()
    taking = _claim(
        service.extract("The patient takes glipizide.", "med-action"),
        ClaimType.MEDICATION,
        "glipizide",
    )
    stopped = _claim(
        service.extract("Metformin was stopped. Start ceftriaxone.", "med-stop-start"),
        ClaimType.MEDICATION,
        "ceftriaxone",
    )

    assert taking.attributes["medication_action"] == "TAKE"
    assert stopped.attributes["medication_action"] == "START"


def test_negation_possible_and_family_history_scope() -> None:
    text = "No trauma. Possible pneumonia, but fever is absent. Mother had diabetes."
    claims = ClinicalNLPService().extract(text, "assertion-1")

    assert _claim(claims, ClaimType.CLINICAL_FACT, "trauma").assertion == AssertionStatus.ABSENT
    assert _claim(claims, ClaimType.CONDITION, "pneumonia").assertion == AssertionStatus.POSSIBLE
    assert _claim(claims, ClaimType.SYMPTOM, "fever").assertion == AssertionStatus.ABSENT
    family_claim = _claim(claims, ClaimType.CONDITION, "diabetes")
    assert family_claim.assertion == AssertionStatus.FAMILY_HISTORY
    assert family_claim.subject == SubjectType.FAMILY


def test_section_context_handles_family_and_historical_assertions() -> None:
    text = "Family history:\nAsthma.\nPast medical history:\nHypertension."
    claims = ClinicalNLPService().extract(text, "sections-1")

    asthma = _claim(claims, ClaimType.CONDITION, "asthma")
    hypertension = _claim(claims, ClaimType.CONDITION, "hypertension")
    assert (asthma.assertion, asthma.subject) == (
        AssertionStatus.FAMILY_HISTORY,
        SubjectType.FAMILY,
    )
    assert hypertension.assertion == AssertionStatus.HISTORICAL


def test_lab_value_date_age_and_duration_are_structured() -> None:
    text = "A 45-year-old had HbA1c 7.2% on 15/01/2024 for 3 months."
    claims = ClinicalNLPService(day_first_dates=True).extract(text, "numeric-1")

    lab = _claim(claims, ClaimType.LAB_TEST, "hba1c")
    assert lab.attributes == {
        "value": "7.2",
        "normalized_value": "7.2",
        "unit": "%",
        "normalized_unit": "%",
        "assertion_rule": "default-present",
    }
    assert _claim(claims, ClaimType.DATE, "2024-01-15").source_span.text == "15/01/2024"
    assert _claim(claims, ClaimType.AGE, "45 YEAR").attributes["value"] == 45
    assert _claim(claims, ClaimType.DURATION, "P3M").attributes["unit"] == "MONTH"


def test_clinical_numbers_are_parsed_completely_and_without_float_coercion() -> None:
    cases = (
        (
            "Temperature is \N{MINUS SIGN}2.125 °C.",
            "temperature",
            "-2.125",
            "\N{MINUS SIGN}2.125 °C",
        ),
        ("INR is 2.0-3.0.", "inr", "2..3", "2.0-3.0"),
        ("SpO2 is 91.25%.", "oxygen saturation", "91.25", "91.25%"),
        ("BP is 120/80 mmHg.", "blood pressure", "120/80", "120/80 mmHg"),
    )
    service = ClinicalNLPService()

    for index, (text, concept, normalized, evidence) in enumerate(cases):
        claims = service.extract(text, f"numeric-exact-{index}")
        test_claim = _claim(claims, ClaimType.LAB_TEST, concept)
        value_claim = next(
            claim
            for claim in claims
            if claim.type is ClaimType.LAB_VALUE
            and claim.attributes.get("parent_concept") == concept
        )

        assert test_claim.attributes["normalized_value"] == normalized
        assert not isinstance(test_claim.attributes["normalized_value"], float)
        assert Claim.model_validate(test_claim.model_dump(mode="json")) == test_claim
        assert value_claim.source_span.text == evidence
        assert text[value_claim.source_span.start : value_claim.source_span.end] == evidence


def test_claims_have_exact_source_spans_and_stable_document_id() -> None:
    text = "Patient reports left knee pain. Ibuprofen 400 mg BID."
    service = ClinicalNLPService()
    first = service.extract(text, "offset-1")
    second = service.extract(text, "offset-1")

    assert [claim.claim_id for claim in first] == [claim.claim_id for claim in second]
    for claim in first:
        span = claim.source_span
        assert span.document_id == "offset-1"
        assert text[span.start : span.end] == span.text


def test_default_service_is_offline_and_reports_versions() -> None:
    service = ClinicalNLPService()

    assert service.hf_enabled is False
    ner_model = service.versions["ner_model"]
    assert ner_model is not None
    assert ner_model.startswith("deterministic-clinical-extractor@")
    assert HF_NER_MODEL_REVISION in ClinicalNLPService.optional_ner_model_version


def test_lazy_hf_adapter_is_not_loaded_during_construction() -> None:
    calls = 0

    def loader():  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return lambda _text: []

    adapter = HuggingFaceClinicalNERAdapter(loader=loader)
    service = ClinicalNLPService(ner_adapter=adapter)
    assert calls == 0

    service.extract("Fever.", "hf-lazy")
    assert calls == 1


def test_optional_ner_failure_degrades_to_deterministic_claims() -> None:
    def loader():  # type: ignore[no-untyped-def]
        raise OSError("model unavailable")

    adapter = HuggingFaceClinicalNERAdapter(loader=loader)
    service = ClinicalNLPService(ner_adapter=adapter)
    claims = service.extract("No fever.", "hf-failure")

    assert _claim(claims, ClaimType.SYMPTOM, "fever").assertion == AssertionStatus.ABSENT
    assert service.last_ner_error == "NER_ADAPTER_FAILED"
    assert "model unavailable" not in service.last_ner_error


def test_injected_ner_adds_candidate_but_rules_own_assertion() -> None:
    class StubAdapter:
        version = "stub-ner@1"
        last_error: str | None = None

        def extract(self, text: str):  # type: ignore[no-untyped-def]
            start = text.index("melanoma")
            return [
                {
                    "start": start,
                    "end": start + len("melanoma"),
                    "label": "Disease_disorder",
                    "score": 0.91,
                }
            ]

    claims = ClinicalNLPService(ner_adapter=StubAdapter()).extract("No melanoma.", "hf-injected")
    melanoma = _claim(claims, ClaimType.CONDITION, "melanoma")

    assert melanoma.assertion == AssertionStatus.ABSENT
    assert melanoma.extractor == "pretrained_hf_ner"
    assert melanoma.extractor_version == "stub-ner@1"


@pytest.mark.parametrize(("text", "document_id"), [(None, "doc"), ("text", None)])
def test_invalid_argument_types_raise(text: object, document_id: object) -> None:
    with pytest.raises(TypeError):
        ClinicalNLPService().extract(text, document_id)  # type: ignore[arg-type]


def test_empty_text_returns_no_claims_but_empty_document_id_is_invalid() -> None:
    service = ClinicalNLPService()
    assert service.extract("   ", "empty") == []
    with pytest.raises(ValueError, match="document_id"):
        service.extract("fever", "")

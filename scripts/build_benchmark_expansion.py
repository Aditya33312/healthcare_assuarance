"""Build the balanced 200-case clinical-faithfulness engineering corpus.

The first 50 records are the original engineering corpus.  The additional case
specifications below are deliberately readable and reviewable; this script only
computes identifiers, exact offsets, aggregate counts, hashes, and split files.
The immutable splits are regression partitions only: every record was visible
before the v0.4 split freeze and remains engineering-labeled until review.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark" / "clinical_faithfulness" / "v0"
CASES_PATH = BENCHMARK_DIR / "cases.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
SPLITS_PATH = BENCHMARK_DIR / "splits.json"
CALIBRATION_TEMPLATE_PATH = BENCHMARK_DIR / "clinician-calibration-template.json"
BASE_CASE_COUNT = 50
BENCHMARK_VERSION = "0.4.0"
SOURCE_REFERENCE = "ENGINEERING_BALANCED_EXPANSION_V0_4"


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    source: str
    generated: str
    finding_type: str
    category: str
    severity: str
    reason_code: str
    source_evidence: str | None
    generated_evidence: str | None
    alignment_method: str


def _spec(
    name: str,
    source: str,
    generated: str,
    finding_type: str,
    category: str,
    severity: str,
    reason_code: str,
    source_evidence: str | None,
    generated_evidence: str | None,
    alignment_method: str,
) -> CaseSpec:
    return CaseSpec(
        name,
        source,
        generated,
        finding_type,
        category,
        severity,
        reason_code,
        source_evidence,
        generated_evidence,
        alignment_method,
    )


def _supported_specs() -> list[CaseSpec]:
    rows = [
        (
            "Headache preserved",
            "Patient reports headache.",
            "Headache is reported.",
            "symptom",
            "EXACT_MATCH",
            "headache",
            "Headache",
            "EXACT",
        ),
        (
            "Hypertension paraphrase",
            "The patient has hypertension.",
            "High blood pressure is documented.",
            "diagnosis",
            "SEMANTIC_EQUIVALENCE",
            "hypertension",
            "High blood pressure",
            "NORMALIZED",
        ),
        (
            "Myocardial infarction synonym",
            "The patient had a heart attack.",
            "Myocardial infarction is documented.",
            "diagnosis",
            "SEMANTIC_EQUIVALENCE",
            "heart attack",
            "Myocardial infarction",
            "NORMALIZED",
        ),
        (
            "Dyspnea synonym",
            "The patient reports dyspnea.",
            "Shortness of breath is reported.",
            "symptom",
            "SEMANTIC_EQUIVALENCE",
            "dyspnea",
            "Shortness of breath",
            "NORMALIZED",
        ),
        (
            "Edema synonym",
            "The patient has leg edema.",
            "Leg swelling is present.",
            "symptom",
            "SEMANTIC_EQUIVALENCE",
            "leg edema",
            "Leg swelling",
            "NORMALIZED",
        ),
        (
            "Acetaminophen alias",
            "Take acetaminophen.",
            "Take paracetamol.",
            "medication_identity",
            "NORMALIZED_EQUIVALENCE",
            "acetaminophen",
            "paracetamol",
            "NORMALIZED",
        ),
        (
            "Ibuprofen brand alias",
            "Take ibuprofen.",
            "Take Advil.",
            "medication_identity",
            "NORMALIZED_EQUIVALENCE",
            "ibuprofen",
            "Advil",
            "NORMALIZED",
        ),
        (
            "Warfarin brand alias",
            "Take warfarin.",
            "Take Coumadin.",
            "medication_identity",
            "NORMALIZED_EQUIVALENCE",
            "warfarin",
            "Coumadin",
            "NORMALIZED",
        ),
        (
            "Albuterol alias",
            "Use albuterol.",
            "Use salbutamol.",
            "medication_identity",
            "NORMALIZED_EQUIVALENCE",
            "albuterol",
            "salbutamol",
            "NORMALIZED",
        ),
        (
            "Half gram dose equivalence",
            "Take metformin 0.5 g daily.",
            "Take metformin 500 mg daily.",
            "medication_dose",
            "NORMALIZED_EQUIVALENCE",
            "0.5 g",
            "500 mg",
            "ATTRIBUTE",
        ),
        (
            "Aspirin gram equivalence",
            "Take aspirin 0.1 g daily.",
            "Take aspirin 100 mg daily.",
            "medication_dose",
            "NORMALIZED_EQUIVALENCE",
            "0.1 g",
            "100 mg",
            "ATTRIBUTE",
        ),
        (
            "Levothyroxine microgram equivalence",
            "Take levothyroxine 1000 mcg daily.",
            "Take levothyroxine 1 mg daily.",
            "medication_dose",
            "NORMALIZED_EQUIVALENCE",
            "1000 mcg",
            "1 mg",
            "ATTRIBUTE",
        ),
        (
            "Amoxicillin gram equivalence",
            "Take amoxicillin 1 g daily.",
            "Take amoxicillin 1000 mg daily.",
            "medication_dose",
            "NORMALIZED_EQUIVALENCE",
            "1 g",
            "1000 mg",
            "ATTRIBUTE",
        ),
        (
            "BID frequency equivalence",
            "Take metformin BID.",
            "Take metformin twice daily.",
            "medication_frequency",
            "NORMALIZED_EQUIVALENCE",
            "BID",
            "twice daily",
            "ATTRIBUTE",
        ),
        (
            "Eight-hour frequency equivalence",
            "Take amoxicillin q8h.",
            "Take amoxicillin every 8 hours.",
            "medication_frequency",
            "NORMALIZED_EQUIVALENCE",
            "q8h",
            "every 8 hours",
            "ATTRIBUTE",
        ),
        (
            "Bedtime frequency equivalence",
            "Take gabapentin qhs.",
            "Take gabapentin at bedtime.",
            "medication_frequency",
            "NORMALIZED_EQUIVALENCE",
            "qhs",
            "at bedtime",
            "ATTRIBUTE",
        ),
        (
            "Oral route equivalence",
            "Take metformin oral.",
            "Take metformin PO.",
            "medication_route",
            "NORMALIZED_EQUIVALENCE",
            "oral",
            "PO",
            "ATTRIBUTE",
        ),
        (
            "Intravenous route equivalence",
            "Give ceftriaxone IV.",
            "Give ceftriaxone intravenously.",
            "medication_route",
            "NORMALIZED_EQUIVALENCE",
            "IV",
            "intravenously",
            "ATTRIBUTE",
        ),
        (
            "Subcutaneous route equivalence",
            "Use insulin SC.",
            "Use insulin subcutaneously.",
            "medication_route",
            "NORMALIZED_EQUIVALENCE",
            "SC",
            "subcutaneously",
            "ATTRIBUTE",
        ),
        (
            "Left laterality equivalence",
            "The patient has left knee pain.",
            "The patient has Lt knee pain.",
            "laterality",
            "NORMALIZED_EQUIVALENCE",
            "left knee",
            "Lt knee",
            "ATTRIBUTE",
        ),
        (
            "Right laterality equivalence",
            "The patient has right shoulder pain.",
            "The patient has Rt shoulder pain.",
            "laterality",
            "NORMALIZED_EQUIVALENCE",
            "right shoulder",
            "Rt shoulder",
            "ATTRIBUTE",
        ),
        (
            "Chest pain negation equivalence",
            "The patient denies chest pain.",
            "The patient reports no chest pain.",
            "assertion",
            "ASSERTION_EQUIVALENCE",
            "denies chest pain",
            "no chest pain",
            "ATTRIBUTE",
        ),
        (
            "Fever negation equivalence",
            "The patient is negative for fever.",
            "The patient has no fever.",
            "assertion",
            "ASSERTION_EQUIVALENCE",
            "negative for fever",
            "no fever",
            "ATTRIBUTE",
        ),
        (
            "Sodium value preserved",
            "Serum sodium is 138 mmol/L.",
            "The sodium level is 138 mmol/L.",
            "lab_value",
            "EXACT_MATCH",
            "138 mmol/L",
            "138 mmol/L",
            "EXACT",
        ),
        (
            "Date representation preserved",
            "Follow-up is on April 7, 2026.",
            "Follow-up is on 7 April 2026.",
            "date",
            "NORMALIZED_EQUIVALENCE",
            "April 7, 2026",
            "7 April 2026",
            "NORMALIZED",
        ),
    ]
    return [
        _spec(name, source, generated, "SUPPORTED", category, "LOW", reason, src, gen, alignment)
        for name, source, generated, category, reason, src, gen, alignment in rows
    ]


def _altered_specs() -> list[CaseSpec]:
    rows = [
        (
            "Antihypertensive substitution",
            "Take lisinopril daily.",
            "Take losartan daily.",
            "medication_identity",
            "HIGH",
            "MEDICATION_MISMATCH",
            "lisinopril",
            "losartan",
        ),
        (
            "Statin substitution",
            "Take atorvastatin daily.",
            "Take simvastatin daily.",
            "medication_identity",
            "HIGH",
            "MEDICATION_MISMATCH",
            "atorvastatin",
            "simvastatin",
        ),
        (
            "Metformin dose doubled",
            "Take metformin 500 mg daily.",
            "Take metformin 1000 mg daily.",
            "medication_dose",
            "HIGH",
            "DOSE_MISMATCH",
            "500 mg",
            "1000 mg",
        ),
        (
            "Warfarin dose changed",
            "Take warfarin 2 mg daily.",
            "Take warfarin 5 mg daily.",
            "medication_dose",
            "CRITICAL",
            "DOSE_MISMATCH",
            "2 mg",
            "5 mg",
        ),
        (
            "Insulin dose reduced tenfold",
            "Use insulin 80 units nightly.",
            "Use insulin 8 units nightly.",
            "medication_dose",
            "CRITICAL",
            "DOSE_MISMATCH",
            "80 units",
            "8 units",
        ),
        (
            "Daily changed to BID",
            "Take lisinopril once daily.",
            "Take lisinopril BID.",
            "medication_frequency",
            "HIGH",
            "FREQUENCY_MISMATCH",
            "once daily",
            "BID",
        ),
        (
            "Eight-hour changed to twelve-hour",
            "Take amoxicillin q8h.",
            "Take amoxicillin q12h.",
            "medication_frequency",
            "HIGH",
            "FREQUENCY_MISMATCH",
            "q8h",
            "q12h",
        ),
        (
            "Oral changed to intravenous",
            "Take ciprofloxacin oral daily.",
            "Take ciprofloxacin IV daily.",
            "medication_route",
            "HIGH",
            "ROUTE_MISMATCH",
            "oral",
            "IV",
        ),
        (
            "Intramuscular changed to subcutaneous",
            "Give ceftriaxone IM daily.",
            "Give ceftriaxone SC daily.",
            "medication_route",
            "HIGH",
            "ROUTE_MISMATCH",
            "IM",
            "SC",
        ),
        (
            "Knee laterality reversed",
            "The patient has left knee swelling.",
            "The patient has right knee swelling.",
            "laterality",
            "HIGH",
            "LATERALITY_MISMATCH",
            "left knee",
            "right knee",
        ),
        (
            "Shoulder laterality reversed",
            "The patient has right shoulder pain.",
            "The patient has left shoulder pain.",
            "laterality",
            "HIGH",
            "LATERALITY_MISMATCH",
            "right shoulder",
            "left shoulder",
        ),
        (
            "Cough negation reversed",
            "The patient denies cough.",
            "The patient reports cough.",
            "assertion",
            "HIGH",
            "NEGATION_REVERSAL",
            "denies cough",
            "cough",
        ),
        (
            "Anaphylaxis negation reversed",
            "The patient has no anaphylaxis.",
            "The patient has anaphylaxis.",
            "assertion",
            "CRITICAL",
            "NEGATION_REVERSAL",
            "no anaphylaxis",
            "anaphylaxis",
        ),
        (
            "Stroke negation reversed",
            "There is no evidence of stroke.",
            "Stroke is present.",
            "assertion",
            "CRITICAL",
            "NEGATION_REVERSAL",
            "no evidence of stroke",
            "Stroke",
        ),
        (
            "Possible pneumonia made definite",
            "Possible pneumonia is documented.",
            "Pneumonia is documented.",
            "certainty",
            "MODERATE",
            "CERTAINTY_CHANGE",
            "Possible pneumonia",
            "Pneumonia",
        ),
        (
            "Historical asthma made current",
            "History of asthma is documented.",
            "Asthma is present.",
            "certainty",
            "MODERATE",
            "CERTAINTY_CHANGE",
            "History of asthma",
            "Asthma",
        ),
        (
            "Maternal cancer attributed to patient",
            "The patient's mother had cancer.",
            "The patient had cancer.",
            "subject_attribution",
            "HIGH",
            "SUBJECT_ATTRIBUTION_ERROR",
            "mother had cancer",
            "patient had cancer",
        ),
        (
            "Paternal stroke attributed to patient",
            "The patient's father had a stroke.",
            "The patient had a stroke.",
            "subject_attribution",
            "HIGH",
            "SUBJECT_ATTRIBUTION_ERROR",
            "father had a stroke",
            "patient had a stroke",
        ),
        (
            "Sodium numeric mismatch",
            "Serum sodium is 140 mmol/L.",
            "Serum sodium is 125 mmol/L.",
            "lab_value",
            "HIGH",
            "NUMERIC_VALUE_MISMATCH",
            "140 mmol/L",
            "125 mmol/L",
        ),
        (
            "Critical potassium mismatch",
            "Serum potassium is 4.0 mmol/L.",
            "Serum potassium is 6.5 mmol/L.",
            "lab_value",
            "CRITICAL",
            "NUMERIC_VALUE_MISMATCH",
            "4.0 mmol/L",
            "6.5 mmol/L",
        ),
        (
            "INR numeric mismatch",
            "INR is 2.1.",
            "INR is 5.8.",
            "lab_value",
            "HIGH",
            "NUMERIC_VALUE_MISMATCH",
            "2.1",
            "5.8",
        ),
        (
            "Follow-up date changed",
            "Follow-up is on May 2, 2026.",
            "Follow-up is on May 9, 2026.",
            "date",
            "MODERATE",
            "DATE_MISMATCH",
            "May 2, 2026",
            "May 9, 2026",
        ),
        (
            "Symptom duration changed",
            "Cough has lasted 3 days.",
            "Cough has lasted 3 weeks.",
            "duration",
            "MODERATE",
            "DURATION_MISMATCH",
            "3 days",
            "3 weeks",
        ),
        (
            "Penicillin allergy contradicted",
            "A penicillin allergy is documented.",
            "The patient has no known allergies.",
            "allergy",
            "CRITICAL",
            "ALLERGY_CONTRADICTION",
            "penicillin allergy",
            "no known allergies",
        ),
        (
            "Amoxicillin allergy contradicted",
            "The patient is allergic to amoxicillin.",
            "The patient has no known drug allergies.",
            "allergy",
            "CRITICAL",
            "ALLERGY_CONTRADICTION",
            "amoxicillin",
            "no known drug allergies",
        ),
    ]
    return [
        _spec(name, source, generated, "ALTERED", category, severity, reason, src, gen, "ATTRIBUTE")
        for name, source, generated, category, severity, reason, src, gen in rows
    ]


def _added_specs() -> list[CaseSpec]:
    additions = [
        ("nausea", "HIGH"),
        ("vomiting", "HIGH"),
        ("dizziness", "MODERATE"),
        ("fatigue", "MODERATE"),
        ("rash", "HIGH"),
        ("palpitations", "HIGH"),
        ("syncope", "HIGH"),
        ("diarrhea", "MODERATE"),
        ("constipation", "MODERATE"),
        ("asthma", "HIGH"),
        ("pneumonia", "HIGH"),
        ("heart failure", "HIGH"),
        ("atrial fibrillation", "HIGH"),
        ("stroke", "CRITICAL"),
        ("pulmonary embolism", "CRITICAL"),
        ("metformin", "HIGH"),
        ("warfarin", "CRITICAL"),
        ("apixaban", "CRITICAL"),
        ("digoxin", "CRITICAL"),
        ("morphine", "HIGH"),
        ("prednisone", "HIGH"),
        ("ceftriaxone", "HIGH"),
        ("penicillin allergy", "HIGH"),
        ("amoxicillin allergy", "HIGH"),
        ("right knee pain", "HIGH"),
    ]
    symptom_additions = {
        "nausea",
        "vomiting",
        "dizziness",
        "fatigue",
        "rash",
        "palpitations",
        "syncope",
        "diarrhea",
        "constipation",
        "right knee pain",
    }
    medication_additions = {
        "metformin",
        "warfarin",
        "apixaban",
        "digoxin",
        "morphine",
        "prednisone",
        "ceftriaxone",
    }
    specs: list[CaseSpec] = []
    for index, (addition, severity) in enumerate(additions, 1):
        source = f"Patient reports headache; baseline note {index}."
        generated = f"Patient reports headache and {addition}; generated note {index}."
        specs.append(
            _spec(
                f"Unsupported addition {index}: {addition}",
                source,
                generated,
                "ADDED",
                (
                    "allergy"
                    if "allergy" in addition
                    else "medication_identity"
                    if addition in medication_additions
                    else "symptom"
                    if addition in symptom_additions
                    else "diagnosis"
                ),
                severity,
                "UNSUPPORTED_ADDITION",
                None,
                addition,
                "NONE",
            )
        )
    return specs


def _omitted_specs() -> list[CaseSpec]:
    omissions = [
        ("severe penicillin allergy", "CRITICAL"),
        ("serious amoxicillin allergy", "CRITICAL"),
        ("crushing chest pain", "CRITICAL"),
        ("severe abdominal pain", "HIGH"),
        ("insulin", "HIGH"),
        ("warfarin", "HIGH"),
        ("apixaban", "HIGH"),
        ("digoxin", "HIGH"),
        ("stroke", "HIGH"),
        ("heart failure", "HIGH"),
        ("atrial fibrillation", "HIGH"),
        ("pulmonary embolism", "HIGH"),
        ("syncope", "HIGH"),
        ("palpitations", "HIGH"),
        ("fever", "HIGH"),
        ("vomiting", "HIGH"),
        ("rash", "HIGH"),
        ("metformin", "HIGH"),
        ("lisinopril", "HIGH"),
        ("prednisone", "HIGH"),
        ("ceftriaxone", "HIGH"),
        ("potassium", "HIGH"),
        ("sodium", "HIGH"),
        ("pregnancy test", "HIGH"),
        ("right knee pain", "HIGH"),
    ]
    medication_omissions = {
        "insulin",
        "warfarin",
        "apixaban",
        "digoxin",
        "metformin",
        "lisinopril",
        "prednisone",
        "ceftriaxone",
    }
    lab_omissions = {"potassium", "sodium", "pregnancy test"}
    symptom_omissions = {
        "crushing chest pain",
        "severe abdominal pain",
        "syncope",
        "palpitations",
        "fever",
        "vomiting",
        "rash",
        "right knee pain",
    }
    specs: list[CaseSpec] = []
    for index, (omission, severity) in enumerate(omissions, 1):
        if "allergy" in omission:
            source = f"The patient has a {omission}."
        elif omission in {
            "insulin",
            "warfarin",
            "apixaban",
            "digoxin",
            "metformin",
            "lisinopril",
            "prednisone",
            "ceftriaxone",
        }:
            source = f"The patient takes {omission}."
        elif omission in {"potassium", "sodium"}:
            value = "6.2 mmol/L" if omission == "potassium" else "125 mmol/L"
            source = f"Serum {omission} is {value}."
        elif omission == "pregnancy test":
            source = "The pregnancy test is positive."
        else:
            source = f"The patient reports {omission}."
        generated = f"Follow-up planning was documented in summary {index}."
        specs.append(
            _spec(
                f"Important omission {index}: {omission}",
                source,
                generated,
                "OMITTED",
                (
                    "allergy"
                    if "allergy" in omission
                    else "medication_identity"
                    if omission in medication_omissions
                    else "lab_value"
                    if omission in lab_omissions
                    else "symptom"
                    if omission in symptom_omissions
                    else "diagnosis"
                ),
                severity,
                "IMPORTANT_OMISSION",
                omission,
                None,
                "NONE",
            )
        )
    return specs


def _inferred_specs() -> list[CaseSpec]:
    rows = [
        (
            "Fever, cough, and a lobar infiltrate are present.",
            "The findings suggest pneumonia.",
            "pneumonia",
        ),
        (
            "Hemoglobin A1c is 10.1% despite therapy.",
            "Glycemic control is poor.",
            "Glycemic control is poor",
        ),
        (
            "Blood pressure is 178 mmHg despite treatment.",
            "Hypertension is poorly controlled.",
            "Hypertension",
        ),
        ("Fever, chills, and cough are present.", "An infection is suspected.", "infection"),
        (
            "The patient has dyspnea and chest pain.",
            "A cardiopulmonary cause is possible.",
            "cardiopulmonary cause",
        ),
        (
            "The patient has fever and a urinary tract infection.",
            "The infection may be active.",
            "infection",
        ),
        (
            "The patient has cough and shortness of breath.",
            "A respiratory illness is possible.",
            "respiratory illness",
        ),
        ("The patient has edema and dyspnea.", "Volume overload is possible.", "Volume overload"),
        ("The patient has palpitations and syncope.", "An arrhythmia is possible.", "arrhythmia"),
        (
            "The patient has rash after amoxicillin.",
            "A drug reaction is possible.",
            "drug reaction",
        ),
        (
            "Serum sodium is 122 mmol/L and dizziness is present.",
            "Symptoms may relate to hyponatremia.",
            "hyponatremia",
        ),
        (
            "Serum potassium is 6.3 mmol/L and palpitations are present.",
            "Hyperkalemia may be clinically significant.",
            "Hyperkalemia",
        ),
        ("Glucose is 320 mg/dL with fatigue.", "Hyperglycemia is likely.", "Hyperglycemia"),
        (
            "INR is 5.4 while taking warfarin.",
            "Anticoagulation may be excessive.",
            "Anticoagulation may be excessive",
        ),
        ("Heart rate is 130 bpm with palpitations.", "Tachycardia is present.", "Tachycardia"),
        ("Temperature is 39 C with chills.", "A febrile illness is present.", "febrile illness"),
        ("Oxygen saturation is 86% with dyspnea.", "Hypoxemia is likely.", "Hypoxemia"),
        (
            "The patient has right calf pain and swelling.",
            "Venous thrombosis is possible.",
            "Venous thrombosis",
        ),
        (
            "The patient has crushing chest pain and nausea.",
            "Acute coronary syndrome is possible.",
            "Acute coronary syndrome",
        ),
        (
            "The patient has sudden weakness and headache.",
            "A neurologic event is possible.",
            "neurologic event",
        ),
        (
            "The patient has abdominal pain and vomiting.",
            "An acute abdominal process is possible.",
            "acute abdominal process",
        ),
        (
            "The patient has cough for 3 weeks.",
            "A persistent respiratory condition is possible.",
            "persistent respiratory condition",
        ),
        (
            "The patient has fever after ceftriaxone was stopped.",
            "The infection may be unresolved.",
            "infection",
        ),
        (
            "The patient has asthma and uses albuterol daily.",
            "Airway disease remains active.",
            "Airway disease",
        ),
        (
            "Creatinine is 3.2 mg/dL with fatigue.",
            "Renal dysfunction is possible.",
            "Renal dysfunction",
        ),
    ]
    return [
        _spec(
            f"Provisional reasonable inference {index}",
            source,
            generated,
            "INFERRED",
            "inference",
            "MODERATE",
            "DERIVED_CLINICAL_SUMMARY",
            source.rstrip("."),
            evidence,
            "SEMANTIC",
        )
        for index, (source, generated, evidence) in enumerate(rows, 1)
    ]


def _uncertain_specs() -> list[CaseSpec]:
    concepts = [
        "pneumonia",
        "asthma",
        "infection",
        "heart failure",
        "atrial fibrillation",
        "stroke",
        "pulmonary embolism",
        "myocardial infarction",
        "cancer",
        "headache",
        "fever",
        "cough",
        "nausea",
        "vomiting",
        "dizziness",
        "fatigue",
        "rash",
        "palpitations",
        "syncope",
        "abdominal pain",
        "chest pain",
        "dyspnea",
        "urinary tract infection",
        "anaphylaxis",
        "hypertension",
    ]
    return [
        _spec(
            f"Ambiguous assertion {index}: {concept}",
            f"The source documents {concept}.",
            "An unspecified clinical relationship is possible.",
            "UNCERTAIN",
            "inference",
            "MODERATE",
            "SEMANTIC_INFERENCE_UNASSESSED",
            concept,
            "clinical relationship",
            "SEMANTIC",
        )
        for index, concept in enumerate(concepts, 1)
    ]


def expansion_specs() -> list[CaseSpec]:
    groups = (
        _supported_specs(),
        _altered_specs(),
        _added_specs(),
        _omitted_specs(),
        _inferred_specs(),
        _uncertain_specs(),
    )
    assert all(len(group) == 25 for group in groups)
    specs = [spec for group in groups for spec in group]
    assert Counter(spec.finding_type for spec in specs) == {
        "SUPPORTED": 25,
        "ALTERED": 25,
        "ADDED": 25,
        "OMITTED": 25,
        "INFERRED": 25,
        "UNCERTAIN": 25,
    }
    return specs


_TAG_BY_CATEGORY = {
    "symptom": "supported_paraphrase",
    "diagnosis": "supported_paraphrase",
    "clinical_fact": "supported_paraphrase",
    "medication_identity": "medication_identity",
    "medication_dose": "dose",
    "medication_frequency": "frequency",
    "medication_route": "route",
    "laterality": "laterality",
    "assertion": "negation",
    "certainty": "uncertainty",
    "subject_attribution": "subject_attribution",
    "lab_value": "lab_values",
    "date": "dates",
    "duration": "duration",
    "allergy": "negation",
    "inference": "reasonable_inference",
}


def _evidence(document_id: str, content: str, needle: str | None) -> dict[str, object] | None:
    if needle is None:
        return None
    if content.count(needle) != 1:
        raise ValueError(f"Evidence must occur exactly once: {needle!r} in {content!r}")
    start = content.index(needle)
    return {
        "document_id": document_id,
        "text": needle,
        "start": start,
        "end": start + len(needle),
    }


def _record(index: int, spec: CaseSpec) -> dict[str, object]:
    case_id = f"cf-v0-{index:03d}"
    description = f"Engineering-authored development case: {spec.name}."
    return {
        "schema_version": "1.0.0",
        "test_case_id": case_id,
        "test_case_version": "1.0.0",
        "name": spec.name,
        "description": description,
        "test_family": "CLINICAL_FAITHFULNESS",
        "risk_tier": spec.severity,
        "input": {
            "source_documents": [
                {"document_id": "source_001", "type": "clinical_text", "content": spec.source}
            ],
            "generated_output": {
                "document_id": "generated_output",
                "type": "clinical_text",
                "content": spec.generated,
            },
        },
        "expectation": {
            "schema_version": "1.0.0",
            "expectation_type": "EXPECTED_FINDINGS",
            "expected_findings": [
                {
                    "schema_version": "1.0.0",
                    "finding_id": f"gold-{case_id}-01",
                    "type": spec.finding_type,
                    "category": spec.category,
                    "severity": spec.severity,
                    "source": _evidence("source_001", spec.source, spec.source_evidence),
                    "generated": _evidence(
                        "generated_output", spec.generated, spec.generated_evidence
                    ),
                    "reason_code": spec.reason_code,
                    "explanation": description,
                    "confidence": 1,
                    "evaluator": "benchmark_annotation",
                    "evaluator_version": BENCHMARK_VERSION,
                    "severity_policy_version": "benchmark-v0@0.4.0",
                    "alignment_method": spec.alignment_method,
                }
            ],
            "allow_additional_findings": False,
        },
        "tags": [
            (
                "unsupported_additions"
                if spec.finding_type == "ADDED"
                else "important_omissions"
                if spec.finding_type == "OMITTED"
                else "reasonable_inference"
                if spec.finding_type == "INFERRED"
                else _TAG_BY_CATEGORY[spec.category]
            ),
            spec.finding_type.casefold(),
            "balanced_expansion",
        ],
        "provenance": {
            "data_classification": "SYNTHETIC",
            "contains_phi": False,
            "authoring_method": "MANUAL",
            "source_reference": SOURCE_REFERENCE,
        },
        "active": True,
    }


def build_expansion_records() -> list[dict[str, object]]:
    """Return the canonical expansion records without modifying the filesystem."""

    return [
        _record(index, spec) for index, spec in enumerate(expansion_specs(), BASE_CASE_COUNT + 1)
    ]


def _base_records() -> list[dict[str, object]]:
    records = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines()]
    base = [
        record
        for record in records
        if int(str(record["test_case_id"]).rsplit("-", 1)[-1]) <= BASE_CASE_COUNT
    ]
    if len(base) != BASE_CASE_COUNT:
        raise RuntimeError("The canonical first 50 benchmark records are missing or duplicated.")
    canonical_categories = {
        "cf-v0-001": "symptom",
        "cf-v0-002": "symptom",
        "cf-v0-032": "allergy",
        "cf-v0-033": "diagnosis",
        "cf-v0-034": "clinical_fact",
        "cf-v0-035": "allergy",
        "cf-v0-036": "medication_identity",
        "cf-v0-037": "symptom",
        "cf-v0-038": "inference",
        "cf-v0-039": "inference",
        "cf-v0-045": "allergy",
    }
    legacy_categories = {
        "supported_paraphrase": "clinical_fact",
        "unsupported_addition": "clinical_fact",
        "important_omission": "clinical_fact",
        "reasonable_inference": "inference",
        "allergy_assertion": "allergy",
    }
    for record in base:
        case_id = str(record["test_case_id"])
        if case_id == "cf-v0-002":
            source = "The patient has ankle swelling."
            generated = "Ankle edema is present."
            record["input"]["source_documents"][0]["content"] = source
            record["input"]["generated_output"]["content"] = generated
            finding = record["expectation"]["expected_findings"][0]
            finding["source"] = {
                "document_id": "source_001",
                "text": "ankle swelling",
                "start": source.index("ankle swelling"),
                "end": source.index("ankle swelling") + len("ankle swelling"),
            }
            finding["generated"] = {
                "document_id": "generated_output",
                "text": "Ankle edema",
                "start": generated.index("Ankle edema"),
                "end": generated.index("Ankle edema") + len("Ankle edema"),
            }
        findings = record["expectation"]["expected_findings"]
        for finding in findings:
            old_category = str(finding["category"])
            finding["category"] = canonical_categories.get(
                case_id, legacy_categories.get(old_category, old_category)
            )
            finding["evaluator_version"] = BENCHMARK_VERSION
            finding["severity_policy_version"] = "benchmark-v0@0.4.0"
            if case_id in {"cf-v0-038", "cf-v0-039"}:
                finding["reason_code"] = "DERIVED_CLINICAL_SUMMARY"
            if case_id == "cf-v0-021":
                finding.update(
                    {
                        "type": "SUPPORTED",
                        "severity": "LOW",
                        "reason_code": "ASSERTION_EQUIVALENCE",
                        "explanation": (
                            "The possible assertion is preserved; the previous UNCERTAIN "
                            "engineering label conflicted with the frozen relationship definition."
                        ),
                    }
                )
        record["provenance"]["source_reference"] = "ENGINEERING_BASE_CANONICALIZED_V0_4"
    return base


def _case_ids(records: list[dict[str, object]], *, tag: str | None = None) -> list[str]:
    return [
        str(record["test_case_id"])
        for record in records
        if tag is None or tag in record["tags"]  # type: ignore[operator]
    ]


def _write_manifest(records: list[dict[str, object]]) -> None:
    findings = [
        finding
        for record in records
        for finding in record["expectation"]["expected_findings"]  # type: ignore[index]
    ]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["benchmark_version"] = BENCHMARK_VERSION
    manifest["case_count"] = len(records)
    manifest["finding_count"] = len(findings)
    manifest["reproducibility"]["content_hash"] = hashlib.sha256(
        CASES_PATH.read_bytes()
    ).hexdigest()
    manifest["reproducibility"]["generation"] = (
        "engineering-authored case specifications with deterministic ID, offset, count, "
        "hash, manifest, and split generation"
    )
    manifest["annotation_policy"]["authoring_method"] = "ENGINEERING_AUTHORED_TEMPLATE_SPECS"
    manifest["annotation_policy"]["adjudication_status"] = (
        "ENGINEERING_DEVELOPMENT_NOT_CLINICIAN_CALIBRATED"
    )
    manifest["annotation_policy"]["provisional_severity_policy"] = "benchmark-v0@0.4.0"
    manifest["annotation_policy"]["expansion_severity_review"] = (
        "Unsupported additions of configured high-risk diagnoses or medications are "
        "provisionally CRITICAL. Isolated medication-mention omissions without dose or "
        "reconciliation context, and isolated severe abdominal pain, remain HIGH pending "
        "clinician calibration."
    )
    manifest["gold_taxonomy"]["categories"] = sorted({finding["category"] for finding in findings})
    manifest["gold_taxonomy"]["taxonomy_version"] = "1.0.0"
    manifest["gold_taxonomy"]["reason_codes"] = sorted(
        {finding["reason_code"] for finding in findings}
    )
    finding_types = manifest["gold_taxonomy"]["finding_types"]
    manifest["finding_type_counts"] = {
        finding_type: sum(finding["type"] == finding_type for finding in findings)
        for finding_type in finding_types
    }
    required_categories = list(manifest["required_category_coverage"])
    manifest["required_category_coverage"] = {
        category: _case_ids(records, tag=category) for category in required_categories
    }
    manifest["finding_type_coverage"] = {
        finding_type: [
            str(record["test_case_id"])
            for record in records
            if any(
                finding["type"] == finding_type
                for finding in record["expectation"]["expected_findings"]  # type: ignore[index]
            )
        ]
        for finding_type in finding_types
    }

    manifest["engine_to_gold_normalization"]["mapping_version"] = "0.2.0"
    manifest["engine_to_gold_normalization"]["contextual_category_aliases"] = []
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_splits(records: list[dict[str, object]]) -> None:
    # Frozen at benchmark 0.4.0. These explicit ranges are deliberately not
    # recomputed from hashes, record order, or proportions as the corpus grows.
    development_ids = [f"cf-v0-{index:03d}" for index in range(1, 33)]
    validation_ids = [f"cf-v0-{index:03d}" for index in range(33, 39)]
    held_out_ids = [f"cf-v0-{index:03d}" for index in range(39, 45)]
    calibration_ids = [f"cf-v0-{index:03d}" for index in range(45, 51)]
    for block_start in range(51, 201, 25):
        development_ids.extend(
            f"cf-v0-{index:03d}" for index in range(block_start, block_start + 13)
        )
        validation_ids.extend(
            f"cf-v0-{index:03d}" for index in range(block_start + 13, block_start + 17)
        )
        held_out_ids.extend(
            f"cf-v0-{index:03d}" for index in range(block_start + 17, block_start + 21)
        )
        calibration_ids.extend(
            f"cf-v0-{index:03d}" for index in range(block_start + 21, block_start + 25)
        )
    assignment = {
        "development": development_ids,
        "validation": validation_ids,
        "held_out_test": held_out_ids,
        "clinician_calibration": calibration_ids,
    }
    serialized_assignment = json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    assigned_ids = {case_id for ids in assignment.values() for case_id in ids}
    record_ids = set(_case_ids(records))
    if assigned_ids != record_ids:
        raise RuntimeError("Frozen split assignment does not match the canonical case IDs.")
    splits = {
        "schema_version": "1.0.0",
        "benchmark_version": BENCHMARK_VERSION,
        "split_assignment_version": "1.0.0",
        "split_assignment_sha256": hashlib.sha256(serialized_assignment).hexdigest(),
        "split_policy": (
            "Explicit immutable case IDs frozen at benchmark 0.4.0. No builder-time random "
            "assignment is permitted. All 200 cases were visible before this split freeze, "
            "so held_out_test is a regression holdout from 0.4.0 forward, not an independent "
            "clinical validation set. Clinician calibration remains pending review."
        ),
        **assignment,
        "historical_exposure": "ALL_CASES_VISIBLE_BEFORE_SPLIT_FREEZE",
        "reserved_split_status": "POPULATED_ENGINEERING_LABELS_ONLY",
    }
    SPLITS_PATH.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def _write_calibration_template(records: list[dict[str, object]]) -> None:
    split_config = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    calibration_ids = set(split_config["clinician_calibration"])
    cases: list[dict[str, object]] = []
    for record in records:
        if record["test_case_id"] not in calibration_ids:
            continue
        cases.append(
            {
                "test_case_id": record["test_case_id"],
                "source_documents": record["input"]["source_documents"],  # type: ignore[index]
                "generated_output": record["input"]["generated_output"],  # type: ignore[index]
                "reviewer_a": None,
                "reviewer_b": None,
                "adjudication": None,
            }
        )
    template: dict[str, object] = {
        "schema_version": "1.0.0",
        "benchmark_version": BENCHMARK_VERSION,
        "status": "PENDING_TWO_INDEPENDENT_CLINICIAN_REVIEWS",
        "blinding": {
            "automated_gatekeeper_result_included": False,
            "engineering_gold_included": False,
            "reviewers_must_complete_initial_labels_independently": True,
        },
        "review_policy": {
            "minimum_independent_reviewers": 2,
            "reviewer_ids_must_differ": True,
            "gold_change_requires_adjudication": True,
            "single_reviewer_gold_change_allowed": False,
            "adjudicator_must_not_be_an_initial_reviewer": True,
        },
        "allowed_classifications": [
            "SUPPORTED",
            "ADDED",
            "ALTERED",
            "OMITTED",
            "INFERRED",
            "UNCERTAIN",
        ],
        "allowed_severities": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        "review_record_schema": {
            "reviewer_id": "",
            "credential_attestation_reference": "",
            "classification": None,
            "severity": None,
            "supporting_evidence": [],
            "rationale": "",
            "blinded_to_automation": True,
            "submitted_at": None,
        },
        "cases": cases,
        "required_metrics": [
            "classification_percent_agreement",
            "severity_percent_agreement",
            "classification_cohens_kappa",
            "severity_weighted_kappa",
            "gatekeeper_vs_adjudicated_precision_recall_f1",
        ],
        "completion_attestation": None,
    }
    CALIBRATION_TEMPLATE_PATH.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")


def build() -> list[dict[str, object]]:
    records = _base_records()
    records.extend(build_expansion_records())
    assert len(records) == 200
    pairs = {
        (
            record["input"]["source_documents"][0]["content"],  # type: ignore[index]
            record["input"]["generated_output"]["content"],  # type: ignore[index]
        )
        for record in records
    }
    assert len(pairs) == len(records)
    CASES_PATH.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_manifest(records)
    _write_splits(records)
    _write_calibration_template(records)
    return records


def main() -> None:
    records = build()
    print(f"Built {len(records)} engineering cases at {CASES_PATH}")


if __name__ == "__main__":
    main()

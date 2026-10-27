"""Build the frozen engineering semantic-evaluation subset.

The labels in this file are deliberately not described as clinical gold.  The
clinician-calibration split is an annotation queue until two independent
reviewers and adjudication have completed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark" / "clinical_faithfulness" / "v0"
DATA_PATH = BENCHMARK_DIR / "semantic-gold.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "semantic-gold-manifest.json"
DATASET_VERSION = "0.1.0"

RELATIONSHIPS = ("SUPPORTED", "INFERRED", "UNCERTAIN", "CONTRADICTED")
SPLITS = ("development", "validation", "held_out_test", "clinician_calibration")


@dataclass(frozen=True, slots=True)
class SemanticSpec:
    source: str
    generated: str
    relationship: str
    reason: str
    support_count: int = 1


def _supported() -> list[SemanticSpec]:
    pairs = [
        ("The patient reports shortness of breath.", "The patient has dyspnoea."),
        ("High blood pressure is documented.", "The patient has hypertension."),
        (
            "The patient previously had a heart attack.",
            "Prior myocardial infarction is documented.",
        ),
        ("There is swelling of both ankles.", "Bilateral ankle edema is present."),
        ("Take acetaminophen daily.", "Take paracetamol daily."),
        ("Take metformin twice daily.", "Take metformin BID."),
        ("Take metformin 0.5 g daily.", "Take metformin 500 mg daily."),
        ("The patient has left knee pain.", "The patient has Lt knee pain."),
        ("The patient denies fever.", "The patient has no fever."),
        ("History of asthma is documented.", "The patient has a prior history of asthma."),
        ("Cough has lasted two weeks.", "Cough has lasted a fortnight."),
        ("Take ciprofloxacin by mouth.", "Take ciprofloxacin PO."),
        ("Administer insulin subcutaneously.", "Administer insulin SC."),
        ("Myocardial infarction is documented.", "The patient had a heart attack."),
        ("No known drug allergies are documented.", "The patient has no drug allergies."),
        ("Possible pneumonia is documented.", "The patient may have pneumonia."),
    ]
    return [
        SemanticSpec(source, generated, "SUPPORTED", "SEMANTIC_OR_NORMALIZED_EQUIVALENCE")
        for source, generated in pairs
    ]


def _inferred() -> list[SemanticSpec]:
    triples = [
        ("Fever, cough, and a lobar infiltrate are present.", "The findings suggest pneumonia."),
        ("HbA1c is 10.1% despite metformin therapy.", "Diabetes remains poorly controlled."),
        ("Blood pressure is 178 mmHg despite treatment.", "Hypertension is poorly controlled."),
        (
            "Serum sodium is 122 mmol/L and dizziness is present.",
            "Symptoms may relate to hyponatremia.",
        ),
        (
            "Serum potassium is 6.3 mmol/L and palpitations are present.",
            "Hyperkalemia may be clinically significant.",
        ),
        ("Glucose is 320 mg/dL with fatigue.", "Hyperglycemia is likely."),
        ("INR is 5.4 while taking warfarin.", "Anticoagulation may be excessive."),
        ("Heart rate is 130 bpm with palpitations.", "Tachycardia is present."),
        ("Temperature is 39 C with chills.", "A febrile illness is present."),
        ("Oxygen saturation is 86% with dyspnea.", "Hypoxemia is likely."),
        ("Right calf pain and swelling began yesterday.", "Venous thrombosis is possible."),
        ("Crushing chest pain and nausea began suddenly.", "Acute coronary syndrome is possible."),
        ("Sudden weakness and headache are present.", "A neurologic event is possible."),
        ("Abdominal pain and vomiting are present.", "An acute abdominal process is possible."),
        ("Creatinine is 3.2 mg/dL with fatigue.", "Renal dysfunction is possible."),
        ("Rash began after amoxicillin was started.", "A drug reaction is possible."),
    ]
    return [
        SemanticSpec(source, generated, "INFERRED", "DERIVED_CLINICAL_SUMMARY", 2)
        for source, generated in triples
    ]


def _uncertain() -> list[SemanticSpec]:
    pairs = [
        (
            "Cancer is mentioned in the family history, but the relative is not named.",
            "The patient has cancer.",
        ),
        (
            "Chest pain was noted around a prior admission; timing is unclear.",
            "The patient currently has chest pain.",
        ),
        (
            "The note says 'she has asthma' after discussing the patient and her mother.",
            "The patient has asthma.",
        ),
        ("Pneumonia versus atelectasis remains under consideration.", "The patient has pneumonia."),
        (
            "The medication list contains two entries named insulin without dose linkage.",
            "Insulin 10 units is prescribed.",
        ),
        ("Pain improved after an unspecified intervention.", "Morphine relieved the pain."),
        (
            "A remote record mentions a positive test without a date.",
            "The test is currently positive.",
        ),
        (
            "The assessment references 'this condition' after listing asthma and pneumonia.",
            "Pneumonia is active.",
        ),
        (
            "The patient or sibling may have had a stroke; attribution is incomplete.",
            "The patient had a stroke.",
        ),
        ("A dose of 5 is recorded without a medication or unit.", "Warfarin 5 mg is prescribed."),
        (
            "The report describes a left-sided finding but does not identify the organ.",
            "The left kidney is abnormal.",
        ),
        (
            "Symptoms occurred after discharge, but the date of this note is unavailable.",
            "Symptoms are present today.",
        ),
        ("The source lists cough twice with conflicting onset dates.", "Cough began yesterday."),
        (
            "An allergy is mentioned, but the allergen is illegible.",
            "The patient is allergic to penicillin.",
        ),
        (
            "The result is described as borderline with no reference range.",
            "The laboratory result is abnormal.",
        ),
        (
            "The plan says to continue it, but the referenced medication is unclear.",
            "Continue metformin.",
        ),
    ]
    return [
        SemanticSpec(source, generated, "UNCERTAIN", "INSUFFICIENT_OR_AMBIGUOUS_EVIDENCE")
        for source, generated in pairs
    ]


def _contradicted() -> list[SemanticSpec]:
    pairs = [
        ("Possible pneumonia is documented.", "Pneumonia is confirmed."),
        ("The patient's mother had breast cancer.", "The patient had breast cancer."),
        ("The cough resolved last week.", "The cough is ongoing."),
        ("Symptoms improved after treatment.", "Symptoms worsened after treatment."),
        ("The disease is stable.", "The disease is rapidly progressing."),
        ("The culture was negative.", "The culture was positive."),
        ("No response to therapy was observed.", "The patient responded to therapy."),
        ("The lesion is benign.", "The lesion is malignant."),
        ("Medication was discontinued.", "Medication is being continued."),
        ("The procedure is planned for next week.", "The procedure was completed yesterday."),
        ("Pain occurs only with exertion.", "Pain occurs at rest."),
        ("The patient can bear weight.", "The patient cannot bear weight."),
        ("The infection has cleared.", "The infection remains active."),
        ("Renal function returned to baseline.", "Renal function continues to decline."),
        ("The finding is on the left.", "The finding is on the right."),
        ("The patient denies taking warfarin.", "The patient takes warfarin."),
    ]
    return [
        SemanticSpec(source, generated, "CONTRADICTED", "MATERIAL_MEANING_CONTRADICTION")
        for source, generated in pairs
    ]


def semantic_specs() -> list[SemanticSpec]:
    groups = (_supported(), _inferred(), _uncertain(), _contradicted())
    assert all(len(group) == 16 for group in groups)
    return [spec for group in groups for spec in group]


def build_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relationship_index, group in enumerate(
        (_supported(), _inferred(), _uncertain(), _contradicted())
    ):
        for group_index, spec in enumerate(group):
            ordinal = relationship_index * 16 + group_index + 1
            split = SPLITS[group_index % len(SPLITS)]
            support_ids = [
                f"sg-v0-{ordinal:03d}-source-{index + 1}" for index in range(spec.support_count)
            ]
            records.append(
                {
                    "schema_version": "1.0.0",
                    "case_id": f"sg-v0-{ordinal:03d}",
                    "source_evidence": spec.source,
                    "generated_claim": spec.generated,
                    "gold_relationship": spec.relationship,
                    "reason": spec.reason,
                    "supporting_span_ids": support_ids,
                    "annotation_status": "engineering",
                    "split": split,
                }
            )
    return records


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    records = build_records()
    data = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    DATA_PATH.write_text(data, encoding="utf-8")
    relationship_counts = Counter(str(record["gold_relationship"]) for record in records)
    split_ids = {
        split: [str(record["case_id"]) for record in records if record["split"] == split]
        for split in SPLITS
    }
    split_serialization = json.dumps(split_ids, sort_keys=True, separators=(",", ":")).encode()
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset_version": DATASET_VERSION,
        "status": "ENGINEERING_LABELS_PENDING_CLINICIAN_REVIEW",
        "case_count": len(records),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "splits": split_ids,
        "split_assignment_sha256": hashlib.sha256(split_serialization).hexdigest(),
        "content_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "held_out_policy": (
            "held_out_test cases must not be used to select thresholds, terminology, "
            "prompts, models, or rules; evaluate once after the candidate is frozen"
        ),
        "annotation_policy": (
            "All labels are engineering annotations. Clinician-review status may only be "
            "changed by the separate reviewer/adjudication workflow."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records, manifest


def main() -> int:
    records, _ = build()
    print(f"Built {len(records)} semantic evaluation cases at {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

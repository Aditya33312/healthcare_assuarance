"""Regression tests for deterministic two-way clinical comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.compare_engine import compare
from backend.app.compare_engine.semantic import (
    ClinicalSemanticCascade,
    ControlledJSONSemanticMatcher,
    SemanticDecision,
    SemanticLabel,
)
from backend.app.domain.enums import AssertionStatus, ClaimType, FindingType
from backend.app.domain.models import (
    Claim,
    CompareResult,
    Finding,
    GeneratedOutput,
    SourceBundle,
    SourceDocument,
    TextSpan,
)


@dataclass(frozen=True)
class ClaimSpec:
    needle: str
    concept: str
    type: ClaimType
    attributes: dict[str, Any] = field(default_factory=dict)
    assertion: AssertionStatus = AssertionStatus.PRESENT


class StubClinicalNLPService:
    def __init__(self, claims_by_text: dict[str, list[ClaimSpec]]) -> None:
        self.claims_by_text = claims_by_text

    def extract(self, text: str, document_id: str) -> list[Claim]:
        claims: list[Claim] = []
        for index, spec in enumerate(self.claims_by_text.get(text, [])):
            start = text.index(spec.needle)
            claims.append(
                Claim(
                    claim_id=f"{document_id}_{index}",
                    concept=spec.concept,
                    type=spec.type,
                    attributes=spec.attributes,
                    assertion=spec.assertion,
                    source_span=TextSpan(
                        document_id=document_id,
                        text=spec.needle,
                        start=start,
                        end=start + len(spec.needle),
                    ),
                )
            )
        return claims


class StubEmbeddingMatcher:
    version = "stub-biomedical-embedding@1"

    def __init__(self, similarity: float) -> None:
        self._similarity = similarity

    def similarity(self, source: Claim, generated: Claim) -> float:
        return self._similarity


class FailingEmbeddingMatcher:
    version = "failing-biomedical-embedding@1"

    def similarity(self, source: Claim, generated: Claim) -> float:
        raise RuntimeError(f"sensitive matcher payload: {source.concept} {generated.concept}")


class StubSemanticMatcher:
    version = "stub-clinical-nli@1"

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        return SemanticDecision(
            label=SemanticLabel.INFERRED,
            confidence=0.93,
            explanation="The controlled test matcher marks this as an inference.",
        )


class ConfigurableSemanticMatcher:
    version = "configurable-semantic@1.0.0"

    def __init__(self, label: SemanticLabel, confidence: float) -> None:
        self.label = label
        self.confidence = confidence

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        return SemanticDecision(self.label, self.confidence, "Controlled test decision.")


class TimeoutSemanticMatcher:
    version = "timeout-semantic@1.0.0"

    def compare(self, source: Claim, generated: Claim) -> SemanticDecision:
        raise TimeoutError("private provider detail")


class MalformedSemanticProvider:
    version = "malformed-provider@1.0.0"

    def evaluate(self, source_evidence: str, generated_claim: str) -> dict[str, object]:
        return {"relationship": "SUPPORTED"}


class FailingClinicalNLPService:
    def extract(self, text: str, document_id: str) -> list[Claim]:
        raise RuntimeError(f"sensitive source leaked: {document_id}: {text}")


class DegradedClinicalNLPService(StubClinicalNLPService):
    last_ner_error: str | None = None

    def extract(self, text: str, document_id: str) -> list[Claim]:
        claims = super().extract(text, document_id)
        self.last_ner_error = "NER_ADAPTER_FAILED:secret-model-path"
        return claims


def _single_reason(result: CompareResult, reason_code: str) -> Finding:
    findings = [finding for finding in result.findings if finding.reason_code == reason_code]
    assert len(findings) == 1
    return findings[0]


def test_extractor_failure_is_unassessed_without_persistable_exception_text() -> None:
    secret = "Patient Jane Doe has a confidential diagnosis."

    result = compare(
        source=secret,
        generated_output="Summary",
        nlp_service=FailingClinicalNLPService(),
    )

    assert result.execution_status.value == "FAILED"
    assert result.coverage.status.value == "UNASSESSED"
    rendered = " ".join(result.coverage.limitations)
    assert rendered == "Clinical claim extraction failed (EXTRACTOR_FAILED)."
    assert secret not in rendered
    assert "Jane Doe" not in rendered


def test_optional_ner_failure_reduces_coverage_without_losing_rule_findings() -> None:
    source = "Patient has hypertension."
    output = "Hypertension is documented."
    service = DegradedClinicalNLPService(
        {
            source: [ClaimSpec("hypertension", "hypertension", ClaimType.CONDITION)],
            output: [ClaimSpec("Hypertension", "hypertension", ClaimType.CONDITION)],
        }
    )

    result = compare(source, output, nlp_service=service)

    assert _single_reason(result, "CLAIM_SUPPORTED").type is FindingType.SUPPORTED
    assert result.coverage.status.value == "REDUCED"
    assert result.coverage.claims_unassessed == 1
    rendered = " ".join(result.coverage.limitations)
    assert rendered == (
        "Optional clinical NER failed (NER_ADAPTER_FAILED); deterministic extraction continued."
    )
    assert "secret-model-path" not in rendered


def test_explicit_inference_cue_is_uncertain_without_semantic_evaluator() -> None:
    source = "Fever and productive cough are present."
    output = "The findings are consistent with pneumonia."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec("Fever", "fever", ClaimType.SYMPTOM),
                ClaimSpec("productive cough", "cough", ClaimType.SYMPTOM),
            ],
            output: [ClaimSpec("pneumonia", "pneumonia", ClaimType.CONDITION)],
        }
    )

    result = compare(source, output, nlp_service=service, enable_semantic=False)

    assert result.coverage.status.value == "REDUCED"
    assert result.coverage.claims_unassessed == 3
    assert {finding.type for finding in result.findings} == {FindingType.UNCERTAIN}
    assert not any(
        finding.type in {FindingType.ADDED, FindingType.OMITTED} for finding in result.findings
    )
    assert "No bounded inference rule resolved" in " ".join(result.coverage.limitations)


def test_supported_fact_has_exact_source_and_generated_evidence() -> None:
    source = "Patient has hypertension."
    output = "Hypertension is documented."
    service = StubClinicalNLPService(
        {
            source: [ClaimSpec("hypertension", "hypertension", ClaimType.CONDITION)],
            output: [ClaimSpec("Hypertension", "hypertension", ClaimType.CONDITION)],
        }
    )

    result = compare(source=source, generated_output=output, nlp_service=service)

    finding = _single_reason(result, "CLAIM_SUPPORTED")
    assert finding.type is FindingType.SUPPORTED
    assert finding.source is not None and finding.source.text == "hypertension"
    assert finding.generated is not None and finding.generated.text == "Hypertension"
    assert source[finding.source.start : finding.source.end] == finding.source.text
    assert output[finding.generated.start : finding.generated.end] == finding.generated.text
    assert result.faithfulness.supported == 1
    assert result.faithfulness.score == 100.0


def test_dose_change_is_high_risk_alteration() -> None:
    source = "Metformin 500 mg twice daily."
    output = "Metformin 1000 mg twice daily."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "Metformin 500 mg twice daily",
                    "metformin",
                    ClaimType.MEDICATION,
                    {"dose": 500, "unit": "mg", "frequency": "TWICE_DAILY"},
                )
            ],
            output: [
                ClaimSpec(
                    "Metformin 1000 mg twice daily",
                    "metformin",
                    ClaimType.MEDICATION,
                    {"dose": 1000, "unit": "mg", "frequency": "TWICE_DAILY"},
                )
            ],
        }
    )

    result = compare(source, output, nlp_service=service)

    finding = _single_reason(result, "DOSE_MISMATCH")
    assert (finding.type.value, finding.category, finding.severity.value) == (
        "ALTERED",
        "medication_dose",
        "HIGH",
    )
    assert finding.rule_id == "DOSE_MISMATCH"
    assert finding.rule_version == "1.0.0"


def test_laterality_change_meets_definition_of_done() -> None:
    source = "Patient reports left knee pain."
    output = "Patient reports right knee pain."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "left knee pain",
                    "knee pain",
                    ClaimType.SYMPTOM,
                    {"laterality": "LEFT"},
                )
            ],
            output: [
                ClaimSpec(
                    "right knee pain",
                    "knee pain",
                    ClaimType.SYMPTOM,
                    {"laterality": "RIGHT"},
                )
            ],
        }
    )

    result = compare(source=source, generated_output=output, nlp_service=service)

    finding = _single_reason(result, "LATERALITY_MISMATCH")
    assert (finding.type.value, finding.category) == ("ALTERED", "laterality")
    assert finding.source is not None and finding.source.text == "left knee pain"
    assert finding.generated is not None and finding.generated.text == "right knee pain"


def test_definition_of_done_uses_public_clinical_nlp_service() -> None:
    source = "Patient reports left knee pain."
    output = "Patient reports right knee pain."

    result = compare(source=source, generated_output=output)

    finding = _single_reason(result, "LATERALITY_MISMATCH")
    assert (finding.type.value, finding.category) == ("ALTERED", "laterality")
    assert finding.source is not None and finding.source.text == "left"
    assert finding.generated is not None and finding.generated.text == "right"


def test_frequency_change_is_not_sent_to_semantic_fallback() -> None:
    source = "Ibuprofen 400 mg twice daily."
    output = "Ibuprofen 400 mg three times daily."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "Ibuprofen 400 mg twice daily",
                    "ibuprofen",
                    ClaimType.MEDICATION,
                    {"dose": 400, "unit": "mg", "frequency": "BID"},
                )
            ],
            output: [
                ClaimSpec(
                    "Ibuprofen 400 mg three times daily",
                    "ibuprofen",
                    ClaimType.MEDICATION,
                    {"dose": 400, "unit": "mg", "frequency": "TID"},
                )
            ],
        }
    )

    result = compare(source, output, nlp_service=service)

    finding = _single_reason(result, "FREQUENCY_MISMATCH")
    assert (finding.type.value, finding.category) == (
        "ALTERED",
        "medication_frequency",
    )
    assert finding.alignment_method.value == "EXACT"


def test_negation_reversal_is_altered_assertion() -> None:
    source = "No chest pain."
    output = "The patient has chest pain."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "chest pain",
                    "chest pain",
                    ClaimType.SYMPTOM,
                    assertion=AssertionStatus.ABSENT,
                )
            ],
            output: [ClaimSpec("chest pain", "chest pain", ClaimType.SYMPTOM)],
        }
    )

    result = compare(source, output, nlp_service=service)

    finding = _single_reason(result, "NEGATION_REVERSAL")
    assert (finding.type.value, finding.category) == ("ALTERED", "assertion")


def test_subject_and_lab_rules_use_canonical_finding_categories() -> None:
    subject = compare(
        "The patient's mother has asthma.",
        "The patient has asthma.",
    )
    lab = compare(
        "Serum potassium is 4.1 mmol/L.",
        "Serum potassium is 6.1 mmol/L.",
    )

    assert _single_reason(subject, "SUBJECT_ATTRIBUTION_ERROR").category == ("subject_attribution")
    assert _single_reason(lab, "NUMERIC_VALUE_MISMATCH").category == "lab_value"


def test_supported_high_risk_dimensions_are_named_when_explicitly_preserved() -> None:
    possible = compare("Possible pneumonia.", "The patient may have pneumonia.")
    family = compare(
        "The patient's mother had breast cancer.",
        "There is a maternal family history of breast cancer.",
    )
    duration = compare("Cough for a fortnight.", "Cough for two weeks.")

    assert _single_reason(possible, "CLAIM_SUPPORTED").category == "certainty"
    assert _single_reason(family, "CLAIM_SUPPORTED").category == "subject_attribution"
    assert _single_reason(duration, "CLAIM_SUPPORTED").category == "duration"


def test_explicit_unique_medication_action_slot_detects_substitution() -> None:
    result = compare(
        "The patient takes metformin.",
        "The patient takes glipizide.",
    )

    finding = _single_reason(result, "MEDICATION_MISMATCH")
    assert (finding.type, finding.category) == (
        FindingType.ALTERED,
        "medication_identity",
    )


def test_singleton_medications_without_shared_structural_context_are_not_guessed() -> None:
    no_context = compare("Metformin.", "Glipizide.")
    conflicting_actions = compare("Stop metformin.", "Start glipizide.")

    for result in (no_context, conflicting_actions):
        assert not any(finding.reason_code == "MEDICATION_MISMATCH" for finding in result.findings)
        assert {finding.type for finding in result.findings} == {
            FindingType.ADDED,
            FindingType.OMITTED,
        }


def test_source_to_output_pass_detects_omission() -> None:
    source = "Severe penicillin allergy."
    output = "No allergy information was included."
    service = StubClinicalNLPService(
        {
            source: [ClaimSpec("penicillin allergy", "penicillin", ClaimType.ALLERGY)],
            output: [],
        }
    )

    result = compare(source, output, nlp_service=service)

    finding = _single_reason(result, "SOURCE_CLAIM_OMITTED")
    assert (finding.type.value, finding.category, finding.severity.value) == (
        "OMITTED",
        "allergy",
        "HIGH",
    )
    assert finding.source is not None
    assert finding.generated is None


def test_output_to_source_pass_detects_addition() -> None:
    source = "Patient reports knee pain."
    output = "Patient reports knee pain and fever."
    service = StubClinicalNLPService(
        {
            source: [ClaimSpec("knee pain", "knee pain", ClaimType.SYMPTOM)],
            output: [
                ClaimSpec("knee pain", "knee pain", ClaimType.SYMPTOM),
                ClaimSpec("fever", "fever", ClaimType.SYMPTOM),
            ],
        }
    )

    result = compare(source, output, nlp_service=service)

    finding = _single_reason(result, "UNSUPPORTED_ADDITION")
    assert (finding.type.value, finding.category) == ("ADDED", "symptom")
    assert result.faithfulness.supported == 1
    assert result.faithfulness.added == 1


def test_structured_inputs_and_brief_example_produce_three_alterations() -> None:
    source = "Patient reports left knee pain. No trauma. Ibuprofen 400 mg twice daily."
    output = "Patient reports right knee pain after trauma. Ibuprofen 800 mg twice daily."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "left knee pain",
                    "knee pain",
                    ClaimType.SYMPTOM,
                    {"laterality": "LEFT"},
                ),
                ClaimSpec(
                    "trauma",
                    "trauma",
                    ClaimType.CLINICAL_FACT,
                    assertion=AssertionStatus.ABSENT,
                ),
                ClaimSpec(
                    "Ibuprofen 400 mg twice daily",
                    "ibuprofen",
                    ClaimType.MEDICATION,
                    {"dose": 400, "unit": "mg", "frequency": "TWICE_DAILY"},
                ),
            ],
            output: [
                ClaimSpec(
                    "right knee pain",
                    "knee pain",
                    ClaimType.SYMPTOM,
                    {"laterality": "RIGHT"},
                ),
                ClaimSpec("trauma", "trauma", ClaimType.CLINICAL_FACT),
                ClaimSpec(
                    "Ibuprofen 800 mg twice daily",
                    "ibuprofen",
                    ClaimType.MEDICATION,
                    {"dose": 800, "unit": "mg", "frequency": "TWICE_DAILY"},
                ),
            ],
        }
    )
    bundle = SourceBundle(documents=[SourceDocument(document_id="brief_source", content=source)])

    result = compare(
        bundle,
        GeneratedOutput(type="clinical_note", content=output),
        nlp_service=service,
    )

    assert {finding.reason_code for finding in result.findings} == {
        "LATERALITY_MISMATCH",
        "NEGATION_REVERSAL",
        "DOSE_MISMATCH",
    }
    assert all(finding.type is FindingType.ALTERED for finding in result.findings)
    assert result.faithfulness.altered == 3


def test_equivalent_mass_units_are_supported() -> None:
    source = "Drug X 0.5 g daily."
    output = "Drug X 500 mg daily."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "Drug X 0.5 g daily",
                    "drug x",
                    ClaimType.MEDICATION,
                    {"dose": 0.5, "unit": "g", "frequency": "ONCE_DAILY"},
                )
            ],
            output: [
                ClaimSpec(
                    "Drug X 500 mg daily",
                    "drug x",
                    ClaimType.MEDICATION,
                    {"dose": 500, "unit": "mg", "frequency": "ONCE_DAILY"},
                )
            ],
        }
    )

    result = compare(source, output, nlp_service=service)

    assert [finding.reason_code for finding in result.findings] == ["CLAIM_SUPPORTED"]


def test_complete_clinical_numeric_signatures_take_deterministic_precedence() -> None:
    cases = (
        ("BP is 120/80 mmHg.", "BP is 120/100 mmHg.", "120/80 mmHg", "120/100 mmHg"),
        ("SpO2 is 91.25%.", "SpO2 is 98.25%.", "91.25%", "98.25%"),
        (
            "Temperature is \N{MINUS SIGN}2.125 °C.",
            "Temperature is 2.125 °C.",
            "\N{MINUS SIGN}2.125 °C",
            "2.125 °C",
        ),
        ("INR is 2.0-3.0.", "INR is 2 to 4.", "2.0-3.0", "2 to 4"),
    )

    for source, output, source_evidence, generated_evidence in cases:
        result = compare(source, output)
        finding = _single_reason(result, "NUMERIC_VALUE_MISMATCH")

        assert finding.type is FindingType.ALTERED
        assert finding.source is not None and finding.source.text == source_evidence
        assert finding.generated is not None and finding.generated.text == generated_evidence
        assert result.coverage.status.value == "FULL"
        assert result.faithfulness.altered == 1


def test_equivalent_numeric_ranges_are_supported() -> None:
    result = compare("INR is 2.0-3.0.", "INR is 2 to 3.")

    assert [finding.reason_code for finding in result.findings] == ["CLAIM_SUPPORTED"]
    assert result.faithfulness.supported == 1


def test_incompatible_lab_units_fail_closed_as_an_alteration() -> None:
    result = compare(
        "Creatinine is 1.8 mg/dL.",
        "Creatinine is 1.8 mmol/L.",
    )

    finding = _single_reason(result, "LAB_UNIT_MISMATCH")
    assert (finding.type, finding.category, finding.severity.value) == (
        FindingType.ALTERED,
        "lab_value",
        "HIGH",
    )
    assert finding.source is not None and finding.source.text == "1.8 mg/dL"
    assert finding.generated is not None and finding.generated.text == "1.8 mmol/L"


def test_compatible_lab_and_medication_unit_conversions_remain_supported() -> None:
    cases = (
        ("Hemoglobin is 0.014 g/dL.", "Hemoglobin is 14 mg/dL."),
        ("Metformin 0.5 g daily.", "Metformin 500 mg daily."),
        ("Metformin 1000 mcg daily.", "Metformin 1 mg daily."),
    )

    for source, output in cases:
        result = compare(source, output)

        assert [finding.reason_code for finding in result.findings] == ["CLAIM_SUPPORTED"]
        assert result.coverage.status.value == "FULL"


def test_ambiguous_alignment_remains_uncertain() -> None:
    source = "Knee pain today. Knee pain yesterday."
    output = "Knee pain persists."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec("Knee pain", "knee pain", ClaimType.SYMPTOM),
                ClaimSpec("Knee pain", "knee pain", ClaimType.SYMPTOM),
            ],
            output: [ClaimSpec("Knee pain", "knee pain", ClaimType.SYMPTOM)],
        }
    )

    result = compare(source, output, nlp_service=service)

    assert result.faithfulness.uncertain == 3
    assert not any(finding.type is FindingType.SUPPORTED for finding in result.findings)
    assert all(finding.reason_code == "ALIGNMENT_UNCERTAIN" for finding in result.findings)


def test_embedding_alignment_uses_auditable_acceptance_threshold() -> None:
    source = "Pulmonary infection is present."
    output = "There is a lung disorder."
    service = StubClinicalNLPService(
        {
            source: [ClaimSpec("Pulmonary infection", "pulmonary infection", ClaimType.CONDITION)],
            output: [ClaimSpec("lung disorder", "lung disorder", ClaimType.CONDITION)],
        }
    )

    result = compare(
        source,
        output,
        nlp_service=service,
        embedding_matcher=StubEmbeddingMatcher(0.95),
    )

    finding = _single_reason(result, "CLAIM_SUPPORTED")
    assert finding.alignment_method.value == "EMBEDDING"
    assert finding.confidence == 0.95


def test_optional_matcher_failure_does_not_persist_exception_text() -> None:
    source = "Pulmonary infection is present."
    output = "There is a lung disorder."
    service = StubClinicalNLPService(
        {
            source: [ClaimSpec("Pulmonary infection", "secret source", ClaimType.CONDITION)],
            output: [ClaimSpec("lung disorder", "secret output", ClaimType.CONDITION)],
        }
    )

    result = compare(
        source,
        output,
        nlp_service=service,
        embedding_matcher=FailingEmbeddingMatcher(),
    )

    rendered = " ".join(result.coverage.limitations)
    assert "Embedding matcher failed (EMBEDDING_MATCHER_FAILED)." in rendered
    assert "secret source" not in rendered
    assert "secret output" not in rendered


def test_semantic_fallback_can_label_a_controlled_inference() -> None:
    source = "Chronic kidney disease is documented."
    output = "Renal impairment is present."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "Chronic kidney disease",
                    "chronic kidney disease",
                    ClaimType.CONDITION,
                )
            ],
            output: [ClaimSpec("Renal impairment", "renal impairment", ClaimType.CONDITION)],
        }
    )

    result = compare(
        source,
        output,
        nlp_service=service,
        semantic_matcher=StubSemanticMatcher(),
    )

    finding = _single_reason(result, "CLAIM_INFERRED")
    assert finding.type is FindingType.INFERRED
    assert finding.alignment_method.value == "SEMANTIC"


def test_default_semantic_cascade_supports_clinical_synonym() -> None:
    result = compare(
        "The patient reports shortness of breath.",
        "The patient has dyspnoea.",
    )

    finding = _single_reason(result, "CLAIM_SUPPORTED")
    assert finding.type is FindingType.SUPPORTED
    assert finding.category == "symptom"


def test_bounded_multi_claim_inference_records_identifiable_source_support() -> None:
    result = compare(
        "Fever and cough are present.",
        "The findings suggest pneumonia.",
    )

    finding = _single_reason(result, "DERIVED_CLINICAL_SUMMARY")
    assert finding.type is FindingType.INFERRED
    assert finding.category == "inference"
    assert {span.text.casefold() for span in finding.source_support} == {"fever", "cough"}
    assert finding.generated is not None and finding.generated.text.casefold() == "pneumonia"


def test_unsupported_and_ambiguous_inference_fail_closed_to_uncertain() -> None:
    unsupported = compare(
        "The patient has headache.",
        "A cardiopulmonary cause is possible.",
    )
    ambiguous = compare(
        "The source documents asthma.",
        "An unspecified clinical relationship is possible.",
    )

    for result in (unsupported, ambiguous):
        finding = _single_reason(result, "SEMANTIC_INFERENCE_UNASSESSED")
        assert finding.type is FindingType.UNCERTAIN
        assert finding.uncertainty_reason == "INFERENCE_EVIDENCE_UNRESOLVED"
        assert finding.candidate_evidence
        assert result.coverage.status.value == "REDUCED"


def test_explicit_attribution_and_temporal_ambiguity_are_uncertain() -> None:
    attribution = compare(
        "Cancer is in the family history, but the relative is unclear.",
        "The patient has cancer.",
    )
    temporal = compare(
        "Chest pain occurred around a prior admission; timing is unclear.",
        "The patient currently has chest pain.",
    )

    for result in (attribution, temporal):
        finding = _single_reason(result, "EXPLICIT_SOURCE_AMBIGUITY")
        assert finding.type is FindingType.UNCERTAIN
        assert finding.uncertainty_reason == "EXPLICIT_SOURCE_AMBIGUITY"
        assert finding.candidate_evidence


def test_semantic_context_contradiction_is_altered() -> None:
    result = compare(
        "Symptoms improved after treatment.",
        "Symptoms worsened after treatment.",
    )

    finding = _single_reason(result, "SEMANTIC_CONTRADICTION")
    assert finding.type is FindingType.ALTERED
    assert finding.category == "clinical_fact"


def test_low_confidence_semantic_disagreement_becomes_uncertain() -> None:
    cascade = ClinicalSemanticCascade(
        nli_matcher=ConfigurableSemanticMatcher(SemanticLabel.EQUIVALENT, 0.7),
        controlled_judge=ConfigurableSemanticMatcher(SemanticLabel.INFERRED, 0.92),
    )
    source = "Chronic renal disease is documented."
    output = "A chronic renal disorder is documented."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec("Chronic renal disease", "chronic renal disease", ClaimType.CONDITION)
            ],
            output: [
                ClaimSpec("chronic renal disorder", "chronic renal disorder", ClaimType.CONDITION)
            ],
        }
    )

    result = compare(source, output, nlp_service=service, semantic_matcher=cascade)

    assert {finding.type for finding in result.findings} == {FindingType.UNCERTAIN}
    assert result.coverage.status.value == "REDUCED"


def test_semantic_timeout_and_malformed_controlled_response_fail_closed() -> None:
    timeout_cascade = ClinicalSemanticCascade(nli_matcher=TimeoutSemanticMatcher())
    malformed_judge = ControlledJSONSemanticMatcher(MalformedSemanticProvider())
    malformed_cascade = ClinicalSemanticCascade(
        nli_matcher=ConfigurableSemanticMatcher(SemanticLabel.UNCERTAIN, 0.6),
        controlled_judge=malformed_judge,
    )
    source = "Chronic renal disease is documented."
    output = "A chronic renal disorder is documented."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec("Chronic renal disease", "chronic renal disease", ClaimType.CONDITION)
            ],
            output: [
                ClaimSpec("chronic renal disorder", "chronic renal disorder", ClaimType.CONDITION)
            ],
        }
    )

    for cascade in (timeout_cascade, malformed_cascade):
        result = compare(source, output, nlp_service=service, semantic_matcher=cascade)
        assert {finding.type for finding in result.findings} == {FindingType.UNCERTAIN}
        assert result.coverage.status.value == "REDUCED"


def test_hard_dose_contradiction_overrides_semantic_equivalence() -> None:
    source = "Insulin 10 units nightly."
    output = "Insulin 100 units nightly."
    service = StubClinicalNLPService(
        {
            source: [
                ClaimSpec(
                    "Insulin 10 units nightly",
                    "insulin",
                    ClaimType.MEDICATION,
                    {"dose": 10, "unit": "unit"},
                )
            ],
            output: [
                ClaimSpec(
                    "Insulin 100 units nightly",
                    "insulin",
                    ClaimType.MEDICATION,
                    {"dose": 100, "unit": "unit"},
                )
            ],
        }
    )

    result = compare(
        source,
        output,
        nlp_service=service,
        semantic_matcher=ConfigurableSemanticMatcher(SemanticLabel.EQUIVALENT, 0.99),
    )

    finding = _single_reason(result, "DOSE_MISMATCH")
    assert finding.type is FindingType.ALTERED
    assert finding.severity.value == "CRITICAL"

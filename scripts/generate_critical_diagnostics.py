"""Generate the deterministic CRITICAL-case pipeline diagnostic artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.clinical_nlp.entities import ExtractedEntity
from backend.app.clinical_nlp.parsers import DeterministicEntityParser
from backend.app.compare_engine.alignment import ClaimAligner
from backend.app.compare_engine.engine import _comparison_claims, compare
from backend.app.compare_engine.rules import RULES, compare_aligned_claims
from backend.app.domain.enums import Severity
from backend.app.domain.models import Claim, Finding
from backend.app.severity import ProvisionalSeverityPolicy, SeverityContext

CORPUS = Path("benchmark/clinical_faithfulness/v0/cases.jsonl")
MANIFEST = Path("benchmark/clinical_faithfulness/v0/manifest.json")
OUTPUT = Path("benchmark/clinical_faithfulness/v0/critical-diagnostics.json")

_BASELINE_PREDICTIONS: dict[str, dict[str, object]] = {
    "cf-v0-007": {
        "finding_type": "ALTERED",
        "category": "medication_dose",
        "reason_code": "DOSE_MISMATCH",
        "severity": "HIGH",
        "discrepancy_detected": True,
        "severity_matched": False,
    },
    "cf-v0-026": {
        "finding_type": "ALTERED",
        "category": "lab_value",
        "reason_code": "NUMERIC_VALUE_MISMATCH",
        "severity": "HIGH",
        "discrepancy_detected": True,
        "severity_matched": False,
    },
    "cf-v0-035": {
        "finding_type": "OMITTED",
        "category": "allergy",
        "reason_code": "SOURCE_CLAIM_OMITTED",
        "severity": "HIGH",
        "discrepancy_detected": True,
        "severity_matched": False,
    },
    "cf-v0-037": {
        "finding_type": "OMITTED",
        "category": "symptom",
        "reason_code": "SOURCE_CLAIM_OMITTED",
        "severity": "MODERATE",
        "discrepancy_detected": True,
        "severity_matched": False,
    },
}


def _json_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads(
            json.dumps(
                attributes,
                default=lambda value: getattr(value, "value", str(value)),
            )
        ),
    )


def _entity_dict(entity: ExtractedEntity) -> dict[str, object]:
    return {
        "type": entity.type.value,
        "concept": entity.concept,
        "text": entity.text,
        "start": entity.start,
        "end": entity.end,
        "attributes": _json_attributes(entity.attributes),
        "confidence": entity.confidence,
    }


def _claim_dict(claim: Claim) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "type": claim.type.value,
        "concept": claim.concept,
        "attributes": claim.attributes,
        "assertion": claim.assertion.value,
        "subject": claim.subject.value,
        "span": claim.source_span.model_dump(mode="json"),
        "confidence": claim.confidence,
        "extractor": claim.extractor,
        "extractor_version": claim.extractor_version,
    }


def _finding_dict(finding: Finding) -> dict[str, object]:
    return {
        "finding_type": finding.type.value,
        "category": finding.category,
        "reason_code": finding.reason_code,
        "severity": finding.severity.value,
        "source": finding.source.model_dump(mode="json") if finding.source else None,
        "generated": (finding.generated.model_dump(mode="json") if finding.generated else None),
        "confidence": finding.confidence,
        "rule_id": finding.rule_id,
        "rule_version": finding.rule_version,
        "alignment_method": finding.alignment_method.value,
    }


def _claim_for_finding(
    finding: Finding,
    claims: list[Claim],
    *,
    source_side: bool,
) -> Claim | None:
    evidence = finding.source if source_side else finding.generated
    if evidence is None:
        return None
    by_id = {claim.claim_id: claim for claim in claims}
    candidates = [
        claim
        for claim in claims
        if claim.source_span.document_id == evidence.document_id
        and claim.source_span.start < evidence.end
        and evidence.start < claim.source_span.end
    ]
    if not candidates:
        return None
    exact = next(
        (
            claim
            for claim in candidates
            if claim.source_span.start == evidence.start and claim.source_span.end == evidence.end
        ),
        candidates[0],
    )
    parent_id = exact.attributes.get("parent_claim_id")
    return by_id.get(str(parent_id), exact) if parent_id is not None else exact


def _severity_trace(
    finding: Finding,
    source_claims: list[Claim],
    generated_claims: list[Claim],
) -> dict[str, object]:
    source_claim = _claim_for_finding(finding, source_claims, source_side=True)
    generated_claim = _claim_for_finding(finding, generated_claims, source_side=False)
    source_value: object | None = None
    generated_value: object | None = None
    if source_claim is not None and generated_claim is not None:
        for outcome in compare_aligned_claims(source_claim, generated_claim):
            if outcome.rule.rule_id == finding.rule_id:
                source_value = outcome.source_value
                generated_value = outcome.generated_value
                break
    context = SeverityContext(
        finding_type=finding.type,
        category=finding.category,
        claim_type=source_claim.type if source_claim is not None else None,
        reason_code=finding.reason_code,
        source_concept=source_claim.concept if source_claim is not None else None,
        generated_concept=(generated_claim.concept if generated_claim is not None else None),
        source_attributes=(source_claim.attributes if source_claim is not None else {}),
        generated_attributes=(generated_claim.attributes if generated_claim is not None else {}),
        source_value=source_value,
        generated_value=generated_value,
    )
    policy = ProvisionalSeverityPolicy()
    decision = policy.decision_for(context)
    return {
        "predicted_severity": finding.severity.value,
        "policy": f"{policy.policy_id}@{policy.version}",
        "policy_rule": decision.rule_id,
        "context": {
            "source_concept": context.source_concept,
            "generated_concept": context.generated_concept,
            "source_value": context.source_value,
            "generated_value": context.generated_value,
            "source_attributes": dict(context.source_attributes),
            "generated_attributes": dict(context.generated_attributes),
        },
    }


def _alignment_trace(source_claims: list[Claim], generated_claims: list[Claim]) -> dict[str, Any]:
    source = _comparison_claims(source_claims)
    generated = _comparison_claims(generated_claims)
    plan = ClaimAligner().align(source, generated)
    selected = [
        {
            "source_claim_id": source[item.source_index].claim_id,
            "generated_claim_id": generated[item.generated_index].claim_id,
            "method": item.method.value,
            "confidence": item.confidence,
            "explanation": item.explanation,
        }
        for item in plan.alignments
    ]
    uncertain = [
        {
            "generated_claim_id": (
                generated[item.generated_index].claim_id
                if item.generated_index is not None
                else None
            ),
            "source_claim_ids": [source[index].claim_id for index in item.source_indices],
            "rejected_candidate_reason": item.explanation,
        }
        for item in plan.uncertain
    ]
    return {
        "candidate_pairs": selected + uncertain,
        "selected_pairs": selected,
        "unaligned_source_claim_ids": [source[index].claim_id for index in plan.unaligned_source],
        "unaligned_generated_claim_ids": [
            generated[index].claim_id for index in plan.unaligned_generated
        ],
        "rejected_candidate_reasons": [item["rejected_candidate_reason"] for item in uncertain]
        or (["NO_GENERATED_COMPARISON_CLAIMS"] if source and not generated else []),
        "limitations": list(plan.limitations),
    }


def _diagnostic_case(record: dict[str, Any]) -> dict[str, Any]:
    source_text = record["input"]["source_documents"][0]["content"]
    generated_text = record["input"]["generated_output"]["content"]
    parser = DeterministicEntityParser()
    source_entities = parser.parse(source_text)
    generated_entities = parser.parse(generated_text)
    service = ClinicalNLPService()
    source_claims = service.extract(source_text, "source_001")
    generated_claims = service.extract(generated_text, "generated_output")
    result = compare(source_text, generated_text)
    gold = [
        finding
        for finding in record["expectation"]["expected_findings"]
        if finding["severity"] == Severity.CRITICAL.value
    ]
    current = [_finding_dict(finding) for finding in result.findings]
    baseline = _BASELINE_PREDICTIONS.get(record["test_case_id"])
    current_critical = [
        finding for finding in result.findings if finding.severity is Severity.CRITICAL
    ]
    return {
        "case_id": record["test_case_id"],
        "source_text": source_text,
        "generated_output": generated_text,
        "gold_findings": [
            {
                "finding_type": finding["type"],
                "category": finding["category"],
                "reason_code": finding["reason_code"],
                "severity": finding["severity"],
                "source": finding.get("source"),
                "generated": finding.get("generated"),
            }
            for finding in gold
        ],
        "baseline_prediction": baseline or {"status": "NOT_IN_0.1.0_BASELINE"},
        "baseline_primary_root_cause": ("SEVERITY_MAPPING_ERROR" if baseline is not None else None),
        "current_predictions": current,
        "current_discrepancy_detected": bool(result.findings),
        "current_critical_severity_detected": bool(current_critical),
        "pipeline_trace": {
            "source_extraction": {
                "entities": [_entity_dict(entity) for entity in source_entities],
                "claims": [_claim_dict(claim) for claim in source_claims],
            },
            "output_extraction": {
                "entities": [_entity_dict(entity) for entity in generated_entities],
                "claims": [_claim_dict(claim) for claim in generated_claims],
            },
            "alignment": _alignment_trace(source_claims, generated_claims),
            "rule_engine": {
                "rule_set": result.versions.rule_set,
                "rules_evaluated": [
                    {
                        "rule_id": rule.rule_id,
                        "version": rule.version,
                        "fired": any(
                            finding.rule_id == rule.rule_id for finding in result.findings
                        ),
                    }
                    for rule in RULES
                ],
                "fired_results": [
                    _finding_dict(finding)
                    for finding in result.findings
                    if finding.rule_id is not None
                ],
            },
            "semantic_evaluator": {
                "called": False,
                "result": None,
                "confidence": None,
                "version": result.versions.semantic_evaluator,
            },
            "finding_classifier": current,
            "severity_engine": [
                _severity_trace(finding, source_claims, generated_claims)
                for finding in result.findings
            ],
            "coverage": result.coverage.model_dump(mode="json"),
        },
    }


def build_artifact() -> dict[str, Any]:
    """Build diagnostics without writing, for byte-for-byte verification."""

    records = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    critical_records = [
        record
        for record in records
        if any(
            finding["severity"] == Severity.CRITICAL.value
            for finding in record["expectation"]["expected_findings"]
        )
    ]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0.0",
        "benchmark_version": manifest["benchmark_version"],
        "benchmark_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "baseline_engine_version": "0.1.0",
        "current_engine_version": "0.1.0",
        "baseline_critical_gold_findings": len(_BASELINE_PREDICTIONS),
        "baseline_critical_discrepancy_detection_recall": 1.0,
        "baseline_critical_severity_recall": 0.0,
        "root_cause_counts": {"SEVERITY_MAPPING_ERROR": 4},
        "cases": [_diagnostic_case(record) for record in critical_records],
    }


def main() -> int:
    OUTPUT.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

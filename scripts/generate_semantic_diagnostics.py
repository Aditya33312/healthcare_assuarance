"""Generate the pre-semantic normalized error-analysis snapshot.

This artifact intentionally describes the checked v0.3 deterministic-only
benchmark result.  It is a baseline input to semantic design, not a claim that
the engineering annotations are clinician-reviewed gold.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.clinical_nlp.parsers import DeterministicEntityParser
from backend.app.compare_engine.engine import compare
from backend.app.compare_engine.rules import RULES
from scripts.generate_critical_diagnostics import (
    _alignment_trace,
    _claim_dict,
    _entity_dict,
    _finding_dict,
)

CORPUS = Path("benchmark/clinical_faithfulness/v0/cases.jsonl")
MANIFEST = Path("benchmark/clinical_faithfulness/v0/manifest.json")
BENCHMARK_RESULT = Path("benchmark/clinical_faithfulness/v0/benchmark-result.json")
OUTPUT = Path("benchmark/clinical_faithfulness/v0/semantic-diagnostics.json")

ROOT_CAUSES = {
    "CORRECT_PARAPHRASE_MISCLASSIFIED",
    "SEMANTIC_EQUIVALENCE_MISSING",
    "INFERENCE_UNSUPPORTED_BY_ENGINE",
    "AMBIGUITY_UNSUPPORTED_BY_ENGINE",
    "ALIGNMENT_ERROR",
    "CLAIM_EXTRACTION_ERROR",
    "ASSERTION_ERROR",
    "NORMALIZATION_ERROR",
    "RULE_FALSE_POSITIVE",
    "RULE_FALSE_NEGATIVE",
    "TAXONOMY_MAPPING_ERROR",
    "SEVERITY_ERROR",
    "BENCHMARK_ANNOTATION_QUESTIONABLE",
    "OTHER",
}


def _unmatched(
    expected: list[list[str]], actual: list[list[str]]
) -> tuple[list[list[str]], list[list[str]]]:
    expected_counter = Counter(tuple(label) for label in expected)
    actual_counter = Counter(tuple(label) for label in actual)
    false_negatives = [
        list(label)
        for label, count in sorted((expected_counter - actual_counter).items())
        for _ in range(count)
    ]
    false_positives = [
        list(label)
        for label, count in sorted((actual_counter - expected_counter).items())
        for _ in range(count)
    ]
    return false_positives, false_negatives


def _primary_root_cause(
    record: dict[str, Any],
    case_result: dict[str, Any],
    *,
    source_claim_count: int,
    generated_claim_count: int,
) -> tuple[str, str]:
    gold = record["expectation"]["expected_findings"]
    gold_types = {finding["type"] for finding in gold}
    gold_reasons = {finding["reason_code"] for finding in gold}
    false_positives = int(case_result["normalized_false_positives"])
    false_negatives = int(case_result["normalized_false_negatives"])
    coverage = case_result["coverage_status"]

    if coverage == "UNASSESSED":
        return (
            "CLAIM_EXTRACTION_ERROR",
            "Neither input produced a clinical claim, so no relationship could be assessed.",
        )
    if "INFERRED" in gold_types:
        return (
            "INFERENCE_UNSUPPORTED_BY_ENGINE",
            "The deterministic engine has no configured clinical inference evaluator.",
        )
    if "UNCERTAIN" in gold_types:
        return (
            "BENCHMARK_ANNOTATION_QUESTIONABLE",
            "The source and output preserve the same possible assertion. Under the frozen "
            "relationship definitions this is a candidate SUPPORTED paraphrase, not evidence "
            "that the source/output relationship itself is uncertain.",
        )
    if "ADDED" in gold_types and false_positives and not false_negatives:
        return (
            "BENCHMARK_ANNOTATION_QUESTIONABLE",
            "The expected addition was detected, but the target-only gold omitted a valid "
            "SUPPORTED relationship for clinical context preserved on both sides.",
        )
    if "OMITTED" in gold_types and false_positives:
        return (
            "BENCHMARK_ANNOTATION_QUESTIONABLE",
            "The gold records one target omission while other extracted source claims are also "
            "absent from the output, producing valid additional omissions.",
        )
    if "SUPPORTED" in gold_types:
        if "SEMANTIC_EQUIVALENCE" in gold_reasons:
            return (
                "SEMANTIC_EQUIVALENCE_MISSING",
                "The expected paraphrase requires semantic equivalence beyond configured "
                "normalization and concept aliases.",
            )
        if "NORMALIZED_EQUIVALENCE" in gold_reasons and false_negatives:
            return (
                "TAXONOMY_MAPPING_ERROR",
                "Normalization succeeded, but one supported pair preserved multiple dimensions "
                "and the engine emitted a different single category than the target-only gold.",
            )
        if false_negatives:
            return (
                "CORRECT_PARAPHRASE_MISCLASSIFIED",
                "A supported relationship was not classified with the expected canonical label.",
            )
        if false_positives:
            return (
                "RULE_FALSE_POSITIVE",
                "The supported relationship was found, but additional findings remained.",
            )
    if coverage == "PARTIAL" and generated_claim_count == 0 and "OMITTED" in gold_types:
        return (
            "OTHER",
            "The generated text intentionally contains no clinical claim; omission was assessed "
            "but the empty generated claim set correctly kept coverage PARTIAL.",
        )
    if false_negatives and (source_claim_count == 0 or generated_claim_count == 0):
        return (
            "CLAIM_EXTRACTION_ERROR",
            "A gold relationship lacked claims on one side after deterministic extraction.",
        )
    if false_negatives and false_positives:
        return (
            "ALIGNMENT_ERROR",
            "Predicted and expected labels diverged after claim extraction and alignment.",
        )
    if false_negatives:
        return "RULE_FALSE_NEGATIVE", "An expected relationship was not emitted."
    if false_positives:
        return "RULE_FALSE_POSITIVE", "One or more additional relationships were emitted."
    return (
        "OTHER",
        "The case is included because coverage is not FULL even though normalized labels match.",
    )


def _partial_reason(
    record: dict[str, Any], source_claim_count: int, generated_claim_count: int
) -> str:
    gold_types = {finding["type"] for finding in record["expectation"]["expected_findings"]}
    if "INFERRED" in gold_types:
        return "UNSUPPORTED_INFERENCE_PATH"
    if source_claim_count == 0 and generated_claim_count == 0:
        return "EMPTY_BOTH_CLAIM_SETS"
    if source_claim_count == 0:
        return "EMPTY_SOURCE_CLAIM_SET"
    if generated_claim_count == 0:
        return "EMPTY_GENERATED_CLAIM_SET_FAIL_CLOSED"
    return "OTHER"


def _normalizations(entities: list[Any]) -> list[dict[str, object]]:
    keys = {
        "normalized_value",
        "normalized_unit",
        "frequency",
        "route",
        "laterality",
        "date",
        "duration",
    }
    return [
        {
            "type": entity.type.value,
            "text": entity.text,
            "concept": entity.concept,
            "normalized_attributes": {
                key: value for key, value in entity.attributes.items() if key in keys
            },
        }
        for entity in entities
        if any(key in entity.attributes for key in keys)
    ]


def build_artifact() -> dict[str, Any]:
    records = {
        record["test_case_id"]: record
        for record in (json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines())
    }
    checked_result_bytes = BENCHMARK_RESULT.read_bytes()
    checked_result = json.loads(checked_result_bytes)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = DeterministicEntityParser()
    service = ClinicalNLPService()
    cases: list[dict[str, Any]] = []
    root_cause_counts: Counter[str] = Counter()
    partial_reason_counts: Counter[str] = Counter()

    for case_result in checked_result["case_results"]:
        if not (
            case_result["normalized_false_positives"]
            or case_result["normalized_false_negatives"]
            or case_result["coverage_status"] != "FULL"
        ):
            continue
        record = records[case_result["test_case_id"]]
        source_text = record["input"]["source_documents"][0]["content"]
        generated_text = record["input"]["generated_output"]["content"]
        source_entities = parser.parse(source_text)
        generated_entities = parser.parse(generated_text)
        source_claims = service.extract(source_text, "source_001")
        generated_claims = service.extract(generated_text, "generated_output")
        comparison = compare(source_text, generated_text)
        root_cause, root_cause_detail = _primary_root_cause(
            record,
            case_result,
            source_claim_count=len(source_claims),
            generated_claim_count=len(generated_claims),
        )
        if root_cause not in ROOT_CAUSES:
            raise AssertionError(f"Unknown root cause: {root_cause}")
        root_cause_counts[root_cause] += 1
        if case_result["coverage_status"] != "FULL":
            partial_reason_counts[
                _partial_reason(record, len(source_claims), len(generated_claims))
            ] += 1
        false_positives, false_negatives = _unmatched(
            case_result["expected_normalized"], case_result["actual_normalized"]
        )
        cases.append(
            {
                "case_id": case_result["test_case_id"],
                "gold_class": case_result["expected_normalized"],
                "predicted_class": case_result["actual_normalized"],
                "normalized_false_positives": false_positives,
                "normalized_false_negatives": false_negatives,
                "coverage_status": case_result["coverage_status"],
                "coverage_limitations": case_result["coverage_limitations"],
                "source": source_text,
                "generated_output": generated_text,
                "source_claims": [_claim_dict(claim) for claim in source_claims],
                "generated_claims": [_claim_dict(claim) for claim in generated_claims],
                "entity_extractions": {
                    "source": [_entity_dict(entity) for entity in source_entities],
                    "generated": [_entity_dict(entity) for entity in generated_entities],
                },
                "assertions": {
                    "source": [
                        {
                            "claim_id": claim.claim_id,
                            "assertion": claim.assertion.value,
                            "subject": claim.subject.value,
                        }
                        for claim in source_claims
                    ],
                    "generated": [
                        {
                            "claim_id": claim.claim_id,
                            "assertion": claim.assertion.value,
                            "subject": claim.subject.value,
                        }
                        for claim in generated_claims
                    ],
                },
                "normalizations": {
                    "source": _normalizations(source_entities),
                    "generated": _normalizations(generated_entities),
                },
                "alignments": _alignment_trace(source_claims, generated_claims),
                "deterministic_rules": {
                    "rule_set": comparison.versions.rule_set,
                    "evaluated": [
                        {
                            "rule_id": rule.rule_id,
                            "version": rule.version,
                            "fired": any(
                                finding.rule_id == rule.rule_id for finding in comparison.findings
                            ),
                        }
                        for rule in RULES
                    ],
                    "fired_findings": [
                        _finding_dict(finding)
                        for finding in comparison.findings
                        if finding.rule_id is not None
                    ],
                },
                "semantic_evaluation": {
                    "attempted": False,
                    "status": "MISSING_NOT_CONFIGURED",
                    "version": comparison.versions.semantic_evaluator,
                },
                "root_cause": root_cause,
                "root_cause_detail": root_cause_detail,
            }
        )

    normalized_fp = sum(len(case["normalized_false_positives"]) for case in cases)
    normalized_fn = sum(len(case["normalized_false_negatives"]) for case in cases)
    coverage_counts = Counter(
        case_result["coverage_status"] for case_result in checked_result["case_results"]
    )
    if normalized_fp != checked_result["normalized_overall"]["false_positive"]:
        raise AssertionError("Semantic diagnostic FP count diverged from benchmark result")
    if normalized_fn != checked_result["normalized_overall"]["false_negative"]:
        raise AssertionError("Semantic diagnostic FN count diverged from benchmark result")
    return {
        "schema_version": "1.0.0",
        "analysis_stage": "PRE_SEMANTIC_V0_3_BASELINE",
        "benchmark_version": manifest["benchmark_version"],
        "benchmark_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "benchmark_result_sha256": hashlib.sha256(checked_result_bytes).hexdigest(),
        "semantic_evaluator": None,
        "annotation_status": "ENGINEERING_NOT_CLINICIAN_REVIEWED",
        "summary": {
            "cases_total": checked_result["cases_total"],
            "cases_in_diagnostic": len(cases),
            "normalized_false_positives": normalized_fp,
            "normalized_false_negatives": normalized_fn,
            "coverage": dict(sorted(coverage_counts.items())),
            "root_cause_counts": dict(sorted(root_cause_counts.items())),
            "non_full_coverage_reason_counts": dict(sorted(partial_reason_counts.items())),
        },
        "cases": cases,
    }


def main() -> int:
    OUTPUT.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

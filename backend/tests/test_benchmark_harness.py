from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from backend.app.benchmark import cli as benchmark_cli
from backend.app.benchmark.harness import run_benchmark
from backend.app.domain.enums import (
    AlignmentMethod,
    CoverageStatus,
    ExecutionStatus,
    FindingType,
    Severity,
)
from backend.app.domain.models import (
    CompareResult,
    CoverageSummary,
    FaithfulnessMetrics,
    Finding,
    TextSpan,
)


def _record() -> dict[str, object]:
    return {
        "test_case_id": "bench-001",
        "input": {
            "source_documents": [
                {
                    "document_id": "src-1",
                    "type": "clinical_text",
                    "content": "Patient reports left knee pain.",
                }
            ],
            "generated_output": {
                "document_id": "generated",
                "type": "clinical_text",
                "content": "Patient reports right knee pain.",
            },
        },
        "expectation": {
            "allow_additional_findings": False,
            "expected_findings": [
                {
                    "type": "ALTERED",
                    "category": "laterality",
                    "reason_code": "LATERALITY_MISMATCH",
                    "severity": "HIGH",
                }
            ],
        },
    }


def _successful_result() -> CompareResult:
    return CompareResult(
        evaluation_id="eval-1",
        execution_status=ExecutionStatus.COMPLETED,
        coverage=CoverageSummary(
            status=CoverageStatus.FULL,
            claims_assessed=1,
            claims_unassessed=0,
        ),
        faithfulness=FaithfulnessMetrics(
            score=0,
            supported=0,
            altered=1,
            added=0,
            omitted=0,
            inferred=0,
            uncertain=0,
            critical_count=0,
            high_count=1,
        ),
        findings=[
            Finding(
                finding_id="find-1",
                type=FindingType.ALTERED,
                category="laterality",
                severity=Severity.HIGH,
                source=TextSpan(document_id="src-1", text="left", start=16, end=20),
                generated=TextSpan(document_id="generated", text="right", start=16, end=21),
                reason_code="LATERALITY_MISMATCH",
                explanation="Laterality changed from LEFT to RIGHT.",
                confidence=1.0,
                alignment_method=AlignmentMethod.ATTRIBUTE,
            )
        ],
        source_claims=[],
        generated_claims=[],
    )


def _write_case(path: Path) -> None:
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")


def test_benchmark_reports_strict_metrics_and_evidence(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: _successful_result())

    assert report.cases_completed == 1
    assert report.cases_unassessed == 0
    assert report.cases_full_coverage == 1
    assert report.cases_reduced_coverage == 0
    assert report.cases_partial_coverage == 0
    assert report.overall.f1 == 1.0
    assert report.overall.false_positive_rate == 0.0
    assert report.overall.false_negative_rate == 0.0
    assert report.critical_error_recall is None
    assert report.critical_gold_findings == 0
    assert report.high_or_critical_gold_findings == 1
    assert report.high_or_critical_error_recall == 1.0
    assert report.evidence_valid_rate == 1.0
    assert report.gold_evidence_match_rate == 1.0
    assert report.normalized_gold_evidence_match_rate == 1.0
    assert len(report.corpus_sha256) == 64
    assert report.component_version_sets == [_successful_result().versions.model_dump(mode="json")]
    serialized = report.as_dict()
    assert serialized["artifact_schema_version"] == "1.2.0"
    assert serialized["source_revision"] == "UNRECORDED"
    assert serialized["overall"]["false_positive_rate_scope"] == (
        "FP_OVER_TP_PLUS_FP_FALSE_DISCOVERY_RATE"
    )


def test_benchmark_keeps_technical_failure_unassessed(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)

    def broken_compare(_source: object, _output: object) -> CompareResult:
        raise RuntimeError("extractor unavailable")

    report = run_benchmark(benchmark_path, compare_fn=broken_compare)

    assert report.cases_unassessed == 1
    assert report.cases_exact_match == 0
    assert report.overall.false_negative == 1
    assert report.overall.false_negative_rate == 1.0
    assert report.case_results[0].execution_status == "UNASSESSED"


def test_benchmark_reports_normalized_metrics_without_hiding_raw_mismatch(
    tmp_path: Path,
) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expectation["expected_findings"] = [
        {
            "type": "SUPPORTED",
            "category": "supported_paraphrase",
            "reason_code": "SEMANTIC_EQUIVALENCE",
            "severity": "LOW",
        }
    ]
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "engine_to_gold_normalization": {
                    "mapping_version": "0.1.0",
                    "reason_code_aliases": {"CLAIM_SUPPORTED": ["SEMANTIC_EQUIVALENCE"]},
                    "contextual_category_aliases": [
                        {
                            "case_ids": ["bench-001"],
                            "engine_categories": ["symptom"],
                            "gold_category": "supported_paraphrase",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    result = _successful_result().model_copy(
        update={
            "findings": [
                Finding(
                    finding_id="find-supported",
                    type=FindingType.SUPPORTED,
                    category="symptom",
                    severity=Severity.LOW,
                    source=TextSpan(document_id="src-1", text="left", start=16, end=20),
                    generated=TextSpan(document_id="generated", text="right", start=16, end=21),
                    reason_code="CLAIM_SUPPORTED",
                    explanation="The clinical claim is supported.",
                    confidence=0.9,
                )
            ]
        }
    )

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: result)

    assert report.overall.f1 == 0.0
    assert report.normalized_overall.f1 == 1.0
    assert report.case_results[0].passed is False
    assert report.case_results[0].normalized_passed is True


def test_benchmark_preserves_labels_but_does_not_pass_unassessed_coverage(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)
    result = _successful_result().model_copy(
        update={
            "coverage": CoverageSummary(
                status=CoverageStatus.UNASSESSED,
                claims_assessed=0,
                claims_unassessed=1,
                limitations=["Evaluator unavailable."],
            )
        }
    )

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: result)

    assert report.overall.f1 == 1.0
    assert report.normalized_overall.f1 == 1.0
    assert report.cases_completed == 0
    assert report.cases_unassessed == 1
    assert report.cases_exact_match == 0
    assert report.case_results[0].coverage_status == "UNASSESSED"
    assert report.case_results[0].claims_unassessed == 1


def test_benchmark_surfaces_full_coverage_with_unassessed_claims_as_reduced(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)
    result = _successful_result().model_copy(
        update={
            "coverage": CoverageSummary(
                status=CoverageStatus.FULL,
                claims_assessed=1,
                claims_unassessed=1,
                limitations=["One claim type is unsupported."],
            )
        }
    )

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: result)

    assert report.cases_completed == 1
    assert report.cases_reduced_coverage == 1
    assert report.cases_exact_match == 0
    assert report.case_results[0].coverage_status == "REDUCED"
    assert report.case_results[0].claims_unassessed == 1


def test_benchmark_never_uses_gold_labels_to_override_evaluator_coverage(
    tmp_path: Path,
) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expectation["expected_findings"] = [
        {
            "type": "INFERRED",
            "category": "reasonable_inference",
            "reason_code": "REASONABLE_INFERENCE",
            "severity": "MODERATE",
        }
    ]
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: _successful_result())

    assert report.cases_completed == 1
    assert report.cases_unassessed == 0
    assert report.cases_full_coverage == 1
    assert report.case_results[0].coverage_status == "FULL"
    assert report.case_results[0].claims_unassessed == 0
    assert report.case_results[0].passed is False


def test_critical_recall_requires_actual_severity_at_least_gold_severity(
    tmp_path: Path,
) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expected_findings = expectation["expected_findings"]
    assert isinstance(expected_findings, list)
    expected_findings[0]["severity"] = "CRITICAL"
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: _successful_result())

    assert report.overall.f1 == 1.0
    assert report.critical_discrepancy_detection_recall == 1.0
    assert report.normalized_critical_discrepancy_detection_recall == 1.0
    assert report.critical_error_recall == 0.0
    assert report.normalized_critical_error_recall == 0.0
    assert report.high_or_critical_error_recall == 0.0
    assert report.normalized_high_or_critical_error_recall == 0.0


def test_critical_recall_uses_only_critical_gold_findings(tmp_path: Path) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expected_findings = expectation["expected_findings"]
    assert isinstance(expected_findings, list)
    expected_findings[0]["severity"] = "CRITICAL"
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    successful = _successful_result()
    critical_finding = successful.findings[0].model_copy(update={"severity": Severity.CRITICAL})
    result = successful.model_copy(update={"findings": [critical_finding]})

    report = run_benchmark(
        benchmark_path,
        compare_fn=lambda _source, _output: result,
    )

    assert report.critical_error_recall == 1.0
    assert report.critical_discrepancy_detection_recall == 1.0
    assert report.critical_gold_findings == 1
    assert report.normalized_critical_severity_aware_matches == 1
    assert report.normalized_critical_error_recall == 1.0
    assert report.high_or_critical_error_recall == 1.0


def test_critical_value_recall_has_an_explicit_lab_value_slice(tmp_path: Path) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expected_findings = expectation["expected_findings"]
    assert isinstance(expected_findings, list)
    expected_findings[0]["category"] = "lab_value"
    expected_findings[0]["reason_code"] = "NUMERIC_VALUE_MISMATCH"
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    successful = _successful_result()
    result = successful.model_copy(
        update={
            "findings": [
                successful.findings[0].model_copy(
                    update={
                        "category": "lab_value",
                        "reason_code": "NUMERIC_VALUE_MISMATCH",
                    }
                )
            ]
        }
    )

    report = run_benchmark(
        benchmark_path,
        compare_fn=lambda _source, _output: result,
        source_revision="test-revision",
    )

    assert report.source_revision == "test-revision"
    assert report.critical_value_gold_findings == 1
    assert report.critical_value_error_recall == 1.0
    assert report.normalized_critical_value_error_recall == 1.0


def test_gold_evidence_match_is_distinct_from_span_slice_validity(tmp_path: Path) -> None:
    record = _record()
    expectation = record["expectation"]
    assert isinstance(expectation, dict)
    expected_findings = expectation["expected_findings"]
    assert isinstance(expected_findings, list)
    expected_findings[0]["source"] = {
        "document_id": "src-1",
        "text": "Patient",
        "start": 0,
        "end": 7,
    }
    expected_findings[0]["generated"] = {
        "document_id": "generated",
        "text": "Patient",
        "start": 0,
        "end": 7,
    }
    benchmark_path = tmp_path / "cases.jsonl"
    benchmark_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_benchmark(benchmark_path, compare_fn=lambda _source, _output: _successful_result())

    assert report.overall.f1 == 1.0
    assert report.evidence_valid_rate == 1.0
    assert report.gold_evidence_match_rate == 0.0
    assert report.normalized_gold_evidence_match_rate == 0.0
    assert report.case_results[0].evidence_valid is True
    assert report.case_results[0].gold_evidence_matched is False


def test_cli_default_gate_rejects_zero_or_absent_critical_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)
    monkeypatch.setattr(sys, "argv", ["gatekeeper-benchmark", str(benchmark_path)])

    assert benchmark_cli.main() == 1
    capsys.readouterr()


def test_cli_rejects_a_disabled_critical_recall_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatekeeper-benchmark", str(benchmark_path), "--min-critical-recall", "0"],
    )

    with pytest.raises(SystemExit, match="greater than 0"):
        benchmark_cli.main()


def test_benchmark_reports_frozen_metrics_by_split(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "cases.jsonl"
    _write_case(benchmark_path)
    (tmp_path / "splits.json").write_text(
        json.dumps(
            {
                "development": ["bench-001"],
                "validation": [],
                "held_out_test": [],
                "clinician_calibration": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_benchmark(
        benchmark_path,
        compare_fn=lambda _source, _output: _successful_result(),
    )
    serialized = report.as_dict()["split_metrics"]

    assert serialized["development"]["case_count"] == 1
    assert serialized["development"]["normalized_overall"]["f1"] == 1.0
    assert serialized["development"]["coverage"] == {
        "FULL": 1,
        "REDUCED": 0,
        "PARTIAL": 0,
        "UNASSESSED": 0,
    }
    assert serialized["held_out_test"]["case_count"] == 0

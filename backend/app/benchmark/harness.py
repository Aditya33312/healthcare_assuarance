"""Deterministic clinical-faithfulness benchmark runner."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from backend.app.domain.enums import CoverageStatus, ExecutionStatus, Severity
from backend.app.domain.models import (
    CompareResult,
    Finding,
    GeneratedOutput,
    SourceBundle,
    SourceDocument,
    TextSpan,
)

Label = tuple[str, str, str]
CompareFunction = Callable[..., CompareResult]
_TYPED_EVIDENCE_TYPES = {
    "transcript",
    "lab_results",
    "medication_list",
    "structured_json",
    "rag_context",
}


@dataclass(slots=True)
class Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Prediction-conditioned FP rate (the false-discovery rate).

        Finding extraction has no enumerable true-negative universe, so the
        classical ``FP / (FP + TN)`` rate is not identifiable.  This explicitly
        reported surrogate uses ``FP / (TP + FP)`` and is labelled in the artifact.
        """

        denominator = self.true_positive + self.false_positive
        return self.false_positive / denominator if denominator else 0.0

    @property
    def false_negative_rate(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.false_negative / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_positive_rate_scope": "FP_OVER_TP_PLUS_FP_FALSE_DISCOVERY_RATE",
            "false_negative_rate": round(self.false_negative_rate, 4),
            "false_negative_rate_scope": "FN_OVER_TP_PLUS_FN",
        }


@dataclass(slots=True)
class RecallSlice:
    """Gold-denominated recall slice with strict and normalized matches."""

    gold_findings: int = 0
    strict_matches: int = 0
    normalized_matches: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "gold_findings": self.gold_findings,
            "strict_matches": self.strict_matches,
            "normalized_matches": self.normalized_matches,
            "strict_recall": (
                round(self.strict_matches / self.gold_findings, 4) if self.gold_findings else None
            ),
            "normalized_recall": (
                round(self.normalized_matches / self.gold_findings, 4)
                if self.gold_findings
                else None
            ),
        }


@dataclass(slots=True)
class CaseResult:
    test_case_id: str
    passed: bool
    normalized_passed: bool
    expected: list[Label]
    actual: list[Label]
    expected_normalized: list[Label]
    actual_normalized: list[Label]
    execution_status: str
    coverage_status: str
    claims_unassessed: int
    coverage_limitations: list[str]
    evidence_valid: bool
    gold_evidence_matched: bool
    normalized_gold_evidence_matched: bool
    false_positives: int = 0
    false_negatives: int = 0
    normalized_false_positives: int = 0
    normalized_false_negatives: int = 0
    error: str | None = None


@dataclass(slots=True)
class BenchmarkReport:
    benchmark_path: str
    cases_total: int
    cases_completed: int
    cases_unassessed: int
    cases_full_coverage: int
    cases_reduced_coverage: int
    cases_partial_coverage: int
    cases_exact_match: int
    cases_normalized_match: int
    cases_gold_evidence_match: int
    cases_normalized_gold_evidence_match: int
    overall: Counts
    normalized_overall: Counts
    by_finding_type: dict[str, Counts] = field(default_factory=dict)
    by_category: dict[str, Counts] = field(default_factory=dict)
    normalized_by_finding_type: dict[str, Counts] = field(default_factory=dict)
    normalized_by_category: dict[str, Counts] = field(default_factory=dict)
    critical_gold_findings: int = 0
    critical_discrepancy_detection_matches: int = 0
    normalized_critical_discrepancy_detection_matches: int = 0
    critical_discrepancy_detection_recall: float | None = None
    normalized_critical_discrepancy_detection_recall: float | None = None
    critical_severity_aware_matches: int = 0
    normalized_critical_severity_aware_matches: int = 0
    critical_error_recall: float | None = None
    normalized_critical_error_recall: float | None = None
    high_or_critical_gold_findings: int = 0
    high_or_critical_severity_aware_matches: int = 0
    normalized_high_or_critical_severity_aware_matches: int = 0
    high_or_critical_error_recall: float | None = None
    normalized_high_or_critical_error_recall: float | None = None
    high_gold_findings: int = 0
    high_severity_aware_matches: int = 0
    normalized_high_severity_aware_matches: int = 0
    high_error_recall: float | None = None
    normalized_high_error_recall: float | None = None
    evidence_valid_rate: float = 0.0
    artifact_schema_version: str = "1.2.0"
    corpus_sha256: str = ""
    manifest_sha256: str | None = None
    benchmark_id: str | None = None
    benchmark_version: str | None = None
    component_version_sets: list[dict[str, Any]] = field(default_factory=list)
    source_revision: str = "UNRECORDED"
    repository_state: str = "UNRECORDED"
    python_version: str = ""
    dependency_lock_sha256: str | None = None
    critical_value_gold_findings: int = 0
    critical_value_severity_aware_matches: int = 0
    normalized_critical_value_severity_aware_matches: int = 0
    critical_value_error_recall: float | None = None
    normalized_critical_value_error_recall: float | None = None
    safety_slice_metrics: dict[str, RecallSlice] = field(default_factory=dict)
    predicted_uncertain_findings: int = 0
    split_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        return self.cases_exact_match / self.cases_total if self.cases_total else 0.0

    @property
    def normalized_match_rate(self) -> float:
        return self.cases_normalized_match / self.cases_total if self.cases_total else 0.0

    @property
    def gold_evidence_match_rate(self) -> float:
        return self.cases_gold_evidence_match / self.cases_total if self.cases_total else 0.0

    @property
    def normalized_gold_evidence_match_rate(self) -> float:
        return (
            self.cases_normalized_gold_evidence_match / self.cases_total
            if self.cases_total
            else 0.0
        )

    @property
    def false_positives_per_case(self) -> float:
        return self.overall.false_positive / self.cases_total if self.cases_total else 0.0

    @property
    def false_negatives_per_case(self) -> float:
        return self.overall.false_negative / self.cases_total if self.cases_total else 0.0

    @property
    def normalized_false_positives_per_case(self) -> float:
        return (
            self.normalized_overall.false_positive / self.cases_total if self.cases_total else 0.0
        )

    @property
    def normalized_false_negatives_per_case(self) -> float:
        return (
            self.normalized_overall.false_negative / self.cases_total if self.cases_total else 0.0
        )

    @property
    def uncertain_rate(self) -> float:
        predicted = sum(len(result.actual) for result in self.case_results if result.error is None)
        return self.predicted_uncertain_findings / predicted if predicted else 0.0

    def _coverage_rate(self, count: int) -> float:
        return count / self.cases_total if self.cases_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "benchmark_path": self.benchmark_path,
            "corpus_sha256": self.corpus_sha256,
            "manifest_sha256": self.manifest_sha256,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "component_version_sets": self.component_version_sets,
            "source_revision": self.source_revision,
            "repository_state": self.repository_state,
            "python_version": self.python_version,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "cases_total": self.cases_total,
            "cases_completed": self.cases_completed,
            "cases_unassessed": self.cases_unassessed,
            "cases_full_coverage": self.cases_full_coverage,
            "cases_reduced_coverage": self.cases_reduced_coverage,
            "cases_partial_coverage": self.cases_partial_coverage,
            "coverage_rates": {
                "full": round(self._coverage_rate(self.cases_full_coverage), 4),
                "reduced": round(self._coverage_rate(self.cases_reduced_coverage), 4),
                "partial": round(self._coverage_rate(self.cases_partial_coverage), 4),
                "unassessed": round(self._coverage_rate(self.cases_unassessed), 4),
            },
            "cases_exact_match": self.cases_exact_match,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "cases_normalized_match": self.cases_normalized_match,
            "normalized_match_rate": round(self.normalized_match_rate, 4),
            "cases_gold_evidence_match": self.cases_gold_evidence_match,
            "gold_evidence_match_rate": round(self.gold_evidence_match_rate, 4),
            "cases_normalized_gold_evidence_match": (self.cases_normalized_gold_evidence_match),
            "normalized_gold_evidence_match_rate": round(
                self.normalized_gold_evidence_match_rate, 4
            ),
            "overall": self.overall.as_dict(),
            "normalized_overall": self.normalized_overall.as_dict(),
            "false_positives_per_case": round(self.false_positives_per_case, 4),
            "false_negatives_per_case": round(self.false_negatives_per_case, 4),
            "normalized_false_positives_per_case": round(
                self.normalized_false_positives_per_case, 4
            ),
            "normalized_false_negatives_per_case": round(
                self.normalized_false_negatives_per_case, 4
            ),
            "by_finding_type": {
                key: value.as_dict() for key, value in sorted(self.by_finding_type.items())
            },
            "by_category": {
                key: value.as_dict() for key, value in sorted(self.by_category.items())
            },
            "normalized_by_finding_type": {
                key: value.as_dict()
                for key, value in sorted(self.normalized_by_finding_type.items())
            },
            "normalized_by_category": {
                key: value.as_dict() for key, value in sorted(self.normalized_by_category.items())
            },
            "critical_gold_findings": self.critical_gold_findings,
            "critical_discrepancy_detection_matches": (self.critical_discrepancy_detection_matches),
            "normalized_critical_discrepancy_detection_matches": (
                self.normalized_critical_discrepancy_detection_matches
            ),
            "critical_discrepancy_detection_recall": (
                None
                if self.critical_discrepancy_detection_recall is None
                else round(self.critical_discrepancy_detection_recall, 4)
            ),
            "normalized_critical_discrepancy_detection_recall": (
                None
                if self.normalized_critical_discrepancy_detection_recall is None
                else round(self.normalized_critical_discrepancy_detection_recall, 4)
            ),
            "critical_severity_aware_matches": self.critical_severity_aware_matches,
            "normalized_critical_severity_aware_matches": (
                self.normalized_critical_severity_aware_matches
            ),
            "critical_error_recall": (
                None if self.critical_error_recall is None else round(self.critical_error_recall, 4)
            ),
            "normalized_critical_error_recall": (
                None
                if self.normalized_critical_error_recall is None
                else round(self.normalized_critical_error_recall, 4)
            ),
            "critical_recall_scope": "CRITICAL_ONLY",
            "high_or_critical_gold_findings": self.high_or_critical_gold_findings,
            "high_or_critical_severity_aware_matches": (
                self.high_or_critical_severity_aware_matches
            ),
            "normalized_high_or_critical_severity_aware_matches": (
                self.normalized_high_or_critical_severity_aware_matches
            ),
            "high_or_critical_error_recall": (
                None
                if self.high_or_critical_error_recall is None
                else round(self.high_or_critical_error_recall, 4)
            ),
            "normalized_high_or_critical_error_recall": (
                None
                if self.normalized_high_or_critical_error_recall is None
                else round(self.normalized_high_or_critical_error_recall, 4)
            ),
            "high_or_critical_recall_scope": "HIGH_AND_CRITICAL",
            "critical_recall_is_severity_aware": True,
            "critical_severity_recall": (
                None if self.critical_error_recall is None else round(self.critical_error_recall, 4)
            ),
            "normalized_critical_severity_recall": (
                None
                if self.normalized_critical_error_recall is None
                else round(self.normalized_critical_error_recall, 4)
            ),
            "high_gold_findings": self.high_gold_findings,
            "high_severity_aware_matches": self.high_severity_aware_matches,
            "normalized_high_severity_aware_matches": (self.normalized_high_severity_aware_matches),
            "high_error_recall": (
                None if self.high_error_recall is None else round(self.high_error_recall, 4)
            ),
            "normalized_high_error_recall": (
                None
                if self.normalized_high_error_recall is None
                else round(self.normalized_high_error_recall, 4)
            ),
            "critical_value_gold_findings": self.critical_value_gold_findings,
            "critical_value_severity_aware_matches": (self.critical_value_severity_aware_matches),
            "normalized_critical_value_severity_aware_matches": (
                self.normalized_critical_value_severity_aware_matches
            ),
            "critical_value_error_recall": (
                None
                if self.critical_value_error_recall is None
                else round(self.critical_value_error_recall, 4)
            ),
            "normalized_critical_value_error_recall": (
                None
                if self.normalized_critical_value_error_recall is None
                else round(self.normalized_critical_value_error_recall, 4)
            ),
            "critical_value_recall_scope": "GOLD_CATEGORY_LAB_VALUE",
            "safety_slice_metrics": {
                key: value.as_dict() for key, value in sorted(self.safety_slice_metrics.items())
            },
            "predicted_uncertain_findings": self.predicted_uncertain_findings,
            "uncertain_rate": round(self.uncertain_rate, 4),
            "evidence_valid_rate": round(self.evidence_valid_rate, 4),
            "split_metrics": self.split_metrics,
            "case_results": [
                {
                    "test_case_id": result.test_case_id,
                    "passed": result.passed,
                    "normalized_passed": result.normalized_passed,
                    "expected": result.expected,
                    "actual": result.actual,
                    "expected_normalized": result.expected_normalized,
                    "actual_normalized": result.actual_normalized,
                    "execution_status": result.execution_status,
                    "coverage_status": result.coverage_status,
                    "claims_unassessed": result.claims_unassessed,
                    "coverage_limitations": result.coverage_limitations,
                    "evidence_valid": result.evidence_valid,
                    "gold_evidence_matched": result.gold_evidence_matched,
                    "normalized_gold_evidence_matched": (result.normalized_gold_evidence_matched),
                    "false_positives": result.false_positives,
                    "false_negatives": result.false_negatives,
                    "normalized_false_positives": result.normalized_false_positives,
                    "normalized_false_negatives": result.normalized_false_negatives,
                    "error": result.error,
                }
                for result in self.case_results
            ],
        }


def _labels(findings: list[dict[str, Any]]) -> list[Label]:
    return [
        (
            str(finding["type"]).upper(),
            str(finding["category"]).lower(),
            str(finding["reason_code"]).upper(),
        )
        for finding in findings
    ]


def _actual_labels(result: CompareResult) -> list[Label]:
    return [
        (finding.type.value, finding.category.lower(), finding.reason_code.upper())
        for finding in result.findings
    ]


def _finding_label(finding: Finding) -> Label:
    return finding.type.value, finding.category.lower(), finding.reason_code.upper()


def _add_counts(counts: Counts, expected: Counter[Label], actual: Counter[Label]) -> None:
    for label in expected.keys() | actual.keys():
        matched = min(expected[label], actual[label])
        counts.true_positive += matched
        counts.false_negative += expected[label] - matched
        counts.false_positive += actual[label] - matched


def _counter_deltas(expected: Counter[Label], actual: Counter[Label]) -> tuple[int, int]:
    false_positives = 0
    false_negatives = 0
    for label in expected.keys() | actual.keys():
        matched = min(expected[label], actual[label])
        false_positives += actual[label] - matched
        false_negatives += expected[label] - matched
    return false_positives, false_negatives


def _evidence_is_valid(
    result: CompareResult,
    sources: dict[str, str],
    generated_content: str,
) -> bool:
    for finding in result.findings:
        if finding.source is not None:
            document_id = finding.source.document_id
            if document_id not in sources:
                return False
            content = sources[document_id]
            if content[finding.source.start : finding.source.end] != finding.source.text:
                return False
        if finding.generated is not None:
            content = generated_content
            if content[finding.generated.start : finding.generated.end] != finding.generated.text:
                return False
    return True


_UNSPECIFIED = object()
_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MODERATE: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def _span_matches_gold(actual: TextSpan | None, expected: object) -> bool:
    if expected is _UNSPECIFIED:
        return True
    if expected is None:
        return actual is None
    if actual is None or not isinstance(expected, dict):
        return False
    expected_document_id = expected.get("document_id")
    if expected_document_id is not None and expected_document_id != actual.document_id:
        return False
    try:
        expected_start = int(expected["start"])
        expected_end = int(expected["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return actual.start < expected_end and expected_start < actual.end


def _finding_evidence_matches_gold(actual: Finding, expected: dict[str, Any]) -> bool:
    return _span_matches_gold(actual.source, expected.get("source", _UNSPECIFIED)) and (
        _span_matches_gold(actual.generated, expected.get("generated", _UNSPECIFIED))
    )


def _all_gold_evidence_matched(
    case_id: str,
    expected_findings: list[dict[str, Any]],
    actual_findings: list[Finding],
    normalizer: _LabelNormalizer | None = None,
) -> bool:
    remaining = list(actual_findings)
    for expected in expected_findings:
        expected_label = _labels([expected])[0]
        if normalizer is not None:
            expected_label = normalizer.label(case_id, expected_label)
        match_index = next(
            (
                index
                for index, actual in enumerate(remaining)
                if (
                    normalizer.label(case_id, _finding_label(actual))
                    if normalizer is not None
                    else _finding_label(actual)
                )
                == expected_label
                and _finding_evidence_matches_gold(actual, expected)
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _consume_severity_aware_match(
    case_id: str,
    expected: dict[str, Any],
    actual_findings: list[Finding],
    normalizer: _LabelNormalizer | None = None,
) -> bool:
    expected_label = _labels([expected])[0]
    if normalizer is not None:
        expected_label = normalizer.label(case_id, expected_label)
    expected_severity = Severity(str(expected["severity"]).upper())
    for index, actual in enumerate(actual_findings):
        actual_label = _finding_label(actual)
        if normalizer is not None:
            actual_label = normalizer.label(case_id, actual_label)
        if (
            actual_label == expected_label
            and _SEVERITY_RANK[actual.severity] >= _SEVERITY_RANK[expected_severity]
            and _finding_evidence_matches_gold(actual, expected)
        ):
            actual_findings.pop(index)
            return True
    return False


def _consume_discrepancy_match(
    case_id: str,
    expected: dict[str, Any],
    actual_findings: list[Finding],
    normalizer: _LabelNormalizer | None = None,
) -> bool:
    """Consume one label/evidence match without considering severity."""

    expected_label = _labels([expected])[0]
    if normalizer is not None:
        expected_label = normalizer.label(case_id, expected_label)
    for index, actual in enumerate(actual_findings):
        actual_label = _finding_label(actual)
        if normalizer is not None:
            actual_label = normalizer.label(case_id, actual_label)
        if actual_label == expected_label and _finding_evidence_matches_gold(actual, expected):
            actual_findings.pop(index)
            return True
    return False


_SAFETY_SLICE_NAMES = (
    "allergy_contradiction",
    "critical_numeric_error",
    "dose_error",
    "frequency_error",
    "laterality_error",
    "medication_error",
    "negation_reversal",
    "route_error",
)


def _safety_slices_for(finding: dict[str, Any]) -> set[str]:
    finding_type, category, reason = _labels([finding])[0]
    slices: set[str] = set()
    if finding_type != "SUPPORTED" and category.startswith("medication_"):
        slices.add("medication_error")
    reason_mapping = {
        "ALLERGY_CONTRADICTION": "allergy_contradiction",
        "DOSE_MISMATCH": "dose_error",
        "FREQUENCY_MISMATCH": "frequency_error",
        "LATERALITY_MISMATCH": "laterality_error",
        "NEGATION_REVERSAL": "negation_reversal",
        "ROUTE_MISMATCH": "route_error",
    }
    if mapped := reason_mapping.get(reason):
        slices.add(mapped)
    if (
        category == "lab_value"
        and reason == "NUMERIC_VALUE_MISMATCH"
        and str(finding.get("severity", "")).upper() == Severity.CRITICAL.value
    ):
        slices.add("critical_numeric_error")
    return slices


def _dependency_lock_sha256(benchmark_path: Path) -> str | None:
    for parent in (benchmark_path.parent, *benchmark_path.parents):
        candidate = parent / "requirements.lock"
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None


def _reported_coverage_status(result: CompareResult) -> CoverageStatus:
    """Normalize only internally inconsistent result coverage.

    Gold annotations must never influence whether the evaluator claims coverage;
    that status is an output of the evaluator under test.
    """

    if result.coverage.claims_unassessed and result.coverage.status is CoverageStatus.FULL:
        return CoverageStatus.REDUCED
    return result.coverage.status


def _update_breakdown(
    target: dict[str, Counts],
    expected: Counter[Label],
    actual: Counter[Label],
    key_index: int,
) -> None:
    keys = {label[key_index] for label in expected.keys() | actual.keys()}
    for key in keys:
        expected_subset = Counter(
            {label: count for label, count in expected.items() if label[key_index] == key}
        )
        actual_subset = Counter(
            {label: count for label, count in actual.items() if label[key_index] == key}
        )
        _add_counts(target[key], expected_subset, actual_subset)


class _LabelNormalizer:
    """Apply only the explicit, versioned aliases from the benchmark manifest."""

    def __init__(self, manifest: dict[str, Any] | None) -> None:
        config = (manifest or {}).get("engine_to_gold_normalization", {})
        self.type_aliases: dict[str, str] = {
            str(key).upper(): str(value).upper()
            for key, value in config.get("type_aliases", {}).items()
            if isinstance(value, str)
        }
        self.reason_groups: dict[str, str] = {}
        for engine_reason, gold_reasons in config.get("reason_code_aliases", {}).items():
            canonical = f"ALIAS:{str(engine_reason).upper()}"
            self.reason_groups[str(engine_reason).upper()] = canonical
            for gold_reason in gold_reasons:
                self.reason_groups[str(gold_reason).upper()] = canonical
        self.contextual_categories: list[dict[str, Any]] = config.get(
            "contextual_category_aliases", []
        )

    def label(self, case_id: str, label: Label) -> Label:
        finding_type, category, reason = label
        finding_type = self.type_aliases.get(finding_type.upper(), finding_type.upper())
        category = category.lower()
        for mapping in self.contextual_categories:
            if case_id not in mapping.get("case_ids", []):
                continue
            engine_categories = {
                str(candidate).lower() for candidate in mapping.get("engine_categories", [])
            }
            if category in engine_categories:
                category = str(mapping["gold_category"]).lower()
                break
        reason = self.reason_groups.get(reason.upper(), reason.upper())
        return finding_type, category, reason

    def labels(self, case_id: str, labels: list[Label]) -> list[Label]:
        return [self.label(case_id, label) for label in labels]


def _load_manifest(benchmark_path: Path) -> dict[str, Any] | None:
    manifest_path = benchmark_path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def _split_metrics(
    benchmark_path: Path,
    case_results: list[CaseResult],
) -> dict[str, dict[str, Any]]:
    split_path = benchmark_path.with_name("splits.json")
    if not split_path.exists():
        return {}
    config = json.loads(split_path.read_text(encoding="utf-8"))
    by_id = {result.test_case_id: result for result in case_results}
    output: dict[str, dict[str, Any]] = {}
    for split_name in (
        "development",
        "validation",
        "held_out_test",
        "clinician_calibration",
    ):
        ids = [str(case_id) for case_id in config.get(split_name, [])]
        results = [by_id[case_id] for case_id in ids if case_id in by_id]
        strict = Counts()
        normalized = Counts()
        normalized_by_type: defaultdict[str, Counts] = defaultdict(Counts)
        for result in results:
            expected = Counter(result.expected)
            actual = Counter(result.actual)
            expected_normalized = Counter(result.expected_normalized)
            actual_normalized = Counter(result.actual_normalized)
            _add_counts(strict, expected, actual)
            _add_counts(normalized, expected_normalized, actual_normalized)
            _update_breakdown(
                normalized_by_type,
                expected_normalized,
                actual_normalized,
                0,
            )
        coverage = Counter(result.coverage_status for result in results)
        output[split_name] = {
            "case_count": len(results),
            "case_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "overall": strict.as_dict(),
            "normalized_overall": normalized.as_dict(),
            "normalized_by_finding_type": {
                key: counts.as_dict() for key, counts in sorted(normalized_by_type.items())
            },
            "coverage": {
                status: coverage.get(status, 0)
                for status in ("FULL", "REDUCED", "PARTIAL", "UNASSESSED")
            },
        }
    return output


def run_benchmark(
    path: str | Path,
    compare_fn: CompareFunction | None = None,
    *,
    source_revision: str = "UNRECORDED",
    repository_state: str = "UNRECORDED",
    python_version: str | None = None,
) -> BenchmarkReport:
    """Evaluate the Compare Engine against JSONL gold annotations.

    A technical exception is recorded as unassessed and cannot count as a match.
    Matching is deliberately strict on finding type, category, and reason code.
    """

    benchmark_path = Path(path)
    if compare_fn is None:
        from backend.app.compare_engine.engine import compare

        compare_fn = compare

    corpus_bytes = benchmark_path.read_bytes()
    records = [
        json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    manifest_path = benchmark_path.with_name("manifest.json")
    manifest = _load_manifest(benchmark_path)
    normalizer = _LabelNormalizer(manifest)
    overall = Counts()
    normalized_overall = Counts()
    by_type: dict[str, Counts] = defaultdict(Counts)
    by_category: dict[str, Counts] = defaultdict(Counts)
    normalized_by_type: dict[str, Counts] = defaultdict(Counts)
    normalized_by_category: dict[str, Counts] = defaultdict(Counts)
    case_results: list[CaseResult] = []
    completed = 0
    unassessed_cases = 0
    full_coverage_cases = 0
    reduced_coverage_cases = 0
    partial_coverage_cases = 0
    exact_matches = 0
    normalized_matches = 0
    evidence_valid_cases = 0
    gold_evidence_matches = 0
    normalized_gold_evidence_matches = 0
    critical_expected = 0
    critical_detection_matched = 0
    normalized_critical_detection_matched = 0
    critical_matched = 0
    normalized_critical_matched = 0
    high_or_critical_expected = 0
    high_or_critical_matched = 0
    normalized_high_or_critical_matched = 0
    high_expected = 0
    high_matched = 0
    normalized_high_matched = 0
    critical_value_expected = 0
    critical_value_matched = 0
    normalized_critical_value_matched = 0
    safety_slices = {name: RecallSlice() for name in _SAFETY_SLICE_NAMES}
    predicted_uncertain_findings = 0
    component_version_sets: dict[str, dict[str, Any]] = {}

    for record in records:
        expected_findings = record["expectation"]["expected_findings"]
        expected_labels = _labels(expected_findings)
        expected_counter = Counter(expected_labels)
        expected_normalized_labels = normalizer.labels(record["test_case_id"], expected_labels)
        expected_normalized_counter = Counter(expected_normalized_labels)
        raw_source_documents = record["input"]["source_documents"]
        generated = record["input"]["generated_output"]
        try:
            uses_typed_ingestion = any(
                document.get("type") in _TYPED_EVIDENCE_TYPES
                for document in raw_source_documents
            ) or generated.get("type") == "structured_json"
            if uses_typed_ingestion:
                from pydantic import TypeAdapter

                from backend.app.clinical_nlp import ClinicalNLPService
                from backend.app.ingestion import (
                    CanonicalEvidenceExtractor,
                    EvidenceSource,
                    GeneratedEvidenceOutput,
                    UniversalEvidenceIngestionService,
                )

                evidence_adapter: TypeAdapter[EvidenceSource] = TypeAdapter(EvidenceSource)
                output_adapter: TypeAdapter[GeneratedEvidenceOutput] = TypeAdapter(
                    GeneratedEvidenceOutput
                )
                typed_sources = [
                    evidence_adapter.validate_python(document)
                    for document in raw_source_documents
                ]
                typed_generated = output_adapter.validate_python(
                    {
                        key: value
                        for key, value in generated.items()
                        if key != "document_id"
                    }
                )
                ingestion_service = UniversalEvidenceIngestionService()
                ingestion = ingestion_service.ingest(typed_sources, typed_generated)
                source_documents = list(ingestion.source_bundle.documents)
                generated_output = ingestion.generated_output
                result = compare_fn(
                    SourceBundle(documents=source_documents),
                    generated_output,
                    nlp_service=CanonicalEvidenceExtractor(
                        ClinicalNLPService(),
                        (*ingestion.source_documents, ingestion.generated_document),
                        max_claims=ingestion_service.limits.max_extracted_claims,
                        processing_timeout_seconds=(
                            ingestion_service.limits.max_processing_seconds
                        ),
                    ),
                )
            else:
                source_documents = [
                    SourceDocument(
                        document_id=document["document_id"],
                        type=document.get("type", "clinical_text"),
                        content=document["content"],
                    )
                    for document in raw_source_documents
                ]
                generated_output = GeneratedOutput(
                    type=generated.get("type", "clinical_text"),
                    content=generated["content"],
                )
                result = compare_fn(
                    SourceBundle(documents=source_documents),
                    generated_output,
                )
            if result.execution_status is not ExecutionStatus.COMPLETED:
                raise RuntimeError(f"evaluation ended as {result.execution_status.value}")
            coverage_status = _reported_coverage_status(result)
            reported_claims_unassessed = result.coverage.claims_unassessed
            coverage_limitations = list(result.coverage.limitations)
            version_set = result.versions.model_dump(mode="json")
            component_version_sets[
                json.dumps(version_set, sort_keys=True, separators=(",", ":"))
            ] = version_set
            if coverage_status is CoverageStatus.UNASSESSED:
                unassessed_cases += 1
            else:
                completed += 1
                if coverage_status is CoverageStatus.FULL:
                    full_coverage_cases += 1
                elif coverage_status is CoverageStatus.REDUCED:
                    reduced_coverage_cases += 1
                elif coverage_status is CoverageStatus.PARTIAL:
                    partial_coverage_cases += 1
            actual_labels = _actual_labels(result)
            actual_counter = Counter(actual_labels)
            actual_normalized_labels = normalizer.labels(record["test_case_id"], actual_labels)
            actual_normalized_counter = Counter(actual_normalized_labels)
            predicted_uncertain_findings += sum(
                finding_type == "UNCERTAIN" for finding_type, _category, _reason in actual_labels
            )
            case_false_positives, case_false_negatives = _counter_deltas(
                expected_counter, actual_counter
            )
            (
                case_normalized_false_positives,
                case_normalized_false_negatives,
            ) = _counter_deltas(expected_normalized_counter, actual_normalized_counter)
            _add_counts(overall, expected_counter, actual_counter)
            _update_breakdown(by_type, expected_counter, actual_counter, 0)
            _update_breakdown(by_category, expected_counter, actual_counter, 1)
            _add_counts(
                normalized_overall,
                expected_normalized_counter,
                actual_normalized_counter,
            )
            _update_breakdown(
                normalized_by_type,
                expected_normalized_counter,
                actual_normalized_counter,
                0,
            )
            _update_breakdown(
                normalized_by_category,
                expected_normalized_counter,
                actual_normalized_counter,
                1,
            )

            allow_additional = record["expectation"].get("allow_additional_findings", False)
            passed = (
                all(actual_counter[label] >= count for label, count in expected_counter.items())
                if allow_additional
                else expected_counter == actual_counter
            )
            normalized_passed = (
                all(
                    actual_normalized_counter[label] >= count
                    for label, count in expected_normalized_counter.items()
                )
                if allow_additional
                else expected_normalized_counter == actual_normalized_counter
            )
            sources = {document.document_id: document.content for document in source_documents}
            evidence_valid = _evidence_is_valid(result, sources, generated["content"])
            gold_evidence_matched = _all_gold_evidence_matched(
                record["test_case_id"], expected_findings, result.findings
            )
            normalized_gold_evidence_matched = _all_gold_evidence_matched(
                record["test_case_id"], expected_findings, result.findings, normalizer
            )
            coverage_allows_match = (
                coverage_status in {CoverageStatus.FULL, CoverageStatus.PARTIAL}
                and reported_claims_unassessed == 0
            )
            case_passed = passed and evidence_valid and coverage_allows_match
            normalized_case_passed = normalized_passed and evidence_valid and coverage_allows_match
            exact_matches += int(case_passed)
            normalized_matches += int(normalized_case_passed)
            if coverage_status is not CoverageStatus.UNASSESSED:
                evidence_valid_cases += int(evidence_valid)
                gold_evidence_matches += int(gold_evidence_matched)
                normalized_gold_evidence_matches += int(normalized_gold_evidence_matched)

            critical_actual_remaining = list(result.findings)
            normalized_critical_actual_remaining = list(result.findings)
            critical_detection_actual_remaining = list(result.findings)
            normalized_critical_detection_actual_remaining = list(result.findings)
            high_or_critical_actual_remaining = list(result.findings)
            normalized_high_or_critical_actual_remaining = list(result.findings)
            high_actual_remaining = list(result.findings)
            normalized_high_actual_remaining = list(result.findings)
            critical_value_actual_remaining = list(result.findings)
            normalized_critical_value_actual_remaining = list(result.findings)
            safety_actual_remaining = {name: list(result.findings) for name in _SAFETY_SLICE_NAMES}
            normalized_safety_actual_remaining = {
                name: list(result.findings) for name in _SAFETY_SLICE_NAMES
            }
            for finding in expected_findings:
                for slice_name in _safety_slices_for(finding):
                    slice_metric = safety_slices[slice_name]
                    slice_metric.gold_findings += 1
                    if _consume_discrepancy_match(
                        record["test_case_id"],
                        finding,
                        safety_actual_remaining[slice_name],
                    ):
                        slice_metric.strict_matches += 1
                    if _consume_discrepancy_match(
                        record["test_case_id"],
                        finding,
                        normalized_safety_actual_remaining[slice_name],
                        normalizer,
                    ):
                        slice_metric.normalized_matches += 1
                if str(finding["category"]).casefold() == "lab_value":
                    critical_value_expected += 1
                    if _consume_severity_aware_match(
                        record["test_case_id"],
                        finding,
                        critical_value_actual_remaining,
                    ):
                        critical_value_matched += 1
                    if _consume_severity_aware_match(
                        record["test_case_id"],
                        finding,
                        normalized_critical_value_actual_remaining,
                        normalizer,
                    ):
                        normalized_critical_value_matched += 1
                gold_severity = str(finding["severity"]).upper()
                if gold_severity == Severity.HIGH.value:
                    high_expected += 1
                    if _consume_severity_aware_match(
                        record["test_case_id"], finding, high_actual_remaining
                    ):
                        high_matched += 1
                    if _consume_severity_aware_match(
                        record["test_case_id"],
                        finding,
                        normalized_high_actual_remaining,
                        normalizer,
                    ):
                        normalized_high_matched += 1
                if gold_severity not in {
                    Severity.CRITICAL.value,
                    Severity.HIGH.value,
                }:
                    continue
                high_or_critical_expected += 1
                if _consume_severity_aware_match(
                    record["test_case_id"],
                    finding,
                    high_or_critical_actual_remaining,
                ):
                    high_or_critical_matched += 1
                if _consume_severity_aware_match(
                    record["test_case_id"],
                    finding,
                    normalized_high_or_critical_actual_remaining,
                    normalizer,
                ):
                    normalized_high_or_critical_matched += 1
                if gold_severity != Severity.CRITICAL.value:
                    continue
                critical_expected += 1
                if _consume_discrepancy_match(
                    record["test_case_id"],
                    finding,
                    critical_detection_actual_remaining,
                ):
                    critical_detection_matched += 1
                if _consume_discrepancy_match(
                    record["test_case_id"],
                    finding,
                    normalized_critical_detection_actual_remaining,
                    normalizer,
                ):
                    normalized_critical_detection_matched += 1
                if _consume_severity_aware_match(
                    record["test_case_id"], finding, critical_actual_remaining
                ):
                    critical_matched += 1
                if _consume_severity_aware_match(
                    record["test_case_id"],
                    finding,
                    normalized_critical_actual_remaining,
                    normalizer,
                ):
                    normalized_critical_matched += 1

            case_results.append(
                CaseResult(
                    test_case_id=record["test_case_id"],
                    passed=case_passed,
                    normalized_passed=normalized_case_passed,
                    expected=expected_labels,
                    actual=actual_labels,
                    expected_normalized=expected_normalized_labels,
                    actual_normalized=actual_normalized_labels,
                    execution_status=result.execution_status.value,
                    coverage_status=coverage_status.value,
                    claims_unassessed=reported_claims_unassessed,
                    coverage_limitations=coverage_limitations,
                    evidence_valid=evidence_valid,
                    gold_evidence_matched=gold_evidence_matched,
                    normalized_gold_evidence_matched=(normalized_gold_evidence_matched),
                    false_positives=case_false_positives,
                    false_negatives=case_false_negatives,
                    normalized_false_positives=case_normalized_false_positives,
                    normalized_false_negatives=case_normalized_false_negatives,
                )
            )
        # The harness deliberately records any case-level failure as unassessed
        # so one broken case cannot hide the rest of the benchmark report.
        except Exception as exc:
            unassessed_cases += 1
            expected_counter = Counter(expected_labels)
            _add_counts(overall, expected_counter, Counter())
            _update_breakdown(by_type, expected_counter, Counter(), 0)
            _update_breakdown(by_category, expected_counter, Counter(), 1)
            _add_counts(normalized_overall, expected_normalized_counter, Counter())
            _update_breakdown(normalized_by_type, expected_normalized_counter, Counter(), 0)
            _update_breakdown(normalized_by_category, expected_normalized_counter, Counter(), 1)
            expected_severities = [
                str(finding["severity"]).upper() for finding in expected_findings
            ]
            critical_expected += sum(
                severity == Severity.CRITICAL.value for severity in expected_severities
            )
            high_or_critical_expected += sum(
                severity in {Severity.CRITICAL.value, Severity.HIGH.value}
                for severity in expected_severities
            )
            high_expected += sum(
                severity == Severity.HIGH.value for severity in expected_severities
            )
            critical_value_expected += sum(
                str(finding["category"]).casefold() == "lab_value" for finding in expected_findings
            )
            for finding in expected_findings:
                for slice_name in _safety_slices_for(finding):
                    safety_slices[slice_name].gold_findings += 1
            case_results.append(
                CaseResult(
                    test_case_id=record["test_case_id"],
                    passed=False,
                    normalized_passed=False,
                    expected=expected_labels,
                    actual=[],
                    expected_normalized=expected_normalized_labels,
                    actual_normalized=[],
                    execution_status="UNASSESSED",
                    coverage_status=CoverageStatus.UNASSESSED.value,
                    claims_unassessed=0,
                    coverage_limitations=[f"BENCHMARK_CASE_FAILED:{type(exc).__name__}"],
                    evidence_valid=False,
                    gold_evidence_matched=False,
                    normalized_gold_evidence_matched=False,
                    false_positives=0,
                    false_negatives=sum(expected_counter.values()),
                    normalized_false_positives=0,
                    normalized_false_negatives=sum(expected_normalized_counter.values()),
                    error=f"BENCHMARK_CASE_FAILED:{type(exc).__name__}",
                )
            )

    return BenchmarkReport(
        benchmark_path=str(benchmark_path),
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        manifest_sha256=(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.exists()
            else None
        ),
        benchmark_id=(str(manifest.get("benchmark_id")) if manifest else None),
        benchmark_version=(str(manifest.get("benchmark_version")) if manifest else None),
        component_version_sets=[
            component_version_sets[key] for key in sorted(component_version_sets)
        ],
        source_revision=source_revision,
        repository_state=repository_state,
        python_version=python_version or platform.python_version(),
        dependency_lock_sha256=_dependency_lock_sha256(benchmark_path),
        cases_total=len(records),
        cases_completed=completed,
        cases_unassessed=unassessed_cases,
        cases_full_coverage=full_coverage_cases,
        cases_reduced_coverage=reduced_coverage_cases,
        cases_partial_coverage=partial_coverage_cases,
        cases_exact_match=exact_matches,
        cases_normalized_match=normalized_matches,
        cases_gold_evidence_match=gold_evidence_matches,
        cases_normalized_gold_evidence_match=normalized_gold_evidence_matches,
        overall=overall,
        normalized_overall=normalized_overall,
        by_finding_type=dict(by_type),
        by_category=dict(by_category),
        normalized_by_finding_type=dict(normalized_by_type),
        normalized_by_category=dict(normalized_by_category),
        critical_gold_findings=critical_expected,
        critical_discrepancy_detection_matches=critical_detection_matched,
        normalized_critical_discrepancy_detection_matches=(normalized_critical_detection_matched),
        critical_discrepancy_detection_recall=(
            critical_detection_matched / critical_expected if critical_expected else None
        ),
        normalized_critical_discrepancy_detection_recall=(
            normalized_critical_detection_matched / critical_expected if critical_expected else None
        ),
        critical_severity_aware_matches=critical_matched,
        normalized_critical_severity_aware_matches=normalized_critical_matched,
        critical_error_recall=(critical_matched / critical_expected if critical_expected else None),
        normalized_critical_error_recall=(
            normalized_critical_matched / critical_expected if critical_expected else None
        ),
        high_or_critical_gold_findings=high_or_critical_expected,
        high_or_critical_severity_aware_matches=high_or_critical_matched,
        normalized_high_or_critical_severity_aware_matches=(normalized_high_or_critical_matched),
        high_or_critical_error_recall=(
            high_or_critical_matched / high_or_critical_expected
            if high_or_critical_expected
            else None
        ),
        normalized_high_or_critical_error_recall=(
            normalized_high_or_critical_matched / high_or_critical_expected
            if high_or_critical_expected
            else None
        ),
        high_gold_findings=high_expected,
        high_severity_aware_matches=high_matched,
        normalized_high_severity_aware_matches=normalized_high_matched,
        high_error_recall=(high_matched / high_expected if high_expected else None),
        normalized_high_error_recall=(
            normalized_high_matched / high_expected if high_expected else None
        ),
        critical_value_gold_findings=critical_value_expected,
        critical_value_severity_aware_matches=critical_value_matched,
        normalized_critical_value_severity_aware_matches=(normalized_critical_value_matched),
        critical_value_error_recall=(
            critical_value_matched / critical_value_expected if critical_value_expected else None
        ),
        normalized_critical_value_error_recall=(
            normalized_critical_value_matched / critical_value_expected
            if critical_value_expected
            else None
        ),
        safety_slice_metrics=safety_slices,
        predicted_uncertain_findings=predicted_uncertain_findings,
        split_metrics=_split_metrics(benchmark_path, case_results),
        evidence_valid_rate=(evidence_valid_cases / completed if completed else 0.0),
        case_results=case_results,
    )

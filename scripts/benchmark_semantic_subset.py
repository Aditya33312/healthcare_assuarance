"""Evaluate the frozen semantic candidate once across immutable splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.compare_engine import compare

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark" / "clinical_faithfulness" / "v0"
DATA_PATH = BENCHMARK_DIR / "semantic-gold.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "semantic-subset-result.json"
LABELS = ("SUPPORTED", "INFERRED", "UNCERTAIN", "CONTRADICTED")
SPLITS = ("development", "validation", "held_out_test", "clinician_calibration")


def _relationship(finding_types: set[str]) -> str:
    if "ALTERED" in finding_types:
        return "CONTRADICTED"
    if "INFERRED" in finding_types:
        return "INFERRED"
    if "UNCERTAIN" in finding_types:
        return "UNCERTAIN"
    if finding_types == {"SUPPORTED"}:
        return "SUPPORTED"
    return "UNCERTAIN"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        true_positive = sum(
            row["gold_relationship"] == label and row["predicted_relationship"] == label
            for row in rows
        )
        false_positive = sum(
            row["gold_relationship"] != label and row["predicted_relationship"] == label
            for row in rows
        )
        false_negative = sum(
            row["gold_relationship"] == label and row["predicted_relationship"] != label
            for row in rows
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    correct = sum(row["gold_relationship"] == row["predicted_relationship"] for row in rows)
    return {
        "case_count": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_f1": round(sum(item["f1"] for item in by_label.values()) / len(LABELS), 4),
        "by_relationship": by_label,
        "coverage": dict(sorted(Counter(str(row["coverage_status"]) for row in rows).items())),
    }


def build_artifact() -> dict[str, Any]:
    data = DATA_PATH.read_bytes()
    records = [json.loads(line) for line in data.decode().splitlines()]
    case_results: list[dict[str, Any]] = []
    for record in records:
        result = compare(record["source_evidence"], record["generated_claim"])
        finding_types = {finding.type.value for finding in result.findings}
        case_results.append(
            {
                "case_id": record["case_id"],
                "split": record["split"],
                "gold_relationship": record["gold_relationship"],
                "predicted_relationship": _relationship(finding_types),
                "coverage_status": result.coverage.status.value,
                "finding_types": sorted(finding_types),
                "reason_codes": sorted({finding.reason_code for finding in result.findings}),
            }
        )
    return {
        "schema_version": "1.0.0",
        "candidate_freeze": "clinical-faithfulness-v1-candidate@0.1.0",
        "semantic_dataset_sha256": hashlib.sha256(data).hexdigest(),
        "semantic_evaluator": "clinical-semantic-cascade@0.1.0",
        "inference_evaluator": "bounded-clinical-inference@0.1.0",
        "annotation_status": "ENGINEERING_NOT_CLINICIAN_GOLD",
        "evaluation_policy": (
            "Development and validation preceded candidate freeze. held_out_test was run "
            "once after freeze and was not used for subsequent semantic tuning."
        ),
        "overall": _metrics(case_results),
        "by_split": {
            split: _metrics([row for row in case_results if row["split"] == split])
            for split in SPLITS
        },
        "case_results": case_results,
    }


def main() -> int:
    OUTPUT_PATH.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote semantic subset result to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from scripts.build_benchmark_expansion import build_expansion_records
from scripts.build_semantic_gold_subset import build_records as build_semantic_records
from scripts.generate_critical_diagnostics import build_artifact
from scripts.generate_semantic_diagnostics import ROOT_CAUSES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = ROOT / "backend" / "app" / "schemas" / "contracts"
BENCHMARK_DIR = ROOT / "benchmark" / "clinical_faithfulness" / "v0"
CASES_PATH = BENCHMARK_DIR / "cases.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
SPLITS_PATH = BENCHMARK_DIR / "splits.json"
CRITICAL_DIAGNOSTICS_PATH = BENCHMARK_DIR / "critical-diagnostics.json"
SEMANTIC_DIAGNOSTICS_PATH = BENCHMARK_DIR / "semantic-diagnostics.json"
BENCHMARK_RESULT_PATH = BENCHMARK_DIR / "benchmark-result.json"
SEMANTIC_GOLD_PATH = BENCHMARK_DIR / "semantic-gold.jsonl"
SEMANTIC_GOLD_MANIFEST_PATH = BENCHMARK_DIR / "semantic-gold-manifest.json"
SEMANTIC_CANDIDATE_PATH = BENCHMARK_DIR / "semantic-candidate-benchmark.json"
SEMANTIC_RESULT_PATH = BENCHMARK_DIR / "semantic-subset-result.json"
REQUIRED_CATEGORIES = {
    "supported_paraphrase",
    "medication_identity",
    "dose",
    "frequency",
    "route",
    "laterality",
    "negation",
    "uncertainty",
    "subject_attribution",
    "lab_values",
    "dates",
    "duration",
    "unsupported_additions",
    "important_omissions",
    "reasonable_inference",
}
REQUIRED_SEEDED_REASONS = {
    "MEDICATION_MISMATCH",
    "DOSE_MISMATCH",
    "FREQUENCY_MISMATCH",
    "ROUTE_MISMATCH",
    "LATERALITY_MISMATCH",
    "NEGATION_REVERSAL",
    "CERTAINTY_CHANGE",
    "SUBJECT_ATTRIBUTION_ERROR",
    "NUMERIC_VALUE_MISMATCH",
    "DATE_MISMATCH",
    "DURATION_MISMATCH",
    "UNSUPPORTED_ADDITION",
    "IMPORTANT_OMISSION",
    "DERIVED_CLINICAL_SUMMARY",
    "SEMANTIC_INFERENCE_UNASSESSED",
}
FINDING_TYPES = {"SUPPORTED", "ADDED", "ALTERED", "OMITTED", "INFERRED", "UNCERTAIN"}


def _contracts() -> dict[str, dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CONTRACT_DIR.glob("*.json"))
    }


def _validator() -> Draft202012Validator:
    contracts = _contracts()
    registry = Registry()
    for schema in contracts.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return Draft202012Validator(contracts["test_case_schema.json"], registry=registry)


def _records() -> list[dict[str, Any]]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in lines), "JSONL must not contain blank records"
    return [json.loads(line) for line in lines]


def _findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [finding for record in records for finding in record["expectation"]["expected_findings"]]


def test_all_200_records_validate_against_the_versioned_test_case_contract() -> None:
    records = _records()
    validator = _validator()

    assert len(records) == 200
    assert [record["test_case_id"] for record in records] == [
        f"cf-v0-{index:03d}" for index in range(1, 201)
    ]
    assert len({record["test_case_id"] for record in records}) == 200
    assert (
        len(
            {
                (
                    record["input"]["source_documents"][0]["content"],
                    record["input"]["generated_output"]["content"],
                )
                for record in records
            }
        )
        == 200
    )
    for record in records:
        validator.validate(record)


def test_benchmark_is_manual_phi_free_and_covers_every_required_category() -> None:
    records = _records()
    covered_tags = {tag for record in records for tag in record["tags"]}

    assert covered_tags >= REQUIRED_CATEGORIES
    for record in records:
        assert record["test_family"] == "CLINICAL_FAITHFULNESS"
        assert record["provenance"]["data_classification"] == "SYNTHETIC"
        assert record["provenance"]["contains_phi"] is False
        assert record["provenance"]["authoring_method"] == "MANUAL"
        assert record["active"] is True


def test_engineering_expansion_is_balanced_distinct_and_provenance_tracked() -> None:
    records = _records()
    expansion = records[50:]
    findings = _findings(expansion)

    assert len(expansion) == 150
    assert expansion == build_expansion_records()
    assert {
        finding_type: sum(finding["type"] == finding_type for finding in findings)
        for finding_type in FINDING_TYPES
    } == {finding_type: 25 for finding_type in FINDING_TYPES}
    assert all("balanced_expansion" in record["tags"] for record in expansion)
    assert all(
        record["provenance"].get("source_reference") == "ENGINEERING_BALANCED_EXPANSION_V0_4"
        for record in expansion
    )
    assert len(
        {
            (
                record["input"]["source_documents"][0]["content"],
                record["input"]["generated_output"]["content"],
            )
            for record in expansion
        }
    ) == len(expansion)


def test_gold_findings_cover_all_labels_and_deliberately_seed_known_errors() -> None:
    findings = _findings(_records())
    finding_types = {finding["type"] for finding in findings}
    reason_codes = {finding["reason_code"] for finding in findings}

    assert finding_types == FINDING_TYPES
    assert reason_codes >= REQUIRED_SEEDED_REASONS
    assert any(finding["type"] == "SUPPORTED" for finding in findings)
    assert any(finding["severity"] == "CRITICAL" for finding in findings)


def test_every_gold_evidence_offset_slices_exact_record_content() -> None:
    for record in _records():
        documents = {
            document["document_id"]: document["content"]
            for document in record["input"]["source_documents"]
        }
        generated = record["input"]["generated_output"]
        documents[generated["document_id"]] = generated["content"]

        for finding in record["expectation"]["expected_findings"]:
            if finding["type"] == "ADDED":
                assert finding["source"] is None
                assert finding["generated"] is not None
            elif finding["type"] == "OMITTED":
                assert finding["source"] is not None
                assert finding["generated"] is None
            elif finding["type"] in {"SUPPORTED", "ALTERED"}:
                assert finding["source"] is not None
                assert finding["generated"] is not None

            for evidence in (finding["source"], finding["generated"]):
                if evidence is None:
                    continue
                content = documents[evidence["document_id"]]
                assert 0 <= evidence["start"] < evidence["end"] <= len(content)
                assert content[evidence["start"] : evidence["end"]] == evidence["text"]


def test_manifest_counts_coverage_taxonomy_and_hash_match_canonical_jsonl() -> None:
    records = _records()
    findings = _findings(records)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()

    assert manifest["benchmark_version"] == "0.4.0"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["case_count"] == len(records) == 200
    assert manifest["finding_count"] == len(findings) == 202
    assert manifest["reproducibility"]["content_hash_algorithm"] == "SHA-256"
    assert manifest["reproducibility"]["content_hash"] == digest
    assert manifest["annotation_policy"]["supported_findings_included"] is True
    assert manifest["annotation_policy"]["exact_evidence_offsets_required"] is True
    assert manifest["gold_taxonomy"]["taxonomy_version"] == "1.0.0"
    assert set(manifest["gold_taxonomy"]["finding_types"]) == FINDING_TYPES
    assert set(manifest["gold_taxonomy"]["categories"]) == {
        finding["category"] for finding in findings
    }
    assert set(manifest["gold_taxonomy"]["reason_codes"]) == {
        finding["reason_code"] for finding in findings
    }

    expected_category_coverage = {
        category: [record["test_case_id"] for record in records if category in record["tags"]]
        for category in REQUIRED_CATEGORIES
    }
    assert manifest["required_category_coverage"] == expected_category_coverage

    expected_type_coverage = {
        finding_type: [
            record["test_case_id"]
            for record in records
            if any(
                finding["type"] == finding_type
                for finding in record["expectation"]["expected_findings"]
            )
        ]
        for finding_type in FINDING_TYPES
    }
    assert manifest["finding_type_coverage"] == expected_type_coverage
    assert manifest["finding_type_counts"] == {
        finding_type: sum(finding["type"] == finding_type for finding in findings)
        for finding_type in FINDING_TYPES
    }
    assert (
        manifest["engine_to_gold_normalization"]["reporting_requirement"]
        == "REPORT_RAW_STRICT_AND_NORMALIZED_METRICS"
    )
    assert manifest["engine_to_gold_normalization"]["contextual_category_aliases"] == []


def test_split_manifest_is_explicit_disjoint_and_honest_about_reserved_sets() -> None:
    records = _records()
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    split_names = ("development", "validation", "held_out_test", "clinician_calibration")
    split_sets = [set(splits[name]) for name in split_names]

    assert splits["benchmark_version"] == "0.4.0"
    assert set.union(*split_sets) == {record["test_case_id"] for record in records}
    for index, split in enumerate(split_sets):
        assert all(split.isdisjoint(other) for other in split_sets[index + 1 :])
    assert [len(splits[name]) for name in split_names] == [110, 30, 30, 30]
    assert splits["historical_exposure"] == "ALL_CASES_VISIBLE_BEFORE_SPLIT_FREEZE"
    assert splits["reserved_split_status"] == "POPULATED_ENGINEERING_LABELS_ONLY"
    assignment = {name: splits[name] for name in split_names}
    digest = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert splits["split_assignment_sha256"] == digest


def test_critical_diagnostics_are_reproducible_and_trace_every_critical_case() -> None:
    checked = json.loads(CRITICAL_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    expected_critical_ids = {
        record["test_case_id"]
        for record in _records()
        if any(
            finding["severity"] == "CRITICAL"
            for finding in record["expectation"]["expected_findings"]
        )
    }

    assert checked == build_artifact()
    assert checked["benchmark_version"] == "0.4.0"
    assert checked["benchmark_sha256"] == hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    assert checked["root_cause_counts"] == {"SEVERITY_MAPPING_ERROR": 4}
    assert {case["case_id"] for case in checked["cases"]} == expected_critical_ids
    for case in checked["cases"]:
        trace = case["pipeline_trace"]
        assert "entities" in trace["source_extraction"]
        assert "claims" in trace["source_extraction"]
        assert "candidate_pairs" in trace["alignment"]
        assert "rules_evaluated" in trace["rule_engine"]
        assert trace["semantic_evaluator"]["called"] is False
        assert trace["finding_classifier"]
        assert trace["severity_engine"]


def test_semantic_gap_snapshot_accounts_for_every_baseline_error_and_coverage_gap() -> None:
    artifact = json.loads(SEMANTIC_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    summary = artifact["summary"]

    assert artifact["analysis_stage"] == "PRE_SEMANTIC_V0_3_BASELINE"
    # This is intentionally an immutable pre-semantic v0.3 snapshot. The active
    # corpus may advance while this before-state remains available for comparison.
    assert artifact["benchmark_sha256"] == (
        "ee06853da156a92027880851728a16dd4407973e4c99642fb84b382eeb451eb5"
    )
    assert artifact["benchmark_result_sha256"] == (
        "0ec66003de352cd5220bc3dcbdbe711fae371c1f46d23427cf97a4bd6a716e53"
    )
    assert summary["normalized_false_positives"] == 118
    assert summary["normalized_false_negatives"] == 62
    assert summary["coverage"] == {
        "FULL": 148,
        "PARTIAL": 50,
        "REDUCED": 1,
        "UNASSESSED": 1,
    }
    assert sum(summary["root_cause_counts"].values()) == len(artifact["cases"]) == 118
    assert all(case["root_cause"] in ROOT_CAUSES for case in artifact["cases"])


def test_semantic_gold_subset_is_balanced_frozen_and_honest_about_annotation_status() -> None:
    data = SEMANTIC_GOLD_PATH.read_text(encoding="utf-8")
    records = [json.loads(line) for line in data.splitlines()]
    manifest = json.loads(SEMANTIC_GOLD_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert records == build_semantic_records()
    assert len(records) == 64
    assert {
        relationship: sum(record["gold_relationship"] == relationship for record in records)
        for relationship in ("SUPPORTED", "INFERRED", "UNCERTAIN", "CONTRADICTED")
    } == {
        relationship: 16 for relationship in ("SUPPORTED", "INFERRED", "UNCERTAIN", "CONTRADICTED")
    }
    assert all(record["annotation_status"] == "engineering" for record in records)
    assert all(record["source_evidence"] and record["generated_claim"] for record in records)
    assert all(record["supporting_span_ids"] for record in records)
    split_names = ("development", "validation", "held_out_test", "clinician_calibration")
    assert [len(manifest["splits"][name]) for name in split_names] == [16, 16, 16, 16]
    split_sets = [set(manifest["splits"][name]) for name in split_names]
    assert set.union(*split_sets) == {record["case_id"] for record in records}
    for index, split in enumerate(split_sets):
        assert all(split.isdisjoint(other) for other in split_sets[index + 1 :])
    assert manifest["content_sha256"] == hashlib.sha256(data.encode()).hexdigest()


def test_semantic_benchmark_artifacts_preserve_scope_provenance_and_freeze() -> None:
    manifest = json.loads(SEMANTIC_GOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
    candidates = json.loads(SEMANTIC_CANDIDATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(SEMANTIC_RESULT_PATH.read_text(encoding="utf-8"))

    dataset_hash = manifest["content_sha256"]
    assert candidates["semantic_dataset_sha256"] == dataset_hash
    assert candidates["evaluated_splits"] == ["development", "validation"]
    assert candidates["held_out_test_touched"] is False
    assert candidates["clinician_calibration_touched"] is False
    assert candidates["candidates"]["clinical_biomedical_nli"]["status"].startswith("NOT_RUN")
    embedding = candidates["candidates"]["biomedical_embedding_alignment"]
    assert embedding["status"] == "RUN_COMPLETE"
    assert embedding["clinical_entailment_metrics"] is None

    assert result["candidate_freeze"] == "clinical-faithfulness-v1-candidate@0.1.0"
    assert result["semantic_dataset_sha256"] == dataset_hash
    assert result["annotation_status"] == "ENGINEERING_NOT_CLINICIAN_GOLD"
    assert result["overall"]["case_count"] == len(result["case_results"]) == 64
    assert set(result["by_split"]) == set(manifest["splits"])
    assert {item["case_id"] for item in result["case_results"]} == {
        case_id for case_ids in manifest["splits"].values() for case_id in case_ids
    }

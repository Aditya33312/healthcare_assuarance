from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from backend.app.domain.enums import (
    AlignmentMethod,
    AssertionStatus,
    ClaimType,
    FindingType,
    Severity,
    SubjectType,
)
from backend.app.domain.models import Claim, Finding, TextSpan

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = ROOT / "backend" / "app" / "schemas" / "contracts"
CONTRACT_NAMES = {
    "entity_schema.json",
    "assertion_schema.json",
    "claim_schema.json",
    "finding_schema.json",
    "test_case_schema.json",
    "expectation_schema.json",
}


def _load_contracts() -> dict[str, dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CONTRACT_DIR.glob("*.json"))
    }


def _registry(contracts: dict[str, dict[str, Any]]) -> Registry:
    registry = Registry()
    for schema in contracts.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def _valid_examples() -> dict[str, dict[str, Any]]:
    source_span = {"document_id": "source_001", "text": "metformin", "start": 0, "end": 9}
    generated_span = {
        "document_id": "generated_output",
        "text": "metformin",
        "start": 0,
        "end": 9,
    }
    finding = {
        "schema_version": "1.0.0",
        "finding_id": "find_001",
        "type": "SUPPORTED",
        "category": "medication_identity",
        "severity": "LOW",
        "source": source_span,
        "generated": generated_span,
        "reason_code": "CLAIM_SUPPORTED",
        "explanation": "The medication identity is preserved.",
        "confidence": 1.0,
        "evaluator": "clinical_compare",
        "evaluator_version": "0.1.0",
        "severity_policy_version": "provisional-clinical-severity@0.1.0",
        "alignment_method": "EXACT",
    }
    expectation = {
        "schema_version": "1.0.0",
        "expectation_type": "EXPECTED_FINDINGS",
        "expected_findings": [finding],
        "allow_additional_findings": False,
    }
    return {
        "entity_schema.json": {
            "schema_version": "1.0.0",
            "entity_id": "entity_001",
            "type": "MEDICATION",
            "text": "metformin",
            "assertion": "PRESENT",
            "temporality": "CURRENT",
            "subject": "PATIENT",
            "certainty": "CERTAIN",
            "laterality": "NOT_APPLICABLE",
            "normalized_value": "metformin",
            "normalized_unit": None,
            "source_span": source_span,
            "confidence": 1.0,
            "extractor": "fixture",
            "extractor_version": "0.1.0",
        },
        "assertion_schema.json": {
            "schema_version": "1.0.0",
            "assertion_id": "assertion_001",
            "status": "ABSENT",
            "subject": "PATIENT",
            "temporality": "CURRENT",
            "certainty": "CERTAIN",
            "cue": {"text": "No", "start": 0, "end": 2},
            "scope": {"text": "No chest pain", "start": 0, "end": 13},
            "confidence": 1.0,
            "detector": "fixture",
            "detector_version": "0.1.0",
        },
        "claim_schema.json": {
            "schema_version": "1.0.0",
            "claim_id": "claim_001",
            "concept": "metformin",
            "type": "MEDICATION",
            "attributes": {
                "dose": 500,
                "unit": "mg",
                "frequency": "TWICE_DAILY",
                "route": "ORAL",
            },
            "assertion": "PRESENT",
            "subject": "PATIENT",
            "source_span": source_span,
            "confidence": 1.0,
            "extractor": "fixture",
            "extractor_version": "0.1.0",
        },
        "finding_schema.json": finding,
        "expectation_schema.json": expectation,
        "test_case_schema.json": {
            "schema_version": "1.0.0",
            "test_case_id": "cf-fixture-001",
            "test_case_version": "1.0.0",
            "name": "Fixture case",
            "test_family": "CLINICAL_FAITHFULNESS",
            "risk_tier": "HIGH",
            "input": {
                "source_documents": [
                    {"document_id": "source_001", "type": "clinical_text", "content": "metformin"}
                ],
                "generated_output": {
                    "document_id": "generated_output",
                    "type": "clinical_text",
                    "content": "metformin",
                },
            },
            "expectation": expectation,
            "tags": ["fixture"],
            "provenance": {
                "data_classification": "SYNTHETIC",
                "contains_phi": False,
                "authoring_method": "MANUAL",
            },
            "active": True,
        },
    }


def _object_schemas(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_object_schemas(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_object_schemas(value))
    return found


def test_exactly_six_versioned_draft_2020_12_contracts_exist() -> None:
    contracts = _load_contracts()

    assert set(contracts) == CONTRACT_NAMES
    assert len({schema["$id"] for schema in contracts.values()}) == 6
    for schema in contracts.values():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].endswith("/1.0.0")


def test_every_declared_object_shape_rejects_unknown_fields() -> None:
    for name, schema in _load_contracts().items():
        object_schemas = _object_schemas(schema)
        assert object_schemas, f"{name} must declare at least one object shape"
        assert all(item.get("additionalProperties") is False for item in object_schemas), name


@pytest.mark.parametrize("contract_name", sorted(CONTRACT_NAMES))
def test_contracts_accept_canonical_examples_and_reject_extra_fields(contract_name: str) -> None:
    contracts = _load_contracts()
    validator = Draft202012Validator(contracts[contract_name], registry=_registry(contracts))
    example = _valid_examples()[contract_name]

    validator.validate(example)
    invalid = copy.deepcopy(example)
    invalid["unexpected_field"] = "must be rejected"
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_contract_enums_include_required_clinical_taxonomies() -> None:
    contracts = _load_contracts()
    assertion_statuses = set(contracts["entity_schema.json"]["$defs"]["assertionStatus"]["enum"])
    entity_types = set(contracts["entity_schema.json"]["$defs"]["entityType"]["enum"])
    finding_categories = set(contracts["finding_schema.json"]["properties"]["category"]["enum"])

    assert assertion_statuses == {
        "PRESENT",
        "ABSENT",
        "POSSIBLE",
        "HISTORICAL",
        "FAMILY_HISTORY",
        "HYPOTHETICAL",
        "CONDITIONAL",
        "UNCERTAIN",
    }
    assert entity_types == {
        "CONDITION",
        "SYMPTOM",
        "MEDICATION",
        "DOSE",
        "UNIT",
        "FREQUENCY",
        "ROUTE",
        "PROCEDURE",
        "BODY_SITE",
        "LATERALITY",
        "ALLERGY",
        "LAB_TEST",
        "LAB_VALUE",
        "DATE",
        "TIME",
        "DURATION",
        "AGE",
    }
    assert finding_categories == {
        "symptom",
        "diagnosis",
        "procedure",
        "allergy",
        "clinical_fact",
        "medication_identity",
        "medication_dose",
        "medication_unit",
        "medication_frequency",
        "medication_route",
        "laterality",
        "assertion",
        "certainty",
        "subject_attribution",
        "lab_value",
        "date",
        "duration",
        "inference",
        "alignment",
    }


def test_finding_evidence_nullability_depends_on_finding_direction() -> None:
    contracts = _load_contracts()
    validator = Draft202012Validator(
        contracts["finding_schema.json"], registry=_registry(contracts)
    )
    supported = _valid_examples()["finding_schema.json"]

    invalid_added = copy.deepcopy(supported)
    invalid_added["type"] = "ADDED"
    with pytest.raises(ValidationError):
        validator.validate(invalid_added)

    valid_added = copy.deepcopy(invalid_added)
    valid_added["source"] = None
    validator.validate(valid_added)

    invalid_omitted = copy.deepcopy(supported)
    invalid_omitted["type"] = "OMITTED"
    with pytest.raises(ValidationError):
        validator.validate(invalid_omitted)

    valid_omitted = copy.deepcopy(invalid_omitted)
    valid_omitted["generated"] = None
    validator.validate(valid_omitted)


def test_runtime_claim_and_finding_serialization_match_locked_contracts() -> None:
    contracts = _load_contracts()
    registry = _registry(contracts)
    source_span = TextSpan(document_id="source_001", text="metformin", start=0, end=9)
    generated_span = TextSpan(document_id="generated_output", text="metformin", start=0, end=9)
    claim = Claim(
        claim_id="claim_runtime_001",
        concept="metformin",
        type=ClaimType.MEDICATION,
        attributes={"dose": 500, "unit": "mg"},
        assertion=AssertionStatus.PRESENT,
        subject=SubjectType.PATIENT,
        source_span=source_span,
    )
    finding = Finding(
        finding_id="finding_runtime_001",
        type=FindingType.SUPPORTED,
        category="medication",
        severity=Severity.LOW,
        source=source_span,
        generated=generated_span,
        reason_code="CLAIM_SUPPORTED",
        explanation="The medication claim is supported.",
        confidence=1.0,
        alignment_method=AlignmentMethod.EXACT,
    )

    Draft202012Validator(contracts["claim_schema.json"], registry=registry).validate(
        claim.model_dump(mode="json")
    )
    Draft202012Validator(contracts["finding_schema.json"], registry=registry).validate(
        finding.model_dump(mode="json")
    )

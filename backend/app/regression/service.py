"""Durably promote only human-validated, PHI-free findings to regression cases."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auditing.service import record_audit_event
from backend.app.database.base import utc_now
from backend.app.database.models import (
    Evaluation,
    FindingRecord,
    GeneratedOutputRecord,
    HumanReview,
    SourceDocumentRecord,
    TestCase,
    TestCaseVersion,
    TestSuite,
    TestSuiteVersion,
    User,
)
from backend.app.domain.enums import Severity
from backend.app.domain.models import Finding
from backend.app.ingestion.normalization import content_hash, raw_serialization
from backend.app.regression.models import RegressionArtifact, RegressionPromotionRequest
from backend.app.security import EncryptedJsonStore
from backend.app.test_selection import TestFamily
from backend.app.test_suites import (
    case_version_ids_from_definition,
    with_case_version_snapshot,
)

_SEMVER_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_TYPED_EVIDENCE_TYPES = {
    "transcript",
    "lab_results",
    "medication_list",
    "structured_json",
    "rag_context",
}
_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MODERATE: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def _valid_no_phi_provenance(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    actor = value.get("attested_by_api_key_id") or value.get("attested_by_user_id")
    return (
        value.get("data_classification") in {"SYNTHETIC", "DE_IDENTIFIED"}
        and value.get("contains_phi") is False
        and value.get("no_phi_attested") is True
        and isinstance(actor, str)
        and bool(actor)
    )


class IneligibleFindingError(ValueError):
    """The requested finding is missing or is not eligible for permanence."""


class DuplicatePromotionError(ValueError):
    """The same tenant finding was already made into a permanent case."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_case_id(organization_id: str, finding_id: str) -> str:
    digest = _sha256(f"{organization_id}\x1f{finding_id}")[:32]
    return f"case_reg_{digest}"


def _next_patch_version(versions: list[str]) -> str:
    if not versions:
        return "1.0.0"
    parsed: list[tuple[int, int, int]] = []
    for version in versions:
        match = _SEMVER_PATTERN.fullmatch(version)
        if match is None:
            raise IneligibleFindingError("The target suite has an unsupported version history.")
        major, minor, patch = (int(part) for part in match.groups())
        parsed.append((major, minor, patch))
    major, minor, patch = max(parsed)
    return f"{major}.{minor}.{patch + 1}"


class RegressionPromotionService:
    """Build a schema-compatible permanent case inside the caller's transaction.

    Clinical content is decrypted only after tenant and validation checks. The
    service emits no logs and writes plaintext only to a suite whose caller has
    explicitly attested that every promoted field is synthetic or de-identified.
    """

    validated_disposition = "VALIDATED"

    def __init__(self, session: Session, encryptor: EncryptedJsonStore) -> None:
        self.session = session
        self.encryptor = encryptor

    def promote(
        self,
        organization_id: str,
        request: RegressionPromotionRequest,
    ) -> RegressionArtifact:
        if not organization_id:
            raise IneligibleFindingError("An organization scope is required.")

        finding_row = self.session.scalar(
            select(FindingRecord).where(
                FindingRecord.organization_id == organization_id,
                FindingRecord.id == request.finding_id,
            )
        )
        review = self.session.scalar(
            select(HumanReview).where(
                HumanReview.organization_id == organization_id,
                HumanReview.id == request.validation_review_id,
                HumanReview.finding_id == request.finding_id,
            )
        )
        if finding_row is None or review is None:
            # Missing and cross-tenant artifacts intentionally look identical.
            raise IneligibleFindingError("The finding or validation review was not found.")
        if review.disposition != self.validated_disposition:
            raise IneligibleFindingError(
                "The finding has not received an eligible human validation."
            )

        eligible_users = set(
            self.session.scalars(
                select(User.id).where(
                    User.organization_id == organization_id,
                    User.id.in_((review.reviewer_user_id, request.promoted_by_user_id)),
                    User.status == "ACTIVE",
                )
            )
        )
        if eligible_users != {review.reviewer_user_id, request.promoted_by_user_id}:
            raise IneligibleFindingError(
                "The validation actors were not found in this organization."
            )

        case_id = _stable_case_id(organization_id, finding_row.id)
        if self.session.scalar(select(TestCase.id).where(TestCase.id == case_id)) is not None:
            raise DuplicatePromotionError("The validated finding was already promoted.")

        evaluation = self.session.scalar(
            select(Evaluation).where(
                Evaluation.organization_id == organization_id,
                Evaluation.id == finding_row.evaluation_id,
            )
        )
        if evaluation is None:
            raise IneligibleFindingError("The finding evaluation was not found.")

        finding = Finding.model_validate(
            self.encryptor.decrypt(finding_row.original_finding_encrypted)
        )
        if not self.encryptor.matches_digest(
            finding.finding_id,
            finding_row.external_finding_id,
            namespace="external-finding-id",
            organization_id=organization_id,
        ):
            raise IneligibleFindingError("The finding artifact failed its integrity check.")
        if review.severity_override is not None:
            try:
                finding = finding.model_copy(
                    update={
                        "severity": Severity(review.severity_override),
                        "severity_policy_version": "human-review@1.0.0",
                    }
                )
            except ValueError as exc:
                raise IneligibleFindingError("The validated severity is not supported.") from exc

        source_documents, document_id_map, normalized_by_document = self._source_documents(
            organization_id,
            evaluation.id,
            finding,
        )
        generated_output, generated_normalized = self._generated_output(
            organization_id,
            evaluation.id,
        )
        document_id_map["generated_output"] = "generated_output"
        normalized_by_document["generated_output"] = generated_normalized
        self._remap_finding_document_ids(finding, document_id_map)
        self._assert_exact_evidence(finding, normalized_by_document)

        suite = self.session.scalar(
            select(TestSuite).where(
                TestSuite.organization_id == organization_id,
                TestSuite.slug == request.target_suite_slug,
            )
        )
        if suite is None:
            suite = TestSuite(
                organization_id=organization_id,
                slug=request.target_suite_slug,
                name=request.target_suite_name,
                suite_type="REGRESSION",
            )
            self.session.add(suite)
            self.session.flush()
        elif not suite.active or suite.suite_type != "REGRESSION":
            raise IneligibleFindingError("The target suite must be an active REGRESSION suite.")

        prior_versions = list(
            self.session.scalars(
                select(TestSuiteVersion)
                .where(
                    TestSuiteVersion.organization_id == organization_id,
                    TestSuiteVersion.test_suite_id == suite.id,
                )
                .order_by(TestSuiteVersion.created_at, TestSuiteVersion.id)
            )
        )
        suite_version_number = _next_patch_version([version.version for version in prior_versions])
        prior_case_entries: list[dict[str, Any]] = []
        prior_case_version_ids: tuple[str, ...] = ()
        inherited_case_attestations: list[dict[str, Any]] = []
        if prior_versions:
            latest = max(
                prior_versions,
                key=lambda item: tuple(int(part) for part in item.version.split(".")),
            )
            if latest.status not in {"ACTIVE", "PUBLISHED"}:
                raise IneligibleFindingError("The target suite's latest version is not executable.")
            prior_case_version_ids = case_version_ids_from_definition(latest.definition)
            raw_case_entries = latest.definition.get("cases", [])
            if isinstance(raw_case_entries, list):
                prior_case_entries = [
                    dict(entry)
                    for entry in raw_case_entries
                    if isinstance(entry, dict)
                    and entry.get("test_case_version_id") in prior_case_version_ids
                ]
            raw_attestations = latest.definition.get("inherited_case_attestations", [])
            if isinstance(raw_attestations, list):
                inherited_case_attestations = [
                    dict(item)
                    for item in raw_attestations
                    if isinstance(item, dict)
                    and _valid_no_phi_provenance(item.get("provenance"))
                    and isinstance(item.get("case_version_ids"), list)
                    and all(
                        isinstance(case_version_id, str)
                        and case_version_id in prior_case_version_ids
                        for case_version_id in item["case_version_ids"]
                    )
                ]

            covered_case_version_ids = {
                entry["test_case_version_id"]
                for entry in prior_case_entries
                if _valid_no_phi_provenance(entry.get("provenance"))
            }
            covered_case_version_ids.update(
                case_version_id
                for item in inherited_case_attestations
                for case_version_id in item["case_version_ids"]
            )
            unattested_case_version_ids = [
                case_version_id
                for case_version_id in prior_case_version_ids
                if case_version_id not in covered_case_version_ids
            ]
            if unattested_case_version_ids:
                prior_suite_provenance = latest.definition.get("provenance")
                if not isinstance(prior_suite_provenance, Mapping) or not _valid_no_phi_provenance(
                    prior_suite_provenance
                ):
                    raise IneligibleFindingError(
                        "The target suite lacks traceable no-PHI attestation lineage."
                    )
                inherited_case_attestations.append(
                    {
                        "source_suite_version_id": latest.id,
                        "source_suite_version": latest.version,
                        "case_version_ids": unattested_case_version_ids,
                        "provenance": dict(prior_suite_provenance),
                    }
                )

        risk_tier = max(
            (request.requested_risk_tier, finding.severity),
            key=_SEVERITY_RANK.__getitem__,
        )
        source_reference = (
            f"evaluation:{evaluation.id}/finding:{finding_row.external_finding_id}"
            f"/review:{review.id}"
        )
        provenance = {
            "data_classification": request.data_classification.value,
            "contains_phi": False,
            "no_phi_attested": request.no_phi_attested,
            "attested_by_user_id": request.promoted_by_user_id,
            "attested_at": utc_now().isoformat(),
            "authoring_method": request.authoring_method.value,
            "source_reference": source_reference,
        }
        expectation = {
            "schema_version": "1.0.0",
            "expectation_type": "EXPECTED_FINDINGS",
            "expected_findings": [
                finding.model_dump(
                    mode="json",
                    exclude={"source_provenance", "generated_provenance"},
                )
            ],
            "allow_additional_findings": True,
        }
        test_input = {
            "source_documents": source_documents,
            "generated_output": generated_output,
        }

        test_case = TestCase(
            id=case_id,
            organization_id=organization_id,
            test_suite_id=suite.id,
            test_family=TestFamily.CLINICAL_FAITHFULNESS.value,
            name=request.test_case_name,
        )
        self.session.add(test_case)
        self.session.flush()
        case_version = TestCaseVersion(
            organization_id=organization_id,
            test_case_id=test_case.id,
            version="1.0.0",
            input_data=test_input,
            expectation=expectation,
        )
        self.session.add(case_version)
        self.session.flush()

        case_entry = {
            "test_case_id": test_case.id,
            "test_case_version_id": case_version.id,
            "test_case_version": case_version.version,
            "risk_tier": risk_tier.value,
            "tags": list(request.tags),
            "provenance": provenance,
            "source_finding_id": finding_row.external_finding_id,
            "validation_review_id": review.id,
        }
        suite_definition = with_case_version_snapshot(
            {
                "schema_version": "1.0.0",
                "suite_kind": "PERMANENT_REGRESSION",
                "cases": [*prior_case_entries, case_entry],
                "inherited_case_attestations": inherited_case_attestations,
            },
            [
                *prior_case_version_ids,
                case_version.id,
            ],
        )
        suite_version = TestSuiteVersion(
            organization_id=organization_id,
            test_suite_id=suite.id,
            version=suite_version_number,
            status="PUBLISHED",
            definition=suite_definition,
        )
        self.session.add(suite_version)
        self.session.flush()

        contract = {
            "schema_version": "1.0.0",
            "test_case_id": test_case.id,
            "test_case_version": case_version.version,
            "name": test_case.name,
            "description": "Regression from a human-validated clinical finding.",
            "test_family": TestFamily(test_case.test_family).name,
            "risk_tier": risk_tier.value,
            "input": test_input,
            "expectation": expectation,
            "tags": list(request.tags),
            "provenance": provenance,
            "active": test_case.active,
        }
        record_audit_event(
            self.session,
            organization_id=organization_id,
            actor_type="USER",
            actor_id=request.promoted_by_user_id,
            action="regression.finding_promoted",
            target_type="test_case_version",
            target_id=case_version.id,
            request_id=request.request_id,
            metadata={
                "suite_version": suite_version.version,
                "case_version": case_version.version,
                "risk_tier": risk_tier.value,
                "validation_disposition": review.disposition,
                "data_classification": request.data_classification.value,
                "no_phi_attested": request.no_phi_attested,
            },
        )
        return RegressionArtifact(
            test_suite_id=suite.id,
            test_suite_version_id=suite_version.id,
            test_suite_version=suite_version.version,
            test_case_id=test_case.id,
            test_case_version_id=case_version.id,
            test_case_version=case_version.version,
            source_finding_id=finding_row.external_finding_id,
            validation_review_id=review.id,
            test_case_contract=contract,
        )

    def _source_documents(
        self,
        organization_id: str,
        evaluation_id: str,
        finding: Finding,
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
        rows = list(
            self.session.scalars(
                select(SourceDocumentRecord)
                .where(
                    SourceDocumentRecord.organization_id == organization_id,
                    SourceDocumentRecord.evaluation_id == evaluation_id,
                )
                .order_by(SourceDocumentRecord.created_at, SourceDocumentRecord.id)
            )
        )
        if not rows:
            raise IneligibleFindingError("The evaluation has no promotable source material.")
        evidence_document_ids = {
            span.document_id
            for span in self._finding_evidence_spans(finding)
            if span.document_id not in {None, "generated_output"}
        }
        documents: list[dict[str, Any]] = []
        document_id_map: dict[str, str] = {}
        normalized_by_document: dict[str, str] = {}
        for index, row in enumerate(rows, start=1):
            stored = self.encryptor.decrypt(row.content_encrypted or b"")
            content, digest_material, stable_hash = self._canonical_stored_content(stored)
            keyed_digest_valid = self.encryptor.matches_digest(
                digest_material,
                row.content_hash,
                namespace="source-content",
                organization_id=organization_id,
            )
            if not keyed_digest_valid and stable_hash != content_hash(digest_material):
                raise IneligibleFindingError("The source artifact failed its integrity check.")
            stored_document_id = (
                stored.get("document_id") if isinstance(stored, dict) else None
            )
            document_id = f"source_{index:03d}"
            if isinstance(stored_document_id, str):
                document_id_map[stored_document_id] = document_id
            else:
                for evidence_document_id in evidence_document_ids:
                    if self.encryptor.matches_digest(
                        evidence_document_id,
                        row.external_document_id_hash,
                        namespace="source-document-id",
                        organization_id=organization_id,
                    ):
                        document_id_map[evidence_document_id] = document_id
            normalized_by_document[document_id] = content
            typed_content = (
                stored.get("raw_content")
                if isinstance(stored, dict) and row.document_type in _TYPED_EVIDENCE_TYPES
                else content
            )
            artifact_type = (
                row.document_type
                if row.document_type in _TYPED_EVIDENCE_TYPES
                else self._regression_artifact_type(row.document_type)
            )
            documents.append(
                {
                    "document_id": document_id,
                    "type": artifact_type,
                    "content": typed_content,
                }
            )
        if not evidence_document_ids.issubset(document_id_map):
            raise IneligibleFindingError("The finding evidence could not be matched to its source.")
        return documents, document_id_map, normalized_by_document

    def _generated_output(
        self,
        organization_id: str,
        evaluation_id: str,
    ) -> tuple[dict[str, Any], str]:
        rows = list(
            self.session.scalars(
                select(GeneratedOutputRecord).where(
                    GeneratedOutputRecord.organization_id == organization_id,
                    GeneratedOutputRecord.evaluation_id == evaluation_id,
                )
            )
        )
        if len(rows) != 1:
            raise IneligibleFindingError("The evaluation output is not uniquely promotable.")
        row = rows[0]
        stored = self.encryptor.decrypt(row.content_encrypted)
        content, digest_material, stable_hash = self._canonical_stored_content(stored)
        keyed_digest_valid = self.encryptor.matches_digest(
            digest_material,
            row.content_hash,
            namespace="generated-content",
            organization_id=organization_id,
        )
        if not keyed_digest_valid and stable_hash != content_hash(digest_material):
            raise IneligibleFindingError("The generated artifact failed its integrity check.")
        typed_content = (
            stored.get("raw_content")
            if isinstance(stored, dict) and row.output_type == "structured_json"
            else content
        )
        artifact_type = (
            row.output_type
            if row.output_type == "structured_json"
            else self._regression_artifact_type(row.output_type)
        )
        return (
            {
                "document_id": "generated_output",
                "type": artifact_type,
                "content": typed_content,
            },
            content,
        )

    @staticmethod
    def _canonical_stored_content(stored: object) -> tuple[str, str, str | None]:
        """Read legacy strings and encrypted universal-evidence envelopes."""

        if isinstance(stored, str):
            return stored, stored, None
        if not isinstance(stored, dict):
            raise IneligibleFindingError("The clinical artifact has an unsupported format.")
        if stored.get("schema_version") != "universal-evidence@1.0":
            raise IneligibleFindingError("The clinical artifact has an unsupported format.")
        normalized = stored.get("normalized_content")
        if not isinstance(normalized, str):
            raise IneligibleFindingError("The clinical artifact has an unsupported format.")
        raw = stored.get("raw_serialized")
        if raw is not None and not isinstance(raw, str):
            raise IneligibleFindingError("The clinical artifact has an unsupported format.")
        stable_hash = stored.get("content_hash")
        if stable_hash is not None and not isinstance(stable_hash, str):
            raise IneligibleFindingError("The clinical artifact has an unsupported format.")
        return normalized, raw or raw_serialization(stored.get("raw_content")), stable_hash

    @staticmethod
    def _regression_artifact_type(source_type: str) -> str:
        return {
            "transcript": "clinical_transcript",
            "lab_results": "structured_data",
            "medication_list": "structured_data",
            "structured_json": "structured_data",
            "rag_context": "document",
        }.get(source_type, source_type)

    @staticmethod
    def _finding_evidence_spans(finding: Finding) -> tuple[Any, ...]:
        return tuple(
            span
            for span in (
                finding.source,
                finding.generated,
                *finding.source_support,
                *finding.candidate_evidence,
            )
            if span is not None
        )

    @classmethod
    def _remap_finding_document_ids(
        cls,
        finding: Finding,
        document_id_map: Mapping[str, str],
    ) -> None:
        for evidence in cls._finding_evidence_spans(finding):
            document_id = evidence.document_id
            if document_id is not None and document_id in document_id_map:
                evidence.document_id = document_id_map[document_id]

    @staticmethod
    def _assert_exact_evidence(
        finding: Finding,
        content_by_document: Mapping[str, str],
    ) -> None:
        for evidence in RegressionPromotionService._finding_evidence_spans(finding):
            if evidence is None or evidence.document_id is None:
                continue
            content = content_by_document.get(evidence.document_id)
            if content is None or content[evidence.start : evidence.end] != evidence.text:
                raise IneligibleFindingError("The finding evidence failed its exact-span check.")

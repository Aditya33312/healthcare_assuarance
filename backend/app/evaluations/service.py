"""Run and persist a clinical comparison without exposing PHI in plaintext storage."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.compare_engine import compare
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.base import new_id
from backend.app.database.models import (
    AISystem,
    AISystemVersion,
    AlignmentRecord,
    ClaimRecord,
    EntityExtraction,
    Evaluation,
    FindingRecord,
    GeneratedOutputRecord,
    SourceDocumentRecord,
)
from backend.app.domain.enums import CoverageStatus, ExecutionStatus
from backend.app.domain.models import (
    Claim,
    CompareResult,
    ComponentVersions,
    CoverageLimitationDetail,
    CoverageSummary,
    EvidenceReference,
    FaithfulnessMetrics,
    Finding,
)
from backend.app.evaluations.models import SingleEvaluationRequest
from backend.app.ingestion import (
    CanonicalEvidenceExtractor,
    IngestionResult,
    UniversalEvidenceIngestionService,
)
from backend.app.observability import MetricName, OperationalMetrics
from backend.app.security import EncryptedJsonStore

CompareCallable = Callable[..., CompareResult]
_COVERAGE_DOCUMENT_REF_PREFIX = "hmac:"


class EvaluationService:
    def __init__(
        self,
        session: Session,
        encryptor: EncryptedJsonStore,
        *,
        compare_fn: CompareCallable = compare,
        retention_days: int = 30,
        metrics: OperationalMetrics | None = None,
        ingestion_service: UniversalEvidenceIngestionService | None = None,
        nlp_service: Any | None = None,
    ) -> None:
        self.session = session
        self.encryptor = encryptor
        self.compare_fn = compare_fn
        self.retention_days = retention_days
        self.metrics = metrics
        self.ingestion_service = ingestion_service or UniversalEvidenceIngestionService()
        self.nlp_service = (
            nlp_service
            if nlp_service is not None
            else ClinicalNLPService() if compare_fn is compare else None
        )
        self.last_ingestion_result: IngestionResult | None = None
        self.last_processing_seconds: float | None = None

    def configure_ingestion(
        self,
        ingestion_service: UniversalEvidenceIngestionService,
        *,
        nlp_service: Any,
    ) -> None:
        """Apply request-scoped limits while keeping constructor compatibility."""

        self.ingestion_service = ingestion_service
        self.nlp_service = nlp_service

    def run(
        self,
        organization_id: str,
        request: SingleEvaluationRequest,
        *,
        system_version_id: str | None = None,
        environment: str = "TEST",
    ) -> CompareResult:
        started_at = time.monotonic()
        normalized_environment = environment.upper()
        if normalized_environment not in {"TEST", "STAGING", "PRODUCTION"}:
            raise ValueError("environment must be TEST, STAGING, or PRODUCTION")
        system = self.session.scalar(
            select(AISystem).where(
                AISystem.organization_id == organization_id,
                AISystem.id == request.system_id,
            )
        )
        if system is None:
            raise GatekeeperError(
                ErrorCode.SYSTEM_NOT_FOUND,
                "The requested AI system was not found.",
                http_status=404,
            )
        if system_version_id is not None:
            registered_version = self.session.scalar(
                select(AISystemVersion).where(
                    AISystemVersion.organization_id == organization_id,
                    AISystemVersion.id == system_version_id,
                    AISystemVersion.system_id == request.system_id,
                    AISystemVersion.version == request.system_version,
                    AISystemVersion.environment == normalized_environment,
                    AISystemVersion.active.is_(True),
                )
            )
            if registered_version is None:
                raise GatekeeperError(
                    ErrorCode.VERSION_NOT_FOUND,
                    "The requested AI system version was not found.",
                    http_status=404,
                )

        ingestion_result = self.ingestion_service.ingest(
            request.sources,
            request.generated_output,
        )
        self.last_ingestion_result = ingestion_result
        compare_kwargs: dict[str, Any] = {}
        if self.nlp_service is not None:
            compare_kwargs["nlp_service"] = CanonicalEvidenceExtractor(
                self.nlp_service,
                (*ingestion_result.source_documents, ingestion_result.generated_document),
                max_claims=self.ingestion_service.limits.max_extracted_claims,
                processing_timeout_seconds=(
                    self.ingestion_service.limits.max_processing_seconds
                ),
            )
        try:
            result = self.compare_fn(
                ingestion_result.source_bundle,
                ingestion_result.generated_output,
                **compare_kwargs,
            )
        except Exception as exc:
            if self.metrics is not None:
                self.metrics.increment(
                    MetricName.EVALUATIONS,
                    execution_status=ExecutionStatus.FAILED.value,
                    test_family="clinical_faithfulness",
                )
                self.metrics.observe(
                    MetricName.EVALUATION_LATENCY,
                    time.monotonic() - started_at,
                    outcome="technical_failure",
                )
            failed = Evaluation(
                organization_id=organization_id,
                system_id=request.system_id,
                system_version_id=system_version_id,
                system_version=request.system_version,
                environment=normalized_environment,
                execution_status=ExecutionStatus.FAILED.value,
                coverage={
                    "status": CoverageStatus.UNASSESSED.value,
                    "claims_assessed": 0,
                    "claims_unassessed": 0,
                    "limitations": ["Clinical comparison could not complete."],
                },
                metrics={},
                component_versions=ComponentVersions().model_dump(mode="json"),
                error_code=ErrorCode.WORKER_FAILED.value,
            )
            self.session.add(failed)
            self.session.flush()
            raise GatekeeperError(
                ErrorCode.WORKER_FAILED,
                "Assurance could not complete the evaluation.",
                retryable=False,
                http_status=500,
                resource_id=failed.id,
            ) from exc

        # CompareEngine IDs are deterministic for reproducibility. Persistence IDs
        # must instead be globally unique because evaluations from different tenants
        # share one primary-key namespace.
        result = result.model_copy(update={"evaluation_id": new_id("eval")})
        if result.execution_status is ExecutionStatus.FAILED:
            result = result.model_copy(update={"error_code": ErrorCode.EVALUATION_FAILED.value})
        result = self._apply_ingestion_context(result, ingestion_result)
        self._persist_result(
            organization_id,
            request,
            result,
            ingestion_result,
            system_version_id=system_version_id,
            environment=normalized_environment,
        )
        if self.metrics is not None:
            self.metrics.increment(
                MetricName.EVALUATIONS,
                execution_status=result.execution_status.value,
                test_family="clinical_faithfulness",
            )
            self.metrics.observe(
                MetricName.EVALUATION_LATENCY,
                time.monotonic() - started_at,
                outcome="success",
            )
        self.last_processing_seconds = time.monotonic() - started_at
        return result

    @staticmethod
    def _provenance_for_claim(claim: Claim | None) -> EvidenceReference | None:
        if claim is None:
            return None
        raw = claim.attributes.get("evidence_provenance")
        if not isinstance(raw, dict):
            return None
        return EvidenceReference.model_validate(raw)

    def _apply_ingestion_context(
        self,
        result: CompareResult,
        ingestion: IngestionResult,
    ) -> CompareResult:
        versions = result.versions.model_copy(
            update={"ingestion": self.ingestion_service.version}
        )
        source_by_span = {
            (
                claim.source_span.document_id,
                claim.source_span.start,
                claim.source_span.end,
            ): claim
            for claim in result.source_claims
        }
        generated_by_span = {
            (
                claim.source_span.document_id,
                claim.source_span.start,
                claim.source_span.end,
            ): claim
            for claim in result.generated_claims
        }
        findings: list[Finding] = []
        for finding in result.findings:
            source_claim = (
                source_by_span.get(
                    (
                        finding.source.document_id,
                        finding.source.start,
                        finding.source.end,
                    )
                )
                if finding.source is not None
                else None
            )
            generated_claim = (
                generated_by_span.get(
                    (
                        finding.generated.document_id,
                        finding.generated.start,
                        finding.generated.end,
                    )
                )
                if finding.generated is not None
                else None
            )
            findings.append(
                finding.model_copy(
                    update={
                        "source_provenance": self._provenance_for_claim(source_claim),
                        "generated_provenance": self._provenance_for_claim(generated_claim),
                    }
                )
            )

        if not ingestion.limitations:
            return result.model_copy(update={"findings": findings, "versions": versions})
        coverage = result.coverage
        status = coverage.status
        if status is CoverageStatus.FULL:
            status = CoverageStatus.REDUCED
        details = [
            CoverageLimitationDetail.model_validate(limitation.model_dump(mode="json"))
            for limitation in ingestion.limitations
        ]
        messages = [
            *coverage.limitations,
            *(
                f"{item.message.rstrip('.')} ({item.code})."
                for item in ingestion.limitations
            ),
        ]
        merged_coverage = coverage.model_copy(
            update={
                "status": status,
                "limitations": list(dict.fromkeys(messages)),
                "limitation_details": [*coverage.limitation_details, *details],
            }
        )
        return result.model_copy(
            update={
                "coverage": merged_coverage,
                "findings": findings,
                "versions": versions,
            }
        )

    def get(self, organization_id: str, evaluation_id: str) -> CompareResult | None:
        evaluation = self.session.scalar(
            select(Evaluation).where(
                Evaluation.organization_id == organization_id,
                Evaluation.id == evaluation_id,
            )
        )
        if evaluation is None or evaluation.execution_status != ExecutionStatus.COMPLETED.value:
            return None
        claims = tuple(
            self.session.scalars(
                select(ClaimRecord)
                .where(
                    ClaimRecord.organization_id == organization_id,
                    ClaimRecord.evaluation_id == evaluation_id,
                )
                .order_by(ClaimRecord.side, ClaimRecord.ordinal)
            )
        )
        source_claims = [
            Claim.model_validate(self.encryptor.decrypt(record.claim_encrypted))
            for record in claims
            if record.side == "SOURCE"
        ]
        generated_claims = [
            Claim.model_validate(self.encryptor.decrypt(record.claim_encrypted))
            for record in claims
            if record.side == "GENERATED"
        ]
        finding_rows = self.session.scalars(
            select(FindingRecord)
            .where(
                FindingRecord.organization_id == organization_id,
                FindingRecord.evaluation_id == evaluation_id,
            )
            .order_by(FindingRecord.ordinal)
        )
        findings = [
            Finding.model_validate(self.encryptor.decrypt(row.original_finding_encrypted))
            for row in finding_rows
        ]
        return CompareResult(
            evaluation_id=evaluation.id,
            execution_status=ExecutionStatus.COMPLETED,
            coverage=self._coverage_from_storage(
                organization_id,
                evaluation_id,
                evaluation.coverage,
            ),
            faithfulness=FaithfulnessMetrics.model_validate(evaluation.metrics),
            findings=findings,
            source_claims=source_claims,
            generated_claims=generated_claims,
            versions=ComponentVersions.model_validate(evaluation.component_versions),
        )

    def _coverage_document_ref(self, organization_id: str, document_id: str) -> str:
        return _COVERAGE_DOCUMENT_REF_PREFIX + self.encryptor.digest(
            document_id,
            namespace="coverage-document-id",
            organization_id=organization_id,
        )

    def _coverage_for_storage(
        self,
        organization_id: str,
        coverage: CoverageSummary,
    ) -> dict[str, Any]:
        stored = coverage.model_dump(mode="json")
        raw_details = stored.get("limitation_details")
        if not isinstance(raw_details, list):
            return stored
        for raw_detail in raw_details:
            if not isinstance(raw_detail, dict):
                continue
            document_id = raw_detail.get("document_id")
            if isinstance(document_id, str) and document_id != "generated_output":
                raw_detail["document_id"] = self._coverage_document_ref(
                    organization_id,
                    document_id,
                )
        return stored

    def _coverage_from_storage(
        self,
        organization_id: str,
        evaluation_id: str,
        stored_coverage: dict[str, Any],
    ) -> CoverageSummary:
        restored = {**stored_coverage}
        raw_details = stored_coverage.get("limitation_details")
        if not isinstance(raw_details, list):
            return CoverageSummary.model_validate(restored)

        reference_to_document_id: dict[str, str] = {}
        source_rows = self.session.scalars(
            select(SourceDocumentRecord).where(
                SourceDocumentRecord.organization_id == organization_id,
                SourceDocumentRecord.evaluation_id == evaluation_id,
            )
        )
        for row in source_rows:
            encrypted = row.content_encrypted
            if encrypted is None:
                continue
            envelope = self.encryptor.decrypt(encrypted)
            if not isinstance(envelope, dict):
                continue
            document_id = envelope.get("document_id")
            if isinstance(document_id, str):
                reference_to_document_id[
                    self._coverage_document_ref(organization_id, document_id)
                ] = document_id

        restored_details: list[Any] = []
        for raw_detail in raw_details:
            if not isinstance(raw_detail, dict):
                restored_details.append(raw_detail)
                continue
            detail = {**raw_detail}
            document_ref = detail.get("document_id")
            if isinstance(document_ref, str):
                detail["document_id"] = reference_to_document_id.get(
                    document_ref,
                    document_ref,
                )
            restored_details.append(detail)
        restored["limitation_details"] = restored_details
        return CoverageSummary.model_validate(restored)

    def _persist_result(
        self,
        organization_id: str,
        request: SingleEvaluationRequest,
        result: CompareResult,
        ingestion: IngestionResult,
        *,
        system_version_id: str | None,
        environment: str,
    ) -> None:
        evaluation = Evaluation(
            id=result.evaluation_id,
            organization_id=organization_id,
            system_id=request.system_id,
            system_version_id=system_version_id,
            system_version=request.system_version,
            environment=environment,
            execution_status=result.execution_status.value,
            coverage=self._coverage_for_storage(organization_id, result.coverage),
            metrics=result.faithfulness.model_dump(mode="json"),
            component_versions=result.versions.model_dump(mode="json"),
            error_code=(
                ErrorCode.EVALUATION_FAILED.value
                if result.execution_status is ExecutionStatus.FAILED
                else None
            ),
        )
        self.session.add(evaluation)
        self.session.flush()
        retention_expires_at = (
            datetime.now(UTC) + timedelta(days=self.retention_days)
            if self.retention_days
            else datetime.now(UTC)
        )
        canonical_by_id = {
            document.document_id: document for document in ingestion.source_documents
        }
        for document in request.sources:
            canonical = canonical_by_id[document.document_id]
            self.session.add(
                SourceDocumentRecord(
                    organization_id=organization_id,
                    evaluation_id=evaluation.id,
                    external_document_id_hash=self.encryptor.digest(
                        document.document_id,
                        namespace="source-document-id",
                        organization_id=organization_id,
                    ),
                    document_type=document.type,
                    content_encrypted=self.encryptor.encrypt(canonical.encrypted_envelope()),
                    content_hash=self.encryptor.digest(
                        canonical.raw_serialized,
                        namespace="source-content",
                        organization_id=organization_id,
                    ),
                    retention_expires_at=retention_expires_at,
                )
            )
        output = request.generated_output
        canonical_output = ingestion.generated_document
        self.session.add(
            GeneratedOutputRecord(
                organization_id=organization_id,
                evaluation_id=evaluation.id,
                output_type=output.type,
                content_encrypted=self.encryptor.encrypt(canonical_output.encrypted_envelope()),
                content_hash=self.encryptor.digest(
                    canonical_output.raw_serialized,
                    namespace="generated-content",
                    organization_id=organization_id,
                ),
                retention_expires_at=retention_expires_at,
            )
        )
        claim_record_ids_by_span: dict[tuple[str, str, int, int], str] = {}
        for side, claims in (
            ("SOURCE", result.source_claims),
            ("GENERATED", result.generated_claims),
        ):
            for ordinal, claim in enumerate(claims):
                serialized_claim = claim.model_dump(mode="json")
                document_id = claim.source_span.document_id or ""
                document_id_hash = self.encryptor.digest(
                    document_id,
                    namespace="claim-document-id",
                    organization_id=organization_id,
                )
                claim_record = ClaimRecord(
                    id=new_id("claim"),
                    organization_id=organization_id,
                    evaluation_id=evaluation.id,
                    external_claim_id=self.encryptor.digest(
                        claim.claim_id,
                        namespace="external-claim-id",
                        organization_id=organization_id,
                    ),
                    document_id_hash=document_id_hash,
                    side=side,
                    ordinal=ordinal,
                    claim_type=claim.type.value,
                    concept_hash=self.encryptor.digest(
                        claim.concept.casefold(),
                        namespace="claim-concept",
                        organization_id=organization_id,
                    ),
                    claim_encrypted=self.encryptor.encrypt(serialized_claim),
                    span_start=claim.source_span.start,
                    span_end=claim.source_span.end,
                    confidence=claim.confidence,
                    extractor_version=claim.extractor_version,
                )
                self.session.add(claim_record)
                claim_record_ids_by_span[
                    (
                        side,
                        document_id,
                        claim.source_span.start,
                        claim.source_span.end,
                    )
                ] = claim_record.id
                self.session.add(
                    EntityExtraction(
                        organization_id=organization_id,
                        evaluation_id=evaluation.id,
                        document_id_hash=document_id_hash,
                        entity_type=claim.type.value,
                        extraction_encrypted=self.encryptor.encrypt(serialized_claim),
                        span_start=claim.source_span.start,
                        span_end=claim.source_span.end,
                        model_version=claim.extractor_version,
                    )
                )
        # Alignment rows reference ClaimRecord primary keys, but there are no ORM
        # relationships for SQLAlchemy to use when ordering these
        # pending objects, so persist the claims before adding their dependants.
        self.session.flush()
        for ordinal, finding in enumerate(result.findings):
            serialized = finding.model_dump(mode="json")
            source_claim_id = None
            if finding.source is not None:
                source_claim_id = claim_record_ids_by_span.get(
                    (
                        "SOURCE",
                        finding.source.document_id or "",
                        finding.source.start,
                        finding.source.end,
                    )
                )
            generated_claim_id = None
            if finding.generated is not None:
                generated_claim_id = claim_record_ids_by_span.get(
                    (
                        "GENERATED",
                        finding.generated.document_id or "",
                        finding.generated.start,
                        finding.generated.end,
                    )
                )
            self.session.add(
                AlignmentRecord(
                    organization_id=organization_id,
                    evaluation_id=evaluation.id,
                    source_claim_id=source_claim_id,
                    generated_claim_id=generated_claim_id,
                    method=finding.alignment_method.value,
                    confidence=finding.confidence,
                    status=finding.type.value,
                    explanation_encrypted=self.encryptor.encrypt(
                        {
                            "finding_id": finding.finding_id,
                            "reason_code": finding.reason_code,
                            "explanation": finding.explanation,
                        }
                    ),
                )
            )
            self.session.add(
                FindingRecord(
                    organization_id=organization_id,
                    evaluation_id=evaluation.id,
                    external_finding_id=self.encryptor.digest(
                        finding.finding_id,
                        namespace="external-finding-id",
                        organization_id=organization_id,
                    ),
                    ordinal=ordinal,
                    finding_type=finding.type.value,
                    category=finding.category.value,
                    severity=finding.severity.value,
                    reason_code=finding.reason_code,
                    explanation_hash=self.encryptor.digest(
                        finding.explanation,
                        namespace="finding-explanation",
                        organization_id=organization_id,
                    ),
                    confidence=finding.confidence,
                    source_evidence_encrypted=(
                        self.encryptor.encrypt(finding.source.model_dump(mode="json"))
                        if finding.source
                        else None
                    ),
                    generated_evidence_encrypted=(
                        self.encryptor.encrypt(finding.generated.model_dump(mode="json"))
                        if finding.generated
                        else None
                    ),
                    evaluator=finding.evaluator,
                    evaluator_version=finding.evaluator_version,
                    rule_id=finding.rule_id,
                    rule_version=finding.rule_version,
                    severity_policy_version=finding.severity_policy_version,
                    original_finding_encrypted=self.encryptor.encrypt(serialized),
                )
            )
        self.session.flush()

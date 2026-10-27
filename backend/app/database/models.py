"""Versioned SQLAlchemy entities for Phase 1 Gatekeeper artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base, TenantMixin, TimestampMixin, new_id, utc_now


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("org"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", nullable=False)


class User(Base, TenantMixin, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("organization_id", "email", name="uq_user_org_email"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", nullable=False)


class Role(Base, TenantMixin, TimestampMixin):
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_role_org_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("role"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    permissions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)


class UserRole(Base, TenantMixin):
    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role_id", name="uq_user_role"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ur"))
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("roles.id", ondelete="CASCADE"), nullable=False
    )


class ApiKey(Base, TenantMixin, TimestampMixin):
    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("organization_id", "key_hash", name="uq_api_key_org_hash"),
        Index("ix_api_key_prefix_status", "key_prefix", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("key"))
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AISystem(Base, TenantMixin, TimestampMixin):
    __tablename__ = "ai_systems"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_system_org_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("sys"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    application_type: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AISystemVersion(Base, TenantMixin, TimestampMixin):
    __tablename__ = "ai_system_versions"
    __table_args__ = (
        UniqueConstraint(
            "system_id",
            "version",
            "environment",
            name="uq_system_version_environment",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("sysv"))
    system_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_systems.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    environment: Mapped[str] = mapped_column(String(32), nullable=False)
    endpoint_url: Mapped[str | None] = mapped_column(Text)
    authentication_config_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    request_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    response_schema: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ApplicationProfileRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "application_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("profile"))
    system_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_systems.id", ondelete="CASCADE"), nullable=False, index=True
    )
    profile_version: Mapped[str] = mapped_column(String(32), nullable=False)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class TestCatalogEntry(Base, TimestampMixin):
    __tablename__ = "test_catalog"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    family: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    evaluator_key: Mapped[str] = mapped_column(String(100), nullable=False)
    implementation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class TestSuite(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_suites"
    __table_args__ = (UniqueConstraint("organization_id", "slug", name="uq_suite_org_slug"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("suite"))
    slug: Mapped[str] = mapped_column(String(160), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    suite_type: Mapped[str] = mapped_column(String(32), default="CUSTOM", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class TestSuiteVersion(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_suite_versions"
    __table_args__ = (UniqueConstraint("test_suite_id", "version", name="uq_suite_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("suitev"))
    test_suite_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_suites.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", nullable=False)


class TestCase(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("case"))
    test_suite_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_suites.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_family: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class TestCaseVersion(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_case_versions"
    __table_args__ = (UniqueConstraint("test_case_id", "version", name="uq_case_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("casev"))
    test_case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    expectation: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class TestSelectionRule(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_selection_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("selrule"))
    rule_id: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class TestPlanRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_plans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("plan"))
    system_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_systems.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    confirmed_by_user_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("users.id"))
    confirmed_by_api_key_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("api_keys.id")
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReleasePolicy(Base, TenantMixin, TimestampMixin):
    __tablename__ = "release_policies"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_policy_org_name"),
        UniqueConstraint("organization_id", "policy_key", name="uq_policy_org_policy_key"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("policy"))
    policy_key: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ReleasePolicyVersion(Base, TenantMixin, TimestampMixin):
    __tablename__ = "release_policy_versions"
    __table_args__ = (UniqueConstraint("release_policy_id", "version", name="uq_policy_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("policyv"))
    release_policy_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("release_policies.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    rules: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    required_coverage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class TestRun(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("run"))
    system_id: Mapped[str] = mapped_column(String(64), ForeignKey("ai_systems.id"), nullable=False)
    system_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("ai_system_versions.id"), nullable=False
    )
    test_suite_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_suite_versions.id"), nullable=False
    )
    release_policy_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("release_policy_versions.id")
    )
    execution_status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    release_decision: Mapped[str | None] = mapped_column(String(16))
    trigger_source: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    coverage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    component_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(100))
    failure_message: Mapped[str | None] = mapped_column(Text)


class TestRunCase(Base, TenantMixin, TimestampMixin):
    __tablename__ = "test_run_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("runcase"))
    test_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_case_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_case_versions.id"), nullable=False
    )
    execution_status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    release_decision: Mapped[str | None] = mapped_column(String(16))
    output_data_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(100))


class Evaluation(Base, TenantMixin, TimestampMixin):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("eval"))
    system_id: Mapped[str] = mapped_column(String(64), ForeignKey("ai_systems.id"), nullable=False)
    system_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("ai_system_versions.id"), index=True
    )
    system_version: Mapped[str] = mapped_column(String(100), nullable=False)
    environment: Mapped[str] = mapped_column(String(32), default="TEST", nullable=False)
    test_run_case_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("test_run_cases.id", ondelete="SET NULL"), index=True
    )
    execution_status: Mapped[str] = mapped_column(String(32), nullable=False)
    coverage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    component_versions: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100))


class SourceDocumentRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("src"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_document_id_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GeneratedOutputRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "generated_outputs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("output"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    output_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ClaimRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "claims"
    __table_args__ = (
        UniqueConstraint("evaluation_id", "external_claim_id", name="uq_evaluation_claim"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("claim"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_claim_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_id_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(64), nullable=False)
    concept_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    claim_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(160), nullable=False)


class EntityExtraction(Base, TenantMixin, TimestampMixin):
    __tablename__ = "entity_extractions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("entity"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(160), nullable=False)


class AlignmentRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "alignments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("align"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_claim_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("claims.id"))
    generated_claim_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("claims.id"))
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    explanation_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class FindingRecord(Base, TenantMixin, TimestampMixin):
    __tablename__ = "findings"
    __table_args__ = (
        UniqueConstraint("evaluation_id", "external_finding_id", name="uq_evaluation_finding"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("finding"))
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_finding_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    finding_type: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(100), nullable=False)
    explanation_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source_evidence_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    generated_evidence_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    evaluator: Mapped[str] = mapped_column(String(100), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(100), nullable=False)
    rule_id: Mapped[str | None] = mapped_column(String(160))
    rule_version: Mapped[str | None] = mapped_column(String(32))
    severity_policy_version: Mapped[str] = mapped_column(String(160), nullable=False)
    original_finding_encrypted: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class HumanReview(Base, TenantMixin, TimestampMixin):
    __tablename__ = "human_reviews"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("review"))
    finding_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("findings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id"), nullable=False
    )
    disposition: Mapped[str] = mapped_column(String(32), nullable=False)
    severity_override: Mapped[str | None] = mapped_column(String(32))
    comment_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)
    vendor_response_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary)


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_hash: Mapped[str | None] = mapped_column(String(128))
    acceptance_status: Mapped[str] = mapped_column(String(32), nullable=False)
    benchmark_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class RuleVersion(Base, TimestampMixin):
    __tablename__ = "rule_versions"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    rule_id: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    definition_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class SeverityPolicyVersion(Base, TimestampMixin):
    __tablename__ = "severity_policy_versions"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class AuditEvent(Base, TenantMixin):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("audit"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(160), nullable=False)
    target_type: Mapped[str] = mapped_column(String(100), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(128))
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class IdempotencyRecord(Base, TenantMixin):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "operation", "key_digest", name="uq_idempotency_scope"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("idem"))
    operation: Mapped[str] = mapped_column(String(160), nullable=False)
    key_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_body: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

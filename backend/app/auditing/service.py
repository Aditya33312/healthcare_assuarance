"""Create append-only audit records without clinical payloads."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from backend.app.database.models import AuditEvent

_FORBIDDEN_METADATA_FRAGMENTS = {
    "answer",
    "clinical",
    "content",
    "document",
    "evidence",
    "generated",
    "patient",
    "phi",
    "prompt",
    "source",
    "text",
    "transcript",
}
_SAFE_ATTESTATION_FIELDS = {"no_phi_attested"}


def _assert_metadata_safe(value: Any, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SAFE_ATTESTATION_FIELDS:
                if child is not True:
                    raise ValueError(f"audit safety attestation must be true: {path}.{key}")
                continue
            if any(fragment in normalized for fragment in _FORBIDDEN_METADATA_FRAGMENTS):
                raise ValueError(f"audit metadata field is not PHI-safe: {path}.{key}")
            _assert_metadata_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_metadata_safe(child, f"{path}[{index}]")
    elif isinstance(value, str) and len(value) > 256:
        raise ValueError(f"audit metadata string is too long: {path}")


def record_audit_event(
    session: Session,
    *,
    organization_id: str,
    actor_type: str,
    actor_id: str | None,
    action: str,
    target_type: str,
    target_id: str | None,
    request_id: str,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    safe_metadata = metadata or {}
    _assert_metadata_safe(safe_metadata)
    event = AuditEvent(
        organization_id=organization_id,
        actor_type=actor_type,
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        request_id=request_id,
        event_metadata=safe_metadata,
    )
    session.add(event)
    session.flush()
    return event

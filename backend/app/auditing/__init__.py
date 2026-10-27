"""Append-only, PHI-safe audit events."""

from backend.app.auditing.service import record_audit_event

__all__ = ["record_audit_event"]

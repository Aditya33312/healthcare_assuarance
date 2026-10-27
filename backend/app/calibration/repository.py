"""Tenant-scoped repository boundary for calibration workflow records."""

from __future__ import annotations

from typing import Protocol

from backend.app.calibration.models import CalibrationVersion


class CalibrationRepository(Protocol):
    def add(self, organization_id: str, record: CalibrationVersion) -> None: ...

    def get(self, organization_id: str, calibration_id: str) -> CalibrationVersion | None: ...

    def save(self, organization_id: str, record: CalibrationVersion) -> None: ...


class InMemoryCalibrationRepository:
    """Reference adapter for tests and pilots; snapshots prevent shared mutation."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], CalibrationVersion] = {}
        self._versions: set[tuple[str, str, str]] = set()

    def add(self, organization_id: str, record: CalibrationVersion) -> None:
        self._assert_scope(organization_id, record)
        key = (organization_id, record.calibration_id)
        version_key = (organization_id, record.severity_policy_id, record.version)
        if key in self._records or version_key in self._versions:
            raise ValueError("calibration version already exists")
        self._records[key] = record.model_copy(deep=True)
        self._versions.add(version_key)

    def get(self, organization_id: str, calibration_id: str) -> CalibrationVersion | None:
        if not organization_id:
            raise ValueError("organization scope is required")
        record = self._records.get((organization_id, calibration_id))
        return record.model_copy(deep=True) if record is not None else None

    def save(self, organization_id: str, record: CalibrationVersion) -> None:
        self._assert_scope(organization_id, record)
        key = (organization_id, record.calibration_id)
        if key not in self._records:
            raise LookupError("calibration version not found")
        self._records[key] = record.model_copy(deep=True)

    @staticmethod
    def _assert_scope(organization_id: str, record: CalibrationVersion) -> None:
        if not organization_id or record.organization_id != organization_id:
            raise ValueError("calibration organization does not match authenticated tenant")

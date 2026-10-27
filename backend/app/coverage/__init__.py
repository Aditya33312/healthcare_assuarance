"""Coverage aggregation public API."""

from backend.app.coverage.engine import (
    COVERAGE_ENGINE_VERSION,
    CoverageReport,
    TestCoverageRecord,
    calculate_coverage,
)
from backend.app.domain.enums import CoverageStatus

__all__ = [
    "COVERAGE_ENGINE_VERSION",
    "CoverageReport",
    "CoverageStatus",
    "TestCoverageRecord",
    "calculate_coverage",
]

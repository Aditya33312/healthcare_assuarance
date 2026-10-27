"""Stable API scopes used by integrations and CI keys."""

from enum import StrEnum


class ApiScope(StrEnum):
    SYSTEMS_CREATE = "systems:create"
    SYSTEMS_READ = "systems:read"
    EVALUATIONS_CREATE = "evaluations:create"
    EVALUATIONS_READ = "evaluations:read"
    TEST_RUNS_CREATE = "test-runs:create"
    TEST_RUNS_READ = "test-runs:read"
    TEST_SUITES_MANAGE = "test-suites:manage"
    TEST_PLANS_MANAGE = "test-plans:manage"
    RELEASE_POLICIES_MANAGE = "release-policies:manage"
    REPORTS_READ = "reports:read"
    API_KEYS_MANAGE = "api-keys:manage"


ALL_SCOPES = frozenset(scope.value for scope in ApiScope)

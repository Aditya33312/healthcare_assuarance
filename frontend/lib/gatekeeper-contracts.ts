import "server-only";

const EXECUTION_STATUSES = new Set([
  "QUEUED",
  "PREPARING",
  "RUNNING",
  "EVALUATING",
  "AGGREGATING",
  "COMPLETED",
  "FAILED",
  "CANCELLED"
]);
const COVERAGE_STATUSES = new Set(["FULL", "REDUCED", "PARTIAL", "UNASSESSED"]);
const RELEASE_DECISIONS = new Set(["PASS", "WARN", "BLOCK"]);
const PLAN_STATUSES = new Set(["DRAFT", "CONFIRMED"]);
const EXCLUSION_REASONS = new Set([
  "QUOTA_EXCEEDED",
  "NOT_APPLICABLE",
  "NOT_SELECTED",
  "INACTIVE"
]);
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

export type JsonRecord = Record<string, unknown>;

export function isJsonRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && typeof value === "number" && value >= 0;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isNonEmptyString);
}

function isUniqueStringArray(value: unknown): value is string[] {
  return isStringArray(value) && new Set(value).size === value.length;
}

function isStringList(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isListPayload(
  value: unknown,
  itemValidator: (item: unknown) => boolean
): boolean {
  if (Array.isArray(value)) return value.every(itemValidator);
  if (!isJsonRecord(value) || !Array.isArray(value.items)) return false;
  if (!value.items.every(itemValidator)) return false;
  return value.count === undefined || (
    isNonNegativeInteger(value.count) && value.count === value.items.length
  );
}

export function isSystemPayload(value: unknown): value is JsonRecord {
  if (!isJsonRecord(value)) return false;
  if (
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isNonEmptyString(value.application_type) ||
    !isNonEmptyString(value.risk_tier) ||
    typeof value.active !== "boolean"
  ) {
    return false;
  }
  if (value.application_profile !== null && !isJsonRecord(value.application_profile)) {
    return false;
  }
  return value.test_selection === null || isJsonRecord(value.test_selection);
}

export function isSystemVersionPayload(value: unknown): value is JsonRecord {
  return isJsonRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.system_id) &&
    isNonEmptyString(value.version) &&
    isNonEmptyString(value.environment) &&
    typeof value.active === "boolean";
}

function isSuitePayload(value: unknown): boolean {
  if (
    !isJsonRecord(value) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    typeof value.active !== "boolean"
  ) {
    return false;
  }
  if (value.latest_version === null) return true;
  return isJsonRecord(value.latest_version) &&
    isNonEmptyString(value.latest_version.version) &&
    isNonEmptyString(value.latest_version.status);
}

function isPolicyPayload(value: unknown): boolean {
  if (
    !isJsonRecord(value) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    typeof value.active !== "boolean"
  ) {
    return false;
  }
  return value.latest_version === null || (
    isJsonRecord(value.latest_version) &&
    isNonEmptyString(value.latest_version.version)
  );
}

function isTextSpan(value: unknown): boolean {
  return value === null || (
    isJsonRecord(value) &&
    typeof value.text === "string" &&
    isNonNegativeInteger(value.start) &&
    isNonNegativeInteger(value.end) &&
    value.start <= value.end
  );
}

function isFindingPayload(value: unknown): boolean {
  return isJsonRecord(value) &&
    isNonEmptyString(value.finding_id) &&
    isNonEmptyString(value.type) &&
    isNonEmptyString(value.category) &&
    isNonEmptyString(value.severity) &&
    isNonEmptyString(value.reason_code) &&
    typeof value.explanation === "string" &&
    typeof value.confidence === "number" &&
    Number.isFinite(value.confidence) &&
    value.confidence >= 0 &&
    value.confidence <= 1 &&
    isTextSpan(value.source ?? null) &&
    isTextSpan(value.generated ?? null);
}

interface PlannedFamilyPayload {
  test_family: string;
  cases: number;
}

interface CaseBindingPayload {
  case_version_id: string;
  test_family: string;
}

interface CaseExclusionPayload extends CaseBindingPayload {
  reason: string;
}

function isPlannedFamily(value: unknown): value is PlannedFamilyPayload {
  return isJsonRecord(value) &&
    isNonEmptyString(value.test_family) &&
    isNonNegativeInteger(value.cases) &&
    value.cases > 0;
}

function isCaseBinding(value: unknown): value is CaseBindingPayload {
  return isJsonRecord(value) &&
    isNonEmptyString(value.case_version_id) &&
    isNonEmptyString(value.test_family);
}

function isCaseExclusion(value: unknown): value is CaseExclusionPayload {
  return isJsonRecord(value) &&
    isCaseBinding(value) &&
    isNonEmptyString(value.reason) &&
    EXCLUSION_REASONS.has(value.reason);
}

export function isTestPlanPayload(value: unknown): value is JsonRecord {
  if (!isJsonRecord(value)) return false;
  if (
    !isNonEmptyString(value.id) ||
    value.test_plan_id !== value.id ||
    !isNonEmptyString(value.application_id) ||
    !isNonEmptyString(value.application_version) ||
    !isNonEmptyString(value.application_environment) ||
    !isNonEmptyString(value.test_plan_version) ||
    !isNonEmptyString(value.status) ||
    !PLAN_STATUSES.has(value.status) ||
    !isNonEmptyString(value.profile_fingerprint) ||
    !SHA256_PATTERN.test(value.profile_fingerprint) ||
    !isNonEmptyString(value.test_suite_id) ||
    !isNonEmptyString(value.test_suite_version) ||
    !isNonEmptyString(value.source_suite_case_snapshot_digest) ||
    !SHA256_PATTERN.test(value.source_suite_case_snapshot_digest) ||
    !isNonEmptyString(value.suite_case_snapshot_digest) ||
    !SHA256_PATTERN.test(value.suite_case_snapshot_digest) ||
    !isNonEmptyString(value.release_policy_snapshot_digest) ||
    !SHA256_PATTERN.test(value.release_policy_snapshot_digest) ||
    !isNonEmptyString(value.test_suite_content_snapshot_digest) ||
    !SHA256_PATTERN.test(value.test_suite_content_snapshot_digest) ||
    !isUniqueStringArray(value.source_suite_case_version_ids) ||
    !isUniqueStringArray(value.suite_case_version_ids) ||
    !Array.isArray(value.suite_case_bindings) ||
    !value.suite_case_bindings.every(isCaseBinding) ||
    !Array.isArray(value.excluded_suite_cases) ||
    !value.excluded_suite_cases.every(isCaseExclusion) ||
    !Array.isArray(value.required) ||
    !value.required.every(isPlannedFamily) ||
    !Array.isArray(value.recommended) ||
    !value.recommended.every(isPlannedFamily) ||
    !Array.isArray(value.optional) ||
    !value.optional.every(isPlannedFamily) ||
    !isUniqueStringArray(value.unselected_optional) ||
    !isUniqueStringArray(value.not_applicable) ||
    !isJsonRecord(value.release_policy) ||
    !isNonEmptyString(value.release_policy.policy_id) ||
    !isNonEmptyString(value.release_policy.policy_version)
  ) {
    return false;
  }

  const suiteCaseVersionIds = value.suite_case_version_ids as string[];
  const sourceSuiteCaseVersionIds = value.source_suite_case_version_ids as string[];
  const suiteCaseBindings = value.suite_case_bindings as CaseBindingPayload[];
  const excludedSuiteCases = value.excluded_suite_cases as CaseExclusionPayload[];
  const bindingIds = suiteCaseBindings.map((item) => item.case_version_id);
  if (
    bindingIds.length !== suiteCaseVersionIds.length ||
    bindingIds.some((item, index) => item !== suiteCaseVersionIds[index])
  ) {
    return false;
  }
  const excludedIds = excludedSuiteCases.map((item) => item.case_version_id);
  if (new Set(excludedIds).size !== excludedIds.length) return false;
  const partitionIds = [...bindingIds, ...excludedIds];
  if (
    new Set(partitionIds).size !== partitionIds.length ||
    partitionIds.length !== sourceSuiteCaseVersionIds.length
  ) {
    return false;
  }
  const sourceIds = new Set(sourceSuiteCaseVersionIds);
  if (!partitionIds.every((item) => sourceIds.has(item))) return false;

  const plannedFamilies = [value.required, value.recommended, value.optional]
    .flat()
    .map((item) => item.test_family);
  const familyPartition = [
    ...plannedFamilies,
    ...value.unselected_optional,
    ...value.not_applicable
  ];
  if (new Set(familyPartition).size !== familyPartition.length) return false;

  if (value.status === "CONFIRMED") {
    return isNonEmptyString(value.confirmed_by_api_key_id) &&
      isNonEmptyString(value.confirmed_at);
  }
  return value.confirmed_by_api_key_id === null && value.confirmed_at === null;
}

function isCoveragePayload(value: unknown): value is JsonRecord {
  if (
    !isJsonRecord(value) ||
    !isNonEmptyString(value.engine_version) ||
    !isNonEmptyString(value.status) ||
    !COVERAGE_STATUSES.has(value.status) ||
    !isNonNegativeInteger(value.planned_tests) ||
    !isNonNegativeInteger(value.planned_cases) ||
    !isNonNegativeInteger(value.completed_cases) ||
    !isNonNegativeInteger(value.technical_failed_cases) ||
    !isNonNegativeInteger(value.unassessed_cases) ||
    !isNonNegativeInteger(value.pending_cases) ||
    !isUniqueStringArray(value.missing_required_tests) ||
    !isUniqueStringArray(value.degraded_tests) ||
    typeof value.valid_for_release !== "boolean" ||
    !isStringList(value.limitations)
  ) {
    return false;
  }

  const accounted = value.completed_cases +
    value.technical_failed_cases +
    value.unassessed_cases +
    value.pending_cases;
  if (accounted !== value.planned_cases) return false;
  if (value.planned_tests === 0 && value.planned_cases !== 0) return false;

  if (value.status === "FULL") {
    if (
      value.planned_tests === 0 ||
      value.planned_cases === 0 ||
      value.completed_cases !== value.planned_cases ||
      value.technical_failed_cases !== 0 ||
      value.unassessed_cases !== 0 ||
      value.pending_cases !== 0 ||
      value.missing_required_tests.length !== 0 ||
      value.degraded_tests.length !== 0 ||
      !value.valid_for_release
    ) {
      return false;
    }
  } else if (value.status === "REDUCED") {
    if (
      value.planned_cases === 0 ||
      value.completed_cases === 0 ||
      value.missing_required_tests.length !== 0 ||
      (
        value.technical_failed_cases === 0 &&
        value.unassessed_cases === 0 &&
        value.pending_cases === 0 &&
        value.degraded_tests.length === 0
      )
    ) {
      return false;
    }
  } else if (value.status === "PARTIAL") {
    if (
      value.planned_cases === 0 ||
      value.completed_cases === 0 ||
      value.missing_required_tests.length === 0 ||
      value.valid_for_release
    ) {
      return false;
    }
  } else if (
    (value.planned_cases > 0 && value.completed_cases > 0) ||
    value.valid_for_release
  ) {
    return false;
  }

  if (value.valid_for_release && (
    !["FULL", "REDUCED"].includes(value.status) ||
    value.technical_failed_cases !== 0 ||
    value.pending_cases !== 0 ||
    value.missing_required_tests.length !== 0
  )) {
    return false;
  }
  return true;
}

function isMetricsPayload(value: unknown): boolean {
  if (value === null) return true;
  if (!isJsonRecord(value)) return false;
  for (const key of ["critical_count", "high_count"]) {
    const metric = value[key];
    if (metric !== undefined && !isNonNegativeInteger(metric)) return false;
  }
  return true;
}

export function isTestRunPayload(value: unknown): value is JsonRecord {
  if (
    !isJsonRecord(value) ||
    !isNonEmptyString(value.run_id) ||
    !isNonEmptyString(value.system_id) ||
    !isNonEmptyString(value.system_version) ||
    !isNonEmptyString(value.execution_status) ||
    !EXECUTION_STATUSES.has(value.execution_status) ||
    !["ASSESSED", "UNASSESSED"].includes(String(value.assessment_status)) ||
    !isCoveragePayload(value.coverage) ||
    !isMetricsPayload(value.metrics) ||
    !(
      value.release_decision === null ||
      (isNonEmptyString(value.release_decision) && RELEASE_DECISIONS.has(value.release_decision))
    )
  ) {
    return false;
  }

  const decision = value.release_decision;
  if (value.execution_status !== "COMPLETED") return decision === null;
  if (decision === null) return true;
  if (decision === "PASS") {
    return value.assessment_status === "ASSESSED" &&
      value.coverage.status === "FULL" &&
      value.coverage.valid_for_release === true;
  }
  if (decision === "WARN") {
    return value.assessment_status === "ASSESSED" &&
      ["FULL", "REDUCED"].includes(String(value.coverage.status)) &&
      value.coverage.valid_for_release === true;
  }
  return value.coverage.technical_failed_cases === 0;
}

export function validateGatekeeperPayload(path: `/${string}`, value: unknown): boolean {
  const pathname = path.split("?", 1)[0];
  if (pathname === "/v1/systems") return isListPayload(value, isSystemPayload);
  if (pathname === "/v1/test-suites") return isListPayload(value, isSuitePayload);
  if (pathname === "/v1/release-policies") return isListPayload(value, isPolicyPayload);
  if (pathname === "/v1/test-plans") return isListPayload(value, isTestPlanPayload);
  if (pathname === "/v1/test-runs") return isListPayload(value, isTestRunPayload);
  if (pathname === "/v1/findings") return isListPayload(value, isFindingPayload);
  if (/^\/v1\/test-plans\/[^/]+$/.test(pathname)) return isTestPlanPayload(value);
  if (/^\/v1\/test-runs\/[^/]+$/.test(pathname)) return isTestRunPayload(value);
  if (/^\/v1\/test-runs\/[^/]+\/findings$/.test(pathname)) {
    return isListPayload(value, isFindingPayload);
  }
  return false;
}

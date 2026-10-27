export type Decision = "PASS" | "WARN" | "BLOCK" | null;
export type ExecutionStatus =
  | "QUEUED"
  | "PREPARING"
  | "RUNNING"
  | "EVALUATING"
  | "AGGREGATING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface SystemSummary {
  id: string;
  name: string;
  application_type: string;
  risk_tier: string;
  application_profile?: { version?: string } | null;
  updated_at?: string;
}

export interface RunSummary {
  run_id: string;
  system_id: string;
  system_version?: string;
  execution_status: ExecutionStatus;
  release_decision: Decision;
  metrics?: {
    critical_count?: number;
    high_count?: number;
    [key: string]: unknown;
  } | null;
  created_at?: string;
  coverage?: { status: string; limitations?: string[] } | null;
}

export interface FindingSummary {
  finding_id: string;
  type: string;
  category: string;
  severity: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  reason_code: string;
  explanation: string;
  confidence: number;
  source?: { text: string; start: number; end: number } | null;
  generated?: { text: string; start: number; end: number } | null;
}

export interface PlannedFamily {
  test_family: string;
  cases: number;
}

export interface SuiteCaseBinding {
  case_version_id: string;
  test_family: string;
}

export interface TestPlanSummary {
  id: string;
  test_plan_id: string;
  application_id: string;
  application_version: string;
  application_environment: string;
  test_plan_version: string;
  status: "DRAFT" | "CONFIRMED";
  profile_fingerprint: string;
  test_suite_id: string;
  test_suite_version: string;
  source_suite_case_version_ids: string[];
  source_suite_case_snapshot_digest: string;
  suite_case_version_ids: string[];
  suite_case_bindings: SuiteCaseBinding[];
  suite_case_snapshot_digest: string;
  excluded_suite_cases: Array<{
    case_version_id: string;
    test_family: string;
    reason: string;
  }>;
  required: PlannedFamily[];
  recommended: PlannedFamily[];
  optional: PlannedFamily[];
  unselected_optional: string[];
  not_applicable: string[];
  release_policy: { policy_id: string; policy_version: string };
  release_policy_snapshot_digest: string;
  test_suite_content_snapshot_digest: string;
  confirmed_by_api_key_id?: string | null;
  confirmed_at?: string | null;
  created_at?: string;
}

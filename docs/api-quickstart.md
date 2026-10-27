# API quickstart: governance artifacts to queued run

This local example creates every prerequisite explicitly. It uses `jq` to carry
opaque IDs between calls and deliberately creates only one synthetic clinical-
faithfulness case. The resulting run demonstrates the workflow; it is not a
clinically complete suite and cannot justify release.

Use a governance key with `systems:create`, `systems:read`,
`test-suites:manage`, `release-policies:manage`, `test-plans:manage`,
`test-runs:create`, and `test-runs:read`. A CI key should normally have only the
last two scopes and must consume already approved artifacts.

```bash
export GATEKEEPER_URL=http://127.0.0.1:8000
export GATEKEEPER_KEY='<one-time Gatekeeper API key>'

SYSTEM_ID=$(
  curl -fsS "$GATEKEEPER_URL/v1/systems" \
    -H "Authorization: Bearer $GATEKEEPER_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
      "name":"Quickstart scribe",
      "application_type":"CLINICAL_SCRIBE",
      "risk_tier":"HIGH",
      "profile":{
        "version":"1.0.0",
        "intended_use":"Synthetic quickstart only",
        "input_type":"CLINICAL_TEXT",
        "output_type":"CLINICAL_NOTE",
        "clinician_facing":true,
        "environment":"TEST",
        "authentication_type":"NONE"
      }
    }' | jq -r .id
)

curl -fsS "$GATEKEEPER_URL/v1/systems/$SYSTEM_ID/versions" \
  -H "Authorization: Bearer $GATEKEEPER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"version":"1.0.0","environment":"TEST"}' >/dev/null

SUITE_ID=$(
  curl -fsS "$GATEKEEPER_URL/v1/test-suites" \
    -H "Authorization: Bearer $GATEKEEPER_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
      "slug":"quickstart-faithfulness",
      "name":"Quickstart faithfulness",
      "version":"1.0.0",
      "status":"ACTIVE",
      "data_classification":"SYNTHETIC",
      "contains_phi":false,
      "no_phi_attested":true,
      "cases":[{
        "name":"Preserved hypertension",
        "test_family":"clinical_faithfulness",
        "version":"1.0.0",
        "input_data":{
          "source_documents":[{"document_id":"source_001","content":"Hypertension is documented."}],
          "generated_output":{"content":"The patient has hypertension."}
        },
        "expectation":{"expected_findings":[]}
      }]
    }' | jq -r .id
)

POLICY_ID=$(
  curl -fsS "$GATEKEEPER_URL/v1/release-policies" \
    -H "Authorization: Bearer $GATEKEEPER_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
      "name":"Quickstart fail-closed policy",
      "version":"1.0.0",
      "missing_required_coverage_action":"NO_DECISION",
      "critical_finding_override":true,
      "rules":[{
        "rule_id":"no-critical-findings",
        "metric":"critical_count",
        "operator":"<=",
        "threshold":0,
        "on_failure":"BLOCK",
        "tier":"BLOCKING"
      }]
    }' | jq -r .id
)

PLAN=$(
  curl -fsS "$GATEKEEPER_URL/v1/systems/$SYSTEM_ID/test-plans" \
    -H "Authorization: Bearer $GATEKEEPER_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"release_policy_id\":\"$POLICY_ID\",\"test_suite_id\":\"$SUITE_ID\",\"test_suite_version\":\"1.0.0\",\"test_plan_version\":\"1.0\",\"case_counts\":{\"clinical_faithfulness\":1}}"
)
PLAN_ID=$(printf '%s' "$PLAN" | jq -r .test_plan_id)

# Recovery/review is a separate read. Inspect bindings, exclusions, policy and
# all content/snapshot digests before confirmation.
PLAN=$(
  curl -fsS "$GATEKEEPER_URL/v1/test-plans/$PLAN_ID" \
    -H "Authorization: Bearer $GATEKEEPER_KEY"
)
printf '%s\n' "$PLAN" | jq '{test_plan_id,status,suite_case_bindings,excluded_suite_cases,release_policy_snapshot_digest,test_suite_content_snapshot_digest}'

FINGERPRINT=$(printf '%s' "$PLAN" | jq -r .profile_fingerprint)
PLAN_VERSION=$(printf '%s' "$PLAN" | jq -r .test_plan_version)
curl -fsS "$GATEKEEPER_URL/v1/test-plans/$PLAN_ID/confirm" \
  -H "Authorization: Bearer $GATEKEEPER_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"confirm\":true,\"expected_profile_fingerprint\":\"$FINGERPRINT\",\"expected_test_plan_version\":\"$PLAN_VERSION\"}" >/dev/null

RUN_ID=$(
  curl -fsS "$GATEKEEPER_URL/v1/test-runs" \
    -H "Authorization: Bearer $GATEKEEPER_KEY" \
    -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: quickstart-run-attempt-1' \
    -d "{\"system_id\":\"$SYSTEM_ID\",\"system_version\":\"1.0.0\",\"environment\":\"TEST\",\"test_suite_id\":\"$SUITE_ID\",\"test_suite_version\":\"1.0.0\",\"test_plan_id\":\"$PLAN_ID\",\"source\":{\"trigger\":\"quickstart\"}}" | jq -r .run_id
)

curl -fsS "$GATEKEEPER_URL/v1/test-runs/$RUN_ID" \
  -H "Authorization: Bearer $GATEKEEPER_KEY" | jq .
```

`POST /v1/test-runs` returns `202 QUEUED`; QUEUED is not a release decision.
Run a worker in a second process to execute it:

```bash
python scripts/gatekeeper_worker.py
```

The worker claims queued runs, calls the registered endpoint, evaluates the
captured outputs, persists coverage and metrics, and applies the release policy.
Use `--once` for a one-shot local verification. A production deployment should
run the worker as a separate supervised service beside the API.

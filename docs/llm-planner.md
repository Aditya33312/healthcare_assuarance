# Optional grounded LLM test planner

Gatekeeper now includes an opt-in Anthropic planner for reviewing application
test-family suggestions. It is advisory only:

1. The application profile is sanitized before it is sent to Anthropic.
2. Endpoint URLs, credentials, patient data, and application outputs are not
   included in the prompt.
3. The deterministic selection engine remains authoritative.
4. The planner output is schema-validated against Gatekeeper's closed test
   family catalog.
5. A planner preview does not mutate or confirm a test plan.
6. The release policy and worker remain fully deterministic.

## Configuration

Keep the feature disabled by default. Add these values to a local `.env` only
when the integration has been approved:

```dotenv
GATEKEEPER_LLM_PLANNER_ENABLED=false
GATEKEEPER_LLM_PROVIDER=anthropic
GATEKEEPER_LLM_MODEL=
GATEKEEPER_ANTHROPIC_API_KEY=
GATEKEEPER_LLM_TIMEOUT_SECONDS=30
```

Never commit the API key. The key is read only by the server-side planner and
is never returned in API responses.

## Preview endpoint

With the feature enabled and a configured Anthropic key, call:

```text
POST /v1/systems/{system_id}/test-selection/llm-preview
```

The endpoint returns the deterministic selection, the validated LLM proposal,
accepted suggestions, and rejected suggestions. It does not create, change, or
confirm a test plan. An operator must review the preview and continue through
the normal test-plan workflow.

The API key used for this endpoint needs the `test-plans:manage` scope.

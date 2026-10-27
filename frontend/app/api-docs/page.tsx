export default function ApiGuidePage() {
  return <>
    <div className="page-header"><div><span className="eyebrow">Developer guide</span><h1>Gatekeeper API</h1><p>Use a scoped Bearer key and an Idempotency-Key for evaluations and test runs.</p></div></div>
    <article className="panel prose"><div className="panel-body">
      <h2>Authentication</h2><pre className="callout mono">Authorization: Bearer gk_test_…</pre>
      <h2>Issue a least-privilege CI key</h2><p>An existing <code>api-keys:manage</code> key can create, list, and revoke tenant keys. The plaintext secret appears only in the create response.</p>
      <pre className="callout mono">POST /v1/api-keys{"\n"}{'{"environment":"test","scopes":["test-runs:create","test-runs:read"]}'}{"\n\n"}GET /v1/api-keys{"\n"}POST /v1/api-keys/&lt;key_id&gt;/revoke</pre>
      <h2>Create prerequisites</h2><p>First register a system and executable version, create an ACTIVE/PUBLISHED no-PHI test suite, and create a versioned release policy. Suite and policy authoring require their management scopes and should not use the CI key.</p>
      <pre className="callout mono">POST /v1/systems{"\n"}POST /v1/systems/&lt;system_id&gt;/versions{"\n"}POST /v1/test-suites{"\n"}POST /v1/release-policies</pre>
      <h2>Review and confirm a test plan</h2><p>Create a draft for the registered system, recover and review the exact ordered case bindings, exclusions, fingerprints, and content digests, then confirm that same version before execution.</p>
      <pre className="callout mono">POST /v1/systems/&lt;system_id&gt;/test-plans{"\n"}{'{"release_policy_id":"<policy_id>","test_suite_id":"<suite_id>","test_suite_version":"<suite_version>"}'}{"\n\n"}GET /v1/test-plans/&lt;test_plan_id&gt;{"\n\n"}POST /v1/test-plans/&lt;test_plan_id&gt;/confirm{"\n"}{'{"confirm":true,"expected_profile_fingerprint":"sha256:<fingerprint>","expected_test_plan_version":"1.0"}'}</pre>
      <h2>Create a test run</h2><pre className="callout mono">POST /v1/test-runs{"\n"}Idempotency-Key: &lt;unique attempt key&gt;{"\n\n"}{'{"system_id":"<system_id>","system_version":"<version>","environment":"staging","test_suite_id":"<suite_id>","test_suite_version":"<suite_version>","test_plan_id":"<confirmed_plan_id>"}'}</pre>
      <h2>Fail-closed behavior</h2><p>A completed run may produce PASS, WARN, or BLOCK. FAILED/CANCELLED execution and missing required coverage produce no valid PASS.</p>
    </div></article>
  </>;
}

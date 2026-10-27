"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

interface PlanSystem {
  id: string;
  name: string;
  test_selection?: { optional_tests?: string[] } | null;
}

interface PlanSuite {
  id: string;
  name: string;
  latest_version?: { version?: string; status?: string } | null;
}

interface PlanPolicy {
  id: string;
  name: string;
  latest_version?: { version?: string } | null;
}

export function NewPlanForm({
  systems,
  suites,
  policies
}: {
  systems: PlanSystem[];
  suites: PlanSuite[];
  policies: PlanPolicy[];
}) {
  const router = useRouter();
  const executableSuites = useMemo(
    () => suites.filter((suite) => {
      const status = suite.latest_version?.status?.toUpperCase();
      return Boolean(
        suite.latest_version?.version &&
        status &&
        ["ACTIVE", "PUBLISHED"].includes(status)
      );
    }),
    [suites]
  );
  const versionedPolicies = useMemo(
    () => policies.filter((policy) => policy.latest_version?.version),
    [policies]
  );
  const [systemId, setSystemId] = useState(systems[0]?.id ?? "");
  const [suiteId, setSuiteId] = useState(executableSuites[0]?.id ?? "");
  const [policyId, setPolicyId] = useState(versionedPolicies[0]?.id ?? "");
  const [planVersion, setPlanVersion] = useState("1.0");
  const [caseCounts, setCaseCounts] = useState("{}");
  const [selectedOptional, setSelectedOptional] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedSystem = systems.find((system) => system.id === systemId);
  const selectedSuite = executableSuites.find((suite) => suite.id === suiteId);
  const selectedPolicy = versionedPolicies.find((policy) => policy.id === policyId);
  const optionalFamilies = selectedSystem?.test_selection?.optional_tests ?? [];

  function changeSystem(value: string) {
    setSystemId(value);
    setSelectedOptional([]);
  }

  function toggleOptional(family: string, checked: boolean) {
    setSelectedOptional((current) => checked
      ? [...current, family]
      : current.filter((item) => item !== family));
  }

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const parsedCounts = JSON.parse(caseCounts) as unknown;
      if (
        parsedCounts === null ||
        Array.isArray(parsedCounts) ||
        typeof parsedCounts !== "object"
      ) {
        throw new Error("Case-count overrides must be a JSON object.");
      }
      if (!selectedSuite?.latest_version?.version || !selectedPolicy?.latest_version?.version) {
        throw new Error("Choose a versioned suite and release policy.");
      }
      const response = await fetch(
        `/api/gatekeeper/systems/${encodeURIComponent(systemId)}/test-plans`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            release_policy_id: policyId,
            release_policy_version: selectedPolicy.latest_version.version,
            test_suite_id: suiteId,
            test_suite_version: selectedSuite.latest_version.version,
            test_plan_version: planVersion,
            case_counts: parsedCounts,
            include_optional: selectedOptional
          })
        }
      );
      const body = await response.json().catch(() => ({}));
      if (!response.ok || typeof body?.test_plan_id !== "string") {
        throw new Error(body?.error?.message ?? "The draft plan could not be generated.");
      }
      router.push(`/plans/${body.test_plan_id}`);
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The draft plan could not be generated.");
    } finally {
      setSubmitting(false);
    }
  }

  const unavailable = !systems.length || !executableSuites.length || !versionedPolicies.length;
  return <section className="panel"><div className="panel-header"><h2>Draft inputs</h2><span className="muted">No execution occurs here</span></div><div className="panel-body">
    {unavailable ? <div className="callout warning">A registered system, executable suite version, and release-policy version are all required before drafting.</div> : <div className="form-grid">
      <div className="field"><label htmlFor="plan-system">AI system</label><select id="plan-system" value={systemId} onChange={(event) => changeSystem(event.target.value)}>{systems.map((system) => <option value={system.id} key={system.id}>{system.name} · {system.id}</option>)}</select></div>
      <div className="field"><label htmlFor="plan-version">Plan version</label><input id="plan-version" value={planVersion} onChange={(event) => setPlanVersion(event.target.value)} /></div>
      <div className="field"><label htmlFor="plan-suite">Test suite</label><select id="plan-suite" value={suiteId} onChange={(event) => setSuiteId(event.target.value)}>{executableSuites.map((suite) => <option value={suite.id} key={suite.id}>{suite.name} · v{suite.latest_version?.version}</option>)}</select></div>
      <div className="field"><label htmlFor="plan-policy">Release policy</label><select id="plan-policy" value={policyId} onChange={(event) => setPolicyId(event.target.value)}>{versionedPolicies.map((policy) => <option value={policy.id} key={policy.id}>{policy.name} · v{policy.latest_version?.version}</option>)}</select></div>
      <div className="field full"><label htmlFor="case-counts">Case-count overrides</label><textarea className="mono" id="case-counts" value={caseCounts} onChange={(event) => setCaseCounts(event.target.value)} /><span className="muted">JSON object keyed by selected test family, for example {`{"clinical_faithfulness": 40}`}.</span></div>
      <div className="field full"><label>Optional families</label>{optionalFamilies.length ? optionalFamilies.map((family) => <label className="check-row" key={family}><input type="checkbox" checked={selectedOptional.includes(family)} onChange={(event) => toggleOptional(family, event.target.checked)} />{family}</label>) : <span className="muted">This profile has no optional families.</span>}</div>
    </div>}
    {error ? <div className="callout warning" style={{marginTop: 16}}>{error}</div> : null}
    <div className="wizard-actions"><span className="muted">The next screen exposes exact cases, exclusions, quotas, and digests for review.</span><button className="button primary" disabled={submitting || unavailable || !planVersion} onClick={submit}>{submitting ? "Generating…" : "Generate draft"}</button></div>
  </div></section>;
}

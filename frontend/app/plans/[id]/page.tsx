import Link from "next/link";
import { notFound } from "next/navigation";

import { ApiErrorState } from "@/components/api-error-state";
import { ConfirmPlanButton } from "@/components/confirm-plan-button";
import { gatekeeperFetch } from "@/lib/api";
import type { PlannedFamily, TestPlanSummary } from "@/lib/types";

function FamilyList({ title, families }: { title: string; families: PlannedFamily[] }) {
  return <article className="panel"><div className="panel-header"><h2>{title}</h2><span className="muted">{families.length}</span></div><div className="panel-body">
    {families.length ? <ul>{families.map((family) => <li key={family.test_family}><span className="mono">{family.test_family}</span> · {family.cases} planned cases</li>)}</ul> : <p className="muted">None</p>}
  </div></article>;
}

export default async function TestPlanDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const response = await gatekeeperFetch<TestPlanSummary>(`/v1/test-plans/${encodeURIComponent(id)}`);
  if (!response.ok && response.error.code === "UPSTREAM_NOT_FOUND") notFound();
  if (!response.ok) return <><div className="page-header"><div><span className="eyebrow">Test plan</span><h1 className="mono">{id}</h1></div></div><ApiErrorState error={response.error} title="Test plan unavailable" /></>;
  const plan = response.data;
  return <>
    <div className="page-header">
      <div><span className="eyebrow">Immutable review artifact</span><h1 className="mono">{plan.test_plan_id}</h1><p>System {plan.application_id} · {plan.application_environment} · profile {plan.application_version}</p></div>
      <div className="stack"><span className={`badge ${plan.status === "CONFIRMED" ? "pass" : "warn"}`}>{plan.status}</span>{plan.status === "CONFIRMED" ? <Link className="button primary" href={`/runs/new?test_plan_id=${encodeURIComponent(plan.test_plan_id)}&system_id=${encodeURIComponent(plan.application_id)}&environment=${encodeURIComponent(plan.application_environment.toLowerCase())}&test_suite_id=${encodeURIComponent(plan.test_suite_id)}&test_suite_version=${encodeURIComponent(plan.test_suite_version)}`}>Configure run</Link> : null}</div>
    </div>
    <section className="metrics">
      <article className="metric-card"><span>Source snapshot</span><strong>{plan.source_suite_case_version_ids.length}</strong><small>Cases present in the suite version</small></article>
      <article className="metric-card good"><span>Executable snapshot</span><strong>{plan.suite_case_version_ids.length}</strong><small>Exact ordered cases approved to run</small></article>
      <article className="metric-card warn"><span>Excluded</span><strong>{plan.excluded_suite_cases.length}</strong><small>Quota, applicability, inactive, or optional</small></article>
      <article className="metric-card"><span>Policy</span><strong>{plan.release_policy.policy_version}</strong><small className="mono">{plan.release_policy.policy_id}</small></article>
    </section>
    <div className="grid-2">
      <div className="stack"><FamilyList title="Required" families={plan.required} /><FamilyList title="Recommended" families={plan.recommended} /><FamilyList title="Selected optional" families={plan.optional} /></div>
      <div className="stack">
        <article className="panel"><div className="panel-header"><h2>Snapshot integrity</h2></div><div className="panel-body prose"><p><strong>Profile</strong><br /><span className="mono">{plan.profile_fingerprint}</span></p><p><strong>Release policy content</strong><br /><span className="mono">{plan.release_policy_snapshot_digest}</span></p><p><strong>Suite and case content</strong><br /><span className="mono">{plan.test_suite_content_snapshot_digest}</span></p><p><strong>Source case IDs</strong><br /><span className="mono">{plan.source_suite_case_snapshot_digest}</span></p><p><strong>Executable case IDs</strong><br /><span className="mono">{plan.suite_case_snapshot_digest}</span></p></div></article>
        <article className="panel"><div className="panel-header"><h2>Exact executable cases</h2><span className="muted">ordered</span></div><div className="panel-body">{plan.suite_case_bindings.length ? <ol>{plan.suite_case_bindings.map((item) => <li key={item.case_version_id}><span className="mono">{item.case_version_id}</span><br /><span className="muted">{item.test_family}</span></li>)}</ol> : <p className="muted">No executable case versions are bound.</p>}</div></article>
        <article className="panel"><div className="panel-header"><h2>Explicit exclusions</h2></div><div className="panel-body">{plan.excluded_suite_cases.length ? <ul>{plan.excluded_suite_cases.map((item) => <li key={item.case_version_id}><span className="mono">{item.case_version_id}</span><br />{item.test_family} · <strong>{item.reason}</strong></li>)}</ul> : <p className="muted">No suite cases were excluded.</p>}</div></article>
        {plan.status === "DRAFT" ? <article className="panel"><div className="panel-header"><h2>Confirmation</h2></div><div className="panel-body"><div className="callout warning">Confirm only after reviewing the family quotas, exact executable count, exclusions, policy version, and all five integrity digests.</div><div style={{marginTop: 16}}><ConfirmPlanButton planId={plan.test_plan_id} profileFingerprint={plan.profile_fingerprint} planVersion={plan.test_plan_version} /></div></div></article> : <div className="callout">Confirmed {plan.confirmed_at ? new Date(plan.confirmed_at).toLocaleString() : "at an unrecorded time"} by API key <span className="mono">{plan.confirmed_by_api_key_id ?? "unknown"}</span>.</div>}
      </div>
    </div>
  </>;
}

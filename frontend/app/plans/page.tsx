import Link from "next/link";

import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { gatekeeperFetch } from "@/lib/api";
import type { TestPlanSummary } from "@/lib/types";

type PlansResponse = TestPlanSummary[] | { items: TestPlanSummary[] };

export default async function TestPlansPage() {
  const response = await gatekeeperFetch<PlansResponse>("/v1/test-plans");
  const header = <div className="page-header">
    <div><span className="eyebrow">Governance review</span><h1>Test plans</h1><p>Drafts bind an application profile, policy, suite snapshot, exact executable cases, quotas, and exclusions before confirmation.</p></div>
    <Link className="button primary" href="/plans/new">Generate draft</Link>
  </div>;
  if (!response.ok) return <>{header}<ApiErrorState error={response.error} title="Test plans unavailable" /></>;
  const plans = Array.isArray(response.data) ? response.data : response.data.items;
  return <>
    {header}
    <section className="panel">
      <div className="panel-header"><h2>Review queue</h2><span className="muted">{plans.length} plan{plans.length === 1 ? "" : "s"}</span></div>
      {plans.length ? <div className="table-wrap"><table>
        <thead><tr><th>Plan</th><th>System</th><th>Status</th><th>Suite</th><th>Executable</th><th>Excluded</th></tr></thead>
        <tbody>{plans.map((plan) => <tr key={plan.test_plan_id}>
          <td><Link className="mono" href={`/plans/${plan.test_plan_id}`}>{plan.test_plan_id}</Link><br /><span className="muted">v{plan.test_plan_version}</span></td>
          <td><span className="mono">{plan.application_id}</span><br /><span className="muted">{plan.application_environment}</span></td>
          <td><span className={`badge ${plan.status === "CONFIRMED" ? "pass" : "warn"}`}>{plan.status}</span></td>
          <td><span className="mono">{plan.test_suite_id}</span><br /><span className="muted">v{plan.test_suite_version}</span></td>
          <td>{plan.suite_case_version_ids.length}</td>
          <td>{plan.excluded_suite_cases.length}</td>
        </tr>)}</tbody>
      </table></div> : <EmptyState title="No test plans" detail="Generate a draft from a registered system, active suite version, and release policy. Review its exact snapshot before confirmation." action="Generate draft" href="/plans/new" />}
    </section>
  </>;
}

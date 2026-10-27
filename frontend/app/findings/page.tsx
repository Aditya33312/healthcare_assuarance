import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { SeverityBadge } from "@/components/status-badge";
import { gatekeeperFetch } from "@/lib/api";
import type { FindingSummary } from "@/lib/types";

type FindingsResponse = FindingSummary[] | { items: FindingSummary[] };

export default async function FindingsPage() {
  const response = await gatekeeperFetch<FindingsResponse>("/v1/findings");
  const header = <div className="page-header"><div><span className="eyebrow">Explainable discrepancies</span><h1>Findings</h1><p>Every item retains exact evidence, reason code, rule/evaluator version, confidence, and provisional severity.</p></div></div>;
  if (!response.ok) return <>{header}<ApiErrorState error={response.error} title="Finding register unavailable" /></>;
  const findings = Array.isArray(response.data) ? response.data : response.data.items;
  return <>
    {header}
    <section className="panel"><div className="panel-header"><h2>Finding register</h2><span className="muted">Original automated evidence is immutable</span></div>
      {findings.length ? <div className="table-wrap"><table><thead><tr><th>Severity</th><th>Type</th><th>Category</th><th>Reason</th><th>Confidence</th></tr></thead><tbody>
        {findings.map((finding) => <tr key={finding.finding_id}><td><SeverityBadge severity={finding.severity} /></td><td>{finding.type}</td><td>{finding.category}</td><td className="mono">{finding.reason_code}</td><td>{Math.round(finding.confidence * 100)}%</td></tr>)}
      </tbody></table></div> : <EmptyState title="No findings to review" detail="Findings appear after a completed assessment; technical failures and unassessed cases are shown as coverage gaps instead." />}
    </section>
  </>;
}

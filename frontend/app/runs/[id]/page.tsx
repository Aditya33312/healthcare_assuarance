import { notFound } from "next/navigation";

import { ApiErrorState } from "@/components/api-error-state";
import { EvidenceDiff } from "@/components/evidence-diff";
import { DecisionBadge, ExecutionBadge, SeverityBadge } from "@/components/status-badge";
import { gatekeeperFetch } from "@/lib/api";
import type { FindingSummary, RunSummary } from "@/lib/types";

type FindingsResponse = FindingSummary[] | { items: FindingSummary[] };

export default async function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [runResult, findingResult] = await Promise.all([
    gatekeeperFetch<RunSummary>(`/v1/test-runs/${encodeURIComponent(id)}`),
    gatekeeperFetch<FindingsResponse>(`/v1/test-runs/${encodeURIComponent(id)}/findings`)
  ]);
  if (!runResult.ok && runResult.error.code === "UPSTREAM_NOT_FOUND") notFound();
  if (!runResult.ok) {
    return <>
      <div className="page-header"><div><span className="eyebrow">Test run</span><h1 className="mono">{id}</h1></div></div>
      <ApiErrorState error={runResult.error} title="Test run unavailable" />
    </>;
  }
  const run = runResult.data;
  const findings = findingResult.ok
    ? (Array.isArray(findingResult.data) ? findingResult.data : findingResult.data.items)
    : [];
  return <>
    <div className="page-header"><div><span className="eyebrow">Test run</span><h1 className="mono">{run.run_id}</h1><p>System {run.system_id} · version {run.system_version ?? "not recorded"}</p></div><div className="stack"><ExecutionBadge status={run.execution_status} /><DecisionBadge decision={run.release_decision} /></div></div>
    <nav className="tabs" aria-label="Run result sections">
      {['Overview','Findings','Cases','Metrics','Coverage','Policy evaluation','Versions','Audit'].map((tab, index) => <span className={index === 0 ? "tab active" : "tab"} key={tab}>{tab}</span>)}
    </nav>
    <section className="metrics" style={{marginTop: 24}}>
      <article className="metric-card"><span>Coverage</span><strong>{run.coverage?.status ?? "UNASSESSED"}</strong><small>Required assessment completeness</small></article>
      <article className="metric-card danger"><span>Critical</span><strong>{run.metrics?.critical_count ?? 0}</strong><small>Independent blocking candidates</small></article>
      <article className="metric-card warn"><span>High</span><strong>{run.metrics?.high_count ?? 0}</strong><small>High-severity findings</small></article>
      <article className="metric-card"><span>Findings</span><strong>{findingResult.ok ? findings.length : "—"}</strong><small>Evidence-linked discrepancies</small></article>
    </section>
    {run.execution_status === "FAILED" ? <div className="callout warning">Assurance could not complete. This is a technical failure, not a clinical PASS or FAIL.</div> : null}
    <section className="stack" style={{marginTop: 20}}>
      {!findingResult.ok ? <ApiErrorState error={findingResult.error} title="Run findings unavailable" /> : null}
      {findings.map((finding) => <article className="panel" key={finding.finding_id}>
        <div className="panel-header"><div><SeverityBadge severity={finding.severity} /> <strong style={{marginLeft: 8}}>{finding.type} · {finding.category}</strong></div><span className="mono">{finding.reason_code}</span></div>
        <EvidenceDiff source={finding.source?.text} generated={finding.generated?.text} />
        <div className="panel-body"><p>{finding.explanation}</p><span className="muted">Confidence {Math.round(finding.confidence * 100)}% · automated severity remains provisional</span></div>
      </article>)}
      {findingResult.ok && !findings.length ? <div className="callout">No finding payload is available. Check execution status, coverage, and unassessed cases before interpreting the result.</div> : null}
    </section>
  </>;
}

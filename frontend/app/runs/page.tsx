import Link from "next/link";

import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { DecisionBadge, ExecutionBadge } from "@/components/status-badge";
import { gatekeeperFetch } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

type RunsResponse = RunSummary[] | { items: RunSummary[] };

export default async function RunsPage() {
  const response = await gatekeeperFetch<RunsResponse>("/v1/test-runs");
  const header = <div className="page-header"><div><span className="eyebrow">Execution history</span><h1>Test runs</h1><p>Execution status and release decision are deliberately independent.</p></div><Link className="button primary" href="/runs/new">Run test</Link></div>;
  if (!response.ok) return <>{header}<ApiErrorState error={response.error} title="Test-run history unavailable" /></>;
  const runs = Array.isArray(response.data) ? response.data : response.data.items;
  return <>
    {header}
    <section className="panel"><div className="panel-header"><h2>All runs</h2><span className="muted">{runs.length} records</span></div>
      {runs.length ? <div className="table-wrap"><table><thead><tr><th>Run</th><th>System</th><th>Version</th><th>Execution</th><th>Decision</th><th>Coverage</th></tr></thead><tbody>
        {runs.map((run) => <tr key={run.run_id}><td><Link className="mono" href={`/runs/${run.run_id}`}>{run.run_id}</Link></td><td>{run.system_id}</td><td className="mono">{run.system_version ?? "—"}</td><td><ExecutionBadge status={run.execution_status} /></td><td><DecisionBadge decision={run.release_decision} /></td><td>{run.coverage?.status ?? "UNASSESSED"}</td></tr>)}
      </tbody></table></div> : <EmptyState title="No runs available" detail="A queued run is not a clinical result. Start only after confirming the generated test plan." action="Configure a run" href="/runs/new" />}
    </section>
  </>;
}

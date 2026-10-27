import Link from "next/link";

import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { MetricCard } from "@/components/metric-card";
import { DecisionBadge, ExecutionBadge } from "@/components/status-badge";
import { gatekeeperFetch } from "@/lib/api";
import type { RunSummary, SystemSummary } from "@/lib/types";

type ListResponse<T> = T[] | { items: T[] };

function items<T>(value: ListResponse<T>): T[] {
  return Array.isArray(value) ? value : value.items;
}

export default async function DashboardPage() {
  const [systemsResult, runsResult] = await Promise.all([
    gatekeeperFetch<ListResponse<SystemSummary>>("/v1/systems"),
    gatekeeperFetch<ListResponse<RunSummary>>("/v1/test-runs")
  ]);
  if (!systemsResult.ok || !runsResult.ok) {
    return (
      <>
        <div className="page-header">
          <div>
            <span className="eyebrow">Release assurance</span>
            <h1>Assurance overview</h1>
            <p>Clinical failures, technical failures, and coverage gaps remain separate and visible.</p>
          </div>
        </div>
        <div className="stack">
          {!systemsResult.ok ? <ApiErrorState error={systemsResult.error} title="System registry unavailable" /> : null}
          {!runsResult.ok ? <ApiErrorState error={runsResult.error} title="Test-run history unavailable" /> : null}
        </div>
      </>
    );
  }
  const systems = items(systemsResult.data);
  const runs = items(runsResult.data);
  const completed = runs.filter((run) => run.execution_status === "COMPLETED");
  const critical = runs.reduce((total, run) => total + (run.metrics?.critical_count ?? 0), 0);

  return (
    <>
      <div className="page-header">
        <div>
          <span className="eyebrow">Release assurance</span>
          <h1>Assurance overview</h1>
          <p>Clinical failures, technical failures, and coverage gaps remain separate and visible.</p>
        </div>
        <Link className="button primary" href="/runs/new">Run assurance</Link>
      </div>

      <section className="metrics" aria-label="Key metrics">
        <MetricCard label="Registered systems" value={systems.length} hint="Tenant-scoped applications" />
        <MetricCard label="PASS" value={completed.filter((run) => run.release_decision === "PASS").length} hint="Completed valid decisions" accent="good" />
        <MetricCard label="WARN / BLOCK" value={completed.filter((run) => run.release_decision === "WARN" || run.release_decision === "BLOCK").length} hint="Needs review or prevents release" accent="warn" />
        <MetricCard label="Critical findings" value={critical} hint="Never hidden by average score" accent={critical ? "danger" : "good"} />
      </section>

      <section className="grid-2">
        <article className="panel">
          <div className="panel-header"><h2>Recent test runs</h2><Link className="muted" href="/runs">View all</Link></div>
          {runs.length ? (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Run</th><th>System</th><th>Execution</th><th>Decision</th></tr></thead>
                <tbody>
                  {runs.slice(0, 8).map((run) => (
                    <tr key={run.run_id}>
                      <td><Link className="mono" href={`/runs/${run.run_id}`}>{run.run_id}</Link></td>
                      <td>{run.system_id}</td>
                      <td><ExecutionBadge status={run.execution_status} /></td>
                      <td><DecisionBadge decision={run.release_decision} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <EmptyState title="No test runs yet" detail="Connect the server API, register a system, and run a confirmed test plan." action="Register an AI system" href="/systems/new" />}
        </article>

        <div className="stack">
          <article className="panel">
            <div className="panel-header"><h2>Coverage discipline</h2></div>
            <div className="panel-body">
              <div className="callout">Required tests that are missing, technically failed, or unsupported cannot silently produce a full PASS.</div>
            </div>
          </article>
          <article className="panel">
            <div className="panel-header"><h2>Evaluator status</h2></div>
            <div className="panel-body">
              <p><strong>Clinical faithfulness</strong><br /><span className="muted">Implemented · deterministic Compare Engine</span></p>
              <p><strong>Remaining families</strong><br /><span className="muted">Structurally routed · explicitly unassessed until implemented</span></p>
            </div>
          </article>
        </div>
      </section>
    </>
  );
}

import Link from "next/link";

import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { gatekeeperFetch } from "@/lib/api";
import type { SystemSummary } from "@/lib/types";

type SystemsResponse = SystemSummary[] | { items: SystemSummary[] };

export default async function SystemsPage() {
  const response = await gatekeeperFetch<SystemsResponse>("/v1/systems");
  const header = (
    <div className="page-header">
      <div><span className="eyebrow">Application registry</span><h1>AI systems</h1><p>Profiles drive deterministic test selection, plans, adapters, and risk-aware release policy.</p></div>
      <Link className="button primary" href="/systems/new">Add application</Link>
    </div>
  );
  if (!response.ok) return <>{header}<ApiErrorState error={response.error} title="System registry unavailable" /></>;
  const systems = Array.isArray(response.data) ? response.data : response.data.items;
  return (
    <>
      {header}
      <section className="panel">
        <div className="panel-header"><h2>All systems</h2><span className="muted">{systems.length} registered</span></div>
        {systems.length ? (
          <div className="table-wrap"><table>
            <thead><tr><th>Name</th><th>Type</th><th>Version</th><th>Risk tier</th><th>Updated</th></tr></thead>
            <tbody>{systems.map((system) => <tr key={system.id}>
              <td><strong>{system.name}</strong><br /><span className="mono muted">{system.id}</span></td>
              <td>{system.application_type.replaceAll("_", " ")}</td>
              <td className="mono">{system.application_profile?.version ?? "—"}</td>
              <td><span className={`badge severity-${system.risk_tier.toLowerCase()}`}>{system.risk_tier}</span></td>
              <td className="muted">{system.updated_at ? new Date(system.updated_at).toLocaleDateString() : "—"}</td>
            </tr>)}</tbody>
          </table></div>
        ) : <EmptyState title="No systems registered" detail="Registration captures intended use, audience, AI characteristics, endpoint mapping, and risk tier." action="Start registration" href="/systems/new" />}
      </section>
    </>
  );
}

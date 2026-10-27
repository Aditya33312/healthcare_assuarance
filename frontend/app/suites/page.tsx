import { ApiErrorState } from "@/components/api-error-state";
import { EmptyState } from "@/components/empty-state";
import { gatekeeperFetch } from "@/lib/api";

interface Suite { id: string; name: string; slug?: string; suite_type?: string; latest_version?: { version?: string } | null; }
type SuitesResponse = Suite[] | { items: Suite[] };

export default async function SuitesPage() {
  const response = await gatekeeperFetch<SuitesResponse>("/v1/test-suites");
  const header = <div className="page-header"><div><span className="eyebrow">Versioned test assets</span><h1>Test suites</h1><p>Required, recommended, custom, and permanent regression cases remain traceable by version.</p></div></div>;
  if (!response.ok) return <>{header}<ApiErrorState error={response.error} title="Test suites unavailable" /></>;
  const suites = Array.isArray(response.data) ? response.data : response.data.items;
  return <>
    {header}
    <section className="panel"><div className="panel-header"><h2>Available suites</h2><span className="muted">Required · Recommended · Custom · Regression</span></div>
      {suites.length ? <div className="table-wrap"><table><thead><tr><th>Suite</th><th>Type</th><th>Version</th><th>Identifier</th></tr></thead><tbody>
        {suites.map((suite) => <tr key={suite.id}><td><strong>{suite.name}</strong></td><td>{suite.suite_type ?? "CUSTOM"}</td><td className="mono">{suite.latest_version?.version ?? "—"}</td><td className="mono muted">{suite.slug ?? suite.id}</td></tr>)}
      </tbody></table></div> : <EmptyState title="No suites stored" detail="Load Benchmark v0 or create a versioned customer/regression suite through the API." />}
    </section>
  </>;
}

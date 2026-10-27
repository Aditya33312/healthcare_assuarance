import { ApiErrorState } from "@/components/api-error-state";
import { NewPlanForm } from "@/components/new-plan-form";
import { gatekeeperFetch } from "@/lib/api";

interface PlanSystem {
  id: string;
  name: string;
  test_selection?: { optional_tests?: string[] } | null;
}
interface PlanSuite { id: string; name: string; latest_version?: { version?: string; status?: string } | null; }
interface PlanPolicy { id: string; name: string; latest_version?: { version?: string } | null; }
type ListResponse<T> = T[] | { items: T[] };

function items<T>(value: ListResponse<T>): T[] {
  return Array.isArray(value) ? value : value.items;
}

export default async function NewPlanPage() {
  const [systemsResult, suitesResult, policiesResult] = await Promise.all([
    gatekeeperFetch<ListResponse<PlanSystem>>("/v1/systems"),
    gatekeeperFetch<ListResponse<PlanSuite>>("/v1/test-suites"),
    gatekeeperFetch<ListResponse<PlanPolicy>>("/v1/release-policies")
  ]);
  const header = <div className="page-header"><div><span className="eyebrow">Deterministic planning</span><h1>Generate test-plan draft</h1><p>The profile selects applicable families. Overrides can change selected quotas or opt into profile-eligible optional families; they cannot enable not-applicable tests.</p></div></div>;
  if (!systemsResult.ok || !suitesResult.ok || !policiesResult.ok) return <>{header}<div className="stack">{!systemsResult.ok ? <ApiErrorState error={systemsResult.error} title="Systems unavailable" /> : null}{!suitesResult.ok ? <ApiErrorState error={suitesResult.error} title="Suites unavailable" /> : null}{!policiesResult.ok ? <ApiErrorState error={policiesResult.error} title="Policies unavailable" /> : null}</div></>;
  return <>{header}<NewPlanForm systems={items(systemsResult.data)} suites={items(suitesResult.data)} policies={items(policiesResult.data)} /></>;
}

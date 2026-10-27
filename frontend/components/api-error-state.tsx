import type { GatekeeperFetchError } from "@/lib/api";

export function ApiErrorState({
  error,
  title = "Gatekeeper data unavailable"
}: {
  error: GatekeeperFetchError;
  title?: string;
}) {
  return (
    <section className="panel" role="alert">
      <div className="panel-header"><h2>{title}</h2></div>
      <div className="panel-body">
        <div className="callout warning">
          {error.message} <span className="mono">({error.code}{error.status ? ` · HTTP ${error.status}` : ""})</span>
        </div>
      </div>
    </section>
  );
}

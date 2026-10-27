import Link from "next/link";

export default function DevelopersPage() {
  return <>
    <div className="page-header"><div><span className="eyebrow">Integration surface</span><h1>Developers</h1><p>API keys, API usage, documentation, and CI/CD release gating.</p></div></div>
    <section className="resource-grid">
      <article className="resource-card"><span className="eyebrow">API keys</span><h3>Scoped credentials</h3><p>Keys are shown once, hashed at rest, tenant-bound, revocable, and separated into test/live environments.</p><Link className="button secondary" href="/api-docs">View lifecycle API</Link></article>
      <article className="resource-card"><span className="eyebrow">API docs</span><h3>OpenAPI contract</h3><p>FastAPI publishes the same endpoints used by the portal and CI clients.</p><a className="button secondary" href="/api-docs">Open API guide</a></article>
      <article className="resource-card"><span className="eyebrow">CI/CD</span><h3>GitHub Actions</h3><p>PASS exits 0, BLOCK exits 1, and technical failure exits 1 with “assurance could not complete.”</p><code className="mono">scripts/gatekeeper_ci_gate.py</code></article>
    </section>
    <section className="panel" style={{marginTop: 20}}><div className="panel-header"><h2>Minimum CI scopes</h2></div><div className="panel-body prose"><ul><li><code>test-runs:create</code></li><li><code>test-runs:read</code></li><li><code>reports:read</code> only when report download is required</li></ul><p>Secrets belong in the Authorization header and server-side secret store—never a URL, browser bundle, or normal log.</p></div></section>
  </>;
}

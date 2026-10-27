export default function ResourcesPage() {
  return <>
    <div className="page-header"><div><span className="eyebrow">Transparency</span><h1>Resources</h1><p>Methodology, coverage semantics, model/rule boundaries, and known limitations.</p></div></div>
    <section className="resource-grid">
      <article className="resource-card"><span className="eyebrow">Methodology</span><h3>How comparisons work</h3><p>Two-way claim comparison, deterministic high-risk rules, exact evidence, and optional semantic fallback.</p></article>
      <article className="resource-card"><span className="eyebrow">Coverage</span><h3>What actually ran</h3><p>FULL, REDUCED, PARTIAL, and UNASSESSED remain independent of release score.</p></article>
      <article className="resource-card"><span className="eyebrow">Limitations</span><h3>What this cannot prove</h3><p>No automatic legal compliance claim; model acceptance and severity require clinician calibration.</p></article>
    </section>
    <article className="panel prose" style={{marginTop: 20}}><div className="panel-body">
      <h2>Clinical Compare Engine</h2><p>Known source and generated output are normalized into structured claims. Output-to-source detects supported, added, altered, inferred, and uncertain content. Source-to-output independently detects omissions.</p>
      <h2>Deterministic boundary</h2><p>Dose, unit, frequency, route, laterality, assertion, subject, numeric values, dates, and durations are resolved by explicit versioned rules. A semantic model cannot override these comparisons.</p>
      <h2>Semantic and model boundary</h2><p>The optional pretrained NER model is a candidate extractor, not production-accepted clinical evidence. Semantic evaluation is reserved for unresolved narrative equivalence and may reduce coverage when unavailable.</p>
      <h2>Human calibration</h2><p>Benchmark v0 is an engineering baseline. Clinician agreement, severity agreement, and held-out validation are required before production claims.</p>
    </div>
    </article>
  </>;
}

export function EvidenceDiff({
  source,
  generated
}: {
  source?: string | null;
  generated?: string | null;
}) {
  return (
    <div className="evidence-grid">
      <div>
        <span className="eyebrow">Known source</span>
        <p>{source || "No source span"}</p>
      </div>
      <div>
        <span className="eyebrow">Generated output</span>
        <p>{generated || "No generated span"}</p>
      </div>
    </div>
  );
}


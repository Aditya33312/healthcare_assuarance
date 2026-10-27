import type { ReactNode } from "react";

export function MetricCard({
  label,
  value,
  hint,
  accent
}: {
  label: string;
  value: ReactNode;
  hint: string;
  accent?: "good" | "warn" | "danger";
}) {
  return (
    <article className={`metric-card ${accent ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </article>
  );
}


import type { Decision, ExecutionStatus } from "@/lib/types";

export function DecisionBadge({ decision }: { decision: Decision }) {
  if (!decision) return <span className="badge neutral">NO DECISION</span>;
  return <span className={`badge ${decision.toLowerCase()}`}>{decision}</span>;
}

export function ExecutionBadge({ status }: { status: ExecutionStatus }) {
  const tone = status === "FAILED" || status === "CANCELLED" ? "block" : "neutral";
  return <span className={`badge ${tone}`}>{status}</span>;
}

export function SeverityBadge({ severity }: { severity: string }) {
  return <span className={`badge severity-${severity.toLowerCase()}`}>{severity}</span>;
}


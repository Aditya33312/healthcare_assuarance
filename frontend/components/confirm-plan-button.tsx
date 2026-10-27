"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export function ConfirmPlanButton({
  planId,
  profileFingerprint,
  planVersion
}: {
  planId: string;
  profileFingerprint: string;
  planVersion: string;
}) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirm() {
    setConfirming(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/gatekeeper/test-plans/${encodeURIComponent(planId)}/confirm`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            confirm: true,
            expected_profile_fingerprint: profileFingerprint,
            expected_test_plan_version: planVersion
          })
        }
      );
      const body = await response.json().catch(() => ({}));
      if (!response.ok || body?.status !== "CONFIRMED") {
        throw new Error(body?.error?.message ?? "The plan could not be confirmed.");
      }
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The plan could not be confirmed.");
    } finally {
      setConfirming(false);
    }
  }

  return <div className="stack">
    {error ? <div className="callout warning">{error}</div> : null}
    <button className="button primary" disabled={confirming} onClick={confirm}>
      {confirming ? "Confirming…" : "Confirm reviewed plan"}
    </button>
  </div>;
}

import "server-only";

import { NextResponse } from "next/server";

import { isJsonRecord, type JsonRecord } from "@/lib/gatekeeper-contracts";

export async function readUpstreamJson(response: Response): Promise<JsonRecord | null> {
  try {
    const value: unknown = await response.json();
    return isJsonRecord(value) ? value : null;
  } catch {
    return null;
  }
}

export function upstreamUnavailableResponse(extra: JsonRecord = {}) {
  return NextResponse.json(
    {
      error: {
        code: "UPSTREAM_UNAVAILABLE",
        message: "Gatekeeper could not be reached before the bounded request deadline.",
        retryable: true
      },
      ...extra
    },
    { status: 502 }
  );
}

export function upstreamInvalidResponse(
  message = "Gatekeeper returned an invalid response.",
  extra: JsonRecord = {}
) {
  return NextResponse.json(
    {
      error: {
        code: "UPSTREAM_INVALID_RESPONSE",
        message,
        retryable: false
      },
      ...extra
    },
    { status: 502 }
  );
}

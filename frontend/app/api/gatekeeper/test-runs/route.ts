import { NextRequest, NextResponse } from "next/server";

import {
  isJsonRecord,
  isTestRunPayload
} from "@/lib/gatekeeper-contracts";
import {
  readUpstreamJson,
  upstreamInvalidResponse,
  upstreamUnavailableResponse
} from "@/lib/gatekeeper-proxy";
import {
  gatekeeperUpstreamFetch,
  getGatekeeperServerConfiguration,
  portalMutationRequestIsTrusted
} from "@/lib/gatekeeper-server";

const ATTEMPT_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function POST(request: NextRequest) {
  if (!portalMutationRequestIsTrusted(request)) return NextResponse.json({error:{code:"PORTAL_MUTATION_REJECTED",message:"The portal write requires trusted same-origin JSON.",retryable:false}}, {status:403});
  const configuration = getGatekeeperServerConfiguration();
  if (!configuration.ok) return NextResponse.json({error:{code:"API_NOT_CONFIGURED",message:"The server-side Gatekeeper API connection is not configured.",retryable:false}}, {status:503});
  if (process.env.GATEKEEPER_PORTAL_WRITES_ENABLED !== "true") return NextResponse.json({error:{code:"PORTAL_WRITES_DISABLED",message:"Portal writes are disabled. Enable them only behind an authenticated access boundary.",retryable:false}}, {status:403});
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({error:{code:"REQ_INVALID",message:"The request body must be valid JSON.",retryable:false}}, {status:400});
  }
  if (!isJsonRecord(payload)) {
    return NextResponse.json({error:{code:"REQ_INVALID",message:"The request body must be a JSON object.",retryable:false}}, {status:400});
  }
  const attemptId = request.headers.get("x-gatekeeper-portal-attempt-id") ?? "";
  if (!ATTEMPT_ID_PATTERN.test(attemptId)) {
    return NextResponse.json({error:{code:"REQ_INVALID",message:"A valid portal attempt identifier is required.",retryable:false}}, {status:400});
  }
  const upstream = await gatekeeperUpstreamFetch(
    configuration.connection,
    "/v1/test-runs",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": `portal:${attemptId}`
      },
      body: JSON.stringify(payload)
    }
  );
  if (!upstream.ok) return upstreamUnavailableResponse();
  const response = upstream.response;
  const body = await readUpstreamJson(response);
  if (body === null) return upstreamInvalidResponse();
  if (!response.ok) return NextResponse.json(body,{status:response.status});
  if (!isTestRunPayload(body)) {
    return upstreamInvalidResponse("Gatekeeper did not return a valid test-run artifact.");
  }
  return NextResponse.json(body,{status:response.status});
}

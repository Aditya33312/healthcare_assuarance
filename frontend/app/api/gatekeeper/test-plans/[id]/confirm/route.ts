import { NextRequest, NextResponse } from "next/server";

import { isTestPlanPayload } from "@/lib/gatekeeper-contracts";
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

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
  const { id } = await params;
  const upstream = await gatekeeperUpstreamFetch(
    configuration.connection,
    `/v1/test-plans/${encodeURIComponent(id)}/confirm`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    }
  );
  if (!upstream.ok) return upstreamUnavailableResponse();
  const response = upstream.response;
  const body = await readUpstreamJson(response);
  if (body === null) return upstreamInvalidResponse("Gatekeeper returned an invalid confirmation response.");
  if (!response.ok) return NextResponse.json(body, {status: response.status});
  if (!isTestPlanPayload(body) || body.test_plan_id !== id || body.status !== "CONFIRMED") {
    return upstreamInvalidResponse("Gatekeeper did not return the requested confirmed plan.");
  }
  return NextResponse.json(body, {status: response.status});
}

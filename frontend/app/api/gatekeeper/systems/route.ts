import { NextRequest, NextResponse } from "next/server";

import {
  isJsonRecord,
  isSystemPayload,
  isSystemVersionPayload
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

export async function POST(request: NextRequest) {
  if (!portalMutationRequestIsTrusted(request)) {
    return NextResponse.json(
      {
        error: {
          code: "PORTAL_MUTATION_REJECTED",
          message: "The portal write requires trusted same-origin JSON.",
          retryable: false
        }
      },
      { status: 403 }
    );
  }
  const configuration = getGatekeeperServerConfiguration();
  if (!configuration.ok) {
    return NextResponse.json(
      {
        error: {
          code: "API_NOT_CONFIGURED",
          message: "The server-side Gatekeeper API connection is not configured.",
          retryable: false
        }
      },
      { status: 503 }
    );
  }
  if (process.env.GATEKEEPER_PORTAL_WRITES_ENABLED !== "true") {
    return NextResponse.json(
      {
        error: {
          code: "PORTAL_WRITES_DISABLED",
          message: "Portal writes are disabled. Enable them only behind an authenticated access boundary.",
          retryable: false
        }
      },
      { status: 403 }
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { error: { code: "REQ_INVALID", message: "The request body must be valid JSON.", retryable: false } },
      { status: 400 }
    );
  }
  if (!isJsonRecord(payload)) {
    return NextResponse.json(
      { error: { code: "REQ_INVALID", message: "The request body must be a JSON object.", retryable: false } },
      { status: 400 }
    );
  }

  const profile = {
    version: payload.version,
    intended_use: payload.intended_use,
    clinical_specialty: payload.clinical_specialty || null,
    input_type: payload.input_type,
    output_type: payload.output_type,
    patient_facing: payload.patient_facing,
    clinician_facing: payload.clinician_facing,
    uses_rag: payload.uses_rag,
    uses_tools: payload.uses_tools,
    uses_llm: payload.uses_llm,
    uses_predictive_model: payload.uses_predictive_model,
    environment: payload.environment,
    endpoint_url: payload.endpoint_url,
    authentication_type: payload.authentication_type,
    request_schema: payload.request_schema,
    response_schema: payload.response_schema
  };
  const created = await gatekeeperUpstreamFetch(
    configuration.connection,
    "/v1/systems",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: payload.name,
        description: payload.description,
        application_type: payload.application_type,
        risk_tier: payload.risk_tier,
        profile
      })
    }
  );
  if (!created.ok) return upstreamUnavailableResponse();
  const response = created.response;
  const body = await readUpstreamJson(response);
  if (body === null) return upstreamInvalidResponse();
  if (!response.ok) return NextResponse.json(body, { status: response.status });
  if (!isSystemPayload(body)) {
    return upstreamInvalidResponse("Gatekeeper did not return a valid system artifact.");
  }

  const versionCreated = await gatekeeperUpstreamFetch(
    configuration.connection,
    `/v1/systems/${encodeURIComponent(String(body.id))}/versions`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        version: payload.version,
        environment: payload.environment,
        endpoint_url: payload.endpoint_url,
        request_schema: payload.request_schema,
        response_schema: payload.response_schema
      })
    }
  );
  if (!versionCreated.ok) {
    return upstreamUnavailableResponse({ system: body });
  }
  const versionResponse = versionCreated.response;
  const versionBody = await readUpstreamJson(versionResponse);
  if (versionBody === null) {
    return upstreamInvalidResponse(
      "Gatekeeper returned an invalid version response.",
      { system: body }
    );
  }
  if (!versionResponse.ok) {
    return NextResponse.json(
      {
        error: {
          code: "SYSTEM_VERSION_CREATE_FAILED",
          message: "The application was registered, but its executable version could not be created.",
          retryable: false
        },
        system: body,
        upstream: versionBody
      },
      { status: versionResponse.status }
    );
  }
  if (!isSystemVersionPayload(versionBody)) {
    return upstreamInvalidResponse(
      "Gatekeeper did not return a valid system version.",
      { system: body }
    );
  }
  return NextResponse.json({ ...body, version: versionBody }, { status: response.status });
}

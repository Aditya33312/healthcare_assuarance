import "server-only";

import { headers } from "next/headers";

import { validateGatekeeperPayload } from "@/lib/gatekeeper-contracts";
import {
  gatekeeperUpstreamFetch,
  getGatekeeperServerConfiguration,
  portalRequestIsTrusted
} from "@/lib/gatekeeper-server";

export interface GatekeeperFetchError {
  code:
    | "API_NOT_CONFIGURED"
    | "PORTAL_ACCESS_DENIED"
    | "UPSTREAM_AUTH_FAILED"
    | "UPSTREAM_NOT_FOUND"
    | "UPSTREAM_REJECTED"
    | "UPSTREAM_UNAVAILABLE"
    | "UPSTREAM_INVALID_RESPONSE";
  message: string;
  status?: number;
}

export type GatekeeperFetchResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: GatekeeperFetchError };

export async function gatekeeperFetch<T>(
  path: `/${string}`
): Promise<GatekeeperFetchResult<T>> {
  const requestHeaders = await headers();
  if (!portalRequestIsTrusted(requestHeaders)) {
    return {
      ok: false,
      error: {
        code: "PORTAL_ACCESS_DENIED",
        message: "The portal request was not accepted by the trusted access boundary.",
        status: 403
      }
    };
  }

  const configuration = getGatekeeperServerConfiguration();
  if (!configuration.ok) {
    return {
      ok: false,
      error: {
        code: "API_NOT_CONFIGURED",
        message: "The server-side Gatekeeper API connection is not configured."
      }
    };
  }

  const upstream = await gatekeeperUpstreamFetch(configuration.connection, path);
  if (!upstream.ok) {
    return {
      ok: false,
      error: {
        code: "UPSTREAM_UNAVAILABLE",
        message: "Gatekeeper could not be reached before the bounded request deadline.",
        status: 502
      }
    };
  }
  const response = upstream.response;

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      return {
        ok: false,
        error: {
          code: "UPSTREAM_AUTH_FAILED",
          message: "Gatekeeper rejected the portal API credentials.",
          status: response.status
        }
      };
    }
    if (response.status === 404) {
      return {
        ok: false,
        error: {
          code: "UPSTREAM_NOT_FOUND",
          message: "The requested Gatekeeper resource was not found.",
          status: response.status
        }
      };
    }
    return {
      ok: false,
      error: {
        code: response.status >= 500 ? "UPSTREAM_UNAVAILABLE" : "UPSTREAM_REJECTED",
        message: response.status >= 500
          ? "Gatekeeper is unavailable. Data has not been loaded."
          : "Gatekeeper rejected the portal request. Data has not been loaded.",
        status: response.status
      }
    };
  }

  try {
    const data: unknown = await response.json();
    if (!validateGatekeeperPayload(path, data)) throw new Error("invalid upstream contract");
    return { ok: true, data: data as T };
  } catch {
    return {
      ok: false,
      error: {
        code: "UPSTREAM_INVALID_RESPONSE",
        message: "Gatekeeper returned an invalid response. Data has not been loaded.",
        status: 502
      }
    };
  }
}

export function serverApiConfigurationPresent(): boolean {
  return getGatekeeperServerConfiguration().ok;
}

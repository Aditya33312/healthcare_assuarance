import "server-only";

import { timingSafeEqual } from "node:crypto";

const HEADER_NAME_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const SECRET_VALUE_PATTERN = /^[A-Za-z0-9+/_=-]+$/;
const MINIMUM_ENCODED_SECRET_LENGTH = 43;
const MAXIMUM_ENCODED_SECRET_LENGTH = 512;
const UPSTREAM_TIMEOUT_MS = 10_000;

export interface HeaderReader {
  get(name: string): string | null;
}

export interface PortalMutationRequest {
  headers: HeaderReader;
  nextUrl: { origin: string };
}

export interface GatekeeperServerConnection {
  apiBaseUrl: string;
  apiKey: string;
}

export type GatekeeperServerConfiguration =
  | { ok: true; connection: GatekeeperServerConnection }
  | { ok: false };

function isLocalDevelopmentHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (
    normalized === "localhost" ||
    normalized === "::1"
  ) {
    return true;
  }

  const octets = normalized.split(".");
  return (
    octets.length === 4 &&
    octets.every((octet) => /^\d{1,3}$/.test(octet) && Number(octet) <= 255) &&
    Number(octets[0]) === 127
  );
}

function normalizeApiBaseUrl(rawValue: string): string | null {
  if (
    !/^https?:\/\//i.test(rawValue) ||
    rawValue !== rawValue.trim() ||
    /[\u0000-\u0020\u007f\\]/.test(rawValue) ||
    rawValue.includes("?") ||
    rawValue.includes("#")
  ) {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(rawValue);
  } catch {
    return null;
  }

  const authorityStart = rawValue.indexOf("://");
  const authority = rawValue.slice(authorityStart + 3).split(/[/?#]/, 1)[0];
  if (
    !parsed.hostname ||
    parsed.username ||
    parsed.password ||
    authority.includes("@")
  ) {
    return null;
  }

  const production = process.env.NODE_ENV === "production";
  if (parsed.protocol !== "https:") {
    if (
      production ||
      parsed.protocol !== "http:" ||
      !isLocalDevelopmentHostname(parsed.hostname)
    ) {
      return null;
    }
  }

  const basePath = parsed.pathname === "/"
    ? ""
    : parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${basePath}`;
}

export function getGatekeeperServerConfiguration(): GatekeeperServerConfiguration {
  // Read secrets at request time so a module-level value cannot outlive runtime
  // secret rotation or be accidentally reused by client code.
  const rawBaseUrl = process.env.GATEKEEPER_API_URL;
  const apiKey = process.env.GATEKEEPER_API_KEY;
  if (!rawBaseUrl || !apiKey) {
    return { ok: false };
  }

  const apiBaseUrl = normalizeApiBaseUrl(rawBaseUrl);
  if (!apiBaseUrl) {
    return { ok: false };
  }

  return { ok: true, connection: { apiBaseUrl, apiKey } };
}

function gatewaySecretHasRequiredStrength(secret: string): boolean {
  return (
    secret.length >= MINIMUM_ENCODED_SECRET_LENGTH &&
    secret.length <= MAXIMUM_ENCODED_SECRET_LENGTH &&
    SECRET_VALUE_PATTERN.test(secret) &&
    new Set(secret).size >= 12
  );
}

function constantTimeExactMatch(actual: string, expected: string): boolean {
  const actualBytes = Buffer.from(actual, "utf8");
  const expectedBytes = Buffer.from(expected, "utf8");
  if (
    actualBytes.length > MAXIMUM_ENCODED_SECRET_LENGTH ||
    expectedBytes.length > MAXIMUM_ENCODED_SECRET_LENGTH
  ) {
    return false;
  }

  // Include the length in fixed-size buffers so prefixes and values with
  // different lengths are rejected by the same constant-time operation.
  const compareLength = MAXIMUM_ENCODED_SECRET_LENGTH + 2;
  const actualPadded = Buffer.alloc(compareLength);
  const expectedPadded = Buffer.alloc(compareLength);
  actualBytes.copy(actualPadded);
  expectedBytes.copy(expectedPadded);
  actualPadded.writeUInt16BE(actualBytes.length, MAXIMUM_ENCODED_SECRET_LENGTH);
  expectedPadded.writeUInt16BE(expectedBytes.length, MAXIMUM_ENCODED_SECRET_LENGTH);

  return timingSafeEqual(actualPadded, expectedPadded);
}

export function portalRequestIsTrusted(requestHeaders: HeaderReader): boolean {
  const headerName = process.env.GATEKEEPER_PORTAL_TRUSTED_HEADER ?? "";
  const expectedValue = process.env.GATEKEEPER_PORTAL_TRUSTED_VALUE ?? "";
  const headerConfigured = headerName.length > 0;
  const valueConfigured = expectedValue.length > 0;

  if (!headerConfigured && !valueConfigured) {
    return process.env.NODE_ENV !== "production";
  }

  // A partial or malformed gateway configuration is unsafe in every runtime.
  if (
    !headerConfigured ||
    !valueConfigured ||
    headerName.length > 128 ||
    !HEADER_NAME_PATTERN.test(headerName) ||
    !gatewaySecretHasRequiredStrength(expectedValue)
  ) {
    return false;
  }

  return constantTimeExactMatch(requestHeaders.get(headerName) ?? "", expectedValue);
}

function normalizePortalOrigin(rawValue: string): string | null {
  if (
    rawValue !== rawValue.trim() ||
    /[\u0000-\u0020\u007f\\]/.test(rawValue) ||
    rawValue.includes("?") ||
    rawValue.includes("#")
  ) {
    return null;
  }
  let parsed: URL;
  try {
    parsed = new URL(rawValue);
  } catch {
    return null;
  }
  if (
    !parsed.hostname ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/"
  ) {
    return null;
  }
  if (process.env.NODE_ENV === "production") {
    if (parsed.protocol !== "https:") return null;
  } else if (
    parsed.protocol !== "https:" &&
    (parsed.protocol !== "http:" || !isLocalDevelopmentHostname(parsed.hostname))
  ) {
    return null;
  }
  return parsed.origin;
}

export function portalMutationRequestIsTrusted(
  request: PortalMutationRequest
): boolean {
  if (!portalRequestIsTrusted(request.headers)) return false;
  const mediaType = (request.headers.get("content-type") ?? "")
    .split(";", 1)[0]
    .trim()
    .toLowerCase();
  if (mediaType !== "application/json") return false;
  const fetchSite = request.headers.get("sec-fetch-site");
  if (fetchSite !== null && fetchSite !== "same-origin") return false;

  const configuredOrigin = process.env.GATEKEEPER_PORTAL_ALLOWED_ORIGIN;
  const expectedOrigin = configuredOrigin
    ? normalizePortalOrigin(configuredOrigin)
    : process.env.NODE_ENV === "production"
      ? null
      : normalizePortalOrigin(request.nextUrl.origin);
  const actualOrigin = request.headers.get("origin");
  return expectedOrigin !== null && actualOrigin === expectedOrigin;
}

export function gatekeeperUpstreamUrl(
  connection: GatekeeperServerConnection,
  path: `/${string}`
): string {
  return `${connection.apiBaseUrl}${path}`;
}

export type GatekeeperUpstreamFetchResult =
  | { ok: true; response: Response }
  | { ok: false };

export async function gatekeeperUpstreamFetch(
  connection: GatekeeperServerConnection,
  path: `/${string}`,
  init: Omit<RequestInit, "signal"> = {}
): Promise<GatekeeperUpstreamFetchResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  const requestHeaders = new Headers(init.headers);
  requestHeaders.set("Authorization", `Bearer ${connection.apiKey}`);
  if (!requestHeaders.has("Accept")) requestHeaders.set("Accept", "application/json");

  try {
    const response = await fetch(gatekeeperUpstreamUrl(connection, path), {
      ...init,
      headers: requestHeaders,
      signal: controller.signal,
      cache: "no-store",
      redirect: "error"
    });
    return { ok: true, response };
  } catch {
    // Network errors, redirect refusal, and timeout are deliberately
    // indistinguishable at the browser-facing boundary.
    return { ok: false };
  } finally {
    clearTimeout(timeout);
  }
}

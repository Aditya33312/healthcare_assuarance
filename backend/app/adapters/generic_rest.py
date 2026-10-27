"""Generic JSON-over-HTTP adapter with sanitized failure behavior."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from backend.app.adapters.errors import (
    AdapterRateLimitError,
    AdapterRequestError,
    AdapterSchemaError,
    AdapterServerError,
    AdapterTimeoutError,
    AdapterTransportError,
    AdapterUnauthorizedError,
)
from backend.app.adapters.json_paths import JSONPath, get_path
from backend.app.adapters.models import AdapterResponse, AdapterUsage
from backend.app.adapters.protocols import AsyncHTTPClient, HTTPResponse
from backend.app.security.endpoints import validate_public_https_endpoint

GENERIC_REST_ADAPTER_VERSION = "generic-rest-adapter@0.1.0"
UsageExtractor = Callable[[Any], AdapterUsage]


def _retry_after(response: HTTPResponse) -> float | None:
    raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class GenericRESTAdapter:
    """Pass a standardized JSON payload to a configured customer endpoint."""

    name = "generic_rest"
    version = GENERIC_REST_ADAPTER_VERSION

    def __init__(
        self,
        endpoint_url: str,
        *,
        response_path: JSONPath = "output",
        response_field: str | None = None,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: AsyncHTTPClient | None = None,
        usage_extractor: UsageExtractor | None = None,
        max_response_bytes: int = 2_097_152,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.endpoint_url = validate_public_https_endpoint(endpoint_url)
        self.response_path = response_field if response_field is not None else response_path
        self.method = method.upper()
        self._headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self._client = cast(AsyncHTTPClient, client or httpx.AsyncClient())
        self._owns_client = client is None
        self._usage_extractor = usage_extractor
        self.max_response_bytes = max_response_bytes

    def __repr__(self) -> str:
        parsed = urlsplit(self.endpoint_url)
        origin = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port is not None:
            origin += f":{parsed.port}"
        return f"{type(self).__name__}(endpoint_origin={origin!r}, method={self.method!r})"

    def build_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("adapter payload must be a mapping")
        return dict(payload)

    def extract_output(self, response_data: Any) -> Any:
        output = get_path(response_data, self.response_path)
        if output is None or (isinstance(output, str) and not output.strip()):
            raise AdapterSchemaError("AI endpoint response did not contain a usable output.")
        return output

    def extract_usage(self, response_data: Any) -> AdapterUsage:
        if self._usage_extractor is None:
            return AdapterUsage()
        usage = self._usage_extractor(response_data)
        if not isinstance(usage, AdapterUsage):
            raise AdapterSchemaError("Configured usage extractor returned an invalid result.")
        return usage

    async def invoke(self, payload: Mapping[str, Any]) -> AdapterResponse:
        request_data = self.build_request(payload)
        started = perf_counter()
        try:
            if self._owns_client:
                status_code, response_headers, response_data = await self._stream_json(request_data)
            else:
                response = await self._client.request(
                    self.method,
                    self.endpoint_url,
                    json=request_data,
                    headers=self._headers,
                    timeout=self.timeout_seconds,
                )
                self._validate_status(response)
                self._validate_declared_size(response.headers)
                try:
                    response_data = response.json()
                    encoded_size = len(
                        json.dumps(
                            response_data,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode()
                    )
                except (TypeError, ValueError) as exc:
                    raise AdapterSchemaError("AI endpoint response was not valid JSON.") from exc
                if encoded_size > self.max_response_bytes:
                    raise AdapterSchemaError("AI endpoint response exceeded the size limit.")
                status_code = response.status_code
                response_headers = response.headers
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise AdapterTimeoutError from exc
        except httpx.RequestError as exc:
            raise AdapterTransportError from exc

        latency_ms = (perf_counter() - started) * 1000
        output = self.extract_output(response_data)
        usage = self.extract_usage(response_data)
        response_id = response_headers.get("x-request-id") or response_headers.get("X-Request-ID")
        return AdapterResponse(
            output=output,
            status_code=status_code,
            latency_ms=latency_ms,
            usage=usage,
            response_id=response_id,
        )

    def _validate_declared_size(self, headers: Mapping[str, str]) -> None:
        raw = headers.get("content-length") or headers.get("Content-Length")
        if raw is None:
            return
        try:
            size = int(raw)
        except ValueError as exc:
            raise AdapterSchemaError("AI endpoint returned an invalid Content-Length.") from exc
        if size < 0 or size > self.max_response_bytes:
            raise AdapterSchemaError("AI endpoint response exceeded the size limit.")

    async def _stream_json(
        self,
        request_data: Mapping[str, Any],
    ) -> tuple[int, Mapping[str, str], Any]:
        client = cast(httpx.AsyncClient, self._client)
        async with client.stream(
            self.method,
            self.endpoint_url,
            json=request_data,
            headers=self._headers,
            timeout=self.timeout_seconds,
        ) as response:
            self._validate_status(cast(HTTPResponse, response))
            self._validate_declared_size(response.headers)
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > self.max_response_bytes:
                    raise AdapterSchemaError("AI endpoint response exceeded the size limit.")
            try:
                response_data = json.loads(content)
            except (TypeError, ValueError) as exc:
                raise AdapterSchemaError("AI endpoint response was not valid JSON.") from exc
            return response.status_code, dict(response.headers), response_data

    async def execute(self, payload: Mapping[str, Any]) -> AdapterResponse:
        return await self.invoke(payload)

    async def generate(self, payload: Mapping[str, Any]) -> AdapterResponse:
        return await self.invoke(payload)

    async def call(self, payload: Mapping[str, Any]) -> AdapterResponse:
        return await self.invoke(payload)

    def _validate_status(self, response: HTTPResponse) -> None:
        status_code = response.status_code
        if status_code == 401:
            raise AdapterUnauthorizedError
        if status_code == 429:
            raise AdapterRateLimitError(retry_after_seconds=_retry_after(response))
        if status_code >= 500:
            raise AdapterServerError(status_code)
        if status_code >= 400:
            raise AdapterRequestError(status_code)
        if status_code < 200 or status_code >= 300:
            raise AdapterRequestError(status_code)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> GenericRESTAdapter:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


GenericRestAdapter = GenericRESTAdapter

"""Request safety controls and PHI-safe operational instrumentation."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from uuid import uuid4

from fastapi import Request
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.api.errors import error_response
from backend.app.observability import MetricName, OperationalMetrics

logger = logging.getLogger("gatekeeper.api")
_SAFE_REQUEST_ID = re.compile(r"^req_[0-9a-f]{16,32}$")


def request_id_for_scope(scope: Scope) -> str:
    state = scope.setdefault("state", {})
    request_id = state.get("request_id")
    if isinstance(request_id, str) and _SAFE_REQUEST_ID.fullmatch(request_id):
        return request_id
    request_id = f"req_{uuid4().hex}"
    state["request_id"] = request_id
    return request_id


class RequestTooLargeError(Exception):
    """Raised while consuming a chunked request that exceeds the configured limit."""


class RequestBodyLimitMiddleware:
    """Reject declared and streamed request bodies above a fixed byte limit."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                response = error_response(
                    code="REQ_INVALID",
                    message="Content-Length must be a valid non-negative integer.",
                    request_id=request_id_for_scope(scope),
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = error_response(
                    code="REQ_INVALID",
                    message="Content-Length must be a valid non-negative integer.",
                    request_id=request_id_for_scope(scope),
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_size > self.max_bytes:
                scope.setdefault("state", {})["error_code"] = "REQ_TOO_LARGE"
                response = error_response(
                    code="REQ_TOO_LARGE",
                    message="The request body exceeds the configured size limit.",
                    request_id=request_id_for_scope(scope),
                    status_code=413,
                )
                await response(scope, receive, send)
                return

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    scope.setdefault("state", {})["error_code"] = "REQ_TOO_LARGE"
                    raise RequestTooLargeError
            return message

        await self.app(scope, limited_receive, send)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class RateLimiter(Protocol):
    def check(self, key: str) -> RateLimitDecision:
        """Return a decision for an opaque, non-secret caller key."""


class InMemoryRateLimiter:
    """Small fixed-window limiter replaceable by a distributed implementation."""

    def __init__(self, *, requests_per_window: int = 120, window_seconds: int = 60) -> None:
        if requests_per_window < 1 or window_seconds < 1:
            raise ValueError("rate-limit values must be positive")
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._buckets: dict[tuple[str, int], int] = {}

    def check(self, key: str) -> RateLimitDecision:
        now = int(time.monotonic())
        window = now // self.window_seconds
        bucket = (key, window)
        with self._lock:
            count = self._buckets.get(bucket, 0) + 1
            self._buckets[bucket] = count
            if len(self._buckets) > 10_000:
                self._buckets = {
                    existing: value
                    for existing, value in self._buckets.items()
                    if existing[1] >= window - 1
                }
        if count <= self.requests_per_window:
            return RateLimitDecision(allowed=True)
        retry_after = self.window_seconds - (now % self.window_seconds)
        return RateLimitDecision(allowed=False, retry_after_seconds=max(1, retry_after))


class RateLimitMiddleware:
    """Apply a per-Bearer-key limit without retaining the credential itself."""

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return
        # Authentication has not run yet, so a supplied Bearer token is not a
        # trustworthy caller identity. Bucket by the transport peer to prevent
        # rotating invalid credentials from bypassing the pre-authentication limit.
        client = scope.get("client")
        address = client[0] if client else "unknown"
        identity = f"client:{hashlib.sha256(address.encode()).hexdigest()}"
        decision = self.limiter.check(identity)
        if decision.allowed:
            await self.app(scope, receive, send)
            return
        scope.setdefault("state", {})["error_code"] = "REQ_RATE_LIMITED"
        response = error_response(
            code="REQ_RATE_LIMITED",
            message="The request rate limit was exceeded.",
            request_id=request_id_for_scope(scope),
            status_code=429,
            retryable=True,
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )
        await response(scope, receive, send)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign request IDs, safe logs, and low-cardinality request metrics."""

    def __init__(self, app: ASGIApp, *, metrics: OperationalMetrics) -> None:
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request_id_for_scope(request.scope)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Vary"] = "Authorization"

        route = request.scope.get("route")
        route_template = getattr(route, "path", "__unmatched__")
        if not isinstance(route_template, str) or not route_template.startswith("/"):
            route_template = "__unmatched__"
        if response.status_code >= 500:
            outcome = "server_error"
        elif response.status_code >= 400:
            outcome = "client_error"
        else:
            outcome = "success"
        raw_error_code = getattr(request.state, "error_code", "NONE")
        error_code = raw_error_code if isinstance(raw_error_code, str) else "INTERNAL_ERROR"
        labels: dict[str, str] = {
            "route_template": route_template,
            "method": request.method,
            "outcome": outcome,
            "error_code": error_code,
        }
        self.metrics.increment(
            MetricName.REQUESTS,
            route_template=route_template,
            method=request.method,
            outcome=outcome,
            error_code=error_code,
        )
        self.metrics.observe(MetricName.REQUEST_LATENCY, elapsed, **labels)
        logger.info(
            "request_complete request_id=%s method=%s route=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            route_template,
            response.status_code,
            elapsed * 1000,
        )
        return response

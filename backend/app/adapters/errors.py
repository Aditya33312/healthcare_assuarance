"""Sanitized adapter failures used by retry policy and orchestration."""

from __future__ import annotations

from backend.app.core.errors import ErrorCode


class AdapterError(Exception):
    """An endpoint failure whose message never embeds request or response content."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        status_code: int | None = None,
        retryable: bool = False,
        fatal: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.fatal = fatal
        self.retry_after_seconds = retry_after_seconds


class AdapterTimeoutError(AdapterError):
    def __init__(self) -> None:
        super().__init__(
            "Configured AI endpoint did not respond before the timeout.",
            code=ErrorCode.AI_TIMEOUT,
            retryable=True,
        )


class AdapterRateLimitError(AdapterError):
    def __init__(self, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(
            "Configured AI endpoint rate limited the request.",
            code=ErrorCode.AI_RATE_LIMITED,
            status_code=429,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )


class AdapterUnauthorizedError(AdapterError):
    def __init__(self) -> None:
        super().__init__(
            "Configured AI endpoint rejected its credentials.",
            code=ErrorCode.AI_UNAUTHORIZED,
            status_code=401,
            fatal=True,
        )


class AdapterServerError(AdapterError):
    def __init__(self, status_code: int) -> None:
        retryable = status_code in {502, 503}
        super().__init__(
            f"Configured AI endpoint returned HTTP {status_code}.",
            code=ErrorCode.AI_SERVER_ERROR,
            status_code=status_code,
            retryable=retryable,
        )


class AdapterRequestError(AdapterError):
    def __init__(self, status_code: int) -> None:
        super().__init__(
            f"Configured AI endpoint rejected the request with HTTP {status_code}.",
            code=ErrorCode.AI_INVALID_RESPONSE,
            status_code=status_code,
        )


class AdapterTransportError(AdapterError):
    def __init__(self) -> None:
        super().__init__(
            "Configured AI endpoint could not be reached.",
            code=ErrorCode.AI_SERVER_ERROR,
        )


class AdapterSchemaError(AdapterError):
    def __init__(
        self,
        message: str = "AI endpoint response did not match its configured schema.",
    ) -> None:
        super().__init__(
            message,
            code=ErrorCode.AI_INVALID_RESPONSE,
            fatal=True,
        )

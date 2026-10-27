"""Explicit transient-error retry policy."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.adapters.errors import AdapterError, AdapterTimeoutError
from backend.app.core.errors import ErrorCode
from backend.app.orchestration.protocols import JitterCallback


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be below base_delay_seconds")

    def should_retry(self, error: AdapterError, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if isinstance(error, AdapterTimeoutError) or error.code is ErrorCode.AI_TIMEOUT:
            return True
        return error.status_code in {429, 502, 503} and error.retryable

    def delay_for(
        self,
        failed_attempt: int,
        *,
        jitter: JitterCallback,
        retry_after_seconds: float | None = None,
    ) -> float:
        exponential = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** max(0, failed_attempt - 1)),
        )
        jitter_value = max(0.0, float(jitter(exponential)))
        delay: float = min(self.max_delay_seconds, exponential + jitter_value)
        if retry_after_seconds is not None:
            delay = max(delay, min(self.max_delay_seconds, retry_after_seconds))
        return float(delay)

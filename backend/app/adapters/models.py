"""Transport-neutral adapter results.

Only the explicit ``output`` field may contain customer clinical content. Operational
metadata is deliberately restricted to counts, timing, status, cost, and opaque request IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.total_tokens) < 0:
            raise ValueError("token counts cannot be negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        minimum_total = self.input_tokens + self.output_tokens
        if self.total_tokens == 0 and minimum_total:
            object.__setattr__(self, "total_tokens", minimum_total)
        elif self.total_tokens < minimum_total:
            raise ValueError("total_tokens cannot be less than input plus output tokens")

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens


@dataclass(frozen=True, slots=True)
class AdapterResponse:
    """Captured AI output plus PHI-safe operational accounting."""

    output: Any = field(repr=False)
    status_code: int
    latency_ms: float
    usage: AdapterUsage = field(default_factory=AdapterUsage)
    response_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 100 <= self.status_code <= 599:
            raise ValueError("status_code must be a valid HTTP status")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")

    @property
    def generated_output(self) -> Any:
        return self.output

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd

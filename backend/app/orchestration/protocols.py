"""Ports for evaluators, durable state, time, and replaceable job queues."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from backend.app.adapters.models import AdapterResponse
from backend.app.adapters.protocols import AIAdapter
from backend.app.orchestration.models import OrchestrationCase, StateTransition


class EvaluatorCallback(Protocol):
    def __call__(
        self, case: OrchestrationCase, response: AdapterResponse
    ) -> object | Awaitable[object]:
        """Evaluate a captured output or raise on technical failure."""


class StateTransitionCallback(Protocol):
    def __call__(self, transition: StateTransition) -> None | Awaitable[None]:
        """Durably persist a PHI-free transition before execution proceeds.

        Synchronous implementations run in a worker thread and therefore must create
        and close any thread-confined database session inside the callback.
        """


class StateTransitionStore(Protocol):
    def record_transition(self, transition: StateTransition) -> None | Awaitable[None]:
        """Method-based alternative with the same async/thread-safety contract."""


class BackgroundQueue(Protocol):
    async def enqueue(
        self,
        job_name: str,
        payload: Mapping[str, Any],
        *,
        job_id: str | None = None,
    ) -> str:
        """Enqueue serializable work and return the backend job identifier."""


type SleepCallback = Callable[[float], Awaitable[None]]
type JitterCallback = Callable[[float], float]

__all__ = [
    "AIAdapter",
    "BackgroundQueue",
    "EvaluatorCallback",
    "JitterCallback",
    "SleepCallback",
    "StateTransitionCallback",
    "StateTransitionStore",
]

"""Thin dispatcher over a replaceable Redis/Celery/RQ/ARQ-compatible queue port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.app.orchestration.protocols import BackgroundQueue

ORCHESTRATION_JOB_NAME = "gatekeeper.execute_test_run"


class BackgroundRunDispatcher:
    def __init__(self, queue: BackgroundQueue) -> None:
        self._queue = queue

    async def submit(self, run_id: str, payload: Mapping[str, Any]) -> str:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(payload, Mapping):
            raise TypeError("queue payload must be a mapping")
        return await self._queue.enqueue(
            ORCHESTRATION_JOB_NAME,
            payload,
            job_id=run_id,
        )


QueueDispatcher = BackgroundRunDispatcher

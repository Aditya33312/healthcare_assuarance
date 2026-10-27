"""Replaceable transport and AI adapter protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from backend.app.adapters.models import AdapterResponse


class AIAdapter(Protocol):
    name: str
    version: str

    async def invoke(self, payload: Mapping[str, Any]) -> AdapterResponse:
        """Call the system under test and return its captured output."""


class HTTPResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any:
        """Decode the endpoint response without logging it."""


class AsyncHTTPClient(Protocol):
    async def request(self, method: str, url: str, **kwargs: Any) -> HTTPResponse:
        """Execute an HTTP request."""

    async def aclose(self) -> None:
        """Close owned transport resources."""

"""FastAPI route support for precision-preserving JSON decimal tokens."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from decimal import Decimal
from typing import Any

from fastapi import Request, Response
from fastapi.routing import APIRoute


class DecimalJSONRequest(Request):
    """Decode non-integral JSON numbers directly to Decimal."""

    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            body = await self.body()
            self._json = json.loads(body, parse_float=Decimal)
        return self._json


class DecimalJSONRoute(APIRoute):
    """Provide DecimalJSONRequest to FastAPI's normal validation pipeline."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def precision_preserving_handler(request: Request) -> Response:
            decimal_request = DecimalJSONRequest(request.scope, request.receive)
            return await original(decimal_request)

        return precision_preserving_handler


__all__ = ["DecimalJSONRequest", "DecimalJSONRoute"]

"""HTTP error contracts that never reflect request bodies or secret material."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.responses import JSONResponse


@dataclass(slots=True)
class ApiError(Exception):
    code: str
    message: str
    http_status: int = 400
    retryable: bool = False


def error_response(
    *,
    code: str,
    message: str,
    request_id: str,
    status_code: int,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the single public error envelope used by middleware and handlers."""

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
        headers=headers,
    )

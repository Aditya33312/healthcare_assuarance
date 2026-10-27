"""Gatekeeper FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.api_keys import router as api_keys_router
from backend.app.api.errors import ApiError, error_response
from backend.app.api.middleware import (
    InMemoryRateLimiter,
    RateLimiter,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
    RequestTooLargeError,
    request_id_for_scope,
)
from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.observability import OperationalMetrics
from backend.app.version import ENGINE_VERSION

logger = logging.getLogger("gatekeeper.api")


def _request_id(request: Request) -> str:
    return request_id_for_scope(request.scope)


def _mark_error(request: Request, code: str) -> None:
    request.state.error_code = code


def _integrity_error_kind(exc: IntegrityError) -> str:
    """Classify only stable driver codes; never inspect a PHI-bearing message."""

    original = exc.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    sqlite_name = getattr(original, "sqlite_errorname", None)
    if sqlstate == "23505" or sqlite_name in {
        "SQLITE_CONSTRAINT_PRIMARYKEY",
        "SQLITE_CONSTRAINT_UNIQUE",
    }:
        return "unique"
    if sqlstate == "23503" or sqlite_name == "SQLITE_CONSTRAINT_FOREIGNKEY":
        return "foreign_key"
    return "unknown"


def create_app(
    *,
    settings: Settings | None = None,
    rate_limiter: RateLimiter | None = None,
    metrics: OperationalMetrics | None = None,
) -> FastAPI:
    """Create an isolated app whose database/encryption dependencies are overridable."""

    runtime_settings = settings or get_settings()
    operational_metrics = metrics or OperationalMetrics()
    limiter = rate_limiter or InMemoryRateLimiter()
    database_engine = create_database_engine(runtime_settings.database_url)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            database_engine.dispose()

    application = FastAPI(
        title="Healthcare AI Gatekeeper API",
        version=ENGINE_VERSION,
        description=(
            "Tenant-scoped assurance API. Test runs are durably queued; this HTTP process "
            "does not simulate worker completion or release decisions."
        ),
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.database_engine = database_engine
    application.state.session_factory = create_session_factory(database_engine)
    application.state.operational_metrics = operational_metrics
    application.state.rate_limiter = limiter

    @application.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "gatekeeper", "version": ENGINE_VERSION}

    application.include_router(router, prefix="/v1", tags=["v1"])
    application.include_router(api_keys_router, prefix="/v1", tags=["api-keys"])

    @application.exception_handler(GatekeeperError)
    async def handle_gatekeeper_error(request: Request, exc: GatekeeperError) -> JSONResponse:
        code = exc.code.value
        _mark_error(request, code)
        headers = {"WWW-Authenticate": "Bearer"} if exc.http_status == 401 else None
        return error_response(
            code=code,
            message=exc.message,
            retryable=exc.retryable,
            request_id=_request_id(request),
            status_code=exc.http_status,
            headers=headers,
        )

    @application.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        _mark_error(request, exc.code)
        return error_response(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            request_id=_request_id(request),
            status_code=exc.http_status,
        )

    @application.exception_handler(RequestTooLargeError)
    async def handle_streamed_size_error(
        request: Request, exc: RequestTooLargeError
    ) -> JSONResponse:
        del exc
        _mark_error(request, ErrorCode.REQ_TOO_LARGE.value)
        return error_response(
            code=ErrorCode.REQ_TOO_LARGE.value,
            message="The request body exceeds the configured size limit.",
            request_id=_request_id(request),
            status_code=413,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        validation_errors = exc.errors()
        locations = [tuple(error.get("loc", ())) for error in validation_errors]
        error_types = [str(error.get("type", "")) for error in validation_errors]
        source_errors = [
            (location, error_type)
            for location, error_type in zip(locations, error_types, strict=True)
            if "sources" in location
        ]
        output_errors = [
            (location, error_type)
            for location, error_type in zip(locations, error_types, strict=True)
            if "generated_output" in location
        ]
        if source_errors and any(
            error_type in {"union_tag_invalid", "union_tag_not_found"}
            for _, error_type in source_errors
        ):
            code = ErrorCode.UNSUPPORTED_SOURCE_TYPE.value
            message = "A source document used an unsupported evidence type."
        elif output_errors and any(
            error_type in {"union_tag_invalid", "union_tag_not_found"}
            for _, error_type in output_errors
        ):
            code = ErrorCode.UNSUPPORTED_OUTPUT_TYPE.value
            message = "The generated output used an unsupported evidence type."
        elif any(
            location[-1:] == ("sources",) and error_type in {"missing", "too_short"}
            for location, error_type in source_errors
        ):
            code = ErrorCode.REQ_MISSING_SOURCE.value
            message = "At least one valid source document is required."
        elif any(
            location[-1:] == ("generated_output",) and error_type == "missing"
            for location, error_type in output_errors
        ):
            code = ErrorCode.REQ_MISSING_OUTPUT.value
            message = "A valid generated output is required."
        elif source_errors:
            code = ErrorCode.REQ_INVALID.value
            message = "One or more source documents did not match the required schema."
        elif output_errors:
            code = ErrorCode.REQ_INVALID.value
            message = "The generated output did not match the required schema."
        else:
            code = ErrorCode.REQ_INVALID.value
            message = "The request did not match the required schema."
        _mark_error(request, code)
        return error_response(
            code=code,
            message=message,
            request_id=_request_id(request),
            status_code=422,
        )

    @application.exception_handler(IntegrityError)
    async def handle_integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
        kind = _integrity_error_kind(exc)
        if kind == "unique":
            code = ErrorCode.RESOURCE_CONFLICT.value
            message = "The resource conflicts with an existing tenant resource."
            retryable = False
            status_code = 409
        elif kind == "foreign_key":
            code = ErrorCode.RESOURCE_REFERENCE_INVALID.value
            message = "A referenced resource is no longer available."
            retryable = False
            status_code = 409
        else:
            code = ErrorCode.DATABASE_UNAVAILABLE.value
            message = "The persistence service could not safely complete the request."
            retryable = True
            status_code = 503
        _mark_error(request, code)
        return error_response(
            code=code,
            message=message,
            retryable=retryable,
            request_id=_request_id(request),
            status_code=status_code,
        )

    @application.exception_handler(SQLAlchemyError)
    async def handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        # Never render or log driver messages: bound parameters may contain protected data.
        del exc
        code = ErrorCode.DATABASE_UNAVAILABLE.value
        _mark_error(request, code)
        logger.error("database_operation_failed request_id=%s", _request_id(request))
        return error_response(
            code=code,
            message="The persistence service is unavailable.",
            retryable=True,
            request_id=_request_id(request),
            status_code=503,
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if getattr(request.state, "error_code", None) == ErrorCode.REQ_TOO_LARGE.value:
            code = ErrorCode.REQ_TOO_LARGE.value
            message = "The request body exceeds the configured size limit."
            status_code = 413
        elif exc.status_code == 404:
            code = "REQ_NOT_FOUND"
            message = "The requested API route was not found."
            status_code = exc.status_code
        elif exc.status_code == 405:
            code = "REQ_METHOD_NOT_ALLOWED"
            message = "The HTTP method is not supported for this API route."
            status_code = exc.status_code
        else:
            code = ErrorCode.REQ_INVALID.value
            message = "The request could not be processed."
            status_code = exc.status_code
        _mark_error(request, code)
        return error_response(
            code=code,
            message=message,
            request_id=_request_id(request),
            status_code=status_code,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Exception messages and reprs can contain request data; record only the type.
        code = "INTERNAL_ERROR"
        _mark_error(request, code)
        logger.error(
            "unexpected_application_error request_id=%s exception_type=%s",
            _request_id(request),
            type(exc).__name__,
        )
        return error_response(
            code=code,
            message="The request could not be completed.",
            request_id=_request_id(request),
            status_code=500,
        )

    # Starlette applies the most recently added middleware first. Context is
    # outermost so every response, including an early size/rate rejection, is measured.
    application.add_middleware(RateLimitMiddleware, limiter=limiter)
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=runtime_settings.max_request_bytes,
    )
    application.add_middleware(RequestContextMiddleware, metrics=operational_metrics)
    return application


app = create_app()

__all__ = ["app", "create_app"]

"""FastAPI dependencies with explicit test/runtime replacement seams."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import cast

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.auth import ApiKeyAuthenticator, ApiKeyPrincipal, ApiScope
from backend.app.core.config import Settings
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.security import EncryptedJsonStore

_bearer = HTTPBearer(auto_error=False, description="Gatekeeper API key")


def get_api_settings(request: Request) -> Settings:
    """Return app-local settings so separate test apps cannot leak configuration."""

    return cast(Settings, request.app.state.settings)


def get_db_session(request: Request) -> Generator[Session, None, None]:
    """Yield a unit-of-work session; route handlers own their commit boundary."""

    session_factory = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_encryptor(
    settings: Settings = Depends(get_api_settings),
) -> EncryptedJsonStore:
    """Construct the protected artifact store; tests can override this dependency."""

    return EncryptedJsonStore(settings.data_encryption_keys)


def require_scopes(*required_scopes: ApiScope) -> Callable[..., ApiKeyPrincipal]:
    """Create an Authorization Bearer dependency for an exact scope set."""

    def authenticate(
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
        session: Session = Depends(get_db_session),
        settings: Settings = Depends(get_api_settings),
    ) -> ApiKeyPrincipal:
        if credentials is None or credentials.scheme.casefold() != "bearer":
            raise GatekeeperError(
                ErrorCode.AUTH_INVALID_KEY,
                "A valid Bearer API key is required.",
                http_status=401,
            )
        return ApiKeyAuthenticator(session, settings.api_key_pepper).authenticate(
            credentials.credentials,
            required_scopes=set(required_scopes),
        )

    return authenticate

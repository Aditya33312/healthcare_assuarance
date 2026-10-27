"""API-key authentication and scope enforcement."""

from backend.app.auth.api_keys import (
    ApiKeyAuthenticator,
    ApiKeyPrincipal,
    CreatedApiKey,
    create_api_key,
    revoke_api_key,
)
from backend.app.auth.bootstrap import (
    BootstrapAlreadyInitializedError,
    BootstrapResult,
    bootstrap_first_admin,
)
from backend.app.auth.scopes import ApiScope

__all__ = [
    "ApiKeyAuthenticator",
    "ApiKeyPrincipal",
    "ApiScope",
    "BootstrapAlreadyInitializedError",
    "BootstrapResult",
    "CreatedApiKey",
    "bootstrap_first_admin",
    "create_api_key",
    "revoke_api_key",
]

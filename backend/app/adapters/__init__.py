"""Customer AI adapter layer."""

from backend.app.adapters.errors import (
    AdapterError,
    AdapterRateLimitError,
    AdapterRequestError,
    AdapterSchemaError,
    AdapterServerError,
    AdapterTimeoutError,
    AdapterTransportError,
    AdapterUnauthorizedError,
)
from backend.app.adapters.field_mapping import (
    CUSTOM_FIELD_MAPPING_ADAPTER_VERSION,
    CustomJSONAdapter,
    CustomJSONFieldMappingAdapter,
)
from backend.app.adapters.generic_rest import (
    GENERIC_REST_ADAPTER_VERSION,
    GenericRESTAdapter,
    GenericRestAdapter,
)
from backend.app.adapters.models import AdapterResponse, AdapterUsage
from backend.app.adapters.openai_compatible import (
    OPENAI_COMPATIBLE_ADAPTER_VERSION,
    OpenAICompatibleAdapter,
)
from backend.app.adapters.protocols import AIAdapter, AsyncHTTPClient, HTTPResponse

__all__ = [
    "CUSTOM_FIELD_MAPPING_ADAPTER_VERSION",
    "GENERIC_REST_ADAPTER_VERSION",
    "OPENAI_COMPATIBLE_ADAPTER_VERSION",
    "AIAdapter",
    "AdapterError",
    "AdapterRateLimitError",
    "AdapterRequestError",
    "AdapterResponse",
    "AdapterSchemaError",
    "AdapterServerError",
    "AdapterTimeoutError",
    "AdapterTransportError",
    "AdapterUnauthorizedError",
    "AdapterUsage",
    "AsyncHTTPClient",
    "CustomJSONAdapter",
    "CustomJSONFieldMappingAdapter",
    "GenericRESTAdapter",
    "GenericRestAdapter",
    "HTTPResponse",
    "OpenAICompatibleAdapter",
]

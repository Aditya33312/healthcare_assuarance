"""Custom request and response JSON field-mapping adapter."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from backend.app.adapters.generic_rest import GenericRESTAdapter, UsageExtractor
from backend.app.adapters.json_paths import JSONPath, get_path, set_path
from backend.app.adapters.protocols import AsyncHTTPClient

CUSTOM_FIELD_MAPPING_ADAPTER_VERSION = "custom-json-field-mapping-adapter@0.1.0"


class CustomJSONFieldMappingAdapter(GenericRESTAdapter):
    """Map standard input paths to a customer's nested request schema.

    ``request_mapping`` is ``{source_path: target_path}``. For example,
    ``{"source": "consultation_text"}`` maps the standard ``source`` input to the
    customer's ``consultation_text`` request field.
    """

    name = "custom_json_field_mapping"
    version = CUSTOM_FIELD_MAPPING_ADAPTER_VERSION

    def __init__(
        self,
        endpoint_url: str,
        *,
        request_mapping: Mapping[str, JSONPath] | None = None,
        field_mapping: Mapping[str, JSONPath] | None = None,
        request_field_mapping: Mapping[str, JSONPath] | None = None,
        response_path: JSONPath = "output",
        response_field: str | None = None,
        static_fields: Mapping[str, Any] | None = None,
        include_unmapped_fields: bool = False,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: AsyncHTTPClient | None = None,
        usage_extractor: UsageExtractor | None = None,
        max_response_bytes: int = 2_097_152,
    ) -> None:
        supplied_mappings = [
            mapping
            for mapping in (request_mapping, field_mapping, request_field_mapping)
            if mapping is not None
        ]
        if len(supplied_mappings) > 1:
            raise ValueError(
                "provide only one of request_mapping, field_mapping, or request_field_mapping"
            )
        mapping = supplied_mappings[0] if supplied_mappings else None
        self.request_mapping = dict(mapping or {})
        self.static_fields = deepcopy(dict(static_fields or {}))
        self.include_unmapped_fields = include_unmapped_fields
        super().__init__(
            endpoint_url,
            response_path=response_field if response_field is not None else response_path,
            method=method,
            headers=headers,
            timeout_seconds=timeout_seconds,
            client=client,
            usage_extractor=usage_extractor,
            max_response_bytes=max_response_bytes,
        )

    def build_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("adapter payload must be a mapping")
        request_data = deepcopy(dict(payload)) if self.include_unmapped_fields else {}
        for path, value in self.static_fields.items():
            set_path(request_data, path, deepcopy(value))
        for source_path, target_path in self.request_mapping.items():
            set_path(request_data, target_path, deepcopy(get_path(payload, source_path)))
        return request_data


CustomJSONAdapter = CustomJSONFieldMappingAdapter

"""OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.app.adapters.errors import AdapterSchemaError
from backend.app.adapters.generic_rest import GenericRESTAdapter
from backend.app.adapters.models import AdapterUsage
from backend.app.adapters.protocols import AsyncHTTPClient

OPENAI_COMPATIBLE_ADAPTER_VERSION = "openai-compatible-adapter@0.1.0"


def _chat_completions_url(endpoint_url: str | None, base_url: str | None) -> str:
    if endpoint_url and base_url:
        raise ValueError("provide endpoint_url or base_url, not both")
    if endpoint_url:
        return endpoint_url
    if not base_url:
        raise ValueError("endpoint_url or base_url is required")
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


class OpenAICompatibleAdapter(GenericRESTAdapter):
    """Translate standard prompts to OpenAI-compatible chat requests."""

    name = "openai_compatible"
    version = OPENAI_COMPATIBLE_ADAPTER_VERSION

    def __init__(
        self,
        endpoint_url: str | None = None,
        *,
        base_url: str | None = None,
        model: str,
        api_key: str | None = None,
        prompt_field: str = "prompt",
        system_prompt: str | None = None,
        request_options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        input_cost_per_1k_tokens: float = 0.0,
        output_cost_per_1k_tokens: float = 0.0,
        client: AsyncHTTPClient | None = None,
        max_response_bytes: int = 2_097_152,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if min(input_cost_per_1k_tokens, output_cost_per_1k_tokens) < 0:
            raise ValueError("token pricing cannot be negative")
        request_headers = dict(headers or {})
        request_headers.setdefault("content-type", "application/json")
        if api_key is not None:
            request_headers.setdefault("authorization", f"Bearer {api_key}")
        self.model = model
        self.prompt_field = prompt_field
        self.system_prompt = system_prompt
        self.request_options = dict(request_options or {})
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens
        super().__init__(
            _chat_completions_url(endpoint_url, base_url),
            response_path=None,
            headers=request_headers,
            timeout_seconds=timeout_seconds,
            client=client,
            max_response_bytes=max_response_bytes,
        )

    def build_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("adapter payload must be a mapping")
        messages_value = payload.get("messages")
        if messages_value is not None:
            if not isinstance(messages_value, Sequence) or isinstance(
                messages_value, (str, bytes, bytearray)
            ):
                raise AdapterSchemaError("OpenAI-compatible messages must be a JSON array.")
            messages = [dict(message) for message in messages_value if isinstance(message, Mapping)]
            if len(messages) != len(messages_value) or not messages:
                raise AdapterSchemaError("OpenAI-compatible messages had the wrong shape.")
        else:
            prompt = payload.get(self.prompt_field)
            if prompt is None:
                for fallback in ("source", "input", "content", "question"):
                    if payload.get(fallback) is not None:
                        prompt = payload[fallback]
                        break
            if not isinstance(prompt, str) or not prompt.strip():
                raise AdapterSchemaError("OpenAI-compatible request requires a non-empty prompt.")
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

        request_data = dict(self.request_options)
        request_data.update({"model": self.model, "messages": messages})
        return request_data

    def extract_output(self, response_data: Any) -> str:
        try:
            choices = response_data["choices"]
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", first_choice.get("text"))
        except (IndexError, KeyError, TypeError) as exc:
            raise AdapterSchemaError(
                "OpenAI-compatible response did not contain a completion."
            ) from exc

        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                text_part = item.get("text")
                if isinstance(text_part, str):
                    text_parts.append(text_part)
            joined = "".join(text_parts)
            if joined.strip():
                return joined
        raise AdapterSchemaError("OpenAI-compatible completion content was empty or invalid.")

    def extract_usage(self, response_data: Any) -> AdapterUsage:
        usage = response_data.get("usage", {}) if isinstance(response_data, Mapping) else {}
        if not isinstance(usage, Mapping):
            raise AdapterSchemaError("OpenAI-compatible usage had the wrong shape.")
        try:
            input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        except (TypeError, ValueError) as exc:
            raise AdapterSchemaError("OpenAI-compatible token usage was invalid.") from exc
        cost = (
            input_tokens * self.input_cost_per_1k_tokens
            + output_tokens * self.output_cost_per_1k_tokens
        ) / 1000
        try:
            return AdapterUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
            )
        except ValueError as exc:
            raise AdapterSchemaError("OpenAI-compatible token usage was invalid.") from exc

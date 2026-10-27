"""Meaning-preserving transport normalization and deterministic serialization."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from decimal import Decimal
from math import isfinite
from typing import Any

from pydantic import BaseModel

MAX_DECIMAL_DIGITS = 100
MAX_DECIMAL_ADJUSTED_EXPONENT = 100
MIN_DECIMAL_ADJUSTED_EXPONENT = -100


class UnsafeTextError(ValueError):
    """Raised when text contains characters unsafe for clinical processing."""


def validate_decimal_bounds(value: Decimal) -> Decimal:
    """Reject finite but expansion-prone numbers before rendering or hashing."""

    if not value.is_finite():
        raise ValueError("clinical numeric values must be finite")
    if len(value.as_tuple().digits) > MAX_DECIMAL_DIGITS:
        raise ValueError("clinical numeric values exceed the precision limit")
    if value and not (
        MIN_DECIMAL_ADJUSTED_EXPONENT
        <= value.adjusted()
        <= MAX_DECIMAL_ADJUSTED_EXPONENT
    ):
        raise ValueError("clinical numeric values exceed the magnitude limit")
    return value


def validate_text(value: str, *, allow_blank: bool = False) -> str:
    """Reject surrogates and unsafe controls without rewriting clinical content."""

    if not isinstance(value, str):
        raise TypeError("clinical text must be a string")
    if not allow_blank and not value.strip():
        raise UnsafeTextError("clinical text must not be blank")
    for character in value:
        codepoint = ord(character)
        if unicodedata.category(character) == "Cs":
            raise UnsafeTextError("clinical text contains an invalid Unicode surrogate")
        if codepoint == 0 or (codepoint < 32 and character not in {"\t", "\n", "\r"}):
            raise UnsafeTextError("clinical text contains an unsupported control character")
    return value


def normalize_text(value: str) -> str:
    """Normalize transport line endings only; never strip or case-fold evidence."""

    validate_text(value)
    return value.replace("\r\n", "\n").replace("\r", "\n")


def json_safe(value: Any) -> Any:
    """Convert supported request values to deterministic JSON-safe primitives."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically while retaining numeric JSON types."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        validate_decimal_bounds(value)
        return format(value, "f")
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numeric values must be finite")
        return format(validate_decimal_bounds(Decimal(str(value))), "f")
    if isinstance(value, dict):
        ordered = sorted(((str(key), item) for key, item in value.items()))
        return "{" + ",".join(
            f"{json.dumps(key, ensure_ascii=False)}:{canonical_json(item)}"
            for key, item in ordered
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def raw_serialization(value: Any) -> str:
    """Use the original text bytes for text and canonical JSON for structures."""

    return value if isinstance(value, str) else canonical_json(value)


def content_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def display_scalar(value: Any) -> str:
    """Render a structured scalar without lossy numeric formatting."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        validate_text(value, allow_blank=True)
        return value
    return str(value)

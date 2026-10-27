"""Minimal deterministic JSON path helpers for adapter field mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.app.adapters.errors import AdapterSchemaError

type PathPart = str | int
type JSONPath = str | Sequence[PathPart] | None


def parse_path(path: JSONPath) -> tuple[PathPart, ...]:
    if path is None or (isinstance(path, str) and path in {"", "$"}):
        return ()
    if isinstance(path, str):
        if path.startswith("/"):
            return tuple(part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:])
        return tuple(part for part in path.removeprefix("$.").split(".") if part)
    return tuple(path)


def get_path(value: Any, path: JSONPath) -> Any:
    current = value
    for part in parse_path(path):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                index = int(part)
                current = current[index]
                continue
            except (IndexError, TypeError, ValueError):
                pass
        raise AdapterSchemaError("Configured JSON field was missing or had the wrong shape.")
    return current


def set_path(target: dict[str, Any], path: JSONPath, value: Any) -> None:
    parts = parse_path(path)
    if not parts:
        if not isinstance(value, Mapping):
            raise AdapterSchemaError("A root field mapping must produce a JSON object.")
        target.clear()
        target.update(value)
        return

    current = target
    for part in parts[:-1]:
        key = str(part)
        existing = current.get(key)
        if existing is None:
            nested: dict[str, Any] = {}
            current[key] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise AdapterSchemaError("Configured request mappings conflict with each other.")
    current[str(parts[-1])] = value

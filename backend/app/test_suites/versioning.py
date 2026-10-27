"""Canonical immutable case membership for a test-suite version."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

CASE_VERSION_IDS_KEY = "case_version_ids"


def case_version_snapshot_digest(case_version_ids: Sequence[str]) -> str:
    """Return a deterministic digest for an ordered immutable membership list."""

    values = list(case_version_ids)
    if not all(isinstance(item, str) and item for item in values):
        raise ValueError("case version IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("case version IDs must be unique")
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def case_version_ids_from_definition(definition: Mapping[str, Any]) -> tuple[str, ...]:
    """Return ordered membership, supporting the prior structured-case form."""

    raw = definition.get(CASE_VERSION_IDS_KEY)
    if raw is None:
        cases = definition.get("cases")
        if isinstance(cases, list):
            raw = [item.get("test_case_version_id") for item in cases if isinstance(item, Mapping)]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return ()
    values = tuple(raw)
    if len(values) != len(set(values)):
        return ()
    return values


def with_case_version_snapshot(
    definition: Mapping[str, Any],
    case_version_ids: Sequence[str],
) -> dict[str, Any]:
    """Copy a caller definition and attach ordered immutable membership."""

    values = list(case_version_ids)
    if not all(isinstance(item, str) and item for item in values):
        raise ValueError("case version IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("case version IDs must be unique")
    snapshot = dict(definition)
    snapshot[CASE_VERSION_IDS_KEY] = values
    return snapshot

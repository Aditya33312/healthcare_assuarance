#!/usr/bin/env python3
"""Submit local evidence files to Gatekeeper without putting API keys in argv."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import httpx

SOURCE_TYPES = (
    "clinical_text",
    "clinical_note",
    "transcript",
    "lab_results",
    "medication_list",
    "structured_json",
    "rag_context",
)
OUTPUT_TYPES = ("clinical_text", "clinical_note", "structured_json")
_LOCAL_API_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True, slots=True)
class CliConfig:
    api_url: str
    system_id: str
    system_version: str
    environment: str
    source_paths: tuple[Path, ...]
    source_type: str | None
    output_path: Path
    output_type: str | None
    timeout_seconds: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read local text or JSON evidence, serialize it safely, and submit one "
            "Gatekeeper evaluation. The API key is read only from GATEKEEPER_API_KEY."
        )
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GATEKEEPER_API_URL", "http://127.0.0.1:8000"),
        help="Gatekeeper origin (default: GATEKEEPER_API_URL or local development API)",
    )
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--system-version", required=True)
    parser.add_argument(
        "--environment",
        choices=("test", "staging", "production"),
        default="staging",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        dest="source_paths",
        help="Source .txt or .json file; repeat for multiple source documents",
    )
    parser.add_argument(
        "--source-type",
        choices=SOURCE_TYPES,
        help=(
            "Type for raw source files. Defaults to clinical_text for text and "
            "structured_json for raw JSON. A source object in JSON supplies its own type."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument(
        "--output-type",
        choices=OUTPUT_TYPES,
        help=(
            "Type for raw generated output. Defaults to clinical_text for text and "
            "structured_json for raw JSON."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP response timeout for a synchronous evaluation (default: 300)",
    )
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> CliConfig:
    args = _parser().parse_args(argv)
    source_paths = tuple(cast(list[Path], args.source_paths))
    timeout_seconds = cast(float, args.timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be greater than zero")
    return CliConfig(
        api_url=cast(str, args.api_url),
        system_id=cast(str, args.system_id),
        system_version=cast(str, args.system_version),
        environment=cast(str, args.environment),
        source_paths=source_paths,
        source_type=cast(str | None, args.source_type),
        output_path=cast(Path, args.output_path),
        output_type=cast(str | None, args.output_type),
        timeout_seconds=timeout_seconds,
    )


def _api_origin(value: str) -> str:
    if value != value.strip() or not value:
        raise ValueError("--api-url must not be empty or contain surrounding whitespace")
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("--api-url must be an origin without credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("--api-url must not contain a path, query, or fragment")
    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold().rstrip(".")
    if scheme != "https" and not (scheme == "http" and hostname in _LOCAL_API_HOSTS):
        raise ValueError("--api-url must use HTTPS, except for a local development API")
    # Accessing port validates malformed/out-of-range port spellings.
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("--api-url contains an invalid port") from exc
    return f"{scheme}://{parsed.netloc}"


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        content = handle.read()
    if not content.strip():
        raise ValueError(f"{path} is empty or contains only whitespace")
    return content


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return cast(object, json.load(handle))


def _string_key_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _derived_document_id(path: Path, index: int) -> str:
    stem = path.stem.strip() or "source"
    safe_stem = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in stem
    )
    safe_stem = safe_stem.strip("-._") or "source"
    return f"source-{index:03d}-{safe_stem[:80]}"


def _complete_source(
    value: object,
    *,
    path: Path,
    index: int,
) -> dict[str, object]:
    source = _string_key_mapping(value, label=f"source entry in {path}")
    content = source.get("content")
    if content is None:
        raise ValueError(f"source entry in {path} is missing content")
    document_id = source.get("document_id", _derived_document_id(path, index))
    source_type = source.get("type", "clinical_text")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(f"source entry in {path} has an invalid document_id")
    if not isinstance(source_type, str) or not source_type.strip():
        raise ValueError(f"source entry in {path} has an invalid type")
    return {**source, "document_id": document_id, "type": source_type}


def _sources_from_json(
    value: object,
    *,
    path: Path,
    start_index: int,
    raw_source_type: str,
) -> list[dict[str, object]]:
    if isinstance(value, dict) and "sources" in value:
        envelope = _string_key_mapping(value, label=str(path))
        value = envelope["sources"]

    if isinstance(value, list):
        source_like = all(
            isinstance(item, dict) and "content" in item
            for item in value
        )
        if source_like:
            if not value:
                raise ValueError(f"{path} contains an empty source list")
            return [
                _complete_source(item, path=path, index=start_index + offset)
                for offset, item in enumerate(value)
            ]

    if isinstance(value, dict) and "content" in value:
        return [_complete_source(value, path=path, index=start_index)]

    return [
        {
            "document_id": _derived_document_id(path, start_index),
            "type": raw_source_type,
            "content": value,
        }
    ]


def _load_sources(config: CliConfig) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for path in config.source_paths:
        if path.suffix.casefold() == ".json":
            value = _read_json(path)
            raw_type = config.source_type or "structured_json"
            sources.extend(
                _sources_from_json(
                    value,
                    path=path,
                    start_index=len(sources) + 1,
                    raw_source_type=raw_type,
                )
            )
        else:
            source_type = config.source_type or "clinical_text"
            if source_type in {
                "transcript",
                "lab_results",
                "medication_list",
                "rag_context",
                "structured_json",
            }:
                raise ValueError(
                    f"{path} is text, but {source_type} requires JSON content; "
                    "use clinical_text for a raw multiline transcript"
                )
            sources.append(
                {
                    "document_id": _derived_document_id(path, len(sources) + 1),
                    "type": source_type,
                    "content": _read_text(path),
                }
            )

    document_ids = [source["document_id"] for source in sources]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("source document_id values must be unique")
    return sources


def _load_generated_output(config: CliConfig) -> dict[str, object]:
    path = config.output_path
    if path.suffix.casefold() != ".json":
        output_type = config.output_type or "clinical_text"
        if output_type == "structured_json":
            raise ValueError(f"{path} is text, but structured_json output requires a JSON file")
        return {"type": output_type, "content": _read_text(path)}

    value = _read_json(path)
    if isinstance(value, dict) and "generated_output" in value:
        envelope = _string_key_mapping(value, label=str(path))
        value = envelope["generated_output"]
    if isinstance(value, dict) and "content" in value:
        output = _string_key_mapping(value, label=f"generated output in {path}")
        declared_output_type = output.get("type", config.output_type or "clinical_text")
        if not isinstance(declared_output_type, str) or not declared_output_type.strip():
            raise ValueError(f"generated output in {path} has an invalid type")
        return {**output, "type": declared_output_type}
    return {"type": config.output_type or "structured_json", "content": value}


def _safe_error(response: httpx.Response) -> tuple[str, str, str | None]:
    try:
        payload = cast(object, response.json())
    except ValueError:
        return "HTTP_ERROR", "Gatekeeper returned a non-JSON error response.", None
    if not isinstance(payload, dict):
        return "HTTP_ERROR", "Gatekeeper returned an invalid error response.", None
    error = payload.get("error")
    if not isinstance(error, dict):
        return "HTTP_ERROR", "Gatekeeper returned an invalid error response.", None
    code = error.get("code")
    message = error.get("message")
    request_id = error.get("request_id")
    return (
        code if isinstance(code, str) else "HTTP_ERROR",
        message if isinstance(message, str) else "Gatekeeper rejected the request.",
        request_id if isinstance(request_id, str) else None,
    )


def _mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        return {}
    nested = value.get(key)
    return cast(Mapping[str, object], nested) if isinstance(nested, dict) else {}


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _redact_secret(value: str, secret: str | None) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _print_summary(payload: object, *, api_key: str) -> bool:
    if not isinstance(payload, dict):
        print("Gatekeeper returned an invalid success response.", file=sys.stderr)
        return False
    evaluation_id = payload.get("evaluation_id")
    execution_status = payload.get("execution_status")
    coverage = _mapping(payload, "coverage")
    metrics = _mapping(payload, "faithfulness")
    coverage_status = coverage.get("status")
    limitations = coverage.get("limitations")
    limitation_count = len(limitations) if isinstance(limitations, list) else 0

    evaluation_label = evaluation_id if isinstance(evaluation_id, str) else "unknown"
    execution_label = execution_status if isinstance(execution_status, str) else "unknown"
    coverage_label = coverage_status if isinstance(coverage_status, str) else "unknown"
    print(f"Evaluation: {_redact_secret(evaluation_label, api_key)}")
    print(f"Execution: {_redact_secret(execution_label, api_key)}")
    print(f"Coverage: {_redact_secret(coverage_label, api_key)}")
    print(f"Coverage limitations: {limitation_count}")
    score = metrics.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool):
        print(f"Faithfulness score: {score}")
    print(
        "Findings: "
        f"supported={_integer(metrics.get('supported'))}, "
        f"altered={_integer(metrics.get('altered'))}, "
        f"added={_integer(metrics.get('added'))}, "
        f"omitted={_integer(metrics.get('omitted'))}, "
        f"inferred={_integer(metrics.get('inferred'))}, "
        f"uncertain={_integer(metrics.get('uncertain'))}, "
        f"critical={_integer(metrics.get('critical_count'))}, "
        f"high={_integer(metrics.get('high_count'))}"
    )
    return execution_status == "COMPLETED" and coverage_status != "UNASSESSED"


def run(config: CliConfig, *, api_key: str, client: httpx.Client | None = None) -> int:
    api_origin = _api_origin(config.api_url)
    payload: dict[str, object] = {
        "system_id": config.system_id,
        "system_version": config.system_version,
        "environment": config.environment,
        "sources": _load_sources(config),
        "generated_output": _load_generated_output(config),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Idempotency-Key": f"file-helper:{uuid.uuid4().hex}",
    }
    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=httpx.Timeout(config.timeout_seconds, connect=10.0),
        follow_redirects=False,
    )
    try:
        response = http_client.post(
            f"{api_origin}/v1/evaluations",
            headers=headers,
            json=payload,
        )
    finally:
        if owns_client:
            http_client.close()

    if response.status_code != 201:
        code, message, request_id = _safe_error(response)
        safe_code = _redact_secret(code, api_key)
        safe_message = _redact_secret(message, api_key)
        safe_request_id = _redact_secret(request_id, api_key) if request_id else None
        request_suffix = f" (request_id={safe_request_id})" if safe_request_id else ""
        print(
            f"Evaluation rejected: HTTP {response.status_code} "
            f"{safe_code}: {safe_message}{request_suffix}",
            file=sys.stderr,
        )
        return 1
    try:
        response_payload = cast(object, response.json())
    except ValueError:
        print("Gatekeeper returned a non-JSON success response.", file=sys.stderr)
        return 1
    return 0 if _print_summary(response_payload, api_key=api_key) else 1


def main(argv: Sequence[str] | None = None) -> int:
    api_key = os.environ.get("GATEKEEPER_API_KEY")
    try:
        config = _parse_config(argv)
        if not api_key:
            print("GATEKEEPER_API_KEY is not configured.", file=sys.stderr)
            return 2
        return run(config, api_key=api_key)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Input error: {_redact_secret(str(exc), api_key)}", file=sys.stderr)
        return 2
    except httpx.HTTPError as exc:
        print(f"Evaluation transport failed: {type(exc).__name__}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

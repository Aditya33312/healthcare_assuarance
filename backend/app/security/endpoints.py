"""Validation for customer-controlled inference endpoints."""

from __future__ import annotations

import re
from ipaddress import ip_address
from urllib.parse import urlsplit

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_LEGACY_NUMERIC_LABEL = re.compile(r"^(?:[0-9]+|0x[0-9a-f]+)$")
_LOCAL_HOST_SUFFIXES = (
    ".home",
    ".home.arpa",
    ".internal",
    ".lan",
    ".local",
    ".localdomain",
    ".localhost",
)


def _canonical_hostname(raw_hostname: str) -> str:
    """Match the IDNA/dot normalization performed by HTTP clients."""

    try:
        hostname = raw_hostname.encode("idna").decode("ascii").casefold().rstrip(".")
    except UnicodeError as exc:
        raise ValueError("endpoint_url contains an invalid hostname") from exc
    if not hostname or len(hostname) > 253:
        raise ValueError("endpoint_url contains an invalid hostname")
    return hostname


def _validate_dns_hostname(hostname: str) -> None:
    labels = hostname.split(".")
    # libc accepts historical one-part, short dotted, octal, and hexadecimal IPv4
    # spellings that ipaddress deliberately does not. They are unsafe ambiguity at an
    # HTTP security boundary, so reject every all-numeric legacy representation.
    if all(_LEGACY_NUMERIC_LABEL.fullmatch(label) for label in labels):
        raise ValueError("endpoint_url cannot target a numeric legacy address")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("endpoint_url contains an invalid public hostname")
    if hostname == "localhost" or hostname.endswith(_LOCAL_HOST_SUFFIXES):
        raise ValueError("endpoint_url cannot target a local address")


def validate_public_https_endpoint(endpoint_url: str) -> str:
    """Return a safe endpoint URL or reject credential and obvious SSRF targets.

    Query strings and fragments are deliberately forbidden: endpoint credentials and
    routing metadata belong in the encrypted adapter configuration, where they are not
    echoed in API responses or persisted as part of the public system profile.
    """

    if not isinstance(endpoint_url, str) or not endpoint_url.strip():
        raise ValueError("endpoint_url must not be empty")
    if endpoint_url != endpoint_url.strip():
        raise ValueError("endpoint_url must not contain surrounding whitespace")

    parsed = urlsplit(endpoint_url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValueError("endpoint_url must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("endpoint credentials must be supplied via adapter configuration")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint_url must not contain a query string or fragment")

    hostname = _canonical_hostname(parsed.hostname)
    try:
        address = ip_address(hostname)
    except ValueError:
        _validate_dns_hostname(hostname)
    else:
        if not address.is_global:
            raise ValueError("endpoint_url cannot target a non-public IP address")

    return endpoint_url

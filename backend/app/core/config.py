"""Environment-backed, fail-safe application settings."""

from __future__ import annotations

import base64
import binascii
import hmac
import unicodedata
from functools import lru_cache
from typing import Literal

from cryptography.fernet import Fernet
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from backend.app.clinical_nlp.versions import (
    HF_NER_MODEL_NAME,
    HF_NER_MODEL_REVISION,
)

DEVELOPMENT_API_KEY_PEPPER = "development-only-change-me"
DEVELOPMENT_DATABASE_PASSWORD = "gatekeeper"
DEVELOPMENT_ENCRYPTION_KEY = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="

_DEVELOPMENT_ENCRYPTION_KEY_MATERIAL = base64.urlsafe_b64decode(DEVELOPMENT_ENCRYPTION_KEY)
_MINIMUM_PRODUCTION_API_KEY_PEPPER_LENGTH = 43
_MINIMUM_PRODUCTION_API_KEY_PEPPER_DIVERSITY = 12
_API_KEY_PEPPER_PLACEHOLDER_MARKERS = (
    "changeme",
    "example",
    "notaproductionsecret",
    "placeholder",
    "replacewith",
)
_PLACEHOLDER_CHARACTER_TRANSLATION = str.maketrans("013457", "oieast")


def _validated_production_api_key_pepper(value: str) -> str:
    normalized = value.strip()
    folded = unicodedata.normalize("NFKC", normalized).casefold()
    placeholder_form = "".join(
        character
        for character in folded.translate(_PLACEHOLDER_CHARACTER_TRANSLATION)
        if character.isalnum()
    )
    if (
        len(normalized) < _MINIMUM_PRODUCTION_API_KEY_PEPPER_LENGTH
        or len(set(folded)) < _MINIMUM_PRODUCTION_API_KEY_PEPPER_DIVERSITY
        or any(not character.isprintable() or character.isspace() for character in normalized)
        or any(marker in placeholder_form for marker in _API_KEY_PEPPER_PLACEHOLDER_MARKERS)
    ):
        raise ValueError(
            "production requires a randomly generated API key pepper from at least 32 bytes"
        )
    return normalized


class Settings(BaseSettings):
    """Runtime settings.

    Secrets are loaded from the environment and are never serialized into API
    responses or normal logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GATEKEEPER_",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "test", "staging", "production"] = "development"
    database_url: str = Field(
        default="postgresql+psycopg://gatekeeper:gatekeeper@localhost:5432/gatekeeper",
        repr=False,
    )
    api_key_pepper: str = Field(default=DEVELOPMENT_API_KEY_PEPPER, min_length=16, repr=False)
    data_encryption_keys: str = Field(
        default=DEVELOPMENT_ENCRYPTION_KEY,
        repr=False,
    )
    # The HTTP envelope can be larger than the decoded evidence because JSON
    # escaping expands line breaks and some Unicode code points.  Per-artifact and
    # aggregate decoded-content limits below remain the authoritative ingestion
    # safeguards.
    max_request_bytes: int = Field(default=8_388_608, ge=1_024)
    max_source_count: int = Field(default=100, ge=1, le=10_000)
    max_source_bytes: int = Field(default=1_048_576, ge=1_024)
    max_generated_output_bytes: int = Field(default=1_048_576, ge=1_024)
    max_total_evidence_bytes: int = Field(default=2_097_152, ge=2_048)
    max_transcript_segments: int = Field(default=10_000, ge=1, le=100_000)
    max_structured_depth: int = Field(default=32, ge=1, le=128)
    max_json_array_items: int = Field(default=100_000, ge=1, le=1_000_000)
    max_canonical_evidence_units: int = Field(default=25_000, ge=1, le=1_000_000)
    max_extracted_claims: int = Field(default=50_000, ge=1, le=1_000_000)
    max_processing_seconds: float = Field(default=120.0, gt=0, le=3_600)
    small_bytes: int = Field(default=10_240, ge=1)
    medium_bytes: int = Field(default=102_400, ge=2)
    large_bytes: int = Field(default=1_048_576, ge=3)
    chunk_characters: int = Field(default=12_000, ge=256, le=1_000_000)
    chunk_overlap_characters: int = Field(default=500, ge=0)
    max_chunks: int = Field(default=5_000, ge=1, le=100_000)
    ner_backend: Literal["rules", "huggingface"] = "rules"
    ner_model: str = HF_NER_MODEL_NAME
    ner_model_revision: str = HF_NER_MODEL_REVISION
    # The LLM planner is an opt-in advisory layer.  It cannot replace the
    # deterministic selection rules or make release decisions.
    llm_planner_enabled: bool = False
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str = ""
    anthropic_api_key: str | None = Field(default=None, repr=False)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120.0)
    semantic_evaluation_enabled: bool = False
    customer_ai_timeout_seconds: float = Field(default=20.0, gt=0)
    customer_ai_max_retries: int = Field(default=2, ge=0, le=10)
    clinical_data_retention_days: int = Field(default=30, ge=0, le=365)

    @model_validator(mode="after")
    def validate_secret_material(self) -> Settings:
        if not self.small_bytes < self.medium_bytes < self.large_bytes:
            raise ValueError("ingestion size thresholds must be strictly increasing")
        if self.chunk_overlap_characters >= self.chunk_characters:
            raise ValueError("ingestion chunk overlap must be smaller than the chunk size")
        if self.max_source_bytes > self.max_total_evidence_bytes:
            raise ValueError("max source bytes cannot exceed max total evidence bytes")
        if self.max_generated_output_bytes > self.max_total_evidence_bytes:
            raise ValueError(
                "max generated-output bytes cannot exceed max total evidence bytes"
            )
        if self.ner_model != HF_NER_MODEL_NAME or self.ner_model_revision != HF_NER_MODEL_REVISION:
            raise ValueError(
                "the optional clinical NER model and revision are pinned by this release"
            )
        if self.semantic_evaluation_enabled:
            raise ValueError(
                "external semantic model runtime is not configured in this release; "
                "the built-in offline cascade remains enabled"
            )
        if self.llm_planner_enabled:
            if self.llm_provider != "anthropic":
                raise ValueError("the LLM planner currently supports only Anthropic")
            if not self.llm_model.strip():
                raise ValueError("an LLM model is required when the LLM planner is enabled")
            if not self.anthropic_api_key or not self.anthropic_api_key.strip():
                raise ValueError(
                    "an Anthropic API key is required when the LLM planner is enabled"
                )

        keys = [item.strip() for item in self.data_encryption_keys.split(",") if item.strip()]
        if not keys:
            raise ValueError("at least one data encryption key is required")
        key_materials: list[bytes] = []
        try:
            for key in keys:
                Fernet(key.encode())
                key_materials.append(base64.urlsafe_b64decode(key.encode()))
        except (binascii.Error, TypeError, ValueError) as exc:
            raise ValueError("data encryption keys must be valid Fernet keys") from exc

        if self.environment == "production":
            self.api_key_pepper = _validated_production_api_key_pepper(self.api_key_pepper)
            if any(
                hmac.compare_digest(key_material, _DEVELOPMENT_ENCRYPTION_KEY_MATERIAL)
                for key_material in key_materials
            ):
                raise ValueError("production cannot use the development encryption key")
            try:
                database = make_url(self.database_url)
            except ArgumentError as exc:
                raise ValueError("production requires a valid PostgreSQL database URL") from exc
            query_password = database.query.get("password")
            if database.get_backend_name() != "postgresql" or (
                database.password == DEVELOPMENT_DATABASE_PASSWORD
                or query_password == DEVELOPMENT_DATABASE_PASSWORD
            ):
                raise ValueError("production requires a non-default PostgreSQL database URL")
            ssl_mode = str(database.query.get("sslmode", "")).casefold()
            if ssl_mode not in {"require", "verify-ca", "verify-full"}:
                raise ValueError(
                    "production PostgreSQL requires sslmode=require, verify-ca, or verify-full"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

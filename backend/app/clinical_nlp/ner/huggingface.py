"""Lazy, fail-safe Hugging Face token-classification adapter.

The adapter is opt-in. By default it only checks the local Hugging Face cache; downloading
weights requires the caller to explicitly set ``allow_download=True``. The model is a
candidate entity source only: deterministic code remains authoritative for assertion,
laterality, dose, units, frequency, and other high-risk scalar values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from backend.app.clinical_nlp.versions import (
    HF_NER_MODEL_NAME,
    HF_NER_MODEL_REVISION,
    HF_NER_MODEL_VERSION,
)


@dataclass(frozen=True, slots=True)
class NEREntity:
    text: str
    start: int
    end: int
    label: str
    score: float


PipelineLoader = Callable[[], Callable[[str], Sequence[Mapping[str, Any]]]]


class HuggingFaceClinicalNERAdapter:
    """Load the revision-pinned MACCROBAT model only on first extraction."""

    model_name = HF_NER_MODEL_NAME
    revision = HF_NER_MODEL_REVISION
    version = HF_NER_MODEL_VERSION
    benchmark_status = "benchmark-gated"

    def __init__(
        self,
        *,
        allow_download: bool = False,
        loader: PipelineLoader | None = None,
    ) -> None:
        self.allow_download = allow_download
        self._loader = loader
        self._pipeline: Callable[[str], Sequence[Mapping[str, Any]]] | None = None
        self._load_attempted = False
        self.last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _default_loader(self) -> Callable[[str], Sequence[Mapping[str, Any]]]:
        # Optional dependencies are deliberately imported inside the opt-in execution path.
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        local_files_only = not self.allow_download
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=local_files_only,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=local_files_only,
        )
        return cast(
            Callable[[str], Sequence[Mapping[str, Any]]],
            pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1,
            ),
        )

    def _ensure_pipeline(self) -> Callable[[str], Sequence[Mapping[str, Any]]] | None:
        if self._pipeline is not None:
            return self._pipeline
        if self._load_attempted:
            return None
        self._load_attempted = True
        try:
            self._pipeline = (self._loader or self._default_loader)()
        # Model absence/dependency failure must reduce coverage, not crash extraction.
        except Exception as exc:
            self.last_error = f"NER_MODEL_LOAD_FAILED:{type(exc).__name__}"
            self._pipeline = None
        return self._pipeline

    def extract(self, text: str) -> list[NEREntity]:
        pipeline_instance = self._ensure_pipeline()
        if pipeline_instance is None:
            return []
        try:
            raw_entities = pipeline_instance(text)
        except Exception as exc:
            self.last_error = f"NER_INFERENCE_FAILED:{type(exc).__name__}"
            return []

        entities: list[NEREntity] = []
        for raw in raw_entities:
            try:
                start = int(raw["start"])
                end = int(raw["end"])
                if not 0 <= start < end <= len(text):
                    continue
                label = str(raw.get("entity_group") or raw.get("entity") or "")
                score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
                entities.append(
                    NEREntity(
                        text=text[start:end],
                        start=start,
                        end=end,
                        label=label,
                        score=score,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return entities

#!/usr/bin/env python3
"""Measure the local universal-ingestion and comparison path on synthetic text."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.compare_engine import compare
from backend.app.core.config import Settings
from backend.app.domain.models import Claim, GeneratedOutput, SourceBundle, SourceDocument
from backend.app.ingestion.extractor import CanonicalEvidenceExtractor
from backend.app.ingestion.limits import IngestionLimits
from backend.app.ingestion.schemas import TextEvidenceSource, TextGeneratedOutput
from backend.app.ingestion.service import UniversalEvidenceIngestionService

_SOURCE_PREFIX = (
    "Patient reports left knee pain. Use insulin glargine 10 units nightly.\n\n"
)
_SOURCE_FILLER = (
    "Synthetic context for deterministic ingestion timing; this line contains no person data.\n"
)
_GENERATED = "Patient reports right knee pain. Use insulin glargine 100 units nightly."


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    profiles: tuple[str, ...]
    repeat: int
    output: Path | None


@dataclass(frozen=True, slots=True)
class Measurement:
    profile: str
    iteration: int
    requested_payload_bytes: int
    payload_bytes: int
    size_class: str
    ingestion_seconds: float
    normalization_seconds: float
    extraction_seconds: float
    alignment_comparison_residual_seconds: float
    compare_seconds: float
    total_evaluation_seconds: float
    claim_count: int
    finding_count: int
    chunk_count: int
    evidence_unit_count: int
    python_tracemalloc_peak_bytes: int
    coverage_status: str
    alignment_comparison_residual_is_estimate: bool = True


class TimedExtractor:
    """Measure only extractor calls made by the real Compare Engine invocation."""

    version = "timed-canonical-evidence-extractor@1.0.0"

    def __init__(self, wrapped: CanonicalEvidenceExtractor) -> None:
        self._wrapped = wrapped
        self.elapsed_seconds = 0.0
        self.claim_count = 0

    @property
    def last_ner_error(self) -> str | None:
        return self._wrapped.last_ner_error

    @property
    def model_versions(self) -> Mapping[str, str | None]:
        return self._wrapped.model_versions

    def extract(self, text: str, document_id: str) -> list[Claim]:
        started = time.perf_counter()
        try:
            claims = self._wrapped.extract(text, document_id)
        finally:
            self.elapsed_seconds += time.perf_counter() - started
        self.claim_count += len(claims)
        return claims


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight, synthetic small/medium/large benchmark through "
            "universal ingestion and the existing Clinical Compare Engine."
        )
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=("small", "medium", "large"),
        dest="profiles",
        help="Profile to run; repeat to select several (default: all three)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Measured runs per profile (default: 1)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser


def _parse_config(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    args = _parser().parse_args(argv)
    repeat = cast(int, args.repeat)
    if not 1 <= repeat <= 20:
        raise ValueError("--repeat must be between 1 and 20")
    selected = cast(list[str] | None, args.profiles)
    profiles = tuple(dict.fromkeys(selected or ["small", "medium", "large"]))
    return BenchmarkConfig(
        profiles=profiles,
        repeat=repeat,
        output=cast(Path | None, args.output),
    )


def _target_payloads(limits: IngestionLimits) -> dict[str, int]:
    minimum = len(_SOURCE_PREFIX.encode()) + len(_GENERATED.encode())
    small = max(minimum, limits.small_bytes // 2)
    # Cross each configured boundary without turning this developer benchmark into
    # a CI-scale load test. Deployments can use --repeat for additional samples.
    medium = limits.small_bytes + min(
        32 * 1_024,
        max(1, (limits.medium_bytes - limits.small_bytes) // 2),
    )
    large = limits.medium_bytes + min(
        32 * 1_024,
        max(1, (limits.large_bytes - limits.medium_bytes) // 2),
    )
    if small > limits.small_bytes or medium > limits.medium_bytes or large > limits.large_bytes:
        raise ValueError("configured size classes are too small for the benchmark fixture")
    return {"small": small, "medium": medium, "large": large}


def _sized_source(target_payload_bytes: int) -> str:
    output_bytes = len(_GENERATED.encode("utf-8"))
    target_source_bytes = target_payload_bytes - output_bytes
    prefix_bytes = len(_SOURCE_PREFIX.encode("utf-8"))
    if target_source_bytes < prefix_bytes:
        raise ValueError("target payload is smaller than the synthetic clinical prefix")
    remaining = target_source_bytes - prefix_bytes
    repetitions, remainder = divmod(remaining, len(_SOURCE_FILLER))
    return _SOURCE_PREFIX + (_SOURCE_FILLER * repetitions) + _SOURCE_FILLER[:remainder]


def _run_once(
    *,
    profile: str,
    iteration: int,
    target_payload_bytes: int,
    limits: IngestionLimits,
) -> Measurement:
    source = TextEvidenceSource(
        document_id=f"benchmark-{profile}",
        type="clinical_text",
        content=_sized_source(target_payload_bytes),
    )
    generated = TextGeneratedOutput(type="clinical_note", content=_GENERATED)
    service = UniversalEvidenceIngestionService(limits=limits)

    tracemalloc.start()
    total_started = time.perf_counter()
    try:
        ingestion = service.ingest([source], generated)
        source_bundle = SourceBundle(
            documents=[
                SourceDocument(
                    document_id=document.document_id,
                    type="clinical_text",
                    content=document.normalized_text,
                )
                for document in ingestion.source_documents
            ]
        )
        normalized_output = GeneratedOutput(
            type="clinical_text",
            content=ingestion.generated_document.normalized_text,
        )
        canonical_extractor = CanonicalEvidenceExtractor(
            ClinicalNLPService(),
            [*ingestion.source_documents, ingestion.generated_document],
            max_claims=limits.max_extracted_claims,
            processing_timeout_seconds=limits.max_processing_seconds,
        )
        timed_extractor = TimedExtractor(canonical_extractor)
        compare_started = time.perf_counter()
        result = compare(
            source_bundle,
            normalized_output,
            nlp_service=timed_extractor,
        )
        compare_seconds = time.perf_counter() - compare_started
        total_seconds = time.perf_counter() - total_started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    extraction_seconds = timed_extractor.elapsed_seconds
    return Measurement(
        profile=profile,
        iteration=iteration,
        requested_payload_bytes=target_payload_bytes,
        payload_bytes=ingestion.stats.payload_bytes,
        size_class=ingestion.stats.size_class,
        ingestion_seconds=ingestion.stats.ingestion_seconds,
        normalization_seconds=ingestion.stats.normalization_seconds,
        extraction_seconds=extraction_seconds,
        alignment_comparison_residual_seconds=max(0.0, compare_seconds - extraction_seconds),
        compare_seconds=compare_seconds,
        total_evaluation_seconds=total_seconds,
        claim_count=len(result.source_claims) + len(result.generated_claims),
        finding_count=len(result.findings),
        chunk_count=ingestion.stats.chunk_count,
        evidence_unit_count=ingestion.stats.evidence_unit_count,
        python_tracemalloc_peak_bytes=peak_bytes,
        coverage_status=result.coverage.status.value,
    )


def run(config: BenchmarkConfig) -> dict[str, object]:
    settings = Settings()
    limits = IngestionLimits.from_settings(settings)
    targets = _target_payloads(limits)
    measurements = [
        _run_once(
            profile=profile,
            iteration=iteration,
            target_payload_bytes=targets[profile],
            limits=limits,
        )
        for profile in config.profiles
        for iteration in range(1, config.repeat + 1)
    ]
    return {
        "schema_version": "ingestion-performance@1.0",
        "fixture": "SYNTHETIC_NO_PHI",
        "profiles": list(config.profiles),
        "repeat": config.repeat,
        "timing_note": (
            "alignment_comparison_residual_seconds is compare wall time minus measured "
            "extractor call time; it is an estimate covering alignment, deterministic rules, "
            "semantic fallback, finding construction, and timing overhead. Timings are "
            "observed while Python allocation tracing is active."
        ),
        "memory_note": (
            "python_tracemalloc_peak_bytes is traced Python allocation peak, not process RSS."
        ),
        "limits": asdict(limits),
        "measurements": [asdict(measurement) for measurement in measurements],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_config(argv)
        report = run(config)
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if config.output is not None:
            config.output.parent.mkdir(parents=True, exist_ok=True)
            config.output.write_text(serialized, encoding="utf-8")
            print(f"Wrote ingestion benchmark to {config.output}")
        else:
            print(serialized, end="")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ingestion benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

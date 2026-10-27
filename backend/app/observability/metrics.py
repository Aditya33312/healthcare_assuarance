"""Exporter-neutral metrics registry for Phase 1 instrumentation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock


class MetricName(StrEnum):
    REQUESTS = "requests_total"
    REQUEST_LATENCY = "request_latency_seconds"
    EVALUATIONS = "evaluations_total"
    EVALUATION_LATENCY = "evaluation_latency_seconds"
    INGESTION_PAYLOAD_BYTES = "ingestion_payload_bytes"
    INGESTION_SOURCE_COUNT = "ingestion_source_count"
    INGESTION_EVIDENCE_UNITS = "ingestion_evidence_units"
    INGESTION_CHUNKS = "ingestion_chunks"
    INGESTION_NORMALIZATION_LATENCY = "ingestion_normalization_latency_seconds"
    INGESTION_LATENCY = "ingestion_latency_seconds"
    TEST_RUN_DURATION = "test_run_duration_seconds"
    CUSTOMER_AI_FAILURES = "customer_ai_endpoint_failures_total"
    NER_FAILURES = "ner_failures_total"
    SEMANTIC_FAILURES = "semantic_evaluator_failures_total"
    UNASSESSED_CASES = "unassessed_cases_total"
    WORKER_FAILURES = "worker_failures_total"
    QUEUE_DEPTH = "queue_depth"
    MODEL_USAGE = "model_usage_total"
    TEST_COST = "test_cost"


_ALLOWED_LABELS = {
    "adapter",
    "environment",
    "error_code",
    "execution_status",
    "method",
    "outcome",
    "route_template",
    "test_family",
}


@dataclass(frozen=True, slots=True)
class Observation:
    count: int
    total: float
    maximum: float


class OperationalMetrics:
    """Thread-safe in-memory metrics behind an exporter-neutral API.

    Only allowlisted, low-cardinality labels are accepted. Identifiers, clinical
    concepts, content, and evidence therefore cannot become metric labels.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(
            float
        )
        self._observations: defaultdict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = (
            defaultdict(list)
        )
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def increment(
        self,
        name: MetricName,
        amount: float = 1,
        **labels: str,
    ) -> None:
        key = (name.value, self._labels(labels))
        with self._lock:
            self._counters[key] += amount

    def observe(self, name: MetricName, value: float, **labels: str) -> None:
        if value < 0:
            raise ValueError("metric observations cannot be negative")
        key = (name.value, self._labels(labels))
        with self._lock:
            self._observations[key].append(value)

    def gauge(self, name: MetricName, value: float, **labels: str) -> None:
        key = (name.value, self._labels(labels))
        with self._lock:
            self._gauges[key] = value

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = {self._display_key(key): value for key, value in self._counters.items()}
            observations = {
                self._display_key(key): Observation(
                    count=len(values),
                    total=sum(values),
                    maximum=max(values),
                )
                for key, values in self._observations.items()
                if values
            }
            gauges = {self._display_key(key): value for key, value in self._gauges.items()}
        return {
            "counters": counters,
            "observations": observations,
            "gauges": gauges,
        }

    @staticmethod
    def _labels(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
        unknown = set(labels) - _ALLOWED_LABELS
        if unknown:
            raise ValueError(f"metric labels are not allowed: {', '.join(sorted(unknown))}")
        if any(len(value) > 100 for value in labels.values()):
            raise ValueError("metric label values must be low-cardinality operational values")
        return tuple(sorted(labels.items()))

    @staticmethod
    def _display_key(key: tuple[str, tuple[tuple[str, str], ...]]) -> str:
        name, labels = key
        if not labels:
            return name
        rendered = ",".join(f"{label}={value}" for label, value in labels)
        return f"{name}{{{rendered}}}"

"""Regression coverage for universal, provenance-preserving evidence ingestion."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select

import backend.app.ingestion.service as ingestion_service_module
from backend.app.api.schemas import EVALUATION_OPENAPI_EXAMPLES, SingleEvaluationCreate
from backend.app.clinical_nlp import ClinicalNLPService
from backend.app.compare_engine import compare
from backend.app.core.errors import ErrorCode, GatekeeperError
from backend.app.database.models import GeneratedOutputRecord, SourceDocumentRecord
from backend.app.domain.enums import AssertionStatus, ClaimType, FindingCategory, FindingType
from backend.app.domain.models import Claim, TextSpan
from backend.app.ingestion import (
    CanonicalDocument,
    CanonicalEvidenceExtractor,
    CanonicalEvidenceUnit,
    ClinicalProcessingLimitError,
    EvidenceSource,
    GeneratedEvidenceOutput,
    IngestionLimits,
    IngestionResult,
    LabResultItem,
    LabResultsSource,
    MedicationItem,
    MedicationListSource,
    RagChunk,
    RagContextSource,
    StructuredJsonGeneratedOutput,
    StructuredJsonSource,
    TextEvidenceSource,
    TextGeneratedOutput,
    TranscriptSegment,
    TranscriptSource,
    UniversalEvidenceIngestionService,
)
from backend.tests.test_fastapi_api import (
    _ACTIVE_HARNESSES,
    ApiHarness,
    _build_harness,
    _create_system,
    _create_version,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_transcript_88.json"
EVIDENCE_ADAPTER: TypeAdapter[EvidenceSource] = TypeAdapter(EvidenceSource)
OUTPUT_ADAPTER: TypeAdapter[GeneratedEvidenceOutput] = TypeAdapter(GeneratedEvidenceOutput)


class _NoopExtractor:
    version = "noop-clinical-extractor@test"
    last_ner_error: str | None = None

    @property
    def model_versions(self) -> Mapping[str, str | None]:
        return {"ner_model": self.version}

    def extract(self, text: str, document_id: str) -> list[Claim]:
        del text, document_id
        return []


class _KeywordExtractor(_NoopExtractor):
    version = "keyword-clinical-extractor@test"

    def __init__(self, keyword: str = "headache") -> None:
        self.keyword = keyword

    def extract(self, text: str, document_id: str) -> list[Claim]:
        claims: list[Claim] = []
        search_at = 0
        while (start := text.casefold().find(self.keyword.casefold(), search_at)) >= 0:
            end = start + len(self.keyword)
            claims.append(
                Claim(
                    claim_id=f"local_{len(claims)}",
                    concept=self.keyword.casefold(),
                    type=ClaimType.SYMPTOM,
                    source_span=TextSpan(
                        document_id=document_id,
                        text=text[start:end],
                        start=start,
                        end=end,
                    ),
                    extractor=self.version,
                    extractor_version="test",
                )
            )
            search_at = end
        return claims


def _output(content: str = "Synthetic clinical output.") -> TextGeneratedOutput:
    return TextGeneratedOutput(type="clinical_note", content=content)


def _ingest(
    source: EvidenceSource,
    *,
    output: GeneratedEvidenceOutput | None = None,
    limits: IngestionLimits | None = None,
) -> IngestionResult:
    return UniversalEvidenceIngestionService(limits).ingest(
        [source],
        output or _output(),
    )


def _load_long_case() -> dict[str, Any]:
    loaded = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("synthetic transcript fixture must contain a JSON object")
    return cast(dict[str, Any], loaded)


def _assert_complete_chunk_coverage(document: CanonicalDocument) -> None:
    assert document.chunks
    assert document.chunks[0].start_offset == 0
    covered_until = 0
    for chunk in document.chunks:
        assert chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]
        assert chunk.start_offset <= covered_until
        covered_until = max(covered_until, chunk.end_offset)
    assert covered_until == len(document.normalized_text)


def _long_soap_note() -> str:
    sections = [
        "SUBJECTIVE:\nSynthetic patient reports headache, photophobia, and nausea.",
        "OBJECTIVE:\nHome blood pressure was 150/95 mmHg; the patient remains alert.",
        "ASSESSMENT:\nHeadache with migraine features; no final diagnosis is asserted.",
        "PLAN:\nContinue lisinopril 10 mg once daily and review warning symptoms.",
    ]
    review_lines = [
        f"Review item {index:02d}: synthetic context was retained for deterministic testing."
        for index in range(80)
    ]
    return "\n\n".join([*sections, "REVIEW:\n" + "\n".join(review_lines)])


@pytest.fixture(scope="module")
def ingestion_api() -> Iterator[tuple[ApiHarness, str]]:
    harness = _build_harness(max_request_bytes=2_000_000)
    system = _create_system(harness)
    _create_version(harness, str(system["id"]))
    try:
        yield harness, str(system["id"])
    finally:
        harness.close()
        if harness in _ACTIVE_HARNESSES:
            _ACTIVE_HARNESSES.remove(harness)


def test_public_schema_keeps_legacy_text_defaults_backward_compatible() -> None:
    request = SingleEvaluationCreate.model_validate(
        {
            "system_id": "sys_synthetic",
            "system_version": "1.0.0",
            "sources": [{"document_id": "doc-1", "content": "x"}],
            "generated_output": {"content": "y"},
        }
    )

    assert isinstance(request.sources[0], TextEvidenceSource)
    assert request.sources[0].type == "clinical_text"
    assert isinstance(request.generated_output, TextGeneratedOutput)
    assert request.generated_output.type == "clinical_text"


@pytest.mark.parametrize(
    "raw",
    [
        "x",
        "One synthetic sentence.",
        "Line one.\nLine two.\nLine three.",
        "Line one.\r\nLine two.\r\nLine three.",
        "Column one\tColumn two\t91%",
        "The clinician said, \"don't flatten this text.\"",
        "Crème brûlée; 血圧 120/80 mmHg; नमस्ते.",
        "Synthetic status: stable 🩺 ✅",
        "\n".join(f"Synthetic line {index}: value {index}." for index in range(300)),
        "\n".join(f"Synthetic row {index}." for index in range(1_000)),
        "A" * 14_000 + ".",
    ],
    ids=(
        "one-character",
        "sentence",
        "lf",
        "crlf",
        "tabs",
        "quotes-apostrophe",
        "unicode",
        "emoji",
        "hundreds-of-lines",
        "thousands-of-lines",
        "long-single-sentence",
    ),
)
def test_text_formats_preserve_raw_content_and_only_normalize_line_endings(raw: str) -> None:
    result = _ingest(TextEvidenceSource(document_id="text-format", content=raw))
    document = result.source_documents[0]

    assert document.raw_content == raw
    assert document.raw_serialized == raw
    assert document.normalized_text == raw.replace("\r\n", "\n").replace("\r", "\n")
    assert all(
        unit.raw_text == raw[unit.raw_start_offset : unit.raw_end_offset]
        for unit in document.evidence_units
    )
    assert all(
        unit.normalized_text == document.normalized_text[unit.start_offset : unit.end_offset]
        for unit in document.evidence_units
    )


@pytest.mark.parametrize(
    ("byte_count", "expected_size_class"),
    [
        (1_024, "SMALL"),
        (10_240, "SMALL"),
        (51_200, "MEDIUM"),
        (102_400, "MEDIUM"),
        (1_048_576, "LARGE"),
    ],
    ids=("1-kb", "10-kb", "50-kb", "100-kb", "configured-large-threshold"),
)
def test_configured_size_classes_accept_representative_documents(
    byte_count: int,
    expected_size_class: str,
) -> None:
    result = _ingest(TextEvidenceSource(document_id="sized", content="x" * byte_count))

    assert result.source_documents[0].size_class == expected_size_class
    assert len(result.source_documents[0].normalized_text.encode("utf-8")) == byte_count


def test_exact_source_size_limit_is_accepted_and_one_byte_excess_is_rejected() -> None:
    limits = IngestionLimits(
        max_source_bytes=2_048,
        max_generated_output_bytes=128,
        max_total_evidence_bytes=2_176,
        small_bytes=256,
        medium_bytes=512,
        large_bytes=2_048,
        chunk_characters=256,
        chunk_overlap_characters=16,
        max_chunks=100,
    )
    service = UniversalEvidenceIngestionService(limits)

    accepted = service.ingest(
        [TextEvidenceSource(document_id="boundary", content="x" * 2_048)],
        _output("y"),
    )
    assert accepted.source_documents[0].size_class == "LARGE"
    assert len(accepted.source_documents[0].normalized_text) == 2_048

    with pytest.raises(GatekeeperError) as error:
        service.ingest(
            [TextEvidenceSource(document_id="excess", content="x" * 2_049)],
            _output("y"),
        )
    assert error.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert error.value.http_status == 413


@pytest.mark.parametrize("segment_count", [1, 100, 1_000], ids=("one", "hundred", "thousand"))
def test_transcript_segment_counts_retain_every_segment_and_provenance(
    segment_count: int,
) -> None:
    segments = [
        TranscriptSegment(
            segment=index,
            timestamp=f"{index // 60:02d}:{index % 60:02d}" if index % 7 else None,
            speaker="PATIENT" if index % 2 else "CLINICIAN",
            text=f"Synthetic segment {index} records headache observation {index}.",
        )
        for index in range(1, segment_count + 1)
    ]
    result = _ingest(
        TranscriptSource(
            document_id=f"transcript-{segment_count}",
            type="transcript",
            content=segments,
        )
    )
    document = result.source_documents[0]

    assert len(document.evidence_units) == segment_count
    assert [unit.normalized_text for unit in document.evidence_units] == [
        segment.text for segment in segments
    ]
    final_text = segments[-1].text
    start = document.normalized_text.rindex(final_text)
    provenance = document.provenance_for_span(start, start + len(final_text))
    assert provenance is not None
    assert provenance.document_id == document.document_id
    assert provenance.segment_id == str(segment_count)
    assert provenance.timestamp == segments[-1].timestamp
    assert provenance.speaker == segments[-1].speaker.value
    assert provenance.source_path == f"$[{segment_count - 1}].text"


def test_realistic_88_segment_fixture_is_synthetic_multiline_and_complete() -> None:
    case = _load_long_case()
    source = EVIDENCE_ADAPTER.validate_python(case["source"])
    output = OUTPUT_ADAPTER.validate_python(case["generated_output"])
    assert isinstance(source, TranscriptSource)
    assert len(source.content) == 88
    assert {segment.speaker.value for segment in source.content} == {
        "PATIENT",
        "CLINICIAN",
        "OTHER",
        "UNKNOWN",
    }
    assert any("\n" in segment.text for segment in source.content)
    assert any(segment.timestamp is None for segment in source.content)

    result = _ingest(source, output=output)
    document = result.source_documents[0]
    assert len(document.evidence_units) == 88
    assert "photophobia" in document.normalized_text
    assert "neck stiffness" in document.normalized_text
    assert "150/95 mmHg" in document.normalized_text
    assert document.normalized_text.endswith(source.content[-1].text)


def test_transcript_limit_rejects_excess_without_partial_processing() -> None:
    limits = replace(IngestionLimits(), max_transcript_segments=2)
    source = TranscriptSource(
        document_id="bounded-transcript",
        type="transcript",
        content=[
            TranscriptSegment(segment=index, speaker="UNKNOWN", text=f"Synthetic segment {index}.")
            for index in range(3)
        ],
    )

    with pytest.raises(GatekeeperError) as error:
        _ingest(source, limits=limits)
    assert error.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert "Synthetic segment" not in error.value.message


def test_laboratory_values_are_decimal_safe_and_units_are_normalized() -> None:
    source = LabResultsSource(
        document_id="labs",
        type="lab_results",
        content=[
            LabResultItem(
                test="creatinine",
                value=Decimal("1.800"),
                unit="MG/DL",
                reference_range="0.6-1.3",
                flag="HIGH",
                timestamp="2026-01-01T09:30:00Z",
            ),
            LabResultItem(test="potassium", value=Decimal("4.60"), unit="mmol/l"),
            LabResultItem(test="temperature", value=Decimal("-2.5"), unit="°C"),
        ],
    )

    result = _ingest(source)
    units = result.source_documents[0].evidence_units
    assert units[0].metadata["value"] == Decimal("1.800")
    assert units[0].metadata["unit"] == "mg/dL"
    assert units[0].metadata["reference_range"] == "0.6-1.3"
    assert units[0].metadata["flag"] == "HIGH"
    assert units[1].metadata["value"] == Decimal("4.60")
    assert units[1].metadata["unit"] == "mmol/L"
    assert units[2].metadata["value"] == Decimal("-2.5")
    assert units[2].metadata["unit"] == "°C"
    assert result.limitations == ()


def test_medications_preserve_raw_values_and_normalize_dose_frequency_and_route() -> None:
    source = MedicationListSource(
        document_id="medications",
        type="medication_list",
        content=[
            MedicationItem(
                name="Metformin",
                dose=Decimal("0.5"),
                unit="g",
                frequency="BID",
                route="oral",
                status="ACTIVE",
            ),
            MedicationItem(
                name="Acetaminophen",
                dose=1_000,
                unit="mcg",
                frequency="2x/day",
                route="PO",
            ),
            MedicationItem(
                name="Ceftriaxone",
                dose=500,
                unit="mg",
                frequency="once daily",
                route="intravenous",
                instructions="Synthetic instruction only.",
            ),
        ],
    )

    result = _ingest(source)
    first, second, third = result.source_documents[0].evidence_units
    assert first.metadata["raw_dose"] == "0.5"
    assert first.metadata["raw_unit"] == "g"
    assert first.metadata["dose"] == 500
    assert first.metadata["unit"] == "mg"
    assert first.metadata["frequency"] == "TWICE_DAILY"
    assert first.metadata["route"] == "ORAL"
    assert second.metadata["dose"] == 1
    assert second.metadata["unit"] == "mg"
    assert second.metadata["frequency"] == "TWICE_DAILY"
    assert second.metadata["route"] == "ORAL"
    assert third.metadata["frequency"] == "ONCE_DAILY"
    assert third.metadata["route"] == "INTRAVENOUS"
    assert result.limitations == ()


def test_duplicate_structured_items_are_preserved_with_distinct_unit_ids() -> None:
    item = MedicationItem(name="Metformin", dose=500, unit="mg", frequency="BD")
    source = MedicationListSource(
        document_id="duplicate-items",
        type="medication_list",
        content=[item, item.model_copy()],
    )

    document = _ingest(source).source_documents[0]
    assert len(document.evidence_units) == 2
    assert document.evidence_units[0].raw_text == document.evidence_units[1].raw_text
    assert (
        document.evidence_units[0].evidence_unit_id
        != document.evidence_units[1].evidence_unit_id
    )


def test_unsupported_units_and_attributes_create_safe_reduced_coverage_limitations() -> None:
    sources: list[EvidenceSource] = [
        LabResultsSource(
            document_id="unsupported-lab",
            type="lab_results",
            content=[LabResultItem(test="synthetic marker", value="trace", unit="arbitrary-u")],
        ),
        MedicationListSource(
            document_id="unsupported-medication",
            type="medication_list",
            content=[
                MedicationItem(
                    name="Synthetic medicine",
                    dose=2,
                    unit="arbitrary-u",
                    frequency="on alternating Tuesdays",
                    route="experimental route",
                )
            ],
        ),
    ]

    result = UniversalEvidenceIngestionService().ingest(sources, _output())
    codes = [limitation.code for limitation in result.limitations]
    assert codes.count("UNSUPPORTED_UNIT") == 2
    assert "UNSUPPORTED_FREQUENCY" in codes
    assert "UNSUPPORTED_ROUTE" in codes
    assert all("arbitrary-u" not in limitation.message for limitation in result.limitations)


def test_structured_claim_extraction_uses_normalized_attributes_and_provenance() -> None:
    sources: list[EvidenceSource] = [
        LabResultsSource(
            document_id="extract-lab",
            type="lab_results",
            content=[LabResultItem(test="potassium", value=Decimal("4.6"), unit="mmol/L")],
        ),
        MedicationListSource(
            document_id="extract-medication",
            type="medication_list",
            content=[
                MedicationItem(
                    name="Metformin",
                    dose=Decimal("0.5"),
                    unit="g",
                    frequency="BD",
                    route="PO",
                )
            ],
        ),
    ]
    ingestion = UniversalEvidenceIngestionService().ingest(sources, _output())
    extractor = CanonicalEvidenceExtractor(
        _NoopExtractor(),
        ingestion.source_documents,
        max_claims=100,
        processing_timeout_seconds=10,
    )

    claims = [
        claim
        for document in ingestion.source_documents
        for claim in extractor.extract(document.normalized_text, document.document_id)
    ]
    by_type = {claim.type: claim for claim in claims}
    assert by_type[ClaimType.LAB_TEST].attributes["normalized_value"] == "4.6"
    assert by_type[ClaimType.LAB_TEST].attributes["normalized_unit"] == "mmol/L"
    assert by_type[ClaimType.MEDICATION].attributes["dose"] == 500
    assert by_type[ClaimType.MEDICATION].attributes["unit"] == "mg"
    assert by_type[ClaimType.MEDICATION].attributes["frequency"] == "TWICE_DAILY"
    assert by_type[ClaimType.MEDICATION].attributes["route"] == "ORAL"
    assert all("evidence_provenance" in claim.attributes for claim in claims)
    assert {
        cast(dict[str, Any], claim.attributes["evidence_provenance"])["document_id"]
        for claim in claims
    } == {"extract-lab", "extract-medication"}


def test_mixed_source_types_remain_separate_and_ordered() -> None:
    sources: list[EvidenceSource] = [
        TextEvidenceSource(document_id="text", content="Synthetic clinical note."),
        TranscriptSource(
            document_id="transcript",
            type="transcript",
            content=[TranscriptSegment(segment="s-1", speaker="PATIENT", text="Headache.")],
        ),
        LabResultsSource(
            document_id="labs",
            type="lab_results",
            content=[LabResultItem(test="potassium", value="4.6", unit="mmol/L")],
        ),
        MedicationListSource(
            document_id="meds",
            type="medication_list",
            content=[MedicationItem(name="Metformin", dose=500, unit="mg")],
        ),
        StructuredJsonSource(
            document_id="ehr",
            type="structured_json",
            content={"encounter": {"status": "complete", "sequence": 3}},
        ),
        RagContextSource(
            document_id="rag",
            type="rag_context",
            content=[
                RagChunk(
                    chunk_id="rag-1",
                    text="Synthetic retrieved context.",
                    source_uri="ehr://synthetic/context",
                    source_document_id="upstream-1",
                    retrieval_score=Decimal("0.87"),
                )
            ],
        ),
    ]
    output = StructuredJsonGeneratedOutput(
        type="structured_json",
        content={"summary": "Synthetic clinical note", "measurements": [91, "98%"]},
    )

    result = UniversalEvidenceIngestionService().ingest(sources, output)
    assert [document.document_id for document in result.source_documents] == [
        "text",
        "transcript",
        "labs",
        "meds",
        "ehr",
        "rag",
    ]
    assert result.stats.source_types == (
        "clinical_text",
        "transcript",
        "lab_results",
        "medication_list",
        "structured_json",
        "rag_context",
    )
    assert [document.type for document in result.source_bundle.documents] == [
        "clinical_text",
        "clinical_transcript",
        "structured_data",
        "structured_data",
        "structured_data",
        "document",
    ]
    assert result.generated_output.type == "structured_data"
    unit_document_ids = {
        unit.document_id
        for document in result.source_documents
        for unit in document.evidence_units
    }
    assert len(unit_document_ids) == 6


def test_structured_json_is_deterministic_and_retains_nested_paths_and_arrays() -> None:
    left = StructuredJsonSource(
        document_id="structured",
        type="structured_json",
        content={
            "patient": {"active": True, "age": 42},
            "labs": [
                {"value": Decimal("4.60"), "test": "potassium"},
                {"test": "oxygen", "value": "91%"},
            ],
            "mixed": [None, -2, "text"],
        },
    )
    right = StructuredJsonSource(
        document_id="structured",
        type="structured_json",
        content={
            "mixed": [None, -2, "text"],
            "labs": [
                {"test": "potassium", "value": Decimal("4.60")},
                {"value": "91%", "test": "oxygen"},
            ],
            "patient": {"age": 42, "active": True},
        },
    )

    first = _ingest(left).source_documents[0]
    second = _ingest(right).source_documents[0]
    assert first.raw_serialized == second.raw_serialized
    assert first.content_hash == second.content_hash
    assert first.normalized_text == second.normalized_text
    paths = {unit.source_path for unit in first.evidence_units}
    expected_paths = {
        "$.patient",
        "$.labs[0]",
        "$.labs[1]",
        "$.mixed[0]",
        "$.mixed[1]",
        "$.mixed[2]",
    }
    assert expected_paths <= paths
    assert "$.labs[0]: test=potassium; value=4.60" in first.normalized_text


def test_structured_json_depth_and_array_limits_accept_boundary_and_reject_excess() -> None:
    depth_limits = replace(IngestionLimits(), max_structured_depth=3)
    accepted_depth = StructuredJsonSource(
        document_id="depth-ok",
        type="structured_json",
        content={"level_2": {"level_3": {"value": 1}}},
    )
    excessive_depth = StructuredJsonSource(
        document_id="depth-excess",
        type="structured_json",
        content={"level_2": {"level_3": {"level_4": {"value": 1}}}},
    )
    assert len(_ingest(accepted_depth, limits=depth_limits).source_documents[0].evidence_units) == 1
    with pytest.raises(GatekeeperError) as depth_error:
        _ingest(excessive_depth, limits=depth_limits)
    assert depth_error.value.code is ErrorCode.PAYLOAD_TOO_LARGE

    array_limits = replace(IngestionLimits(), max_json_array_items=3)
    accepted_array = StructuredJsonSource(
        document_id="array-ok", type="structured_json", content={"items": [1, 2, 3]}
    )
    excessive_array = StructuredJsonSource(
        document_id="array-excess", type="structured_json", content={"items": [1, 2, 3, 4]}
    )
    assert len(_ingest(accepted_array, limits=array_limits).source_documents[0].evidence_units) == 3
    with pytest.raises(GatekeeperError) as array_error:
        _ingest(excessive_array, limits=array_limits)
    assert array_error.value.code is ErrorCode.PAYLOAD_TOO_LARGE


def test_rag_context_retains_retrieval_metadata_and_span_provenance() -> None:
    source = RagContextSource(
        document_id="rag-document",
        type="rag_context",
        content=[
            RagChunk(
                chunk_id="chunk-123",
                text="Insulin 10 units nightly is present in synthetic evidence.",
                source_uri="not-a-url-but-a-valid-reference",
                source_document_id="upstream-document",
                retrieval_score=Decimal("0.8700"),
                metadata={"rank": 1, "tags": ["synthetic", "medication"]},
            )
        ],
    )

    document = _ingest(source).source_documents[0]
    unit = document.evidence_units[0]
    assert unit.segment_id == "chunk-123"
    assert unit.source_uri == "not-a-url-but-a-valid-reference"
    assert unit.metadata["source_document_id"] == "upstream-document"
    assert unit.metadata["retrieval_score"] == Decimal("0.8700")
    assert unit.metadata["retrieval_metadata"] == {
        "rank": 1,
        "tags": ["synthetic", "medication"],
    }
    start = document.normalized_text.index("10 units")
    provenance = document.provenance_for_span(start, start + len("10 units"))
    assert provenance is not None
    assert provenance.segment_id == "chunk-123"
    assert provenance.source_uri == "not-a-url-but-a-valid-reference"


def test_chunking_is_deterministic_offset_exact_and_never_truncates() -> None:
    text = " ".join(
        f"Sentence {index:03d} retains negation and value {index} mg."
        for index in range(120)
    )
    limits = replace(
        IngestionLimits(),
        chunk_characters=180,
        chunk_overlap_characters=24,
        max_chunks=100,
    )
    source = TextEvidenceSource(document_id="chunked", content=text)

    first = _ingest(source, limits=limits).source_documents[0]
    second = _ingest(source, limits=limits).source_documents[0]
    assert first.chunks == second.chunks
    assert first.evidence_units == second.evidence_units
    assert first.normalized_text == text
    assert len(first.chunks) > 1
    assert all(len(chunk.text) <= limits.chunk_characters for chunk in first.chunks)
    assert any(
        current.start_offset < previous.end_offset
        for previous, current in zip(first.chunks, first.chunks[1:], strict=False)
    )
    _assert_complete_chunk_coverage(first)


def test_canonical_extractor_remaps_chunk_offsets_and_transcript_provenance() -> None:
    limits = replace(
        IngestionLimits(),
        chunk_characters=20,
        chunk_overlap_characters=0,
        max_chunks=10,
    )
    source = TranscriptSource(
        document_id="extract-transcript",
        type="transcript",
        content=[
            TranscriptSegment(
                segment="opening",
                timestamp="00:01",
                speaker="CLINICIAN",
                text="No concerns today.",
            ),
            TranscriptSegment(
                segment="symptom",
                timestamp="00:02",
                speaker="PATIENT",
                text="Headache persists.",
            ),
        ],
    )
    document = _ingest(source, limits=limits).source_documents[0]
    extractor = CanonicalEvidenceExtractor(
        _KeywordExtractor(),
        [document],
        max_claims=10,
        processing_timeout_seconds=10,
    )

    claims = extractor.extract(document.normalized_text, document.document_id)
    assert len(claims) == 1
    claim = claims[0]
    expected_start = document.normalized_text.index("Headache")
    assert claim.source_span.start == expected_start
    assert claim.source_span.end == expected_start + len("headache")
    assert claim.source_span.text == "Headache"
    provenance = cast(dict[str, Any], claim.attributes["evidence_provenance"])
    assert provenance["segment_id"] == "symptom"
    assert provenance["timestamp"] == "00:02"
    assert provenance["speaker"] == "PATIENT"
    assert provenance["chunk_id"] == document.chunks[1].chunk_id

    repeated = CanonicalEvidenceExtractor(
        _KeywordExtractor(),
        [document],
        max_claims=10,
        processing_timeout_seconds=10,
    ).extract(document.normalized_text, document.document_id)
    assert repeated[0].claim_id == claim.claim_id


def test_canonical_extractor_rejects_mismatched_text_and_claim_limit_excess() -> None:
    document = _ingest(
        TextEvidenceSource(document_id="bounded-extraction", content="Headache then headache.")
    ).source_documents[0]

    mismatch = CanonicalEvidenceExtractor(
        _KeywordExtractor(),
        [document],
        max_claims=10,
        processing_timeout_seconds=10,
    )
    with pytest.raises(ValueError, match="does not match"):
        mismatch.extract("Different text.", document.document_id)

    bounded = CanonicalEvidenceExtractor(
        _KeywordExtractor(),
        [document],
        max_claims=1,
        processing_timeout_seconds=10,
    )
    with pytest.raises(ClinicalProcessingLimitError, match="claim-count"):
        bounded.extract(document.normalized_text, document.document_id)


@pytest.mark.parametrize(
    "invalid_source",
    [
        {"document_id": "blank", "type": "clinical_text", "content": ""},
        {"document_id": "whitespace", "type": "clinical_text", "content": " \t\n "},
        {"document_id": "control", "type": "clinical_text", "content": "safe\x00unsafe"},
        {"document_id": "surrogate", "type": "clinical_text", "content": "safe\ud800unsafe"},
        {"document_id": "empty-transcript", "type": "transcript", "content": []},
        {"document_id": "empty-labs", "type": "lab_results", "content": []},
        {"document_id": "empty-meds", "type": "medication_list", "content": []},
        {"document_id": "empty-rag", "type": "rag_context", "content": []},
        {
            "document_id": "bad-number",
            "type": "lab_results",
            "content": [{"test": "x", "value": True}],
        },
        {"document_id": "bad-root", "type": "structured_json", "content": [1, 2]},
        {"document_id": "unknown", "type": "FHIR", "content": {}},
        {"type": "clinical_text", "content": "missing document identifier"},
        {"document_id": "extra", "type": "clinical_text", "content": "x", "extra": "no"},
    ],
    ids=(
        "empty-text",
        "whitespace-text",
        "control-character",
        "unicode-surrogate",
        "empty-transcript",
        "empty-labs",
        "empty-medications",
        "empty-rag",
        "boolean-clinical-number",
        "structured-root-array",
        "unknown-source-type",
        "missing-document-id",
        "extra-field",
    ),
)
def test_public_schema_rejects_empty_malformed_or_unknown_sources(
    invalid_source: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        SingleEvaluationCreate.model_validate(
            {
                "system_id": "sys_synthetic",
                "system_version": "1.0.0",
                "sources": [invalid_source],
                "generated_output": {"content": "valid synthetic output"},
            }
        )


@pytest.mark.parametrize(
    "invalid_output",
    [
        {"type": "clinical_note", "content": ""},
        {"type": "clinical_text", "content": " \n\t "},
        {"type": "structured_json", "content": []},
        {"type": "unsupported", "content": "x"},
        None,
    ],
    ids=("empty", "whitespace", "structured-root-array", "unknown-type", "null"),
)
def test_public_schema_rejects_invalid_generated_output(invalid_output: Any) -> None:
    with pytest.raises(ValidationError):
        SingleEvaluationCreate.model_validate(
            {
                "system_id": "sys_synthetic",
                "system_version": "1.0.0",
                "sources": [{"document_id": "valid", "content": "x"}],
                "generated_output": invalid_output,
            }
        )


def test_empty_structured_object_and_duplicate_document_ids_fail_closed() -> None:
    service = UniversalEvidenceIngestionService()
    empty = StructuredJsonSource(document_id="empty-object", type="structured_json", content={})
    with pytest.raises(GatekeeperError) as empty_error:
        service.ingest([empty], _output())
    assert empty_error.value.code is ErrorCode.INGESTION_FAILED
    assert empty_error.value.http_status == 422

    duplicate_sources = [
        TextEvidenceSource(document_id="duplicate-id", content="Sensitive marker one."),
        TextEvidenceSource(document_id="duplicate-id", content="Sensitive marker two."),
    ]
    with pytest.raises(GatekeeperError) as duplicate_error:
        service.ingest(duplicate_sources, _output())
    assert duplicate_error.value.code is ErrorCode.INGESTION_FAILED
    assert duplicate_error.value.http_status == 422
    assert "Sensitive marker" not in duplicate_error.value.message


def test_source_document_id_cannot_collide_with_generated_output_namespace() -> None:
    source = TextEvidenceSource(
        document_id="generated_output",
        content="Synthetic source content must not enter the generated namespace.",
    )

    with pytest.raises(GatekeeperError) as error:
        UniversalEvidenceIngestionService().ingest([source], _output())
    assert error.value.code is ErrorCode.INGESTION_FAILED
    assert error.value.http_status == 422
    assert "Synthetic source content" not in error.value.message


def test_canonical_unit_limit_stops_single_and_cumulative_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_draft = ingestion_service_module._UnitDraft
    real_unit = CanonicalEvidenceUnit
    counts = {"drafts": 0, "units": 0}

    def counted_draft(*args: Any, **kwargs: Any) -> Any:
        counts["drafts"] += 1
        return real_draft(*args, **kwargs)

    def counted_unit(*args: Any, **kwargs: Any) -> CanonicalEvidenceUnit:
        counts["units"] += 1
        return real_unit(*args, **kwargs)

    monkeypatch.setattr(ingestion_service_module, "_UnitDraft", counted_draft)
    monkeypatch.setattr(ingestion_service_module, "CanonicalEvidenceUnit", counted_unit)
    limits = replace(IngestionLimits(), max_canonical_evidence_units=3)
    service = UniversalEvidenceIngestionService(limits)

    single = StructuredJsonSource(
        document_id="single-budget",
        type="structured_json",
        content={"items": list(range(100))},
    )
    with pytest.raises(GatekeeperError) as single_error:
        service.ingest([single], _output())
    assert single_error.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert counts["drafts"] <= limits.max_canonical_evidence_units + 1
    assert counts["units"] <= limits.max_canonical_evidence_units + 1

    counts.update(drafts=0, units=0)
    cumulative_sources: list[EvidenceSource] = [
        TranscriptSource(
            document_id="cumulative-one",
            type="transcript",
            content=[
                TranscriptSegment(segment=index, text=f"First source segment {index}.")
                for index in range(2)
            ],
        ),
        TranscriptSource(
            document_id="cumulative-two",
            type="transcript",
            content=[
                TranscriptSegment(segment=index, text=f"Second source segment {index}.")
                for index in range(2)
            ],
        ),
        TranscriptSource(
            document_id="must-not-be-built",
            type="transcript",
            content=[
                TranscriptSegment(segment=index, text=f"Later source segment {index}.")
                for index in range(50)
            ],
        ),
    ]
    with pytest.raises(GatekeeperError) as cumulative_error:
        service.ingest(cumulative_sources, _output())
    assert cumulative_error.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert counts["drafts"] <= limits.max_canonical_evidence_units + 1
    assert counts["units"] <= limits.max_canonical_evidence_units + 1


def test_arbitrary_structured_scalars_emit_one_machine_readable_limitation_each() -> None:
    source = StructuredJsonSource(
        document_id="opaque-structured-source",
        type="structured_json",
        content={"vendor_specific_scalar": "alpha"},
    )
    output = StructuredJsonGeneratedOutput(
        type="structured_json",
        content={"vendor_specific_scalar": "beta"},
    )

    result = _ingest(source, output=output)
    assert [
        (limitation.code, limitation.document_id) for limitation in result.limitations
    ] == [
        ("UNINTERPRETED_STRUCTURED_FIELDS", "opaque-structured-source"),
        ("UNINTERPRETED_STRUCTURED_FIELDS", "generated_output"),
    ]


def test_zero_overlap_chunking_keeps_negation_and_medication_tuples_assessable() -> None:
    source_text = (
        "Context filler words. Patient does not have chest pain. "
        "Please continue insulin 10 units nightly."
    )
    output_text = (
        "Context filler words. Patient does not have chest pain. "
        "Please continue insulin 100 units nightly."
    )
    limits = replace(
        IngestionLimits(),
        chunk_characters=26,
        chunk_overlap_characters=0,
        max_chunks=100,
    )
    ingestion = _ingest(
        TextEvidenceSource(document_id="adversarial-chunks", content=source_text),
        output=_output(output_text),
        limits=limits,
    )
    source_document = ingestion.source_documents[0]
    generated_document = ingestion.generated_document
    source_chunks = [chunk.text for chunk in source_document.chunks]
    generated_chunks = [chunk.text for chunk in generated_document.chunks]

    assert all(len(chunk) <= limits.chunk_characters for chunk in source_chunks)
    assert all(len(chunk) <= limits.chunk_characters for chunk in generated_chunks)
    assert any("does not have chest pain" in chunk for chunk in source_chunks)
    assert any("does not have chest pain" in chunk for chunk in generated_chunks)
    assert any("insulin 10 units" in chunk for chunk in source_chunks)
    assert any("insulin 100 units" in chunk for chunk in generated_chunks)
    _assert_complete_chunk_coverage(source_document)
    _assert_complete_chunk_coverage(generated_document)

    extractor = CanonicalEvidenceExtractor(
        ClinicalNLPService(),
        [source_document, generated_document],
        max_claims=100,
        processing_timeout_seconds=10,
    )
    source_claims = extractor.extract(source_document.normalized_text, source_document.document_id)
    generated_claims = extractor.extract(
        generated_document.normalized_text,
        generated_document.document_id,
    )
    source_doses = {
        claim.attributes.get("normalized_value")
        for claim in source_claims
        if claim.type is ClaimType.DOSE
    }
    generated_doses = {
        claim.attributes.get("normalized_value")
        for claim in generated_claims
        if claim.type is ClaimType.DOSE
    }
    assert source_doses == {10}
    assert generated_doses == {100}
    assert any(
        claim.type is ClaimType.SYMPTOM and claim.assertion is AssertionStatus.ABSENT
        for claim in source_claims
    )

    comparison = compare(
        ingestion.source_bundle,
        ingestion.generated_output,
        nlp_service=CanonicalEvidenceExtractor(
            ClinicalNLPService(),
            [source_document, generated_document],
            max_claims=100,
            processing_timeout_seconds=10,
        ),
        enable_semantic=False,
    )
    assert any(
        finding.type is FindingType.ALTERED
        and finding.category is FindingCategory.MEDICATION_DOSE
        for finding in comparison.findings
    )


def test_api_accepts_actual_multiline_json_and_a_long_soap_output(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    multiline_source = """Segment 1
00:03
PATIENT: I have a headache and nausea.

Segment 2
00:22
CLINICIAN: I'll review the blood pressure of 150/95 mmHg.

Segment 3
00:31
PATIENT: Bright lights make the headache worse. 🩺
"""
    long_output = _long_soap_note()
    payload = {
        "system_id": system_id,
        "system_version": "1.4.0",
        "sources": [
            {
                "document_id": "actual-multiline-json",
                "type": "clinical_text",
                "content": multiline_source,
            }
        ],
        "generated_output": {"type": "clinical_note", "content": long_output},
    }

    response = api.client.post("/v1/evaluations", headers=api.authorization(), json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["execution_status"] == "COMPLETED"
    assert body["coverage"]["status"] != "UNASSESSED"
    assert len(long_output.encode("utf-8")) > 4_000

    with api.sessions() as session:
        source_record = session.scalar(
            select(SourceDocumentRecord).where(
                SourceDocumentRecord.evaluation_id == body["evaluation_id"]
            )
        )
        output_record = session.scalar(
            select(GeneratedOutputRecord).where(
                GeneratedOutputRecord.evaluation_id == body["evaluation_id"]
            )
        )
        assert source_record is not None and source_record.content_encrypted is not None
        assert output_record is not None
        source_envelope = api.encryptor.decrypt(source_record.content_encrypted)
        output_envelope = api.encryptor.decrypt(output_record.content_encrypted)
        assert source_envelope["raw_content"] == multiline_source
        assert output_envelope["raw_content"] == long_output


def test_api_accepts_88_segment_transcript_with_provenance_and_idempotency(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    case = _load_long_case()
    payload = {
        "system_id": system_id,
        "system_version": "1.4.0",
        "sources": [case["source"]],
        "generated_output": case["generated_output"],
    }
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    headers = {**api.authorization(), "Idempotency-Key": "synthetic-transcript-88"}

    created = api.client.post("/v1/evaluations", headers=headers, json=payload)
    replayed = api.client.post("/v1/evaluations", headers=headers, json=payload)
    assert created.status_code == replayed.status_code == 201, created.text
    assert payload_bytes > 10_000
    assert created.json()["execution_status"] == "COMPLETED"
    assert created.json()["coverage"]["status"] != "UNASSESSED"
    assert created.json()["findings"]
    provenance = [
        claim["attributes"].get("evidence_provenance")
        for claim in created.json()["source_claims"]
        if claim["attributes"].get("evidence_provenance") is not None
    ]
    assert provenance
    assert any(item["segment_id"] is not None for item in provenance)
    assert replayed.headers["Idempotent-Replayed"] == "true"
    assert replayed.json()["evaluation_id"] == created.json()["evaluation_id"]


def test_api_never_reports_full_coverage_for_changed_arbitrary_structured_scalars(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    response = api.client.post(
        "/v1/evaluations",
        headers=api.authorization(),
        json={
            "system_id": system_id,
            "system_version": "1.4.0",
            "sources": [
                {
                    "document_id": "opaque-api-source",
                    "type": "structured_json",
                    "content": {"vendor_specific_scalar": "alpha"},
                }
            ],
            "generated_output": {
                "type": "structured_json",
                "content": {"vendor_specific_scalar": "beta"},
            },
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["execution_status"] == "COMPLETED"
    assert body["coverage"]["status"] in {"REDUCED", "UNASSESSED"}
    assert body["coverage"]["status"] != "FULL"
    structured_limitations = [
        (detail["code"], detail["document_id"])
        for detail in body["coverage"]["limitation_details"]
        if detail["code"] == "UNINTERPRETED_STRUCTURED_FIELDS"
    ]
    assert structured_limitations == [
        ("UNINTERPRETED_STRUCTURED_FIELDS", "opaque-api-source"),
        ("UNINTERPRETED_STRUCTURED_FIELDS", "generated_output"),
    ]


def test_api_accepts_all_structured_types_and_reduces_unsupported_unit_coverage(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    payload = {
        "system_id": system_id,
        "system_version": "1.4.0",
        "sources": [
            {
                "document_id": "api-transcript",
                "type": "transcript",
                "content": [
                    {
                        "segment": 1,
                        "timestamp": "00:01",
                        "speaker": "PATIENT",
                        "text": "I take metformin twice daily.",
                    }
                ],
            },
            {
                "document_id": "api-labs",
                "type": "lab_results",
                "content": [
                    {"test": "potassium", "value": 4.6, "unit": "mmol/L"},
                    {"test": "synthetic marker", "value": -2.5, "unit": "arbitrary-u"},
                ],
            },
            {
                "document_id": "api-medications",
                "type": "medication_list",
                "content": [
                    {
                        "name": "Metformin",
                        "dose": 0.5,
                        "unit": "g",
                        "frequency": "BD",
                        "route": "PO",
                        "status": "ACTIVE",
                    }
                ],
            },
            {
                "document_id": "api-structured",
                "type": "structured_json",
                "content": {"encounter": {"status": "complete"}, "scores": [91, "98%"]},
            },
            {
                "document_id": "api-rag",
                "type": "rag_context",
                "content": [
                    {
                        "chunk_id": "chunk-api-1",
                        "text": "Synthetic context reports metformin 500 mg twice daily.",
                        "source_uri": "ehr://synthetic/medications",
                        "source_document_id": "upstream-api-1",
                        "retrieval_score": 0.87,
                        "metadata": {"rank": 1},
                    }
                ],
            },
        ],
        "generated_output": {
            "type": "structured_json",
            "content": {
                "summary": "Continue metformin 500 mg twice daily.",
                "potassium": {"value": 4.6, "unit": "mmol/L"},
            },
        },
    }

    response = api.client.post("/v1/evaluations", headers=api.authorization(), json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["execution_status"] == "COMPLETED"
    assert body["coverage"]["status"] == "REDUCED"
    assert any(
        detail["code"] == "UNSUPPORTED_UNIT"
        and detail["document_id"] == "api-labs"
        for detail in body["coverage"]["limitation_details"]
    )


def test_api_returns_safe_413_for_decoded_evidence_over_limit(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    private_marker = "PRIVATE-SYNTHETIC-MARKER"
    content = private_marker + "x" * (1_048_576 + 1)
    response = api.client.post(
        "/v1/evaluations",
        headers=api.authorization(),
        json={
            "system_id": system_id,
            "system_version": "1.4.0",
            "sources": [
                {"document_id": "too-large", "type": "clinical_text", "content": content}
            ],
            "generated_output": {"type": "clinical_note", "content": "Synthetic note."},
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert response.json()["error"]["retryable"] is False
    assert private_marker not in response.text


def test_api_validation_errors_are_specific_and_do_not_echo_evidence(
    ingestion_api: tuple[ApiHarness, str],
) -> None:
    api, system_id = ingestion_api
    private_marker = "PRIVATE-SYNTHETIC-EVIDENCE"
    response = api.client.post(
        "/v1/evaluations",
        headers=api.authorization(),
        json={
            "system_id": system_id,
            "system_version": "1.4.0",
            "sources": [
                {"document_id": "unknown", "type": "FHIR", "content": private_marker}
            ],
            "generated_output": {"content": "Synthetic note."},
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_SOURCE_TYPE"
    assert private_marker not in response.text


def test_openapi_exposes_named_examples_that_all_validate() -> None:
    expected = {
        "smallClinicalText",
        "structuredTranscript",
        "labResults",
        "medicationList",
        "multipleSources",
        "longText",
        "ragContext",
    }
    assert set(EVALUATION_OPENAPI_EXAMPLES) == expected
    for example in EVALUATION_OPENAPI_EXAMPLES.values():
        SingleEvaluationCreate.model_validate(example["value"])

    api = _build_harness()
    try:
        schema = api.client.get("/openapi.json")
        assert schema.status_code == 200
        examples = schema.json()["paths"]["/v1/evaluations"]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]
        assert set(examples) == expected
    finally:
        api.close()
        if api in _ACTIVE_HARNESSES:
            _ACTIVE_HARNESSES.remove(api)

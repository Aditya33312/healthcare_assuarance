import pytest
from pydantic import ValidationError

from backend.app.domain.enums import ClaimType, FindingType, Severity
from backend.app.domain.models import (
    Claim,
    Finding,
    GeneratedOutput,
    SourceDocument,
    TextSpan,
)


def test_text_span_requires_exact_offsets() -> None:
    span = TextSpan(document_id="src-1", text="left", start=8, end=12)
    assert span.text == "left"

    with pytest.raises(ValidationError):
        TextSpan(document_id="src-1", text="left", start=8, end=13)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SourceDocument,
            {
                "document_id": "source",
                "type": "Jane Doe MRN 12345 discharge summary",
                "content": "Synthetic content.",
            },
        ),
        (
            GeneratedOutput,
            {
                "type": "output-for-patient-jane-doe",
                "content": "Synthetic content.",
            },
        ),
    ],
)
def test_artifact_type_is_a_bounded_metadata_enum(
    model: type[SourceDocument] | type[GeneratedOutput],
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_claim_rejects_unversioned_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(
            {
                "claim_id": "claim-1",
                "concept": "pain",
                "type": ClaimType.SYMPTOM,
                "source_span": TextSpan(text="pain", start=0, end=4),
                "undocumented_field": True,
            }
        )


def test_finding_contains_explanation_and_exact_evidence() -> None:
    finding = Finding(
        finding_id="finding-1",
        type=FindingType.ALTERED,
        category="laterality",
        severity=Severity.HIGH,
        source=TextSpan(document_id="src-1", text="left", start=0, end=4),
        generated=TextSpan(document_id="generated", text="right", start=0, end=5),
        reason_code="LATERALITY_MISMATCH",
        explanation="Laterality changed from LEFT to RIGHT.",
        confidence=1.0,
        rule_id="CLINICAL.LATERALITY_MISMATCH",
        rule_version="1.0.0",
    )

    assert finding.explanation
    assert finding.source is not None
    assert finding.generated is not None

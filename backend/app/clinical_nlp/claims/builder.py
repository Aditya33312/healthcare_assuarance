"""Convert validated internal entities into public versioned Claim contracts."""

from __future__ import annotations

import hashlib
from typing import Any

from backend.app.clinical_nlp.entities import ExtractedEntity
from backend.app.clinical_nlp.limits import ClinicalNLPLimitError, ensure_deadline
from backend.app.domain.models import Claim, TextSpan


def _claim_id(document_id: str, entity: ExtractedEntity, ordinal: int) -> str:
    identity = (
        f"{document_id}\x1f{entity.start}\x1f{entity.end}\x1f{entity.type.value}\x1f"
        f"{entity.concept}\x1f{ordinal}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"claim_{digest}"


class ClaimBuilder:
    def build(
        self,
        entities: list[ExtractedEntity],
        *,
        text: str,
        document_id: str,
        max_claims: int | None = None,
        deadline: float | None = None,
    ) -> list[Claim]:
        ordered = sorted(
            enumerate(entities),
            key=lambda item: (item[1].start, item[1].end, item[1].type.value, item[0]),
        )
        claim_ids = {
            original_index: _claim_id(document_id, entity, ordinal)
            for ordinal, (original_index, entity) in enumerate(ordered, start=1)
        }
        claims: list[Claim] = []
        for original_index, entity in ordered:
            ensure_deadline(deadline)
            if max_claims is not None and len(claims) >= max_claims:
                raise ClinicalNLPLimitError("clinical claim-count limit exceeded")
            if not entity.validate_against(text):
                continue
            attributes: dict[str, Any] = {
                key: value
                for key, value in entity.attributes.items()
                if not key.startswith("_") and value is not None
            }
            if entity.parent_index is not None and entity.parent_index in claim_ids:
                attributes["parent_claim_id"] = claim_ids[entity.parent_index]
            claims.append(
                Claim(
                    claim_id=claim_ids[original_index],
                    concept=entity.concept,
                    type=entity.type,
                    attributes=attributes,
                    assertion=entity.assertion,
                    subject=entity.subject,
                    source_span=TextSpan(
                        document_id=document_id,
                        text=entity.text,
                        start=entity.start,
                        end=entity.end,
                    ),
                    confidence=max(0.0, min(1.0, entity.confidence)),
                    extractor=entity.origin,
                    extractor_version=entity.origin_version,
                )
            )
        return claims

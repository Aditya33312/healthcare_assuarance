"""Link high-risk scalar entities to nearby semantic anchors."""

from __future__ import annotations

from backend.app.clinical_nlp.entities import ExtractedEntity
from backend.app.clinical_nlp.limits import ensure_deadline
from backend.app.clinical_nlp.preprocessing import PreprocessedText
from backend.app.clinical_nlp.versions import RELATION_RULES_VERSION
from backend.app.domain.enums import ClaimType

_PARENT_TYPES: dict[ClaimType, set[ClaimType]] = {
    ClaimType.DOSE: {ClaimType.MEDICATION},
    ClaimType.UNIT: {ClaimType.MEDICATION},
    ClaimType.FREQUENCY: {ClaimType.MEDICATION},
    ClaimType.ROUTE: {ClaimType.MEDICATION},
    ClaimType.LATERALITY: {
        ClaimType.SYMPTOM,
        ClaimType.CONDITION,
        ClaimType.CLINICAL_FACT,
        ClaimType.BODY_SITE,
        ClaimType.PROCEDURE,
    },
    ClaimType.BODY_SITE: {ClaimType.SYMPTOM, ClaimType.CONDITION},
    ClaimType.DATE: {
        ClaimType.CONDITION,
        ClaimType.SYMPTOM,
        ClaimType.CLINICAL_FACT,
        ClaimType.PROCEDURE,
        ClaimType.MEDICATION,
    },
    ClaimType.DURATION: {
        ClaimType.CONDITION,
        ClaimType.SYMPTOM,
        ClaimType.CLINICAL_FACT,
        ClaimType.MEDICATION,
    },
    ClaimType.LAB_VALUE: {ClaimType.LAB_TEST},
}
_MAX_DISTANCE = {
    ClaimType.DOSE: 45,
    ClaimType.UNIT: 50,
    ClaimType.FREQUENCY: 65,
    ClaimType.ROUTE: 55,
    ClaimType.LATERALITY: 30,
    ClaimType.BODY_SITE: 35,
    ClaimType.DATE: 100,
    ClaimType.DURATION: 90,
    ClaimType.LAB_VALUE: 45,
}


def _distance(modifier: ExtractedEntity, anchor: ExtractedEntity) -> int:
    if modifier.overlaps(anchor):
        return 0
    if modifier.start >= anchor.end:
        return modifier.start - anchor.end
    return anchor.start - modifier.end + 4


class RelationLinker:
    version = RELATION_RULES_VERSION

    def link(
        self,
        entities: list[ExtractedEntity],
        document: PreprocessedText,
        *,
        deadline: float | None = None,
    ) -> list[ExtractedEntity]:
        for modifier_index, modifier in enumerate(entities):
            ensure_deadline(deadline)
            allowed_parents = _PARENT_TYPES.get(modifier.type)
            if not allowed_parents:
                continue
            modifier_sentence = document.sentence_at(modifier.start)
            candidates: list[tuple[int, int]] = []
            for anchor_index, anchor in enumerate(entities):
                if anchor_index % 128 == 0:
                    ensure_deadline(deadline)
                if anchor_index == modifier_index or anchor.type not in allowed_parents:
                    continue
                anchor_sentence = document.sentence_at(anchor.start)
                if modifier_sentence is None or anchor_sentence is not modifier_sentence:
                    continue
                distance = _distance(modifier, anchor)
                if distance > _MAX_DISTANCE[modifier.type]:
                    continue
                between_start = min(modifier.end, anchor.end)
                between_end = max(modifier.start, anchor.start)
                if ";" in document.text[between_start:between_end]:
                    continue
                candidates.append((distance, anchor_index))
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (item[0], abs(entities[item[1]].start - modifier.start))
            )
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                modifier.attributes["relation_ambiguous"] = True
                continue
            parent_index = candidates[0][1]
            modifier.parent_index = parent_index
            parent = entities[parent_index]
            modifier.attributes["parent_concept"] = parent.concept
            modifier.attributes["relation"] = "MODIFIES"
            self._apply(parent, modifier)

        return entities

    def _apply(self, parent: ExtractedEntity, modifier: ExtractedEntity) -> None:
        attributes = modifier.attributes
        if modifier.type == ClaimType.DOSE:
            parent.attributes.update(
                {
                    "dose": attributes["dose"],
                    "unit": attributes["unit"],
                    "normalized_value": attributes["normalized_value"],
                    "normalized_unit": attributes["normalized_unit"],
                    "raw_dose": attributes["raw_value"],
                    "raw_unit": attributes["raw_unit"],
                }
            )
        elif modifier.type == ClaimType.UNIT:
            parent.attributes.setdefault("unit", attributes["normalized_unit"])
            parent.attributes.setdefault("normalized_unit", attributes["normalized_unit"])
        elif modifier.type == ClaimType.FREQUENCY:
            parent.attributes["frequency"] = attributes["frequency"]
        elif modifier.type == ClaimType.ROUTE:
            parent.attributes["route"] = attributes["route"]
        elif modifier.type == ClaimType.LATERALITY:
            parent.attributes["laterality"] = attributes["laterality"]
        elif modifier.type == ClaimType.BODY_SITE:
            parent.attributes.setdefault("body_site", modifier.concept)
        elif modifier.type == ClaimType.DATE:
            parent.attributes["date"] = attributes["date"]
        elif modifier.type == ClaimType.DURATION:
            parent.attributes["duration"] = attributes["duration"]
        elif modifier.type == ClaimType.LAB_VALUE:
            parent.attributes["value"] = attributes["value"]
            parent.attributes["normalized_value"] = attributes["normalized_value"]
            if "unit" in attributes:
                parent.attributes["unit"] = attributes["unit"]
                parent.attributes["normalized_unit"] = attributes["normalized_unit"]

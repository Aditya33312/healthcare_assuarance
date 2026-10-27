"""Conservative assertion, uncertainty, history, and subject scope rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.clinical_nlp.preprocessing import SentenceSpan
from backend.app.clinical_nlp.versions import ASSERTION_RULES_VERSION
from backend.app.domain.enums import AssertionStatus, SubjectType

_CONTRAST_RE = re.compile(r"(?:[;]|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\byet\b)", re.I)
_NEGATION_RE = re.compile(
    r"\b(?:no|denies|denied|deny|without|negative\s+for|free\s+of|absence\s+of|"
    r"not|never|none|no\s+evidence\s+of|no\s+signs?\s+of)\b",
    re.IGNORECASE,
)
_POSSIBLE_RE = re.compile(
    r"\b(?:possible|possibly|may|might|could|suspected|suspicion\s+of|concern\s+for|"
    r"cannot\s+exclude|can't\s+exclude|rule\s+out|consider|suggestive\s+of|"
    r"compatible\s+with|versus|vs\.?)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:uncertain|unclear|equivocal|unknown\s+whether|questionable|unlikely|"
    r"indeterminate)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:history\s+(?:of|(?:is\s+)?(?:significant\s+)?for)|hx\s+of|h/o|"
    r"previous(?:ly)?|prior|remote|resolved|status\s+post|s/p)\b",
    re.IGNORECASE,
)
_FAMILY_RE = re.compile(
    r"\b(?:family\s+(?:history|hx)|mother|father|sister|brother|maternal|paternal|"
    r"grandmother|grandfather|daughter|son|aunt|uncle)\b",
    re.IGNORECASE,
)
_OTHER_PERSON_RE = re.compile(r"\b(?:caregiver|partner|spouse|roommate|contact)\b", re.IGNORECASE)
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:if\s+(?:the\s+)?patient\s+(?:develops?|experiences?)|should\s+(?:the\s+)?"
    r"patient\s+develop)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(r"\b(?:if|provided\s+that|in\s+case\s+of)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AssertionResult:
    assertion: AssertionStatus
    subject: SubjectType
    confidence: float
    rule: str


def _recent_match(pattern: re.Pattern[str], text: str, max_distance: int) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    if len(text) - match.end() > max_distance:
        return None
    return match


class AssertionRuleEngine:
    """Resolve assertion state from a bounded sentence/clause prefix."""

    version = ASSERTION_RULES_VERSION

    def resolve(
        self,
        *,
        entity_start: int,
        entity_end: int | None = None,
        sentence: SentenceSpan | None,
        forced_assertion: AssertionStatus | None = None,
        forced_subject: SubjectType | None = None,
    ) -> AssertionResult:
        if forced_assertion is not None:
            return AssertionResult(
                forced_assertion,
                forced_subject or SubjectType.PATIENT,
                1.0,
                "forced-parser-state",
            )
        if sentence is None:
            return AssertionResult(
                AssertionStatus.PRESENT,
                forced_subject or SubjectType.PATIENT,
                0.85,
                "default-present",
            )

        relative_start = max(0, min(len(sentence.text), entity_start - sentence.start))
        resolved_end = entity_start if entity_end is None else entity_end
        relative_end = max(relative_start, min(len(sentence.text), resolved_end - sentence.start))
        prefix = sentence.text[:relative_start]
        suffix = sentence.text[relative_end : relative_end + 90]
        contrast_matches = list(_CONTRAST_RE.finditer(prefix))
        clause_prefix = prefix[contrast_matches[-1].end() :] if contrast_matches else prefix
        clause_prefix = clause_prefix[-180:]
        section = (sentence.section or "").casefold()

        family_after = bool(
            re.match(r"\s+(?:in|of)\s+(?:his\s+|her\s+|the\s+)?", suffix, re.IGNORECASE)
            and _FAMILY_RE.search(suffix)
        )
        family_context = (
            bool(_FAMILY_RE.search(clause_prefix)) or family_after or "family" in section
        )
        if forced_subject is not None:
            subject = forced_subject
        elif family_context:
            subject = SubjectType.FAMILY
        elif _OTHER_PERSON_RE.search(clause_prefix):
            subject = SubjectType.OTHER_PERSON
        else:
            subject = SubjectType.PATIENT

        negation = _recent_match(_NEGATION_RE, clause_prefix, 110)
        if negation is not None:
            tail = clause_prefix[negation.start() :].casefold()
            false_positive = any(
                phrase in tail
                for phrase in ("no change in", "not only", "may not", "might not", "could not")
            )
            if "not ruled out" in tail or "not excluded" in tail:
                return AssertionResult(AssertionStatus.POSSIBLE, subject, 0.94, "possibility-cue")
            if not false_positive:
                return AssertionResult(AssertionStatus.ABSENT, subject, 0.99, "negation-cue")

        if re.match(
            r"\s*(?:is|was|are|were|has\s+been)?\s*(?:absent|negative|not\s+present|"
            r"denied|ruled\s+out|excluded)\b",
            suffix,
            re.IGNORECASE,
        ):
            return AssertionResult(AssertionStatus.ABSENT, subject, 0.98, "post-negation-cue")
        if re.match(
            r"\s*(?:cannot|can't|could\s+not)\s+be\s+(?:excluded|ruled\s+out)\b",
            suffix,
            re.IGNORECASE,
        ):
            return AssertionResult(AssertionStatus.POSSIBLE, subject, 0.94, "post-possibility-cue")
        if re.match(
            r"\s*(?:may|might|could)\s+be\s+(?:present|active|responsible)\b",
            suffix,
            re.IGNORECASE,
        ):
            return AssertionResult(AssertionStatus.POSSIBLE, subject, 0.94, "post-possibility-cue")
        if suffix.lstrip().startswith("?"):
            return AssertionResult(AssertionStatus.UNCERTAIN, subject, 0.9, "question-mark-cue")

        if family_context:
            return AssertionResult(
                AssertionStatus.FAMILY_HISTORY,
                SubjectType.FAMILY,
                0.97,
                "family-history-context",
            )

        if _recent_match(_HYPOTHETICAL_RE, clause_prefix, 120):
            return AssertionResult(AssertionStatus.HYPOTHETICAL, subject, 0.94, "hypothetical-cue")
        if _recent_match(_CONDITIONAL_RE, clause_prefix, 100):
            return AssertionResult(AssertionStatus.CONDITIONAL, subject, 0.91, "conditional-cue")
        if _recent_match(_UNCERTAIN_RE, clause_prefix, 100):
            return AssertionResult(AssertionStatus.UNCERTAIN, subject, 0.94, "uncertainty-cue")
        if _recent_match(_POSSIBLE_RE, clause_prefix, 100):
            return AssertionResult(AssertionStatus.POSSIBLE, subject, 0.95, "possibility-cue")
        if _recent_match(_HISTORICAL_RE, clause_prefix, 120) or any(
            label in section for label in ("past medical", "medical history", "history")
        ):
            return AssertionResult(AssertionStatus.HISTORICAL, subject, 0.96, "history-cue")
        return AssertionResult(AssertionStatus.PRESENT, subject, 0.98, "default-present")

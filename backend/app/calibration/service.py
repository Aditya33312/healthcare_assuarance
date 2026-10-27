"""Application service for blinded labels, adjudication, and calibration evidence."""

from __future__ import annotations

import hashlib
import secrets
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import combinations
from random import SystemRandom

from backend.app.calibration.models import (
    AdjudicationReason,
    AdjudicationRecord,
    AgreementMetrics,
    BlindedAssignment,
    BlindedReviewItem,
    CalibrationCandidate,
    CalibrationStatus,
    CalibrationVersion,
    ExternalClinicalValidationAttestation,
    ReviewerAssignmentRecord,
    ReviewerLabel,
    StatusTransition,
    ValidationBasis,
)
from backend.app.calibration.repository import CalibrationRepository
from backend.app.domain.enums import FindingType, Severity

Clock = Callable[[], datetime]
TokenFactory = Callable[[], str]


class CalibrationNotFoundError(LookupError):
    """A missing and a cross-tenant calibration intentionally share this error."""


class CalibrationConflictError(ValueError):
    """The requested workflow transition is not permitted."""


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _stable_calibration_id(
    organization_id: str,
    severity_policy_id: str,
    version: str,
) -> str:
    material = f"{organization_id}\x1f{severity_policy_id}\x1f{version}"
    return f"cal_{hashlib.sha256(material.encode()).hexdigest()[:32]}"


def _chance_corrected_agreement(matches: int, pairs: int, values: Iterable[str]) -> float | None:
    if pairs == 0:
        return None
    value_counts = Counter(values)
    total = sum(value_counts.values())
    if total == 0:
        return None
    expected = sum((count / total) ** 2 for count in value_counts.values())
    observed = matches / pairs
    if expected >= 1:
        return None
    return round((observed - expected) / (1 - expected), 6)


class ClinicianCalibrationService:
    """Coordinate calibration evidence without claiming to provide clinical review.

    Reviewer qualification and final clinical validation are customer-supplied
    attestations. This service enforces workflow shape and blinding boundaries;
    it neither verifies credentials nor substitutes for a clinician panel.
    """

    def __init__(
        self,
        repository: CalibrationRepository,
        *,
        clock: Clock = _default_clock,
        token_factory: TokenFactory | None = None,
    ) -> None:
        self.repository = repository
        self.clock = clock
        self.token_factory = token_factory or (lambda: secrets.token_urlsafe(18))
        self._random = SystemRandom()

    def create_version(
        self,
        organization_id: str,
        *,
        severity_policy_id: str,
        version: str,
        protocol_ref: str,
        candidates: tuple[CalibrationCandidate, ...],
        created_by_user_id: str,
        minimum_reviewers_per_item: int = 2,
    ) -> CalibrationVersion:
        now = self.clock()
        calibration_id = _stable_calibration_id(
            organization_id,
            severity_policy_id,
            version,
        )
        metrics = AgreementMetrics(
            item_count=len(candidates),
            reviewed_item_count=0,
            multi_reviewed_item_count=0,
            reviewer_count=0,
            review_count=0,
            reviewer_pair_count=0,
        )
        record = CalibrationVersion(
            calibration_id=calibration_id,
            organization_id=organization_id,
            severity_policy_id=severity_policy_id,
            version=version,
            protocol_ref=protocol_ref,
            minimum_reviewers_per_item=minimum_reviewers_per_item,
            status=CalibrationStatus.DRAFT,
            candidates=candidates,
            agreement_metrics=metrics,
            status_history=(
                StatusTransition(
                    status=CalibrationStatus.DRAFT,
                    actor_id=created_by_user_id,
                    occurred_at=now,
                ),
            ),
            created_at=now,
            updated_at=now,
        )
        try:
            self.repository.add(organization_id, record)
        except ValueError as exc:
            raise CalibrationConflictError("The calibration version already exists.") from exc
        return record

    def create_blinded_assignment(
        self,
        organization_id: str,
        calibration_id: str,
        *,
        reviewer_id: str,
        qualification_attested: bool,
        candidate_ids: tuple[str, ...] | None = None,
    ) -> BlindedAssignment:
        record = self._require(organization_id, calibration_id)
        if record.status not in {CalibrationStatus.DRAFT, CalibrationStatus.IN_REVIEW}:
            raise CalibrationConflictError(
                "The calibration no longer accepts reviewer assignments."
            )
        if qualification_attested is not True:
            raise CalibrationConflictError("A reviewer qualification attestation is required.")
        if any(assignment.reviewer_id == reviewer_id for assignment in record.assignments):
            raise CalibrationConflictError("The reviewer already has an assignment.")

        candidate_by_id = {candidate.candidate_id: candidate for candidate in record.candidates}
        selected_ids = list(candidate_ids or tuple(candidate_by_id))
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise CalibrationConflictError("Assignments require unique calibration candidates.")
        if any(candidate_id not in candidate_by_id for candidate_id in selected_ids):
            raise CalibrationConflictError("An assignment candidate was not found.")
        self._random.shuffle(selected_ids)

        assignment_id = f"assign_{self.token_factory()}"
        tokens: dict[str, str] = {}
        blinded_items: list[BlindedReviewItem] = []
        for candidate_id in selected_ids:
            item_token = f"item_{self.token_factory()}"
            if item_token in tokens:
                raise CalibrationConflictError("The assignment token source produced a collision.")
            tokens[item_token] = candidate_id
            blinded_items.append(
                BlindedReviewItem(
                    item_token=item_token,
                    blinded_payload_ref=candidate_by_id[candidate_id].blinded_payload_ref,
                )
            )
        now = self.clock()
        assignment = ReviewerAssignmentRecord(
            assignment_id=assignment_id,
            reviewer_id=reviewer_id,
            qualification_attested=True,
            item_tokens=tokens,
            created_at=now,
        )
        history = record.status_history
        status: CalibrationStatus = record.status
        if status is CalibrationStatus.DRAFT:
            status = CalibrationStatus.IN_REVIEW
            history = (
                *history,
                StatusTransition(
                    status=status,
                    actor_id=reviewer_id,
                    occurred_at=now,
                ),
            )
        updated = record.model_copy(
            update={
                "status": status,
                "assignments": (*record.assignments, assignment),
                "status_history": history,
                "updated_at": now,
            }
        )
        self.repository.save(organization_id, updated)
        return BlindedAssignment(
            assignment_id=assignment_id,
            calibration_version=record.version,
            items=tuple(blinded_items),
        )

    def submit_blinded_label(
        self,
        organization_id: str,
        calibration_id: str,
        *,
        assignment_id: str,
        item_token: str,
        reviewer_id: str,
        label: FindingType,
        severity: Severity,
    ) -> ReviewerLabel:
        record = self._require(organization_id, calibration_id)
        if record.status is not CalibrationStatus.IN_REVIEW:
            raise CalibrationConflictError("The calibration is not accepting blinded labels.")
        assignment = next(
            (item for item in record.assignments if item.assignment_id == assignment_id),
            None,
        )
        if assignment is None or item_token not in assignment.item_tokens:
            raise CalibrationConflictError("The blinded assignment item was not found.")
        if assignment.reviewer_id != reviewer_id:
            raise CalibrationConflictError("The blinded assignment belongs to another reviewer.")
        candidate_id = assignment.item_tokens[item_token]
        if any(
            review.assignment_id == assignment_id and review.candidate_id == candidate_id
            for review in record.reviews
        ):
            raise CalibrationConflictError("The blinded item already has a submitted label.")

        now = self.clock()
        review = ReviewerLabel(
            review_id=f"creview_{self.token_factory()}",
            assignment_id=assignment_id,
            candidate_id=candidate_id,
            reviewer_id=assignment.reviewer_id,
            label=label,
            severity=severity,
            submitted_at=now,
        )
        reviews = (*record.reviews, review)
        metrics = self._agreement_metrics(record.candidates, reviews)
        status: CalibrationStatus = record.status
        history = record.status_history
        if self._ready_for_adjudication(record, reviews):
            status = CalibrationStatus.READY_FOR_ADJUDICATION
            history = (
                *history,
                StatusTransition(
                    status=status,
                    actor_id=assignment.reviewer_id,
                    occurred_at=now,
                ),
            )
        updated = record.model_copy(
            update={
                "reviews": reviews,
                "agreement_metrics": metrics,
                "status": status,
                "status_history": history,
                "updated_at": now,
            }
        )
        self.repository.save(organization_id, updated)
        return review

    def agreement_metrics(
        self,
        organization_id: str,
        calibration_id: str,
    ) -> AgreementMetrics:
        return self._require(organization_id, calibration_id).agreement_metrics

    def adjudicate(
        self,
        organization_id: str,
        calibration_id: str,
        *,
        candidate_id: str,
        adjudicator_id: str,
        qualification_attested: bool,
        label: FindingType,
        severity: Severity,
        reason: AdjudicationReason,
    ) -> AdjudicationRecord:
        record = self._require(organization_id, calibration_id)
        if record.status is not CalibrationStatus.READY_FOR_ADJUDICATION:
            raise CalibrationConflictError("The calibration is not ready for adjudication.")
        if qualification_attested is not True:
            raise CalibrationConflictError("An adjudicator qualification attestation is required.")
        if candidate_id not in {candidate.candidate_id for candidate in record.candidates}:
            raise CalibrationConflictError("The adjudication candidate was not found.")
        if any(item.candidate_id == candidate_id for item in record.adjudications):
            raise CalibrationConflictError("The candidate was already adjudicated.")
        if any(
            review.candidate_id == candidate_id and review.reviewer_id == adjudicator_id
            for review in record.reviews
        ):
            raise CalibrationConflictError(
                "The adjudicator must be independent of this candidate's blinded reviewers."
            )

        now = self.clock()
        adjudication = AdjudicationRecord(
            adjudication_id=f"adjud_{self.token_factory()}",
            candidate_id=candidate_id,
            adjudicator_id=adjudicator_id,
            qualification_attested=True,
            label=label,
            severity=severity,
            reason=reason,
            adjudicated_at=now,
        )
        adjudications = (*record.adjudications, adjudication)
        status: CalibrationStatus = record.status
        history = record.status_history
        if len(adjudications) == len(record.candidates):
            status = CalibrationStatus.ADJUDICATED
            history = (
                *history,
                StatusTransition(
                    status=status,
                    actor_id=adjudicator_id,
                    occurred_at=now,
                ),
            )
        updated = record.model_copy(
            update={
                "adjudications": adjudications,
                "status": status,
                "status_history": history,
                "updated_at": now,
            }
        )
        self.repository.save(organization_id, updated)
        return adjudication

    def mark_calibrated(
        self,
        organization_id: str,
        calibration_id: str,
        *,
        attestation: ExternalClinicalValidationAttestation,
    ) -> CalibrationVersion:
        record = self._require(organization_id, calibration_id)
        if record.status is not CalibrationStatus.ADJUDICATED:
            raise CalibrationConflictError("All candidates must be adjudicated first.")
        now = self.clock()
        updated = record.model_copy(
            update={
                "status": CalibrationStatus.CALIBRATED,
                "validation_basis": ValidationBasis.EXTERNAL_ATTESTATION_UNVERIFIED,
                "external_validation": attestation,
                "status_history": (
                    *record.status_history,
                    StatusTransition(
                        status=CalibrationStatus.CALIBRATED,
                        actor_id=attestation.approved_by_user_id,
                        occurred_at=now,
                    ),
                ),
                "updated_at": now,
            }
        )
        self.repository.save(organization_id, updated)
        return updated

    def get_version(
        self,
        organization_id: str,
        calibration_id: str,
    ) -> CalibrationVersion:
        return self._require(organization_id, calibration_id)

    def _require(self, organization_id: str, calibration_id: str) -> CalibrationVersion:
        record = self.repository.get(organization_id, calibration_id)
        if record is None:
            raise CalibrationNotFoundError("Calibration version not found.")
        return record

    @staticmethod
    def _ready_for_adjudication(
        record: CalibrationVersion,
        reviews: tuple[ReviewerLabel, ...],
    ) -> bool:
        reviewers_by_candidate: defaultdict[str, set[str]] = defaultdict(set)
        for review in reviews:
            reviewers_by_candidate[review.candidate_id].add(review.reviewer_id)
        return all(
            len(reviewers_by_candidate[candidate.candidate_id]) >= record.minimum_reviewers_per_item
            for candidate in record.candidates
        )

    @staticmethod
    def _agreement_metrics(
        candidates: tuple[CalibrationCandidate, ...],
        reviews: tuple[ReviewerLabel, ...],
    ) -> AgreementMetrics:
        by_candidate: defaultdict[str, list[ReviewerLabel]] = defaultdict(list)
        for review in reviews:
            by_candidate[review.candidate_id].append(review)

        pair_count = 0
        label_matches = 0
        severity_matches = 0
        exact_matches = 0
        for candidate_reviews in by_candidate.values():
            for first, second in combinations(candidate_reviews, 2):
                pair_count += 1
                label_matches += first.label is second.label
                severity_matches += first.severity is second.severity
                exact_matches += first.label is second.label and first.severity is second.severity

        label_values = [review.label.value for review in reviews]
        severity_values = [review.severity.value for review in reviews]
        return AgreementMetrics(
            item_count=len(candidates),
            reviewed_item_count=sum(bool(by_candidate[item.candidate_id]) for item in candidates),
            multi_reviewed_item_count=sum(
                len(by_candidate[item.candidate_id]) >= 2 for item in candidates
            ),
            reviewer_count=len({review.reviewer_id for review in reviews}),
            review_count=len(reviews),
            reviewer_pair_count=pair_count,
            label_percent_agreement=(round(label_matches / pair_count, 6) if pair_count else None),
            severity_percent_agreement=(
                round(severity_matches / pair_count, 6) if pair_count else None
            ),
            exact_percent_agreement=(round(exact_matches / pair_count, 6) if pair_count else None),
            label_pairwise_kappa=_chance_corrected_agreement(
                label_matches,
                pair_count,
                label_values,
            ),
            severity_pairwise_kappa=_chance_corrected_agreement(
                severity_matches,
                pair_count,
                severity_values,
            ),
        )

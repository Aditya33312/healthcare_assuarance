"""Single-evaluation application service."""

from backend.app.evaluations.models import SingleEvaluationRequest
from backend.app.evaluations.service import EvaluationService

__all__ = ["EvaluationService", "SingleEvaluationRequest"]

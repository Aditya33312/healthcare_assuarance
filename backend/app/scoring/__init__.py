"""Test-specific scoring public API."""

from backend.app.scoring.engine import (
    score_bias,
    score_clinical_faithfulness,
    score_multiple_choice,
    score_robustness,
    score_safety,
)
from backend.app.scoring.models import (
    BiasPairObservation,
    BiasScore,
    ClinicalFaithfulnessScore,
    MultipleChoiceScore,
    RobustnessObservation,
    RobustnessScore,
    SafetyClassification,
    SafetyScore,
)

__all__ = [
    "BiasPairObservation",
    "BiasScore",
    "ClinicalFaithfulnessScore",
    "MultipleChoiceScore",
    "RobustnessObservation",
    "RobustnessScore",
    "SafetyClassification",
    "SafetyScore",
    "score_bias",
    "score_clinical_faithfulness",
    "score_multiple_choice",
    "score_robustness",
    "score_safety",
]

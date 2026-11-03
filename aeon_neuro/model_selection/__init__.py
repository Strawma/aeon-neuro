"""Plan explicit neuro classification evaluations without fitting models."""

__all__ = ["plan_classification_evaluation", "validate_classification_evaluation"]

from aeon_neuro.model_selection._evaluation import (
    plan_classification_evaluation,
    validate_classification_evaluation,
)

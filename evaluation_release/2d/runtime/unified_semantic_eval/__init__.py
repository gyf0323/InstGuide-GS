"""Shared protocol for cross-repository 2D semantic evaluation."""

from .protocol import (
    PredictionWriter,
    UnifiedSemanticGT,
    score_prediction_directory,
)

__all__ = [
    "PredictionWriter",
    "UnifiedSemanticGT",
    "score_prediction_directory",
]

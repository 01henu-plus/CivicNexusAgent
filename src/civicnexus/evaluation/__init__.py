"""Small, deterministic evaluation surface used by the admin console."""

from civicnexus.evaluation.gold import GOLD_CASES, GOLD_DATASET, GoldCase
from civicnexus.evaluation.metrics import runtime_metrics
from civicnexus.evaluation.runner import (
    CaseScore,
    EvaluationReport,
    EvaluationRunner,
    Metric,
    Prediction,
    default_predictor,
)

__all__ = [
    "GOLD_CASES",
    "GOLD_DATASET",
    "CaseScore",
    "EvaluationReport",
    "EvaluationRunner",
    "GoldCase",
    "Metric",
    "Prediction",
    "default_predictor",
    "runtime_metrics",
]

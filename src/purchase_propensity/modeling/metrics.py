"""Evaluation metrics for binary purchase propensity models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_evaluation_inputs(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate and normalize binary labels and predicted probabilities."""
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_probability, dtype=np.float64)

    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")

    if labels.size == 0:
        raise ValueError("Evaluation inputs cannot be empty.")

    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("Labels and probabilities must have the same length.")

    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Labels must contain both binary classes: 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must contain only finite values.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")

    return labels, probabilities


def evaluate_binary_classifier(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calculate ranking, classification and probability-quality metrics."""
    labels, probabilities = validate_evaluation_inputs(
        y_true,
        y_probability,
        threshold,
    )
    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "positive_prediction_rate": float(predictions.mean()),
        "decision_threshold": float(threshold),
    }

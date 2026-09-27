"""Evaluation metrics for binary purchase propensity models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
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
    probabilities = np.asarray(
        y_probability,
        dtype=np.float64,
    )

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


def select_f1_threshold(
    y_true: ArrayLike,
    y_probability: ArrayLike,
) -> float:
    """Select the validation threshold that maximizes the F1 score."""
    labels, probabilities = validate_evaluation_inputs(
        y_true,
        y_probability,
        threshold=0.50,
    )

    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominators = precision + recall
    f1_scores = np.divide(
        2.0 * precision * recall,
        denominators,
        out=np.zeros_like(denominators),
        where=denominators > 0.0,
    )

    valid_thresholds = (thresholds > 0.0) & (thresholds < 1.0)

    if not valid_thresholds.any():
        raise ValueError("No valid threshold candidates were produced.")

    valid_indexes = np.flatnonzero(valid_thresholds)
    valid_scores = f1_scores[valid_indexes]
    maximum_score = valid_scores.max()

    best_indexes = valid_indexes[np.isclose(valid_scores, maximum_score)]

    selected_index = int(best_indexes[-1])

    return float(thresholds[selected_index])


def evaluate_binary_classifier(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calculate ranking, classification and probability metrics."""
    labels, probabilities = validate_evaluation_inputs(
        y_true,
        y_probability,
        threshold,
    )

    predictions = (probabilities >= threshold).astype(np.int64)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "positive_prediction_rate": float(predictions.mean()),
        "decision_threshold": float(threshold),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_positive": float(true_positive),
    }

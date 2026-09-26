"""Unit tests for binary classification evaluation metrics."""

import numpy as np
import pytest

from purchase_propensity.modeling.metrics import (
    evaluate_binary_classifier,
    validate_evaluation_inputs,
)


def test_evaluate_binary_classifier_returns_expected_metrics() -> None:
    """Calculate perfect ranking and classification metrics."""
    labels = [0, 0, 1, 1]
    probabilities = [0.10, 0.20, 0.80, 0.90]

    metrics = evaluate_binary_classifier(
        labels,
        probabilities,
        threshold=0.50,
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["positive_prediction_rate"] == pytest.approx(0.50)
    assert metrics["decision_threshold"] == pytest.approx(0.50)
    assert 0.0 < metrics["log_loss"] < 1.0
    assert 0.0 < metrics["brier_score"] < 1.0


def test_evaluate_binary_classifier_applies_custom_threshold() -> None:
    """Apply the configured threshold when producing class predictions."""
    labels = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.45, 0.90]

    metrics = evaluate_binary_classifier(
        labels,
        probabilities,
        threshold=0.50,
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.50)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["positive_prediction_rate"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("labels", "probabilities", "threshold", "expected_message"),
    [
        (
            [],
            [],
            0.50,
            "Evaluation inputs cannot be empty",
        ),
        (
            [0, 1],
            [0.20],
            0.50,
            "must have the same length",
        ),
        (
            [[0, 1]],
            [[0.20, 0.80]],
            0.50,
            "must be one-dimensional",
        ),
        (
            [0, 0],
            [0.20, 0.30],
            0.50,
            "must contain both binary classes",
        ),
        (
            [0, 2],
            [0.20, 0.80],
            0.50,
            "must contain both binary classes",
        ),
        (
            [0, 1],
            [0.20, np.nan],
            0.50,
            "only finite values",
        ),
        (
            [0, 1],
            [0.20, 1.20],
            0.50,
            "must be between 0 and 1",
        ),
        (
            [0, 1],
            [0.20, 0.80],
            0.00,
            "threshold must be between 0 and 1",
        ),
        (
            [0, 1],
            [0.20, 0.80],
            1.00,
            "threshold must be between 0 and 1",
        ),
    ],
)
def test_validate_evaluation_inputs_rejects_invalid_values(
    labels: object,
    probabilities: object,
    threshold: float,
    expected_message: str,
) -> None:
    """Reject malformed labels, probabilities and thresholds."""
    with pytest.raises(ValueError, match=expected_message):
        validate_evaluation_inputs(
            labels,
            probabilities,
            threshold,
        )

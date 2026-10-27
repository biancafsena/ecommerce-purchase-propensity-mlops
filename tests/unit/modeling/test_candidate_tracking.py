"""Tests for candidate model tracking."""

from unittest.mock import MagicMock, Mock

import pytest
from sklearn.pipeline import Pipeline

from purchase_propensity.modeling import candidate_tracking


def test_log_candidate_run_records_model_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log candidate parameters, validation metrics and model."""
    classifier = Mock()
    classifier.get_params.return_value = {
        "n_estimators": 300,
        "class_weight": None,
    }

    pipeline = Mock(spec=Pipeline)
    pipeline.named_steps = {"classifier": classifier}

    active_run = MagicMock()
    active_run.info.run_id = "candidate-run-123"

    run_context = MagicMock()
    run_context.__enter__.return_value = active_run

    start_run_mock = Mock(return_value=run_context)
    log_params_mock = Mock()
    log_metrics_mock = Mock()
    log_model_mock = Mock()

    monkeypatch.setattr(candidate_tracking.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(candidate_tracking.mlflow, "set_experiment", Mock())
    monkeypatch.setattr(candidate_tracking.mlflow, "start_run", start_run_mock)
    monkeypatch.setattr(candidate_tracking.mlflow, "log_params", log_params_mock)
    monkeypatch.setattr(candidate_tracking.mlflow, "log_metrics", log_metrics_mock)
    monkeypatch.setattr(
        candidate_tracking.mlflow.sklearn,
        "log_model",
        log_model_mock,
    )

    metrics = {
        "average_precision": 0.35,
        "f1": 0.40,
    }

    run_id = candidate_tracking.log_candidate_run(
        model_name="random_forest",
        pipeline=pipeline,
        metrics=metrics,
        tracking_uri="http://127.0.0.1:5001",
        experiment_name="test-experiment",
    )

    assert run_id == "candidate-run-123"

    start_run_mock.assert_called_once_with(
        run_name="candidate-random_forest",
        tags={
            "model_family": "random_forest",
            "dataset_split": "validation",
            "pipeline_stage": "model_comparison",
        },
    )

    log_params_mock.assert_called_once_with(
        {
            "n_estimators": 300,
            "class_weight": "none",
        }
    )

    log_metrics_mock.assert_called_once_with(metrics)
    log_model_mock.assert_called_once()
    assert log_model_mock.call_args.kwargs["sk_model"] is pipeline

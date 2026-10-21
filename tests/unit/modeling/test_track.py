"""Unit tests for MLflow experiment tracking and model registration."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from purchase_propensity.config import Settings
from purchase_propensity.features.build import FEATURE_COLUMNS
from purchase_propensity.modeling import track
from purchase_propensity.modeling.train import TrainingConfig


@pytest.fixture
def training_config() -> TrainingConfig:
    """Provide a valid baseline training configuration."""
    return TrainingConfig(
        target_column="Revenue",
        random_seed=42,
        primary_metric="average_precision",
        model_type="logistic_regression",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        decision_threshold=0.5,
    )


def test_load_metrics_returns_numerical_values(
    tmp_path: Path,
) -> None:
    """Load integer and floating-point metrics as floats."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"average_precision": 0.64, "train_row_count": 100}',
        encoding="utf-8",
    )

    assert track.load_metrics(metrics_path) == {
        "average_precision": 0.64,
        "train_row_count": 100.0,
    }


def test_load_metrics_reads_nested_section(
    tmp_path: Path,
) -> None:
    """Load numerical metrics from a nested report section."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        """
{
"model": {
    "name": "random_forest"
},
"test": {
    "average_precision": 0.71,
    "f1": 0.66,
    "decision_threshold": 0.64
}
}
""".lstrip(),
        encoding="utf-8",
    )

    metrics = track.load_metrics(
        metrics_path,
        section="test",
    )

    assert metrics == {
        "average_precision": 0.71,
        "f1": 0.66,
        "decision_threshold": 0.64,
    }


@pytest.mark.parametrize(
    ("document", "expected_message"),
    [
        (
            '["not", "a", "mapping"]',
            "Metrics document must be a JSON object",
        ),
        (
            '{"status": "passed"}',
            "Metric 'status' must be numerical",
        ),
        (
            '{"enabled": true}',
            "Metric 'enabled' must be numerical",
        ),
    ],
)
def test_load_metrics_rejects_invalid_documents(
    tmp_path: Path,
    document: str,
    expected_message: str,
) -> None:
    """Reject malformed or non-numerical metric documents."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        document,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        track.load_metrics(metrics_path)


def test_load_metrics_rejects_missing_section(
    tmp_path: Path,
) -> None:
    """Reject a requested metrics section that does not exist."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"validation": {"f1": 0.60}}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Metrics section not found: test",
    ):
        track.load_metrics(
            metrics_path,
            section="test",
        )


def test_load_pipeline_rejects_non_pipeline_artifact(
    tmp_path: Path,
) -> None:
    """Reject a serialized object that is not a pipeline."""
    model_path = tmp_path / "invalid.joblib"

    joblib.dump(
        {"not": "a pipeline"},
        model_path,
    )

    with pytest.raises(
        TypeError,
        match="Serialized model must be a scikit-learn Pipeline",
    ):
        track.load_pipeline(model_path)


def test_build_logged_parameters_maps_training_configuration(
    training_config: TrainingConfig,
) -> None:
    """Map the training configuration to MLflow parameters."""
    parameters = track.build_logged_parameters(training_config)

    assert parameters == {
        "target_column": "Revenue",
        "random_seed": 42,
        "primary_metric": "average_precision",
        "model_type": "logistic_regression",
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 1000,
        "decision_threshold": 0.5,
    }


def test_build_logged_parameters_normalizes_missing_class_weight(
    training_config: TrainingConfig,
) -> None:
    """Represent a missing class weight with a stable value."""
    config = TrainingConfig(
        target_column=training_config.target_column,
        random_seed=training_config.random_seed,
        primary_metric=training_config.primary_metric,
        model_type=training_config.model_type,
        class_weight=None,
        solver=training_config.solver,
        max_iter=training_config.max_iter,
        decision_threshold=training_config.decision_threshold,
    )

    parameters = track.build_logged_parameters(config)

    assert parameters["class_weight"] == "none"


def test_build_model_logged_parameters_uses_classifier_configuration(
    training_config: TrainingConfig,
) -> None:
    """Log actual parameters for a non-baseline classifier."""
    classifier = Mock()
    classifier.get_params.return_value = {
        "n_estimators": 300,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
        "unsupported": object(),
    }

    pipeline = Mock(spec=Pipeline)
    pipeline.named_steps = {
        "classifier": classifier,
    }

    parameters = track.build_model_logged_parameters(
        training_config,
        pipeline,
        "random_forest",
    )

    assert parameters == {
        "target_column": "Revenue",
        "random_seed": 42,
        "primary_metric": "average_precision",
        "model_type": "random_forest",
        "decision_threshold": 0.5,
        "classifier__class_weight": "balanced",
        "classifier__max_depth": "none",
        "classifier__n_estimators": 300,
        "classifier__n_jobs": -1,
    }


def test_build_input_example_excludes_target_column(
    tmp_path: Path,
) -> None:
    """Build a bounded example containing only model features."""
    validation_path = tmp_path / "validation.parquet"

    dataframe = pd.DataFrame(
        {
            **{column: range(10) for column in FEATURE_COLUMNS},
            "Revenue": [False, True] * 5,
        }
    )
    dataframe.to_parquet(
        validation_path,
        index=False,
    )

    example = track.build_input_example(
        validation_path,
        "Revenue",
    )

    assert list(example.columns) == list(FEATURE_COLUMNS)
    assert "Revenue" not in example.columns
    assert len(example) == track.INPUT_EXAMPLE_ROW_COUNT


def test_log_dataset_lineage_registers_dataset_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register a dataset with source, target and context."""
    dataset_path = tmp_path / "train.parquet"
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2],
            "Revenue": [False, True],
        }
    )
    dataframe.to_parquet(
        dataset_path,
        index=False,
    )

    dataset = Mock()
    from_pandas_mock = Mock(return_value=dataset)
    log_input_mock = Mock()

    monkeypatch.setattr(
        track,
        "from_pandas",
        from_pandas_mock,
    )
    monkeypatch.setattr(
        track.mlflow,
        "log_input",
        log_input_mock,
    )

    track.log_dataset_lineage(
        dataset_path,
        name="training-dataset",
        context="training",
        target_column="Revenue",
    )

    from_pandas_mock.assert_called_once()

    call = from_pandas_mock.call_args

    pd.testing.assert_frame_equal(
        call.args[0],
        dataframe,
    )
    assert call.kwargs == {
        "source": str(dataset_path),
        "targets": "Revenue",
        "name": "training-dataset",
    }

    log_input_mock.assert_called_once_with(
        dataset,
        context="training",
    )


def test_track_and_register_model_logs_run_and_sets_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_config: TrainingConfig,
) -> None:
    """Log artifacts, register the model and assign its alias."""
    settings = Settings(
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment_name="test-experiment",
        mlflow_registered_model_name="test-model",
        mlflow_model_alias="champion",
    )

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    params_path = tmp_path / "params.yaml"
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    pipeline = Mock(spec=Pipeline)
    pipeline.predict.return_value = [False, True]

    input_example = pd.DataFrame({"feature": [1.0, 2.0]})

    active_run = MagicMock()
    active_run.info.run_id = "run-123"

    run_context = MagicMock()
    run_context.__enter__.return_value = active_run

    model_info = Mock()
    model_info.model_uri = "models:/model-id"

    registered_version = Mock()
    registered_version.version = "1"

    start_run_mock = Mock(return_value=run_context)
    log_model_mock = Mock(return_value=model_info)
    register_model_mock = Mock(return_value=registered_version)

    client = Mock()
    client_factory = Mock(return_value=client)
    log_dataset_lineage_mock = Mock()

    monkeypatch.setattr(
        track,
        "load_training_config",
        Mock(return_value=training_config),
    )
    monkeypatch.setattr(
        track,
        "load_metrics",
        Mock(
            return_value={
                "average_precision": 0.64,
                "decision_threshold": 0.18,
            }
        ),
    )
    monkeypatch.setattr(
        track,
        "load_pipeline",
        Mock(return_value=pipeline),
    )
    monkeypatch.setattr(
        track,
        "build_input_example",
        Mock(return_value=input_example),
    )
    monkeypatch.setattr(
        track,
        "log_dataset_lineage",
        log_dataset_lineage_mock,
    )
    monkeypatch.setattr(
        track,
        "infer_signature",
        Mock(return_value="signature"),
    )
    monkeypatch.setattr(
        track.mlflow,
        "set_tracking_uri",
        Mock(),
    )
    monkeypatch.setattr(
        track.mlflow,
        "set_experiment",
        Mock(),
    )
    monkeypatch.setattr(
        track.mlflow,
        "start_run",
        start_run_mock,
    )
    monkeypatch.setattr(
        track.mlflow,
        "log_params",
        Mock(),
    )
    monkeypatch.setattr(
        track.mlflow,
        "log_metrics",
        Mock(),
    )
    monkeypatch.setattr(
        track.mlflow,
        "log_artifact",
        Mock(),
    )
    monkeypatch.setattr(
        track.mlflow.sklearn,
        "log_model",
        log_model_mock,
    )
    monkeypatch.setattr(
        track.mlflow,
        "register_model",
        register_model_mock,
    )
    monkeypatch.setattr(
        track,
        "MlflowClient",
        client_factory,
    )

    result = track.track_and_register_model(
        model_path=model_path,
        metrics_path=metrics_path,
        params_path=params_path,
        train_path=train_path,
        validation_path=validation_path,
        run_name="baseline-test",
        settings=settings,
    )

    assert result == track.TrackingResult(
        run_id="run-123",
        model_name="test-model",
        model_version="1",
        model_alias="champion",
    )

    start_run_mock.assert_called_once()

    assert log_dataset_lineage_mock.call_count == 2

    log_dataset_lineage_mock.assert_any_call(
        train_path,
        name="training-dataset",
        context="training",
        target_column="Revenue",
    )
    log_dataset_lineage_mock.assert_any_call(
        validation_path,
        name="validation-dataset",
        context="validation",
        target_column="Revenue",
    )

    log_model_mock.assert_called_once()

    register_model_mock.assert_called_once_with(
        model_uri="models:/model-id",
        name="test-model",
        tags={
            "model_family": "logistic_regression",
            "pipeline_stage": "train_baseline",
            "source_run_id": "run-123",
            "decision_threshold": "0.18",
            "prediction_rule": "probability >= decision_threshold",
            "positive_class": "1",
        },
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="test-model",
        alias="champion",
        version="1",
    )

    logged_parameters = track.mlflow.log_params.call_args.args[0]

    assert logged_parameters["decision_threshold"] == 0.18

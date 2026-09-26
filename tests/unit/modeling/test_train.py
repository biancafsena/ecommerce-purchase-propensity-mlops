"""Unit tests for baseline model training orchestration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from purchase_propensity.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from purchase_propensity.modeling.train import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAMS_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    TrainingConfig,
    load_training_config,
    parse_arguments,
    train_baseline,
    validate_training_config,
)


def create_dataset(row_count: int) -> pd.DataFrame:
    """Create a deterministic dataset accepted by the training pipeline."""
    row_ids = np.arange(row_count)
    dataframe = pd.DataFrame(
        {
            feature: row_ids.astype(float) + feature_index
            for feature_index, feature in enumerate(NUMERIC_FEATURES)
        }
    )

    dataframe["Month"] = [("Feb", "Mar", "May", "Nov")[row_id % 4] for row_id in row_ids]
    dataframe["OperatingSystems"] = (row_ids % 3) + 1
    dataframe["Browser"] = (row_ids % 4) + 1
    dataframe["Region"] = (row_ids % 3) + 1
    dataframe["TrafficType"] = (row_ids % 4) + 1
    dataframe["VisitorType"] = [
        ("Returning_Visitor", "New_Visitor")[row_id % 2] for row_id in row_ids
    ]
    dataframe["Weekend"] = row_ids % 2 == 0
    dataframe["Revenue"] = row_ids % 4 == 0

    return dataframe.loc[
        :,
        [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "Revenue"],
    ]


def write_params(params_path: Path) -> None:
    """Write valid baseline training parameters."""
    document = {
        "data": {
            "target_column": "Revenue",
        },
        "modeling": {
            "random_seed": 42,
            "primary_metric": "average_precision",
            "baseline": {
                "model_type": "logistic_regression",
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
            },
        },
        "evaluation": {
            "decision_threshold": 0.50,
        },
    }
    params_path.write_text(
        yaml.safe_dump(document),
        encoding="utf-8",
    )


def test_load_training_config_returns_expected_values(
    tmp_path: Path,
) -> None:
    """Load model and evaluation parameters from YAML."""
    params_path = tmp_path / "params.yaml"
    write_params(params_path)

    config = load_training_config(params_path)

    assert config == TrainingConfig(
        target_column="Revenue",
        random_seed=42,
        primary_metric="average_precision",
        model_type="logistic_regression",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        decision_threshold=0.50,
    )


@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        (
            TrainingConfig(
                "Revenue",
                42,
                "average_precision",
                "random_forest",
                "balanced",
                "lbfgs",
                1000,
                0.50,
            ),
            "Unsupported baseline model type",
        ),
        (
            TrainingConfig(
                "Revenue",
                42,
                "invalid_metric",
                "logistic_regression",
                "balanced",
                "lbfgs",
                1000,
                0.50,
            ),
            "Unsupported primary metric",
        ),
        (
            TrainingConfig(
                "Revenue",
                42,
                "average_precision",
                "logistic_regression",
                "balanced",
                "lbfgs",
                1000,
                1.00,
            ),
            "Decision threshold must be between 0 and 1",
        ),
    ],
)
def test_validate_training_config_rejects_invalid_values(
    config: TrainingConfig,
    expected_message: str,
) -> None:
    """Reject unsupported training configurations."""
    with pytest.raises(ValueError, match=expected_message):
        validate_training_config(config)


def test_train_baseline_persists_model_and_metrics(
    tmp_path: Path,
) -> None:
    """Train the complete pipeline and persist reproducible artifacts."""
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    params_path = tmp_path / "params.yaml"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    create_dataset(40).to_parquet(train_path, index=False)
    create_dataset(20).to_parquet(validation_path, index=False)
    write_params(params_path)

    config = load_training_config(params_path)
    metrics = train_baseline(
        train_path=train_path,
        validation_path=validation_path,
        config=config,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
    )

    persisted_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert persisted_metrics == metrics
    assert metrics["train_row_count"] == pytest.approx(40.0)
    assert metrics["validation_row_count"] == pytest.approx(20.0)
    assert metrics["primary_metric_value"] == pytest.approx(metrics["average_precision"])


def test_parse_arguments_uses_project_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use repository paths when CLI arguments are omitted."""
    monkeypatch.setattr("sys.argv", ["train"])

    arguments = parse_arguments()

    assert arguments.train == DEFAULT_TRAIN_PATH
    assert arguments.validation == DEFAULT_VALIDATION_PATH
    assert arguments.params == DEFAULT_PARAMS_PATH
    assert arguments.model_output == DEFAULT_MODEL_PATH
    assert arguments.metrics_output == DEFAULT_METRICS_PATH

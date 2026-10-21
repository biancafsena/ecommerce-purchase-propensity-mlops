"""Unit tests for candidate model comparison."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import FEATURE_COLUMNS
from purchase_propensity.modeling import compare
from purchase_propensity.modeling.build import TreeModelConfig
from purchase_propensity.modeling.train import TrainingConfig


def write_params(path: Path) -> None:
    """Write a complete model-comparison configuration."""
    path.write_text(
        """
data:
  target_column: Revenue
  random_seed: 42
  train_size: 0.70
  validation_size: 0.15
  test_size: 0.15

modeling:
  random_seed: 42
  primary_metric: average_precision

  baseline:
    model_type: logistic_regression
    class_weight: balanced
    solver: lbfgs
    max_iter: 1000

  candidates:
    random_forest:
      n_estimators: 300
      max_depth: null
      min_samples_leaf: 2
      class_weight: balanced
      n_jobs: -1

    extra_trees:
      n_estimators: 250
      max_depth: 18
      min_samples_leaf: 3
      class_weight: balanced_subsample
      n_jobs: 2

evaluation:
  decision_threshold: 0.50
  threshold_strategy: maximize_f1
""".lstrip(),
        encoding="utf-8",
    )


def build_dataset(row_count: int = 10) -> pd.DataFrame:
    """Build a structurally valid synthetic model dataset."""
    dataframe = pd.DataFrame({column: range(row_count) for column in FEATURE_COLUMNS})
    dataframe["Revenue"] = [index % 2 == 0 for index in range(row_count)]
    return dataframe


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
        decision_threshold=0.50,
    )


@pytest.fixture
def comparison_config(
    training_config: TrainingConfig,
) -> compare.ComparisonConfig:
    """Provide a complete candidate comparison configuration."""
    return compare.ComparisonConfig(
        training=training_config,
        random_forest=TreeModelConfig(
            random_seed=42,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )


def test_load_comparison_config_returns_expected_values(
    tmp_path: Path,
) -> None:
    """Load baseline and tree candidate parameters."""
    params_path = tmp_path / "params.yaml"
    write_params(params_path)

    config = compare.load_comparison_config(params_path)

    assert config.training.target_column == "Revenue"
    assert config.training.primary_metric == "average_precision"

    assert config.random_forest == TreeModelConfig(
        random_seed=42,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
    )


@pytest.mark.parametrize(
    ("value", "expected_message"),
    [
        (
            True,
            "must be an integer or null",
        ),
        (
            "ten",
            "must be an integer or null",
        ),
        (
            0,
            "must be greater than zero",
        ),
        (
            -3,
            "must be greater than zero",
        ),
    ],
)
def test_require_optional_positive_integer_rejects_invalid_values(
    value: object,
    expected_message: str,
) -> None:
    """Reject invalid optional integer parameters."""
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        compare.require_optional_positive_integer(
            {"max_depth": value},
            "max_depth",
        )


def test_require_optional_positive_integer_accepts_null() -> None:
    """Accept null as an unlimited tree depth."""
    result = compare.require_optional_positive_integer(
        {"max_depth": None},
        "max_depth",
    )

    assert result is None


def test_require_class_weight_rejects_non_string() -> None:
    """Reject a class-weight value that is neither string nor null."""
    with pytest.raises(
        ValueError,
        match="must be a string or null",
    ):
        compare.require_class_weight(
            {"class_weight": 42},
        )


def test_build_candidate_pipelines_returns_all_models(
    comparison_config: compare.ComparisonConfig,
) -> None:
    """Build the logistic regression and random forest candidates."""
    pipelines = compare.build_candidate_pipelines(
        comparison_config,
    )

    assert set(pipelines) == {
        "logistic_regression",
        "random_forest",
    }

    assert isinstance(
        pipelines["logistic_regression"],
        Pipeline,
    )
    assert isinstance(
        pipelines["logistic_regression"].named_steps["classifier"],
        LogisticRegression,
    )
    assert isinstance(
        pipelines["random_forest"].named_steps["classifier"],
        RandomForestClassifier,
    )


def test_select_best_model_uses_primary_metric() -> None:
    """Select the model with the highest primary metric."""
    metrics = {
        "logistic_regression": {
            "average_precision": 0.64,
        },
        "random_forest": {
            "average_precision": 0.71,
        },
        "extra_trees": {
            "average_precision": 0.69,
        },
    }

    selected = compare.select_best_model(
        metrics,
        "average_precision",
    )

    assert selected == "random_forest"


def test_select_best_model_uses_name_for_tie_breaking() -> None:
    """Resolve equal scores deterministically by model name."""
    metrics = {
        "random_forest": {
            "average_precision": 0.70,
        },
        "extra_trees": {
            "average_precision": 0.70,
        },
    }

    selected = compare.select_best_model(
        metrics,
        "average_precision",
    )

    assert selected == "extra_trees"


def test_select_best_model_rejects_empty_collection() -> None:
    """Reject model selection without candidate results."""
    with pytest.raises(
        ValueError,
        match="At least one model is required",
    ):
        compare.select_best_model(
            {},
            "average_precision",
        )


def test_select_best_model_rejects_missing_primary_metric() -> None:
    """Reject comparison results missing the primary metric."""
    metrics = {
        "random_forest": {
            "roc_auc": 0.90,
        },
    }

    with pytest.raises(
        ValueError,
        match="Primary metric is missing",
    ):
        compare.select_best_model(
            metrics,
            "average_precision",
        )


def test_save_comparison_report_creates_deterministic_json(
    tmp_path: Path,
) -> None:
    """Persist a sorted and newline-terminated comparison report."""
    output_path = tmp_path / "metrics" / "comparison.json"

    compare.save_comparison_report(
        {
            "selection": {
                "selected_model": "random_forest",
            },
            "models": {
                "random_forest": {
                    "average_precision": 0.71,
                },
            },
        },
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")

    assert content.endswith("\n")
    assert content.index('"models"') < content.index('"selection"')


def test_compare_candidate_models_selects_and_saves_best_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comparison_config: compare.ComparisonConfig,
) -> None:
    """Train candidates and persist the validation winner."""
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    params_path = tmp_path / "params.yaml"
    model_output_path = tmp_path / "models" / "selected.joblib"
    report_output_path = tmp_path / "reports" / "comparison.json"

    build_dataset(row_count=10).to_parquet(
        train_path,
        index=False,
    )
    build_dataset(row_count=6).to_parquet(
        validation_path,
        index=False,
    )

    logistic_pipeline = Mock(spec=Pipeline)
    random_forest_pipeline = Mock(spec=Pipeline)

    logistic_pipeline.predict_proba.return_value = np.array(
        [
            [0.80, 0.20],
            [0.70, 0.30],
            [0.40, 0.60],
            [0.30, 0.70],
            [0.20, 0.80],
            [0.10, 0.90],
        ]
    )
    random_forest_pipeline.predict_proba.return_value = np.array(
        [
            [0.90, 0.10],
            [0.75, 0.25],
            [0.35, 0.65],
            [0.25, 0.75],
            [0.15, 0.85],
            [0.05, 0.95],
        ]
    )

    pipelines = {
        "logistic_regression": logistic_pipeline,
        "random_forest": random_forest_pipeline,
    }

    evaluation_mock = Mock(
        side_effect=[
            {
                "average_precision": 0.64,
                "roc_auc": 0.89,
            },
            {
                "average_precision": 0.72,
                "roc_auc": 0.92,
            },
            {
                "average_precision": 0.69,
                "roc_auc": 0.91,
            },
        ]
    )
    dump_mock = Mock()
    save_report_mock = Mock()

    monkeypatch.setattr(
        compare,
        "load_comparison_config",
        Mock(return_value=comparison_config),
    )
    monkeypatch.setattr(
        compare,
        "build_candidate_pipelines",
        Mock(return_value=pipelines),
    )
    monkeypatch.setattr(
        compare,
        "evaluate_binary_classifier",
        evaluation_mock,
    )
    monkeypatch.setattr(
        compare.joblib,
        "dump",
        dump_mock,
    )
    monkeypatch.setattr(
        compare,
        "save_comparison_report",
        save_report_mock,
    )

    report = compare.compare_candidate_models(
        train_path=train_path,
        validation_path=validation_path,
        params_path=params_path,
        model_output_path=model_output_path,
        report_output_path=report_output_path,
    )

    logistic_pipeline.fit.assert_called_once()
    random_forest_pipeline.fit.assert_called_once()

    assert evaluation_mock.call_count == 2

    assert report["selection"] == {
        "primary_metric": "average_precision",
        "selected_model": "random_forest",
        "selected_score": 0.72,
    }
    assert report["row_counts"] == {
        "train": 10.0,
        "validation": 6.0,
    }

    dump_mock.assert_called_once_with(
        random_forest_pipeline,
        model_output_path,
    )
    save_report_mock.assert_called_once_with(
        report,
        report_output_path,
    )

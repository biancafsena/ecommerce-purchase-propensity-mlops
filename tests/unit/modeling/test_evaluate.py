"""Unit tests for baseline threshold selection and final evaluation."""

from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import FEATURE_COLUMNS
from purchase_propensity.modeling import evaluate


def write_params(
    path: Path,
    threshold_strategy: str = "maximize_f1",
) -> None:
    """Write a complete test configuration."""
    path.write_text(
        f"""
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

evaluation:
  decision_threshold: 0.50
  threshold_strategy: {threshold_strategy}
""".lstrip(),
        encoding="utf-8",
    )


def build_dataset(row_count: int = 10) -> pd.DataFrame:
    """Build a structurally valid synthetic model dataset."""
    dataframe = pd.DataFrame({column: range(row_count) for column in FEATURE_COLUMNS})
    dataframe["Revenue"] = [index % 2 == 0 for index in range(row_count)]
    return dataframe


def test_load_evaluation_config_returns_expected_values(
    tmp_path: Path,
) -> None:
    """Load the target and threshold strategy."""
    params_path = tmp_path / "params.yaml"
    write_params(params_path)

    config = evaluate.load_evaluation_config(params_path)

    assert config == evaluate.EvaluationConfig(
        target_column="Revenue",
        threshold_strategy="maximize_f1",
    )


def test_load_evaluation_config_rejects_unknown_strategy(
    tmp_path: Path,
) -> None:
    """Reject an unsupported threshold strategy."""
    params_path = tmp_path / "params.yaml"
    write_params(
        params_path,
        threshold_strategy="use_test_set",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported threshold strategy",
    ):
        evaluate.load_evaluation_config(params_path)


def test_load_pipeline_rejects_non_pipeline_artifact(
    tmp_path: Path,
) -> None:
    """Reject a serialized object that is not a pipeline."""
    model_path = tmp_path / "invalid.joblib"
    joblib.dump(
        {"invalid": "model"},
        model_path,
    )

    with pytest.raises(
        TypeError,
        match="Serialized model must be a scikit-learn Pipeline",
    ):
        evaluate.load_pipeline(model_path)


def test_predict_probabilities_returns_aligned_target(
    tmp_path: Path,
) -> None:
    """Calculate positive probabilities while preserving row alignment."""
    dataset_path = tmp_path / "dataset.parquet"
    dataframe = build_dataset(row_count=4)
    dataframe.to_parquet(
        dataset_path,
        index=False,
    )

    pipeline = Mock(spec=Pipeline)
    pipeline.predict_proba.return_value = np.array(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.20, 0.80],
            [0.05, 0.95],
        ]
    )

    target, probabilities = evaluate.predict_probabilities(
        pipeline,
        dataset_path,
        "Revenue",
    )

    assert target.tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert probabilities.tolist() == pytest.approx([0.10, 0.30, 0.80, 0.95])
    assert probabilities.name == "purchase_probability"


def test_save_json_report_creates_deterministic_document(
    tmp_path: Path,
) -> None:
    """Persist a sorted and newline-terminated JSON report."""
    output_path = tmp_path / "metrics" / "report.json"

    evaluate.save_json_report(
        {
            "test": {"f1": 0.70},
            "threshold": 0.40,
        },
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")

    assert content.endswith("\n")
    assert content.index('"test"') < content.index('"threshold"')


def test_save_evaluation_figures_creates_all_images(
    tmp_path: Path,
) -> None:
    """Generate every expected final evaluation figure."""
    labels = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
    )
    probabilities = pd.Series(
        [0.05, 0.15, 0.35, 0.60, 0.40, 0.70, 0.85, 0.95],
    )
    output_directory = tmp_path / "figures"

    evaluate.save_evaluation_figures(
        labels,
        probabilities,
        threshold=0.50,
        output_directory=output_directory,
    )

    expected_files = {
        "precision_recall_curve.png",
        "roc_curve.png",
        "calibration_curve.png",
        "confusion_matrix.png",
    }

    assert {path.name for path in output_directory.iterdir()} == expected_files

    for filename in expected_files:
        assert (output_directory / filename).stat().st_size > 0


def test_evaluate_baseline_selects_threshold_only_from_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select on validation before applying the cutoff to test data."""
    model_path = tmp_path / "model.joblib"
    validation_path = tmp_path / "validation.parquet"
    test_path = tmp_path / "test.parquet"
    params_path = tmp_path / "params.yaml"
    report_path = tmp_path / "report.json"
    figures_directory = tmp_path / "figures"

    config = evaluate.EvaluationConfig(
        target_column="Revenue",
        threshold_strategy="maximize_f1",
    )
    pipeline = Mock(spec=Pipeline)

    validation_target = pd.Series([0, 0, 1, 1])
    validation_probabilities = pd.Series([0.10, 0.40, 0.45, 0.90])
    test_target = pd.Series([0, 1, 0, 1])
    test_probabilities = pd.Series([0.20, 0.80, 0.30, 0.70])

    validation_metrics = {
        "f1": 0.80,
        "average_precision": 0.90,
    }
    test_metrics = {
        "f1": 0.75,
        "average_precision": 0.85,
    }

    predict_mock = Mock(
        side_effect=[
            (
                validation_target,
                validation_probabilities,
            ),
            (
                test_target,
                test_probabilities,
            ),
        ]
    )
    select_threshold_mock = Mock(return_value=0.45)
    evaluate_metrics_mock = Mock(
        side_effect=[
            validation_metrics,
            test_metrics,
        ]
    )
    save_report_mock = Mock()
    save_figures_mock = Mock()

    monkeypatch.setattr(
        evaluate,
        "load_evaluation_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        evaluate,
        "load_pipeline",
        Mock(return_value=pipeline),
    )
    monkeypatch.setattr(
        evaluate,
        "predict_probabilities",
        predict_mock,
    )
    monkeypatch.setattr(
        evaluate,
        "select_f1_threshold",
        select_threshold_mock,
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_binary_classifier",
        evaluate_metrics_mock,
    )
    monkeypatch.setattr(
        evaluate,
        "save_json_report",
        save_report_mock,
    )
    monkeypatch.setattr(
        evaluate,
        "save_evaluation_figures",
        save_figures_mock,
    )

    report = evaluate.evaluate_baseline(
        model_path=model_path,
        validation_path=validation_path,
        test_path=test_path,
        params_path=params_path,
        report_path=report_path,
        figures_directory=figures_directory,
    )

    select_threshold_mock.assert_called_once_with(
        validation_target,
        validation_probabilities,
    )

    assert evaluate_metrics_mock.call_count == 2

    validation_call = evaluate_metrics_mock.call_args_list[0]
    test_call = evaluate_metrics_mock.call_args_list[1]

    pd.testing.assert_series_equal(
        validation_call.args[0],
        validation_target,
    )
    pd.testing.assert_series_equal(
        validation_call.args[1],
        validation_probabilities,
    )
    assert validation_call.args[2] == pytest.approx(0.45)

    pd.testing.assert_series_equal(
        test_call.args[0],
        test_target,
    )
    pd.testing.assert_series_equal(
        test_call.args[1],
        test_probabilities,
    )
    assert test_call.args[2] == pytest.approx(0.45)

    assert report["threshold_selection"] == {
        "strategy": "maximize_f1",
        "selected_threshold": 0.45,
        "validation_f1": 0.80,
    }

    assert report["model"] == {
        "name": "logistic_regression",
        "artifact_path": model_path.as_posix(),
    }
    assert report["validation"] == validation_metrics
    assert report["test"] == test_metrics
    assert report["row_counts"] == {
        "validation": 4,
        "test": 4,
    }

    save_report_mock.assert_called_once_with(
        report,
        report_path,
    )
    save_figures_mock.assert_called_once_with(
        test_target,
        test_probabilities,
        0.45,
        figures_directory,
    )

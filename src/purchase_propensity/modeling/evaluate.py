"""Evaluate a trained model on validation and test datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import split_features_and_target
from purchase_propensity.modeling.metrics import (
    evaluate_binary_classifier,
    select_f1_threshold,
)
from purchase_propensity.modeling.train import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAMS_PATH,
    DEFAULT_VALIDATION_PATH,
    load_training_config,
    require_mapping,
    require_value,
)

DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_REPORT_PATH = Path("reports/metrics/baseline_test_metrics.json")
DEFAULT_FIGURES_DIRECTORY = Path("reports/figures/baseline")
SUPPORTED_THRESHOLD_STRATEGY = "maximize_f1"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration required for final model evaluation."""

    target_column: str
    threshold_strategy: str


def load_evaluation_config(
    params_path: Path,
) -> EvaluationConfig:
    """Load and validate final evaluation configuration."""
    training_config = load_training_config(params_path)

    raw_document: object = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    document = require_mapping(
        raw_document,
        "root",
    )
    evaluation_section = require_mapping(
        document.get("evaluation"),
        "evaluation",
    )
    threshold_strategy = require_value(
        evaluation_section,
        "threshold_strategy",
        str,
    )

    if threshold_strategy != SUPPORTED_THRESHOLD_STRATEGY:
        raise ValueError(f"Unsupported threshold strategy: {threshold_strategy}")

    return EvaluationConfig(
        target_column=training_config.target_column,
        threshold_strategy=threshold_strategy,
    )


def load_pipeline(
    model_path: Path,
) -> Pipeline:
    """Load and validate the serialized model pipeline."""
    model: object = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Serialized model must be a scikit-learn Pipeline.")

    return model


def predict_probabilities(
    pipeline: Pipeline,
    dataset_path: Path,
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Load one dataset and calculate positive-class probabilities."""
    dataframe = pd.read_parquet(dataset_path)
    features, target = split_features_and_target(
        dataframe,
        target_column,
    )

    probabilities = pd.Series(
        pipeline.predict_proba(features)[:, 1],
        index=target.index,
        name="purchase_probability",
    )

    return target, probabilities


def save_json_report(
    report: dict[str, object],
    output_path: Path,
) -> None:
    """Persist the evaluation report as deterministic JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_precision_recall_figure(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    """Save the test-set precision-recall curve."""
    precision, recall, _ = precision_recall_curve(
        labels,
        probabilities,
    )
    threshold_metrics = evaluate_binary_classifier(
        labels,
        probabilities,
        threshold,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6),
    )
    axis.plot(
        recall,
        precision,
        color="#0057B8",
        linewidth=2,
        label="Test set",
    )
    axis.scatter(
        threshold_metrics["recall"],
        threshold_metrics["precision"],
        color="#D62728",
        s=80,
        zorder=3,
        label=f"Selected threshold = {threshold:.3f}",
    )
    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_roc_figure(
    labels: pd.Series,
    probabilities: pd.Series,
    output_path: Path,
) -> None:
    """Save the test-set ROC curve."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        probabilities,
    )
    metrics = evaluate_binary_classifier(
        labels,
        probabilities,
        threshold=0.50,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6),
    )
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#0057B8",
        linewidth=2,
        label=f"ROC AUC = {metrics['roc_auc']:.3f}",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#666666",
        label="Random classifier",
    )
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_calibration_figure(
    labels: pd.Series,
    probabilities: pd.Series,
    output_path: Path,
) -> None:
    """Save the test-set probability calibration curve."""
    observed_rate, predicted_probability = calibration_curve(
        labels,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    figure, axis = plt.subplots(
        figsize=(8, 6),
    )
    axis.plot(
        predicted_probability,
        observed_rate,
        marker="o",
        color="#0057B8",
        linewidth=2,
        label="Evaluated model",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#666666",
        label="Perfect calibration",
    )
    axis.set_title("Calibration Curve")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Purchase Rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix_figure(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    """Save the test-set confusion matrix."""
    predictions = (probabilities.to_numpy() >= threshold).astype(int)
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix — Threshold {threshold:.3f}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_evaluation_figures(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    output_directory: Path,
) -> None:
    """Create all final test-set evaluation figures."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_precision_recall_figure(
        labels,
        probabilities,
        threshold,
        output_directory / "precision_recall_curve.png",
    )
    save_roc_figure(
        labels,
        probabilities,
        output_directory / "roc_curve.png",
    )
    save_calibration_figure(
        labels,
        probabilities,
        output_directory / "calibration_curve.png",
    )
    save_confusion_matrix_figure(
        labels,
        probabilities,
        threshold,
        output_directory / "confusion_matrix.png",
    )


def evaluate_model(
    *,
    model_name: str,
    model_path: Path,
    validation_path: Path,
    test_path: Path,
    params_path: Path,
    report_path: Path,
    figures_directory: Path,
) -> dict[str, object]:
    """Select a validation threshold and evaluate once on test data."""
    config = load_evaluation_config(params_path)
    pipeline = load_pipeline(model_path)

    validation_target, validation_probabilities = predict_probabilities(
        pipeline,
        validation_path,
        config.target_column,
    )
    test_target, test_probabilities = predict_probabilities(
        pipeline,
        test_path,
        config.target_column,
    )

    selected_threshold = select_f1_threshold(
        validation_target,
        validation_probabilities,
    )

    validation_metrics = evaluate_binary_classifier(
        validation_target,
        validation_probabilities,
        selected_threshold,
    )
    test_metrics = evaluate_binary_classifier(
        test_target,
        test_probabilities,
        selected_threshold,
    )

    report: dict[str, object] = {
        "model": {
            "name": model_name,
            "artifact_path": model_path.as_posix(),
        },
        "threshold_selection": {
            "strategy": config.threshold_strategy,
            "selected_threshold": selected_threshold,
            "validation_f1": validation_metrics["f1"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "row_counts": {
            "validation": float(len(validation_target)),
            "test": float(len(test_target)),
        },
    }

    save_json_report(
        report,
        report_path,
    )
    save_evaluation_figures(
        test_target,
        test_probabilities,
        selected_threshold,
        figures_directory,
    )

    return report


def evaluate_baseline(
    *,
    model_path: Path,
    validation_path: Path,
    test_path: Path,
    params_path: Path,
    report_path: Path,
    figures_directory: Path,
) -> dict[str, object]:
    """Evaluate the historical logistic regression baseline."""
    return evaluate_model(
        model_name="logistic_regression",
        model_path=model_path,
        validation_path=validation_path,
        test_path=test_path,
        params_path=params_path,
        report_path=report_path,
        figures_directory=figures_directory,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description=("Select a validation threshold and evaluate a trained model on test data.")
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regression",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--figures-directory",
        type=Path,
        default=DEFAULT_FIGURES_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    """Run threshold selection and final test evaluation."""
    arguments = parse_arguments()

    report = evaluate_model(
        model_name=arguments.model_name,
        model_path=arguments.model,
        validation_path=arguments.validation,
        test_path=arguments.test,
        params_path=arguments.params,
        report_path=arguments.report,
        figures_directory=arguments.figures_directory,
    )

    threshold_selection = report["threshold_selection"]
    test_metrics = report["test"]

    if not isinstance(threshold_selection, dict):
        raise TypeError("Threshold selection report must be a mapping.")
    if not isinstance(test_metrics, dict):
        raise TypeError("Test metrics report must be a mapping.")

    print(
        "Model evaluation completed: "
        f"model={arguments.model_name}, "
        f"threshold={threshold_selection['selected_threshold']:.4f}, "
        f"test_average_precision="
        f"{test_metrics['average_precision']:.4f}, "
        f"test_f1={test_metrics['f1']:.4f}."
    )


if __name__ == "__main__":
    main()

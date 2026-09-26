"""Train and evaluate the logistic regression baseline model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import split_features_and_target
from purchase_propensity.modeling.build import (
    BaselineModelConfig,
    build_baseline_pipeline,
)
from purchase_propensity.modeling.metrics import evaluate_binary_classifier

DEFAULT_TRAIN_PATH = Path("data/processed/train.parquet")
DEFAULT_VALIDATION_PATH = Path("data/processed/validation.parquet")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_MODEL_PATH = Path("models/baseline_logistic_regression.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics/baseline_validation_metrics.json")

ValueType = TypeVar("ValueType")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Complete configuration for baseline training and evaluation."""

    target_column: str
    random_seed: int
    primary_metric: str
    model_type: str
    class_weight: str | None
    solver: str
    max_iter: int
    decision_threshold: float


def require_mapping(
    value: object,
    section_name: str,
) -> dict[str, object]:
    """Require a YAML value to be a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section must be a mapping: {section_name}")

    return cast(dict[str, object], value)


def require_value(
    mapping: dict[str, object],
    key: str,
    expected_type: type[ValueType],
) -> ValueType:
    """Read a required configuration value with runtime type validation."""
    value = mapping.get(key)

    if not isinstance(value, expected_type):
        expected_name = expected_type.__name__
        raise ValueError(f"Configuration value '{key}' must be of type {expected_name}.")

    return value


def require_number(
    mapping: dict[str, object],
    key: str,
) -> float:
    """Read a required numerical configuration value."""
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration value '{key}' must be numerical.")

    return float(value)


def load_training_config(params_path: Path) -> TrainingConfig:
    """Load baseline training parameters from YAML."""
    raw_document: object = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    document = require_mapping(raw_document, "root")
    data_section = require_mapping(document.get("data"), "data")
    modeling_section = require_mapping(
        document.get("modeling"),
        "modeling",
    )
    baseline_section = require_mapping(
        modeling_section.get("baseline"),
        "modeling.baseline",
    )
    evaluation_section = require_mapping(
        document.get("evaluation"),
        "evaluation",
    )

    class_weight_value = baseline_section.get("class_weight")
    if class_weight_value is not None and not isinstance(
        class_weight_value,
        str,
    ):
        raise ValueError("Configuration value 'class_weight' must be a string or null.")

    config = TrainingConfig(
        target_column=require_value(
            data_section,
            "target_column",
            str,
        ),
        random_seed=require_value(
            modeling_section,
            "random_seed",
            int,
        ),
        primary_metric=require_value(
            modeling_section,
            "primary_metric",
            str,
        ),
        model_type=require_value(
            baseline_section,
            "model_type",
            str,
        ),
        class_weight=class_weight_value,
        solver=require_value(
            baseline_section,
            "solver",
            str,
        ),
        max_iter=require_value(
            baseline_section,
            "max_iter",
            int,
        ),
        decision_threshold=require_number(
            evaluation_section,
            "decision_threshold",
        ),
    )

    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    """Validate parameters specific to the baseline training workflow."""
    if config.model_type != "logistic_regression":
        raise ValueError(f"Unsupported baseline model type: {config.model_type}")

    supported_metrics = {
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "log_loss",
        "brier_score",
    }
    if config.primary_metric not in supported_metrics:
        raise ValueError(f"Unsupported primary metric: {config.primary_metric}")

    BaselineModelConfig(
        random_seed=config.random_seed,
        class_weight=config.class_weight,
        solver=config.solver,
        max_iter=config.max_iter,
    )

    if not 0.0 < config.decision_threshold < 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")


def build_training_pipeline(config: TrainingConfig) -> Pipeline:
    """Build the baseline pipeline from the training configuration."""
    model_config = BaselineModelConfig(
        random_seed=config.random_seed,
        class_weight=config.class_weight,
        solver=config.solver,
        max_iter=config.max_iter,
    )
    return build_baseline_pipeline(model_config)


def save_metrics(
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Persist evaluation metrics as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_baseline(
    train_path: Path,
    validation_path: Path,
    config: TrainingConfig,
    model_output_path: Path,
    metrics_output_path: Path,
) -> dict[str, float]:
    """Train the baseline and evaluate it on the validation partition."""
    train_dataframe = pd.read_parquet(train_path)
    validation_dataframe = pd.read_parquet(validation_path)

    train_features, train_target = split_features_and_target(
        train_dataframe,
        config.target_column,
    )
    validation_features, validation_target = split_features_and_target(
        validation_dataframe,
        config.target_column,
    )

    pipeline = build_training_pipeline(config)
    pipeline.fit(train_features, train_target)

    validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
    metrics = evaluate_binary_classifier(
        validation_target,
        validation_probabilities,
        config.decision_threshold,
    )
    metrics["train_row_count"] = float(len(train_dataframe))
    metrics["validation_row_count"] = float(len(validation_dataframe))
    metrics["primary_metric_value"] = metrics[config.primary_metric]

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    save_metrics(metrics, metrics_output_path)

    return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for baseline training."""
    parser = argparse.ArgumentParser(description="Train the purchase propensity baseline model.")
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline training from the command line."""
    arguments = parse_arguments()
    config = load_training_config(arguments.params)
    metrics = train_baseline(
        train_path=arguments.train,
        validation_path=arguments.validation,
        config=config,
        model_output_path=arguments.model_output,
        metrics_output_path=arguments.metrics_output,
    )

    print(
        "Baseline training completed: "
        f"average_precision={metrics['average_precision']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"f1={metrics['f1']:.4f}."
    )


if __name__ == "__main__":
    main()

"""Track and register the trained baseline model with MLflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.data import from_pandas  # type: ignore[attr-defined]
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from purchase_propensity.config import Settings, get_settings
from purchase_propensity.features.build import split_features_and_target
from purchase_propensity.modeling.train import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAMS_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    TrainingConfig,
    load_training_config,
)

DEFAULT_RUN_NAME = "baseline-logistic-regression"
DEFAULT_LOGGED_MODEL_NAME = "model"
INPUT_EXAMPLE_ROW_COUNT = 5


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Identifiers produced while tracking and registering a model."""

    run_id: str
    model_name: str
    model_version: str
    model_alias: str


def load_metrics(metrics_path: Path) -> dict[str, float]:
    """Load and validate numerical metrics from JSON."""
    raw_metrics: object = json.loads(metrics_path.read_text(encoding="utf-8"))

    if not isinstance(raw_metrics, dict):
        raise ValueError("Metrics document must be a JSON object.")

    metrics: dict[str, float] = {}

    for key, value in raw_metrics.items():
        if not isinstance(key, str):
            raise ValueError("Metric names must be strings.")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric '{key}' must be numerical.")

        metrics[key] = float(value)

    return metrics


def load_pipeline(model_path: Path) -> Pipeline:
    """Load and validate the serialized scikit-learn pipeline."""
    model: object = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Serialized model must be a scikit-learn Pipeline.")

    return model


def build_logged_parameters(
    config: TrainingConfig,
) -> dict[str, str | int | float]:
    """Build stable MLflow parameters from the training configuration."""
    return {
        "target_column": config.target_column,
        "random_seed": config.random_seed,
        "primary_metric": config.primary_metric,
        "model_type": config.model_type,
        "class_weight": config.class_weight or "none",
        "solver": config.solver,
        "max_iter": config.max_iter,
        "decision_threshold": config.decision_threshold,
    }


def build_input_example(
    validation_path: Path,
    target_column: str,
) -> pd.DataFrame:
    """Build a representative input example from validation data."""
    validation_dataframe = pd.read_parquet(validation_path)

    validation_features, _ = split_features_and_target(
        validation_dataframe,
        target_column,
    )

    input_example = validation_features.head(INPUT_EXAMPLE_ROW_COUNT).copy()

    integer_columns = input_example.select_dtypes(include="integer").columns

    input_example[integer_columns] = input_example[integer_columns].astype("float64")

    return input_example


def log_dataset_lineage(
    dataset_path: Path,
    *,
    name: str,
    context: str,
    target_column: str,
) -> None:
    """Log a versioned dataset input with its role in the workflow."""
    dataframe = pd.read_parquet(dataset_path)

    dataset = from_pandas(
        dataframe,
        source=str(dataset_path),
        targets=target_column,
        name=name,
    )

    mlflow.log_input(
        dataset,
        context=context,
    )


def track_and_register_model(
    *,
    model_path: Path,
    metrics_path: Path,
    params_path: Path,
    train_path: Path,
    validation_path: Path,
    run_name: str,
    settings: Settings,
) -> TrackingResult:
    """Log the baseline run and register its model version."""
    config = load_training_config(params_path)
    metrics = load_metrics(metrics_path)
    pipeline = load_pipeline(model_path)

    input_example = build_input_example(
        validation_path,
        config.target_column,
    )

    predictions = pipeline.predict(input_example)
    signature = infer_signature(
        input_example,
        predictions,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    tags = {
        "project": settings.project_name,
        "environment": settings.environment,
        "pipeline_stage": "train_baseline",
        "model_family": "logistic_regression",
        "dataset_split": "train-validation",
    }

    with mlflow.start_run(
        run_name=run_name,
        tags=tags,
    ) as active_run:
        log_dataset_lineage(
            train_path,
            name="training-dataset",
            context="training",
            target_column=config.target_column,
        )

        log_dataset_lineage(
            validation_path,
            name="validation-dataset",
            context="validation",
            target_column=config.target_column,
        )

        mlflow.log_params(build_logged_parameters(config))
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(params_path),
            artifact_path="configuration",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=DEFAULT_LOGGED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
            serialization_format=(mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS),
            skops_trusted_types=["numpy.dtype"],
        )

        run_id = active_run.info.run_id

    registered_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=settings.mlflow_registered_model_name,
        tags={
            "model_family": "logistic_regression",
            "source_run_id": run_id,
        },
    )

    model_version = registered_version.version

    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
    )
    client.set_registered_model_alias(
        name=settings.mlflow_registered_model_name,
        alias=settings.mlflow_model_alias,
        version=model_version,
    )

    return TrackingResult(
        run_id=run_id,
        model_name=settings.mlflow_registered_model_name,
        model_version=model_version,
        model_alias=settings.mlflow_model_alias,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for MLflow tracking."""
    parser = argparse.ArgumentParser(
        description=("Track and register the trained baseline model with MLflow.")
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
    )
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
        "--run-name",
        default=DEFAULT_RUN_NAME,
    )

    return parser.parse_args()


def main() -> None:
    """Track and register the baseline from the command line."""
    arguments = parse_arguments()

    result = track_and_register_model(
        model_path=arguments.model,
        metrics_path=arguments.metrics,
        params_path=arguments.params,
        train_path=arguments.train,
        validation_path=arguments.validation,
        run_name=arguments.run_name,
        settings=get_settings(),
    )

    print(
        "MLflow registration completed: "
        f"run_id={result.run_id}, "
        f"model={result.model_name}, "
        f"version={result.model_version}, "
        f"alias={result.model_alias}."
    )


if __name__ == "__main__":
    main()

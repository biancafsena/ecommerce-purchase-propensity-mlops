"""Track candidate classification models with MLflow."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline


def log_candidate_run(
    *,
    model_name: str,
    pipeline: Pipeline,
    metrics: dict[str, float],
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Log a candidate model and its validation metrics."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    classifier = pipeline.named_steps["classifier"]

    parameters = {
        key: value if value is not None else "none"
        for key, value in classifier.get_params(deep=False).items()
    }

    with mlflow.start_run(
        run_name=f"candidate-{model_name}",
        tags={
            "model_family": model_name,
            "dataset_split": "validation",
            "pipeline_stage": "model_comparison",
        },
    ) as active_run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=(mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS),
            skops_trusted_types=["numpy.dtype"],
        )

        run_id = str(active_run.info.run_id)

    return run_id

"""Compare candidate models using the validation partition."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import split_features_and_target
from purchase_propensity.modeling.build import (
    BaselineModelConfig,
    TreeModelConfig,
    build_baseline_pipeline,
    build_extra_trees_pipeline,
    build_random_forest_pipeline,
)
from purchase_propensity.modeling.metrics import evaluate_binary_classifier
from purchase_propensity.modeling.train import (
    TrainingConfig,
    load_training_config,
    require_mapping,
    require_value,
)

DEFAULT_TRAIN_PATH = Path("data/processed/train.parquet")
DEFAULT_VALIDATION_PATH = Path("data/processed/validation.parquet")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_MODEL_PATH = Path("models/selected_model.joblib")
DEFAULT_REPORT_PATH = Path("reports/metrics/model_comparison.json")


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Configuration for candidate model comparison."""

    training: TrainingConfig
    random_forest: TreeModelConfig
    extra_trees: TreeModelConfig


def require_optional_positive_integer(
    mapping: dict[str, object],
    key: str,
) -> int | None:
    """Read an optional positive integer configuration value."""
    value = mapping.get(key)

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration value '{key}' must be an integer or null.")

    if value <= 0:
        raise ValueError(f"Configuration value '{key}' must be greater than zero.")

    return value


def require_class_weight(
    mapping: dict[str, object],
) -> str | None:
    """Read a nullable class-weight configuration value."""
    value = mapping.get("class_weight")

    if value is not None and not isinstance(value, str):
        raise ValueError("Configuration value 'class_weight' must be a string or null.")

    return value


def load_tree_model_config(
    section: dict[str, object],
    random_seed: int,
) -> TreeModelConfig:
    """Build and validate a tree ensemble configuration."""
    return TreeModelConfig(
        random_seed=random_seed,
        n_estimators=require_value(
            section,
            "n_estimators",
            int,
        ),
        max_depth=require_optional_positive_integer(
            section,
            "max_depth",
        ),
        min_samples_leaf=require_value(
            section,
            "min_samples_leaf",
            int,
        ),
        class_weight=require_class_weight(section),
        n_jobs=require_value(
            section,
            "n_jobs",
            int,
        ),
    )


def load_comparison_config(
    params_path: Path,
) -> ComparisonConfig:
    """Load baseline and candidate model configurations."""
    training_config = load_training_config(params_path)

    raw_document: object = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    document = require_mapping(
        raw_document,
        "root",
    )
    modeling_section = require_mapping(
        document.get("modeling"),
        "modeling",
    )
    candidates_section = require_mapping(
        modeling_section.get("candidates"),
        "modeling.candidates",
    )
    random_forest_section = require_mapping(
        candidates_section.get("random_forest"),
        "modeling.candidates.random_forest",
    )
    extra_trees_section = require_mapping(
        candidates_section.get("extra_trees"),
        "modeling.candidates.extra_trees",
    )

    return ComparisonConfig(
        training=training_config,
        random_forest=load_tree_model_config(
            random_forest_section,
            training_config.random_seed,
        ),
        extra_trees=load_tree_model_config(
            extra_trees_section,
            training_config.random_seed,
        ),
    )


def build_candidate_pipelines(
    config: ComparisonConfig,
) -> dict[str, Pipeline]:
    """Build every model participating in the comparison."""
    baseline_config = BaselineModelConfig(
        random_seed=config.training.random_seed,
        class_weight=config.training.class_weight,
        solver=config.training.solver,
        max_iter=config.training.max_iter,
    )

    return {
        "logistic_regression": build_baseline_pipeline(baseline_config),
        "random_forest": build_random_forest_pipeline(config.random_forest),
        "extra_trees": build_extra_trees_pipeline(config.extra_trees),
    }


def select_best_model(
    metrics_by_model: dict[str, dict[str, float]],
    primary_metric: str,
) -> str:
    """Select the highest-scoring model with deterministic tie-breaking."""
    if not metrics_by_model:
        raise ValueError("At least one model is required for selection.")

    missing_metric_models = [
        model_name
        for model_name, metrics in metrics_by_model.items()
        if primary_metric not in metrics
    ]
    if missing_metric_models:
        raise ValueError(f"Primary metric is missing for models: {sorted(missing_metric_models)}")

    ranking = sorted(
        metrics_by_model,
        key=lambda model_name: (
            -metrics_by_model[model_name][primary_metric],
            model_name,
        ),
    )
    return ranking[0]


def save_comparison_report(
    report: dict[str, object],
    output_path: Path,
) -> None:
    """Persist the model comparison as deterministic JSON."""
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


def compare_candidate_models(
    *,
    train_path: Path,
    validation_path: Path,
    params_path: Path,
    model_output_path: Path,
    report_output_path: Path,
) -> dict[str, object]:
    """Train, compare and persist the best validation model."""
    config = load_comparison_config(params_path)

    train_dataframe = pd.read_parquet(train_path)
    validation_dataframe = pd.read_parquet(validation_path)

    train_features, train_target = split_features_and_target(
        train_dataframe,
        config.training.target_column,
    )
    validation_features, validation_target = split_features_and_target(
        validation_dataframe,
        config.training.target_column,
    )

    pipelines = build_candidate_pipelines(config)
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, pipeline in pipelines.items():
        pipeline.fit(
            train_features,
            train_target,
        )
        probabilities = pipeline.predict_proba(validation_features)[:, 1]

        metrics = evaluate_binary_classifier(
            validation_target,
            probabilities,
            config.training.decision_threshold,
        )
        metrics["primary_metric_value"] = metrics[config.training.primary_metric]
        metrics_by_model[model_name] = metrics

    selected_model_name = select_best_model(
        metrics_by_model,
        config.training.primary_metric,
    )
    selected_pipeline = pipelines[selected_model_name]
    selected_score = metrics_by_model[selected_model_name][config.training.primary_metric]

    report: dict[str, object] = {
        "models": metrics_by_model,
        "row_counts": {
            "train": float(len(train_dataframe)),
            "validation": float(len(validation_dataframe)),
        },
        "selection": {
            "primary_metric": config.training.primary_metric,
            "selected_model": selected_model_name,
            "selected_score": selected_score,
        },
    }

    model_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        selected_pipeline,
        model_output_path,
    )
    save_comparison_report(
        report,
        report_output_path,
    )

    return report


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for model comparison."""
    parser = argparse.ArgumentParser(
        description=("Compare purchase propensity models using the validation partition.")
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
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    """Run candidate model comparison from the command line."""
    arguments = parse_arguments()
    report = compare_candidate_models(
        train_path=arguments.train,
        validation_path=arguments.validation,
        params_path=arguments.params,
        model_output_path=arguments.model_output,
        report_output_path=arguments.report_output,
    )

    selection = report["selection"]
    if not isinstance(selection, dict):
        raise TypeError("Comparison report selection must be a mapping.")

    selected_model = selection["selected_model"]
    selected_score = selection["selected_score"]

    print(
        "Model comparison completed: "
        f"selected_model={selected_model}, "
        f"average_precision={selected_score:.4f}."
    )


if __name__ == "__main__":
    main()

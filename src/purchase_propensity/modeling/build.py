"""Build purchase propensity classification models."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import build_feature_preprocessor


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    """Configuration for the logistic regression baseline."""

    random_seed: int = 42
    class_weight: str | None = None
    solver: str = "lbfgs"
    max_iter: int = 2000

    def __post_init__(self) -> None:
        """Validate baseline model parameters."""
        if self.random_seed < 0:
            raise ValueError("Random seed must be non-negative.")

        if self.class_weight not in {None, "balanced"}:
            raise ValueError("Class weight must be either 'balanced' or None.")

        if self.solver not in {
            "lbfgs",
            "liblinear",
            "newton-cg",
            "newton-cholesky",
            "sag",
            "saga",
        }:
            raise ValueError(f"Unsupported logistic regression solver: {self.solver}")

        if self.max_iter <= 0:
            raise ValueError("Maximum iterations must be greater than zero.")


@dataclass(frozen=True, slots=True)
class TreeModelConfig:
    """Shared configuration for tree ensemble candidates."""

    random_seed: int = 42
    n_estimators: int = 300
    max_depth: int | None = 12
    min_samples_leaf: int = 5
    class_weight: str | None = None
    n_jobs: int = -1

    def __post_init__(self) -> None:
        """Validate tree ensemble parameters."""
        if self.random_seed < 0:
            raise ValueError("Random seed must be non-negative.")

        if self.n_estimators <= 0:
            raise ValueError("Number of estimators must be greater than zero.")

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("Maximum depth must be greater than zero or None.")

        if self.min_samples_leaf <= 0:
            raise ValueError("Minimum samples per leaf must be greater than zero.")

        if self.class_weight not in {
            None,
            "balanced",
            "balanced_subsample",
        }:
            raise ValueError("Class weight must be 'balanced', 'balanced_subsample' or None.")

        if self.n_jobs == 0:
            raise ValueError("Number of jobs cannot be zero.")


def build_baseline_classifier(
    config: BaselineModelConfig,
) -> LogisticRegression:
    """Build the configured logistic regression classifier."""
    return LogisticRegression(
        class_weight=config.class_weight,
        solver=config.solver,
        max_iter=config.max_iter,
        random_state=config.random_seed,
    )


def build_random_forest_classifier(
    config: TreeModelConfig,
) -> RandomForestClassifier:
    """Build the configured random forest classifier."""
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
    )


def build_extra_trees_classifier(
    config: TreeModelConfig,
) -> ExtraTreesClassifier:
    """Build the configured extremely randomized trees classifier."""
    return ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
    )


def build_baseline_pipeline(
    config: BaselineModelConfig,
) -> Pipeline:
    """Combine feature preprocessing and baseline classification."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_feature_preprocessor(),
            ),
            (
                "classifier",
                build_baseline_classifier(config),
            ),
        ]
    )


def build_random_forest_pipeline(
    config: TreeModelConfig,
) -> Pipeline:
    """Combine unscaled numerical features and random forest classification."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_feature_preprocessor(scale_numeric=False),
            ),
            (
                "classifier",
                build_random_forest_classifier(config),
            ),
        ]
    )


def build_extra_trees_pipeline(
    config: TreeModelConfig,
) -> Pipeline:
    """Combine feature preprocessing and extra trees classification."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_feature_preprocessor(),
            ),
            (
                "classifier",
                build_extra_trees_classifier(config),
            ),
        ]
    )

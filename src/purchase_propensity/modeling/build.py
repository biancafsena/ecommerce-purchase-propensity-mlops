"""Build baseline purchase propensity classification models."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import build_feature_preprocessor


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    """Configuration for the logistic regression baseline."""

    random_seed: int = 42
    class_weight: str | None = "balanced"
    solver: str = "lbfgs"
    max_iter: int = 1000

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


def build_baseline_pipeline(
    config: BaselineModelConfig,
) -> Pipeline:
    """Combine feature preprocessing and baseline classification."""
    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor()),
            ("classifier", build_baseline_classifier(config)),
        ]
    )

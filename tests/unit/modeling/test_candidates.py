"""Unit tests for candidate tree ensemble builders."""

import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline

from purchase_propensity.modeling.build import (
    TreeModelConfig,
    build_extra_trees_classifier,
    build_extra_trees_pipeline,
    build_random_forest_classifier,
    build_random_forest_pipeline,
)


def test_tree_model_config_accepts_valid_parameters() -> None:
    """Accept a complete valid tree ensemble configuration."""
    config = TreeModelConfig(
        random_seed=42,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
    )

    assert config.random_seed == 42
    assert config.n_estimators == 300
    assert config.max_depth is None
    assert config.min_samples_leaf == 2
    assert config.class_weight == "balanced"
    assert config.n_jobs == -1


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"random_seed": -1},
            "Random seed must be non-negative",
        ),
        (
            {"n_estimators": 0},
            "Number of estimators must be greater than zero",
        ),
        (
            {"max_depth": 0},
            "Maximum depth must be greater than zero or None",
        ),
        (
            {"min_samples_leaf": 0},
            "Minimum samples per leaf must be greater than zero",
        ),
        (
            {"class_weight": "invalid"},
            "Class weight must be",
        ),
        (
            {"n_jobs": 0},
            "Number of jobs cannot be zero",
        ),
    ],
)
def test_tree_model_config_rejects_invalid_parameters(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """Reject invalid tree ensemble configuration values."""
    parameters: dict[str, object] = {
        "random_seed": 42,
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    parameters.update(overrides)

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        TreeModelConfig(**parameters)  # type: ignore[arg-type]


def test_build_random_forest_classifier_applies_configuration() -> None:
    """Build a random forest with the configured parameters."""
    config = TreeModelConfig(
        random_seed=17,
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=2,
    )

    classifier = build_random_forest_classifier(config)

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.random_state == 17
    assert classifier.n_estimators == 120
    assert classifier.max_depth == 8
    assert classifier.min_samples_leaf == 3
    assert classifier.class_weight == "balanced_subsample"
    assert classifier.n_jobs == 2


def test_build_extra_trees_classifier_applies_configuration() -> None:
    """Build an extra trees model with the configured parameters."""
    config = TreeModelConfig(
        random_seed=23,
        n_estimators=140,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=3,
    )

    classifier = build_extra_trees_classifier(config)

    assert isinstance(classifier, ExtraTreesClassifier)
    assert classifier.random_state == 23
    assert classifier.n_estimators == 140
    assert classifier.max_depth == 10
    assert classifier.min_samples_leaf == 4
    assert classifier.class_weight == "balanced"
    assert classifier.n_jobs == 3


def test_build_random_forest_pipeline_has_expected_steps() -> None:
    """Build preprocessing followed by random forest classification."""
    pipeline = build_random_forest_pipeline(
        TreeModelConfig(),
    )

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [
        "preprocessor",
        "classifier",
    ]
    assert isinstance(
        pipeline.named_steps["classifier"],
        RandomForestClassifier,
    )


def test_build_extra_trees_pipeline_has_expected_steps() -> None:
    """Build preprocessing followed by extra trees classification."""
    pipeline = build_extra_trees_pipeline(
        TreeModelConfig(),
    )

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [
        "preprocessor",
        "classifier",
    ]
    assert isinstance(
        pipeline.named_steps["classifier"],
        ExtraTreesClassifier,
    )

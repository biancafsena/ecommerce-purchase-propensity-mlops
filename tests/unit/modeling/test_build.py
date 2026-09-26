"""Unit tests for baseline model construction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_propensity.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from purchase_propensity.modeling.build import (
    BaselineModelConfig,
    build_baseline_classifier,
    build_baseline_pipeline,
)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create model-ready features with both numerical and categorical data."""
    row_count = 8
    dataframe = pd.DataFrame(
        {feature: np.linspace(0.0, 1.0, row_count) for feature in NUMERIC_FEATURES}
    )

    dataframe["Month"] = [
        "Feb",
        "Mar",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
    ]
    dataframe["OperatingSystems"] = [1, 2, 1, 2, 3, 1, 2, 3]
    dataframe["Browser"] = [1, 2, 1, 3, 2, 1, 3, 2]
    dataframe["Region"] = [1, 2, 3, 1, 2, 3, 1, 2]
    dataframe["TrafficType"] = [1, 2, 3, 4, 1, 2, 3, 4]
    dataframe["VisitorType"] = [
        "Returning_Visitor",
        "New_Visitor",
        "Returning_Visitor",
        "New_Visitor",
        "Returning_Visitor",
        "New_Visitor",
        "Returning_Visitor",
        "Other",
    ]
    dataframe["Weekend"] = [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
    ]

    return dataframe.loc[
        :,
        list(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
    ]


def test_build_baseline_classifier_uses_configuration() -> None:
    """Transfer validated configuration to Logistic Regression."""
    config = BaselineModelConfig(
        random_seed=21,
        class_weight="balanced",
        solver="liblinear",
        max_iter=500,
    )

    classifier = build_baseline_classifier(config)

    assert isinstance(classifier, LogisticRegression)
    assert classifier.random_state == 21
    assert classifier.class_weight == "balanced"
    assert classifier.solver == "liblinear"
    assert classifier.max_iter == 500


def test_build_baseline_pipeline_contains_expected_steps() -> None:
    """Combine preprocessing and classification in the required order."""
    pipeline = build_baseline_pipeline(BaselineModelConfig())

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert isinstance(
        pipeline.named_steps["preprocessor"],
        ColumnTransformer,
    )
    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression,
    )


def test_baseline_pipeline_fits_and_predicts_probabilities(
    sample_features: pd.DataFrame,
) -> None:
    """Fit the complete pipeline and produce valid probabilities."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pipeline = build_baseline_pipeline(BaselineModelConfig())

    pipeline.fit(sample_features, labels)
    probabilities = pipeline.predict_proba(sample_features)[:, 1]

    assert probabilities.shape == (len(sample_features),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


@pytest.mark.parametrize(
    ("configuration", "expected_message"),
    [
        (
            {"random_seed": -1},
            "Random seed must be non-negative",
        ),
        (
            {"class_weight": "invalid"},
            "Class weight must be either 'balanced' or None",
        ),
        (
            {"solver": "invalid"},
            "Unsupported logistic regression solver",
        ),
        (
            {"max_iter": 0},
            "Maximum iterations must be greater than zero",
        ),
    ],
)
def test_baseline_model_config_rejects_invalid_values(
    configuration: dict[str, object],
    expected_message: str,
) -> None:
    """Reject unsupported or unsafe model parameters."""
    with pytest.raises(ValueError, match=expected_message):
        BaselineModelConfig(**configuration)

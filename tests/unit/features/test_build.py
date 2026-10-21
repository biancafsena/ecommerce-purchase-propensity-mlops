"""Unit tests for the feature preprocessing pipeline."""

import pandas as pd
import pytest

from purchase_propensity.features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_feature_preprocessor,
    split_features_and_target,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a representative dataset with all required model columns."""
    return pd.DataFrame(
        {
            "Administrative": [0, 1, 2, 3],
            "Administrative_Duration": [0.0, 10.0, 20.0, 30.0],
            "Informational": [0, 0, 1, 1],
            "Informational_Duration": [0.0, 0.0, 5.0, 10.0],
            "ProductRelated": [1, 5, 10, 20],
            "ProductRelated_Duration": [0.0, 50.0, 100.0, 200.0],
            "BounceRates": [0.20, 0.10, 0.05, 0.01],
            "ExitRates": [0.20, 0.15, 0.10, 0.02],
            "PageValues": [0.0, 0.0, 5.0, 20.0],
            "SpecialDay": [0.0, 0.0, 0.2, 0.8],
            "Month": ["Feb", "Mar", "May", "Dec"],
            "OperatingSystems": [1, 2, 3, 4],
            "Browser": [1, 2, 3, 4],
            "Region": [1, 2, 3, 4],
            "TrafficType": [1, 2, 3, 4],
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
                "Returning_Visitor",
                "Other",
            ],
            "Weekend": [False, False, True, True],
            "Revenue": [False, False, True, True],
        }
    )


def test_feature_groups_cover_all_predictors_without_overlap() -> None:
    """Keep numerical and categorical feature definitions consistent."""
    assert len(NUMERIC_FEATURES) == 9
    assert len(CATEGORICAL_FEATURES) == 7
    assert len(FEATURE_COLUMNS) == 16
    assert "PageValues" not in FEATURE_COLUMNS
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_split_features_and_target_returns_expected_data(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Separate predictors and convert the target to binary integers."""
    features, target = split_features_and_target(sample_dataframe)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert "Revenue" not in features.columns
    assert "PageValues" not in features.columns
    assert target.tolist() == [0, 0, 1, 1]
    assert str(target.dtype) == "int8"


def test_split_features_and_target_rejects_missing_target(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Reject data without the configured target column."""
    dataframe = sample_dataframe.drop(columns="Revenue")

    with pytest.raises(ValueError, match="Target column not found"):
        split_features_and_target(dataframe)


def test_split_features_and_target_rejects_missing_feature(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Reject data without a required predictor."""
    dataframe = sample_dataframe.drop(columns="ProductRelated")

    with pytest.raises(ValueError, match="Required feature columns not found"):
        split_features_and_target(dataframe)


def test_feature_preprocessor_handles_unseen_categories(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Transform unseen categories without changing the output structure."""
    train_dataframe = sample_dataframe.iloc[:3]
    validation_dataframe = sample_dataframe.iloc[3:]

    train_features, _ = split_features_and_target(train_dataframe)
    validation_features, _ = split_features_and_target(validation_dataframe)

    preprocessor = build_feature_preprocessor()
    transformed_train = preprocessor.fit_transform(train_features)
    transformed_validation = preprocessor.transform(validation_features)

    assert transformed_train.shape[0] == 3
    assert transformed_validation.shape[0] == 1
    assert transformed_train.shape[1] == transformed_validation.shape[1]
    assert transformed_train.shape[1] > len(FEATURE_COLUMNS)


def test_feature_preprocessor_exposes_transformed_feature_names(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Expose stable names for model interpretation and auditing."""
    features, _ = split_features_and_target(sample_dataframe)
    preprocessor = build_feature_preprocessor()

    preprocessor.fit(features)
    transformed_names = preprocessor.get_feature_names_out().tolist()

    assert set(NUMERIC_FEATURES).issubset(transformed_names)
    assert any(name.startswith("Month_") for name in transformed_names)
    assert any(name.startswith("VisitorType_") for name in transformed_names)

"""Build the feature preprocessing pipeline for purchase propensity modeling."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

CATEGORICAL_FEATURES = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_TARGET_COLUMN = "Revenue"


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary target."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    missing_features = sorted(set(FEATURE_COLUMNS) - set(dataframe.columns))
    if missing_features:
        raise ValueError(f"Required feature columns not found: {missing_features}")

    features = dataframe.loc[:, FEATURE_COLUMNS].copy()
    target = dataframe.loc[:, target_column].astype("int8").copy()

    return features, target


def build_numeric_pipeline() -> Pipeline:
    """Build preprocessing steps for numerical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Build preprocessing steps for categorical features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


def build_feature_preprocessor() -> ColumnTransformer:
    """Build the complete numerical and categorical feature transformer."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                build_numeric_pipeline(),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                build_categorical_pipeline(),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

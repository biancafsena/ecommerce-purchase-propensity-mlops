"""Unit tests for the raw dataset schema."""

import pandas as pd
import pandera.pandas as pa
import pytest

from purchase_propensity.data import schema


@pytest.fixture
def valid_raw_dataframe() -> pd.DataFrame:
    """Build a minimal valid dataset containing both target classes."""
    return pd.DataFrame(
        {
            "Administrative": [0, 2],
            "Administrative_Duration": [0.0, 45.5],
            "Informational": [0, 1],
            "Informational_Duration": [0.0, 12.0],
            "ProductRelated": [1, 10],
            "ProductRelated_Duration": [0.0, 320.5],
            "BounceRates": [0.2, 0.01],
            "ExitRates": [0.2, 0.03],
            "PageValues": [0.0, 25.5],
            "SpecialDay": [0.0, 0.8],
            "Month": ["Feb", "Nov"],
            "OperatingSystems": [1, 2],
            "Browser": [1, 3],
            "Region": [1, 4],
            "TrafficType": [1, 2],
            "VisitorType": ["Returning_Visitor", "New_Visitor"],
            "Weekend": [False, True],
            "Revenue": [False, True],
        }
    )


@pytest.fixture(autouse=True)
def use_test_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt the fixed production row count for small unit-test frames."""
    monkeypatch.setattr(schema, "EXPECTED_ROW_COUNT", 2)


def test_validate_raw_dataset_accepts_valid_data(
    valid_raw_dataframe: pd.DataFrame,
) -> None:
    """Accept data that complies with every schema constraint."""
    validated_dataframe = schema.validate_raw_dataset(valid_raw_dataframe)

    pd.testing.assert_frame_equal(validated_dataframe, valid_raw_dataframe)


def test_validate_raw_dataset_rejects_unexpected_column(
    valid_raw_dataframe: pd.DataFrame,
) -> None:
    """Reject columns that are not part of the data contract."""
    invalid_dataframe = valid_raw_dataframe.assign(UnexpectedFeature=1)

    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate_raw_dataset(invalid_dataframe)


def test_validate_raw_dataset_rejects_rate_outside_range(
    valid_raw_dataframe: pd.DataFrame,
) -> None:
    """Reject rates outside the inclusive interval from zero to one."""
    invalid_dataframe = valid_raw_dataframe.copy()
    invalid_dataframe.loc[0, "BounceRates"] = 1.5

    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate_raw_dataset(invalid_dataframe)


def test_validate_raw_dataset_rejects_unknown_month(
    valid_raw_dataframe: pd.DataFrame,
) -> None:
    """Reject a month outside the official dataset domain."""
    invalid_dataframe = valid_raw_dataframe.copy()
    invalid_dataframe.loc[0, "Month"] = "Jan"

    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate_raw_dataset(invalid_dataframe)


def test_validate_raw_dataset_requires_both_target_classes(
    valid_raw_dataframe: pd.DataFrame,
) -> None:
    """Reject data that contains only one target class."""
    invalid_dataframe = valid_raw_dataframe.copy()
    invalid_dataframe["Revenue"] = False

    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate_raw_dataset(invalid_dataframe)

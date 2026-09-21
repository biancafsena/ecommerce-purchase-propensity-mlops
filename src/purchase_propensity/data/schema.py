"""Runtime schema for the raw Online Shoppers dataset."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pandera.pandas as pa

EXPECTED_ROW_COUNT = 12_330

VALID_MONTHS = (
    "Feb",
    "Mar",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

VALID_OPERATING_SYSTEMS = tuple(range(1, 9))
VALID_BROWSERS = tuple(range(1, 14))
VALID_REGIONS = tuple(range(1, 10))
VALID_TRAFFIC_TYPES = tuple(range(1, 21))

VALID_VISITOR_TYPES = (
    "New_Visitor",
    "Other",
    "Returning_Visitor",
)

RAW_DATASET_SCHEMA = pa.DataFrameSchema(
    columns={
        "Administrative": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "Administrative_Duration": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "Informational": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "Informational_Duration": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "ProductRelated": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "ProductRelated_Duration": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "BounceRates": pa.Column(
            float,
            checks=pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "ExitRates": pa.Column(
            float,
            checks=pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "PageValues": pa.Column(
            float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "SpecialDay": pa.Column(
            float,
            checks=pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "Month": pa.Column(
            str,
            checks=pa.Check.isin(VALID_MONTHS),
            nullable=False,
        ),
        "OperatingSystems": pa.Column(
            int,
            checks=pa.Check.isin(VALID_OPERATING_SYSTEMS),
            nullable=False,
        ),
        "Browser": pa.Column(
            int,
            checks=pa.Check.isin(VALID_BROWSERS),
            nullable=False,
        ),
        "Region": pa.Column(
            int,
            checks=pa.Check.isin(VALID_REGIONS),
            nullable=False,
        ),
        "TrafficType": pa.Column(
            int,
            checks=pa.Check.isin(VALID_TRAFFIC_TYPES),
            nullable=False,
        ),
        "VisitorType": pa.Column(
            str,
            checks=pa.Check.isin(VALID_VISITOR_TYPES),
            nullable=False,
        ),
        "Weekend": pa.Column(
            bool,
            nullable=False,
        ),
        "Revenue": pa.Column(
            bool,
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            lambda dataframe: len(dataframe) == EXPECTED_ROW_COUNT,
            error=f"Dataset must contain exactly {EXPECTED_ROW_COUNT} rows.",
        ),
        pa.Check(
            lambda dataframe: dataframe["Revenue"].nunique() == 2,
            error="Target column Revenue must contain both classes.",
        ),
    ],
    strict=True,
    ordered=True,
    coerce=False,
    unique_column_names=True,
    name="raw_online_shoppers_dataset",
)


def validate_raw_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate the structure and business constraints of the raw dataset."""
    validated_dataframe = RAW_DATASET_SCHEMA.validate(dataframe, lazy=True)
    return cast(pd.DataFrame, validated_dataframe)

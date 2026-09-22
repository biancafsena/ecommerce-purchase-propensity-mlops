"""Unit tests for dataset preprocessing and stratified splitting."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from purchase_propensity.data.preprocess import (
    SplitConfig,
    load_split_config,
    split_dataset,
    write_dataset_splits,
)

ROW_COUNT = 100
POSITIVE_COUNT = 20


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a representative binary-class dataset."""
    return pd.DataFrame(
        {
            "feature": range(ROW_COUNT),
            "Revenue": [False] * (ROW_COUNT - POSITIVE_COUNT) + [True] * POSITIVE_COUNT,
        }
    )


def write_params(path: Path, data_config: dict[str, Any]) -> None:
    """Write a temporary YAML parameter file."""
    path.write_text(
        yaml.safe_dump({"data": data_config}),
        encoding="utf-8",
    )


def test_load_split_config_returns_expected_values(tmp_path: Path) -> None:
    """Load and type the data-splitting configuration."""
    params_path = tmp_path / "params.yaml"
    write_params(
        params_path,
        {
            "target_column": "Revenue",
            "random_seed": 42,
            "train_size": 0.70,
            "validation_size": 0.15,
            "test_size": 0.15,
        },
    )

    config = load_split_config(params_path)

    assert config == SplitConfig(
        target_column="Revenue",
        random_seed=42,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    )


@pytest.mark.parametrize(
    ("train_size", "validation_size", "test_size"),
    [
        (0.80, 0.15, 0.15),
        (0.70, 0.00, 0.30),
        (1.00, 0.00, 0.00),
    ],
)
def test_load_split_config_rejects_invalid_sizes(
    tmp_path: Path,
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    """Reject invalid proportions and totals."""
    params_path = tmp_path / "params.yaml"
    write_params(
        params_path,
        {
            "target_column": "Revenue",
            "random_seed": 42,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
        },
    )

    with pytest.raises(ValueError):
        load_split_config(params_path)


def test_split_dataset_is_deterministic_and_stratified(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Preserve class balance and reproduce identical partitions."""
    config = SplitConfig(
        target_column="Revenue",
        random_seed=42,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    )

    first_split = split_dataset(sample_dataframe, config)
    second_split = split_dataset(sample_dataframe, config)

    split_names = ("train", "validation", "test")
    expected_rows = (70, 15, 15)

    for split_name, expected_count, first_dataframe, second_dataframe in zip(
        split_names,
        expected_rows,
        first_split,
        second_split,
        strict=True,
    ):
        assert len(first_dataframe) == expected_count, split_name
        assert first_dataframe["Revenue"].mean() == pytest.approx(0.20)
        pd.testing.assert_frame_equal(first_dataframe, second_dataframe)

    split_ids = [set(split_dataframe["feature"]) for split_dataframe in first_split]

    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert set.union(*split_ids) == set(range(ROW_COUNT))


def test_write_dataset_splits_creates_readable_parquet_files(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Persist every partition as a readable Parquet file."""
    splits = (
        sample_dataframe.iloc[:70].reset_index(drop=True),
        sample_dataframe.iloc[70:85].reset_index(drop=True),
        sample_dataframe.iloc[85:].reset_index(drop=True),
    )
    filenames = ("train.parquet", "validation.parquet", "test.parquet")

    write_dataset_splits(splits, tmp_path)

    for filename, expected_dataframe in zip(
        filenames,
        splits,
        strict=True,
    ):
        output_path = tmp_path / filename

        assert output_path.is_file()
        pd.testing.assert_frame_equal(
            pd.read_parquet(output_path),
            expected_dataframe,
        )

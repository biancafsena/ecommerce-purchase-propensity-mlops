"""Validate and split the dataset into reproducible stratified subsets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from purchase_propensity.data.schema import validate_raw_dataset

DEFAULT_INPUT_PATH = Path("data/raw/online_shoppers_intention.csv")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed")

ValueType = TypeVar("ValueType")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic dataset splitting."""

    target_column: str
    random_seed: int
    train_size: float
    validation_size: float
    test_size: float


def require_typed_value(
    values: dict[str, object],
    key: str,
    expected_type: type[ValueType],
) -> ValueType:
    """Return a required configuration value with runtime type validation."""
    value = values.get(key)

    if not isinstance(value, expected_type):
        raise ValueError(f"Configuration key '{key}' must be {expected_type.__name__}.")

    return value


def load_split_config(params_path: Path) -> SplitConfig:
    """Load split configuration from the project parameters file."""
    raw_params = yaml.safe_load(params_path.read_text(encoding="utf-8"))

    if not isinstance(raw_params, dict) or not isinstance(raw_params.get("data"), dict):
        raise ValueError("params.yaml must contain a 'data' mapping.")

    values = cast(dict[str, object], raw_params["data"])
    config = SplitConfig(
        target_column=require_typed_value(values, "target_column", str),
        random_seed=require_typed_value(values, "random_seed", int),
        train_size=require_typed_value(values, "train_size", float),
        validation_size=require_typed_value(values, "validation_size", float),
        test_size=require_typed_value(values, "test_size", float),
    )
    validate_split_config(config)
    return config


def validate_split_config(config: SplitConfig) -> None:
    """Validate split proportions before processing data."""
    split_sizes = (config.train_size, config.validation_size, config.test_size)

    if not all(0 < size < 1 for size in split_sizes):
        raise ValueError("Every split size must be between zero and one.")

    if not math.isclose(sum(split_sizes), 1.0):
        raise ValueError("Train, validation and test sizes must sum to one.")


def split_dataset(
    dataframe: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic train, validation and test subsets."""
    temporary_size = config.validation_size + config.test_size
    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        test_size=temporary_size,
        random_state=config.random_seed,
        stratify=dataframe[config.target_column],
    )
    relative_test_size = config.test_size / temporary_size
    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=relative_test_size,
        random_state=config.random_seed,
        stratify=temporary_dataframe[config.target_column],
    )
    return (
        train_dataframe.reset_index(drop=True),
        validation_dataframe.reset_index(drop=True),
        test_dataframe.reset_index(drop=True),
    )


def write_dataset_splits(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    output_directory: Path,
) -> None:
    """Persist split datasets using the typed Parquet format."""
    output_directory.mkdir(parents=True, exist_ok=True)
    filenames = ("train.parquet", "validation.parquet", "test.parquet")

    for dataframe, filename in zip(splits, filenames, strict=True):
        dataframe.to_parquet(output_directory / filename, index=False)


def run_preprocessing(
    input_path: Path,
    params_path: Path,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate, split and persist the raw dataset."""
    config = load_split_config(params_path)
    raw_dataframe = pd.read_csv(input_path)
    validated_dataframe = validate_raw_dataset(raw_dataframe)
    splits = split_dataset(validated_dataframe, config)
    write_dataset_splits(splits, output_directory)
    return splits


def parse_arguments() -> argparse.Namespace:
    """Parse preprocessing command-line arguments."""
    parser = argparse.ArgumentParser(description="Create reproducible stratified dataset splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing from the command line."""
    arguments = parse_arguments()
    train, validation, test = run_preprocessing(
        arguments.input,
        arguments.params,
        arguments.output_directory,
    )
    print(
        "Dataset split completed: "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}."
    )


if __name__ == "__main__":
    main()

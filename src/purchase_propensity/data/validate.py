"""Validate the raw dataset and produce an auditable quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from purchase_propensity.data.download import calculate_sha256
from purchase_propensity.data.schema import validate_raw_dataset

DEFAULT_INPUT_PATH = Path("data/raw/online_shoppers_intention.csv")
DEFAULT_REPORT_PATH = Path("reports/metrics/raw_data_validation.json")


def build_validation_report(
    dataframe: pd.DataFrame,
    dataset_sha256: str,
) -> dict[str, object]:
    """Build quality and class-distribution metrics for validated data."""
    positive_count = int(dataframe["Revenue"].sum())
    row_count = len(dataframe)

    return {
        "status": "passed",
        "dataset_sha256": dataset_sha256,
        "row_count": row_count,
        "column_count": len(dataframe.columns),
        "missing_value_count": int(dataframe.isna().sum().sum()),
        "negative_class_count": row_count - positive_count,
        "positive_class_count": positive_count,
        "positive_class_rate": round(positive_count / row_count, 6),
    }


def write_validation_report(
    report: dict[str, object],
    report_path: Path,
) -> None:
    """Write the validation report as deterministic JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_report = json.dumps(report, indent=2, sort_keys=True)
    report_path.write_text(f"{serialized_report}\n", encoding="utf-8")


def run_validation(input_path: Path, report_path: Path) -> dict[str, object]:
    """Validate raw data and persist its quality report."""
    dataframe = pd.read_csv(input_path)
    validated_dataframe = validate_raw_dataset(dataframe)
    dataset_sha256 = calculate_sha256(input_path)
    report = build_validation_report(validated_dataframe, dataset_sha256)
    write_validation_report(report, report_path)
    return report


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate raw data and create a quality report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination path for the validation report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run raw data validation from the command line."""
    arguments = parse_arguments()
    report = run_validation(arguments.input, arguments.report)
    print(
        "Raw dataset validation passed: "
        f"{report['row_count']} rows, "
        f"{report['positive_class_rate']:.2%} positive class."
    )


if __name__ == "__main__":
    main()

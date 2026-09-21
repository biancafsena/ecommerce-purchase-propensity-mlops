"""Unit tests for the raw data validation report."""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from purchase_propensity.data import validate


@pytest.fixture
def class_distribution_dataframe() -> pd.DataFrame:
    """Build a small dataframe with a balanced binary target."""
    return pd.DataFrame(
        {
            "Feature": [1, 2, 3, 4],
            "Revenue": [False, True, False, True],
        }
    )


def test_build_validation_report_calculates_quality_metrics(
    class_distribution_dataframe: pd.DataFrame,
) -> None:
    """Calculate deterministic quality and target metrics."""
    report = validate.build_validation_report(
        class_distribution_dataframe,
        dataset_sha256="expected-checksum",
    )

    assert report == {
        "status": "passed",
        "dataset_sha256": "expected-checksum",
        "row_count": 4,
        "column_count": 2,
        "missing_value_count": 0,
        "negative_class_count": 2,
        "positive_class_count": 2,
        "positive_class_rate": 0.5,
    }


def test_write_validation_report_creates_deterministic_json(
    tmp_path: Path,
) -> None:
    """Persist sorted and newline-terminated JSON."""
    report_path = tmp_path / "nested" / "report.json"
    report = {
        "status": "passed",
        "row_count": 4,
    }

    validate.write_validation_report(report, report_path)

    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report == report
    assert report_path.read_text(encoding="utf-8").endswith("\n")


def test_run_validation_reads_validates_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the validation workflow with isolated collaborators."""
    input_path = tmp_path / "dataset.csv"
    report_path = tmp_path / "report.json"
    input_path.write_text(
        "Feature,Revenue\n1,False\n2,True\n",
        encoding="utf-8",
    )
    schema_mock = Mock(side_effect=lambda dataframe: dataframe)
    checksum_mock = Mock(return_value="test-checksum")

    monkeypatch.setattr(validate, "validate_raw_dataset", schema_mock)
    monkeypatch.setattr(validate, "calculate_sha256", checksum_mock)

    report = validate.run_validation(input_path, report_path)

    assert report["status"] == "passed"
    assert report["dataset_sha256"] == "test-checksum"
    assert report_path.exists()
    schema_mock.assert_called_once()
    checksum_mock.assert_called_once_with(input_path)


def test_main_forwards_command_line_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward CLI paths to the validation workflow."""
    input_path = tmp_path / "input.csv"
    report_path = tmp_path / "report.json"
    validation_mock = Mock(
        return_value={
            "row_count": 2,
            "positive_class_rate": 0.5,
        }
    )

    monkeypatch.setattr(validate, "run_validation", validation_mock)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate",
            "--input",
            str(input_path),
            "--report",
            str(report_path),
        ],
    )

    validate.main()

    validation_mock.assert_called_once_with(input_path, report_path)

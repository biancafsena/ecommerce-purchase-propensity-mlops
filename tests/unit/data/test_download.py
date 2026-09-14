"""Unit tests for the raw dataset acquisition module."""

from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from purchase_propensity.data import download

KNOWN_CONTENT = b"abc"
KNOWN_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_calculate_sha256_returns_expected_digest(tmp_path: Path) -> None:
    """Calculate a deterministic checksum for known content."""
    file_path = tmp_path / "sample.csv"
    file_path.write_bytes(KNOWN_CONTENT)

    assert download.calculate_sha256(file_path) == KNOWN_SHA256


def test_validate_dataset_rejects_modified_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a dataset whose checksum differs from the expected value."""
    file_path = tmp_path / "modified.csv"
    file_path.write_bytes(KNOWN_CONTENT)
    monkeypatch.setattr(download, "EXPECTED_SHA256", "invalid-checksum")

    with pytest.raises(ValueError, match="Dataset checksum mismatch"):
        download.validate_dataset(file_path)


def test_acquire_dataset_reuses_existing_valid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid another download when a valid dataset already exists."""
    output_path = tmp_path / "dataset.csv"
    output_path.write_bytes(KNOWN_CONTENT)
    download_mock = Mock()

    monkeypatch.setattr(download, "EXPECTED_SHA256", KNOWN_SHA256)
    monkeypatch.setattr(download, "download_archive", download_mock)

    download.acquire_dataset(output_path)

    download_mock.assert_not_called()


def test_acquire_dataset_downloads_and_replaces_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download and atomically replace the destination when forced."""
    output_path = tmp_path / "dataset.csv"
    output_path.write_text("outdated", encoding="utf-8")

    def create_test_archive(archive_path: Path) -> None:
        with ZipFile(archive_path, mode="w") as archive:
            archive.writestr(download.ARCHIVE_MEMBER, KNOWN_CONTENT)

    monkeypatch.setattr(download, "EXPECTED_SHA256", KNOWN_SHA256)
    monkeypatch.setattr(download, "download_archive", create_test_archive)

    download.acquire_dataset(output_path, force=True)

    assert output_path.read_bytes() == KNOWN_CONTENT


def test_main_forwards_command_line_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward parsed CLI arguments to the acquisition service."""
    output_path = tmp_path / "custom.csv"
    acquire_mock = Mock()

    monkeypatch.setattr(download, "acquire_dataset", acquire_mock)
    monkeypatch.setattr(
        "sys.argv",
        ["download", "--output", str(output_path), "--force"],
    )

    download.main()

    acquire_mock.assert_called_once_with(output_path, True)

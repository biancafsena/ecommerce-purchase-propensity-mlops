"""Download and verify the raw Online Shoppers dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
ARCHIVE_MEMBER = "online_shoppers_intention.csv"
EXPECTED_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
DEFAULT_OUTPUT_PATH = Path("data/raw/online_shoppers_intention.csv")
DOWNLOAD_TIMEOUT_SECONDS = 60
HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_SIZE_BYTES):
            digest.update(chunk)

    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    """Download the source archive to a temporary path."""
    with (
        urlopen(DATASET_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        destination.open("wb") as archive_file,
    ):
        shutil.copyfileobj(response, archive_file)


def extract_dataset(archive_path: Path, destination: Path) -> None:
    """Extract only the expected CSV from the source archive."""
    with (
        ZipFile(archive_path) as archive,
        archive.open(ARCHIVE_MEMBER) as source,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(source, output)


def validate_dataset(file_path: Path) -> None:
    """Raise an error when the downloaded dataset is not authentic."""
    actual_sha256 = calculate_sha256(file_path)

    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, received {actual_sha256}"
        )


def acquire_dataset(output_path: Path, force: bool = False) -> None:
    """Download, extract and atomically store the verified dataset."""
    if output_path.exists() and not force:
        validate_dataset(output_path)
        print(f"Dataset already exists and is valid: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / "dataset.zip"
        extracted_path = temporary_path / ARCHIVE_MEMBER

        download_archive(archive_path)
        extract_dataset(archive_path, extracted_path)
        validate_dataset(extracted_path)
        extracted_path.replace(output_path)

    print(f"Dataset acquired and verified: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and verify the Online Shoppers dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the extracted CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing dataset after verification.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset acquisition command."""
    arguments = parse_arguments()
    acquire_dataset(arguments.output, arguments.force)


if __name__ == "__main__":
    main()

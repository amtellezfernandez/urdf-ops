from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import pytest

from backend.scripts import dataset_mixer


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_extract_dataset_zip_rejects_parent_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_ENTRY_COUNT", 10)
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_ENTRY_BYTES", 1024)
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_TOTAL_BYTES", 4096)

    with TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "demo.zip"
        _write_zip(zip_path, {"../escape.txt": b"blocked"})

        with pytest.raises(ValueError, match="unsafe path"):
            dataset_mixer._extract_dataset_zip(zip_path)


def test_extract_dataset_zip_rejects_oversized_member(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_ENTRY_COUNT", 10)
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_ENTRY_BYTES", 4)
    monkeypatch.setattr(dataset_mixer, "DATASET_MIX_ARCHIVE_MAX_TOTAL_BYTES", 4096)

    with TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "demo.zip"
        _write_zip(zip_path, {"dataset/file.bin": b"12345"})

        with pytest.raises(ValueError, match="file-size limit"):
            dataset_mixer._extract_dataset_zip(zip_path)

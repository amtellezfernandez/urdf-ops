from __future__ import annotations

import json
from pathlib import Path

from backend.services.datasets import list_dataset_catalog


def test_dataset_catalog_discovers_studio_lerobot_exports(tmp_path: Path) -> None:
    dataset_root = tmp_path / "studio-exports" / "pick-place-001"
    meta_root = dataset_root / "meta"
    (dataset_root / "data").mkdir(parents=True)
    meta_root.mkdir(parents=True)
    (meta_root / "info.json").write_text(
        json.dumps(
            {
                "dataset_format_version": "lerobot_dataset_v3",
                "robot_type": "so101",
                "total_episodes": 1,
                "total_frames": 42,
                "fps": 10,
            }
        ),
        encoding="utf-8",
    )
    (meta_root / "urdf_studio_replay.json").write_text(
        json.dumps(
            {
                "recording_id": "pick-place-001",
                "export_mode": "studio_kinematic",
            }
        ),
        encoding="utf-8",
    )

    catalog = list_dataset_catalog(local_roots=[tmp_path / "studio-exports"])

    assert len(catalog.datasets) == 1
    dataset = catalog.datasets[0]
    assert dataset.name == "pick-place-001"
    assert dataset.source == "studio_export"
    assert dataset.path == str(dataset_root.resolve(strict=False))
    assert dataset.robot_type == "so101"
    assert dataset.total_frames == 42

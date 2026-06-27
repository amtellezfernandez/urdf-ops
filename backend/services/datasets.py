from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models.datasets import (
    DatasetCatalogItem,
    DatasetCatalogResponse,
    DatasetMixRequest,
    DatasetMixResponse,
)
from backend.services.dataset_source_contract import (
    normalize_local_dataset_paths as normalize_dataset_source_local_paths,
)
from backend.services.datasets_params import (
    DATASET_MIX_ALLOWED_LOCAL_ROOTS,
    PATH_LIST_SEPARATOR,
)
from backend.services.teleop_replay_params import (
    TELEOP_REPLAY_INFO_FILENAME,
    TELEOP_REPLAY_META_FILENAME,
)


DATASET_CATALOG_LOCAL_ROOTS_ENV = "URDF_DATASET_CATALOG_LOCAL_ROOTS"
DATASET_CATALOG_MAX_SCAN_DEPTH = 8
DATASET_CATALOG_MAX_ITEMS = 200
LEROBOT_META_DIRNAME = "meta"
LEROBOT_DATA_DIRNAME = "data"


def get_dataset_mix_control_plane():
    from backend.services.dataset_mix_control_plane import (
        get_dataset_mix_control_plane as get_control_plane,
    )

    return get_control_plane()


def normalize_local_dataset_paths(local_paths: list[str]) -> list[str]:
    return normalize_dataset_source_local_paths(local_paths)


def mix_datasets(req: DatasetMixRequest) -> DatasetMixResponse:
    """Submit a dataset mix job through the local control plane."""
    return get_dataset_mix_control_plane().submit_mix_job(req)


def get_dataset_mix_job(job_id: str) -> DatasetMixResponse:
    """Fetch a submitted dataset mix job."""
    return get_dataset_mix_control_plane().get_mix_job(job_id)


def _read_catalog_env_roots() -> tuple[Path, ...]:
    raw = os.getenv(DATASET_CATALOG_LOCAL_ROOTS_ENV, "").strip()
    if not raw:
        return ()
    roots: list[Path] = []
    for entry in raw.split(PATH_LIST_SEPARATOR):
        normalized = entry.strip()
        if normalized:
            roots.append(Path(normalized).expanduser().resolve(strict=False))
    return tuple(dict.fromkeys(roots))


def _resolve_catalog_roots(local_roots: Sequence[Path] | None = None) -> tuple[Path, ...]:
    if local_roots is not None:
        roots = [root.expanduser().resolve(strict=False) for root in local_roots]
    else:
        roots = [
            *DATASET_MIX_ALLOWED_LOCAL_ROOTS,
            *_read_catalog_env_roots(),
        ]
    return tuple(dict.fromkeys(roots))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_lerobot_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / LEROBOT_META_DIRNAME / TELEOP_REPLAY_INFO_FILENAME).is_file()
        and (path / LEROBOT_DATA_DIRNAME).is_dir()
    )


def _within_scan_depth(root: Path, dataset_root: Path) -> bool:
    try:
        relative_parts = dataset_root.relative_to(root).parts
    except ValueError:
        return False
    return len(relative_parts) <= DATASET_CATALOG_MAX_SCAN_DEPTH


def _iter_lerobot_dataset_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if _is_lerobot_dataset_root(root):
        return [root.resolve(strict=False)]

    discovered: list[Path] = []
    seen: set[Path] = set()
    for info_path in root.glob(f"**/{LEROBOT_META_DIRNAME}/{TELEOP_REPLAY_INFO_FILENAME}"):
        dataset_root = info_path.parent.parent.resolve(strict=False)
        if dataset_root in seen:
            continue
        if not _within_scan_depth(root, dataset_root):
            continue
        if not _is_lerobot_dataset_root(dataset_root):
            continue
        seen.add(dataset_root)
        discovered.append(dataset_root)
        if len(discovered) >= DATASET_CATALOG_MAX_ITEMS:
            break
    return discovered


def _dataset_created_at(path: Path) -> str | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _build_catalog_item(dataset_root: Path) -> DatasetCatalogItem:
    info = _read_json_object(dataset_root / LEROBOT_META_DIRNAME / TELEOP_REPLAY_INFO_FILENAME)
    replay_meta = _read_json_object(dataset_root / LEROBOT_META_DIRNAME / TELEOP_REPLAY_META_FILENAME)
    recording_id = replay_meta.get("recording_id")
    name = str(recording_id).strip() if isinstance(recording_id, str) else ""
    if not name:
        name = dataset_root.name or str(dataset_root)

    return DatasetCatalogItem(
        id=str(dataset_root),
        name=name,
        source="studio_export" if replay_meta else "local",
        path=str(dataset_root),
        format_version=(
            str(info["dataset_format_version"])
            if isinstance(info.get("dataset_format_version"), str)
            else None
        ),
        robot_type=(
            str(info["robot_type"]) if isinstance(info.get("robot_type"), str) else None
        ),
        total_episodes=_safe_int(info.get("total_episodes")),
        total_frames=_safe_int(info.get("total_frames")),
        fps=_safe_float(info.get("fps")),
        export_mode=(
            str(replay_meta["export_mode"])
            if isinstance(replay_meta.get("export_mode"), str)
            else None
        ),
        recording_id=name if replay_meta else None,
        created_at=_dataset_created_at(dataset_root),
    )


def list_dataset_catalog(
    *,
    local_roots: Sequence[Path] | None = None,
) -> DatasetCatalogResponse:
    """List local LeRobot datasets visible to URDF Ops."""
    roots = _resolve_catalog_roots(local_roots)
    items: list[DatasetCatalogItem] = []
    seen_paths: set[str] = set()

    for root in roots:
        for dataset_root in _iter_lerobot_dataset_roots(root):
            dataset_path = str(dataset_root)
            if dataset_path in seen_paths:
                continue
            seen_paths.add(dataset_path)
            items.append(_build_catalog_item(dataset_root))
            if len(items) >= DATASET_CATALOG_MAX_ITEMS:
                break
        if len(items) >= DATASET_CATALOG_MAX_ITEMS:
            break

    items.sort(key=lambda item: item.created_at or "", reverse=True)
    return DatasetCatalogResponse(
        datasets=items,
        roots=[str(root) for root in roots],
    )

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import HTTPException

from backend.models.datasets import (
    DatasetMixJobManifest,
    DatasetMixPartitionManifest,
    DatasetMixPartitionPlan,
    DatasetMixPartitionRef,
    DatasetMixPartitionTask,
    DatasetMixSourceRef,
)
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_CODEBASE_VERSION,
    DATASET_MIX_LEROBOT_DATA_PATH_TEMPLATE,
    DATASET_MIX_LEROBOT_DATA_ROWS_PER_CHUNK,
    DATASET_MIX_LEROBOT_DEFAULT_ROBOT_TYPE,
    DATASET_MIX_LEROBOT_DEFAULT_SPLIT_NAME,
    DATASET_MIX_LEROBOT_DEFAULT_TASK_PREFIX,
    DATASET_MIX_LEROBOT_DEFAULT_VIDEO_KEY,
    DATASET_MIX_LEROBOT_EPISODES_PER_CHUNK,
    DATASET_MIX_LEROBOT_FORMAT_VERSION,
    DATASET_MIX_LEROBOT_INFO_FILENAME,
    DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX,
    DATASET_MIX_LEROBOT_PARTITION_DATA_FILENAME,
    DATASET_MIX_LEROBOT_PARTITION_EPISODES_FILENAME,
    DATASET_MIX_LEROBOT_PARTITION_MANIFEST_FILENAME,
    DATASET_MIX_LEROBOT_PARTITION_MANIFEST_VERSION,
    DATASET_MIX_LEROBOT_PARTITIONS_DIRNAME,
    DATASET_MIX_LEROBOT_STATS_FILENAME,
    DATASET_MIX_LEROBOT_STATS_QUANTILES,
    DATASET_MIX_LEROBOT_STAGING_DIRNAME,
    DATASET_MIX_LEROBOT_TASKS_FILENAME,
    DATASET_MIX_LEROBOT_VIDEO_MODE_NONE,
    DATASET_MIX_LEROBOT_VIDEO_MODE_PRESERVE,
    DATASET_MIX_LEROBOT_VIDEO_MODE_STRIP_MISSING_LOCAL,
    DATASET_MIX_LEROBOT_VIDEO_PATH_TEMPLATE,
    format_dataset_mix_lerobot_index,
)

SCALAR_STATS_KEYS = (
    "frame_index",
    "timestamp",
    "task_index",
    "index",
    "episode_index",
)
VECTOR_STATS_KEYS = ("action", "observation.state")
DATASET_MIX_EPISODES_DIRNAME = "episodes"
DATASET_MIX_META_DIRNAME = "meta"
DATASET_MIX_DATA_DIRNAME = "data"


def _ensure_record(value: Any, *, detail: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail=detail)
    return value


def _read_json_file(path: Path, *, detail: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=400, detail=detail)
    try:
        return _ensure_record(
            json.loads(path.read_text(encoding="utf-8")),
            detail=detail,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=detail) from exc


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _read_parquet_table(path: Path) -> pa.Table:
    return pq.read_table(path)


def _sorted_parquet_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(root.glob("*.parquet")):
        rows.extend(_read_parquet_rows(parquet_path))
    return rows


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field_name} must be numeric")
    if isinstance(value, (int, float)):
        return int(value)
    raise HTTPException(status_code=400, detail=f"{field_name} must be numeric")


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field_name} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    raise HTTPException(status_code=400, detail=f"{field_name} must be numeric")


def _coerce_float_vector(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a numeric list")
    return [_coerce_float(item, field_name=field_name) for item in value]


def _maybe_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_parquet_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _table_from_rows_with_schema(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    normalized_rows = [
        {
            field.name: row.get(field.name)
            for field in schema
        }
        for row in rows
    ]
    return pa.Table.from_pylist(normalized_rows, schema=schema)


def _build_tasks_table(index_to_task: dict[int, str]) -> pa.Table:
    items = sorted(index_to_task.items())
    return pa.table(
        {
            "task": pa.array([task for _, task in items], type=pa.string()),
            "task_index": pa.array([index for index, _ in items], type=pa.int64()),
        }
    )


def _build_episode_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return _table_from_rows_with_schema(rows, schema)


def _build_data_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return _table_from_rows_with_schema(rows, schema)


def _resolve_quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0 for _ in DATASET_MIX_LEROBOT_STATS_QUANTILES]
    sorted_values = sorted(values)
    resolved: list[float] = []
    for quantile in DATASET_MIX_LEROBOT_STATS_QUANTILES:
        quantile_index = int(len(sorted_values) * quantile)
        resolved.append(sorted_values[min(quantile_index, len(sorted_values) - 1)])
    return resolved


def _build_scalar_stats(values: list[float]) -> dict[str, list[float]]:
    if not values:
        quantiles = [0.0 for _ in DATASET_MIX_LEROBOT_STATS_QUANTILES]
        return {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0.0],
            "q01": [quantiles[0]],
            "q10": [quantiles[1]],
            "q50": [quantiles[2]],
            "q90": [quantiles[3]],
            "q99": [quantiles[4]],
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    quantiles = _resolve_quantiles(values)
    return {
        "min": [min(values)],
        "max": [max(values)],
        "mean": [mean],
        "std": [variance ** 0.5],
        "count": [float(len(values))],
        "q01": [quantiles[0]],
        "q10": [quantiles[1]],
        "q50": [quantiles[2]],
        "q90": [quantiles[3]],
        "q99": [quantiles[4]],
    }


def _build_vector_stats(vectors: list[list[float]]) -> dict[str, list[float]]:
    if not vectors:
        return {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "count": [],
            "q01": [],
            "q10": [],
            "q50": [],
            "q90": [],
            "q99": [],
        }
    dimension_count = len(vectors[0])
    dimensions = [[vector[index] for vector in vectors] for index in range(dimension_count)]
    scalar_stats = [_build_scalar_stats(values) for values in dimensions]
    return {
        "min": [stats["min"][0] for stats in scalar_stats],
        "max": [stats["max"][0] for stats in scalar_stats],
        "mean": [stats["mean"][0] for stats in scalar_stats],
        "std": [stats["std"][0] for stats in scalar_stats],
        "count": [stats["count"][0] for stats in scalar_stats],
        "q01": [stats["q01"][0] for stats in scalar_stats],
        "q10": [stats["q10"][0] for stats in scalar_stats],
        "q50": [stats["q50"][0] for stats in scalar_stats],
        "q90": [stats["q90"][0] for stats in scalar_stats],
        "q99": [stats["q99"][0] for stats in scalar_stats],
    }


@dataclass(frozen=True)
class LocalLeRobotEpisodeSpan:
    episode_index: int
    row_from_index: int
    row_to_index: int


@dataclass(frozen=True)
class LocalLeRobotSourceDataset:
    source: DatasetMixSourceRef
    treatment_source: dict[str, Any] | None
    root: Path
    info: dict[str, Any]
    stats: dict[str, Any]
    tasks_by_index: dict[int, str]
    episode_rows_by_index: dict[int, dict[str, Any]]
    episode_schema: pa.Schema
    frame_rows: list[dict[str, Any]]
    episode_spans: tuple[LocalLeRobotEpisodeSpan, ...]
    frame_schema: pa.Schema
    total_frame_rows: int


@dataclass(frozen=True)
class LocalLeRobotCompatibility:
    representation_id: str
    naming_status: str
    robot_type: str
    embodiment_id: str | None
    fps: float
    features: dict[str, Any]
    episode_schema: pa.Schema
    frame_schema: pa.Schema


@dataclass(frozen=True)
class LocalLeRobotVideoPolicy:
    mode: str
    preserved_video_keys: list[str]
    stripped_video_keys: list[str]
    referenced_video_file_count: int
    copied_video_file_count: int = 0


def _read_source_tasks(root: Path) -> dict[int, str]:
    tasks_path = root / DATASET_MIX_META_DIRNAME / DATASET_MIX_LEROBOT_TASKS_FILENAME
    if not tasks_path.exists():
        return {}
    task_rows = _read_parquet_rows(tasks_path)
    tasks_by_index: dict[int, str] = {}
    for row in task_rows:
        task_name = _resolve_task_name_row(row)
        task_index = row.get("task_index")
        if isinstance(task_name, str) and task_name and isinstance(task_index, (int, float)):
            tasks_by_index[int(task_index)] = task_name
    return tasks_by_index


def _read_source_stats(root: Path) -> dict[str, Any]:
    stats_path = root / DATASET_MIX_META_DIRNAME / DATASET_MIX_LEROBOT_STATS_FILENAME
    if not stats_path.exists():
        return {}
    return _read_json_file(
        stats_path,
        detail=f"Dataset root is missing valid meta/{DATASET_MIX_LEROBOT_STATS_FILENAME}: {root}",
    )


def _resolve_task_name_row(row: dict[str, Any]) -> str | None:
    explicit_task_name = _maybe_string(row.get("task"))
    if explicit_task_name:
        return explicit_task_name
    for field_name, field_value in row.items():
        if field_name == "task_index":
            continue
        alternate_task_name = _maybe_string(field_value)
        if alternate_task_name:
            return alternate_task_name
    return None


def _read_source_frame_rows(root: Path) -> tuple[list[dict[str, Any]], pa.Schema, int]:
    data_root = root / DATASET_MIX_DATA_DIRNAME
    if not data_root.exists():
        raise HTTPException(status_code=400, detail=f"Dataset root is missing data directory: {root}")
    frame_schema: pa.Schema | None = None
    rows: list[dict[str, Any]] = []
    total_frame_rows = 0
    for chunk_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for parquet_path in sorted(chunk_dir.glob("*.parquet")):
            table = _read_parquet_table(parquet_path)
            if frame_schema is None:
                frame_schema = table.schema
            elif frame_schema != table.schema:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset frame parquet schema is inconsistent within source: {root}",
                )
            table_rows = table.to_pylist()
            rows.extend(table_rows)
            total_frame_rows += len(table_rows)
    if not rows or total_frame_rows == 0:
        raise HTTPException(status_code=400, detail=f"Dataset root does not contain any parquet frame rows: {root}")
    rows.sort(
        key=lambda row: (
            _coerce_int(row.get("episode_index", 0), field_name="episode_index"),
            _coerce_int(row.get("frame_index", 0), field_name="frame_index"),
            _coerce_int(row.get("index", 0), field_name="index"),
        )
    )
    if frame_schema is None:
        raise HTTPException(status_code=400, detail=f"Dataset root does not contain any readable frame parquet tables: {root}")
    return rows, frame_schema, total_frame_rows


def _read_source_episode_rows(
    root: Path,
) -> tuple[dict[int, dict[str, Any]], pa.Schema]:
    episodes_root = root / DATASET_MIX_META_DIRNAME / DATASET_MIX_EPISODES_DIRNAME
    if not episodes_root.exists():
        raise HTTPException(status_code=400, detail=f"Dataset root is missing meta/episodes directory: {root}")
    episode_schema: pa.Schema | None = None
    rows_by_index: dict[int, dict[str, Any]] = {}
    for parquet_path in sorted(episodes_root.rglob("*.parquet")):
        table = _read_parquet_table(parquet_path)
        if episode_schema is None:
            episode_schema = table.schema
        elif episode_schema != table.schema:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset episode parquet schema is inconsistent within source: {root}",
            )
        for row in table.to_pylist():
            episode_index = _coerce_int(
                row.get("episode_index", 0),
                field_name="episode_index",
            )
            if episode_index in rows_by_index:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset episode parquet contains duplicate episode_index rows: "
                        f"{episode_index} in {root}"
                    ),
                )
            rows_by_index[episode_index] = row
    if episode_schema is None or not rows_by_index:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset root does not contain any readable episode parquet rows: {root}",
        )
    return rows_by_index, episode_schema


def _build_episode_spans(
    frame_rows: list[dict[str, Any]],
) -> tuple[LocalLeRobotEpisodeSpan, ...]:
    if not frame_rows:
        return ()
    spans: list[LocalLeRobotEpisodeSpan] = []
    current_episode_index = _coerce_int(
        frame_rows[0].get("episode_index", 0),
        field_name="episode_index",
    )
    span_start_index = 0
    for row_index, row in enumerate(frame_rows[1:], start=1):
        episode_index = _coerce_int(row.get("episode_index", 0), field_name="episode_index")
        if episode_index == current_episode_index:
            continue
        spans.append(
            LocalLeRobotEpisodeSpan(
                episode_index=current_episode_index,
                row_from_index=span_start_index,
                row_to_index=row_index,
            )
        )
        current_episode_index = episode_index
        span_start_index = row_index
    spans.append(
        LocalLeRobotEpisodeSpan(
            episode_index=current_episode_index,
            row_from_index=span_start_index,
            row_to_index=len(frame_rows),
        )
    )
    return tuple(spans)


def _validate_source_dataset_contract(dataset: LocalLeRobotSourceDataset) -> None:
    span_episode_indexes = {span.episode_index for span in dataset.episode_spans}

    if span_episode_indexes != set(dataset.episode_rows_by_index):
        missing_episode_rows = sorted(span_episode_indexes - set(dataset.episode_rows_by_index))
        orphan_episode_rows = sorted(set(dataset.episode_rows_by_index) - span_episode_indexes)
        details: list[str] = []
        if missing_episode_rows:
            details.append(f"missing metadata rows for episodes {missing_episode_rows[:5]}")
        if orphan_episode_rows:
            details.append(f"episodes without frame rows {orphan_episode_rows[:5]}")
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset episode/frame contract is inconsistent: "
                + ", ".join(details)
                + f" ({dataset.root})"
            ),
        )

    total_frame_rows = 0
    expected_next_dataset_from_index: int | None = None
    for span in dataset.episode_spans:
        episode_index = span.episode_index
        episode_row = dataset.episode_rows_by_index[episode_index]
        source_row_count = span.row_to_index - span.row_from_index
        if source_row_count <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset episode has no frame rows: {episode_index} in {dataset.root}",
            )
        first_source_row = dataset.frame_rows[span.row_from_index]
        last_source_row = dataset.frame_rows[span.row_to_index - 1]
        total_frame_rows += source_row_count
        episode_length = _coerce_int(episode_row.get("length", source_row_count), field_name="length")
        if episode_length != source_row_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset episode length does not match frame row count: "
                    f"episode {episode_index} expects {episode_length} rows but found {source_row_count} in {dataset.root}"
                ),
            )
        dataset_from_index = _coerce_int(
            episode_row.get("dataset_from_index", first_source_row["index"]),
            field_name="dataset_from_index",
        )
        dataset_to_index = _coerce_int(
            episode_row.get("dataset_to_index", last_source_row["index"] + 1),
            field_name="dataset_to_index",
        )
        if (
            expected_next_dataset_from_index is not None
            and dataset_from_index != expected_next_dataset_from_index
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset episode dataset index ranges are not globally contiguous: "
                    f"episode {episode_index} in {dataset.root}"
                ),
            )
        if dataset_to_index - dataset_from_index != source_row_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset frame indexes are not contiguous within episode metadata bounds: "
                    f"episode {episode_index} in {dataset.root}"
                ),
            )
        for frame_offset, row_index in enumerate(range(span.row_from_index, span.row_to_index)):
            source_row = dataset.frame_rows[row_index]
            source_index = _coerce_int(source_row.get("index", 0), field_name="index")
            expected_index = dataset_from_index + frame_offset
            if source_index != expected_index:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset frame indexes are not contiguous within episode metadata bounds: "
                        f"episode {episode_index} in {dataset.root}"
                    ),
                )
            source_frame_index = _coerce_int(
                source_row.get("frame_index", frame_offset),
                field_name="frame_index",
            )
            if source_frame_index != frame_offset:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset frame_index values are not contiguous within the episode: "
                        f"episode {episode_index} in {dataset.root}"
                    ),
                )
        expected_next_dataset_from_index = dataset_to_index

    if total_frame_rows != dataset.total_frame_rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset total frame coverage does not match episode metadata coverage: "
                f"{dataset.root}"
            ),
        )

    info_total_episodes = dataset.info.get("total_episodes")
    if isinstance(info_total_episodes, (int, float)) and int(info_total_episodes) != len(dataset.episode_rows_by_index):
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset info total_episodes does not match actual episode metadata rows: "
                f"{dataset.root}"
            ),
        )
    info_total_frames = dataset.info.get("total_frames")
    if isinstance(info_total_frames, (int, float)) and int(info_total_frames) != len(dataset.frame_rows):
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset info total_frames does not match actual frame row count: "
                f"{dataset.root}"
            ),
        )


def _load_local_source_dataset(
    source: DatasetMixSourceRef,
    treatment_source: dict[str, Any] | None,
) -> LocalLeRobotSourceDataset:
    root = Path(source.canonical_source).resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Local dataset source is not a directory: {source.canonical_source}")
    info = _read_json_file(
        root / DATASET_MIX_META_DIRNAME / DATASET_MIX_LEROBOT_INFO_FILENAME,
        detail=f"Dataset root is missing valid meta/{DATASET_MIX_LEROBOT_INFO_FILENAME}: {root}",
    )
    episode_rows_by_index, episode_schema = _read_source_episode_rows(root)
    frame_rows, frame_schema, total_frame_rows = _read_source_frame_rows(root)
    episode_spans = _build_episode_spans(frame_rows)
    dataset = LocalLeRobotSourceDataset(
        source=source,
        treatment_source=treatment_source,
        root=root,
        info=info,
        stats=_read_source_stats(root),
        tasks_by_index=_read_source_tasks(root),
        episode_rows_by_index=episode_rows_by_index,
        episode_schema=episode_schema,
        frame_rows=frame_rows,
        episode_spans=episode_spans,
        frame_schema=frame_schema,
        total_frame_rows=total_frame_rows,
    )
    _validate_source_dataset_contract(dataset)
    return dataset


def _resolve_source_robot_type(dataset: LocalLeRobotSourceDataset) -> str:
    return _maybe_string(dataset.info.get("robot_type")) or DATASET_MIX_LEROBOT_DEFAULT_ROBOT_TYPE


def _resolve_source_representation_id(dataset: LocalLeRobotSourceDataset) -> str:
    representation_id = _maybe_string(dataset.info.get("representation_id"))
    if representation_id:
        return representation_id
    if dataset.treatment_source and _maybe_string(dataset.treatment_source.get("representation_id")):
        return str(dataset.treatment_source["representation_id"])
    raise HTTPException(
        status_code=400,
        detail=f"Dataset source is missing representation_id metadata: {dataset.root}",
    )


def _resolve_source_naming_status(dataset: LocalLeRobotSourceDataset) -> str:
    naming_status = _maybe_string(dataset.info.get("naming_status"))
    if naming_status:
        return naming_status
    if dataset.treatment_source and _maybe_string(dataset.treatment_source.get("naming_status")):
        return str(dataset.treatment_source["naming_status"])
    return "named"


def _resolve_source_embodiment_id(dataset: LocalLeRobotSourceDataset) -> str | None:
    embodiment_ref = dataset.info.get("embodiment_ref")
    if isinstance(embodiment_ref, dict) and _maybe_string(embodiment_ref.get("embodiment_id")):
        return str(embodiment_ref["embodiment_id"])
    if dataset.treatment_source and _maybe_string(dataset.treatment_source.get("embodiment_id")):
        return str(dataset.treatment_source["embodiment_id"])
    return None


def _resolve_source_fps(dataset: LocalLeRobotSourceDataset) -> float:
    fps = dataset.info.get("fps")
    if isinstance(fps, (int, float)):
        return float(fps)
    raise HTTPException(status_code=400, detail=f"Dataset source is missing fps metadata: {dataset.root}")


def _resolve_source_feature_spec(dataset: LocalLeRobotSourceDataset, sample_row: dict[str, Any]) -> dict[str, Any]:
    features = dataset.info.get("features")
    if isinstance(features, dict) and features:
        return features
    action_vector = _coerce_float_vector(sample_row.get("action", []), field_name="action")
    joint_names = [f"joint_{index}" for index in range(len(action_vector))]
    return {
        "action": {
            "dtype": "float32",
            "names": [f"{joint_name}.pos" for joint_name in joint_names],
            "shape": [len(joint_names)],
        },
        "observation.state": {
            "dtype": "float32",
            "names": [f"{joint_name}.pos" for joint_name in joint_names],
            "shape": [len(joint_names)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


INFO_VARIANT_KEYS = {
    "total_episodes",
    "total_frames",
    "total_tasks",
    "chunks_size",
    "data_files_size_in_mb",
    "video_files_size_in_mb",
    "splits",
}


def _canonicalize_info_contract(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if key not in INFO_VARIANT_KEYS
    }


def _validate_source_compatibility(
    datasets: list[LocalLeRobotSourceDataset],
    manifest: DatasetMixJobManifest,
) -> LocalLeRobotCompatibility:
    first_dataset = datasets[0]
    first_sample_row = first_dataset.frame_rows[0]
    first_representation_id = _resolve_source_representation_id(first_dataset)
    if manifest.required_representation_id and first_representation_id != manifest.required_representation_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Local dataset source representation does not match manifest requirement: "
                f"{first_representation_id} != {manifest.required_representation_id}"
            ),
        )
    first_naming_status = _resolve_source_naming_status(first_dataset)
    first_robot_type = _resolve_source_robot_type(first_dataset)
    first_embodiment_id = _resolve_source_embodiment_id(first_dataset)
    first_fps = _resolve_source_fps(first_dataset)
    first_features = _resolve_source_feature_spec(first_dataset, first_sample_row)
    first_episode_schema = first_dataset.episode_schema
    first_frame_schema = first_dataset.frame_schema
    first_info_contract = _canonicalize_info_contract(first_dataset.info)
    first_action_width = len(_coerce_float_vector(first_sample_row.get("action", []), field_name="action"))
    for dataset in datasets[1:]:
        sample_row = dataset.frame_rows[0]
        action_width = len(_coerce_float_vector(sample_row.get("action", []), field_name="action"))
        if action_width != first_action_width:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same action width")
        if _resolve_source_representation_id(dataset) != first_representation_id:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same representation_id")
        if _resolve_source_naming_status(dataset) != first_naming_status:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same naming_status")
        if _resolve_source_robot_type(dataset) != first_robot_type:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same robot_type")
        if _resolve_source_embodiment_id(dataset) != first_embodiment_id:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same embodiment_id")
        if _resolve_source_fps(dataset) != first_fps:
            raise HTTPException(status_code=400, detail="Local dataset sources must share the same fps")
        if dataset.episode_schema != first_episode_schema:
            raise HTTPException(
                status_code=400,
                detail="Local dataset sources must share the same episode parquet schema",
            )
        if dataset.frame_schema != first_frame_schema:
            raise HTTPException(
                status_code=400,
                detail="Local dataset sources must share the same frame parquet schema",
            )
        if _resolve_source_feature_spec(dataset, sample_row) != first_features:
            raise HTTPException(
                status_code=400,
                detail="Local dataset sources must share the same info.features contract",
            )
        if _canonicalize_info_contract(dataset.info) != first_info_contract:
            raise HTTPException(
                status_code=400,
                detail="Local dataset sources must share the same LeRobot info contract",
            )
    return LocalLeRobotCompatibility(
        representation_id=first_representation_id,
        naming_status=first_naming_status,
        robot_type=first_robot_type,
        embodiment_id=first_embodiment_id,
        fps=first_fps,
        features=first_features,
        episode_schema=first_episode_schema,
        frame_schema=first_frame_schema,
    )


def _task_name_for_index(dataset: LocalLeRobotSourceDataset, task_index: int) -> str:
    return dataset.tasks_by_index.get(task_index, f"{DATASET_MIX_LEROBOT_DEFAULT_TASK_PREFIX}-{task_index}")


def _resolve_episode_task_names(
    dataset: LocalLeRobotSourceDataset,
    episode_row: dict[str, Any],
) -> list[str]:
    raw_tasks = episode_row.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    resolved_task_names: list[str] = []
    for task_entry in raw_tasks:
        if isinstance(task_entry, str) and task_entry:
            resolved_task_names.append(task_entry)
            continue
        if isinstance(task_entry, (int, float)) and not isinstance(task_entry, bool):
            resolved_task_names.append(_task_name_for_index(dataset, int(task_entry)))
    return resolved_task_names


def _resolve_output_info(
    *,
    base_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    output_info = dict(base_info)
    output_info["codebase_version"] = output_info.get(
        "codebase_version",
        DATASET_MIX_LEROBOT_CODEBASE_VERSION,
    )
    output_info["dataset_format_version"] = output_info.get(
        "dataset_format_version",
        DATASET_MIX_LEROBOT_FORMAT_VERSION,
    )
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = total_tasks
    output_info["chunks_size"] = DATASET_MIX_LEROBOT_DATA_ROWS_PER_CHUNK
    output_info["data_path"] = DATASET_MIX_LEROBOT_DATA_PATH_TEMPLATE
    output_info["video_path"] = output_info.get(
        "video_path",
        DATASET_MIX_LEROBOT_VIDEO_PATH_TEMPLATE,
    )
    output_info["splits"] = {
        DATASET_MIX_LEROBOT_DEFAULT_SPLIT_NAME: f"0:{total_episodes}",
    }
    return output_info


def _apply_video_policy_to_info_contract(
    info: dict[str, Any],
    *,
    features: dict[str, Any],
    video_policy: LocalLeRobotVideoPolicy,
) -> dict[str, Any]:
    patched = dict(info)
    patched["features"] = features
    if video_policy.preserved_video_keys:
        patched["video_path"] = patched.get(
            "video_path",
            DATASET_MIX_LEROBOT_VIDEO_PATH_TEMPLATE,
        )
        return patched
    patched["video_path"] = ""
    return patched


def _sequence_stats(
    values: list[int],
) -> dict[str, list[float | int]]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size == 0:
        return {
            "min": [0],
            "max": [0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
        }
    return {
        "min": [int(numeric.min())],
        "max": [int(numeric.max())],
        "mean": [float(numeric.mean())],
        "std": [float(numeric.std())],
        "count": [int(numeric.size)],
    }


def _set_episode_stats(
    row: dict[str, Any],
    feature_key: str,
    stats: dict[str, list[float | int]],
) -> None:
    for stat_name, stat_value in stats.items():
        row[f"stats/{feature_key}/{stat_name}"] = stat_value


def _infer_video_keys(features: dict[str, Any]) -> list[str]:
    video_keys: list[str] = []
    for key, spec in features.items():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            video_keys.append(key)
    return sorted(video_keys)


def _source_video_path(
    root: Path,
    *,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return (
        root
        / "videos"
        / video_key
        / f"chunk-{format_dataset_mix_lerobot_index(chunk_index)}"
        / f"file-{format_dataset_mix_lerobot_index(file_index)}.mp4"
    )


def _strip_video_feature_specs(
    features: dict[str, Any],
    stripped_video_keys: list[str],
) -> dict[str, Any]:
    stripped_key_set = set(stripped_video_keys)
    return {
        key: value
        for key, value in features.items()
        if key not in stripped_key_set
    }


def _strip_video_stats_entries(
    stats: dict[str, Any],
    stripped_video_keys: list[str],
) -> dict[str, Any]:
    stripped_key_set = set(stripped_video_keys)
    return {
        key: value
        for key, value in stats.items()
        if key not in stripped_key_set
    }


def _strip_episode_video_fields(
    row: dict[str, Any],
    stripped_video_keys: list[str],
) -> dict[str, Any]:
    if not stripped_video_keys:
        return row
    stripped_prefixes = tuple(f"videos/{video_key}/" for video_key in stripped_video_keys)
    stripped_stat_prefixes = tuple(f"stats/{video_key}/" for video_key in stripped_video_keys)
    return {
        key: value
        for key, value in row.items()
        if not key.startswith(stripped_prefixes)
        and not key.startswith(stripped_stat_prefixes)
    }


def _strip_video_fields_from_schema(
    schema: pa.Schema,
    stripped_video_keys: list[str],
) -> pa.Schema:
    if not stripped_video_keys:
        return schema
    stripped_prefixes = tuple(f"videos/{video_key}/" for video_key in stripped_video_keys)
    stripped_stat_prefixes = tuple(f"stats/{video_key}/" for video_key in stripped_video_keys)
    return pa.schema(
        [
            field
            for field in schema
            if not field.name.startswith(stripped_prefixes)
            and not field.name.startswith(stripped_stat_prefixes)
        ],
        metadata=schema.metadata,
    )


def _resolve_referenced_video_files(
    dataset: LocalLeRobotSourceDataset,
    *,
    video_key: str,
) -> set[tuple[int, int]]:
    references: set[tuple[int, int]] = set()
    chunk_field = f"videos/{video_key}/chunk_index"
    file_field = f"videos/{video_key}/file_index"
    for episode_row in dataset.episode_rows_by_index.values():
        source_chunk = episode_row.get(chunk_field)
        source_file = episode_row.get(file_field)
        if source_chunk is None or source_file is None:
            continue
        references.add(
            (
                _coerce_int(source_chunk, field_name=chunk_field),
                _coerce_int(source_file, field_name=file_field),
            )
        )
    return references


def _resolve_video_policy(
    datasets: list[LocalLeRobotSourceDataset],
    declared_video_keys: list[str],
) -> LocalLeRobotVideoPolicy:
    if not declared_video_keys:
        return LocalLeRobotVideoPolicy(
            mode=DATASET_MIX_LEROBOT_VIDEO_MODE_NONE,
            preserved_video_keys=[],
            stripped_video_keys=[],
            referenced_video_file_count=0,
        )

    preserved_video_keys: list[str] = []
    stripped_video_keys: list[str] = []
    referenced_video_file_count = 0
    for video_key in declared_video_keys:
        key_has_present_assets = False
        key_has_missing_assets = False
        first_missing_path: Path | None = None
        for dataset in datasets:
            references = _resolve_referenced_video_files(dataset, video_key=video_key)
            referenced_video_file_count += len(references)
            if not references:
                continue
            missing_reference = next(
                (
                    (chunk_index, file_index)
                    for chunk_index, file_index in sorted(references)
                    if not _source_video_path(
                        dataset.root,
                        video_key=video_key,
                        chunk_index=chunk_index,
                        file_index=file_index,
                    ).exists()
                ),
                None,
            )
            if missing_reference is None:
                key_has_present_assets = True
                continue
            key_has_missing_assets = True
            if first_missing_path is None:
                first_missing_path = _source_video_path(
                    dataset.root,
                    video_key=video_key,
                    chunk_index=missing_reference[0],
                    file_index=missing_reference[1],
                )
        if key_has_present_assets and key_has_missing_assets:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Local dataset sources have inconsistent video asset availability for "
                    f"{video_key}. First missing shard: {first_missing_path}"
                ),
            )
        if key_has_present_assets:
            preserved_video_keys.append(video_key)
        else:
            stripped_video_keys.append(video_key)

    mode = (
        DATASET_MIX_LEROBOT_VIDEO_MODE_PRESERVE
        if preserved_video_keys
        else DATASET_MIX_LEROBOT_VIDEO_MODE_STRIP_MISSING_LOCAL
    )
    return LocalLeRobotVideoPolicy(
        mode=mode,
        preserved_video_keys=preserved_video_keys,
        stripped_video_keys=stripped_video_keys,
        referenced_video_file_count=referenced_video_file_count,
    )


def _copy_video_file(
    *,
    dataset: LocalLeRobotSourceDataset,
    output_root: Path,
    video_key: str,
    source_chunk_index: int,
    source_file_index: int,
    assigned_output_chunk_index: int,
) -> None:
    source_path = _source_video_path(
        dataset.root,
        video_key=video_key,
        chunk_index=source_chunk_index,
        file_index=source_file_index,
    )
    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset source is missing required video shard: {source_path}",
        )
    output_path = (
        output_root
        / "videos"
        / video_key
        / f"chunk-{format_dataset_mix_lerobot_index(assigned_output_chunk_index)}"
        / f"file-{format_dataset_mix_lerobot_index(DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX)}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        shutil.copy2(source_path, output_path)


def _patch_episode_video_refs(
    *,
    dataset: LocalLeRobotSourceDataset,
    output_root: Path,
    row: dict[str, Any],
    video_keys: list[str],
    video_file_map: dict[tuple[str, str, int, int], int],
    next_video_chunk_index_by_key: dict[str, int],
) -> None:
    for video_key in video_keys:
        chunk_field = f"videos/{video_key}/chunk_index"
        file_field = f"videos/{video_key}/file_index"
        source_chunk = row.get(chunk_field)
        source_file = row.get(file_field)
        if source_chunk is None or source_file is None:
            continue
        source_chunk_index = _coerce_int(source_chunk, field_name=chunk_field)
        source_file_index = _coerce_int(source_file, field_name=file_field)
        map_key = (
            str(dataset.root),
            video_key,
            source_chunk_index,
            source_file_index,
        )
        assigned_chunk_index = video_file_map.get(map_key)
        if assigned_chunk_index is None:
            assigned_chunk_index = next_video_chunk_index_by_key.get(video_key, 0)
            next_video_chunk_index_by_key[video_key] = assigned_chunk_index + 1
            video_file_map[map_key] = assigned_chunk_index
            _copy_video_file(
                dataset=dataset,
                output_root=output_root,
                video_key=video_key,
                source_chunk_index=source_chunk_index,
                source_file_index=source_file_index,
                assigned_output_chunk_index=assigned_chunk_index,
            )
        row[chunk_field] = assigned_chunk_index
        row[file_field] = DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX


def _merge_stats_entry_arrays(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_count = np.asarray(left["count"], dtype=np.float64)
    right_count = np.asarray(right["count"], dtype=np.float64)
    total_count = left_count + right_count
    left_mean = np.asarray(left["mean"], dtype=np.float64)
    right_mean = np.asarray(right["mean"], dtype=np.float64)
    left_std = np.asarray(left["std"], dtype=np.float64)
    right_std = np.asarray(right["std"], dtype=np.float64)
    numerator = (
        left_count * (left_std**2 + left_mean**2)
        + right_count * (right_std**2 + right_mean**2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(
            left_count * left_mean + right_count * right_mean,
            total_count,
            out=np.zeros_like(left_mean, dtype=np.float64),
            where=total_count > 0,
        )
        variance = np.divide(
            numerator,
            total_count,
            out=np.zeros_like(left_mean, dtype=np.float64),
            where=total_count > 0,
        ) - mean**2
    variance = np.maximum(variance, 0.0)
    merged_min = np.minimum(
        np.asarray(left["min"], dtype=np.float64),
        np.asarray(right["min"], dtype=np.float64),
    )
    merged_max = np.maximum(
        np.asarray(left["max"], dtype=np.float64),
        np.asarray(right["max"], dtype=np.float64),
    )
    return {
        "min": merged_min.tolist(),
        "max": merged_max.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": total_count.astype(np.int64).tolist(),
    }


def _merge_stats_dicts(datasets: list[LocalLeRobotSourceDataset]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for dataset in datasets:
        for key, stats in dataset.stats.items():
            if key in {"index", "episode_index", "task_index"}:
                continue
            if key not in merged:
                merged[key] = json.loads(json.dumps(stats))
                continue
            merged[key] = _merge_stats_entry_arrays(merged[key], stats)
    return merged


def _replace_stats_entry(
    stats: dict[str, Any],
    *,
    key: str,
    values: list[int],
) -> None:
    stats[key] = _sequence_stats(values)


def _resolve_embodiment_ref(
    compatibility: LocalLeRobotCompatibility,
) -> dict[str, Any] | None:
    return _embodiment_ref_from_id(compatibility.embodiment_id)


def _embodiment_ref_from_id(embodiment_id: str | None) -> dict[str, Any] | None:
    if embodiment_id is None:
        return None
    return {"embodiment_id": embodiment_id}


def _artifact_path_string(path: Path) -> str:
    return str(path)


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _partition_root(staging_root: Path, partition_id: str) -> Path:
    return (
        staging_root
        / DATASET_MIX_LEROBOT_PARTITIONS_DIRNAME
        / partition_id
    )


def _write_partition_output(
    *,
    staging_root: Path,
    partition_id: str,
    episode_rows: list[dict[str, Any]],
    episode_schema: pa.Schema,
    frame_rows: list[dict[str, Any]],
    frame_schema: pa.Schema,
    source_ids: set[str],
    task_indices: set[int],
) -> DatasetMixPartitionRef:
    if not episode_rows or not frame_rows:
        raise HTTPException(status_code=500, detail="Partition output cannot be empty")

    partition_root = _partition_root(staging_root, partition_id)
    episodes_path = partition_root / DATASET_MIX_LEROBOT_PARTITION_EPISODES_FILENAME
    data_path = partition_root / DATASET_MIX_LEROBOT_PARTITION_DATA_FILENAME
    _write_parquet_table(episodes_path, _build_episode_table(episode_rows, episode_schema))
    _write_parquet_table(data_path, _build_data_table(frame_rows, frame_schema))

    first_episode_index = _coerce_int(
        episode_rows[0]["episode_index"],
        field_name="episode_index",
    )
    last_episode_index = _coerce_int(
        episode_rows[-1]["episode_index"],
        field_name="episode_index",
    )
    first_frame_index = _coerce_int(frame_rows[0]["index"], field_name="index")
    last_frame_index = _coerce_int(frame_rows[-1]["index"], field_name="index")
    return DatasetMixPartitionRef(
        partition_id=partition_id,
        source_ids=sorted(source_ids),
        task_indices=sorted(task_indices),
        episode_from_index=first_episode_index,
        episode_to_index=last_episode_index + 1,
        frame_from_index=first_frame_index,
        frame_to_index=last_frame_index + 1,
        episode_count=len(episode_rows),
        frame_count=len(frame_rows),
        episodes_artifact_path=_artifact_path_string(episodes_path),
        data_artifact_path=_artifact_path_string(data_path),
    )


def _build_partition_manifest(
    *,
    partition_plan: DatasetMixPartitionPlan,
    compatibility: LocalLeRobotCompatibility,
    features: dict[str, Any],
    task_name_to_index: dict[str, int],
    partitions: list[DatasetMixPartitionRef],
) -> DatasetMixPartitionManifest:
    return DatasetMixPartitionManifest(
        manifest_version=DATASET_MIX_LEROBOT_PARTITION_MANIFEST_VERSION,
        partition_plan=partition_plan,
        tasks=[
            DatasetMixPartitionTask(task_index=index, task_name=task_name)
            for task_name, index in sorted(task_name_to_index.items(), key=lambda item: item[1])
        ],
        representation_id=compatibility.representation_id,
        naming_status=compatibility.naming_status,
        robot_type=compatibility.robot_type,
        embodiment_id=compatibility.embodiment_id,
        fps=compatibility.fps,
        features=features,
        partitions=partitions,
    )


def _flush_episode_chunk(
    *,
    episodes_root: Path,
    chunk_index: int,
    schema: pa.Schema,
    rows: list[dict[str, Any]],
) -> None:
    _write_parquet_table(
        episodes_root
        / f"chunk-{format_dataset_mix_lerobot_index(chunk_index)}"
        / f"file-{format_dataset_mix_lerobot_index(DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX)}.parquet",
        _build_episode_table(rows, schema),
    )


def _finalize_partitioned_output(
    *,
    output_root: Path,
    base_info: dict[str, Any],
    merged_stats: dict[str, Any],
    episode_schema: pa.Schema,
    partition_manifest: DatasetMixPartitionManifest,
) -> dict[str, Any]:
    meta_root = output_root / DATASET_MIX_META_DIRNAME
    data_root = output_root / DATASET_MIX_DATA_DIRNAME
    episodes_root = meta_root / DATASET_MIX_EPISODES_DIRNAME
    meta_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)

    _write_parquet_table(
        meta_root / DATASET_MIX_LEROBOT_TASKS_FILENAME,
        _build_tasks_table(
            {
                task.task_index: task.task_name
                for task in partition_manifest.tasks
            }
        ),
    )

    pending_episode_rows: list[dict[str, Any]] = []
    next_episode_chunk_index = 0
    next_data_chunk_index = 0

    for partition in partition_manifest.partitions:
        if partition.frame_count > DATASET_MIX_LEROBOT_DATA_ROWS_PER_CHUNK:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Native local LeRobot partition exceeds output data chunk capacity: "
                    f"{partition.partition_id} has {partition.frame_count} frames "
                    f"but the configured chunk limit is {DATASET_MIX_LEROBOT_DATA_ROWS_PER_CHUNK}"
                ),
            )
        output_data_path = (
            data_root
            / f"chunk-{format_dataset_mix_lerobot_index(next_data_chunk_index)}"
            / f"file-{format_dataset_mix_lerobot_index(DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX)}.parquet"
        )
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(partition.data_artifact_path), output_data_path)
        partition_episode_rows = _read_parquet_rows(Path(partition.episodes_artifact_path))
        for episode_row in partition_episode_rows:
            finalized_episode_row = dict(episode_row)
            finalized_episode_row["data/chunk_index"] = next_data_chunk_index
            finalized_episode_row["data/file_index"] = DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX
            finalized_episode_row["meta/episodes/chunk_index"] = next_episode_chunk_index
            finalized_episode_row["meta/episodes/file_index"] = (
                DATASET_MIX_LEROBOT_PRIMARY_FILE_INDEX
            )
            pending_episode_rows.append(finalized_episode_row)

            if len(pending_episode_rows) >= DATASET_MIX_LEROBOT_EPISODES_PER_CHUNK:
                _flush_episode_chunk(
                    episodes_root=episodes_root,
                    chunk_index=next_episode_chunk_index,
                    schema=episode_schema,
                    rows=pending_episode_rows,
                )
                pending_episode_rows = []
                next_episode_chunk_index += 1
        next_data_chunk_index += 1

    if pending_episode_rows:
        _flush_episode_chunk(
            episodes_root=episodes_root,
            chunk_index=next_episode_chunk_index,
            schema=episode_schema,
            rows=pending_episode_rows,
        )

    _write_json_file(meta_root / DATASET_MIX_LEROBOT_STATS_FILENAME, merged_stats)
    _write_json_file(
        meta_root / DATASET_MIX_LEROBOT_INFO_FILENAME,
        _resolve_output_info(
            base_info=base_info,
            total_episodes=sum(partition.episode_count for partition in partition_manifest.partitions),
            total_frames=sum(partition.frame_count for partition in partition_manifest.partitions),
            total_tasks=len(partition_manifest.tasks),
        ),
    )
    return {
        "total_episodes": sum(partition.episode_count for partition in partition_manifest.partitions),
        "total_frames": sum(partition.frame_count for partition in partition_manifest.partitions),
        "total_tasks": len(partition_manifest.tasks),
        "data_chunks_written": next_data_chunk_index,
    }


def merge_local_lerobot_datasets(
    manifest: DatasetMixJobManifest,
) -> dict[str, Any]:
    started_at = perf_counter()
    output_path = manifest.output_artifact.uri
    if not output_path:
        raise HTTPException(status_code=500, detail="Dataset mix output artifact is missing a writable URI")
    output_root = Path(output_path).resolve(strict=False)
    _prepare_empty_directory(output_root)

    treatment_sources_by_id = {
        source.source_id: source.model_dump(mode="json")
        for source in manifest.treatment_manifest.sources
    }
    load_started_at = perf_counter()
    local_datasets = [
        _load_local_source_dataset(
            source,
            treatment_sources_by_id.get(source.source_id),
        )
        for source in manifest.sources
    ]
    if not local_datasets:
        raise HTTPException(status_code=400, detail="Native local LeRobot merge requires at least one local source")
    load_duration_sec = perf_counter() - load_started_at

    validation_started_at = perf_counter()
    compatibility = _validate_source_compatibility(local_datasets, manifest)
    declared_video_keys = _infer_video_keys(compatibility.features)
    video_policy = _resolve_video_policy(local_datasets, declared_video_keys)
    output_features = _strip_video_feature_specs(
        compatibility.features,
        video_policy.stripped_video_keys,
    )
    merged_stats = _strip_video_stats_entries(
        _merge_stats_dicts(local_datasets),
        video_policy.stripped_video_keys,
    )
    base_info = _apply_video_policy_to_info_contract(
        _canonicalize_info_contract(local_datasets[0].info),
        features=output_features,
        video_policy=video_policy,
    )
    output_episode_schema = _strip_video_fields_from_schema(
        compatibility.episode_schema,
        video_policy.stripped_video_keys,
    )
    validation_duration_sec = perf_counter() - validation_started_at
    partition_plan = manifest.partition_plan
    if partition_plan is None:
        raise HTTPException(
            status_code=500,
            detail="Native local LeRobot merge requires a compiled partition plan",
        )

    staging_root = output_root.parent / DATASET_MIX_LEROBOT_STAGING_DIRNAME
    _prepare_empty_directory(staging_root)
    task_name_to_index: dict[str, int] = {}
    partition_refs: list[DatasetMixPartitionRef] = []
    current_episode_rows: list[dict[str, Any]] = []
    current_frame_rows: list[dict[str, Any]] = []
    current_source_ids: set[str] = set()
    current_task_indices: set[int] = set()
    video_keys = video_policy.preserved_video_keys
    video_file_map: dict[tuple[str, str, int, int], int] = {}
    next_video_chunk_index_by_key: dict[str, int] = {}
    merged_task_index_values: list[int] = []
    merged_episode_index_values: list[int] = []
    next_dataset_index = 0
    next_episode_index = 0
    next_partition_index = 0

    partition_started_at = perf_counter()

    def flush_partition() -> None:
        nonlocal current_episode_rows
        nonlocal current_frame_rows
        nonlocal current_source_ids
        nonlocal current_task_indices
        nonlocal next_partition_index
        if not current_episode_rows:
            return
        partition_id = f"partition-{format_dataset_mix_lerobot_index(next_partition_index)}"
        partition_refs.append(
            _write_partition_output(
                staging_root=staging_root,
                partition_id=partition_id,
                episode_rows=current_episode_rows,
                episode_schema=output_episode_schema,
                frame_rows=current_frame_rows,
                frame_schema=compatibility.frame_schema,
                source_ids=current_source_ids,
                task_indices=current_task_indices,
            )
        )
        current_episode_rows = []
        current_frame_rows = []
        current_source_ids = set()
        current_task_indices = set()
        next_partition_index += 1

    for dataset in local_datasets:
        for episode_span in dataset.episode_spans:
            source_episode_index = episode_span.episode_index
            source_row_count = episode_span.row_to_index - episode_span.row_from_index
            if current_episode_rows and (
                len(current_episode_rows) >= partition_plan.target_episodes_per_partition
                or len(current_frame_rows) + source_row_count
                > partition_plan.target_frames_per_partition
            ):
                flush_partition()
            source_episode_row = dataset.episode_rows_by_index.get(source_episode_index)
            if source_episode_row is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset episode metadata row is missing for episode "
                        f"{source_episode_index}: {dataset.root}"
                    ),
                )
            dataset_from_index = next_dataset_index
            episode_task_names = _resolve_episode_task_names(dataset, source_episode_row)
            seen_task_names = set(episode_task_names)
            episode_task_index_values: list[int] = []
            current_source_ids.add(dataset.source.source_id)
            for task_name in episode_task_names:
                target_task_index = task_name_to_index.setdefault(
                    task_name,
                    len(task_name_to_index),
                )
                current_task_indices.add(target_task_index)
            for frame_offset, row_index in enumerate(
                range(episode_span.row_from_index, episode_span.row_to_index)
            ):
                source_row = dataset.frame_rows[row_index]
                source_task_index = _coerce_int(source_row.get("task_index", 0), field_name="task_index")
                task_name = _task_name_for_index(dataset, source_task_index)
                target_task_index = task_name_to_index.setdefault(task_name, len(task_name_to_index))
                if task_name not in seen_task_names:
                    seen_task_names.add(task_name)
                    episode_task_names.append(task_name)
                    current_task_indices.add(target_task_index)
                episode_task_index_values.append(target_task_index)
                merged_task_index_values.append(target_task_index)
                merged_episode_index_values.append(next_episode_index)
                merged_row = dict(source_row)
                merged_row["index"] = next_dataset_index
                merged_row["episode_index"] = next_episode_index
                merged_row["frame_index"] = frame_offset
                merged_row["task_index"] = target_task_index
                current_frame_rows.append(merged_row)
                next_dataset_index += 1

            merged_episode_row = _strip_episode_video_fields(
                dict(source_episode_row),
                video_policy.stripped_video_keys,
            )
            merged_episode_row["episode_index"] = next_episode_index
            merged_episode_row["length"] = source_row_count
            merged_episode_row["tasks"] = episode_task_names
            merged_episode_row["dataset_from_index"] = dataset_from_index
            merged_episode_row["dataset_to_index"] = next_dataset_index
            _set_episode_stats(
                merged_episode_row,
                "index",
                _sequence_stats(list(range(dataset_from_index, next_dataset_index))),
            )
            _set_episode_stats(
                merged_episode_row,
                "episode_index",
                _sequence_stats([next_episode_index] * source_row_count),
            )
            _set_episode_stats(
                merged_episode_row,
                "task_index",
                _sequence_stats(episode_task_index_values),
            )
            _patch_episode_video_refs(
                dataset=dataset,
                output_root=output_root,
                row=merged_episode_row,
                video_keys=video_keys,
                video_file_map=video_file_map,
                next_video_chunk_index_by_key=next_video_chunk_index_by_key,
            )
            current_episode_rows.append(merged_episode_row)
            next_episode_index += 1

    flush_partition()
    partition_duration_sec = perf_counter() - partition_started_at
    _replace_stats_entry(
        merged_stats,
        key="index",
        values=list(range(next_dataset_index)),
    )
    _replace_stats_entry(
        merged_stats,
        key="episode_index",
        values=merged_episode_index_values,
    )
    _replace_stats_entry(
        merged_stats,
        key="task_index",
        values=merged_task_index_values,
    )
    partition_manifest = _build_partition_manifest(
        partition_plan=partition_plan,
        compatibility=compatibility,
        features=output_features,
        task_name_to_index=task_name_to_index,
        partitions=partition_refs,
    )
    _write_json_file(
        staging_root / DATASET_MIX_LEROBOT_PARTITION_MANIFEST_FILENAME,
        partition_manifest.model_dump(mode="json"),
    )
    finalize_started_at = perf_counter()
    finalized = _finalize_partitioned_output(
        output_root=output_root,
        base_info=base_info,
        merged_stats=merged_stats,
        episode_schema=output_episode_schema,
        partition_manifest=partition_manifest,
    )
    finalize_duration_sec = perf_counter() - finalize_started_at
    total_duration_sec = perf_counter() - started_at

    return {
        "success": True,
        "message": "Datasets mixed successfully",
        "output_path": str(output_root),
        "info": {
            "total_episodes": finalized["total_episodes"],
            "total_frames": finalized["total_frames"],
            "total_tasks": finalized["total_tasks"],
            "datasets_loaded": len(local_datasets),
            "dataset_sources": [dataset.source.canonical_source for dataset in local_datasets],
            "data_chunks_written": finalized["data_chunks_written"],
            "partition_count": len(partition_refs),
            "partition_manifest_path": _artifact_path_string(
                staging_root / DATASET_MIX_LEROBOT_PARTITION_MANIFEST_FILENAME,
            ),
            "video_mode": video_policy.mode,
        },
        "debug": {
            "timings_sec": {
                "load_sources": round(load_duration_sec, 6),
                "validate": round(validation_duration_sec, 6),
                "build_partitions": round(partition_duration_sec, 6),
                "finalize": round(finalize_duration_sec, 6),
                "total": round(total_duration_sec, 6),
            },
            "source_count": len(local_datasets),
            "source_episode_count": sum(len(dataset.episode_rows_by_index) for dataset in local_datasets),
            "source_frame_count": sum(len(dataset.frame_rows) for dataset in local_datasets),
            "partition_count": len(partition_refs),
            "data_chunks_written": finalized["data_chunks_written"],
            "video_mode": video_policy.mode,
            "preserved_video_keys": video_policy.preserved_video_keys,
            "stripped_video_keys": video_policy.stripped_video_keys,
            "referenced_video_file_count": video_policy.referenced_video_file_count,
            "copied_video_file_count": len(video_file_map),
        },
        "total_episodes": finalized["total_episodes"],
        "total_frames": finalized["total_frames"],
        "total_tasks": finalized["total_tasks"],
    }


def can_execute_native_local_lerobot_sources(sources: list[DatasetMixSourceRef]) -> bool:
    if not sources:
        return False
    for source in sources:
        if source.source_kind != "local":
            return False
        source_path = Path(source.canonical_source).resolve(strict=False)
        if not source_path.exists() or not source_path.is_dir():
            return False
        if (source_path / DATASET_MIX_META_DIRNAME / DATASET_MIX_LEROBOT_INFO_FILENAME).exists():
            continue
        return False
    return True


def can_execute_native_local_lerobot(manifest: DatasetMixJobManifest) -> bool:
    return can_execute_native_local_lerobot_sources(manifest.sources)

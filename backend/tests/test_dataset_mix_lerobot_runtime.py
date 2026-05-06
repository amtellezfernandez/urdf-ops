from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.models.dataset_alignment import DatasetRepresentationValidationResponse
from backend.models.datasets import (
    DatasetMixPartitionManifest,
    DatasetMixAlignmentDataset,
    DatasetMixAlignmentRequest,
    DatasetMixRequest,
    DatasetTreatmentAnalysisResponse,
    DatasetTreatmentManifest,
    DatasetTreatmentSourceManifest,
    DatasetTreatmentStats,
)
from backend.services.dataset_mix_control_plane import create_local_dataset_mix_control_plane
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_VIDEO_MODE_NONE,
    DATASET_MIX_LEROBOT_VIDEO_MODE_PRESERVE,
    DATASET_MIX_LEROBOT_VIDEO_MODE_STRIP_MISSING_LOCAL,
    DATASET_MIX_LEROBOT_VIDEO_PATH_TEMPLATE,
)

TEST_REPRESENTATION_ID = "semantic/joint-position/v1"
TEST_EMBODIMENT_ID = "demo:robot"
TEST_NAMING_STATUS = "named"
TEST_ROBOT_TYPE = "demo-bot"
TEST_FPS = 20
TEST_FLOAT_PRECISION_DIGITS = 6
TEST_PARQUET_CHUNK_INDEX = 0
TEST_PARQUET_FILE_INDEX = 0
TEST_FILES_SIZE_MB = 0
TEST_WEB_FEATURE_GROUP = "motors"
TEST_VIDEO_KEY = "observation.images.rgb_static"
TEST_FRONTEND_FIXTURE_DATASET_NAME_A = "frontend_export_a"
TEST_FRONTEND_FIXTURE_DATASET_NAME_B = "frontend_export_b"
TEST_FRONTEND_FIXTURE_TEST_PATH = (
    "web/src/features/layout/sidebar/datasetArchiveFixtureWriter.test.ts"
)
TEST_REAL_LEROBOT_ROOT_ENV = "URDF_STUDIO_REAL_LEROBOT_ROOT"
TEST_REAL_LEROBOT_DEFAULT_ROOT = "/tmp/real-lerobot-tests"
TEST_REAL_LEROBOT_CASES = (
    ("pusht", "lerobot-pusht", "lerobot-pusht-copy"),
    ("xarm_lift_medium", "lerobot-xarm_lift_medium", "lerobot-xarm_lift_medium-copy"),
    ("taco_play", "lerobot-taco_play", "lerobot-taco_play-copy"),
)
TEST_REAL_LEROBOT_INFO_PATH = Path("meta/info.json")
TEST_REAL_LEROBOT_EPISODES_GLOB = "meta/episodes/**/*.parquet"
TEST_REAL_LEROBOT_VIDEO_DIRNAME = "videos"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _round_vectors(vectors: list[list[float]]) -> list[list[float]]:
    return [
        [round(value, TEST_FLOAT_PRECISION_DIGITS) for value in vector]
        for vector in vectors
    ]


def _build_stats(values: list[float]) -> dict[str, list[float]]:
    if not values:
        return {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": [min(values)],
        "max": [max(values)],
        "mean": [mean],
        "std": [variance**0.5],
        "count": [len(values)],
    }


def _build_vector_stats(values: list[list[float]]) -> dict[str, list[float]]:
    dimensions = list(zip(*values, strict=True))
    stats_by_dimension = [_build_stats([float(value) for value in dimension]) for dimension in dimensions]
    return {
        "min": [stats["min"][0] for stats in stats_by_dimension],
        "max": [stats["max"][0] for stats in stats_by_dimension],
        "mean": [stats["mean"][0] for stats in stats_by_dimension],
        "std": [stats["std"][0] for stats in stats_by_dimension],
        "count": [stats["count"][0] for stats in stats_by_dimension],
    }


def _build_dataset(
    root: Path,
    *,
    task_name: str,
    frame_values: list[list[float]],
    timestamp_start: float,
    tasks_field_name: str = "task",
) -> None:
    timestamps = [timestamp_start + index * 0.1 for index in range(len(frame_values))]
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "dataset_format_version": "lerobot_dataset_v3",
            "robot_type": TEST_ROBOT_TYPE,
            "embodiment_ref": {"embodiment_id": TEST_EMBODIMENT_ID},
            "representation_id": TEST_REPRESENTATION_ID,
            "naming_status": TEST_NAMING_STATUS,
            "fps": TEST_FPS,
            "features": {
                "action": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_episodes": 1,
            "total_frames": len(frame_values),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": "0:1"},
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "frame_index": _build_stats([float(index) for index in range(len(frame_values))]),
            "episode_index": _build_stats([0.0 for _ in frame_values]),
            "index": _build_stats([float(index) for index in range(len(frame_values))]),
            "task_index": _build_stats([0.0 for _ in frame_values]),
            "timestamp": _build_stats(timestamps),
            "action": _build_vector_stats(frame_values),
            "observation.state": _build_vector_stats(frame_values),
        },
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{tasks_field_name: task_name, "task_index": 0}],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "data/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "data/file_index": TEST_PARQUET_FILE_INDEX,
                "tasks": [task_name],
                "length": len(frame_values),
                "dataset_from_index": 0,
                "dataset_to_index": len(frame_values),
                "stats/action/min": _build_vector_stats(frame_values)["min"],
                "stats/action/max": _build_vector_stats(frame_values)["max"],
                "stats/action/mean": _build_vector_stats(frame_values)["mean"],
                "stats/action/std": _build_vector_stats(frame_values)["std"],
                "stats/action/count": _build_vector_stats(frame_values)["count"],
                "stats/observation.state/min": _build_vector_stats(frame_values)["min"],
                "stats/observation.state/max": _build_vector_stats(frame_values)["max"],
                "stats/observation.state/mean": _build_vector_stats(frame_values)["mean"],
                "stats/observation.state/std": _build_vector_stats(frame_values)["std"],
                "stats/observation.state/count": _build_vector_stats(frame_values)["count"],
                "stats/timestamp/min": _build_stats(timestamps)["min"],
                "stats/timestamp/max": _build_stats(timestamps)["max"],
                "stats/timestamp/mean": _build_stats(timestamps)["mean"],
                "stats/timestamp/std": _build_stats(timestamps)["std"],
                "stats/timestamp/count": _build_stats(timestamps)["count"],
                "stats/episode_index/min": [0],
                "stats/episode_index/max": [0],
                "stats/episode_index/mean": [0.0],
                "stats/episode_index/std": [0.0],
                "stats/episode_index/count": [len(frame_values)],
                "stats/frame_index/min": [0],
                "stats/frame_index/max": [len(frame_values) - 1],
                "stats/frame_index/mean": [(len(frame_values) - 1) / 2],
                "stats/frame_index/std": [_build_stats([float(index) for index in range(len(frame_values))])["std"][0]],
                "stats/frame_index/count": [len(frame_values)],
                "stats/index/min": [0],
                "stats/index/max": [len(frame_values) - 1],
                "stats/index/mean": [(len(frame_values) - 1) / 2],
                "stats/index/std": [_build_stats([float(index) for index in range(len(frame_values))])["std"][0]],
                "stats/index/count": [len(frame_values)],
                "stats/task_index/min": [0],
                "stats/task_index/max": [0],
                "stats/task_index/mean": [0.0],
                "stats/task_index/std": [0.0],
                "stats/task_index/count": [len(frame_values)],
                "meta/episodes/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "meta/episodes/file_index": TEST_PARQUET_FILE_INDEX,
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "index": index,
                "episode_index": 0,
                "frame_index": index,
                "timestamp": timestamp_start + index * 0.1,
                "action": values,
                "observation.state": values,
                "task_index": 0,
                "robot_type": TEST_ROBOT_TYPE,
                "embodiment_id": TEST_EMBODIMENT_ID,
                "representation_id": TEST_REPRESENTATION_ID,
                "naming_status": TEST_NAMING_STATUS,
            }
            for index, values in enumerate(frame_values)
        ],
    )


def _build_multi_episode_dataset(
    root: Path,
    *,
    episodes: list[dict[str, object]],
) -> None:
    episode_rows: list[dict[str, object]] = []
    all_frame_rows: list[dict[str, object]] = []
    task_names: list[str] = []
    next_dataset_index = 0
    for episode_index, episode_spec in enumerate(episodes):
        task_name = str(episode_spec["task_name"])
        frame_values = list(episode_spec["frame_values"])
        timestamp_start = float(episode_spec["timestamp_start"])
        timestamps = [timestamp_start + index * 0.1 for index in range(len(frame_values))]
        vector_stats = _build_vector_stats(frame_values)
        timestamp_stats = _build_stats(timestamps)
        frame_index_stats = _build_stats([float(index) for index in range(len(frame_values))])
        if task_name not in task_names:
            task_names.append(task_name)
        task_index = task_names.index(task_name)
        dataset_from_index = next_dataset_index
        dataset_to_index = dataset_from_index + len(frame_values)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": episode_index,
                "data/file_index": TEST_PARQUET_FILE_INDEX,
                "tasks": [task_name],
                "length": len(frame_values),
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_to_index,
                "stats/action/min": vector_stats["min"],
                "stats/action/max": vector_stats["max"],
                "stats/action/mean": vector_stats["mean"],
                "stats/action/std": vector_stats["std"],
                "stats/action/count": vector_stats["count"],
                "stats/observation.state/min": vector_stats["min"],
                "stats/observation.state/max": vector_stats["max"],
                "stats/observation.state/mean": vector_stats["mean"],
                "stats/observation.state/std": vector_stats["std"],
                "stats/observation.state/count": vector_stats["count"],
                "stats/timestamp/min": timestamp_stats["min"],
                "stats/timestamp/max": timestamp_stats["max"],
                "stats/timestamp/mean": timestamp_stats["mean"],
                "stats/timestamp/std": timestamp_stats["std"],
                "stats/timestamp/count": timestamp_stats["count"],
                "stats/episode_index/min": [episode_index],
                "stats/episode_index/max": [episode_index],
                "stats/episode_index/mean": [float(episode_index)],
                "stats/episode_index/std": [0.0],
                "stats/episode_index/count": [len(frame_values)],
                "stats/frame_index/min": [0],
                "stats/frame_index/max": [len(frame_values) - 1],
                "stats/frame_index/mean": frame_index_stats["mean"],
                "stats/frame_index/std": frame_index_stats["std"],
                "stats/frame_index/count": [len(frame_values)],
                "stats/index/min": [dataset_from_index],
                "stats/index/max": [dataset_to_index - 1],
                "stats/index/mean": _build_stats(
                    [float(index) for index in range(dataset_from_index, dataset_to_index)]
                )["mean"],
                "stats/index/std": _build_stats(
                    [float(index) for index in range(dataset_from_index, dataset_to_index)]
                )["std"],
                "stats/index/count": [len(frame_values)],
                "stats/task_index/min": [task_index],
                "stats/task_index/max": [task_index],
                "stats/task_index/mean": [float(task_index)],
                "stats/task_index/std": [0.0],
                "stats/task_index/count": [len(frame_values)],
                "meta/episodes/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "meta/episodes/file_index": TEST_PARQUET_FILE_INDEX,
            }
        )
        chunk_rows: list[dict[str, object]] = []
        for frame_index, values in enumerate(frame_values):
            row = {
                "index": next_dataset_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": timestamp_start + frame_index * 0.1,
                "action": values,
                "observation.state": values,
                "task_index": task_index,
                "robot_type": TEST_ROBOT_TYPE,
                "embodiment_id": TEST_EMBODIMENT_ID,
                "representation_id": TEST_REPRESENTATION_ID,
                "naming_status": TEST_NAMING_STATUS,
            }
            chunk_rows.append(row)
            all_frame_rows.append(row)
            next_dataset_index += 1
        _write_parquet(
            root / "data" / f"chunk-{episode_index:03d}" / "file-000.parquet",
            chunk_rows,
        )

    all_timestamps = [float(row["timestamp"]) for row in all_frame_rows]
    all_vectors = [list(row["action"]) for row in all_frame_rows]
    all_task_indexes = [float(row["task_index"]) for row in all_frame_rows]
    all_episode_indexes = [float(row["episode_index"]) for row in all_frame_rows]
    all_indexes = [float(row["index"]) for row in all_frame_rows]
    all_frame_indexes = [float(row["frame_index"]) for row in all_frame_rows]
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "dataset_format_version": "lerobot_dataset_v3",
            "robot_type": TEST_ROBOT_TYPE,
            "embodiment_ref": {"embodiment_id": TEST_EMBODIMENT_ID},
            "representation_id": TEST_REPRESENTATION_ID,
            "naming_status": TEST_NAMING_STATUS,
            "fps": TEST_FPS,
            "features": {
                "action": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_episodes": len(episodes),
            "total_frames": len(all_frame_rows),
            "total_tasks": len(task_names),
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": f"0:{len(episodes)}"},
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "frame_index": _build_stats(all_frame_indexes),
            "episode_index": _build_stats(all_episode_indexes),
            "index": _build_stats(all_indexes),
            "task_index": _build_stats(all_task_indexes),
            "timestamp": _build_stats(all_timestamps),
            "action": _build_vector_stats(all_vectors),
            "observation.state": _build_vector_stats(all_vectors),
        },
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task": task_name, "task_index": index} for index, task_name in enumerate(task_names)],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        episode_rows,
    )


def _build_web_export_dataset(
    root: Path,
    *,
    task_names: list[str],
    frame_values: list[list[float]],
    timestamp_start: float,
) -> None:
    timestamps = [timestamp_start + index * 0.1 for index in range(len(frame_values))]
    vector_stats = _build_vector_stats(frame_values)
    timestamp_stats = _build_stats(timestamps)
    frame_index_stats = _build_stats([float(index) for index in range(len(frame_values))])
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": TEST_ROBOT_TYPE,
            "fps": TEST_FPS,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {TEST_WEB_FEATURE_GROUP: ["joint_0", "joint_1"]},
                    "fps": TEST_FPS,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": {TEST_WEB_FEATURE_GROUP: ["joint_0", "joint_1"]},
                    "fps": TEST_FPS,
                },
                "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": TEST_FPS},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": TEST_FPS},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": TEST_FPS},
                "index": {"dtype": "int64", "shape": [1], "names": None, "fps": TEST_FPS},
                "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": TEST_FPS},
            },
            "total_episodes": 1,
            "total_frames": len(frame_values),
            "total_tasks": len(task_names),
            "chunks_size": 1000,
            "files_size_in_mb": TEST_FILES_SIZE_MB,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "",
            "splits": {"train": "0:1"},
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "frame_index": frame_index_stats,
            "episode_index": _build_stats([0.0 for _ in frame_values]),
            "index": frame_index_stats,
            "task_index": _build_stats([0.0 for _ in frame_values]),
            "timestamp": timestamp_stats,
            "action": vector_stats,
            "observation.state": vector_stats,
        },
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task": task_name, "task_index": index} for index, task_name in enumerate(task_names)],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "data/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "data/file_index": TEST_PARQUET_FILE_INDEX,
                "dataset_from_index": 0,
                "dataset_to_index": len(frame_values),
                "tasks": task_names,
                "length": len(frame_values),
                "stats/observation.state/min": vector_stats["min"],
                "stats/observation.state/max": vector_stats["max"],
                "stats/observation.state/mean": vector_stats["mean"],
                "stats/observation.state/std": vector_stats["std"],
                "stats/observation.state/count": vector_stats["count"],
                "stats/action/min": vector_stats["min"],
                "stats/action/max": vector_stats["max"],
                "stats/action/mean": vector_stats["mean"],
                "stats/action/std": vector_stats["std"],
                "stats/action/count": vector_stats["count"],
                "stats/episode_index/min": [0],
                "stats/episode_index/max": [0],
                "stats/episode_index/mean": [0.0],
                "stats/episode_index/std": [0.0],
                "stats/episode_index/count": [len(frame_values)],
                "stats/frame_index/min": [0],
                "stats/frame_index/max": [len(frame_values) - 1],
                "stats/frame_index/mean": frame_index_stats["mean"],
                "stats/frame_index/std": frame_index_stats["std"],
                "stats/frame_index/count": [len(frame_values)],
                "stats/timestamp/min": timestamp_stats["min"],
                "stats/timestamp/max": timestamp_stats["max"],
                "stats/timestamp/mean": timestamp_stats["mean"],
                "stats/timestamp/std": timestamp_stats["std"],
                "stats/timestamp/count": [len(frame_values)],
                "stats/index/min": [0],
                "stats/index/max": [len(frame_values) - 1],
                "stats/index/mean": frame_index_stats["mean"],
                "stats/index/std": frame_index_stats["std"],
                "stats/index/count": [len(frame_values)],
                "stats/task_index/min": [0],
                "stats/task_index/max": [0],
                "stats/task_index/mean": [0.0],
                "stats/task_index/std": [0.0],
                "stats/task_index/count": [len(frame_values)],
                "meta/episodes/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "meta/episodes/file_index": TEST_PARQUET_FILE_INDEX,
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "observation.state": values,
                "action": values,
                "episode_index": 0,
                "frame_index": index,
                "timestamp": timestamp_start + index * 0.1,
                "index": index,
                "task_index": 0,
            }
            for index, values in enumerate(frame_values)
        ],
    )


def _build_video_metadata_only_dataset(
    root: Path,
    *,
    task_name: str,
    frame_values: list[list[float]],
    timestamp_start: float,
    create_video_files: bool,
) -> None:
    timestamps = [timestamp_start + index * 0.1 for index in range(len(frame_values))]
    vector_stats = _build_vector_stats(frame_values)
    timestamp_stats = _build_stats(timestamps)
    frame_index_stats = _build_stats([float(index) for index in range(len(frame_values))])
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "dataset_format_version": "lerobot_dataset_v3",
            "robot_type": TEST_ROBOT_TYPE,
            "embodiment_ref": {"embodiment_id": TEST_EMBODIMENT_ID},
            "representation_id": TEST_REPRESENTATION_ID,
            "naming_status": TEST_NAMING_STATUS,
            "fps": TEST_FPS,
            "features": {
                "action": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": ["joint_0.pos", "joint_1.pos"],
                    "shape": [2],
                },
                TEST_VIDEO_KEY: {
                    "dtype": "video",
                    "shape": [64, 64, 3],
                    "names": None,
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_episodes": 1,
            "total_frames": len(frame_values),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": "0:1"},
        },
    )
    _write_json(
        root / "meta" / "stats.json",
        {
            "frame_index": frame_index_stats,
            "episode_index": _build_stats([0.0 for _ in frame_values]),
            "index": frame_index_stats,
            "task_index": _build_stats([0.0 for _ in frame_values]),
            "timestamp": timestamp_stats,
            "action": vector_stats,
            "observation.state": vector_stats,
            TEST_VIDEO_KEY: {
                "min": [],
                "max": [],
                "mean": [],
                "std": [],
                "count": [],
            },
        },
    )
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task": task_name, "task_index": 0}],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "data/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "data/file_index": TEST_PARQUET_FILE_INDEX,
                "dataset_from_index": 0,
                "dataset_to_index": len(frame_values),
                f"videos/{TEST_VIDEO_KEY}/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                f"videos/{TEST_VIDEO_KEY}/file_index": TEST_PARQUET_FILE_INDEX,
                f"videos/{TEST_VIDEO_KEY}/from_timestamp": timestamps[0],
                f"videos/{TEST_VIDEO_KEY}/to_timestamp": timestamps[-1],
                "tasks": [task_name],
                "length": len(frame_values),
                "stats/observation.state/min": vector_stats["min"],
                "stats/observation.state/max": vector_stats["max"],
                "stats/observation.state/mean": vector_stats["mean"],
                "stats/observation.state/std": vector_stats["std"],
                "stats/observation.state/count": vector_stats["count"],
                "stats/action/min": vector_stats["min"],
                "stats/action/max": vector_stats["max"],
                "stats/action/mean": vector_stats["mean"],
                "stats/action/std": vector_stats["std"],
                "stats/action/count": vector_stats["count"],
                f"stats/{TEST_VIDEO_KEY}/min": [],
                f"stats/{TEST_VIDEO_KEY}/max": [],
                f"stats/{TEST_VIDEO_KEY}/mean": [],
                f"stats/{TEST_VIDEO_KEY}/std": [],
                f"stats/{TEST_VIDEO_KEY}/count": [],
                "stats/episode_index/min": [0],
                "stats/episode_index/max": [0],
                "stats/episode_index/mean": [0.0],
                "stats/episode_index/std": [0.0],
                "stats/episode_index/count": [len(frame_values)],
                "stats/frame_index/min": [0],
                "stats/frame_index/max": [len(frame_values) - 1],
                "stats/frame_index/mean": frame_index_stats["mean"],
                "stats/frame_index/std": frame_index_stats["std"],
                "stats/frame_index/count": [len(frame_values)],
                "stats/timestamp/min": timestamp_stats["min"],
                "stats/timestamp/max": timestamp_stats["max"],
                "stats/timestamp/mean": timestamp_stats["mean"],
                "stats/timestamp/std": timestamp_stats["std"],
                "stats/timestamp/count": [len(frame_values)],
                "stats/index/min": [0],
                "stats/index/max": [len(frame_values) - 1],
                "stats/index/mean": frame_index_stats["mean"],
                "stats/index/std": frame_index_stats["std"],
                "stats/index/count": [len(frame_values)],
                "stats/task_index/min": [0],
                "stats/task_index/max": [0],
                "stats/task_index/mean": [0.0],
                "stats/task_index/std": [0.0],
                "stats/task_index/count": [len(frame_values)],
                "meta/episodes/chunk_index": TEST_PARQUET_CHUNK_INDEX,
                "meta/episodes/file_index": TEST_PARQUET_FILE_INDEX,
            }
        ],
    )
    _write_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "observation.state": values,
                "action": values,
                "episode_index": 0,
                "frame_index": index,
                "timestamp": timestamp_start + index * 0.1,
                "index": index,
                "task_index": 0,
            }
            for index, values in enumerate(frame_values)
        ],
    )
    if create_video_files:
        video_path = (
            root
            / "videos"
            / TEST_VIDEO_KEY
            / "chunk-000"
            / "file-000.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video")


def _build_request(local_paths: list[str]) -> DatasetMixRequest:
    return DatasetMixRequest(
        local_paths=local_paths,
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                DatasetMixAlignmentDataset(
                    dataset_id=f"local:dataset-{index}",
                    embodiment_id=TEST_EMBODIMENT_ID,
                    representation_id=TEST_REPRESENTATION_ID,
                    naming_status=TEST_NAMING_STATUS,
                )
                for index, _ in enumerate(local_paths)
            ],
            required_representation_id=TEST_REPRESENTATION_ID,
        ),
    )


def _build_treatment_analysis(local_paths: list[Path]) -> DatasetTreatmentAnalysisResponse:
    return DatasetTreatmentAnalysisResponse(
        success=True,
        warnings=[],
        alignment=DatasetRepresentationValidationResponse(valid=True, errors=[], warnings=[]),
        treatment_manifest=DatasetTreatmentManifest(
            manifest_version="v1",
            required_representation_id=TEST_REPRESENTATION_ID,
            sources=[
                DatasetTreatmentSourceManifest(
                    source_id=f"local:{index}",
                    dataset_id=f"local:dataset-{index}",
                    source_kind="local",
                    source_value=str(path),
                    canonical_source=str(path.resolve()),
                    embodiment_id=TEST_EMBODIMENT_ID,
                    representation_id=TEST_REPRESENTATION_ID,
                    naming_status=TEST_NAMING_STATUS,
                    profile_id="semantic-aligned",
                    profile_version="v1",
                )
                for index, path in enumerate(local_paths)
            ],
            stats=DatasetTreatmentStats(
                total_sources=len(local_paths),
                local_source_count=len(local_paths),
                unique_canonical_sources=len(local_paths),
            ),
        ),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _real_lerobot_root() -> Path:
    return Path(
        os.environ.get(
            TEST_REAL_LEROBOT_ROOT_ENV,
            TEST_REAL_LEROBOT_DEFAULT_ROOT,
        )
    )


def _real_lerobot_case_paths(
    primary_name: str,
    secondary_name: str,
) -> tuple[Path, Path] | None:
    root = _real_lerobot_root()
    primary = root / primary_name
    secondary = root / secondary_name
    required_paths = (
        primary / TEST_REAL_LEROBOT_INFO_PATH,
        secondary / TEST_REAL_LEROBOT_INFO_PATH,
    )
    if not all(path.exists() for path in required_paths):
        return None
    return primary, secondary


def _real_lerobot_info(root: Path) -> dict[str, object]:
    return json.loads((root / TEST_REAL_LEROBOT_INFO_PATH).read_text(encoding="utf-8"))


def _real_lerobot_video_keys(info: dict[str, object]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    return sorted(
        key
        for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    )


def _real_lerobot_expected_video_mode(roots: tuple[Path, Path]) -> str:
    first_info = _real_lerobot_info(roots[0])
    video_keys = _real_lerobot_video_keys(first_info)
    if not video_keys:
        return DATASET_MIX_LEROBOT_VIDEO_MODE_NONE
    for root in roots:
        episode_rows: list[dict[str, object]] = []
        for parquet_path in sorted(root.glob(TEST_REAL_LEROBOT_EPISODES_GLOB)):
            episode_rows.extend(pq.read_table(parquet_path).to_pylist())
        for video_key in video_keys:
            chunk_field = f"videos/{video_key}/chunk_index"
            file_field = f"videos/{video_key}/file_index"
            for row in episode_rows:
                chunk_index = row.get(chunk_field)
                file_index = row.get(file_field)
                if chunk_index is None or file_index is None:
                    continue
                video_path = (
                    root
                    / TEST_REAL_LEROBOT_VIDEO_DIRNAME
                    / video_key
                    / f"chunk-{int(chunk_index):03d}"
                    / f"file-{int(file_index):03d}.mp4"
                )
                if not video_path.exists():
                    return DATASET_MIX_LEROBOT_VIDEO_MODE_STRIP_MISSING_LOCAL
    return DATASET_MIX_LEROBOT_VIDEO_MODE_PRESERVE


def _generate_frontend_export_fixtures(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "npm",
            "test",
            "--",
            TEST_FRONTEND_FIXTURE_TEST_PATH,
        ],
        cwd=_repo_root(),
        env={
            **dict(os.environ),
            "URDF_STUDIO_FRONTEND_EXPORT_FIXTURE_ROOT": str(output_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Failed to generate frontend export fixtures:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_control_plane_merges_real_local_lerobot_datasets_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2], [0.3, 0.4]],
        timestamp_start=0.0,
    )
    _build_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2]],
        timestamp_start=1.0,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )

    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.status == "queued"
    assert queued.manifest_artifact is not None
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "native-local-lerobot"
    assert queued.partition_plan is not None
    assert queued.partition_plan.strategy == "episode-window"

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 3
    assert info["total_tasks"] == 2
    assert info["representation_id"] == TEST_REPRESENTATION_ID
    assert "dataset_treatment_manifest" not in info
    assert "dataset_treatment_sources" not in info
    assert not (output_root / "_staging").exists()

    partition_manifest = DatasetMixPartitionManifest.model_validate_json(
        (
            output_root.parent / "_staging" / "partition-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert partition_manifest.partition_plan.strategy == "episode-window"
    assert len(partition_manifest.partitions) == 1
    assert partition_manifest.partitions[0].frame_count == 3

    tasks_rows = pq.read_table(output_root / "meta" / "tasks.parquet").to_pylist()
    assert tasks_rows == [
        {"task": "pick", "task_index": 0},
        {"task": "place", "task_index": 1},
    ]

    episode_rows = pq.read_table(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert episode_rows == [
        {
            "episode_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "tasks": ["pick"],
            "length": 2,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "stats/action/min": [0.1, 0.2],
            "stats/action/max": [0.3, 0.4],
            "stats/action/mean": [0.2, 0.30000000000000004],
            "stats/action/std": [0.09999999999999999, 0.1],
            "stats/action/count": [2, 2],
            "stats/observation.state/min": [0.1, 0.2],
            "stats/observation.state/max": [0.3, 0.4],
            "stats/observation.state/mean": [0.2, 0.30000000000000004],
            "stats/observation.state/std": [0.09999999999999999, 0.1],
            "stats/observation.state/count": [2, 2],
            "stats/timestamp/min": [0.0],
            "stats/timestamp/max": [0.1],
            "stats/timestamp/mean": [0.05],
            "stats/timestamp/std": [0.05],
            "stats/timestamp/count": [2],
            "stats/episode_index/min": [0],
            "stats/episode_index/max": [0],
            "stats/episode_index/mean": [0.0],
            "stats/episode_index/std": [0.0],
            "stats/episode_index/count": [2],
            "stats/frame_index/min": [0],
            "stats/frame_index/max": [1],
            "stats/frame_index/mean": [0.5],
            "stats/frame_index/std": [0.5],
            "stats/frame_index/count": [2],
            "stats/index/min": [0],
            "stats/index/max": [1],
            "stats/index/mean": [0.5],
            "stats/index/std": [0.5],
            "stats/index/count": [2],
            "stats/task_index/min": [0],
            "stats/task_index/max": [0],
            "stats/task_index/mean": [0.0],
            "stats/task_index/std": [0.0],
            "stats/task_index/count": [2],
        },
        {
            "episode_index": 1,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "tasks": ["place"],
            "length": 1,
            "dataset_from_index": 2,
            "dataset_to_index": 3,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "stats/action/min": [1.1, 1.2],
            "stats/action/max": [1.1, 1.2],
            "stats/action/mean": [1.1, 1.2],
            "stats/action/std": [0.0, 0.0],
            "stats/action/count": [1, 1],
            "stats/observation.state/min": [1.1, 1.2],
            "stats/observation.state/max": [1.1, 1.2],
            "stats/observation.state/mean": [1.1, 1.2],
            "stats/observation.state/std": [0.0, 0.0],
            "stats/observation.state/count": [1, 1],
            "stats/timestamp/min": [1.0],
            "stats/timestamp/max": [1.0],
            "stats/timestamp/mean": [1.0],
            "stats/timestamp/std": [0.0],
            "stats/timestamp/count": [1],
            "stats/episode_index/min": [1],
            "stats/episode_index/max": [1],
            "stats/episode_index/mean": [1.0],
            "stats/episode_index/std": [0.0],
            "stats/episode_index/count": [1],
            "stats/frame_index/min": [0],
            "stats/frame_index/max": [0],
            "stats/frame_index/mean": [0.0],
            "stats/frame_index/std": [0.0],
            "stats/frame_index/count": [1],
            "stats/index/min": [2],
            "stats/index/max": [2],
            "stats/index/mean": [2.0],
            "stats/index/std": [0.0],
            "stats/index/count": [1],
            "stats/task_index/min": [1],
            "stats/task_index/max": [1],
            "stats/task_index/mean": [1.0],
            "stats/task_index/std": [0.0],
            "stats/task_index/count": [1],
        },
    ]

    data_rows = pq.read_table(output_root / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [row["index"] for row in data_rows] == [0, 1, 2]
    assert [row["episode_index"] for row in data_rows] == [0, 0, 1]
    assert [row["task_index"] for row in data_rows] == [0, 0, 1]
    assert _round_vectors([row["action"] for row in data_rows]) == [
        [0.1, 0.2],
        [0.3, 0.4],
        [1.1, 1.2],
    ]

    result_payload = json.loads(
        (
            output_root.parent / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result_payload["total_episodes"] == 2
    assert result_payload["total_frames"] == 3
    assert result_payload["info"]["partition_count"] == 1


@pytest.mark.parametrize(
    ("dataset_name", "primary_name", "secondary_name"),
    TEST_REAL_LEROBOT_CASES,
)
def test_native_local_merge_matches_real_public_lerobot_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    primary_name: str,
    secondary_name: str,
) -> None:
    resolved_paths = _real_lerobot_case_paths(primary_name, secondary_name)
    if resolved_paths is None:
        pytest.skip(
            "real LeRobot dataset cache is missing; set "
            f"{TEST_REAL_LEROBOT_ROOT_ENV} or populate {_real_lerobot_root()}"
        )
    local_paths = list(resolved_paths)
    source_infos = [_real_lerobot_info(path) for path in local_paths]
    expected_video_mode = _real_lerobot_expected_video_mode(resolved_paths)

    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (_real_lerobot_root().resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )

    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / f"control-plane-{dataset_name}",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "native-local-lerobot"
    assert queued.partition_plan is not None
    assert queued.partition_plan.strategy == "episode-window"

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    output_info = json.loads(
        (output_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    result_payload = json.loads(
        (output_root.parent / "result.json").read_text(encoding="utf-8")
    )
    tasks_rows = pq.read_table(output_root / "meta" / "tasks.parquet").to_pylist()
    data_paths = sorted((output_root / "data").glob("chunk-*/file-000.parquet"))

    assert output_info["total_episodes"] == sum(
        int(info["total_episodes"]) for info in source_infos
    )
    assert output_info["total_frames"] == sum(
        int(info["total_frames"]) for info in source_infos
    )
    assert output_info["total_tasks"] == int(source_infos[0]["total_tasks"])
    assert len(tasks_rows) == int(source_infos[0]["total_tasks"])
    assert result_payload["info"]["video_mode"] == expected_video_mode
    assert result_payload["debug"]["video_mode"] == expected_video_mode
    assert result_payload["info"]["partition_count"] > 0
    assert result_payload["info"]["data_chunks_written"] == len(data_paths)
    assert result_payload["debug"]["source_count"] == len(local_paths)
    assert result_payload["debug"]["source_episode_count"] == sum(
        int(info["total_episodes"]) for info in source_infos
    )
    assert result_payload["debug"]["source_frame_count"] == sum(
        int(info["total_frames"]) for info in source_infos
    )
    assert (
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).exists()
    assert not (output_root / "_staging").exists()

    if expected_video_mode == DATASET_MIX_LEROBOT_VIDEO_MODE_PRESERVE:
        assert output_info["video_path"] == DATASET_MIX_LEROBOT_VIDEO_PATH_TEMPLATE
        assert (output_root / "videos").exists()
        assert result_payload["debug"]["copied_video_file_count"] > 0
    else:
        assert output_info["video_path"] == ""
        assert result_payload["debug"]["copied_video_file_count"] == 0


def test_native_local_merge_preserves_alternate_lerobot_task_field_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_dataset(
        dataset_a,
        task_name="Push the T-shaped block onto the T-shaped target.",
        frame_values=[[0.1, 0.2]],
        timestamp_start=0.0,
        tasks_field_name="__index_level_0__",
    )
    _build_dataset(
        dataset_b,
        task_name="Push the T-shaped block onto the T-shaped target.",
        frame_values=[[0.3, 0.4]],
        timestamp_start=1.0,
        tasks_field_name="__index_level_0__",
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )

    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.partition_plan is not None
    control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    tasks_rows = pq.read_table(Path(completed.output_path) / "meta" / "tasks.parquet").to_pylist()
    assert tasks_rows == [
        {
            "task": "Push the T-shaped block onto the T-shaped target.",
            "task_index": 0,
        }
    ]


def test_native_local_merge_preserves_episode_order_across_source_chunk_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_multi_episode_dataset(
        dataset_a,
        episodes=[
            {
                "task_name": "pick",
                "frame_values": [[0.1, 0.2], [0.3, 0.4]],
                "timestamp_start": 0.0,
            },
            {
                "task_name": "place",
                "frame_values": [[0.5, 0.6]],
                "timestamp_start": 1.0,
            },
        ],
    )
    _build_dataset(
        dataset_b,
        task_name="stow",
        frame_values=[[1.1, 1.2]],
        timestamp_start=2.0,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    episode_rows = pq.read_table(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert [row["tasks"] for row in episode_rows] == [["pick"], ["place"], ["stow"]]
    assert [row["episode_index"] for row in episode_rows] == [0, 1, 2]
    assert [row["dataset_from_index"] for row in episode_rows] == [0, 2, 3]
    assert [row["dataset_to_index"] for row in episode_rows] == [2, 3, 4]

    data_rows = pq.read_table(output_root / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [row["episode_index"] for row in data_rows] == [0, 0, 1, 2]
    assert [row["index"] for row in data_rows] == [0, 1, 2, 3]
    assert [row["task_index"] for row in data_rows] == [0, 0, 1, 2]


def test_native_local_merge_promotes_partition_data_files_into_output_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2], [0.3, 0.4]],
        timestamp_start=0.0,
    )
    _build_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2], [1.3, 1.4]],
        timestamp_start=1.0,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_planner.DATASET_MIX_LEROBOT_PARTITION_FRAMES_PER_PARTITION",
        2,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_planner.DATASET_MIX_LEROBOT_PARTITION_EPISODES_PER_PARTITION",
        1,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_lerobot.DATASET_MIX_LEROBOT_DATA_ROWS_PER_CHUNK",
        2,
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    staged_root = output_root.parent / "_staging" / "partitions"
    output_chunk_paths = sorted((output_root / "data").glob("chunk-*/file-000.parquet"))
    staged_chunk_paths = sorted(staged_root.glob("partition-*/data.parquet"))

    assert len(output_chunk_paths) == 2
    assert len(staged_chunk_paths) == 2
    assert [path.read_bytes() for path in output_chunk_paths] == [
        path.read_bytes() for path in staged_chunk_paths
    ]

    result_payload = json.loads(
        (
            output_root.parent / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result_payload["info"]["data_chunks_written"] == 2
    assert result_payload["debug"]["data_chunks_written"] == 2
    assert result_payload["debug"]["partition_count"] == 2


def test_native_local_merge_rejects_inconsistent_episode_metadata_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2], [0.3, 0.4]],
        timestamp_start=0.0,
    )
    _build_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2]],
        timestamp_start=1.0,
    )
    _write_parquet(
        dataset_a / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                **pq.read_table(
                    dataset_a / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
                ).to_pylist()[0],
                "length": 3,
            }
        ],
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "failed"
    assert completed.error is not None
    assert "episode length does not match frame row count" in completed.error


def test_native_local_merge_rejects_non_contiguous_global_episode_index_ranges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_multi_episode_dataset(
        dataset_a,
        episodes=[
            {
                "task_name": "pick",
                "frame_values": [[0.1, 0.2], [0.3, 0.4]],
                "timestamp_start": 0.0,
            },
            {
                "task_name": "place",
                "frame_values": [[0.5, 0.6]],
                "timestamp_start": 1.0,
            },
        ],
    )
    _build_dataset(
        dataset_b,
        task_name="stow",
        frame_values=[[1.1, 1.2]],
        timestamp_start=2.0,
    )
    episode_rows = pq.read_table(
        dataset_a / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    episode_rows[1]["dataset_from_index"] = 3
    episode_rows[1]["dataset_to_index"] = 4
    _write_parquet(
        dataset_a / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        episode_rows,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "failed"
    assert completed.error is not None
    assert "dataset index ranges are not globally contiguous" in completed.error


def test_native_local_merge_cleans_stale_output_and_staging_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "dataset-a"
    dataset_b = tmp_path / "dataset-b"
    _build_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2]],
        timestamp_start=0.0,
    )
    _build_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2]],
        timestamp_start=1.0,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.output_artifact is not None
    stale_output_file = Path(queued.output_artifact.uri) / "stale.txt"
    stale_output_file.parent.mkdir(parents=True, exist_ok=True)
    stale_output_file.write_text("stale", encoding="utf-8")
    stale_staging_file = (
        Path(queued.output_artifact.uri).parent
        / "_staging"
        / "stale.txt"
    )
    stale_staging_file.parent.mkdir(parents=True, exist_ok=True)
    stale_staging_file.write_text("stale", encoding="utf-8")

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None
    assert not stale_output_file.exists()
    assert not stale_staging_file.exists()


def test_native_local_merge_accepts_exact_web_export_no_video_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "web-export-a"
    dataset_b = tmp_path / "web-export-b"
    _build_web_export_dataset(
        dataset_a,
        task_names=["pick", "place"],
        frame_values=[[0.1, 0.2], [0.3, 0.4]],
        timestamp_start=0.0,
    )
    _build_web_export_dataset(
        dataset_b,
        task_names=["stow"],
        frame_values=[[1.1, 1.2], [1.3, 1.4]],
        timestamp_start=1.0,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )

    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "native-local-lerobot"

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["video_path"] == ""
    assert info["files_size_in_mb"] == TEST_FILES_SIZE_MB
    assert "representation_id" not in info
    assert "naming_status" not in info
    assert "embodiment_ref" not in info
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["total_tasks"] == 3
    assert info["features"]["observation.state"]["names"] == {
        TEST_WEB_FEATURE_GROUP: ["joint_0", "joint_1"]
    }

    tasks_rows = pq.read_table(output_root / "meta" / "tasks.parquet").to_pylist()
    assert tasks_rows == [
        {"task": "pick", "task_index": 0},
        {"task": "place", "task_index": 1},
        {"task": "stow", "task_index": 2},
    ]

    data_rows = pq.read_table(output_root / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert set(data_rows[0].keys()) == {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
        "task_index",
    }
    assert [row["task_index"] for row in data_rows] == [0, 0, 2, 2]
    assert _round_vectors([row["action"] for row in data_rows]) == [
        [0.1, 0.2],
        [0.3, 0.4],
        [1.1, 1.2],
        [1.3, 1.4],
    ]

    episode_rows = pq.read_table(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert episode_rows[0]["tasks"] == ["pick", "place"]
    assert episode_rows[1]["tasks"] == ["stow"]


def test_native_local_merge_strips_missing_local_video_assets_into_exact_no_video_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "video-metadata-only-a"
    dataset_b = tmp_path / "video-metadata-only-b"
    _build_video_metadata_only_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2], [0.3, 0.4]],
        timestamp_start=0.0,
        create_video_files=False,
    )
    _build_video_metadata_only_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2], [1.3, 1.4]],
        timestamp_start=1.0,
        create_video_files=False,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "native-local-lerobot"

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["video_path"] == ""
    assert TEST_VIDEO_KEY not in info["features"]

    stats = json.loads((output_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert TEST_VIDEO_KEY not in stats

    episode_rows = pq.read_table(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert all(
        not any(key.startswith(f"videos/{TEST_VIDEO_KEY}/") for key in row)
        and not any(key.startswith(f"stats/{TEST_VIDEO_KEY}/") for key in row)
        for row in episode_rows
    )

    result_payload = json.loads(
        (
            output_root.parent / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result_payload["info"]["video_mode"] == "strip-missing-local-videos"
    assert result_payload["debug"]["video_mode"] == "strip-missing-local-videos"
    assert result_payload["debug"]["stripped_video_keys"] == [TEST_VIDEO_KEY]
    assert result_payload["debug"]["copied_video_file_count"] == 0


def test_native_local_merge_rejects_mixed_video_asset_availability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_a = tmp_path / "video-available-a"
    dataset_b = tmp_path / "video-missing-b"
    _build_video_metadata_only_dataset(
        dataset_a,
        task_name="pick",
        frame_values=[[0.1, 0.2]],
        timestamp_start=0.0,
        create_video_files=True,
    )
    _build_video_metadata_only_dataset(
        dataset_b,
        task_name="place",
        frame_values=[[1.1, 1.2]],
        timestamp_start=1.0,
        create_video_files=False,
    )
    local_paths = [dataset_a, dataset_b]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )
    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "failed"
    assert completed.error is not None
    assert "inconsistent video asset availability" in completed.error
    assert TEST_VIDEO_KEY in completed.error


def test_native_local_merge_accepts_real_frontend_exported_no_video_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "frontend-exports"
    _generate_frontend_export_fixtures(export_root)
    local_paths = [
        export_root / TEST_FRONTEND_FIXTURE_DATASET_NAME_A,
        export_root / TEST_FRONTEND_FIXTURE_DATASET_NAME_B,
    ]
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _normalized_paths: _build_treatment_analysis(local_paths),
    )

    control_plane = create_local_dataset_mix_control_plane(
        root=tmp_path / "control-plane",
        auto_start_worker=False,
    )

    queued = control_plane.submit_mix_job(
        _build_request([str(path) for path in local_paths])
    )
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "native-local-lerobot"

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert completed.status == "succeeded"
    assert completed.output_path is not None

    output_root = Path(completed.output_path)
    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["video_path"] == ""
    assert info["files_size_in_mb"] == TEST_FILES_SIZE_MB
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["total_tasks"] == 3
    assert info["features"]["observation.state"]["names"] == {
        TEST_WEB_FEATURE_GROUP: ["joint_a", "joint_b"]
    }

    tasks_rows = pq.read_table(output_root / "meta" / "tasks.parquet").to_pylist()
    assert tasks_rows == [
        {"task": "pick", "task_index": 0},
        {"task": "place", "task_index": 1},
        {"task": "stow", "task_index": 2},
    ]

    episode_rows = pq.read_table(
        output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pylist()
    assert episode_rows[0]["tasks"] == ["pick", "place"]
    assert episode_rows[1]["tasks"] == ["stow"]

    data_rows = pq.read_table(output_root / "data" / "chunk-000" / "file-000.parquet").to_pylist()
    assert set(data_rows[0].keys()) == {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
        "task_index",
    }
    assert [row["task_index"] for row in data_rows] == [0, 0, 2, 2]

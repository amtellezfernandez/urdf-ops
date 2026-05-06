#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OFFICIAL_LEROBOT_ENGINE = "official-lerobot"
OFFICIAL_LEROBOT_OUTPUT_REPO_ID = "urdf-studio/mixed-dataset"


def _load_official_lerobot() -> tuple[type, Any]:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset, merge_datasets


def _read_payload() -> dict[str, Any]:
    raw_payload = sys.stdin.read()
    if not raw_payload.strip():
        raise ValueError("Official LeRobot runner requires a JSON payload on stdin")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("Official LeRobot runner payload must be a JSON object")
    return payload


def _load_dataset(dataset_class: type, source: dict[str, Any]) -> Any:
    source_kind = source.get("source_kind")
    if source_kind == "local":
        root = source.get("canonical_source")
        if not isinstance(root, str) or not root:
            raise ValueError("Local LeRobot source is missing canonical_source")
        source_root = Path(root)
        return dataset_class(source_root.name, root=source_root)
    if source_kind == "repo":
        repo_id = source.get("source_value")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("Hub LeRobot source is missing source_value")
        return dataset_class(repo_id)
    raise ValueError(f"Official LeRobot runner does not support source kind: {source_kind}")


def _merge(payload: dict[str, Any]) -> dict[str, Any]:
    sources = payload.get("sources")
    output_path = payload.get("output_path")
    output_repo_id = payload.get("output_repo_id") or OFFICIAL_LEROBOT_OUTPUT_REPO_ID
    if not isinstance(sources, list) or not sources:
        raise ValueError("Official LeRobot runner requires at least one source")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("Official LeRobot runner requires output_path")
    if not isinstance(output_repo_id, str) or not output_repo_id:
        raise ValueError("Official LeRobot runner requires output_repo_id")

    dataset_class, merge_datasets = _load_official_lerobot()
    datasets = [_load_dataset(dataset_class, source) for source in sources]
    output_root = Path(output_path).resolve(strict=False)
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=output_root,
    )
    total_episodes = int(getattr(merged_dataset.meta, "total_episodes", 0))
    total_frames = int(getattr(merged_dataset.meta, "total_frames", 0))
    total_tasks = int(getattr(merged_dataset.meta, "total_tasks", 0))
    return {
        "success": True,
        "message": "Datasets mixed successfully with official LeRobot",
        "output_path": str(output_root),
        "info": {
            "engine": OFFICIAL_LEROBOT_ENGINE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "datasets_loaded": len(datasets),
            "dataset_sources": [source.get("canonical_source") for source in sources],
        },
        "debug": {"engine": OFFICIAL_LEROBOT_ENGINE},
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Official LeRobot dataset merge runner")
    parser.add_argument("--probe", action="store_true", help="Verify official LeRobot imports")
    args = parser.parse_args()

    try:
        if args.probe:
            _load_official_lerobot()
            print(json.dumps({"available": True, "engine": OFFICIAL_LEROBOT_ENGINE}))
            return 0
        print(json.dumps(_merge(_read_payload())))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

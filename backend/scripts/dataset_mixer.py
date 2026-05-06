#!/usr/bin/env python3
"""
Dataset Mixer using RoboCandyWrapper
Mixes local datasets with Hugging Face datasets
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from backend.services.datasets_params import (
    DATASET_MIX_ARCHIVE_MAX_ENTRY_BYTES,
    DATASET_MIX_ARCHIVE_MAX_ENTRY_COUNT,
    DATASET_MIX_ARCHIVE_MAX_TOTAL_BYTES,
    is_dataset_mix_local_path_allowed,
    resolve_dataset_mix_local_path,
)


def _load_dataset_builder() -> Callable[[list[str]], object]:
    try:
        from robocandywrapper import make_dataset_without_config
    except ImportError as exc:
        raise RuntimeError(
            "robocandywrapper is not installed. Run: pip install robocandywrapper"
        ) from exc
    return make_dataset_without_config


def _normalize_archive_member_path(member_name: str) -> Path:
    normalized_name = member_name.replace("\\", "/")
    member_path = PurePosixPath(normalized_name)
    parts = [part for part in member_path.parts if part not in ("", ".")]
    if not parts or member_path.is_absolute() or any(part == ".." for part in parts):
        raise ValueError(f"Archive entry has unsafe path: {member_name}")
    return Path(*parts)


def _extract_dataset_zip(path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_zip_"))
    extracted_files = 0
    total_uncompressed_bytes = 0
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            for info in zip_ref.infolist():
                if info.is_dir():
                    continue
                extracted_files += 1
                if extracted_files > DATASET_MIX_ARCHIVE_MAX_ENTRY_COUNT:
                    raise ValueError("Dataset archive exceeds configured file-count limit")
                if info.file_size > DATASET_MIX_ARCHIVE_MAX_ENTRY_BYTES:
                    raise ValueError("Dataset archive contains a file that exceeds the configured file-size limit")
                total_uncompressed_bytes += info.file_size
                if total_uncompressed_bytes > DATASET_MIX_ARCHIVE_MAX_TOTAL_BYTES:
                    raise ValueError("Dataset archive exceeds configured total-size limit")

                relative_path = _normalize_archive_member_path(info.filename)
                output_path = temp_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(info, "r") as source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _collect_dataset_sources(
    repo_ids: list[str],
    local_dataset_paths: list[str] | None = None,
) -> tuple[list[str], list[Path]]:
    all_repos = list(repo_ids) if repo_ids else []
    temp_dirs: list[Path] = []

    if local_dataset_paths:
        for local_path in local_dataset_paths:
            resolved_path = resolve_dataset_mix_local_path(local_path)
            if not is_dataset_mix_local_path_allowed(resolved_path):
                raise ValueError(f"Local dataset path is outside configured allowlisted roots: {local_path}")
            if resolved_path.exists() and resolved_path.is_dir():
                all_repos.append(str(resolved_path))
            elif resolved_path.exists() and resolved_path.is_file() and resolved_path.suffix.lower() == ".zip":
                extracted_dir = _extract_dataset_zip(resolved_path)
                temp_dirs.append(extracted_dir)
                all_repos.append(str(extracted_dir))
            else:
                print(f"WARNING: Local dataset path not found: {local_path}", file=sys.stderr)

    return all_repos, temp_dirs


def mix_datasets(
    repo_ids: list[str],
    local_dataset_paths: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Mix datasets from Hugging Face and local paths.

    Args:
        repo_ids: List of Hugging Face dataset IDs (e.g., ["lerobot/svla_so100_pickplace"])
        local_dataset_paths: List of local dataset paths (directories containing LeRobot datasets)
        output_path: Optional path to save the mixed dataset

    Returns:
        Dict with mixed dataset info
    """
    temp_dirs: list[Path] = []
    try:
        dataset_builder = _load_dataset_builder()
        all_repos, temp_dirs = _collect_dataset_sources(repo_ids, local_dataset_paths)

        if not all_repos:
            return {
                "success": False,
                "error": "No valid datasets provided"
            }

        print(f"Loading and mixing {len(all_repos)} dataset(s)...", file=sys.stderr)
        dataset = dataset_builder(all_repos)
        total_episodes = len(dataset)

        info = {
            "total_episodes": total_episodes,
            "datasets_loaded": len(all_repos),
            "dataset_sources": all_repos,
        }

        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output path specified: {output_path}", file=sys.stderr)

        return {
            "success": True,
            "info": info,
            "total_episodes": total_episodes,
            "output_path": str(output_path) if output_path else None,
        }
    except Exception as exc:
        error_msg = str(exc)
        print(f"ERROR: Failed to mix datasets: {error_msg}", file=sys.stderr)
        return {
            "success": False,
            "error": error_msg,
        }
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Mix LeRobot datasets using RoboCandyWrapper")
    parser.add_argument(
        "--repo-ids",
        type=str,
        nargs="+",
        help="Hugging Face dataset IDs (e.g., lerobot/svla_so100_pickplace)",
    )
    parser.add_argument(
        "--local-paths",
        type=str,
        nargs="+",
        help="Local dataset directory paths or zip files",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for mixed dataset (optional)",
    )

    args = parser.parse_args()

    if not args.repo_ids and not args.local_paths:
        result = {
            "success": False,
            "error": "At least one --repo-ids or --local-paths is required"
        }
        print(json.dumps(result))
        sys.exit(1)

    result = mix_datasets(
        repo_ids=args.repo_ids or [],
        local_dataset_paths=args.local_paths,
        output_path=args.output,
    )

    print(json.dumps(result))

    if not result.get("success", False):
        sys.exit(1)


if __name__ == "__main__":
    main()

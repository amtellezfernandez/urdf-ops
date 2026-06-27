from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.core.paths import BASE_DIR, SCRIPTS_DIR
from backend.models.datasets import DatasetMixJobManifest, DatasetMixSourceRef
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_OUTPUT_REPO_ID,
    DATASET_MIX_LEROBOT_PYTHON_ENV,
    DATASET_MIX_LEROBOT_RUNNER_SCRIPT,
    DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL,
    DATASET_MIX_LEROBOT_TOOLCHAIN_DIRNAME,
)


@dataclass(frozen=True)
class OfficialLeRobotAvailability:
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class OfficialLeRobotBindings:
    dataset_class: type
    merge_datasets: Any


def _default_lerobot_python_path() -> Path:
    return BASE_DIR / DATASET_MIX_LEROBOT_TOOLCHAIN_DIRNAME / "bin" / "python3"


def _resolve_lerobot_python_path() -> Path:
    configured = os.getenv(DATASET_MIX_LEROBOT_PYTHON_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return _default_lerobot_python_path()


class OfficialLeRobotDatasetMixer:
    def availability(self) -> OfficialLeRobotAvailability:
        try:
            self._load_bindings()
        except Exception as import_exc:
            runner_result = self._probe_external_runner()
            if runner_result.available:
                return runner_result
            return OfficialLeRobotAvailability(
                available=False,
                reason=(
                    "Official LeRobot dataset tools are unavailable in-process "
                    f"({import_exc}) and via managed runner ({runner_result.reason})."
                ),
            )
        return OfficialLeRobotAvailability(available=True)

    def can_execute(self, manifest: DatasetMixJobManifest) -> bool:
        return bool(manifest.sources and manifest.output_artifact.uri)

    def merge(self, manifest: DatasetMixJobManifest) -> dict[str, Any]:
        if not self.can_execute(manifest):
            raise HTTPException(
                status_code=400,
                detail="Official LeRobot merge requires sources and a writable output artifact",
            )
        try:
            return self._merge_in_process(manifest)
        except Exception as import_or_merge_exc:
            try:
                return self._merge_external(manifest)
            except Exception as runner_exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Official LeRobot merge failed in-process "
                        f"({import_or_merge_exc}) and via managed runner ({runner_exc})."
                    ),
                ) from runner_exc

    def _merge_in_process(self, manifest: DatasetMixJobManifest) -> dict[str, Any]:
        output_root = Path(str(manifest.output_artifact.uri)).resolve(strict=False)
        bindings = self._load_bindings()
        datasets = [
            self._load_dataset(bindings.dataset_class, source)
            for source in manifest.sources
        ]
        merged_dataset = bindings.merge_datasets(
            datasets,
            output_repo_id=DATASET_MIX_LEROBOT_OUTPUT_REPO_ID,
            output_dir=output_root,
        )
        return self._payload_from_merged_dataset(
            merged_dataset,
            output_root=output_root,
            source_count=len(datasets),
            dataset_sources=[source.canonical_source for source in manifest.sources],
        )

    def _merge_external(self, manifest: DatasetMixJobManifest) -> dict[str, Any]:
        output_path = manifest.output_artifact.uri
        if not output_path:
            raise ValueError("Official LeRobot merge output artifact is missing a URI")
        runner_payload = {
            "output_path": output_path,
            "output_repo_id": DATASET_MIX_LEROBOT_OUTPUT_REPO_ID,
            "sources": [source.model_dump(mode="json") for source in manifest.sources],
        }
        return self._run_external_runner([], payload=runner_payload)

    def _probe_external_runner(self) -> OfficialLeRobotAvailability:
        try:
            self._run_external_runner(["--probe"], payload=None)
        except Exception as exc:
            return OfficialLeRobotAvailability(available=False, reason=str(exc))
        return OfficialLeRobotAvailability(available=True)

    def _run_external_runner(
        self,
        args: list[str],
        *,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        python_path = _resolve_lerobot_python_path()
        if not python_path.exists():
            raise FileNotFoundError(
                f"Managed LeRobot Python is missing at {python_path}. Run npm run setup."
            )
        runner_path = SCRIPTS_DIR / DATASET_MIX_LEROBOT_RUNNER_SCRIPT
        if not runner_path.exists():
            raise FileNotFoundError(f"Official LeRobot runner is missing at {runner_path}")
        process = subprocess.run(
            [str(python_path), str(runner_path), *args],
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if process.returncode != 0:
            raise RuntimeError(stderr or stdout or "Official LeRobot runner failed")
        if not stdout:
            return {}
        parsed = json.loads(stdout)
        if not isinstance(parsed, dict):
            raise RuntimeError("Official LeRobot runner returned a non-object JSON payload")
        return parsed

    def _payload_from_merged_dataset(
        self,
        merged_dataset: Any,
        *,
        output_root: Path,
        source_count: int,
        dataset_sources: list[str],
    ) -> dict[str, Any]:
        total_episodes = int(getattr(merged_dataset.meta, "total_episodes", 0))
        total_frames = int(getattr(merged_dataset.meta, "total_frames", 0))
        total_tasks = int(getattr(merged_dataset.meta, "total_tasks", 0))
        return {
            "success": True,
            "message": "Datasets mixed successfully with official LeRobot",
            "output_path": str(output_root),
            "info": {
                "engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL,
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "total_tasks": total_tasks,
                "datasets_loaded": source_count,
                "dataset_sources": dataset_sources,
            },
            "debug": {
                "engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL,
            },
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
        }

    def _load_dataset(self, dataset_class: type, source: DatasetMixSourceRef) -> Any:
        dataset_kwargs = {"episodes": source.episodes} if source.episodes else {}
        if source.source_kind == "local":
            source_root = Path(source.canonical_source)
            return dataset_class(source_root.name, root=source_root, **dataset_kwargs)
        if source.source_kind == "repo":
            return dataset_class(source.source_value, **dataset_kwargs)
        raise HTTPException(
            status_code=400,
            detail=f"Official LeRobot merge does not support source kind: {source.source_kind}",
        )

    def _load_bindings(self) -> OfficialLeRobotBindings:
        from lerobot.datasets.dataset_tools import merge_datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return OfficialLeRobotBindings(
            dataset_class=LeRobotDataset,
            merge_datasets=merge_datasets,
        )

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.core.paths import BASE_DIR
from backend.models.training import TrainingStartRequest
from backend.services.dataset_source_contract import (
    DatasetSourceContract,
    resolve_dataset_source_contract,
)
from backend.services.training_params import (
    TRAINING_DEFAULT_OUTPUT_DIR,
    TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME,
    TRAINING_OUTPUT_ROOTS_ENV,
)


@dataclass(frozen=True, slots=True)
class TrainingLaunchContract:
    dataset: DatasetSourceContract
    output_dir: Path
    tracker_config: dict[str, Any]
    compute_config: dict[str, Any]
    training_config: dict[str, Any]


def get_enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def dump_internal_model(value: Any) -> dict[str, Any]:
    if not hasattr(value, "model_dump"):
        return {}
    return value.model_dump(by_alias=False)


def _read_training_output_roots() -> tuple[Path, ...]:
    roots = [BASE_DIR / TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME]
    raw = os.getenv(TRAINING_OUTPUT_ROOTS_ENV, "").strip()
    if raw:
        for entry in raw.split(os.pathsep):
            normalized = entry.strip()
            if normalized:
                roots.append(Path(normalized).expanduser())
    return tuple(dict.fromkeys(root.resolve(strict=False) for root in roots))


def _is_relative_to(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def resolve_training_output_dir(output_dir: str | None) -> Path:
    raw_output_dir = (output_dir or TRAINING_DEFAULT_OUTPUT_DIR).strip()
    if not raw_output_dir:
        raw_output_dir = TRAINING_DEFAULT_OUTPUT_DIR

    candidate = Path(raw_output_dir).expanduser()
    if not candidate.is_absolute():
        first_part = candidate.parts[0] if candidate.parts else ""
        if first_part in {".", TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME}:
            candidate = BASE_DIR / candidate
        else:
            candidate = BASE_DIR / TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME / candidate
    resolved = candidate.resolve(strict=False)

    allowed_roots = _read_training_output_roots()
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError(
            f"Training output directory must stay under {TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME} "
            f"or a root configured in {TRAINING_OUTPUT_ROOTS_ENV}."
        )
    return resolved


def _resolve_training_dataset_source(
    request: TrainingStartRequest,
) -> DatasetSourceContract:
    try:
        return resolve_dataset_source_contract(
            source=get_enum_value(request.dataset.source),
            repo_id=request.dataset.repo_id,
            local_path=request.dataset.local_path,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        raise ValueError(detail) from exc


def _build_dataset_config(
    request: TrainingStartRequest,
    dataset: DatasetSourceContract,
) -> dict[str, Any]:
    dataset_config = dump_internal_model(request.dataset)
    if dataset.source_kind == "repo":
        dataset_config["source"] = "huggingface"
        dataset_config["repo_id"] = dataset.canonical_source
        dataset_config["local_path"] = None
    else:
        dataset_config["source"] = "local"
        dataset_config["repo_id"] = None
        dataset_config["local_path"] = dataset.canonical_source
    return dataset_config


def build_training_launch_contract(
    request: TrainingStartRequest,
    *,
    job_id: str,
    lerobot_python_path: Path,
) -> TrainingLaunchContract:
    dataset = _resolve_training_dataset_source(request)
    output_dir = resolve_training_output_dir(request.training.output_dir)
    tracker_config = {
        "type": get_enum_value(request.tracker.type),
        "tracking_uri": request.tracker.tracking_uri,
        "experiment_name": request.tracker.experiment_name,
        "project": request.tracker.project,
        "entity": request.tracker.entity,
        "output_dir": str(output_dir),
    }
    compute_config = {
        "type": get_enum_value(request.compute.type),
        "api_key": request.compute.api_key,
        "default_gpu": request.compute.gpu,
        "output_dir": str(output_dir),
    }
    if get_enum_value(request.compute.type) == "local":
        compute_config["python_path"] = str(lerobot_python_path)

    training_params = dump_internal_model(request.training)
    training_params["output_dir"] = str(output_dir)

    return TrainingLaunchContract(
        dataset=dataset,
        output_dir=output_dir,
        tracker_config=tracker_config,
        compute_config=compute_config,
        training_config={
            "job_id": job_id,
            "dataset": _build_dataset_config(request, dataset),
            "model": dump_internal_model(request.model),
            "training": training_params,
            "tracker": tracker_config,
            "device": request.compute.device,
        },
    )

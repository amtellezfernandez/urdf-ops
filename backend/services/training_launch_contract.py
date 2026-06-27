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
from backend.services.urdf_action_schema import (
    URDF_ACTION_UNITS_NATIVE,
    build_urdf_action_schema,
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


def _model_architecture(request: TrainingStartRequest) -> str:
    return get_enum_value(request.model.architecture)


def _build_embodiment_config(request: TrainingStartRequest) -> dict[str, Any] | None:
    if not request.urdf:
        return None

    model_config = dump_internal_model(request.model).get("config", {})
    action_units = (
        str(model_config.get("action_units")).strip()
        if isinstance(model_config, dict) and model_config.get("action_units")
        else URDF_ACTION_UNITS_NATIVE
    )
    return build_urdf_action_schema(
        request.urdf,
        robot_name=request.robot_name,
        action_units=action_units,
    )


def _build_model_config(
    request: TrainingStartRequest,
    embodiment_config: dict[str, Any] | None,
) -> dict[str, Any]:
    model_config = dump_internal_model(request.model)
    architecture = _model_architecture(request)
    if architecture not in {"dreamzero", "lereal_world_model"}:
        return model_config

    raw_config = model_config.get("config", {})
    architecture_config = dict(raw_config) if isinstance(raw_config, dict) else {}
    if architecture == "lereal_world_model":
        if embodiment_config is not None:
            explicit_action_schema = architecture_config.get("action_schema")
            action_schema = (
                explicit_action_schema
                if isinstance(explicit_action_schema, dict)
                else embodiment_config
            )
            architecture_config["action_schema"] = action_schema
            architecture_config.setdefault("action_dim", action_schema["action_dim"])
            architecture_config.setdefault("action_joint_names", action_schema["joint_names"])
            architecture_config.setdefault(
                "action_units",
                action_schema.get("action_units", URDF_ACTION_UNITS_NATIVE),
            )
            model_config["config"] = architecture_config
        return model_config

    explicit_action_schema = architecture_config.get("action_schema")
    if embodiment_config is None and not isinstance(explicit_action_schema, dict):
        raise ValueError(
            "DreamZero training requires a URDF in the request or model.config.action_schema "
            "so the action dimension and joint order can be derived."
        )

    action_schema = explicit_action_schema if isinstance(explicit_action_schema, dict) else embodiment_config
    if not isinstance(action_schema, dict):
        raise ValueError("DreamZero action schema must be an object.")
    if not action_schema.get("joint_names") or not action_schema.get("action_dim"):
        raise ValueError("DreamZero action schema requires joint_names and action_dim.")
    architecture_config["action_schema"] = action_schema
    architecture_config["action_dim"] = action_schema["action_dim"]
    architecture_config["action_joint_names"] = action_schema["joint_names"]
    architecture_config.setdefault(
        "action_units",
        action_schema.get("action_units", URDF_ACTION_UNITS_NATIVE),
    )
    model_config["config"] = architecture_config
    return model_config


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
    compute_type = get_enum_value(request.compute.type)
    if compute_type == "local":
        compute_config["python_path"] = str(lerobot_python_path)
    elif compute_type == "ssh":
        compute_config.update(
            {
                "host": request.compute.ssh_host,
                "user": request.compute.ssh_user,
                "port": request.compute.ssh_port,
                "key_path": request.compute.ssh_key_path,
                "output_dir": request.compute.remote_output_dir,
                "docker_image": request.compute.docker_image,
                "docker_args": request.compute.docker_args,
                "ssh_options": request.compute.ssh_options,
                "use_gpu": request.compute.device == "cuda",
            }
        )

    training_params = dump_internal_model(request.training)
    training_params["output_dir"] = str(output_dir)
    embodiment_config = (
        _build_embodiment_config(request)
        if _model_architecture(request) in {"dreamzero", "lereal_world_model"}
        else None
    )
    model_config = _build_model_config(request, embodiment_config)

    training_config = {
        "job_id": job_id,
        "dataset": _build_dataset_config(request, dataset),
        "model": model_config,
        "training": training_params,
        "tracker": tracker_config,
        "device": request.compute.device,
    }
    if embodiment_config is not None:
        training_config["embodiment"] = embodiment_config

    return TrainingLaunchContract(
        dataset=dataset,
        output_dir=output_dir,
        tracker_config=tracker_config,
        compute_config=compute_config,
        training_config=training_config,
    )

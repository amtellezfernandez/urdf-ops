#!/usr/bin/env python3
"""Training script for robot policies.

This script is launched as a subprocess by the training service.
It handles the actual training loop using LeRobot.

Usage:
    python train_policy.py --config config.json

The script:
1. Loads configuration from JSON
2. Sets up dataset and model using LeRobot
3. Initializes experiment tracking
4. Runs training loop with progress reporting
5. Saves checkpoints and final model

Requirements:
    - LeRobot >= 0.4.0 must be installed
    - PyTorch with CUDA support recommended
    - HuggingFace token for most datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Support direct execution with `python backend/scripts/train_policy.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.training_policy_compat import normalize_policy_id, prepare_policy_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train robot policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed: {seed}")


def get_tracker(tracker_config: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Initialize experiment tracker from config.

    Args:
        tracker_config: Tracker configuration dict with 'type' key

    Returns:
        Initialized tracker or None if disabled
    """
    if not tracker_config:
        return None

    tracker_type = tracker_config.get("type", "none")
    if tracker_type == "none":
        return None

    try:
        from backend.robotops import get_tracker as _get_tracker
        tracker = _get_tracker(tracker_config)
        logger.info(f"Initialized {tracker_type} tracker")
        return tracker
    except ImportError:
        logger.warning("Could not import tracker backend, metrics will not be logged")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize tracker: {e}")
        return None


def write_progress(
    job_dir: Path,
    current_epoch: int,
    total_epochs: int,
    current_step: int,
    total_steps: int,
    metrics: Dict[str, Any],
) -> None:
    """Write progress to file for status polling."""
    progress = {
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "current_step": current_step,
        "total_steps": total_steps,
        "metrics": metrics,
        "updated_at": datetime.now().isoformat(),
    }

    progress_file = job_dir / "progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def append_metrics(
    job_dir: Path,
    step: int,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Append a metrics snapshot for charting and log replay."""
    entry = {
        "step": step,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    metrics_file = job_dir / "metrics.jsonl"
    with open(metrics_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_policy_config_class(architecture: str):
    """Get the config class for a given policy architecture."""
    architecture = normalize_policy_id(architecture)
    if architecture == "act":
        from lerobot.policies import ACTConfig
        return ACTConfig
    elif architecture == "diffusion":
        from lerobot.policies import DiffusionConfig
        return DiffusionConfig
    elif architecture == "tdmpc":
        from lerobot.policies import TDMPCConfig
        return TDMPCConfig
    elif architecture == "vqbet":
        from lerobot.policies import VQBeTConfig
        return VQBeTConfig
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_policy_class(architecture: str):
    """Get the policy class for a given architecture."""
    architecture = normalize_policy_id(architecture)
    if architecture == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    elif architecture == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    elif architecture == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
        return TDMPCPolicy
    elif architecture == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        return VQBeTPolicy
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _as_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("DreamZero runner_args must be a list of strings when provided.")
    return [str(item) for item in value]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _dreamzero_runner_command(model_runtime_config: Dict[str, Any], config_path: Path) -> list[str]:
    runner_script = (
        _as_optional_string(model_runtime_config.get("runner_script"))
        or _as_optional_string(os.environ.get("URDF_OPS_DREAMZERO_RUNNER_SCRIPT"))
    )
    runner_module = (
        _as_optional_string(model_runtime_config.get("runner_module"))
        or _as_optional_string(os.environ.get("URDF_OPS_DREAMZERO_RUNNER_MODULE"))
    )
    runner_args = _as_string_list(model_runtime_config.get("runner_args"))
    if runner_script and runner_module:
        raise ValueError("Set only one DreamZero runner_script or runner_module.")
    if runner_script:
        return [sys.executable, runner_script, "--config", str(config_path), *runner_args]
    if runner_module:
        return [sys.executable, "-m", runner_module, "--config", str(config_path), *runner_args]
    raise RuntimeError(
        "DreamZero training requires model.config.runner_script, model.config.runner_module, "
        "URDF_OPS_DREAMZERO_RUNNER_SCRIPT, or URDF_OPS_DREAMZERO_RUNNER_MODULE."
    )


def train_with_dreamzero(config: Dict[str, Any], job_dir: Path) -> None:
    """Launch an external DreamZero-compatible runner with a URDF action schema."""

    model_config = config.get("model", {})
    runtime_config = model_config.get("config", {})
    if not isinstance(runtime_config, dict):
        raise ValueError("DreamZero model config must be an object.")

    action_schema = runtime_config.get("action_schema")
    if not isinstance(action_schema, dict):
        raise ValueError("DreamZero model config is missing action_schema.")
    if not action_schema.get("joint_names") or not action_schema.get("action_dim"):
        raise ValueError("DreamZero action_schema must include joint_names and action_dim.")

    dreamzero_config_path = job_dir / "dreamzero_config.json"
    _write_json(dreamzero_config_path, config)
    write_progress(
        job_dir=job_dir,
        current_epoch=0,
        total_epochs=1,
        current_step=0,
        total_steps=1,
        metrics={
            "status": "running",
            "dreamzero_action_dim": float(action_schema["action_dim"]),
        },
    )

    cmd = _dreamzero_runner_command(runtime_config, dreamzero_config_path)
    logger.info("Launching DreamZero runner: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=job_dir,
        env={
            **os.environ,
            "URDF_OPS_DREAMZERO_CONFIG": str(dreamzero_config_path),
            "URDF_OPS_DREAMZERO_ACTION_SCHEMA": json.dumps(action_schema),
        },
        text=True,
    )
    if result.returncode != 0:
        write_progress(
            job_dir=job_dir,
            current_epoch=0,
            total_epochs=1,
            current_step=0,
            total_steps=1,
            metrics={"status": "failed"},
        )
        raise RuntimeError(f"DreamZero runner failed with exit code {result.returncode}")

    write_progress(
        job_dir=job_dir,
        current_epoch=1,
        total_epochs=1,
        current_step=1,
        total_steps=1,
        metrics={
            "status": "completed",
            "dreamzero_action_dim": float(action_schema["action_dim"]),
        },
    )


LEREAL_WORLD_MODEL_ARCHITECTURE = "lereal_world_model"
LEREAL_WORLD_MODEL_DEFAULT_REPO_URL = "https://github.com/amtellezfernandez/LeRealWorldModel.git"
LEREAL_WORLD_MODEL_DEFAULT_REPO_REF = "main"
LEREAL_WORLD_MODEL_REPO_DIRNAME = "LeRealWorldModel"
LEREAL_WORLD_MODEL_DEFAULT_LOCAL_REPO_ID = "local/urdf-ops-dataset"
LEREAL_WORLD_MODEL_STAGE_SETUP = "setup"
LEREAL_WORLD_MODEL_STAGE_STAGE1 = "stage1"
LEREAL_WORLD_MODEL_STAGE_STAGE2 = "stage2"
LEREAL_WORLD_MODEL_STAGE_EXPORT = "export"
LEREAL_WORLD_MODEL_STAGE_COMPLETE = "complete"


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _as_optional_path_string(value: Any) -> str | None:
    string_value = _as_optional_string(value)
    if string_value is None:
        return None
    return str(Path(string_value).expanduser())


def _lwm_write_progress(
    job_dir: Path,
    *,
    stage: str,
    current_step: int,
    total_steps: int,
    status: str = "running",
    error: str | None = None,
) -> None:
    metrics: Dict[str, Any] = {
        "status": status,
        "lereal_world_model_stage": stage,
    }
    if error:
        metrics["error"] = error
    write_progress(
        job_dir=job_dir,
        current_epoch=current_step,
        total_epochs=total_steps,
        current_step=current_step,
        total_steps=total_steps,
        metrics=metrics,
    )
    append_metrics(
        job_dir=job_dir,
        step=current_step,
        epoch=current_step,
        metrics=metrics,
    )


def _run_lereal_world_model_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: Dict[str, str],
) -> subprocess.CompletedProcess[str]:
    logger.info("Running LeRealWorldModel command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        check=False,
    )


def _resolve_lereal_world_model_repo(
    runtime_config: Dict[str, Any],
    job_dir: Path,
    env: Dict[str, str],
) -> Path:
    explicit_repo_path = _as_optional_path_string(
        runtime_config.get("repo_path") or os.environ.get("URDF_OPS_LEREALWORLDMODEL_REPO")
    )
    if explicit_repo_path:
        repo_path = Path(explicit_repo_path).resolve(strict=False)
        if not repo_path.is_dir():
            raise RuntimeError(f"LeRealWorldModel repo_path does not exist: {repo_path}")
        return repo_path

    repo_url = (
        _as_optional_string(runtime_config.get("repo_url"))
        or os.environ.get("URDF_OPS_LEREALWORLDMODEL_REPO_URL")
        or LEREAL_WORLD_MODEL_DEFAULT_REPO_URL
    )
    repo_ref = (
        _as_optional_string(runtime_config.get("repo_ref"))
        or os.environ.get("URDF_OPS_LEREALWORLDMODEL_REPO_REF")
        or LEREAL_WORLD_MODEL_DEFAULT_REPO_REF
    )
    repo_path = job_dir / LEREAL_WORLD_MODEL_REPO_DIRNAME
    if repo_path.is_dir():
        return repo_path

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        repo_ref,
        repo_url,
        str(repo_path),
    ]
    result = _run_lereal_world_model_command(cmd, cwd=job_dir, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone LeRealWorldModel from {repo_url}@{repo_ref}")
    return repo_path


def _install_lereal_world_model_repo(
    repo_path: Path,
    *,
    env: Dict[str, str],
) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_path)]
    result = _run_lereal_world_model_command(cmd, cwd=repo_path, env=env)
    if result.returncode != 0:
        raise RuntimeError("Failed to install LeRealWorldModel dependencies")


def _prepare_lereal_world_model_checkout(repo_path: Path) -> None:
    """Patch checkout layout mismatches before running the upstream project."""

    legacy_package_path = repo_path / "lewm"
    source_package_path = repo_path / "lewm_robot"
    if not legacy_package_path.exists() and source_package_path.is_dir():
        shutil.copytree(source_package_path, legacy_package_path)


def _lwm_dataset_overrides(
    config: Dict[str, Any],
    runtime_config: Dict[str, Any],
    *,
    stage: str,
) -> list[str]:
    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        dataset_config = {}

    overrides: list[str] = []
    if stage == LEREAL_WORLD_MODEL_STAGE_STAGE1:
        overrides.append("data.dataset._target_=lewm_robot.data.lerobot_adapter.LeRobotWMDataset")
    source = str(dataset_config.get("source") or "huggingface")
    if source == "local":
        local_path = _as_optional_string(dataset_config.get("local_path"))
        if not local_path:
            raise ValueError("LeRealWorldModel local training requires dataset.local_path.")
        repo_id = (
            _as_optional_string(runtime_config.get("local_repo_id"))
            or LEREAL_WORLD_MODEL_DEFAULT_LOCAL_REPO_ID
        )
        overrides.extend(
            [
                f"data.dataset.repo_id={repo_id}",
                f"data.dataset.root={Path(local_path).expanduser().resolve(strict=False)}",
            ]
        )
    else:
        repo_id = _as_optional_string(dataset_config.get("repo_id"))
        if not repo_id:
            raise ValueError("LeRealWorldModel Hugging Face training requires dataset.repo_id.")
        overrides.extend(
            [
                f"data.dataset.repo_id={repo_id}",
                "data.dataset.root=null",
            ]
        )

    image_key = _as_optional_string(runtime_config.get("image_key")) or "observation.images.up"
    image_key2 = _as_optional_string(runtime_config.get("image_key2"))
    overrides.append(f"data.dataset.image_key={image_key}")
    if image_key2:
        overrides.append(f"data.dataset.image_key2={image_key2}")
        if stage == LEREAL_WORLD_MODEL_STAGE_STAGE2:
            overrides.append(f"data.dataset.image_keys=[{image_key},{image_key2}]")
    else:
        overrides.append("data.dataset.image_key2=null")
        if stage == LEREAL_WORLD_MODEL_STAGE_STAGE2:
            overrides.append(f"data.dataset.image_keys=[{image_key}]")

    if stage == LEREAL_WORLD_MODEL_STAGE_STAGE1:
        action_key = _as_optional_string(runtime_config.get("action_key"))
        proprio_key = _as_optional_string(runtime_config.get("proprio_key"))
        if action_key:
            overrides.append(f"data.dataset.action_key={action_key}")
        if proprio_key:
            overrides.append(f"data.dataset.proprio_key={proprio_key}")

    frameskip = _as_positive_int(runtime_config.get("frameskip"), 5)
    action_dim = _as_positive_int(runtime_config.get("action_dim"), 6)
    overrides.extend(
        [
            f"data.dataset.frameskip={frameskip}",
            f"wm.action_dim={action_dim}",
        ]
    )
    if stage == LEREAL_WORLD_MODEL_STAGE_STAGE2:
        overrides.append(f"wm.frameskip={frameskip}")
    return overrides


def _latest_lereal_world_model_checkpoint(repo_path: Path, output_model_name: str) -> Path:
    checkpoint_root = repo_path / "checkpoints"
    candidates = sorted(
        checkpoint_root.rglob(f"{output_model_name}_epoch_*_object.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            checkpoint_root.rglob("*_epoch_*_object.ckpt"),
            key=lambda path: path.stat().st_mtime,
        )
    if not candidates:
        raise RuntimeError("LeRealWorldModel Stage 1 did not produce a *_object.ckpt checkpoint.")
    return candidates[-1]


def _normalizers_path_for_world_model(world_model_path: Path, output_model_name: str) -> Path:
    expected = world_model_path.parent / f"{output_model_name}_normalizers.pt"
    if expected.exists():
        return expected
    candidates = sorted(world_model_path.parent.glob("*_normalizers.pt"))
    return candidates[-1] if candidates else expected


def train_with_lereal_world_model(config: Dict[str, Any], job_dir: Path) -> None:
    """Run LeRealWorldModel JEPA/GC-IDM training from an Ops training job."""

    model_config = config.get("model", {})
    runtime_config = model_config.get("config", {}) if isinstance(model_config, dict) else {}
    if not isinstance(runtime_config, dict):
        raise ValueError("LeRealWorldModel model config must be an object.")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    total_steps = 2
    if _as_bool(runtime_config.get("run_stage2"), True):
        total_steps += 1
    if _as_bool(runtime_config.get("export_policy"), True):
        total_steps += 1

    try:
        _lwm_write_progress(
            job_dir,
            stage=LEREAL_WORLD_MODEL_STAGE_SETUP,
            current_step=0,
            total_steps=total_steps,
        )
        repo_path = _resolve_lereal_world_model_repo(runtime_config, job_dir, env)
        _prepare_lereal_world_model_checkout(repo_path)
        env["PYTHONPATH"] = (
            f"{repo_path}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(repo_path)
        )
        if _as_bool(runtime_config.get("auto_install"), True):
            _install_lereal_world_model_repo(repo_path, env=env)

        training_config = config.get("training", {})
        if not isinstance(training_config, dict):
            training_config = {}

        stage1_config = _as_optional_string(runtime_config.get("stage1_config")) or "lewm_so100_topcam"
        output_model_name = (
            _as_optional_string(runtime_config.get("output_model_name"))
            or stage1_config
        )
        run_subdir = _as_optional_string(training_config.get("run_name")) or job_dir.name
        stage1_epochs = _as_positive_int(
            runtime_config.get("stage1_epochs") or training_config.get("epochs"),
            50,
        )
        stage1_dataset_overrides = _lwm_dataset_overrides(
            config,
            runtime_config,
            stage=LEREAL_WORLD_MODEL_STAGE_STAGE1,
        )
        stage1_overrides = [
            *stage1_dataset_overrides,
            f"output_model_name={output_model_name}",
            f"subdir={run_subdir}",
            f"trainer.max_epochs={stage1_epochs}",
        ]
        if "learning_rate" in training_config:
            stage1_overrides.append(f"optimizer.lr={training_config['learning_rate']}")
        if "batch_size" in training_config:
            stage1_overrides.append(f"loader.batch_size={training_config['batch_size']}")
        runner_args = _as_string_list(runtime_config.get("stage1_overrides"))
        stage1_overrides.extend(runner_args)

        _lwm_write_progress(
            job_dir,
            stage=LEREAL_WORLD_MODEL_STAGE_STAGE1,
            current_step=1,
            total_steps=total_steps,
        )
        stage1_cmd = [
            sys.executable,
            "train_lewm.py",
            "--config-name",
            stage1_config,
            *stage1_overrides,
        ]
        stage1_result = _run_lereal_world_model_command(stage1_cmd, cwd=repo_path, env=env)
        if stage1_result.returncode != 0:
            raise RuntimeError(f"LeRealWorldModel Stage 1 failed with exit code {stage1_result.returncode}")

        world_model_path = _latest_lereal_world_model_checkpoint(repo_path, output_model_name)
        normalizers_path = _normalizers_path_for_world_model(world_model_path, output_model_name)
        current_step = 2
        gc_idm_path: Path | None = None

        if _as_bool(runtime_config.get("run_stage2"), True):
            stage2_config = _as_optional_string(runtime_config.get("stage2_config")) or "gc_idm_topcam"
            stage2_steps = _as_positive_int(runtime_config.get("stage2_steps"), 50000)
            stage2_dataset_overrides = _lwm_dataset_overrides(
                config,
                runtime_config,
                stage=LEREAL_WORLD_MODEL_STAGE_STAGE2,
            )
            stage2_overrides = [
                *stage2_dataset_overrides,
                f"world_model_path={world_model_path}",
                f"steps={stage2_steps}",
            ]
            if "batch_size" in training_config:
                stage2_overrides.append(f"batch_size={training_config['batch_size']}")
            stage2_overrides.extend(_as_string_list(runtime_config.get("stage2_overrides")))
            _lwm_write_progress(
                job_dir,
                stage=LEREAL_WORLD_MODEL_STAGE_STAGE2,
                current_step=current_step,
                total_steps=total_steps,
            )
            stage2_cmd = [
                sys.executable,
                "train_gc_idm.py",
                "--config-name",
                stage2_config,
                *stage2_overrides,
            ]
            stage2_result = _run_lereal_world_model_command(stage2_cmd, cwd=repo_path, env=env)
            if stage2_result.returncode != 0:
                raise RuntimeError(f"LeRealWorldModel Stage 2 failed with exit code {stage2_result.returncode}")
            gc_idm_path = world_model_path.parent / "gc_idm.pt"
            current_step += 1

        if _as_bool(runtime_config.get("export_policy"), True):
            if gc_idm_path is None:
                gc_idm_path = Path(
                    _as_optional_string(runtime_config.get("gc_idm_path")) or world_model_path.parent / "gc_idm.pt"
                )
            if not gc_idm_path.exists():
                raise RuntimeError("LeRealWorldModel export requires gc_idm.pt from Stage 2 or model.config.gc_idm_path.")
            export_dir = job_dir / "final_model"
            export_cmd = [
                sys.executable,
                "export_policy.py",
                "--world_model_path",
                str(world_model_path),
                "--gc_idm_path",
                str(gc_idm_path),
                "--normalizers_path",
                str(normalizers_path),
                "--output_dir",
                str(export_dir),
                "--action_dim",
                str(_as_positive_int(runtime_config.get("action_dim"), 6)),
                "--frameskip",
                str(_as_positive_int(runtime_config.get("frameskip"), 5)),
            ]
            goal_image_path = _as_optional_string(runtime_config.get("goal_image_path"))
            if goal_image_path:
                export_cmd.extend(["--goal_image_path", goal_image_path])
            image_key = _as_optional_string(runtime_config.get("image_key")) or "observation.images.up"
            image_key2 = _as_optional_string(runtime_config.get("image_key2"))
            image_keys = [image_key, *([image_key2] if image_key2 else [])]
            export_cmd.extend(["--image_keys", *image_keys])
            _lwm_write_progress(
                job_dir,
                stage=LEREAL_WORLD_MODEL_STAGE_EXPORT,
                current_step=current_step,
                total_steps=total_steps,
            )
            export_result = _run_lereal_world_model_command(export_cmd, cwd=repo_path, env=env)
            if export_result.returncode != 0:
                raise RuntimeError(f"LeRealWorldModel export failed with exit code {export_result.returncode}")

        summary_path = job_dir / "lereal_world_model_summary.json"
        _write_json(
            summary_path,
            {
                "repo_path": str(repo_path),
                "world_model_path": str(world_model_path),
                "normalizers_path": str(normalizers_path),
                "gc_idm_path": str(gc_idm_path) if gc_idm_path else None,
                "final_model_path": str(job_dir / "final_model"),
            },
        )
        _lwm_write_progress(
            job_dir,
            stage=LEREAL_WORLD_MODEL_STAGE_COMPLETE,
            current_step=total_steps,
            total_steps=total_steps,
            status="completed",
        )
    except Exception as exc:
        _lwm_write_progress(
            job_dir,
            stage=LEREAL_WORLD_MODEL_STAGE_COMPLETE,
            current_step=total_steps,
            total_steps=total_steps,
            status="failed",
            error=str(exc),
        )
        raise


def train_with_lerobot(config: Dict[str, Any], job_dir: Path) -> None:
    """Train using LeRobot library.

    Args:
        config: Training configuration dictionary containing:
            - dataset: Dataset configuration (source, repo_id or local_path)
            - model: Model configuration (architecture, config)
            - training: Training parameters (epochs, batch_size, learning_rate, etc.)
            - tracker: Tracker configuration (type, tracking_uri, etc.)
        job_dir: Directory for saving outputs

    Raises:
        ValueError: If dataset configuration is invalid
        RuntimeError: If training fails
    """
    logger.info("Starting LeRobot training")
    import torch
    from torch.utils.data import DataLoader
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import make_dataset
    from lerobot.policies import make_policy

    # Extract configs
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    tracker_config = config.get("tracker", {})

    # =========================================================================
    # 0. Set seed for reproducibility
    # =========================================================================
    seed = training_config.get("seed", 42)
    set_seed(seed)

    # =========================================================================
    # 0b. Initialize experiment tracker
    # =========================================================================
    tracker = get_tracker(tracker_config)
    job_id = os.environ.get("URDF_STUDIO_JOB_ID", "unknown")

    if tracker:
        run_name = training_config.get("run_name") or f"{model_config.get('architecture', 'policy')}_{job_id}"
        tracker.init_run(
            run_name=run_name,
            config={
                "dataset": dataset_config,
                "model": model_config,
                "training": training_config,
                "seed": seed,
            },
            tags={"job_id": job_id},
        )
        # Log dataset lineage
        dataset_id = dataset_config.get("repo_id") or dataset_config.get("local_path") or "unknown"
        tracker.log_dataset_lineage(
            dataset_id=dataset_id,
            version=dataset_config.get("version", "latest"),
            source=dataset_config.get("source", "huggingface"),
        )
        # Log model config
        tracker.log_model_config(
            architecture=model_config.get("architecture", "act"),
            config=model_config.get("config", {}),
        )

    # =========================================================================
    # 1. Setup configs
    # =========================================================================
    repo_id = dataset_config.get("repo_id")
    if dataset_config.get("source") == "huggingface" and repo_id:
        logger.info(f"Loading dataset from HuggingFace: {repo_id}")
    elif dataset_config.get("source") == "local":
        repo_id = dataset_config.get("local_path")
        logger.info(f"Loading dataset from local path: {repo_id}")
    else:
        raise ValueError(f"Invalid dataset config: {dataset_config}")

    # Determine architecture and device
    architecture = normalize_policy_id(model_config.get("architecture", "act"))
    policy_overrides = model_config.get("config", {})
    device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Creating {architecture} policy on {device_str}")

    # Get policy config class
    PolicyConfigClass = get_policy_config_class(architecture)
    policy_overrides = prepare_policy_overrides(PolicyConfigClass, policy_overrides)

    # Create policy config
    policy_cfg = PolicyConfigClass(
        device=device_str,
        push_to_hub=False,
        repo_id=f"local/{job_dir.name}",
        **policy_overrides,
    )

    # Create dataset config for LeRobot
    ds_cfg = DatasetConfig(repo_id=repo_id)

    requested_step_limit = training_config.get("max_steps") or training_config.get("steps")

    # Create training pipeline config with policy
    train_cfg = TrainPipelineConfig(
        dataset=ds_cfg,
        policy=policy_cfg,
        output_dir=job_dir,
        batch_size=training_config.get("batch_size", 8),
        num_workers=training_config.get("num_workers", 4),
        steps=requested_step_limit or training_config.get("epochs", 100) * 1000,
    )

    # =========================================================================
    # 2. Load dataset
    # =========================================================================
    dataset = make_dataset(train_cfg)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Dataset meta: {dataset.meta}")

    # =========================================================================
    # 3. Create policy
    # =========================================================================
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)

    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # =========================================================================
    # 3. Setup training
    # =========================================================================
    # Optimizer
    learning_rate = training_config.get("learning_rate", 1e-5)
    weight_decay = training_config.get("weight_decay", 1e-4)
    grad_clip_norm = training_config.get("max_grad_norm", training_config.get("grad_clip_norm", 10.0))

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # DataLoader
    batch_size = training_config.get("batch_size", 8)
    num_workers = training_config.get("num_workers", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Calculate total steps
    total_epochs = training_config.get("epochs", 1)
    steps_per_epoch = len(dataloader)
    full_total_steps = total_epochs * steps_per_epoch
    total_steps = (
        min(full_total_steps, int(requested_step_limit))
        if requested_step_limit
        else full_total_steps
    )

    logger.info(f"Training config: {total_epochs} epochs, {steps_per_epoch} steps/epoch, batch_size={batch_size}")
    if requested_step_limit:
        logger.info(f"Max steps enabled: {requested_step_limit} (full epoch plan would be {full_total_steps})")
    logger.info(f"Total optimizer steps: {total_steps}")

    # =========================================================================
    # 4. Training loop
    # =========================================================================
    checkpoint_interval = training_config.get("checkpoint_interval", 10)
    log_interval = training_config.get("log_interval", 100)

    global_step = 0

    for epoch in range(total_epochs):
        if global_step >= total_steps:
            break

        policy.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            if global_step >= total_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass - returns (loss, info_dict)
            loss, info = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

            optimizer.step()

            # Accumulate metrics
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            # Write progress to file (for status polling)
            metrics_dict = {
                "loss": loss_val,
                "learning_rate": learning_rate,
                "epoch_avg_loss": epoch_loss / epoch_steps,
            }
            write_progress(
                job_dir=job_dir,
                current_epoch=epoch,
                total_epochs=total_epochs,
                current_step=global_step,
                total_steps=total_steps,
                metrics=metrics_dict,
            )
            append_metrics(
                job_dir=job_dir,
                step=global_step,
                epoch=epoch,
                metrics=metrics_dict,
            )

            # Log metrics to experiment tracker (real-time)
            if tracker:
                tracker.log_metrics(
                    metrics={
                        "train/loss": loss_val,
                        "train/learning_rate": learning_rate,
                        "train/epoch_avg_loss": epoch_loss / epoch_steps,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # Log periodically
            if (step + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{total_epochs} - Step {step + 1}/{steps_per_epoch} - "
                    f"Loss: {loss_val:.4f} - Avg: {epoch_loss / epoch_steps:.4f}"
                )

        if epoch_steps == 0:
            break

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps
        logger.info(f"Epoch {epoch + 1}/{total_epochs} completed - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = job_dir / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save using LeRobot's format
            policy.save_pretrained(str(checkpoint_dir))

            # Also save optimizer state
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": avg_loss,
                },
                checkpoint_dir / "training_state.pt",
            )
            logger.info(f"Saved checkpoint: {checkpoint_dir}")

    # =========================================================================
    # 5. Save final model
    # =========================================================================
    final_model_dir = job_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Save using LeRobot's format
    policy.save_pretrained(str(final_model_dir))
    logger.info(f"Saved final model: {final_model_dir}")

    # Save training config
    config_path = job_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Final progress
    write_progress(
        job_dir=job_dir,
        current_epoch=total_epochs,
        total_epochs=total_epochs,
        current_step=total_steps,
        total_steps=total_steps,
        metrics={"loss": avg_loss, "learning_rate": 0, "status": "completed"},
    )

    # Log final metrics and finish tracker
    if tracker:
        tracker.log_metrics(
            metrics={
                "train/final_loss": avg_loss,
                "train/total_steps": total_steps,
                "train/total_epochs": total_epochs,
            },
            step=total_steps,
        )
        # Log final model as artifact
        tracker.log_artifact(str(final_model_dir), "model")
        tracker.finish_run("completed")
        logger.info(f"Logged training results to tracker: {tracker.get_run_url()}")

    logger.info("Training completed successfully!")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info(f"Starting training with config: {args.config}")

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Get job directory from environment or config
    job_id = os.environ.get("URDF_STUDIO_JOB_ID", "unknown")
    job_dir = Path(os.environ.get("URDF_STUDIO_JOB_DIR", "."))

    # Ensure job directory exists
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Job ID: {job_id}")
    logger.info(f"Job directory: {job_dir}")

    try:
        architecture = str(config.get("model", {}).get("architecture", "act")).replace("-", "_")
        if architecture == "dreamzero":
            train_with_dreamzero(config, job_dir)
        elif architecture == LEREAL_WORLD_MODEL_ARCHITECTURE:
            train_with_lereal_world_model(config, job_dir)
        else:
            train_with_lerobot(config, job_dir)
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

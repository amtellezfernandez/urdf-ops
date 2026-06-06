"""Training orchestration service.

This service manages training jobs, coordinating between:
- Compute backends (local, Modal, RunPod)
- Experiment trackers (MLflow, W&B)
- Job state management
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.training import (
    ComputeConfig,
    DatasetConfig,
    EpisodeResult,
    EvaluateRequest,
    EvaluateResponse,
    JobStatus,
    ModelArchitecture,
    ModelArchitectureInfo,
    ModelConfig,
    ModelsListResponse,
    TrackerConfig,
    TrainingComputeBackendCapability,
    TrainingComputeBackendsResponse,
    TrainingJobsListResponse,
    TrainingJobSummary,
    TrainingLineage,
    TrainingMetrics,
    TrainingParams,
    TrainingProgress,
    TrainingRuntimeCheckResponse,
    TrainingRuntimeDependencyStatus,
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from backend.robotops import get_compute, get_tracker
from backend.robotops.compute_factory import SECRET_CACHE_FIELD_MARKERS
from backend.robotops.compute_protocol import JobState
from backend.services.job_store import get_job_store, JobRecord
from backend.services.training_launch_contract import (
    build_training_launch_contract,
    dump_internal_model,
    get_enum_value,
)
from backend.services.training_params import (
    TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE,
    TRAINING_CLOUD_COMPUTE_TYPES,
    TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES,
    TRAINING_COMPUTE_BACKEND_LABELS,
    TRAINING_LEROBOT_PYTHON_ENV,
    TRAINING_LEROBOT_TOOLCHAIN_DIRNAME,
    TRAINING_LOCAL_COMPUTE_TYPE,
    TRAINING_OPTIONAL_RUNTIME_MODULES,
    TRAINING_PLANNED_COMPUTE_TYPES,
    TRAINING_REQUIRED_RUNTIME_MODULES,
)

logger = logging.getLogger(__name__)

# Flag to track if we've loaded jobs from the store
_jobs_loaded = False


# ============================================================================
# Model Architecture Definitions
# ============================================================================

MODEL_ARCHITECTURES: Dict[str, ModelArchitectureInfo] = {
    "act": ModelArchitectureInfo(
        name="act",
        display_name="ACT (Action Chunking Transformer)",
        description="Transformer-based policy that predicts action chunks. Good for manipulation tasks.",
        default_config={
            "chunk_size": 100,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "n_heads": 8,
            "n_encoder_layers": 4,
            "n_decoder_layers": 7,
            "dropout": 0.1,
        },
        config_schema={
            "chunk_size": {"type": "int", "min": 1, "max": 1000, "default": 100},
            "hidden_dim": {"type": "int", "options": [256, 512, 768, 1024]},
            "n_encoder_layers": {"type": "int", "min": 1, "max": 12},
            "n_decoder_layers": {"type": "int", "min": 1, "max": 12},
        },
        recommended_for=["manipulation", "bimanual", "precise tasks"],
    ),
    "diffusion_policy": ModelArchitectureInfo(
        name="diffusion_policy",
        display_name="Diffusion Policy",
        description="Denoising diffusion for action prediction. Robust to multimodal demonstrations.",
        default_config={
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "num_inference_steps": 10,
            "noise_scheduler": "ddpm",
        },
        config_schema={
            "horizon": {"type": "int", "min": 1, "max": 64},
            "n_obs_steps": {"type": "int", "min": 1, "max": 16},
            "n_action_steps": {"type": "int", "min": 1, "max": 32},
            "num_inference_steps": {"type": "int", "min": 1, "max": 100},
        },
        recommended_for=["diverse demonstrations", "multimodal behavior"],
    ),
    "dreamzero": ModelArchitectureInfo(
        name="dreamzero",
        display_name="DreamZero",
        description=(
            "World action model runner with URDF-derived action schemas. SO-101 weights can be used "
            "as a preset, while other robots require a matching DreamZero-compatible runner."
        ),
        default_config={
            "base_model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
            "adapter_model_id": "Vizuara/dreamzero-so101-lora",
            "action_horizon": 24,
            "video_frames": 33,
            "image_width": 320,
            "image_height": 176,
            "action_units": "urdf-native",
            "runner_script": "",
            "runner_module": "",
        },
        config_schema={
            "action_horizon": {"type": "int", "min": 1, "max": 128},
            "video_frames": {"type": "int", "min": 1, "max": 128},
            "runner_script": {"type": "string"},
            "runner_module": {"type": "string"},
        },
        recommended_for=["world models", "video-conditioned actions", "URDF action schemas"],
    ),
    "tdmpc": ModelArchitectureInfo(
        name="tdmpc",
        display_name="TD-MPC",
        description="Temporal Difference Model Predictive Control. Good for complex dynamics.",
        default_config={
            "horizon": 5,
            "latent_dim": 512,
            "mlp_dim": 512,
            "num_q": 5,
        },
        config_schema={
            "horizon": {"type": "int", "min": 1, "max": 20},
            "latent_dim": {"type": "int", "options": [256, 512, 1024]},
        },
        recommended_for=["long-horizon tasks", "model-based control"],
    ),
    "vq_bet": ModelArchitectureInfo(
        name="vq_bet",
        display_name="VQ-BeT",
        description="Vector-Quantized Behavior Transformer. Discrete action space learning.",
        default_config={
            "n_clusters": 512,
            "hidden_dim": 384,
            "n_heads": 8,
            "n_layers": 6,
        },
        config_schema={
            "n_clusters": {"type": "int", "options": [256, 512, 1024]},
            "hidden_dim": {"type": "int", "options": [256, 384, 512]},
        },
        recommended_for=["discrete actions", "behavior cloning"],
    ),
}


# ============================================================================
# Job Storage (In-memory cache backed by SQLite persistence)
# ============================================================================

_jobs: Dict[str, Dict[str, Any]] = {}

JOB_INFO_COMPUTE_CONFIG_KEY = "compute_config"
PERSISTED_COMPUTE_CONFIG_KEY = "compute_runtime"
TRAINING_PRIMARY_METRIC_KEYS = frozenset({"loss", "learning_rate", "grad_norm"})


def _is_secret_runtime_config_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return any(marker in normalized for marker in SECRET_CACHE_FIELD_MARKERS)


def _sanitize_runtime_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_runtime_config(value)
    if isinstance(value, list):
        return [_sanitize_runtime_config_value(item) for item in value]
    return value


def _sanitize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in config.items():
        if _is_secret_runtime_config_field(key):
            continue
        sanitized[key] = _sanitize_runtime_config_value(value)
    return sanitized


def _load_persisted_compute_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    compute_config = config.get(PERSISTED_COMPUTE_CONFIG_KEY)
    if isinstance(compute_config, dict):
        return _sanitize_runtime_config(compute_config)
    return None


def _load_persisted_output_dir(config: Dict[str, Any]) -> Optional[str]:
    compute_config = _load_persisted_compute_config(config)
    if compute_config and compute_config.get("output_dir"):
        return str(compute_config["output_dir"])

    training_config = config.get("training")
    if isinstance(training_config, dict) and training_config.get("output_dir"):
        return str(training_config["output_dir"])
    return None


def _fallback_compute_config(job_info: Dict[str, Any]) -> Dict[str, Any]:
    compute_config = {"type": job_info.get("compute_backend", "local")}
    output_dir = job_info.get("output_dir")
    if output_dir:
        compute_config["output_dir"] = output_dir
    return compute_config


def _job_compute_config(job_info: Dict[str, Any]) -> Dict[str, Any]:
    compute_config = job_info.get(JOB_INFO_COMPUTE_CONFIG_KEY)
    if isinstance(compute_config, dict):
        return _sanitize_runtime_config(compute_config)
    return _fallback_compute_config(job_info)


def _numeric_metric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _additional_numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    additional: Dict[str, float] = {}
    for key, value in metrics.items():
        if key in TRAINING_PRIMARY_METRIC_KEYS:
            continue
        numeric_value = _numeric_metric_value(value)
        if numeric_value is not None:
            additional[key] = numeric_value
    return additional


def _job_info_from_record(job_record: JobRecord) -> Dict[str, Any]:
    compute_config = _load_persisted_compute_config(job_record.config)
    output_dir = _load_persisted_output_dir(job_record.config)
    return {
        "compute_job_id": job_record.compute_job_id,
        "compute_backend": job_record.compute_backend,
        "tracker": None,
        "tracker_url": job_record.tracker_url,
        "lineage": None,
        "request": None,
        "status": job_record.status,
        "started_at": job_record.started_at,
        "finished_at": job_record.finished_at,
        "error": job_record.error,
        "config": job_record.config,
        "output_dir": output_dir,
        JOB_INFO_COMPUTE_CONFIG_KEY: compute_config,
    }


async def _load_job_from_store(job_id: str) -> bool:
    try:
        store = get_job_store()
        job_record = await store.get_job(job_id)
    except Exception as e:
        logger.warning(f"Failed to load job {job_id} from store: {e}")
        return False
    if job_record is None:
        return False
    _jobs[job_id] = _job_info_from_record(job_record)
    return True


async def _ensure_jobs_loaded() -> None:
    """Load active jobs from persistent store on first access."""
    global _jobs_loaded

    if _jobs_loaded:
        return

    try:
        store = get_job_store()
        # Load running and pending jobs from database
        active_statuses = [JobStatus.RUNNING, JobStatus.PENDING, JobStatus.QUEUED]

        for status in active_statuses:
            jobs = await store.list_jobs(status=status, limit=100)
            for job_record in jobs:
                if job_record.job_id not in _jobs:
                    _jobs[job_record.job_id] = _job_info_from_record(job_record)
                    logger.info(f"Loaded job from store: {job_record.job_id} (status: {job_record.status})")

        _jobs_loaded = True
        logger.info(f"Loaded {len(_jobs)} active jobs from persistent store")

    except Exception as e:
        logger.warning(f"Failed to load jobs from store: {e}")
        _jobs_loaded = True  # Don't retry on failure


async def _persist_job(job_id: str, job_info: Dict[str, Any]) -> None:
    """Persist job state to the store."""
    try:
        store = get_job_store()

        # Check if job exists
        existing = await store.get_job(job_id)

        if existing is None:
            # Create new job record
            request = job_info.get("request")
            lineage = job_info.get("lineage")

            config = {}
            if request:
                config = {
                    "dataset": _dump_internal_model(request.dataset),
                    "model": _dump_internal_model(request.model),
                    "training": _dump_internal_model(request.training),
                }
                compute_config = job_info.get(JOB_INFO_COMPUTE_CONFIG_KEY)
                if isinstance(compute_config, dict):
                    config[PERSISTED_COMPUTE_CONFIG_KEY] = _sanitize_runtime_config(compute_config)

            await store.create_job(
                job_id=job_id,
                config=config,
                compute_backend=job_info.get("compute_backend", "local"),
                compute_job_id=job_info.get("compute_job_id"),
                run_name=request.training.run_name if request and request.training else None,
                model_architecture=lineage.model_architecture if lineage else None,
                dataset_id=lineage.dataset_id if lineage else None,
            )

            # Update with tracker URL if available
            if job_info.get("tracker_url"):
                await store.update_job(job_id, tracker_url=job_info.get("tracker_url"))

        else:
            # Update existing job
            status = job_info.get("status")
            await store.update_job(
                job_id=job_id,
                status=status,
                error=job_info.get("error"),
                finished_at=job_info.get("finished_at"),
                tracker_url=job_info.get("tracker_url"),
                compute_job_id=job_info.get("compute_job_id"),
            )

    except Exception as e:
        logger.error(f"Failed to persist job {job_id}: {e}")


# ============================================================================
# Service Functions
# ============================================================================


def _hash_config(config: Dict[str, Any]) -> str:
    """Create a hash of configuration for lineage tracking."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def _get_enum_value(value) -> str:
    """Get string value from enum or string."""
    return get_enum_value(value)


def _dump_internal_model(value: Any) -> Dict[str, Any]:
    """Dump Pydantic models with backend/script field names."""
    return dump_internal_model(value)


def _default_lerobot_python_path() -> Path:
    from backend.core.paths import BASE_DIR

    return BASE_DIR / TRAINING_LEROBOT_TOOLCHAIN_DIRNAME / "bin" / "python3"


def _resolve_lerobot_python_path() -> Path:
    configured = os.getenv(TRAINING_LEROBOT_PYTHON_ENV, "").strip()
    return Path(configured) if configured else _default_lerobot_python_path()


def _is_training_compute_backend_enabled(compute_type: str) -> bool:
    return compute_type == TRAINING_LOCAL_COMPUTE_TYPE


def list_training_compute_backends() -> TrainingComputeBackendsResponse:
    backends = [
        TrainingComputeBackendCapability(
            type=TRAINING_LOCAL_COMPUTE_TYPE,
            label=TRAINING_COMPUTE_BACKEND_LABELS[TRAINING_LOCAL_COMPUTE_TYPE],
            enabled=True,
            production_ready=True,
        )
    ]

    for compute_type in (*TRAINING_CLOUD_COMPUTE_TYPES, *TRAINING_PLANNED_COMPUTE_TYPES):
        backends.append(
            TrainingComputeBackendCapability(
                type=compute_type,
                label=TRAINING_COMPUTE_BACKEND_LABELS[compute_type],
                enabled=False,
                production_ready=False,
                reason=TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE,
                missing_capabilities=list(TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES),
            )
        )

    return TrainingComputeBackendsResponse(backends=backends)


def validate_training_compute_backend(compute: ComputeConfig) -> Optional[str]:
    compute_type = _get_enum_value(compute.type)
    if _is_training_compute_backend_enabled(compute_type):
        return None
    return TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE


async def list_training_compute_instances() -> Dict[str, list]:
    from backend.robotops.compute.local_compute import LocalCompute

    instances: Dict[str, list] = {
        TRAINING_LOCAL_COMPUTE_TYPE: await LocalCompute().get_available_instances(),
    }
    for compute_type in (*TRAINING_CLOUD_COMPUTE_TYPES, *TRAINING_PLANNED_COMPUTE_TYPES):
        instances[compute_type] = []
    return instances


def _create_lineage(
    request: TrainingStartRequest,
    started_at: str,
    *,
    dataset_id: str | None = None,
) -> TrainingLineage:
    """Create lineage record for a training job."""
    dataset_id = dataset_id or request.dataset.repo_id or request.dataset.local_path or "unknown"

    return TrainingLineage(
        dataset_source=_get_enum_value(request.dataset.source),
        dataset_id=dataset_id,
        dataset_version=request.dataset.version,
        model_architecture=_get_enum_value(request.model.architecture),
        model_config_hash=_hash_config(request.model.config),
        training_config_hash=_hash_config(_dump_internal_model(request.training)),
        robot_name=request.robot_name,
        urdf_hash=_hash_config({"urdf": request.urdf}) if request.urdf else None,
        started_at=started_at,
    )


async def start_training(request: TrainingStartRequest) -> TrainingStartResponse:
    """Start a new training job.

    Args:
        request: Training configuration

    Returns:
        Response with job ID and status
    """
    job_id = f"train_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now().isoformat()

    compute_block_reason = validate_training_compute_backend(request.compute)
    if compute_block_reason:
        compute_backend = _get_enum_value(request.compute.type)
        logger.warning(
            f"Rejected training job {job_id} for disabled compute backend: {compute_backend}"
        )
        return TrainingStartResponse(
            success=False,
            job_id=job_id,
            message=compute_block_reason,
        )

    try:
        launch_contract = build_training_launch_contract(
            request,
            job_id=job_id,
            lerobot_python_path=_resolve_lerobot_python_path(),
        )
    except ValueError as exc:
        logger.warning(f"Rejected training job {job_id}: {exc}")
        return TrainingStartResponse(
            success=False,
            job_id=job_id,
            message=str(exc),
        )

    runtime_check = check_training_runtime()
    if _get_enum_value(request.compute.type) == TRAINING_LOCAL_COMPUTE_TYPE and not runtime_check.available:
        logger.warning(f"Rejected training job {job_id}: {runtime_check.message}")
        return TrainingStartResponse(
            success=False,
            job_id=job_id,
            message=runtime_check.message,
        )

    # Ensure jobs are loaded from persistent store after cheap request validation.
    await _ensure_jobs_loaded()

    logger.info(f"Starting training job {job_id}")

    try:
        # Create lineage
        lineage = _create_lineage(
            request,
            started_at,
            dataset_id=launch_contract.dataset.canonical_source,
        )

        # Initialize experiment tracker
        tracker_config = launch_contract.tracker_config
        tracker = get_tracker(tracker_config)

        # Start tracking run
        run_name = request.training.run_name or f"{_get_enum_value(request.model.architecture)}_{job_id}"
        tracker.init_run(
            run_name=run_name,
            config={
                "dataset": launch_contract.training_config["dataset"],
                "model": launch_contract.training_config["model"],
                "training": launch_contract.training_config["training"],
                "compute": _dump_internal_model(request.compute),
            },
            tags={
                "job_id": job_id,
                "architecture": _get_enum_value(request.model.architecture),
            },
        )

        # Log lineage
        tracker.log_dataset_lineage(
            dataset_id=lineage.dataset_id,
            version=lineage.dataset_version or "latest",
            source=lineage.dataset_source,
        )
        tracker.log_model_config(
            architecture=lineage.model_architecture,
            config=request.model.config,
        )

        # Initialize compute backend
        compute = get_compute(launch_contract.compute_config)

        # Prepare training config for script
        training_config = launch_contract.training_config

        # Launch training job
        from backend.core.paths import SCRIPTS_DIR

        script_path = SCRIPTS_DIR / "train_policy.py"

        # For local compute, we'll use subprocess
        # For cloud, this would submit to Modal/RunPod
        compute_job_id = await compute.launch(
            script=str(script_path),
            config=training_config,
            env={
                "URDF_STUDIO_JOB_ID": job_id,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(SCRIPTS_DIR.parent.parent),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
            },
        )

        # Store job info in memory
        compute_backend = _get_enum_value(request.compute.type)
        _jobs[job_id] = {
            "compute_job_id": compute_job_id,
            "compute_backend": compute_backend,
            "tracker": tracker,
            "tracker_url": tracker.get_run_url(),
            "lineage": lineage,
            "request": request,
            "status": JobStatus.RUNNING,
            "started_at": started_at,
            "output_dir": str(launch_contract.output_dir),
            JOB_INFO_COMPUTE_CONFIG_KEY: _sanitize_runtime_config(launch_contract.compute_config),
        }

        # Persist to database
        await _persist_job(job_id, _jobs[job_id])

        return TrainingStartResponse(
            success=True,
            job_id=job_id,
            message=f"Training started on {compute_backend}",
            tracker_url=tracker.get_run_url(),
            lineage=lineage,
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}")

        _jobs[job_id] = {
            "status": JobStatus.FAILED,
            "error": str(e),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
        }

        # Persist failed job
        await _persist_job(job_id, _jobs[job_id])

        return TrainingStartResponse(
            success=False,
            job_id=job_id,
            message=f"Failed to start training: {e}",
        )


async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """Get status of a training job.

    Args:
        job_id: Job ID to check

    Returns:
        Current status of the job
    """
    # Ensure jobs are loaded from persistent store
    await _ensure_jobs_loaded()

    if job_id not in _jobs:
        await _load_job_from_store(job_id)

    if job_id not in _jobs:
        return TrainingStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error="Job not found",
            compute_backend="unknown",
        )

    job_info = _jobs[job_id]

    # Startup failures have no backend job to query.
    if job_info.get("status") == JobStatus.FAILED and not job_info.get("compute_job_id"):
        return TrainingStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=job_info.get("error"),
            compute_backend=job_info.get("compute_backend", "unknown"),
        )

    # Get status from compute backend
    try:
        request = job_info.get("request")
        output_dir = "./outputs"
        if job_info.get("output_dir"):
            output_dir = job_info["output_dir"]
        elif request and hasattr(request, "training"):
            output_dir = request.training.output_dir

        if output_dir and not job_info.get("output_dir"):
            job_info["output_dir"] = output_dir
        compute = get_compute(_job_compute_config(job_info))

        compute_status = await compute.status(job_info["compute_job_id"])

        # Map compute status to job status
        status_map = {
            JobState.PENDING: JobStatus.PENDING,
            JobState.QUEUED: JobStatus.QUEUED,
            JobState.RUNNING: JobStatus.RUNNING,
            JobState.COMPLETED: JobStatus.COMPLETED,
            JobState.FAILED: JobStatus.FAILED,
            JobState.CANCELLED: JobStatus.CANCELLED,
        }
        status = status_map.get(compute_status.state, JobStatus.RUNNING)

        # Update stored status
        old_status = job_info.get("status")
        job_info["status"] = status
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job_info["finished_at"] = datetime.now().isoformat()
            # Finish tracker run
            tracker = job_info.get("tracker")
            if tracker:
                tracker.finish_run(_get_enum_value(status))

        # Persist status change to database
        if old_status != status:
            await _persist_job(job_id, job_info)

        # Build progress
        progress = None
        if compute_status.progress:
            progress = TrainingProgress(
                current_epoch=compute_status.progress.current_epoch,
                total_epochs=compute_status.progress.total_epochs,
                current_step=compute_status.progress.current_step,
                total_steps=compute_status.progress.total_steps,
                epoch_progress=compute_status.progress.epoch_progress,
                overall_progress=compute_status.progress.overall_progress,
            )

        # Build metrics
        metrics = None
        if compute_status.metrics:
            metrics = TrainingMetrics(
                loss=_numeric_metric_value(compute_status.metrics.get("loss")),
                learning_rate=_numeric_metric_value(
                    compute_status.metrics.get("learning_rate")
                ),
                grad_norm=_numeric_metric_value(compute_status.metrics.get("grad_norm")),
                additional=_additional_numeric_metrics(compute_status.metrics),
            )

        return TrainingStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress,
            metrics=metrics,
            tracker_url=job_info.get("tracker_url"),
            lineage=job_info.get("lineage"),
            error=compute_status.error_message,
            logs_tail=compute_status.logs_tail,
            compute_backend=job_info.get("compute_backend", "local"),
            cost_estimate_usd=compute_status.cost_estimate_usd,
        )

    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return TrainingStatusResponse(
            job_id=job_id,
            status=job_info.get("status", JobStatus.RUNNING),
            error=str(e),
            compute_backend=job_info.get("compute_backend", "unknown"),
        )


async def cancel_training(job_id: str, reason: Optional[str] = None) -> bool:
    """Cancel a running training job.

    Args:
        job_id: Job ID to cancel
        reason: Optional cancellation reason

    Returns:
        True if cancelled successfully
    """
    await _ensure_jobs_loaded()
    if job_id not in _jobs:
        await _load_job_from_store(job_id)
    if job_id not in _jobs:
        return False

    job_info = _jobs[job_id]
    compute_job_id = job_info.get("compute_job_id")
    if not compute_job_id:
        return False

    try:
        compute = get_compute(_job_compute_config(job_info))

        cancelled = await compute.cancel(compute_job_id)

        if cancelled:
            job_info["status"] = JobStatus.CANCELLED
            job_info["finished_at"] = datetime.now().isoformat()
            job_info["cancel_reason"] = reason

            # Finish tracker run
            tracker = job_info.get("tracker")
            if tracker:
                tracker.finish_run("cancelled")

            # Persist cancellation
            await _persist_job(job_id, job_info)

            logger.info(f"Cancelled job {job_id}: {reason}")

        return cancelled

    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return False


async def list_jobs(
    limit: int = 50,
    status_filter: Optional[JobStatus] = None,
) -> TrainingJobsListResponse:
    """List training jobs.

    Args:
        limit: Maximum jobs to return
        status_filter: Optional status filter

    Returns:
        List of job summaries
    """
    # First, load from persistent store
    try:
        store = get_job_store()
        stored_jobs = await store.list_jobs(status=status_filter, limit=limit)

        jobs = []
        for job_record in stored_jobs:
            jobs.append(
                TrainingJobSummary(
                    job_id=job_record.job_id,
                    status=job_record.status,
                    run_name=job_record.run_name,
                    model_architecture=job_record.model_architecture or "unknown",
                    dataset_id=job_record.dataset_id or "unknown",
                    started_at=job_record.started_at or "",
                    finished_at=job_record.finished_at,
                    compute_backend=job_record.compute_backend,
                )
            )

        return TrainingJobsListResponse(
            jobs=jobs[:limit],
            total=len(jobs),
        )

    except Exception as e:
        logger.warning(f"Failed to list jobs from store: {e}, falling back to memory")

    # Fallback to in-memory jobs
    jobs = []

    for job_id, job_info in _jobs.items():
        status = job_info.get("status", JobStatus.PENDING)

        if status_filter and status != status_filter:
            continue

        lineage = job_info.get("lineage")
        request = job_info.get("request")

        run_name = None
        if request and hasattr(request, "training") and request.training:
            run_name = request.training.run_name

        jobs.append(
            TrainingJobSummary(
                job_id=job_id,
                status=status,
                run_name=run_name,
                model_architecture=lineage.model_architecture if lineage else "unknown",
                dataset_id=lineage.dataset_id if lineage else "unknown",
                started_at=job_info.get("started_at", ""),
                finished_at=job_info.get("finished_at"),
                compute_backend=job_info.get("compute_backend", "local"),
            )
        )

    # Sort by start time (newest first)
    jobs.sort(key=lambda j: j.started_at, reverse=True)

    return TrainingJobsListResponse(
        jobs=jobs[:limit],
        total=len(jobs),
    )


def list_models() -> ModelsListResponse:
    """List available model architectures.

    Returns:
        List of model architecture info
    """
    return ModelsListResponse(models=list(MODEL_ARCHITECTURES.values()))


def get_model_info(architecture: str) -> Optional[ModelArchitectureInfo]:
    """Get info for a specific model architecture.

    Args:
        architecture: Architecture name

    Returns:
        Architecture info or None
    """
    return MODEL_ARCHITECTURES.get(architecture)


def _dependency_version(module_name: str, python_path: Path) -> Optional[str]:
    import subprocess

    try:
        output = subprocess.check_output(
            [
                str(python_path),
                "-c",
                (
                    "from importlib.metadata import PackageNotFoundError, version\n"
                    f"try:\n    print(version({module_name!r}))\n"
                    "except PackageNotFoundError:\n    print('')\n"
                ),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return output or None
    except Exception:
        return None


def _resolve_runtime_dependency(
    module_name: str,
    *,
    python_path: Path,
    required: bool,
) -> TrainingRuntimeDependencyStatus:
    import subprocess

    try:
        subprocess.check_call(
            [
                str(python_path),
                "-c",
                f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module_name!r}) else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        installed = True
    except Exception as exc:
        return TrainingRuntimeDependencyStatus(
            name=module_name,
            required=required,
            installed=False,
            error=str(exc) or "module not found",
        )

    return TrainingRuntimeDependencyStatus(
        name=module_name,
        required=required,
        installed=installed,
        version=_dependency_version(module_name, python_path),
    )


def check_training_runtime() -> TrainingRuntimeCheckResponse:
    """Check whether the local backend process can run LeRobot training."""
    import subprocess

    python_path = _resolve_lerobot_python_path()
    if not python_path.exists():
        return TrainingRuntimeCheckResponse(
            available=False,
            python_executable=str(python_path),
            dependencies=[
                TrainingRuntimeDependencyStatus(
                    name=module_name,
                    required=True,
                    installed=False,
                    error=f"Managed LeRobot Python not found. Run npm run setup or set {TRAINING_LEROBOT_PYTHON_ENV}.",
                )
                for module_name in TRAINING_REQUIRED_RUNTIME_MODULES
            ],
            message=f"Managed LeRobot Python not found at {python_path}. Run npm run setup.",
        )

    dependencies = [
        *[
            _resolve_runtime_dependency(module_name, python_path=python_path, required=True)
            for module_name in TRAINING_REQUIRED_RUNTIME_MODULES
        ],
        *[
            _resolve_runtime_dependency(module_name, python_path=python_path, required=False)
            for module_name in TRAINING_OPTIONAL_RUNTIME_MODULES
        ],
    ]
    missing_required = [
        dependency.name
        for dependency in dependencies
        if dependency.required and not dependency.installed
    ]

    cuda_available = None
    torch_status = next((dependency for dependency in dependencies if dependency.name == "torch"), None)
    if torch_status and torch_status.installed:
        try:
            output = subprocess.check_output(
                [str(python_path), "-c", "import torch; print(int(torch.cuda.is_available()))"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            cuda_available = output == "1"
        except Exception:
            cuda_available = None

    if missing_required:
        message = f"Missing required local training modules in {python_path}: {', '.join(missing_required)}. Run npm run setup."
    else:
        message = "Local training runtime dependencies are available."

    return TrainingRuntimeCheckResponse(
        available=len(missing_required) == 0,
        python_executable=str(python_path),
        dependencies=dependencies,
        cuda_available=cuda_available,
        message=message,
    )


async def evaluate_policy(request: EvaluateRequest) -> EvaluateResponse:
    """Run policy evaluation.

    Args:
        request: Evaluation configuration

    Returns:
        Evaluation results with action sequences
    """
    import asyncio
    import subprocess

    from backend.core.paths import SCRIPTS_DIR

    script_path = SCRIPTS_DIR / "eval_policy.py"

    # Build command
    python_path = _resolve_lerobot_python_path()
    cmd = [
        str(python_path),
        str(script_path),
        "--checkpoint",
        request.checkpoint_path,
        "--num-episodes",
        str(request.num_episodes),
        "--max-steps",
        str(request.max_steps),
    ]

    if request.initial_state:
        cmd.extend(["--initial-state", json.dumps(request.initial_state)])

    if request.urdf:
        # Write URDF to temp file
        import tempfile

        urdf_file = Path(tempfile.mktemp(suffix=".urdf"))
        urdf_file.write_text(request.urdf)
        cmd.extend(["--urdf", str(urdf_file)])

    logger.info(f"Running evaluation: {' '.join(cmd)}")

    try:
        # Run evaluation script
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env={
                    **os.environ,
                    "PYTHONPATH": str(SCRIPTS_DIR.parent.parent),
                },
            ),
        )

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return EvaluateResponse(
                success=False,
                error=f"Evaluation script failed: {result.stderr}",
            )

        # Parse output
        output = json.loads(result.stdout)

        # Convert to response model
        episodes = [
            EpisodeResult(
                episode_index=ep["episode_index"],
                actions=ep["actions"],
                observations=ep.get("observations"),
                timestamps=ep.get("timestamps"),
            )
            for ep in output.get("episodes", [])
        ]

        return EvaluateResponse(
            success=output.get("success", False),
            episodes=episodes,
            metrics=output.get("metrics", {}),
            error=output.get("error"),
        )

    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out")
        return EvaluateResponse(
            success=False,
            error="Evaluation timed out after 5 minutes",
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation output: {e}")
        return EvaluateResponse(
            success=False,
            error=f"Invalid evaluation output: {e}",
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluateResponse(
            success=False,
            error=str(e),
        )

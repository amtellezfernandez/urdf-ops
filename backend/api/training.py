"""Training API endpoints.

This module provides REST API endpoints for:
- Starting training jobs
- Monitoring training progress
- Cancelling jobs
- Listing jobs and models
- Policy evaluation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.models.training import (
    EvaluateRequest,
    EvaluateResponse,
    JobStatus,
    ModelArchitectureInfo,
    ModelsListResponse,
    TrainingCancelRequest,
    TrainingComputeBackendsResponse,
    TrainingJobsListResponse,
    TrainingRuntimeCheckResponse,
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from backend.core.simulator_security import require_simulator_operator_access
from backend.services import training as training_service

router = APIRouter(
    prefix="/training",
    tags=["training"],
    dependencies=[Depends(require_simulator_operator_access)],
)


# ============================================================================
# Training Job Endpoints
# ============================================================================


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(request: TrainingStartRequest) -> TrainingStartResponse:
    """Start a new training job.

    This endpoint:
    1. Validates the configuration
    2. Initializes experiment tracking (MLflow/W&B)
    3. Launches the training job on the specified compute backend
    4. Returns immediately with a job ID for tracking

    The actual training runs asynchronously. Use GET /training/status/{job_id}
    to monitor progress.
    """
    return await training_service.start_training(request)


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """Get the status of a training job.

    Returns:
    - Current status (pending, running, completed, failed, cancelled)
    - Training progress (epoch, step)
    - Current metrics (loss, learning rate)
    - Experiment tracker URL
    - Training lineage
    - Recent log output
    - Cost estimate (for cloud compute)
    """
    return await training_service.get_training_status(job_id)


@router.post("/cancel/{job_id}")
async def cancel_training(
    job_id: str,
    request: Optional[TrainingCancelRequest] = None,
) -> dict:
    """Cancel a running training job.

    This will:
    1. Stop the training process
    2. Save the current checkpoint (if possible)
    3. Mark the experiment as cancelled in the tracker
    """
    reason = request.reason if request else None
    success = await training_service.cancel_training(job_id, reason)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or already finished",
        )

    return {"success": True, "message": f"Job {job_id} cancelled"}


@router.get("/jobs", response_model=TrainingJobsListResponse)
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum jobs to return"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
) -> TrainingJobsListResponse:
    """List training jobs.

    Returns a list of training job summaries, sorted by start time (newest first).
    """
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: {[s.value for s in JobStatus]}",
            )

    return await training_service.list_jobs(limit=limit, status_filter=status_filter)


@router.get("/logs/{job_id}")
async def get_training_logs(job_id: str) -> dict:
    """Return the latest available training log tail for a job."""
    status = await training_service.get_training_status(job_id)
    return {
        "jobId": job_id,
        "logs": status.logs_tail or "",
        "hasMore": False,
    }


@router.get("/metrics/{job_id}")
async def get_training_metrics(job_id: str) -> dict:
    """Return the latest available training metrics in chart history format."""
    status = await training_service.get_training_status(job_id)
    if not status.metrics:
        return {
            "jobId": job_id,
            "metrics": {},
            "lastStep": 0,
            "lastEpoch": 0,
        }

    progress = status.progress
    step = progress.current_step if progress else 0
    epoch = progress.current_epoch if progress else 0
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    metric_values = {
        "loss": status.metrics.loss,
        "learning_rate": status.metrics.learning_rate,
        "grad_norm": status.metrics.grad_norm,
        **status.metrics.additional,
    }
    metrics = {
        name: [
            {
                "step": step,
                "epoch": epoch,
                "timestamp": timestamp_ms,
                "value": value,
            }
        ]
        for name, value in metric_values.items()
        if value is not None
    }
    return {
        "jobId": job_id,
        "metrics": metrics,
        "lastStep": step,
        "lastEpoch": epoch,
    }


# ============================================================================
# Model Architecture Endpoints
# ============================================================================


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """List available model architectures.

    Returns information about supported policy architectures including:
    - Name and description
    - Default configuration
    - Configuration schema
    - Recommended use cases
    """
    return training_service.list_models()


@router.get("/models/{architecture}", response_model=ModelArchitectureInfo)
async def get_model_info(architecture: str) -> ModelArchitectureInfo:
    """Get detailed information about a model architecture."""
    info = training_service.get_model_info(architecture)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown architecture: {architecture}",
        )

    return info


# ============================================================================
# Evaluation Endpoints
# ============================================================================


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_policy(request: EvaluateRequest) -> EvaluateResponse:
    """Run policy evaluation and return action sequences.

    This endpoint:
    1. Loads the trained checkpoint
    2. Runs inference for the specified number of episodes
    3. Returns action sequences that can be replayed in the 3D viewer

    The response includes:
    - Action sequences for each episode
    - Evaluation metrics (success rate, etc.)
    """
    return await training_service.evaluate_policy(request)


# ============================================================================
# Compute Info Endpoints
# ============================================================================


@router.get("/compute/backends", response_model=TrainingComputeBackendsResponse)
async def list_compute_backends() -> TrainingComputeBackendsResponse:
    """List production readiness for training compute backends."""
    return training_service.list_training_compute_backends()


@router.get("/compute/instances")
async def list_compute_instances() -> dict:
    """List compute instances that this build can use without cloud side effects."""
    instances = await training_service.list_training_compute_instances()
    return {"instances": instances}


@router.get("/runtime-check", response_model=TrainingRuntimeCheckResponse)
async def check_training_runtime() -> TrainingRuntimeCheckResponse:
    """Check whether the local backend can launch LeRobot training."""
    return training_service.check_training_runtime()

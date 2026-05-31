"""Keypoint observation API for URDF repair and SysID consumers."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.core.simulator_security import require_simulator_operator_access
from backend.models.keypoint_observations import (
    KeypointObservationBatch,
    KeypointObservationValidationResponse,
)
from backend.services.keypoint_observations import validate_keypoint_observation_batch
from backend.services.keypoint_observations_params import URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION

router = APIRouter(
    prefix="/keypoint-observations",
    tags=["keypoint-observations"],
    dependencies=[Depends(require_simulator_operator_access)],
)


@router.get("/schema")
async def get_keypoint_observation_schema() -> dict[str, str]:
    """Return the current keypoint observation contract identifier."""

    return {"schema_version": URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION}


@router.post("/validate", response_model=KeypointObservationValidationResponse)
async def validate_keypoint_observations(
    batch: KeypointObservationBatch,
) -> KeypointObservationValidationResponse:
    """Validate and summarize task-space keypoint observations."""

    return validate_keypoint_observation_batch(batch)

"""Pydantic contract for task-space keypoint observations produced by URDF Ops."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.services.keypoint_observations_params import (
    KEYPOINT_OBSERVATION_DEFAULT_FRAME_ID,
    KEYPOINT_OBSERVATION_MAX_BATCH_FRAMES,
    KEYPOINT_OBSERVATION_MAX_CAMERA_NAME_LENGTH,
    KEYPOINT_OBSERVATION_MAX_CONFIDENCE,
    KEYPOINT_OBSERVATION_MAX_DATASET_REPO_LENGTH,
    KEYPOINT_OBSERVATION_MAX_DATASET_REVISION_LENGTH,
    KEYPOINT_OBSERVATION_MAX_FRAME_ID_LENGTH,
    KEYPOINT_OBSERVATION_MAX_LABEL_LENGTH,
    KEYPOINT_OBSERVATION_MAX_LINK_NAME_LENGTH,
    KEYPOINT_OBSERVATION_MAX_ROBOT_ID_LENGTH,
    KEYPOINT_OBSERVATION_MIN_CONFIDENCE,
    URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION,
)


Float2 = tuple[float, float]
Float3 = tuple[float, float, float]


class KeypointObservationSample(BaseModel):
    """One detected or reconstructed keypoint for a single dataset frame."""

    label: str = Field(..., min_length=1, max_length=KEYPOINT_OBSERVATION_MAX_LABEL_LENGTH)
    confidence: float = Field(
        ...,
        ge=KEYPOINT_OBSERVATION_MIN_CONFIDENCE,
        le=KEYPOINT_OBSERVATION_MAX_CONFIDENCE,
    )
    frame_id: str = Field(
        default=KEYPOINT_OBSERVATION_DEFAULT_FRAME_ID,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_FRAME_ID_LENGTH,
    )
    link_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_LINK_NAME_LENGTH,
    )
    pixel_xy: Float2 | None = None
    position_xyz_m: Float3 | None = None

    @model_validator(mode="after")
    def require_observation_coordinates(self) -> "KeypointObservationSample":
        if self.pixel_xy is None and self.position_xyz_m is None:
            raise ValueError("Keypoint sample requires pixel_xy or position_xyz_m.")
        return self


class KeypointFrameObservation(BaseModel):
    """All keypoints observed for one episode frame and camera."""

    episode_index: int = Field(..., ge=0)
    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0.0)
    camera_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_CAMERA_NAME_LENGTH,
    )
    keypoints: tuple[KeypointObservationSample, ...] = Field(..., min_length=1)


class KeypointObservationBatch(BaseModel):
    """Stable URDF Ops output consumed by downstream calibration systems."""

    schema_version: Literal[URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION] = (
        URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION
    )
    source_dataset_repo: str | None = Field(
        default=None,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_DATASET_REPO_LENGTH,
    )
    source_dataset_revision: str | None = Field(
        default=None,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_DATASET_REVISION_LENGTH,
    )
    robot_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_ROBOT_ID_LENGTH,
    )
    observations: tuple[KeypointFrameObservation, ...] = Field(
        ...,
        min_length=1,
        max_length=KEYPOINT_OBSERVATION_MAX_BATCH_FRAMES,
    )


class KeypointObservationSummary(BaseModel):
    """Coverage summary for a keypoint observation batch."""

    schema_version: str = Field(..., min_length=1)
    observation_count: int = Field(..., ge=0)
    keypoint_count: int = Field(..., ge=0)
    position_keypoint_count: int = Field(..., ge=0)
    pixel_keypoint_count: int = Field(..., ge=0)
    episode_count: int = Field(..., ge=0)
    frame_count: int = Field(..., ge=0)
    labels: tuple[str, ...] = Field(default_factory=tuple)
    link_names: tuple[str, ...] = Field(default_factory=tuple)
    camera_names: tuple[str, ...] = Field(default_factory=tuple)
    ready_for_geometry_repair: bool = False
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class KeypointObservationValidationResponse(BaseModel):
    """Validation response returned by the keypoint observation API."""

    valid: bool
    summary: KeypointObservationSummary

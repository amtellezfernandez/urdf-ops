"""Validation and coverage summaries for URDF Ops keypoint observations."""

from __future__ import annotations

from backend.models.keypoint_observations import (
    KeypointFrameObservation,
    KeypointObservationBatch,
    KeypointObservationSummary,
    KeypointObservationValidationResponse,
)
from backend.services.keypoint_observations_params import (
    KEYPOINT_OBSERVATION_READY_MIN_POSITION_KEYPOINTS,
    URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION,
)


def _sorted_tuple(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values))


def _frame_key(observation: KeypointFrameObservation) -> tuple[int, int, str | None]:
    return (observation.episode_index, observation.frame_index, observation.camera_name)


def summarize_keypoint_observation_batch(batch: KeypointObservationBatch) -> KeypointObservationSummary:
    """Compute deterministic coverage statistics for a validated keypoint batch."""

    labels: set[str] = set()
    link_names: set[str] = set()
    camera_names: set[str] = set()
    frame_keys: set[tuple[int, int, str | None]] = set()
    episodes: set[int] = set()
    keypoint_count = 0
    position_keypoint_count = 0
    pixel_keypoint_count = 0

    for observation in batch.observations:
        frame_keys.add(_frame_key(observation))
        episodes.add(observation.episode_index)
        if observation.camera_name:
            camera_names.add(observation.camera_name)
        for keypoint in observation.keypoints:
            keypoint_count += 1
            labels.add(keypoint.label)
            if keypoint.link_name:
                link_names.add(keypoint.link_name)
            if keypoint.position_xyz_m is not None:
                position_keypoint_count += 1
            if keypoint.pixel_xy is not None:
                pixel_keypoint_count += 1

    ready_for_geometry_repair = position_keypoint_count >= KEYPOINT_OBSERVATION_READY_MIN_POSITION_KEYPOINTS
    warnings = []
    if not ready_for_geometry_repair:
        warnings.append("Geometry repair requires at least one keypoint with position_xyz_m.")
    if not link_names:
        warnings.append("No keypoints include link_name; downstream URDF link attribution will be unavailable.")

    return KeypointObservationSummary(
        schema_version=URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION,
        observation_count=len(batch.observations),
        keypoint_count=keypoint_count,
        position_keypoint_count=position_keypoint_count,
        pixel_keypoint_count=pixel_keypoint_count,
        episode_count=len(episodes),
        frame_count=len(frame_keys),
        labels=_sorted_tuple(labels),
        link_names=_sorted_tuple(link_names),
        camera_names=_sorted_tuple(camera_names),
        ready_for_geometry_repair=ready_for_geometry_repair,
        warnings=tuple(warnings),
    )


def validate_keypoint_observation_batch(batch: KeypointObservationBatch) -> KeypointObservationValidationResponse:
    """Return the stable validation envelope consumed by Studio and tests."""

    return KeypointObservationValidationResponse(
        valid=True,
        summary=summarize_keypoint_observation_batch(batch),
    )

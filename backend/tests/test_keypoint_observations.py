from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from backend.api.keypoint_observations import (
    get_keypoint_observation_schema,
    validate_keypoint_observations,
)
from backend.app import create_app
from backend.models.keypoint_observations import (
    KeypointObservationBatch,
    KeypointObservationSample,
)
from backend.services.keypoint_observations import validate_keypoint_observation_batch
from backend.services.keypoint_observations_params import URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION

TEST_CAMERA_NAME = "wrist_camera"
TEST_DATASET_REPO = "lerobot/svla_so100_pickplace"
TEST_ROBOT_ID = "so100"
TEST_EPISODE_INDEX = 2
TEST_FRAME_INDEX = 7
TEST_CONFIDENCE = 0.92
TEST_PIXEL_XY = (320.0, 241.5)
TEST_POSITION_XYZ_M = (0.12, -0.03, 0.44)
TEST_KEYPOINT_LABEL = "moving_jaw_tip"
TEST_LINK_NAME = "moving_jaw_so101_v1_link"


def _keypoint_batch() -> KeypointObservationBatch:
    return KeypointObservationBatch.model_validate(
        {
            "schema_version": URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION,
            "source_dataset_repo": TEST_DATASET_REPO,
            "robot_id": TEST_ROBOT_ID,
            "observations": [
                {
                    "episode_index": TEST_EPISODE_INDEX,
                    "frame_index": TEST_FRAME_INDEX,
                    "camera_name": TEST_CAMERA_NAME,
                    "keypoints": [
                        {
                            "label": TEST_KEYPOINT_LABEL,
                            "confidence": TEST_CONFIDENCE,
                            "pixel_xy": TEST_PIXEL_XY,
                            "position_xyz_m": TEST_POSITION_XYZ_M,
                            "link_name": TEST_LINK_NAME,
                        }
                    ],
                }
            ],
        }
    )


def test_keypoint_sample_requires_pixel_or_position() -> None:
    with pytest.raises(ValidationError, match="pixel_xy or position_xyz_m"):
        KeypointObservationSample(label=TEST_KEYPOINT_LABEL, confidence=TEST_CONFIDENCE)


def test_validate_keypoint_observation_batch_summarizes_geometry_readiness() -> None:
    response = validate_keypoint_observation_batch(_keypoint_batch())

    assert response.valid is True
    assert response.summary.schema_version == URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION
    assert response.summary.observation_count == 1
    assert response.summary.keypoint_count == 1
    assert response.summary.position_keypoint_count == 1
    assert response.summary.pixel_keypoint_count == 1
    assert response.summary.labels == (TEST_KEYPOINT_LABEL,)
    assert response.summary.link_names == (TEST_LINK_NAME,)
    assert response.summary.camera_names == (TEST_CAMERA_NAME,)
    assert response.summary.ready_for_geometry_repair is True
    assert response.summary.warnings == ()


def test_keypoint_observation_router_is_registered_on_backend_app() -> None:
    app = create_app()
    registered_paths = {route.path for route in app.routes}

    assert "/keypoint-observations/schema" in registered_paths
    assert "/keypoint-observations/validate" in registered_paths


def test_keypoint_observation_api_functions_return_contract_payloads() -> None:
    schema_payload = asyncio.run(get_keypoint_observation_schema())
    response = asyncio.run(validate_keypoint_observations(_keypoint_batch()))

    assert schema_payload["schema_version"] == URDF_OPS_KEYPOINT_OBSERVATION_SCHEMA_VERSION
    assert response.valid is True
    assert response.summary.ready_for_geometry_repair is True
    assert response.summary.link_names == (TEST_LINK_NAME,)

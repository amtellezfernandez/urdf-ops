from __future__ import annotations

import asyncio
import json

import pytest

from backend.robotops.compute.cloud_control import (
    TrainingCloudRuntimeConfig,
    build_training_cloud_run_create_request,
)
from backend.robotops.compute.macrodata_compute import MacrodataCompute
from backend.robotops.compute.cloud_control_params import (
    TRAINING_CLOUD_CONTROL_REDACTION_PLACEHOLDER,
)
from backend.services.training_params import TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE


TEST_HEARTBEAT_SECONDS = 30
TEST_NUM_WORKERS = 2
TEST_CPUS_PER_WORKER = 4
TEST_MEMORY_MB_PER_WORKER = 8192
TEST_GPUS_PER_WORKER = 1
TEST_SECRET_VALUE = "secret-token-value"
TEST_ENV_VALUE = "plain-model-name"
TEST_SHA256_LENGTH = 64


def test_training_cloud_request_separates_env_and_redacts_secrets() -> None:
    request = build_training_cloud_run_create_request(
        provider="macrodata",
        name="training-job",
        body={"token": TEST_SECRET_VALUE, "dataset": "lerobot/pusht"},
        runtime=TrainingCloudRuntimeConfig(
            num_workers=TEST_NUM_WORKERS,
            heartbeat_interval_seconds=TEST_HEARTBEAT_SECONDS,
            cpus_per_worker=TEST_CPUS_PER_WORKER,
            mem_mb_per_worker=TEST_MEMORY_MB_PER_WORKER,
            gpus_per_worker=TEST_GPUS_PER_WORKER,
            gpu_type="h100",
        ),
        manifest={"script": {"text": f"TOKEN={TEST_SECRET_VALUE}"}},
        secrets={"TOKEN": TEST_SECRET_VALUE},
        env={"MODEL_NAME": TEST_ENV_VALUE},
    )

    payload = request.to_dict()
    serialized_payload = json.dumps(payload, sort_keys=True)

    assert payload["executor"] == {"type": "urdfops-cloud", "provider": "macrodata"}
    assert payload["secrets"] == {"TOKEN": TEST_SECRET_VALUE}
    assert payload["env"] == {"MODEL_NAME": TEST_ENV_VALUE}
    assert payload["runtime"]["gpu_type"] == "h100"
    assert payload["payload"]["body"]["token"] == TRAINING_CLOUD_CONTROL_REDACTION_PLACEHOLDER
    assert TEST_SECRET_VALUE not in payload["payload"]["body"].values()
    assert TRAINING_CLOUD_CONTROL_REDACTION_PLACEHOLDER in payload["manifest"]["script"]["text"]
    assert len(payload["payload"]["sha256"]) == TEST_SHA256_LENGTH
    assert TEST_ENV_VALUE in serialized_payload


def test_training_cloud_request_rejects_gpu_type_without_gpu_count() -> None:
    with pytest.raises(ValueError, match="gpus_per_worker is required"):
        TrainingCloudRuntimeConfig(
            num_workers=TEST_NUM_WORKERS,
            heartbeat_interval_seconds=TEST_HEARTBEAT_SECONDS,
            gpu_type="h100",
        )


def test_training_cloud_request_rejects_overlapping_secret_and_env_keys() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        build_training_cloud_run_create_request(
            provider="macrodata",
            name="training-job",
            body={"dataset": "lerobot/pusht"},
            runtime=TrainingCloudRuntimeConfig(
                num_workers=TEST_NUM_WORKERS,
                heartbeat_interval_seconds=TEST_HEARTBEAT_SECONDS,
            ),
            secrets={"API_KEY": "secret"},
            env={"API_KEY": "plain"},
        )


def test_macrodata_compute_backend_fails_closed() -> None:
    compute = MacrodataCompute()

    with pytest.raises(RuntimeError, match="Cloud training runners are disabled"):
        asyncio.run(compute.launch(script="train.py", config={}))

    status = asyncio.run(compute.status("job-1"))

    assert status.compute_backend == "macrodata"
    assert status.error_message == TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE

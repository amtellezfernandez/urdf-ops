from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import backend.services.training as training_service
import backend.api.training as training_api
from backend.models.training import (
    JobStatus,
    ModelArchitecture,
    TrainingStartRequest,
    TrainingStartResponse,
)
from backend.app import create_app
from backend.services.training_policy_compat import normalize_policy_id, prepare_policy_overrides
from backend.services.training import (
    _dump_internal_model,
    check_training_runtime,
    get_model_info,
    get_job_artifacts,
    list_models,
    list_training_compute_backends,
    preflight_training,
    start_training,
    validate_training_compute_backend,
)
from backend.services.training_launch_contract import build_training_launch_contract
from backend.services.training_params import (
    TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE,
    TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES,
)


TEST_BATCH_SIZE = 4
TEST_MAX_STEPS = 2
TEST_DATASET_REPO_ID = "lerobot/pusht"
TEST_JOB_ID = "train_1234"
TEST_COMPUTE_JOB_ID = "local_1234"
TEST_LOCAL_OUTPUT_DIR = "/tmp/urdf-studio-test-output"
TEST_LOCAL_PYTHON_PATH = "/tmp/urdf-studio-test-python"
TEST_JOB_TIMESTAMP = "2026-04-12T00:00:00"
TEST_TRAINING_LOSS = 0.125
TEST_TRAINING_LEARNING_RATE = 0.001
TEST_SO101_URDF = """<?xml version="1.0"?>
<robot name="so101_new_calib">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="upper_arm"/>
  <link name="forearm"/>
  <link name="wrist"/>
  <link name="gripper"/>
  <link name="jaw"/>
  <joint name="gripper" type="revolute">
    <parent link="gripper"/>
    <child link="jaw"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.2" upper="1.7" velocity="1.0" effort="1.0"/>
  </joint>
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist"/>
    <child link="gripper"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="1.0"/>
  </joint>
  <joint name="wrist_flex" type="revolute">
    <parent link="forearm"/>
    <child link="wrist"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.74" upper="1.74" velocity="1.0" effort="1.0"/>
  </joint>
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm"/>
    <child link="forearm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.74" upper="1.74" velocity="1.0" effort="1.0"/>
  </joint>
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder"/>
    <child link="upper_arm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.74" upper="1.74" velocity="1.0" effort="1.0"/>
  </joint>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base"/>
    <child link="shoulder"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="1.0"/>
  </joint>
</robot>
"""


def _runpod_cloud_start_request() -> TrainingStartRequest:
    return TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "runpod", "apiKey": "secret"},
        }
    )


def test_training_start_request_accepts_ui_camel_case_payloads() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"source": "huggingface", "repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.DIFFUSION_POLICY.value},
            "training": {"batchSize": TEST_BATCH_SIZE, "maxSteps": TEST_MAX_STEPS},
        }
    )

    assert request.dataset.repo_id == TEST_DATASET_REPO_ID
    assert request.model.architecture == ModelArchitecture.DIFFUSION_POLICY.value
    assert request.training.batch_size == TEST_BATCH_SIZE
    assert request.training.max_steps == TEST_MAX_STEPS


def test_training_responses_serialize_to_ui_camel_case() -> None:
    payload = TrainingStartResponse(
        success=True,
        job_id=TEST_JOB_ID,
        message="started",
    ).model_dump()

    assert payload["jobId"] == TEST_JOB_ID
    assert "job_id" not in payload
    assert payload["trackerUrl"] is None


def test_training_service_internal_dump_keeps_script_snake_case() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "training": {"batchSize": TEST_BATCH_SIZE, "maxSteps": TEST_MAX_STEPS},
        }
    )

    dataset_config = _dump_internal_model(request.dataset)
    training_config = _dump_internal_model(request.training)

    assert dataset_config["repo_id"] == TEST_DATASET_REPO_ID
    assert "repoId" not in dataset_config
    assert training_config["batch_size"] == TEST_BATCH_SIZE
    assert training_config["max_steps"] == TEST_MAX_STEPS
    assert "batchSize" not in training_config
    assert "maxSteps" not in training_config


def test_training_model_catalog_uses_lerobot_architecture_names() -> None:
    models = list_models()
    architecture_names = {model.name for model in models.models}

    assert {
        "act",
        "diffusion_policy",
        "dreamzero",
        "lereal_world_model",
        "tdmpc",
        "vq_bet",
    }.issubset(architecture_names)
    assert get_model_info("diffusion_policy") is not None
    assert get_model_info("dreamzero") is not None
    assert get_model_info("lereal_world_model") is not None
    assert get_model_info("vq_bet") is not None


def test_training_script_normalizes_ui_policy_ids_for_lerobot() -> None:
    assert normalize_policy_id("act") == "act"
    assert normalize_policy_id("diffusion_policy") == "diffusion"
    assert normalize_policy_id("diffusion-policy") == "diffusion"
    assert normalize_policy_id("vq_bet") == "vqbet"
    assert normalize_policy_id("vq-bet") == "vqbet"


def test_training_script_filters_policy_overrides_for_installed_lerobot_config() -> None:
    class FakePolicyConfig:
        def __init__(
            self,
            *,
            device: str,
            push_to_hub: bool,
            repo_id: str,
            dim_model: int = 256,
            dropout: float = 0.1,
        ) -> None:
            pass

    overrides = prepare_policy_overrides(
        FakePolicyConfig,
        {
            "hidden_dim": 512,
            "dropout": 0.2,
            "unsupported": True,
        },
    )

    assert overrides == {"dim_model": 512, "dropout": 0.2}


def test_training_router_is_registered_on_backend_app() -> None:
    app = create_app()
    registered_paths = {route.path for route in app.routes}

    assert "/training/models" in registered_paths
    assert "/training/start" in registered_paths
    assert "/training/preflight" in registered_paths
    assert "/training/logs/{job_id}" in registered_paths
    assert "/training/metrics/{job_id}" in registered_paths
    assert "/training/artifacts/{job_id}" in registered_paths
    assert "/training/runtime-check" in registered_paths
    assert "/training/compute/backends" in registered_paths


def test_training_runtime_check_reports_required_dependencies() -> None:
    result = check_training_runtime()
    dependencies_by_name = {dependency.name: dependency for dependency in result.dependencies}

    assert "torch" in dependencies_by_name
    assert dependencies_by_name["torch"].required is True
    assert "lerobot" in dependencies_by_name
    assert dependencies_by_name["lerobot"].required is True
    assert result.message


def test_training_compute_backends_fail_closed_for_cloud_providers() -> None:
    response = list_training_compute_backends()
    backends_by_type = {backend.type: backend for backend in response.backends}

    assert backends_by_type["local"].enabled is True
    assert backends_by_type["local"].production_ready is True
    assert backends_by_type["ssh"].enabled is True
    assert backends_by_type["ssh"].production_ready is True
    assert backends_by_type["modal"].enabled is False
    assert backends_by_type["runpod"].enabled is False
    assert backends_by_type["macrodata"].enabled is False
    assert backends_by_type["aws"].enabled is False
    assert backends_by_type["runpod"].reason == TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE
    assert backends_by_type["macrodata"].missing_capabilities == list(
        TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES
    )


def test_training_compute_validation_rejects_cloud_credentials() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "macrodata", "apiKey": "secret"},
        }
    )

    assert validate_training_compute_backend(request.compute) == TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE


def test_training_compute_validation_requires_ssh_target() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "ssh", "sshUser": "ubuntu"},
        }
    )

    assert validate_training_compute_backend(request.compute) == "Remote Docker training requires an SSH host and user."


def test_training_preflight_blocks_disabled_cloud_backend() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "runpod", "apiKey": "secret"},
        }
    )

    result = asyncio.run(preflight_training(request))

    assert result.ready is False
    assert result.compute_backend == "runpod"
    assert result.checks[0].status == "fail"


def test_training_preflight_reports_missing_ssh_target_without_network() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "ssh", "sshUser": "ubuntu"},
        }
    )

    result = asyncio.run(preflight_training(request))

    assert result.ready is False
    assert result.compute_backend == "ssh"
    assert result.checks[0].name == "ssh_config"


def test_training_launch_contract_maps_ssh_compute_config() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {
                "type": "ssh",
                "sshHost": "203.0.113.10",
                "sshUser": "ubuntu",
                "sshPort": 2222,
                "sshKeyPath": "~/.ssh/robotops",
                "remoteOutputDir": "/scratch/robotops",
                "dockerImage": "urdf-ops:training",
                "dockerArgs": "--shm-size 8g",
                "device": "cuda",
            },
        }
    )

    contract = build_training_launch_contract(
        request,
        job_id=TEST_JOB_ID,
        lerobot_python_path=Path("/tmp/python"),
    )

    assert contract.compute_config["type"] == "ssh"
    assert contract.compute_config["host"] == "203.0.113.10"
    assert contract.compute_config["user"] == "ubuntu"
    assert contract.compute_config["port"] == 2222
    assert contract.compute_config["key_path"] == "~/.ssh/robotops"
    assert contract.compute_config["output_dir"] == "/scratch/robotops"
    assert contract.compute_config["docker_image"] == "urdf-ops:training"
    assert contract.compute_config["docker_args"] == "--shm-size 8g"
    assert contract.compute_config["use_gpu"] is True


def test_training_artifacts_list_uses_compute_backend(tmp_path: Path) -> None:
    compute_job_id = "local_artifacts"
    job_dir = tmp_path / compute_job_id
    job_dir.mkdir()
    artifact_path = job_dir / "final_model.safetensors"
    artifact_path.write_text("model")

    training_service._jobs[TEST_JOB_ID] = {
        "compute_job_id": compute_job_id,
        "compute_backend": "local",
        "output_dir": str(tmp_path),
        "status": JobStatus.COMPLETED,
    }

    try:
        result = asyncio.run(get_job_artifacts(TEST_JOB_ID))
    finally:
        training_service._jobs.pop(TEST_JOB_ID, None)

    assert result["jobId"] == TEST_JOB_ID
    assert result["total"] == 1
    assert result["artifacts"][0]["name"] == "final_model.safetensors"


def test_training_metrics_endpoint_uses_history_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def noop_ensure_jobs_loaded() -> None:
        return None

    compute_job_id = "local_metrics"
    job_dir = tmp_path / compute_job_id
    job_dir.mkdir()
    (job_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "epoch": 0,
                        "timestamp": "2026-06-27T10:00:00",
                        "loss": 0.5,
                        "learning_rate": 0.001,
                        "status": "running",
                    }
                ),
                json.dumps(
                    {
                        "step": 2,
                        "epoch": 0,
                        "timestamp": "2026-06-27T10:00:01",
                        "loss": 0.25,
                        "learning_rate": 0.001,
                        "status": "running",
                    }
                ),
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_service, "_ensure_jobs_loaded", noop_ensure_jobs_loaded)
    training_service._jobs[TEST_JOB_ID] = {
        "compute_job_id": compute_job_id,
        "compute_backend": "local",
        "output_dir": str(tmp_path),
        "status": JobStatus.RUNNING,
    }

    try:
        result = asyncio.run(training_api.get_training_metrics(TEST_JOB_ID))
    finally:
        training_service._jobs.pop(TEST_JOB_ID, None)

    assert result["jobId"] == TEST_JOB_ID
    assert result["lastStep"] == 2
    assert result["lastEpoch"] == 0
    assert [point["value"] for point in result["metrics"]["loss"]] == [0.5, 0.25]
    assert [point["step"] for point in result["metrics"]["learning_rate"]] == [1, 2]
    assert "status" not in result["metrics"]


def test_training_launch_contract_rejects_missing_dataset_repo() -> None:
    request = TrainingStartRequest.model_validate(
        {"model": {"architecture": ModelArchitecture.ACT.value}}
    )

    with pytest.raises(ValueError, match="repo ID"):
        build_training_launch_contract(
            request,
            job_id=TEST_JOB_ID,
            lerobot_python_path=Path("/tmp/python"),
        )


def test_training_launch_contract_rejects_output_path_escape() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "training": {"outputDir": "../host-output"},
            "model": {"architecture": ModelArchitecture.ACT.value},
        }
    )

    with pytest.raises(ValueError, match="Training output directory"):
        build_training_launch_contract(
            request,
            job_id=TEST_JOB_ID,
            lerobot_python_path=Path("/tmp/python"),
        )


def test_training_launch_contract_reuses_shared_local_dataset_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.services.dataset_source_contract as dataset_source_contract

    dataset_root = tmp_path / "dataset"
    monkeypatch.setattr(
        dataset_source_contract,
        "DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (tmp_path,),
    )
    monkeypatch.setattr(
        dataset_source_contract,
        "is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"source": "local", "localPath": str(dataset_root)},
            "model": {"architecture": ModelArchitecture.ACT.value},
        }
    )

    contract = build_training_launch_contract(
        request,
        job_id=TEST_JOB_ID,
        lerobot_python_path=Path("/tmp/python"),
    )

    assert contract.training_config["dataset"]["local_path"] == str(
        dataset_root.resolve(strict=False)
    )


def test_training_launch_contract_embeds_urdf_action_schema_for_dreamzero() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.DREAMZERO.value},
            "urdf": TEST_SO101_URDF,
            "robotName": "so101",
        }
    )

    contract = build_training_launch_contract(
        request,
        job_id=TEST_JOB_ID,
        lerobot_python_path=Path("/tmp/python"),
    )

    action_schema = contract.training_config["model"]["config"]["action_schema"]
    assert action_schema["robot_name"] == "so101"
    assert action_schema["action_dim"] == 6
    assert action_schema["joint_names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert contract.training_config["embodiment"] == action_schema


def test_training_launch_contract_requires_dreamzero_action_schema() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.DREAMZERO.value},
        }
    )

    with pytest.raises(ValueError, match="DreamZero training requires"):
        build_training_launch_contract(
            request,
            job_id=TEST_JOB_ID,
            lerobot_python_path=Path("/tmp/python"),
        )


def test_training_launch_contract_embeds_optional_action_schema_for_lereal_world_model() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.LEREAL_WORLD_MODEL.value},
            "urdf": TEST_SO101_URDF,
            "robotName": "so101",
        }
    )

    contract = build_training_launch_contract(
        request,
        job_id=TEST_JOB_ID,
        lerobot_python_path=Path("/tmp/python"),
    )

    action_schema = contract.training_config["model"]["config"]["action_schema"]
    assert action_schema["robot_name"] == "so101"
    assert contract.training_config["model"]["config"]["action_dim"] == 6
    assert contract.training_config["model"]["config"]["action_joint_names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def test_start_training_rejects_cloud_before_job_side_effects() -> None:
    request = TrainingStartRequest.model_validate(
        {
            "dataset": {"repoId": TEST_DATASET_REPO_ID},
            "model": {"architecture": ModelArchitecture.ACT.value},
            "compute": {"type": "modal", "apiKey": "secret"},
        }
    )

    response = asyncio.run(start_training(request))

    assert response.success is False
    assert response.message == TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE
    assert response.job_id.startswith("train_")


def test_training_compute_api_reports_cloud_disabled_and_rejects_start() -> None:
    backends_payload = asyncio.run(training_api.list_compute_backends()).model_dump()
    start_payload = asyncio.run(
        training_api.start_training(_runpod_cloud_start_request())
    ).model_dump()

    assert backends_payload["backends"]
    backends_by_type = {backend["type"]: backend for backend in backends_payload["backends"]}
    assert backends_by_type["runpod"]["enabled"] is False
    assert backends_by_type["macrodata"]["enabled"] is False
    assert backends_by_type["macrodata"]["missingCapabilities"] == list(
        TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES
    )
    assert backends_by_type["aws"]["enabled"] is False
    assert start_payload["success"] is False
    assert start_payload["message"] == TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE


def test_training_compute_api_payloads_serialize_for_ui() -> None:
    start_payload = asyncio.run(
        training_api.start_training(_runpod_cloud_start_request())
    ).model_dump()

    assert "jobId" in start_payload
    assert "job_id" not in start_payload


def test_training_status_uses_launch_compute_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.services.training as training_service
    from backend.robotops.compute_protocol import JobState, JobStatus as ComputeJobStatus

    captured_configs = []

    class FakeCompute:
        async def status(self, job_id: str) -> ComputeJobStatus:
            return ComputeJobStatus(
                job_id=job_id,
                state=JobState.COMPLETED,
                compute_backend="local",
                metrics={
                    "loss": TEST_TRAINING_LOSS,
                    "learning_rate": TEST_TRAINING_LEARNING_RATE,
                    "status": "completed",
                },
            )

    async def noop_ensure_jobs_loaded() -> None:
        return None

    async def noop_persist_job(job_id: str, job_info: dict) -> None:
        return None

    def fake_get_compute(config: dict) -> FakeCompute:
        captured_configs.append(config)
        return FakeCompute()

    monkeypatch.setattr(training_service, "_ensure_jobs_loaded", noop_ensure_jobs_loaded)
    monkeypatch.setattr(training_service, "_persist_job", noop_persist_job)
    monkeypatch.setattr(training_service, "get_compute", fake_get_compute)
    training_service._jobs[TEST_JOB_ID] = {
        "compute_job_id": TEST_COMPUTE_JOB_ID,
        "compute_backend": "local",
        "status": JobStatus.FAILED,
        "output_dir": TEST_LOCAL_OUTPUT_DIR,
        "compute_config": {
            "type": "local",
            "output_dir": TEST_LOCAL_OUTPUT_DIR,
            "python_path": TEST_LOCAL_PYTHON_PATH,
        },
    }

    try:
        response = asyncio.run(training_service.get_training_status(TEST_JOB_ID))
    finally:
        training_service._jobs.pop(TEST_JOB_ID, None)

    assert response.status == JobStatus.COMPLETED
    assert response.metrics is not None
    assert response.metrics.loss == TEST_TRAINING_LOSS
    assert response.metrics.learning_rate == TEST_TRAINING_LEARNING_RATE
    assert "status" not in response.metrics.additional
    assert captured_configs == [
        {
            "type": "local",
            "output_dir": TEST_LOCAL_OUTPUT_DIR,
            "python_path": TEST_LOCAL_PYTHON_PATH,
        }
    ]



def test_training_job_record_uses_legacy_training_output_dir() -> None:
    import backend.services.training as training_service
    from backend.services.job_store import JobRecord

    job_info = training_service._job_info_from_record(
        JobRecord(
            job_id=TEST_JOB_ID,
            status=JobStatus.RUNNING,
            config={"training": {"output_dir": TEST_LOCAL_OUTPUT_DIR}},
            created_at=TEST_JOB_TIMESTAMP,
            updated_at=TEST_JOB_TIMESTAMP,
            compute_backend="local",
            compute_job_id=TEST_COMPUTE_JOB_ID,
        )
    )

    assert job_info["output_dir"] == TEST_LOCAL_OUTPUT_DIR



def test_cancel_training_loads_persisted_job(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.services.training as training_service
    from backend.services.job_store import JobRecord

    captured_configs = []
    cancelled_job_ids = []

    class FakeCompute:
        async def cancel(self, job_id: str) -> bool:
            cancelled_job_ids.append(job_id)
            return True

    class FakeStore:
        async def get_job(self, job_id: str) -> JobRecord:
            return JobRecord(
                job_id=job_id,
                status=JobStatus.RUNNING,
                config={
                    "training": {"output_dir": TEST_LOCAL_OUTPUT_DIR},
                    "compute_runtime": {
                        "type": "local",
                        "output_dir": TEST_LOCAL_OUTPUT_DIR,
                        "python_path": TEST_LOCAL_PYTHON_PATH,
                    },
                },
                created_at=TEST_JOB_TIMESTAMP,
                updated_at=TEST_JOB_TIMESTAMP,
                compute_backend="local",
                compute_job_id=TEST_COMPUTE_JOB_ID,
            )

    async def noop_ensure_jobs_loaded() -> None:
        return None

    async def noop_persist_job(job_id: str, job_info: dict) -> None:
        return None

    def fake_get_compute(config: dict) -> FakeCompute:
        captured_configs.append(config)
        return FakeCompute()

    monkeypatch.setattr(training_service, "_ensure_jobs_loaded", noop_ensure_jobs_loaded)
    monkeypatch.setattr(training_service, "_persist_job", noop_persist_job)
    monkeypatch.setattr(training_service, "get_job_store", lambda: FakeStore())
    monkeypatch.setattr(training_service, "get_compute", fake_get_compute)
    training_service._jobs.pop(TEST_JOB_ID, None)

    try:
        cancelled = asyncio.run(training_service.cancel_training(TEST_JOB_ID))
    finally:
        training_service._jobs.pop(TEST_JOB_ID, None)

    assert cancelled is True
    assert cancelled_job_ids == [TEST_COMPUTE_JOB_ID]
    assert captured_configs == [
        {
            "type": "local",
            "output_dir": TEST_LOCAL_OUTPUT_DIR,
            "python_path": TEST_LOCAL_PYTHON_PATH,
        }
    ]

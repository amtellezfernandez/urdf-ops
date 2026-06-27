import asyncio
import json
import subprocess
import sys
from pathlib import Path

from backend.robotops.compute_factory import (
    _COMPUTE_INSTANCES,
    _build_compute_cache_key,
    get_compute,
    ComputeConfig,
)
from backend.robotops.compute.local_compute import LocalCompute
from backend.robotops.compute.ssh_compute import SSHDockerCompute
from backend.robotops.compute_protocol import JobState


TEST_RESTORED_COMPUTE_JOB_ID = "local_restored"
TEST_CURRENT_EPOCH = 1
TEST_TOTAL_EPOCHS = 1
TEST_CURRENT_STEP = 4
TEST_TOTAL_STEPS = 4
TEST_LOSS = 0.25
TEST_SLEEP_SECONDS = 30


class RecordingSSHCompute(SSHDockerCompute):
    def __init__(self) -> None:
        super().__init__(host="203.0.113.10", user="ubuntu", use_gpu=False)
        self.commands: list[str] = []

    async def _run_ssh(
        self,
        command: str,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(["ssh"], 0, "", "")

    async def _scp_to_remote(
        self,
        local_path: Path,
        remote_path: str,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["scp"], 0, "", "")


def test_local_compute_cache_is_scoped_by_output_dir(tmp_path: Path) -> None:
    _COMPUTE_INSTANCES.clear()
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    first = get_compute({"type": "local", "output_dir": str(first_output_dir)})
    second = get_compute({"type": "local", "output_dir": str(second_output_dir)})

    assert first is not second
    assert first._output_dir == first_output_dir
    assert second._output_dir == second_output_dir


def test_compute_cache_reuses_matching_config(tmp_path: Path) -> None:
    _COMPUTE_INSTANCES.clear()
    output_dir = tmp_path / "same"

    first = get_compute({"type": "local", "output_dir": str(output_dir)})
    second = get_compute({"type": "local", "output_dir": str(output_dir)})

    assert first is second


def test_compute_cache_key_does_not_store_plaintext_api_key() -> None:
    api_key = "super-secret-token"
    access_token = "provider-access-token"

    cache_key = _build_compute_cache_key(
        ComputeConfig(
            type="runpod",
            api_key=api_key,
            nested_credentials={"access_token": access_token},
            provider_secrets=[{"password": "nested-password"}],
        ),
        "runpod",
    )

    assert api_key not in cache_key
    assert access_token not in cache_key
    assert "nested-password" not in cache_key
    assert "api_key_sha256" in cache_key
    assert "access_token_sha256" in cache_key
    assert "password_sha256" in cache_key


def test_get_compute_creates_ssh_backend() -> None:
    _COMPUTE_INSTANCES.clear()

    compute = get_compute(
        {
            "type": "ssh",
            "ssh_host": "203.0.113.10",
            "ssh_user": "ubuntu",
            "ssh_port": 2222,
            "remote_output_dir": "/scratch/robotops",
            "docker_image": "urdf-ops:training",
        }
    )

    assert isinstance(compute, SSHDockerCompute)
    assert compute.host == "203.0.113.10"
    assert compute.user == "ubuntu"
    assert compute.port == 2222
    assert compute.output_dir == "/scratch/robotops"


def test_ssh_compute_launch_sets_container_pythonpath() -> None:
    compute = RecordingSSHCompute()

    asyncio.run(compute.launch("ignored.py", {"device": "cpu"}))

    docker_commands = [command for command in compute.commands if "docker run -d" in command]
    assert len(docker_commands) == 1
    assert "-e PYTHONPATH=/app" in docker_commands[0]
    assert "python /app/backend/scripts/train_policy.py" in docker_commands[0]


def test_local_compute_status_restores_completed_job_from_disk(tmp_path: Path) -> None:
    job_dir = tmp_path / TEST_RESTORED_COMPUTE_JOB_ID
    job_dir.mkdir()
    (job_dir / "stdout.log").write_text("training complete\n")
    (job_dir / "progress.json").write_text(
        json.dumps(
            {
                "current_epoch": TEST_CURRENT_EPOCH,
                "total_epochs": TEST_TOTAL_EPOCHS,
                "current_step": TEST_CURRENT_STEP,
                "total_steps": TEST_TOTAL_STEPS,
                "metrics": {"status": "completed", "loss": TEST_LOSS},
            }
        )
    )

    status = asyncio.run(LocalCompute(output_dir=str(tmp_path)).status(TEST_RESTORED_COMPUTE_JOB_ID))

    assert status.state == JobState.COMPLETED
    assert status.progress is not None
    assert status.progress.current_step == TEST_CURRENT_STEP
    assert status.metrics["loss"] == TEST_LOSS
    assert status.logs_tail == "training complete\n"



def test_local_compute_status_fails_stale_nonterminal_job(tmp_path: Path) -> None:
    job_dir = tmp_path / TEST_RESTORED_COMPUTE_JOB_ID
    job_dir.mkdir()

    status = asyncio.run(LocalCompute(output_dir=str(tmp_path)).status(TEST_RESTORED_COMPUTE_JOB_ID))

    assert status.state == JobState.FAILED
    assert status.error_message == "Local training process was not recoverable after backend restart"


def test_local_compute_logs_and_artifacts_restore_from_disk(tmp_path: Path) -> None:
    job_dir = tmp_path / TEST_RESTORED_COMPUTE_JOB_ID
    job_dir.mkdir()
    (job_dir / "stdout.log").write_text("training complete\n")
    (job_dir / "model.pt").write_text("checkpoint")
    (job_dir / "metrics.jsonl").write_text('{"step": 1, "loss": 0.5}\n')
    (job_dir / "progress.json").write_text(
        json.dumps(
            {
                "current_epoch": TEST_CURRENT_EPOCH,
                "total_epochs": TEST_TOTAL_EPOCHS,
                "current_step": TEST_CURRENT_STEP,
                "total_steps": TEST_TOTAL_STEPS,
                "metrics": {"status": "completed", "loss": TEST_LOSS},
            }
        )
    )

    compute = LocalCompute(output_dir=str(tmp_path))
    logs = asyncio.run(_collect_logs(compute, TEST_RESTORED_COMPUTE_JOB_ID))
    artifacts = asyncio.run(compute.list_artifacts(TEST_RESTORED_COMPUTE_JOB_ID))

    assert logs == ["training complete\n"]
    assert any(artifact.name == "model.pt" for artifact in artifacts)
    assert any(
        artifact.name == "metrics.jsonl" and artifact.artifact_type == "metrics"
        for artifact in artifacts
    )


def test_local_compute_read_job_file_rejects_path_escape(tmp_path: Path) -> None:
    job_dir = tmp_path / TEST_RESTORED_COMPUTE_JOB_ID
    job_dir.mkdir()
    (job_dir / "metrics.jsonl").write_text('{"step": 1, "loss": 0.5}\n')
    (tmp_path / "outside.txt").write_text("secret")

    compute = LocalCompute(output_dir=str(tmp_path))

    assert asyncio.run(compute.read_job_file(TEST_RESTORED_COMPUTE_JOB_ID, "metrics.jsonl")) == (
        '{"step": 1, "loss": 0.5}\n'
    )
    assert asyncio.run(compute.read_job_file(TEST_RESTORED_COMPUTE_JOB_ID, "../outside.txt")) is None


async def _collect_logs(compute: LocalCompute, job_id: str) -> list[str]:
    return [line async for line in compute.logs(job_id)]


def test_local_compute_cancel_restores_live_manifest_job(tmp_path: Path) -> None:
    script_path = tmp_path / "sleep_train.py"
    script_path.write_text(
        "import time\n"
        f"time.sleep({TEST_SLEEP_SECONDS})\n"
    )
    launcher = LocalCompute(output_dir=str(tmp_path), python_path=sys.executable)
    job_id = asyncio.run(launcher.launch(str(script_path), config={}))

    restored = LocalCompute(output_dir=str(tmp_path), python_path=sys.executable)
    try:
        assert asyncio.run(restored.cancel(job_id)) is True
        status = asyncio.run(restored.status(job_id))
        reloaded_status = asyncio.run(
            LocalCompute(output_dir=str(tmp_path), python_path=sys.executable).status(job_id)
        )
        assert status.state == JobState.CANCELLED
        assert reloaded_status.state == JobState.CANCELLED
    finally:
        asyncio.run(launcher.cancel(job_id))

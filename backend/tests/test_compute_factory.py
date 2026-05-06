import asyncio
import json
import sys
from pathlib import Path

from backend.robotops.compute_factory import (
    _COMPUTE_INSTANCES,
    _build_compute_cache_key,
    get_compute,
    ComputeConfig,
)
from backend.robotops.compute.local_compute import LocalCompute
from backend.robotops.compute_protocol import JobState


TEST_RESTORED_COMPUTE_JOB_ID = "local_restored"
TEST_CURRENT_EPOCH = 1
TEST_TOTAL_EPOCHS = 1
TEST_CURRENT_STEP = 4
TEST_TOTAL_STEPS = 4
TEST_LOSS = 0.25
TEST_SLEEP_SECONDS = 30


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

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.scripts import train_policy


def test_lereal_world_model_runner_builds_local_dataset_stage1_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_path = tmp_path / "LeRealWorldModel"
    (repo_path / "lewm_robot" / "data").mkdir(parents=True)
    (repo_path / "train_lewm.py").write_text("", encoding="utf-8")
    dataset_path = tmp_path / "studio-dataset"
    dataset_path.mkdir()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    captured_commands: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured_commands.append(cmd)
        if "train_lewm.py" in cmd:
            checkpoint_dir = repo_path / "checkpoints" / "local-test"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "lewm_so100_topcam_epoch_1_object.ckpt").write_bytes(b"ckpt")
            (checkpoint_dir / "lewm_so100_topcam_normalizers.pt").write_bytes(b"norm")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(train_policy, "_run_lereal_world_model_command", fake_run)

    train_policy.train_with_lereal_world_model(
        {
            "dataset": {
                "source": "local",
                "local_path": str(dataset_path),
                "episodes": [0, 2],
            },
            "model": {
                "architecture": "lereal_world_model",
                "config": {
                    "repo_path": str(repo_path),
                    "auto_install": False,
                    "run_stage2": False,
                    "export_policy": False,
                    "stage1_config": "lewm_so100_topcam",
                    "stage1_epochs": 1,
                    "image_key": "observation.images.up",
                    "frameskip": 5,
                    "action_dim": 6,
                },
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-4,
                "run_name": "local-test",
            },
        },
        job_dir,
    )

    stage1_command = next(command for command in captured_commands if "train_lewm.py" in command)
    assert stage1_command[:4] == [
        sys.executable,
        "train_lewm.py",
        "--config-name",
        "lewm_so100_topcam",
    ]
    assert "data.dataset._target_=lewm_robot.data.lerobot_adapter.LeRobotWMDataset" in stage1_command
    assert f"data.dataset.root={dataset_path.resolve(strict=False)}" in stage1_command
    assert "data.dataset.repo_id=local/urdf-ops-dataset" in stage1_command
    assert "data.dataset.image_key=observation.images.up" in stage1_command
    assert "data.dataset.image_key2=null" in stage1_command
    assert "data.dataset.episodes=[0,2]" in stage1_command
    assert "data.dataset.frameskip=5" in stage1_command
    assert "wm.action_dim=6" in stage1_command
    assert "trainer.max_epochs=1" in stage1_command
    assert "loader.batch_size=2" in stage1_command
    assert "optimizer.lr=0.0001" in stage1_command

    summary = json.loads((job_dir / "lereal_world_model_summary.json").read_text())
    assert summary["world_model_path"].endswith("lewm_so100_topcam_epoch_1_object.ckpt")

    progress = json.loads((job_dir / "progress.json").read_text())
    assert progress["metrics"]["status"] == "completed"
    assert progress["metrics"]["lereal_world_model_stage"] == "complete"

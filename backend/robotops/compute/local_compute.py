"""Local compute backend for running training on the local machine.

This backend runs training jobs as subprocesses on the local machine,
using available GPUs. Suitable for development and single-machine training.

Usage:
    compute = LocalCompute(output_dir="./outputs")
    job_id = await compute.launch("train.py", config={...})
    status = await compute.status(job_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.robotops.compute_protocol import (
    ComputeBackend,
    JobArtifact,
    JobProgress,
    JobState,
    JobStatus,
)

logger = logging.getLogger(__name__)

LOCAL_JOB_ID_HEX_LENGTH = 8
LOCAL_COMPUTE_CANCEL_GRACE_SECONDS = 5
LOCAL_COMPUTE_CANCEL_POLL_SECONDS = 0.1
LOCAL_COMPUTE_LOG_TAIL_LINES = 20
LOCAL_COMPUTE_LOG_FOLLOW_POLL_SECONDS = 0.1
LOCAL_PROGRESS_FILE_NAME = "progress.json"
LOCAL_STDOUT_LOG_NAME = "stdout.log"
LOCAL_STDERR_LOG_NAME = "stderr.log"
LOCAL_CONFIG_FILE_NAME = "config.json"
LOCAL_JOB_MANIFEST_FILE_NAME = "local-job-manifest.json"
LOCAL_JOB_MANIFEST_SCHEMA_VERSION = 1
LOCAL_PROGRESS_COMPLETED_STATUS = "completed"
LOCAL_PROGRESS_FAILED_STATUS = "failed"
LOCAL_PROGRESS_DEFAULT_COUNT = 0
LOCAL_SUCCESS_EXIT_CODE = 0
LOCAL_PROCESS_CHECK_SIGNAL = 0
LOCAL_PROCESS_STAT_START_TIME_TAIL_INDEX = 19
LOCAL_PROCESS_CMDLINE_SEPARATOR = b"\0"
LOCAL_PROCESS_NOT_RECOVERABLE_MESSAGE = (
    "Local training process was not recoverable after backend restart"
)
LOCAL_JOB_NOT_FOUND_MESSAGE = "Job not found"
LOCAL_NO_LOGS_MESSAGE = "No logs available"
LOCAL_ARTIFACT_PATTERNS = {
    "*.pt": "checkpoint",
    "*.pth": "checkpoint",
    "*.safetensors": "checkpoint",
    "*.ckpt": "checkpoint",
    "*.mp4": "video",
    "*.log": "log",
    "*.json": "config",
}


class LocalCompute:
    """Local compute backend using subprocess.

    Runs training jobs as local processes with GPU support.
    Handles job tracking, log streaming, artifact management, and restart-safe
    recovery from an on-disk job manifest.
    """

    name = "local"

    def __init__(
        self,
        output_dir: str = "./outputs",
        python_path: Optional[str] = None,
        default_env: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize local compute backend.

        Args:
            output_dir: Base directory for job outputs
            python_path: Path to Python interpreter (default: sys.executable)
            default_env: Default environment variables for all jobs
            **kwargs: Additional configuration (ignored)
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._python_path = python_path or "python3"
        self._default_env = default_env or {}
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "local"

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _manifest_path(self, job_dir: Path) -> Path:
        return job_dir / LOCAL_JOB_MANIFEST_FILE_NAME

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temp_path.replace(path)

    def _read_json_dict(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def _read_progress_payload(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        return self._read_json_dict(job_dir / LOCAL_PROGRESS_FILE_NAME)

    def _job_state_from_progress_status(
        self,
        progress_status: Any,
    ) -> Optional[JobState]:
        if progress_status == LOCAL_PROGRESS_COMPLETED_STATUS:
            return JobState.COMPLETED
        if progress_status == LOCAL_PROGRESS_FAILED_STATUS:
            return JobState.FAILED
        return None

    def _job_state_from_manifest_value(self, value: Any) -> Optional[JobState]:
        if value in {
            JobState.COMPLETED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        }:
            return JobState(value)
        return None

    def _progress_state(self, progress_data: Optional[Dict[str, Any]]) -> Optional[JobState]:
        if not progress_data:
            return None
        metrics = progress_data.get("metrics", {})
        if not isinstance(metrics, dict):
            return None
        return self._job_state_from_progress_status(metrics.get("status"))

    def _process_start_ticks(self, pid: int) -> Optional[str]:
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            raw = stat_path.read_text()
        except OSError:
            return None
        try:
            tail = raw.rsplit(") ", 1)[1]
            fields = tail.split()
            return fields[LOCAL_PROCESS_STAT_START_TIME_TAIL_INDEX]
        except (IndexError, ValueError):
            return None

    def _process_cmdline(self, pid: int) -> list[str]:
        cmdline_path = Path("/proc") / str(pid) / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            return []
        parts = [part for part in raw.split(LOCAL_PROCESS_CMDLINE_SEPARATOR) if part]
        return [part.decode(errors="replace") for part in parts]

    def _process_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, LOCAL_PROCESS_CHECK_SIGNAL)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _process_matches_manifest(self, manifest: Dict[str, Any]) -> bool:
        pid = manifest.get("pid")
        process_group_id = manifest.get("processGroupId")
        if not isinstance(pid, int) or not isinstance(process_group_id, int):
            return False
        if not self._process_exists(pid):
            return False
        try:
            if os.getpgid(pid) != process_group_id:
                return False
        except OSError:
            return False

        expected_start_ticks = manifest.get("pidStartTicks")
        if expected_start_ticks and self._process_start_ticks(pid) != expected_start_ticks:
            return False

        config_path = manifest.get("configPath")
        if isinstance(config_path, str):
            return config_path in self._process_cmdline(pid)
        return False

    def _write_manifest(
        self,
        *,
        job_id: str,
        job_dir: Path,
        config_file: Path,
        stdout_log: Path,
        stderr_log: Path,
        cmd: list[str],
        process: subprocess.Popen[Any],
        started_at: str,
    ) -> Dict[str, Any]:
        process_group_id = os.getpgid(process.pid)
        manifest: Dict[str, Any] = {
            "schemaVersion": LOCAL_JOB_MANIFEST_SCHEMA_VERSION,
            "jobId": job_id,
            "jobDir": str(job_dir.resolve()),
            "configPath": str(config_file.resolve()),
            "stdoutPath": str(stdout_log.resolve()),
            "stderrPath": str(stderr_log.resolve()),
            "progressPath": str((job_dir / LOCAL_PROGRESS_FILE_NAME).resolve()),
            "pid": process.pid,
            "processGroupId": process_group_id,
            "pidStartTicks": self._process_start_ticks(process.pid),
            "command": cmd,
            "pythonPath": self._python_path,
            "state": JobState.RUNNING.value,
            "startedAt": started_at,
            "lastHeartbeatAt": started_at,
            "finishedAt": None,
            "error": None,
        }
        self._atomic_write_json(self._manifest_path(job_dir), manifest)
        return manifest

    def _update_manifest(self, job_info: Dict[str, Any], **updates: Any) -> None:
        manifest_path = job_info.get("manifest_path")
        if not isinstance(manifest_path, Path):
            return
        manifest = self._read_json_dict(manifest_path)
        if manifest is None:
            return
        manifest.update(updates)
        try:
            self._atomic_write_json(manifest_path, manifest)
        except OSError:
            logger.warning("Failed to update local job manifest %s", manifest_path)

    def _manifest_to_job_info(
        self,
        job_id: str,
        job_dir: Path,
        manifest: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        progress_data = self._read_progress_payload(job_dir)
        progress_state = self._progress_state(progress_data)
        finished_at = None
        error = None
        state = JobState.FAILED

        if progress_state:
            state = progress_state
            progress_file = job_dir / LOCAL_PROGRESS_FILE_NAME
            try:
                finished_at = datetime.fromtimestamp(progress_file.stat().st_mtime).isoformat()
            except OSError:
                finished_at = self._now_iso()
            if progress_state == JobState.FAILED and progress_data:
                metrics = progress_data.get("metrics", {})
                error_value = metrics.get("error") if isinstance(metrics, dict) else None
                error = error_value if isinstance(error_value, str) else None
        elif manifest_state := self._job_state_from_manifest_value(
            manifest.get("state") if manifest else None
        ):
            state = manifest_state
            finished_at_value = manifest.get("finishedAt") if manifest else None
            finished_at = finished_at_value if isinstance(finished_at_value, str) else None
            error_value = manifest.get("error") if manifest else None
            error = error_value if isinstance(error_value, str) else None
        elif manifest and self._process_matches_manifest(manifest):
            state = JobState.RUNNING
        else:
            error = LOCAL_PROCESS_NOT_RECOVERABLE_MESSAGE

        started_at = None
        if manifest:
            started_at = manifest.get("startedAt")
        if not isinstance(started_at, str):
            started_at = datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat()

        config_path = manifest.get("configPath") if manifest else None
        stdout_path = manifest.get("stdoutPath") if manifest else None
        stderr_path = manifest.get("stderrPath") if manifest else None
        process_group_id = manifest.get("processGroupId") if manifest else None
        pid = manifest.get("pid") if manifest else None

        return {
            "job_dir": job_dir,
            "config_file": Path(config_path) if isinstance(config_path, str) else job_dir / LOCAL_CONFIG_FILE_NAME,
            "stdout_log": Path(stdout_path) if isinstance(stdout_path, str) else job_dir / LOCAL_STDOUT_LOG_NAME,
            "stderr_log": Path(stderr_path) if isinstance(stderr_path, str) else job_dir / LOCAL_STDERR_LOG_NAME,
            "manifest_path": self._manifest_path(job_dir),
            "manifest": manifest,
            "pid": pid if isinstance(pid, int) else None,
            "process_group_id": process_group_id if isinstance(process_group_id, int) else None,
            "started_at": started_at,
            "finished_at": finished_at,
            "state": state,
            "error": error,
        }

    def _restore_job_from_disk(self, job_id: str) -> bool:
        job_dir = self._output_dir / job_id
        if not job_dir.is_dir():
            return False

        manifest_path = self._manifest_path(job_dir)
        manifest = self._read_json_dict(manifest_path)
        if manifest is not None:
            manifest_job_dir = manifest.get("jobDir")
            if manifest.get("jobId") != job_id or manifest_job_dir != str(job_dir.resolve()):
                manifest = None

        job_info = self._manifest_to_job_info(job_id, job_dir, manifest)
        if job_info is None:
            return False
        self._jobs[job_id] = job_info
        return True

    def _ensure_job_loaded(self, job_id: str) -> bool:
        return job_id in self._jobs or self._restore_job_from_disk(job_id)

    def _job_progress(self, job_info: Dict[str, Any]) -> tuple[Optional[JobProgress], Dict[str, Any], Optional[JobState]]:
        progress_file = job_info["job_dir"] / LOCAL_PROGRESS_FILE_NAME
        progress_data = self._read_json_dict(progress_file)
        if not progress_data:
            return None, {}, None

        progress = JobProgress(
            current_epoch=progress_data.get("current_epoch", LOCAL_PROGRESS_DEFAULT_COUNT),
            total_epochs=progress_data.get("total_epochs", LOCAL_PROGRESS_DEFAULT_COUNT),
            current_step=progress_data.get("current_step", LOCAL_PROGRESS_DEFAULT_COUNT),
            total_steps=progress_data.get("total_steps", LOCAL_PROGRESS_DEFAULT_COUNT),
        )
        metrics = progress_data.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        return progress, metrics, self._progress_state(progress_data)

    def _restored_process_is_valid(self, job_info: Dict[str, Any]) -> bool:
        manifest = job_info.get("manifest")
        return isinstance(manifest, dict) and self._process_matches_manifest(manifest)

    async def launch(
        self,
        script: str,
        config: Dict[str, Any],
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Launch training job as subprocess."""
        job_id = f"local_{uuid.uuid4().hex[:LOCAL_JOB_ID_HEX_LENGTH]}"
        job_dir = self._output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_file = job_dir / LOCAL_CONFIG_FILE_NAME
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        stdout_log = job_dir / LOCAL_STDOUT_LOG_NAME
        stderr_log = job_dir / LOCAL_STDERR_LOG_NAME

        job_env = os.environ.copy()
        job_env.update(self._default_env)
        if env:
            job_env.update(env)
        job_env["URDF_STUDIO_JOB_ID"] = job_id
        job_env["URDF_STUDIO_JOB_DIR"] = str(job_dir.resolve())

        script_path = Path(script).resolve()
        config_file_abs = config_file.resolve()

        if script_path.exists():
            cmd = [
                self._python_path,
                str(script_path),
                "--config",
                str(config_file_abs),
            ]
        else:
            cmd = [self._python_path, "-m", script, "--config", str(config_file_abs)]

        try:
            with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=job_env,
                    cwd=job_dir,
                    start_new_session=True,
                )

            started_at = self._now_iso()
            try:
                manifest = self._write_manifest(
                    job_id=job_id,
                    job_dir=job_dir,
                    config_file=config_file,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    cmd=cmd,
                    process=process,
                    started_at=started_at,
                )
            except OSError:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise

            self._jobs[job_id] = {
                "process": process,
                "job_dir": job_dir,
                "config_file": config_file,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "manifest_path": self._manifest_path(job_dir),
                "manifest": manifest,
                "pid": process.pid,
                "process_group_id": manifest["processGroupId"],
                "started_at": started_at,
                "state": JobState.RUNNING,
            }

            logger.info(f"Launched local job {job_id} (PID: {process.pid})")
            return job_id

        except Exception as e:
            logger.error(f"Failed to launch job: {e}")
            self._jobs[job_id] = {
                "job_dir": job_dir,
                "started_at": self._now_iso(),
                "finished_at": self._now_iso(),
                "state": JobState.FAILED,
                "error": str(e),
            }
            return job_id

    async def status(self, job_id: str) -> JobStatus:
        """Get job status."""
        if not self._ensure_job_loaded(job_id):
            return JobStatus(
                job_id=job_id,
                state=JobState.FAILED,
                error_message=LOCAL_JOB_NOT_FOUND_MESSAGE,
                compute_backend=self.name,
            )

        job_info = self._jobs[job_id]
        process = job_info.get("process")

        if process:
            return_code = process.poll()
            if return_code is None:
                state = JobState.RUNNING
            elif return_code == LOCAL_SUCCESS_EXIT_CODE:
                state = JobState.COMPLETED
                job_info["state"] = state
                job_info["finished_at"] = self._now_iso()
                self._update_manifest(job_info, state=state.value, finishedAt=job_info["finished_at"])
            else:
                state = JobState.FAILED
                job_info["state"] = state
                job_info["finished_at"] = self._now_iso()
                job_info["error"] = f"Process exited with code {return_code}"
                self._update_manifest(
                    job_info,
                    state=state.value,
                    finishedAt=job_info["finished_at"],
                    error=job_info["error"],
                )
        else:
            state = job_info.get("state", JobState.FAILED)
            if state == JobState.RUNNING and not self._restored_process_is_valid(job_info):
                state = JobState.FAILED
                job_info["state"] = state
                job_info["finished_at"] = self._now_iso()
                job_info["error"] = LOCAL_PROCESS_NOT_RECOVERABLE_MESSAGE
                self._update_manifest(
                    job_info,
                    state=state.value,
                    finishedAt=job_info["finished_at"],
                    error=job_info["error"],
                )

        progress, metrics, progress_state = self._job_progress(job_info)
        if progress_state:
            state = progress_state
            job_info["state"] = progress_state
            if progress_state in {JobState.COMPLETED, JobState.FAILED}:
                progress_file = job_info["job_dir"] / LOCAL_PROGRESS_FILE_NAME
                job_info["finished_at"] = datetime.fromtimestamp(
                    progress_file.stat().st_mtime
                ).isoformat()
                self._update_manifest(
                    job_info,
                    state=progress_state.value,
                    finishedAt=job_info["finished_at"],
                    error=job_info.get("error"),
                )

        logs_tail = None
        stdout_log = job_info.get("stdout_log")
        if stdout_log and stdout_log.exists():
            try:
                with open(stdout_log) as f:
                    lines = f.readlines()
                    logs_tail = "".join(lines[-LOCAL_COMPUTE_LOG_TAIL_LINES:])
            except IOError:
                pass

        return JobStatus(
            job_id=job_id,
            state=state,
            progress=progress,
            metrics=metrics,
            logs_tail=logs_tail,
            error_message=job_info.get("error"),
            started_at=job_info.get("started_at"),
            finished_at=job_info.get("finished_at"),
            compute_backend=self.name,
        )

    async def logs(self, job_id: str, follow: bool = False) -> AsyncIterator[str]:
        """Stream logs from the job."""
        if not self._ensure_job_loaded(job_id):
            yield f"Job {job_id} not found"
            return

        job_info = self._jobs[job_id]
        stdout_log = job_info.get("stdout_log")

        if not stdout_log or not stdout_log.exists():
            yield LOCAL_NO_LOGS_MESSAGE
            return

        with open(stdout_log) as f:
            for line in f:
                yield line

        if follow:
            process = job_info.get("process")
            with open(stdout_log) as f:
                f.seek(LOCAL_SUCCESS_EXIT_CODE, os.SEEK_END)
                while process and process.poll() is None:
                    line = f.readline()
                    if line:
                        yield line
                    else:
                        await asyncio.sleep(LOCAL_COMPUTE_LOG_FOLLOW_POLL_SECONDS)

                for line in f:
                    yield line

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        if not self._ensure_job_loaded(job_id):
            return False

        job_info = self._jobs[job_id]
        process = job_info.get("process")

        if process:
            if process.poll() is not None:
                return False
            process_group_id = os.getpgid(process.pid)
            try:
                os.killpg(process_group_id, signal.SIGTERM)
                try:
                    process.wait(timeout=LOCAL_COMPUTE_CANCEL_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(process_group_id, signal.SIGKILL)
                    process.wait()
            except Exception as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")
                return False
        else:
            process_group_id = job_info.get("process_group_id")
            if not isinstance(process_group_id, int) or not self._restored_process_is_valid(job_info):
                return False
            try:
                os.killpg(process_group_id, signal.SIGTERM)
                deadline = time.monotonic() + LOCAL_COMPUTE_CANCEL_GRACE_SECONDS
                while time.monotonic() < deadline and self._restored_process_is_valid(job_info):
                    await asyncio.sleep(LOCAL_COMPUTE_CANCEL_POLL_SECONDS)
                if self._restored_process_is_valid(job_info):
                    os.killpg(process_group_id, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Failed to cancel restored job {job_id}: {e}")
                return False

        job_info["state"] = JobState.CANCELLED
        job_info["finished_at"] = self._now_iso()
        job_info["error"] = None
        self._update_manifest(
            job_info,
            state=JobState.CANCELLED.value,
            finishedAt=job_info["finished_at"],
            error=None,
        )
        logger.info(f"Cancelled job {job_id}")
        return True

    async def list_artifacts(self, job_id: str) -> List[JobArtifact]:
        """List artifacts in job directory."""
        if not self._ensure_job_loaded(job_id):
            return []

        job_dir = self._jobs[job_id]["job_dir"]
        artifacts = []

        for pattern, artifact_type in LOCAL_ARTIFACT_PATTERNS.items():
            for path in job_dir.rglob(pattern):
                if path.is_file():
                    stat = path.stat()
                    artifacts.append(
                        JobArtifact(
                            name=path.name,
                            path=str(path),
                            size_bytes=stat.st_size,
                            artifact_type=artifact_type,
                            created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        )
                    )

        return artifacts

    async def download_artifact(
        self,
        job_id: str,
        artifact_name: str,
        dest: Path,
    ) -> Path:
        """Copy artifact to destination (local, so just return path)."""
        if not self._ensure_job_loaded(job_id):
            raise FileNotFoundError(f"Job {job_id} not found")

        job_dir = self._jobs[job_id]["job_dir"]

        for path in job_dir.rglob(artifact_name):
            if path.is_file():
                dest_path = dest / artifact_name
                dest.mkdir(parents=True, exist_ok=True)

                import shutil

                shutil.copy2(path, dest_path)
                return dest_path

        raise FileNotFoundError(f"Artifact {artifact_name} not found in job {job_id}")

    async def download_all_artifacts(
        self,
        job_id: str,
        dest: Path,
    ) -> List[Path]:
        """Copy all artifacts to destination."""
        artifacts = await self.list_artifacts(job_id)
        paths = []

        for artifact in artifacts:
            try:
                path = await self.download_artifact(job_id, artifact.name, dest)
                paths.append(path)
            except FileNotFoundError:
                pass

        return paths

    def estimate_cost(
        self,
        config: Dict[str, Any],
        duration_hours: Optional[float] = None,
    ) -> Optional[float]:
        """Local compute has no cost."""
        return None

    async def get_available_instances(self) -> List[Dict[str, Any]]:
        """Return local GPU info if available."""
        instances = []

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    instances.append(
                        {
                            "name": f"cuda:{i}",
                            "device": props.name,
                            "memory_gb": props.total_memory / (1024**3),
                            "available": True,
                            "cost_per_hour": 0,
                        }
                    )
        except ImportError:
            pass

        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                instances.append(
                    {
                        "name": "mps",
                        "device": "Apple Silicon",
                        "available": True,
                        "cost_per_hour": 0,
                    }
                )
        except ImportError:
            pass

        if not instances:
            instances.append(
                {
                    "name": "cpu",
                    "device": "CPU",
                    "available": True,
                    "cost_per_hour": 0,
                }
            )

        return instances

    async def cleanup(self, job_id: str) -> None:
        """Clean up job resources."""
        if job_id in self._jobs:
            await self.cancel(job_id)
            del self._jobs[job_id]

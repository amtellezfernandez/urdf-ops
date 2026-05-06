"""Fail-closed Macrodata Cloud compute backend placeholder.

The control-plane request shape is present so UrdfOps can evolve toward a real
cloud backend without inheriting the old fake-queued provider behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.robotops.compute.cloud_control import (
    TrainingCloudRuntimeConfig,
    TrainingCloudRunCreateRequest,
    build_training_cloud_run_create_request,
)
from backend.robotops.compute_protocol import JobArtifact, JobState, JobStatus
from backend.services.training_params import TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE


class MacrodataCompute:
    """Macrodata Cloud compute contract that remains disabled until lifecycle is wired."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = "./outputs",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._output_dir = output_dir
        self._extra = kwargs

    @property
    def name(self) -> str:
        return "macrodata"

    def build_launch_request(
        self,
        *,
        name: str,
        body: Dict[str, Any],
        runtime: TrainingCloudRuntimeConfig,
        manifest: Dict[str, Any] | None = None,
        secrets: Dict[str, object | None] | None = None,
        env: Dict[str, object | None] | None = None,
    ) -> TrainingCloudRunCreateRequest:
        return build_training_cloud_run_create_request(
            provider=self.name,
            name=name,
            body=body,
            runtime=runtime,
            manifest=manifest,
            secrets=secrets,
            env=env,
        )

    async def launch(
        self,
        script: str,
        config: Dict[str, Any],
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        raise RuntimeError(TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE)

    async def status(self, job_id: str) -> JobStatus:
        return JobStatus(
            job_id=job_id,
            state=JobState.FAILED,
            error_message=TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE,
            compute_backend=self.name,
        )

    async def logs(self, job_id: str, follow: bool = False) -> AsyncIterator[str]:
        yield TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE

    async def cancel(self, job_id: str) -> bool:
        return False

    async def list_artifacts(self, job_id: str) -> List[JobArtifact]:
        return []

    async def download_artifact(
        self,
        job_id: str,
        artifact_name: str,
        dest: Path,
    ) -> Path:
        raise RuntimeError(TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE)

    async def download_all_artifacts(
        self,
        job_id: str,
        dest: Path,
    ) -> List[Path]:
        return []

    def estimate_cost(
        self,
        config: Dict[str, Any],
        duration_hours: Optional[float] = None,
    ) -> Optional[float]:
        return None

    async def get_available_instances(self) -> List[Dict[str, Any]]:
        return []

    async def cleanup(self, job_id: str) -> None:
        return None

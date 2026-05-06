from __future__ import annotations

from backend.models.datasets import DatasetMixRequest, DatasetMixResponse
from backend.services.dataset_mix_control_plane import (
    get_dataset_mix_control_plane,
    normalize_local_dataset_paths,
)


def mix_datasets(req: DatasetMixRequest) -> DatasetMixResponse:
    """Submit a dataset mix job through the local control plane."""
    return get_dataset_mix_control_plane().submit_mix_job(req)


def get_dataset_mix_job(job_id: str) -> DatasetMixResponse:
    """Fetch a submitted dataset mix job."""
    return get_dataset_mix_control_plane().get_mix_job(job_id)

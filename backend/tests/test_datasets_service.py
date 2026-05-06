from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.models.dataset_alignment import DatasetRepresentationValidationResponse
from backend.models.datasets import (
    DatasetMixAlignmentDataset,
    DatasetMixAlignmentRequest,
    DatasetMixRequest,
)
from backend.services.dataset_alignment_params import (
    DEFAULT_INDEXED_REPRESENTATION_ID,
    DEFAULT_SEMANTIC_REPRESENTATION_ID,
)
from backend.services.dataset_mix_control_plane import (
    DatasetMixControlPlane,
    DatasetMixExecutionResult,
    DatasetMixWorkerRuntime,
    LocalDatasetMixJobStore,
    LocalDatasetMixObjectStore,
    LocalDatasetMixQueueStore,
)
from backend.services.dataset_mix_control_plane_params import DATASET_MIX_LEASE_TIMEOUT_SEC
from backend.services.dataset_treatments import analyze_dataset_treatment
from backend.services.dataset_treatments_params import (
    CONTENT_FINGERPRINT_KIND_EPISODE_SERIES_V1,
    TREATMENT_ACTION_CANONICALIZE_LOCAL_PATH,
    TREATMENT_ACTION_REQUIRES_MAPPING,
    TREATMENT_ACTION_REQUIRES_NAMING_REVIEW,
    TREATMENT_CODE_DUPLICATE_SOURCE,
    TREATMENT_CODE_INVALID_CONTENT_FINGERPRINT,
)
from backend.services.datasets import mix_datasets


def _build_alignment_dataset(dataset_id: str, representation_id: str) -> DatasetMixAlignmentDataset:
    return DatasetMixAlignmentDataset(
        dataset_id=dataset_id,
        embodiment_id="demo:robot",
        representation_id=representation_id,
        naming_status="named",
    )


def _install_valid_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    response = DatasetRepresentationValidationResponse(valid=True, errors=[], warnings=[])
    service = SimpleNamespace(validate_dataset_representations=lambda _request: response)
    monkeypatch.setattr(
        "backend.services.dataset_treatments.get_dataset_alignment_service",
        lambda: service,
    )


class _RuntimeStub(DatasetMixWorkerRuntime):
    def __init__(self, result: DatasetMixExecutionResult) -> None:
        self.result = result
        self.manifests = []

    def execute(self, manifest):
        self.manifests.append(manifest)
        return self.result


def _build_control_plane(tmp_path: Path, runtime: DatasetMixWorkerRuntime) -> DatasetMixControlPlane:
    root = tmp_path / "dataset-mix-control-plane"
    return DatasetMixControlPlane(
        object_store=LocalDatasetMixObjectStore(root / "objects"),
        job_store=LocalDatasetMixJobStore(root / "jobs"),
        queue_store=LocalDatasetMixQueueStore(root / "queue", DATASET_MIX_LEASE_TIMEOUT_SEC),
        runtime=runtime,
        auto_start_worker=False,
        worker_poll_interval_sec=0.01,
        worker_id="dataset-mix-service-test-worker",
    )


def test_mix_rejects_alignment_mismatch_without_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.services.datasets.get_dataset_mix_control_plane",
        lambda: _build_control_plane(tmp_path, _RuntimeStub(DatasetMixExecutionResult(success=True))),
    )
    req = DatasetMixRequest(
        repo_ids=["lerobot/demo"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                DatasetMixAlignmentDataset(
                    dataset_id="hf:demo/train",
                    embodiment_id="unknown:demo",
                    representation_id=DEFAULT_INDEXED_REPRESENTATION_ID,
                    naming_status="named",
                )
            ],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    result = mix_datasets(req)

    assert result.success is False
    assert result.status == "rejected"
    assert result.error == "Dataset alignment validation failed"
    assert result.alignment is not None
    assert result.alignment.valid is False
    assert result.treatment_manifest is not None
    assert result.treatment_manifest.stats.alignment_error_count >= 1


def test_mix_rejects_alignment_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.services.datasets.get_dataset_mix_control_plane",
        lambda: _build_control_plane(tmp_path, _RuntimeStub(DatasetMixExecutionResult(success=True))),
    )
    req = DatasetMixRequest(
        repo_ids=["lerobot/demo", "lerobot/demo2"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                DatasetMixAlignmentDataset(
                    dataset_id="hf:demo/train",
                    embodiment_id="unknown:demo",
                    representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
                    naming_status="named",
                )
            ],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        mix_datasets(req)

    assert exc.value.status_code == 400
    assert "Alignment dataset count must match" in str(exc.value.detail)


def test_mix_rejects_local_paths_outside_allowlisted_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_valid_alignment(monkeypatch)
    monkeypatch.setattr(
        "backend.services.datasets.get_dataset_mix_control_plane",
        lambda: _build_control_plane(tmp_path, _RuntimeStub(DatasetMixExecutionResult(success=True))),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (Path("/safe/root"),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.resolve_dataset_mix_local_path",
        lambda path_value: Path("/unsafe/root") / Path(path_value).name,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: False,
    )

    req = DatasetMixRequest(
        local_paths=["/tmp/private-dataset"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[_build_alignment_dataset("local:demo/train", DEFAULT_SEMANTIC_REPRESENTATION_ID)],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        mix_datasets(req)

    assert exc.value.status_code == 403
    assert "outside configured allowlisted roots" in str(exc.value.detail)


def test_mix_passes_canonical_allowlisted_local_paths_to_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_valid_alignment(monkeypatch)
    canonical_path = Path("/safe/root/datasets/demo")
    runtime = _RuntimeStub(
        DatasetMixExecutionResult(
            success=True,
            message="ok",
            output_path="/tmp/mixed",
        )
    )
    control_plane = _build_control_plane(tmp_path, runtime)
    monkeypatch.setattr("backend.services.datasets.get_dataset_mix_control_plane", lambda: control_plane)
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.DATASET_MIX_ALLOWED_LOCAL_ROOTS",
        (Path("/safe/root"),),
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.resolve_dataset_mix_local_path",
        lambda _path: canonical_path,
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.is_dataset_mix_local_path_allowed",
        lambda _path: True,
    )

    req = DatasetMixRequest(
        local_paths=["../unsafe-demo"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[_build_alignment_dataset("local:demo/train", DEFAULT_SEMANTIC_REPRESENTATION_ID)],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    result = mix_datasets(req)

    assert result.success is True
    assert result.status == "queued"
    assert control_plane.process_next_job_once() is True
    assert runtime.manifests
    assert runtime.manifests[0].sources[0].canonical_source == str(canonical_path)
    assert result.treatment_manifest is not None
    assert result.treatment_manifest.sources[0].canonical_source == str(canonical_path)
    assert (
        TREATMENT_ACTION_CANONICALIZE_LOCAL_PATH
        in result.treatment_manifest.sources[0].normalization_actions
    )


def test_mix_includes_duplicate_and_mapping_review_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_valid_alignment(monkeypatch)
    monkeypatch.setattr(
        "backend.services.datasets.get_dataset_mix_control_plane",
        lambda: _build_control_plane(tmp_path, _RuntimeStub(DatasetMixExecutionResult(success=True))),
    )

    req = DatasetMixRequest(
        repo_ids=["lerobot/demo", "lerobot/demo"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                _build_alignment_dataset("hf:demo/train", DEFAULT_INDEXED_REPRESENTATION_ID),
                DatasetMixAlignmentDataset(
                    dataset_id="hf:demo/train-copy",
                    embodiment_id="demo:robot",
                    representation_id=DEFAULT_INDEXED_REPRESENTATION_ID,
                    naming_status="unnamed",
                ),
            ],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    result = mix_datasets(req)

    assert result.success is True
    assert result.status == "queued"
    assert result.treatment_manifest is not None
    assert result.treatment_manifest.stats.duplicate_group_count == 1
    assert result.treatment_manifest.stats.exact_duplicate_group_count == 1
    assert result.treatment_manifest.stats.normalized_duplicate_group_count == 0
    assert result.treatment_manifest.sources[0].duplicate_group_size == 2
    assert result.treatment_manifest.sources[0].duplicate_match_kind == "exact"
    assert result.treatment_manifest.sources[0].canonical_fingerprint is not None
    assert TREATMENT_ACTION_REQUIRES_MAPPING in result.treatment_manifest.sources[0].normalization_actions
    assert (
        TREATMENT_ACTION_REQUIRES_NAMING_REVIEW
        in result.treatment_manifest.sources[1].normalization_actions
    )
    assert any(
        warning.code == TREATMENT_CODE_DUPLICATE_SOURCE
        for warning in result.treatment_manifest.warnings
    )


def test_mix_tracks_normalized_duplicate_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_valid_alignment(monkeypatch)
    monkeypatch.setattr(
        "backend.services.datasets.get_dataset_mix_control_plane",
        lambda: _build_control_plane(tmp_path, _RuntimeStub(DatasetMixExecutionResult(success=True))),
    )

    req = DatasetMixRequest(
        repo_ids=["OpenAI/Demo", "openai/demo"],
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                _build_alignment_dataset("hf:demo/train", DEFAULT_INDEXED_REPRESENTATION_ID),
                _build_alignment_dataset("hf:demo/train-2", DEFAULT_INDEXED_REPRESENTATION_ID),
            ],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    result = mix_datasets(req)

    assert result.success is True
    assert result.status == "queued"
    assert result.treatment_manifest is not None
    assert result.treatment_manifest.stats.duplicate_group_count == 1
    assert result.treatment_manifest.stats.exact_duplicate_group_count == 0
    assert result.treatment_manifest.stats.normalized_duplicate_group_count == 1
    assert result.treatment_manifest.sources[0].duplicate_match_kind == "normalized"


def test_analyze_prefers_backend_computed_content_signature_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_valid_alignment(monkeypatch)
    req = DatasetMixRequest(
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                DatasetMixAlignmentDataset(
                    dataset_id="local-upload:valid",
                    embodiment_id="demo:robot",
                    representation_id=DEFAULT_INDEXED_REPRESENTATION_ID,
                    naming_status="named",
                    content_signature={
                        "kind": CONTENT_FINGERPRINT_KIND_EPISODE_SERIES_V1,
                        "episodes": [
                            {
                                "episode_index": 0,
                                "frames": [
                                    {
                                        "timestamp": 0,
                                        "joints": {"elbow": -0.2, "shoulder": 0.1},
                                    }
                                ],
                            }
                        ],
                    },
                ),
                DatasetMixAlignmentDataset(
                    dataset_id="local-upload:invalid",
                    embodiment_id="demo:robot",
                    representation_id=DEFAULT_INDEXED_REPRESENTATION_ID,
                    naming_status="named",
                    content_fingerprint="content-1",
                    content_fingerprint_kind=CONTENT_FINGERPRINT_KIND_EPISODE_SERIES_V1,
                ),
            ],
            required_representation_id=DEFAULT_SEMANTIC_REPRESENTATION_ID,
        ),
    )

    result = analyze_dataset_treatment(req, normalized_local_paths=[])

    assert result.success is True
    assert result.treatment_manifest is not None
    assert result.treatment_manifest.sources[0].content_fingerprint is not None
    assert result.treatment_manifest.sources[0].content_fingerprint_kind == (
        CONTENT_FINGERPRINT_KIND_EPISODE_SERIES_V1
    )
    assert result.treatment_manifest.sources[1].content_fingerprint is None
    assert any(
        warning.code == TREATMENT_CODE_INVALID_CONTENT_FINGERPRINT
        for warning in result.treatment_manifest.warnings
    )

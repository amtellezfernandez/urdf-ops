from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.models.dataset_alignment import DatasetRepresentationValidationResponse
from backend.models.datasets import (
    DatasetMixAlignmentDataset,
    DatasetMixAlignmentRequest,
    DatasetMixArtifactRef,
    DatasetMixExecutionPlan,
    DatasetMixJobManifest,
    DatasetMixRequest,
    DatasetTreatmentAnalysisResponse,
    DatasetTreatmentManifest,
    DatasetTreatmentSourceManifest,
    DatasetTreatmentStats,
    DatasetMixSourceRef,
)
from backend.services.dataset_mix_control_plane import (
    DatasetMixControlPlane,
    DatasetMixExecutionResult,
    DatasetMixWorkerRuntime,
    OfficialFirstLerobotRuntime,
    LocalDatasetMixJobStore,
    LocalDatasetMixObjectStore,
    LocalDatasetMixQueueStore,
)
from backend.services.dataset_mix_control_plane_params import DATASET_MIX_LEASE_TIMEOUT_SEC
from backend.services.dataset_mix_lerobot_official import (
    OfficialLeRobotAvailability,
    OfficialLeRobotBindings,
    OfficialLeRobotDatasetMixer,
)
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_ENGINE_OFFICIAL_REQUIRED,
    DATASET_MIX_LEROBOT_OUTPUT_REPO_ID,
    DATASET_MIX_LEROBOT_RUNTIME_ENGINE_COMPATIBILITY,
    DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL,
)

TEST_REPRESENTATION_ID = "semantic/joint-position/v1"
TEST_DATASET_ID = "hf:demo/train"


class _RuntimeStub(DatasetMixWorkerRuntime):
    def __init__(self, result: DatasetMixExecutionResult) -> None:
        self.result = result
        self.manifests = []

    def execute(self, manifest):
        self.manifests.append(manifest)
        return self.result


class _OfficialMixerStub:
    def __init__(
        self,
        *,
        available: bool,
        payload: dict | None = None,
        can_execute: bool = True,
        unavailable_reason: str = "official missing",
    ) -> None:
        self.available = available
        self.payload = payload
        self.can_execute_value = can_execute
        self.unavailable_reason = unavailable_reason
        self.merge_manifests = []

    def availability(self) -> OfficialLeRobotAvailability:
        if self.available:
            return OfficialLeRobotAvailability(available=True)
        return OfficialLeRobotAvailability(
            available=False,
            reason=self.unavailable_reason,
        )

    def can_execute(self, _manifest: DatasetMixJobManifest) -> bool:
        return self.can_execute_value

    def merge(self, manifest: DatasetMixJobManifest) -> dict:
        self.merge_manifests.append(manifest)
        if self.payload is None:
            raise AssertionError("Official mixer stub payload was not configured")
        return self.payload


class _RaisingRuntimeStub(DatasetMixWorkerRuntime):
    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, manifest):
        raise self.error


def _build_local_lerobot_manifest(tmp_path: Path) -> DatasetMixJobManifest:
    source_root = tmp_path / "source-dataset"
    source_root.mkdir()
    treatment_manifest = _build_treatment_analysis(valid=True).treatment_manifest
    return DatasetMixJobManifest(
        manifest_version="v1",
        required_representation_id=TEST_REPRESENTATION_ID,
        sources=[
            DatasetMixSourceRef(
                source_id="local:0",
                dataset_id="local-demo",
                source_kind="local",
                source_value=str(source_root),
                canonical_source=str(source_root),
            )
        ],
        execution_plan=DatasetMixExecutionPlan(
            execution_mode="native-local-lerobot",
            reason="test",
        ),
        treatment_manifest=treatment_manifest,
        output_artifact=DatasetMixArtifactRef(
            store_kind="local",
            object_path="output",
            uri=str(tmp_path / "output"),
        ),
    )


def _build_request(*, repo_ids: list[str] | None = None) -> DatasetMixRequest:
    resolved_repo_ids = repo_ids or []
    return DatasetMixRequest(
        repo_ids=resolved_repo_ids,
        alignment=DatasetMixAlignmentRequest(
            datasets=[
                DatasetMixAlignmentDataset(
                    dataset_id=f"{TEST_DATASET_ID}-{index}",
                    embodiment_id="demo:robot",
                    representation_id=TEST_REPRESENTATION_ID,
                    naming_status="named",
                )
                for index, _ in enumerate(resolved_repo_ids)
            ],
            required_representation_id=TEST_REPRESENTATION_ID,
        ),
    )


def _build_treatment_analysis(*, valid: bool) -> DatasetTreatmentAnalysisResponse:
    return DatasetTreatmentAnalysisResponse(
        success=True,
        warnings=["warning-1"],
        alignment=DatasetRepresentationValidationResponse(valid=valid, errors=[], warnings=[]),
        treatment_manifest=DatasetTreatmentManifest(
            manifest_version="v1",
            required_representation_id=TEST_REPRESENTATION_ID,
            sources=[
                DatasetTreatmentSourceManifest(
                    source_id="repo:0",
                    dataset_id=TEST_DATASET_ID,
                    source_kind="repo",
                    source_value="lerobot/demo",
                    canonical_source="lerobot/demo",
                    representation_id=TEST_REPRESENTATION_ID,
                    naming_status="named",
                    profile_id="semantic-aligned",
                    profile_version="v1",
                )
            ],
            stats=DatasetTreatmentStats(total_sources=1, repo_source_count=1, unique_canonical_sources=1),
        ),
    )


def _build_control_plane(tmp_path: Path, runtime: DatasetMixWorkerRuntime) -> DatasetMixControlPlane:
    root = tmp_path / "dataset-mix-control-plane"
    return DatasetMixControlPlane(
        object_store=LocalDatasetMixObjectStore(root / "objects"),
        job_store=LocalDatasetMixJobStore(root / "jobs"),
        queue_store=LocalDatasetMixQueueStore(root / "queue", DATASET_MIX_LEASE_TIMEOUT_SEC),
        runtime=runtime,
        auto_start_worker=False,
        worker_poll_interval_sec=0.01,
        worker_id="dataset-mix-test-worker",
    )


def test_official_lerobot_adapter_loads_local_sources_through_official_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _build_local_lerobot_manifest(tmp_path)
    loaded_datasets = []
    merge_calls = []

    class FakeLeRobotDataset:
        def __init__(self, repo_id: str, root: Path | None = None) -> None:
            self.repo_id = repo_id
            self.root = root
            loaded_datasets.append(self)

    def fake_merge_datasets(
        datasets: list[FakeLeRobotDataset],
        *,
        output_repo_id: str,
        output_dir: Path,
    ) -> SimpleNamespace:
        merge_calls.append(
            {
                "datasets": datasets,
                "output_repo_id": output_repo_id,
                "output_dir": output_dir,
            }
        )
        return SimpleNamespace(
            meta=SimpleNamespace(total_episodes=2, total_frames=3, total_tasks=1),
        )

    mixer = OfficialLeRobotDatasetMixer()
    monkeypatch.setattr(
        mixer,
        "_load_bindings",
        lambda: OfficialLeRobotBindings(
            dataset_class=FakeLeRobotDataset,
            merge_datasets=fake_merge_datasets,
        ),
    )

    payload = mixer.merge(manifest)

    assert payload["success"] is True
    assert payload["info"]["engine"] == DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL
    assert payload["total_episodes"] == 2
    assert loaded_datasets[0].repo_id == Path(manifest.sources[0].canonical_source).name
    assert loaded_datasets[0].root == Path(manifest.sources[0].canonical_source)
    assert merge_calls[0]["datasets"] == loaded_datasets
    assert merge_calls[0]["output_dir"] == Path(
        str(manifest.output_artifact.uri)
    ).resolve(strict=False)


def test_official_lerobot_adapter_uses_managed_runner_when_backend_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _build_local_lerobot_manifest(tmp_path)
    runner_payloads = []

    def _raise_missing_lerobot() -> OfficialLeRobotBindings:
        raise ModuleNotFoundError("lerobot")

    def _fake_runner(_args: list[str], *, payload: dict | None) -> dict:
        runner_payloads.append(payload)
        return {
            "success": True,
            "message": "managed runner merge",
            "output_path": str(manifest.output_artifact.uri),
            "info": {"engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL},
            "debug": {"engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL},
        }

    mixer = OfficialLeRobotDatasetMixer()
    monkeypatch.setattr(mixer, "_load_bindings", _raise_missing_lerobot)
    monkeypatch.setattr(mixer, "_run_external_runner", _fake_runner)

    payload = mixer.merge(manifest)

    assert payload["message"] == "managed runner merge"
    assert (
        payload["info"]["engine"]
        == DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL
    )
    assert runner_payloads == [
        {
            "output_path": str(manifest.output_artifact.uri),
            "output_repo_id": DATASET_MIX_LEROBOT_OUTPUT_REPO_ID,
            "sources": [manifest.sources[0].model_dump(mode="json")],
        }
    ]


def test_official_first_lerobot_runtime_uses_official_adapter(tmp_path: Path) -> None:
    manifest = _build_local_lerobot_manifest(tmp_path)
    output_path = str(tmp_path / "official-output")
    official_mixer = _OfficialMixerStub(
        available=True,
        payload={
            "success": True,
            "message": "official merge",
            "output_path": output_path,
            "info": {"engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL},
            "debug": {"engine": DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL},
        },
    )
    compatibility_runtime = _RuntimeStub(DatasetMixExecutionResult(success=True))
    runtime = OfficialFirstLerobotRuntime(
        official_mixer=official_mixer,
        compatibility_runtime=compatibility_runtime,
    )

    result = runtime.execute(manifest)

    assert result.success is True
    assert result.message == "official merge"
    assert result.output_path == output_path
    assert (
        result.payload["info"]["engine"]
        == DATASET_MIX_LEROBOT_RUNTIME_ENGINE_OFFICIAL
    )
    assert official_mixer.merge_manifests == [manifest]
    assert compatibility_runtime.manifests == []


def test_official_first_lerobot_runtime_labels_compatibility_fallback(
    tmp_path: Path,
) -> None:
    manifest = _build_local_lerobot_manifest(tmp_path)
    compatibility_runtime = _RuntimeStub(
        DatasetMixExecutionResult(
            success=True,
            payload={"success": True, "info": {}, "debug": {}},
        )
    )
    runtime = OfficialFirstLerobotRuntime(
        official_mixer=_OfficialMixerStub(
            available=False,
            unavailable_reason="official LeRobot is not installed",
        ),
        compatibility_runtime=compatibility_runtime,
    )

    result = runtime.execute(manifest)

    assert result.success is True
    assert compatibility_runtime.manifests == [manifest]
    assert (
        result.payload["info"]["engine"]
        == DATASET_MIX_LEROBOT_RUNTIME_ENGINE_COMPATIBILITY
    )
    assert (
        result.payload["info"]["official_lerobot_unavailable_reason"]
        == "official LeRobot is not installed"
    )


def test_official_required_lerobot_runtime_rejects_compatibility_fallback(
    tmp_path: Path,
) -> None:
    manifest = _build_local_lerobot_manifest(tmp_path)
    compatibility_runtime = _RuntimeStub(DatasetMixExecutionResult(success=True))
    runtime = OfficialFirstLerobotRuntime(
        official_mixer=_OfficialMixerStub(
            available=False,
            unavailable_reason="official LeRobot is not installed",
        ),
        compatibility_runtime=compatibility_runtime,
        engine_mode=DATASET_MIX_LEROBOT_ENGINE_OFFICIAL_REQUIRED,
    )

    result = runtime.execute(manifest)

    assert result.success is False
    assert result.error == "official LeRobot is not installed"
    assert compatibility_runtime.manifests == []


def test_submit_mix_job_rejects_invalid_alignment_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _paths: _build_treatment_analysis(valid=False),
    )
    control_plane = _build_control_plane(
        tmp_path,
        _RuntimeStub(DatasetMixExecutionResult(success=True)),
    )

    response = control_plane.submit_mix_job(_build_request(repo_ids=["lerobot/demo"]))

    assert response.status == "rejected"
    assert response.success is False
    assert response.error == "Dataset alignment validation failed"
    assert response.completed_at is not None
    assert control_plane.process_next_job_once() is False
    assert control_plane.get_mix_job(response.job_id).status == "rejected"


def test_process_next_job_persists_manifest_and_marks_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _RuntimeStub(
        DatasetMixExecutionResult(
            success=True,
            message="mixed",
            output_path="/tmp/mixed-output",
            payload={"success": True, "message": "mixed", "output_path": "/tmp/mixed-output"},
        )
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _paths: _build_treatment_analysis(valid=True),
    )
    control_plane = _build_control_plane(tmp_path, runtime)

    queued = control_plane.submit_mix_job(_build_request(repo_ids=["lerobot/demo"]))

    assert queued.status == "queued"
    assert queued.success is True
    assert queued.manifest_artifact is not None
    assert queued.source_refs[0].source_kind == "repo"
    assert queued.execution_plan is not None
    assert queued.execution_plan.execution_mode == "legacy-subprocess"
    assert queued.partition_plan is None

    processed = control_plane.process_next_job_once()
    completed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert runtime.manifests
    assert runtime.manifests[0].execution_plan.execution_mode == "legacy-subprocess"
    assert runtime.manifests[0].partition_plan is None
    assert runtime.manifests[0].sources[0].source_value == "lerobot/demo"
    assert completed.status == "succeeded"
    assert completed.output_path == "/tmp/mixed-output"
    assert completed.message == "mixed"
    assert completed.completed_at is not None


def test_process_next_job_marks_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _RuntimeStub(
        DatasetMixExecutionResult(
            success=False,
            error="dataset mixing failed",
        )
    )
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _paths: _build_treatment_analysis(valid=True),
    )
    control_plane = _build_control_plane(tmp_path, runtime)

    queued = control_plane.submit_mix_job(_build_request(repo_ids=["lerobot/demo"]))
    processed = control_plane.process_next_job_once()
    failed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert failed.status == "failed"
    assert failed.success is False
    assert failed.error == "dataset mixing failed"


def test_process_next_job_converts_runtime_http_exceptions_into_failed_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "backend.services.dataset_mix_control_plane.analyze_dataset_treatment",
        lambda _req, _paths: _build_treatment_analysis(valid=True),
    )
    control_plane = _build_control_plane(
        tmp_path,
        _RaisingRuntimeStub(HTTPException(status_code=504, detail="Dataset mixing timed out")),
    )

    queued = control_plane.submit_mix_job(_build_request(repo_ids=["lerobot/demo"]))
    processed = control_plane.process_next_job_once()
    failed = control_plane.get_mix_job(queued.job_id)

    assert processed is True
    assert failed.status == "failed"
    assert failed.success is False
    assert failed.error == "Dataset mixing timed out"

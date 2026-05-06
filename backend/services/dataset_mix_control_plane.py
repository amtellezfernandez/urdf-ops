from __future__ import annotations

import json
import subprocess
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.core.paths import SCRIPTS_DIR
from backend.models.datasets import (
    DatasetMixArtifactRef,
    DatasetMixJobManifest,
    DatasetMixRequest,
    DatasetMixResponse,
    DatasetMixSourceRef,
)
from backend.services.dataset_mix_lerobot import (
    merge_local_lerobot_datasets,
)
from backend.services.dataset_mix_lerobot_official import (
    OfficialLeRobotDatasetMixer,
)
from backend.services.dataset_mix_planner import compile_dataset_mix_execution_plan
from backend.services.dataset_mix_planner import compile_dataset_mix_partition_plan
from backend.services.dataset_mix_control_plane_params import (
    DATASET_MIX_JOB_ID_PREFIX,
    DATASET_MIX_JOB_RECORDS_DIRNAME,
    DATASET_MIX_JOB_ROOT,
    DATASET_MIX_LEASE_TIMEOUT_SEC,
    DATASET_MIX_MANIFEST_FILENAME,
    DATASET_MIX_MANIFEST_VERSION,
    DATASET_MIX_OBJECTS_DIRNAME,
    DATASET_MIX_OUTPUT_DIRNAME,
    DATASET_MIX_QUEUE_COMPLETED_DIRNAME,
    DATASET_MIX_QUEUE_FAILED_DIRNAME,
    DATASET_MIX_QUEUE_LEASED_DIRNAME,
    DATASET_MIX_QUEUE_PENDING_DIRNAME,
    DATASET_MIX_QUEUE_ROOT_DIRNAME,
    DATASET_MIX_RESULT_FILENAME,
    DATASET_MIX_WORKER_POLL_INTERVAL_SEC,
)
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_ENGINE,
    DATASET_MIX_LEROBOT_ENGINE_COMPATIBILITY_ONLY,
    DATASET_MIX_LEROBOT_ENGINE_OFFICIAL_REQUIRED,
    DATASET_MIX_LEROBOT_RUNTIME_ENGINE_COMPATIBILITY,
)
from backend.services.dataset_source_contract import (
    LOCAL_DATASET_PATHS_DISABLED_DETAIL,
    normalize_local_dataset_paths as normalize_dataset_source_local_paths,
)
from backend.services.dataset_treatments import analyze_dataset_treatment
from backend.services.datasets_params import (
    DATASET_MIX_ALLOWED_LOCAL_ROOTS,
    DATASET_MIX_SCRIPT_TIMEOUT_SEC,
    is_dataset_mix_local_path_allowed,
    resolve_dataset_mix_local_path,
)


def normalize_local_dataset_paths(local_paths: list[str]) -> list[str]:
    return normalize_dataset_source_local_paths(
        local_paths,
        allowed_roots=DATASET_MIX_ALLOWED_LOCAL_ROOTS,
        path_resolver=resolve_dataset_mix_local_path,
        path_allowed=is_dataset_mix_local_path_allowed,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _job_object_path(job_id: str, filename: str) -> str:
    return f"{job_id}/{filename}"


def _iso_after(seconds: float) -> str:
    return (_utc_now() + timedelta(seconds=seconds)).isoformat()


def _artifact_uri(root: Path, object_path: str) -> str:
    return str((root / object_path).resolve(strict=False))


class _DatasetMixQueueEntry(BaseModel):
    job_id: str = Field(..., min_length=1)
    enqueued_at: str = Field(..., min_length=1)
    attempt_count: int = Field(default=0, ge=0)
    lease_owner: str | None = None
    lease_expires_at: str | None = None


class DatasetMixLease(BaseModel):
    job_id: str = Field(..., min_length=1)
    worker_id: str = Field(..., min_length=1)
    lease_expires_at: str = Field(..., min_length=1)
    attempt_count: int = Field(..., ge=1)


class DatasetMixExecutionResult(BaseModel):
    success: bool
    message: str | None = None
    output_path: str | None = None
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class DatasetMixObjectStore(ABC):
    @abstractmethod
    def build_ref(self, object_path: str) -> DatasetMixArtifactRef:
        raise NotImplementedError

    @abstractmethod
    def write_json(self, object_path: str, payload: dict[str, Any]) -> DatasetMixArtifactRef:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, object_path: str) -> dict[str, Any]:
        raise NotImplementedError


class DatasetMixJobStore(ABC):
    @abstractmethod
    def create(self, record: DatasetMixResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> DatasetMixResponse:
        raise NotImplementedError

    @abstractmethod
    def save(self, record: DatasetMixResponse) -> None:
        raise NotImplementedError


class DatasetMixQueueStore(ABC):
    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def claim(self, worker_id: str) -> DatasetMixLease | None:
        raise NotImplementedError

    @abstractmethod
    def complete(self, lease: DatasetMixLease) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail(self, lease: DatasetMixLease) -> None:
        raise NotImplementedError


class DatasetMixWorkerRuntime(ABC):
    @abstractmethod
    def execute(self, manifest: DatasetMixJobManifest) -> DatasetMixExecutionResult:
        raise NotImplementedError


class LocalDatasetMixObjectStore(DatasetMixObjectStore):
    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=False)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, object_path: str) -> Path:
        path = (self._root / object_path).resolve(strict=False)
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(f"Object path escapes object store root: {object_path}") from exc
        return path

    def build_ref(self, object_path: str) -> DatasetMixArtifactRef:
        return DatasetMixArtifactRef(
            store_kind="local",
            object_path=object_path,
            uri=_artifact_uri(self._root, object_path),
        )

    def write_json(self, object_path: str, payload: dict[str, Any]) -> DatasetMixArtifactRef:
        path = self._resolve(object_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self.build_ref(object_path)

    def read_json(self, object_path: str) -> dict[str, Any]:
        return json.loads(self._resolve(object_path).read_text(encoding="utf-8"))


class LocalDatasetMixJobStore(DatasetMixJobStore):
    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=False)
        self._root.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self._root / f"{job_id}.json"

    def create(self, record: DatasetMixResponse) -> None:
        self.save(record)

    def get(self, job_id: str) -> DatasetMixResponse:
        path = self._job_path(job_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset mix job not found: {job_id}")
        return DatasetMixResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: DatasetMixResponse) -> None:
        path = self._job_path(record.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


class LocalDatasetMixQueueStore(DatasetMixQueueStore):
    def __init__(self, root: Path, lease_timeout_sec: float) -> None:
        self._root = root.resolve(strict=False)
        self._lease_timeout_sec = lease_timeout_sec
        self._lock = threading.Lock()
        for dirname in (
            DATASET_MIX_QUEUE_PENDING_DIRNAME,
            DATASET_MIX_QUEUE_LEASED_DIRNAME,
            DATASET_MIX_QUEUE_COMPLETED_DIRNAME,
            DATASET_MIX_QUEUE_FAILED_DIRNAME,
        ):
            (self._root / dirname).mkdir(parents=True, exist_ok=True)

    def _entry_path(self, dirname: str, job_id: str) -> Path:
        return self._root / dirname / f"{job_id}.json"

    def _load_entry(self, path: Path) -> _DatasetMixQueueEntry:
        return _DatasetMixQueueEntry.model_validate_json(path.read_text(encoding="utf-8"))

    def _save_entry(self, path: Path, entry: _DatasetMixQueueEntry) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry.model_dump_json(indent=2), encoding="utf-8")

    def _iter_paths(self, dirname: str) -> list[Path]:
        return sorted((self._root / dirname).glob("*.json"))

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            self._save_entry(
                self._entry_path(DATASET_MIX_QUEUE_PENDING_DIRNAME, job_id),
                _DatasetMixQueueEntry(job_id=job_id, enqueued_at=_utc_now_iso()),
            )

    def claim(self, worker_id: str) -> DatasetMixLease | None:
        with self._lock:
            for path in self._iter_paths(DATASET_MIX_QUEUE_PENDING_DIRNAME):
                entry = self._load_entry(path)
                leased_entry = _DatasetMixQueueEntry(
                    job_id=entry.job_id,
                    enqueued_at=entry.enqueued_at,
                    attempt_count=entry.attempt_count + 1,
                    lease_owner=worker_id,
                    lease_expires_at=_iso_after(self._lease_timeout_sec),
                )
                leased_path = self._entry_path(DATASET_MIX_QUEUE_LEASED_DIRNAME, entry.job_id)
                path.replace(leased_path)
                self._save_entry(leased_path, leased_entry)
                return DatasetMixLease(
                    job_id=leased_entry.job_id,
                    worker_id=worker_id,
                    lease_expires_at=leased_entry.lease_expires_at or _utc_now_iso(),
                    attempt_count=leased_entry.attempt_count,
                )

            now = _utc_now()
            for path in self._iter_paths(DATASET_MIX_QUEUE_LEASED_DIRNAME):
                entry = self._load_entry(path)
                if entry.lease_expires_at is None:
                    continue
                if datetime.fromisoformat(entry.lease_expires_at) > now:
                    continue
                leased_entry = _DatasetMixQueueEntry(
                    job_id=entry.job_id,
                    enqueued_at=entry.enqueued_at,
                    attempt_count=entry.attempt_count + 1,
                    lease_owner=worker_id,
                    lease_expires_at=_iso_after(self._lease_timeout_sec),
                )
                self._save_entry(path, leased_entry)
                return DatasetMixLease(
                    job_id=leased_entry.job_id,
                    worker_id=worker_id,
                    lease_expires_at=leased_entry.lease_expires_at or _utc_now_iso(),
                    attempt_count=leased_entry.attempt_count,
                )

        return None

    def complete(self, lease: DatasetMixLease) -> None:
        self._move(lease.job_id, DATASET_MIX_QUEUE_COMPLETED_DIRNAME)

    def fail(self, lease: DatasetMixLease) -> None:
        self._move(lease.job_id, DATASET_MIX_QUEUE_FAILED_DIRNAME)

    def _move(self, job_id: str, dirname: str) -> None:
        with self._lock:
            leased_path = self._entry_path(DATASET_MIX_QUEUE_LEASED_DIRNAME, job_id)
            if not leased_path.exists():
                return
            leased_path.replace(self._entry_path(dirname, job_id))


class LocalDatasetMixSubprocessRuntime(DatasetMixWorkerRuntime):
    def execute(self, manifest: DatasetMixJobManifest) -> DatasetMixExecutionResult:
        script_path = SCRIPTS_DIR / "dataset_mixer.py"
        if not script_path.exists():
            raise HTTPException(status_code=500, detail=f"Dataset mixer script not found at {script_path}")

        repo_ids: list[str] = []
        local_paths: list[str] = []
        for source in manifest.sources:
            if source.source_kind == "repo":
                repo_ids.append(source.source_value)
            elif source.source_kind == "local":
                local_paths.append(source.canonical_source)

        command = ["python3", str(script_path)]
        if repo_ids:
            command.extend(["--repo-ids", *repo_ids])
        if local_paths:
            command.extend(["--local-paths", *local_paths])
        if manifest.output_artifact.uri:
            command.extend(["--output", manifest.output_artifact.uri])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DATASET_MIX_SCRIPT_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Dataset mixing timed out") from exc

        if result.returncode != 0:
            return DatasetMixExecutionResult(
                success=False,
                error=result.stderr or "Dataset mixing failed",
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return DatasetMixExecutionResult(
                success=False,
                error=f"Failed to parse output: {result.stdout}",
            )

        return DatasetMixExecutionResult(
            success=bool(payload.get("success", True)),
            message=payload.get("message", "Dataset mix job completed"),
            output_path=payload.get("output_path"),
            error=payload.get("error"),
            payload=payload,
        )


def _dataset_mix_payload_result(payload: dict[str, Any]) -> DatasetMixExecutionResult:
    return DatasetMixExecutionResult(
        success=bool(payload.get("success", True)),
        message=payload.get("message", "Dataset mix job completed"),
        output_path=payload.get("output_path"),
        error=payload.get("error"),
        payload=payload,
    )


def _tag_dataset_mix_payload_engine(
    payload: dict[str, Any],
    *,
    engine: str,
    official_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    tagged_payload = dict(payload)
    info = dict(tagged_payload.get("info") or {})
    debug = dict(tagged_payload.get("debug") or {})
    info["engine"] = engine
    debug["engine"] = engine
    if official_unavailable_reason is not None:
        info["official_lerobot_unavailable_reason"] = official_unavailable_reason
        debug["official_lerobot_unavailable_reason"] = official_unavailable_reason
    tagged_payload["info"] = info
    tagged_payload["debug"] = debug
    return tagged_payload


class LocalLerobotCompatibilityRuntime(DatasetMixWorkerRuntime):
    def execute(self, manifest: DatasetMixJobManifest) -> DatasetMixExecutionResult:
        payload = _tag_dataset_mix_payload_engine(
            merge_local_lerobot_datasets(manifest),
            engine=DATASET_MIX_LEROBOT_RUNTIME_ENGINE_COMPATIBILITY,
        )
        return _dataset_mix_payload_result(payload)


class OfficialFirstLerobotRuntime(DatasetMixWorkerRuntime):
    def __init__(
        self,
        *,
        official_mixer: OfficialLeRobotDatasetMixer | None = None,
        compatibility_runtime: DatasetMixWorkerRuntime | None = None,
        engine_mode: str = DATASET_MIX_LEROBOT_ENGINE,
    ) -> None:
        self._official_mixer = official_mixer or OfficialLeRobotDatasetMixer()
        self._compatibility_runtime = (
            compatibility_runtime or LocalLerobotCompatibilityRuntime()
        )
        self._engine_mode = engine_mode

    def execute(self, manifest: DatasetMixJobManifest) -> DatasetMixExecutionResult:
        if self._engine_mode == DATASET_MIX_LEROBOT_ENGINE_COMPATIBILITY_ONLY:
            return self._compatibility_runtime.execute(manifest)

        availability = self._official_mixer.availability()
        reason = availability.reason or "Official LeRobot adapter cannot execute this manifest"
        if availability.available and self._official_mixer.can_execute(manifest):
            try:
                return _dataset_mix_payload_result(self._official_mixer.merge(manifest))
            except HTTPException as exc:
                reason = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            except Exception as exc:
                reason = str(exc)

        if self._engine_mode == DATASET_MIX_LEROBOT_ENGINE_OFFICIAL_REQUIRED:
            return DatasetMixExecutionResult(success=False, error=reason)

        compatibility_result = self._compatibility_runtime.execute(manifest)
        compatibility_result.payload = _tag_dataset_mix_payload_engine(
            compatibility_result.payload,
            engine=DATASET_MIX_LEROBOT_RUNTIME_ENGINE_COMPATIBILITY,
            official_unavailable_reason=reason,
        )
        return compatibility_result


class CompositeDatasetMixRuntime(DatasetMixWorkerRuntime):
    def __init__(
        self,
        native_local_runtime: DatasetMixWorkerRuntime,
        fallback_runtime: DatasetMixWorkerRuntime,
    ) -> None:
        self._native_local_runtime = native_local_runtime
        self._fallback_runtime = fallback_runtime

    def execute(self, manifest: DatasetMixJobManifest) -> DatasetMixExecutionResult:
        if manifest.execution_plan.execution_mode == "native-local-lerobot":
            return self._native_local_runtime.execute(manifest)
        return self._fallback_runtime.execute(manifest)


class DatasetMixControlPlane:
    def __init__(
        self,
        object_store: DatasetMixObjectStore,
        job_store: DatasetMixJobStore,
        queue_store: DatasetMixQueueStore,
        runtime: DatasetMixWorkerRuntime,
        *,
        auto_start_worker: bool = True,
        worker_poll_interval_sec: float = DATASET_MIX_WORKER_POLL_INTERVAL_SEC,
        worker_id: str | None = None,
    ) -> None:
        self._object_store = object_store
        self._job_store = job_store
        self._queue_store = queue_store
        self._runtime = runtime
        self._auto_start_worker = auto_start_worker
        self._worker_poll_interval_sec = worker_poll_interval_sec
        self._worker_id = worker_id or f"{DATASET_MIX_JOB_ID_PREFIX}-worker"
        self._worker_thread: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._stop_event = threading.Event()

    def submit_mix_job(self, req: DatasetMixRequest) -> DatasetMixResponse:
        self._validate_request(req)
        normalized_local_paths = normalize_local_dataset_paths(req.local_paths)
        treatment_analysis = analyze_dataset_treatment(req, normalized_local_paths)
        source_refs = list(self._compile_source_refs(treatment_analysis.treatment_manifest.sources))
        execution_plan = compile_dataset_mix_execution_plan(sources=source_refs)
        partition_plan = compile_dataset_mix_partition_plan(
            execution_plan=execution_plan,
        )
        created_at = _utc_now_iso()
        job_id = self._new_job_id()
        manifest_artifact = self._object_store.build_ref(
            _job_object_path(job_id, DATASET_MIX_MANIFEST_FILENAME),
        )
        output_artifact = self._object_store.build_ref(
            _job_object_path(job_id, DATASET_MIX_OUTPUT_DIRNAME),
        )
        self._persist_manifest(
            job_id,
            alignment=treatment_analysis.alignment,
            treatment_manifest=treatment_analysis.treatment_manifest,
            source_refs=source_refs,
            execution_plan=execution_plan,
            partition_plan=partition_plan,
            output_artifact=output_artifact,
        )
        alignment_response = treatment_analysis.alignment
        if alignment_response is None or not alignment_response.valid:
            record = DatasetMixResponse(
                job_id=job_id,
                status="rejected",
                created_at=created_at,
                updated_at=created_at,
                completed_at=created_at,
                success=False,
                error="Dataset alignment validation failed",
                warnings=treatment_analysis.warnings,
                alignment=alignment_response,
                treatment_manifest=treatment_analysis.treatment_manifest,
                source_refs=source_refs,
                manifest_artifact=manifest_artifact,
                output_artifact=output_artifact,
                execution_plan=execution_plan,
                partition_plan=partition_plan,
            )
            self._job_store.create(record)
            return record

        record = DatasetMixResponse(
            job_id=job_id,
            status="queued",
            created_at=created_at,
            updated_at=created_at,
            success=True,
            message="Dataset mix job queued",
            warnings=treatment_analysis.warnings,
            alignment=alignment_response,
            treatment_manifest=treatment_analysis.treatment_manifest,
            source_refs=source_refs,
            manifest_artifact=manifest_artifact,
            output_artifact=output_artifact,
            execution_plan=execution_plan,
            partition_plan=partition_plan,
        )
        self._job_store.create(record)
        self._queue_store.enqueue(job_id)
        if self._auto_start_worker:
            self.ensure_worker_started()
        return record

    def get_mix_job(self, job_id: str) -> DatasetMixResponse:
        return self._job_store.get(job_id)

    def process_next_job_once(self) -> bool:
        lease = self._queue_store.claim(self._worker_id)
        if lease is None:
            return False
        self._run_claimed_job(lease)
        return True

    def ensure_worker_started(self) -> None:
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=self._worker_id,
                daemon=True,
            )
            self._worker_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        worker_thread = self._worker_thread
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=self._worker_poll_interval_sec)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.process_next_job_once():
                continue
            self._stop_event.wait(self._worker_poll_interval_sec)

    def _run_claimed_job(self, lease: DatasetMixLease) -> None:
        record = self._job_store.get(lease.job_id)
        running_at = _utc_now_iso()
        running_record = record.model_copy(
            update={
                "status": "running",
                "started_at": record.started_at or running_at,
                "updated_at": running_at,
                "message": "Dataset mix job running",
                "error": None,
            }
        )
        self._job_store.save(running_record)

        try:
            if running_record.manifest_artifact is None:
                raise RuntimeError("Dataset mix manifest artifact is missing")
            manifest = DatasetMixJobManifest.model_validate(
                self._object_store.read_json(running_record.manifest_artifact.object_path),
            )
            execution = self._runtime.execute(manifest)
            completed_at = _utc_now_iso()
            if not execution.success:
                failed_record = running_record.model_copy(
                    update={
                        "status": "failed",
                        "updated_at": completed_at,
                        "completed_at": completed_at,
                        "success": False,
                        "message": None,
                        "error": execution.error or "Dataset mixing failed",
                    }
                )
                self._job_store.save(failed_record)
                self._queue_store.fail(lease)
                return

            self._object_store.write_json(
                _job_object_path(lease.job_id, DATASET_MIX_RESULT_FILENAME),
                execution.payload,
            )
            output_path = execution.output_path
            if output_path is None and running_record.output_artifact is not None:
                output_path = running_record.output_artifact.uri
            succeeded_record = running_record.model_copy(
                update={
                    "status": "succeeded",
                    "updated_at": completed_at,
                    "completed_at": completed_at,
                    "success": True,
                    "message": execution.message or "Dataset mix job completed",
                    "output_path": output_path,
                    "error": None,
                }
            )
            self._job_store.save(succeeded_record)
            self._queue_store.complete(lease)
        except HTTPException as exc:
            self._save_failed_job(
                running_record,
                lease,
                exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            )
        except Exception as exc:
            self._save_failed_job(running_record, lease, str(exc))

    def _validate_request(self, req: DatasetMixRequest) -> None:
        if not req.repo_ids and not req.local_paths:
            raise HTTPException(status_code=400, detail="At least one repo ID or local path is required")

        expected_dataset_count = len(req.repo_ids) + len(req.local_paths)
        if len(req.alignment.datasets) != expected_dataset_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Alignment dataset count must match repo_ids + local_paths "
                    f"({expected_dataset_count} expected, {len(req.alignment.datasets)} received)."
                ),
            )

    def _persist_manifest(
        self,
        job_id: str,
        *,
        alignment: Any,
        treatment_manifest: Any,
        source_refs: list[DatasetMixSourceRef],
        execution_plan: Any,
        partition_plan: Any,
        output_artifact: DatasetMixArtifactRef,
    ) -> None:
        manifest = DatasetMixJobManifest(
            manifest_version=DATASET_MIX_MANIFEST_VERSION,
            required_representation_id=treatment_manifest.required_representation_id,
            sources=source_refs,
            execution_plan=execution_plan,
            partition_plan=partition_plan,
            alignment=alignment,
            treatment_manifest=treatment_manifest,
            output_artifact=output_artifact,
        )
        self._object_store.write_json(
            _job_object_path(job_id, DATASET_MIX_MANIFEST_FILENAME),
            manifest.model_dump(mode="json"),
        )

    def _compile_source_refs(self, sources: Iterable[Any]) -> Iterable[DatasetMixSourceRef]:
        for source in sources:
            yield DatasetMixSourceRef(
                source_id=source.source_id,
                dataset_id=source.dataset_id,
                source_kind=source.source_kind,
                source_value=source.source_value,
                canonical_source=source.canonical_source,
            )

    def _new_job_id(self) -> str:
        return f"{DATASET_MIX_JOB_ID_PREFIX}-{uuid.uuid4().hex}"

    def _save_failed_job(
        self,
        running_record: DatasetMixResponse,
        lease: DatasetMixLease,
        error: str,
    ) -> None:
        failed_at = _utc_now_iso()
        failed_record = running_record.model_copy(
            update={
                "status": "failed",
                "updated_at": failed_at,
                "completed_at": failed_at,
                "success": False,
                "message": None,
                "error": error,
            }
        )
        self._job_store.save(failed_record)
        self._queue_store.fail(lease)


def create_local_dataset_mix_control_plane(
    *,
    root: Path = DATASET_MIX_JOB_ROOT,
    auto_start_worker: bool = True,
) -> DatasetMixControlPlane:
    return DatasetMixControlPlane(
        object_store=LocalDatasetMixObjectStore(root / DATASET_MIX_OBJECTS_DIRNAME),
        job_store=LocalDatasetMixJobStore(root / DATASET_MIX_JOB_RECORDS_DIRNAME),
        queue_store=LocalDatasetMixQueueStore(
            root / DATASET_MIX_QUEUE_ROOT_DIRNAME,
            DATASET_MIX_LEASE_TIMEOUT_SEC,
        ),
        runtime=CompositeDatasetMixRuntime(
            native_local_runtime=OfficialFirstLerobotRuntime(),
            fallback_runtime=LocalDatasetMixSubprocessRuntime(),
        ),
        auto_start_worker=auto_start_worker,
    )


_dataset_mix_control_plane: DatasetMixControlPlane | None = None
_dataset_mix_control_plane_lock = threading.Lock()


def get_dataset_mix_control_plane() -> DatasetMixControlPlane:
    global _dataset_mix_control_plane
    with _dataset_mix_control_plane_lock:
        if _dataset_mix_control_plane is None:
            _dataset_mix_control_plane = create_local_dataset_mix_control_plane()
        return _dataset_mix_control_plane

from __future__ import annotations

import os
from pathlib import Path

from backend.core.paths import BASE_DIR
from backend.services.datasets_params import DATASET_MIX_SCRIPT_TIMEOUT_SEC

SECONDS_PER_MINUTE = 60
DEFAULT_DATASET_MIX_JOB_ROOT = BASE_DIR / ".cache" / "dataset-mix-control-plane"
DEFAULT_DATASET_MIX_WORKER_POLL_INTERVAL_SEC = 1.0
DEFAULT_DATASET_MIX_LEASE_TIMEOUT_SEC = max(
    2 * DATASET_MIX_SCRIPT_TIMEOUT_SEC,
    10 * SECONDS_PER_MINUTE,
)
DATASET_MIX_MANIFEST_VERSION = "v1"
DATASET_MIX_QUEUE_ROOT_DIRNAME = "queue"
DATASET_MIX_QUEUE_PENDING_DIRNAME = "pending"
DATASET_MIX_QUEUE_LEASED_DIRNAME = "leased"
DATASET_MIX_QUEUE_COMPLETED_DIRNAME = "completed"
DATASET_MIX_QUEUE_FAILED_DIRNAME = "failed"
DATASET_MIX_OBJECTS_DIRNAME = "objects"
DATASET_MIX_JOB_RECORDS_DIRNAME = "jobs"
DATASET_MIX_RESULT_FILENAME = "result.json"
DATASET_MIX_MANIFEST_FILENAME = "manifest.json"
DATASET_MIX_OUTPUT_DIRNAME = "output"
DATASET_MIX_JOB_ID_PREFIX = "dataset-mix"


def _read_path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return Path(raw).expanduser().resolve(strict=False)


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


DATASET_MIX_JOB_ROOT = _read_path_env(
    "URDF_DATASET_MIX_JOB_ROOT",
    DEFAULT_DATASET_MIX_JOB_ROOT,
)
DATASET_MIX_WORKER_POLL_INTERVAL_SEC = _read_positive_float_env(
    "URDF_DATASET_MIX_WORKER_POLL_INTERVAL_SEC",
    DEFAULT_DATASET_MIX_WORKER_POLL_INTERVAL_SEC,
)
DATASET_MIX_LEASE_TIMEOUT_SEC = _read_positive_float_env(
    "URDF_DATASET_MIX_LEASE_TIMEOUT_SEC",
    float(DEFAULT_DATASET_MIX_LEASE_TIMEOUT_SEC),
)
